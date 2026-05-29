"""Training loop for the PatchTST trend model (plan §4 training regime).

Design split:
  - `train()` is pure: it takes already-split `Sample` lists and returns a
    trained model + metrics. No DB, no filesystem — so it runs in a unit test on
    synthetic data (see tests/test_train.py).
  - `run()` / `main()` are the thin shell: load frames from Supabase, assemble +
    split, train, evaluate on the true holdout, and persist a `model_versions`
    row + a `.pt` file.

Regime: masked multi-horizon cross-entropy with per-horizon inverse-frequency
class weights (counters the up-bias) plus horizon weights {1M:1, 3M:1, 6M:.8,
1Y:.6}, AdamW (lr 3e-4, wd 0.05, clip 1.0), linear warmup over the first 10% of
steps then cosine decay, bf16 autocast on MPS, early stop on mean val lift —
directional accuracy above the majority-class base rate (patience 15, max 100
epochs).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from backend.config import get_settings
from backend.ingestion.db import pool_context
from backend.ml.dataset import (
    Sample,
    SplitConfig,
    assemble_calendar_aligned,
    assemble_samples,
    build_calendar_grid,
    cross_sectional_medians,
    load_frames_cached,
    relabel_cross_sectional,
    split_samples,
    to_arrays,
)
from backend.ml.model import HORIZONS, PatchTST, PatchTSTConfig

# Long horizons have noisier labels and fewer non-overlapping samples; down-
# weight them so their gradients don't dominate (plan §4).
HORIZON_LOSS_WEIGHTS: dict[str, float] = {"1M": 1.0, "3M": 1.0, "6M": 0.8, "1Y": 0.6}


def _sanitize_json(obj: object) -> object:
    """Replace float NaN/inf with None so json.dumps produces valid JSON."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    return obj


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0
    warmup_frac: float = 0.10
    patience: int = 15
    seed: int = 1337
    reg_loss_weight: float = 1.0        # weight of the return-regression term vs classification
    horizon_weights: dict[str, float] = field(default_factory=lambda: dict(HORIZON_LOSS_WEIGHTS))
    class_weight_power: float = 1.0     # 1=inverse-frequency class weights, 0=uniform/off
    shuffle_labels: bool = False        # permutation control: randomize labels (noise floor)
    cross_sectional: bool = False       # select on rank IC instead of directional lift


@dataclass
class TrainResult:
    model: PatchTST
    best_epoch: int
    val_loss: float
    val_accuracy: dict[str, float]      # per-horizon directional accuracy at best epoch
    return_scale: dict[str, float]      # per-horizon train-set return std (loss scaling)
    class_weights: dict[str, list[float]]  # per-horizon [w_down, w_up] used in CE
    history: list[dict]


# =============================================================
# Device / autocast
# =============================================================


def pick_device(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@contextlib.contextmanager
def autocast(device: str):
    # bf16 autocast on the accelerator; plain context on CPU (keeps tests simple).
    if device in ("mps", "cuda"):
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            yield
    else:
        yield


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =============================================================
# Loss + metrics (masked over unavailable horizons)
# =============================================================


def masked_loss(
    logits: dict[str, torch.Tensor],
    returns: dict[str, torch.Tensor] | None,
    y: torch.Tensor,
    r: torch.Tensor,
    mask: torch.Tensor,
    weights: dict[str, float],
    return_scale: dict[str, float],
    reg_loss_weight: float = 1.0,
    class_weights: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Masked multi-task loss: per-horizon CE + (optional) return regression.

    Classification: per-horizon CE over valid samples, optionally with per-class
    weights (`class_weights[h]` = tensor([w_down, w_up])) to offset the universe's
    up-bias so the loss isn't minimized by always predicting up. Regression: Huber
    (SmoothL1)
    on the residual `(r̂ - r)`, divided by the horizon's train-set return std so
    1Y (large scale) doesn't swamp 1M's gradients. Both terms use the same
    horizon weights; the regression term is scaled by `reg_loss_weight`.

    logits[h]: (B, 2); returns[h]: (B,); y: (B, H) long; r: (B, H) float;
    mask: (B, H) float {0,1}.
    """
    total = logits[HORIZONS[0]].new_zeros(())
    for j, h in enumerate(HORIZONS):
        m = mask[:, j]
        denom = m.sum()
        if denom.item() == 0:
            continue
        cw = None if class_weights is None else class_weights[h].to(logits[h].dtype)
        ce = F.cross_entropy(logits[h], y[:, j], weight=cw, reduction="none")
        total = total + weights[h] * (ce * m).sum() / denom
        if returns is not None:
            resid = (returns[h] - r[:, j]) / max(return_scale[h], 1e-4)
            huber = F.smooth_l1_loss(resid, torch.zeros_like(resid), reduction="none")
            total = total + reg_loss_weight * weights[h] * (huber * m).sum() / denom
    return total


def compute_return_scale(samples: list[Sample]) -> dict[str, float]:
    """Per-horizon std of the log-return target over valid training samples.

    Used only to scale regression residuals in the loss so horizons train on a
    comparable footing. Stored in the registry config for reproducibility.
    """
    arr = to_arrays(samples)
    r, m = arr["r"], arr["mask"].astype(bool)
    scale: dict[str, float] = {}
    for j, h in enumerate(HORIZONS):
        vals = r[m[:, j], j]
        scale[h] = max(float(vals.std()), 1e-4) if vals.size > 1 else 1.0
    return scale


def compute_class_weights(
    samples: list[Sample], power: float = 1.0
) -> dict[str, list[float]]:
    """Per-horizon class weights over valid training samples.

    The universe (esp. long horizons) is up-biased, so unweighted CE can collapse
    to "always predict up" (lift 0). `power` controls how hard that's corrected:
      power=1.0 -> full inverse-frequency (a down-error and an up-error cost equally)
      power=0.0 -> uniform [1, 1] (no correction)
      between   -> softened.
    Weights are `w_c ∝ freq_c**(-power)`, normalized so the frequency-weighted mean
    weight is exactly 1 for any power (loss scale unchanged). Returned as
    [w_down, w_up] per horizon and stored in the registry config for reproducibility.
    """
    arr = to_arrays(samples)
    y, m = arr["y"], arr["mask"].astype(bool)
    weights: dict[str, list[float]] = {}
    for j, h in enumerate(HORIZONS):
        yj = y[m[:, j], j]
        n = yj.size
        if n == 0:
            weights[h] = [1.0, 1.0]
            continue
        f_up = max(int(yj.sum()), 1) / n
        f_down = max(n - int(yj.sum()), 1) / n
        z = f_down ** (1.0 - power) + f_up ** (1.0 - power)
        weights[h] = [f_down ** (-power) / z, f_up ** (-power) / z]
    return weights


def _class_weight_tensors(
    class_weights: dict[str, list[float]] | None, device: str
) -> dict[str, torch.Tensor] | None:
    """Materialize the [w_down, w_up] lists as float32 tensors on `device` (once)."""
    if class_weights is None:
        return None
    return {
        h: torch.tensor(class_weights[h], device=device, dtype=torch.float32)
        for h in HORIZONS
    }


def shuffle_labels(samples: list[Sample], seed: int) -> list[Sample]:
    """Permutation control: destroy the feature->label association.

    For each horizon independently, permute the (label, return) values among the
    samples where that horizon is unmasked. Base rates and mask structure are
    preserved, so the only thing removed is learnable signal. A model that still
    beats baseline lift on shuffled labels reveals leakage or a metric bug.
    """
    rng = np.random.default_rng(seed)
    out = [
        Sample(s.ticker_id, s.embedding_idx, s.sample_end, s.features,
               dict(s.labels), dict(s.returns), s.mask)
        for s in samples
    ]
    for h in HORIZONS:
        idx = [i for i, s in enumerate(out) if s.mask[h]]
        if len(idx) < 2:
            continue
        labels = np.array([out[i].labels[h] for i in idx])
        returns = np.array([out[i].returns[h] for i in idx])
        perm = rng.permutation(len(idx))
        for k, i in enumerate(idx):
            out[i].labels[h] = int(labels[perm[k]])
            out[i].returns[h] = float(returns[perm[k]])
    return out


def _rank_ic(
    scores: np.ndarray, targets: np.ndarray, dates: np.ndarray, min_names: int = 10
) -> tuple[float, int]:
    """Mean per-date Spearman rank IC between `scores` and `targets`.

    Groups by date (the cross-section), Spearman-correlates predicted score vs
    realized return within each date that has at least `min_names` names, and
    averages over dates. Returns (mean_ic, n_dates_used) — the cross-sectional
    analogue of directional lift.
    """
    import pandas as pd

    if scores.size == 0:
        return float("nan"), 0
    df = pd.DataFrame({"d": dates, "s": scores, "t": targets})
    ics = [
        g["s"].corr(g["t"], method="spearman")
        for _, g in df.groupby("d")
        if len(g) >= min_names
    ]
    ics = [v for v in ics if v == v]  # drop NaN (zero-variance dates)
    return (float(np.mean(ics)), len(ics)) if ics else (float("nan"), 0)


@torch.no_grad()
def horizon_metrics(
    logits: dict[str, torch.Tensor],
    returns: dict[str, torch.Tensor] | None,
    y: torch.Tensor,
    r: torch.Tensor,
    mask: torch.Tensor,
    dates: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """Per-horizon classification + regression metrics over valid samples.

    Classification: directional accuracy, Brier, base rate, lift. `base_rate` is
    the up-fraction; `lift` is `acc - max(base_rate, 1 - base_rate)` — accuracy
    above the majority-class baseline. At long horizons the universe is heavily
    up-biased (1Y ≈ 0.69 pooled), so raw `acc` hides whether a model beats
    "always predict up"; `lift` is the honest comparator there.

    Regression (when `returns` is given): `ret_rmse` / `ret_mae` of the predicted
    log-return in real return space. `ret_rmse` doubles as a price-interval width
    (predicted_price × exp(±ret_rmse)) on the dashboard.

    Rank IC (when `dates` is given): mean per-date Spearman correlation between the
    predicted score (relative-return head if present, else P(up)) and the realized
    return — the cross-sectional ranking metric. `ic_n` is the count of dates that
    cleared the minimum-names threshold.
    """
    out: dict[str, dict[str, float]] = {}
    for j, h in enumerate(HORIZONS):
        m = mask[:, j].bool()
        n = int(m.sum().item())
        if n == 0:
            out[h] = {
                "acc": float("nan"), "brier": float("nan"),
                "base_rate": float("nan"), "lift": float("nan"),
                "ret_rmse": float("nan"), "ret_mae": float("nan"),
                "ic": float("nan"), "ic_n": 0, "n": 0,
            }
            continue
        lg = logits[h][m]
        target = y[m, j]
        pred = lg.argmax(dim=1)
        acc = (pred == target).float().mean().item()
        p_up = torch.softmax(lg.float(), dim=1)[:, 1]
        brier = ((p_up - target.float()) ** 2).mean().item()
        base_rate = target.float().mean().item()
        lift = acc - max(base_rate, 1.0 - base_rate)
        ret_rmse = ret_mae = float("nan")
        if returns is not None:
            resid = returns[h][m].float() - r[m, j].float()
            ret_rmse = resid.pow(2).mean().sqrt().item()
            ret_mae = resid.abs().mean().item()
        ic, ic_n = float("nan"), 0
        if dates is not None:
            score = (returns[h][m].float() if returns is not None else p_up).cpu().numpy()
            ic, ic_n = _rank_ic(score, r[m, j].float().cpu().numpy(), dates[m.cpu().numpy()])
        out[h] = {
            "acc": acc, "brier": brier,
            "base_rate": base_rate, "lift": lift,
            "ret_rmse": ret_rmse, "ret_mae": ret_mae,
            "ic": ic, "ic_n": ic_n, "n": n,
        }
    return out


def mean_accuracy(metrics: dict[str, dict[str, float]]) -> float:
    accs = [m["acc"] for m in metrics.values() if m["n"] > 0]
    return float(np.mean(accs)) if accs else float("nan")


def mean_lift(metrics: dict[str, dict[str, float]]) -> float:
    """Mean per-horizon lift (acc above the majority-class base rate).

    This is the model-selection signal: raw accuracy rewards predicting the
    up-biased majority class, whereas lift only credits genuine skill.
    """
    lifts = [m["lift"] for m in metrics.values() if m["n"] > 0]
    return float(np.mean(lifts)) if lifts else float("nan")


def mean_ic(metrics: dict[str, dict[str, float]]) -> float:
    """Mean per-horizon rank IC over horizons with a usable cross-section."""
    ics = [m["ic"] for m in metrics.values() if m.get("ic_n", 0) > 0 and m["ic"] == m["ic"]]
    return float(np.mean(ics)) if ics else float("nan")


# =============================================================
# Data → tensors → loader
# =============================================================


def _loader(samples: list[Sample], batch_size: int, shuffle: bool, device: str) -> DataLoader:
    arr = to_arrays(samples)
    ds = TensorDataset(
        torch.from_numpy(arr["x"]),
        torch.from_numpy(arr["ticker_idx"]),
        torch.from_numpy(arr["y"]),
        torch.from_numpy(arr["r"]),
        torch.from_numpy(arr["mask"]),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _evaluate_samples(
    model: PatchTST,
    samples: list[Sample],
    device: str,
    batch_size: int,
    return_scale: dict[str, float],
    reg_loss_weight: float = 1.0,
    class_weights: dict[str, list[float]] | None = None,
) -> tuple[float, dict[str, dict[str, float]]]:
    """Return (mean masked loss, per-horizon metrics) over a sample list."""
    model.eval()
    loader = _loader(samples, batch_size, shuffle=False, device=device)
    cw_t = _class_weight_tensors(class_weights, device)
    # Accumulate predictions/targets on CPU to compute exact (not batch-averaged) metrics.
    all_logits = {h: [] for h in HORIZONS}
    has_returns = model.return_heads is not None
    all_returns = {h: [] for h in HORIZONS}
    ys, rs, masks = [], [], []
    losses, n_batches = 0.0, 0
    with torch.no_grad():
        for x, idx, y, r, mask in loader:
            x, idx = x.to(device), idx.to(device)
            y, r, mask = y.to(device), r.to(device), mask.to(device)
            with autocast(device):
                out = model(x, idx)
            losses += masked_loss(
                out.logits, out.returns, y, r, mask,
                HORIZON_LOSS_WEIGHTS, return_scale, reg_loss_weight, cw_t,
            ).item()
            n_batches += 1
            for h in HORIZONS:
                all_logits[h].append(out.logits[h].float().cpu())
                if has_returns:
                    all_returns[h].append(out.returns[h].float().cpu())
            ys.append(y.cpu())
            rs.append(r.cpu())
            masks.append(mask.cpu())
    cat_logits = {h: torch.cat(all_logits[h]) for h in HORIZONS}
    cat_returns = {h: torch.cat(all_returns[h]) for h in HORIZONS} if has_returns else None
    # Loader is shuffle=False, so concat order matches `samples` order — dates align.
    dates = np.array([s.sample_end.toordinal() for s in samples], dtype=np.int64)
    metrics = horizon_metrics(
        cat_logits, cat_returns, torch.cat(ys), torch.cat(rs), torch.cat(masks), dates=dates
    )
    return (losses / max(n_batches, 1)), metrics


# =============================================================
# Core training loop (pure)
# =============================================================


def train(
    splits: dict[str, list[Sample]],
    num_tickers: int,
    model_cfg: PatchTSTConfig | None = None,
    train_cfg: TrainConfig | None = None,
    device: str | None = None,
    log=print,
) -> TrainResult:
    model_cfg = model_cfg or PatchTSTConfig()
    train_cfg = train_cfg or TrainConfig()
    device = pick_device(device)
    seed_everything(train_cfg.seed)

    train_samples, val_samples = splits["train"], splits["val"]
    if not train_samples:
        raise ValueError("no training samples")

    # Permutation control: randomize labels so any beat-the-baseline lift can only
    # come from leakage or a metric bug. Holdout (scored in run()) stays real.
    if train_cfg.shuffle_labels:
        log("shuffle_labels: permuting train/val labels (noise-floor control)")
        train_samples = shuffle_labels(train_samples, train_cfg.seed)
        val_samples = shuffle_labels(val_samples, train_cfg.seed + 1)

    # Per-horizon return std (from train only) scales the regression residuals.
    return_scale = compute_return_scale(train_samples)
    # Per-horizon class weights (from train only) offset the up-bias in CE.
    class_weights = compute_class_weights(train_samples, train_cfg.class_weight_power)
    class_weight_t = _class_weight_tensors(class_weights, device)

    model = PatchTST(model_cfg, num_tickers=num_tickers).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        betas=train_cfg.betas,
    )

    train_loader = _loader(train_samples, train_cfg.batch_size, shuffle=True, device=device)
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = steps_per_epoch * train_cfg.epochs
    warmup_steps = max(int(total_steps * train_cfg.warmup_frac), 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_metric = float("-inf")   # selection score: mean val lift (or -train_loss with no val)
    best_epoch, best_val_loss = -1, float("inf")
    best_val_metrics: dict[str, dict[str, float]] = {}
    best_state = None
    epochs_since_best = 0
    history: list[dict] = []

    for epoch in range(train_cfg.epochs):
        model.train()
        running = 0.0
        for x, idx, y, r, mask in train_loader:
            x, idx = x.to(device), idx.to(device)
            y, r, mask = y.to(device), r.to(device), mask.to(device)
            opt.zero_grad()
            with autocast(device):
                out = model(x, idx)
                loss = masked_loss(
                    out.logits, out.returns, y, r, mask,
                    train_cfg.horizon_weights, return_scale, train_cfg.reg_loss_weight,
                    class_weight_t,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()
            sched.step()
            running += loss.item()
        train_loss = running / steps_per_epoch

        if val_samples:
            val_loss, val_metrics = _evaluate_samples(
                model, val_samples, device, train_cfg.batch_size,
                return_scale, train_cfg.reg_loss_weight, class_weights,
            )
            val_mean_acc = mean_accuracy(val_metrics)
            val_mean_lift = mean_lift(val_metrics)
            val_mean_ic = mean_ic(val_metrics)
            # Cross-sectional runs select on rank IC; absolute runs on directional lift.
            select_metric = val_mean_ic if train_cfg.cross_sectional else val_mean_lift
        else:  # no val (tiny single-ticker run): select on negative train loss
            val_loss, val_metrics = train_loss, {}
            val_mean_acc = val_mean_lift = val_mean_ic = float("nan")
            select_metric = -train_loss

        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
             "val_mean_acc": val_mean_acc, "val_mean_lift": val_mean_lift,
             "val_mean_ic": val_mean_ic, "lr": sched.get_last_lr()[0]}
        )
        log(
            f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  "
            f"acc {val_mean_acc:.4f}  lift {val_mean_lift:+.4f}  ic {val_mean_ic:+.4f}  "
            f"lr {sched.get_last_lr()[0]:.2e}"
        )

        if select_metric > best_metric + 1e-5:
            best_metric = select_metric
            best_epoch, best_val_loss = epoch, val_loss
            best_val_metrics = val_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= train_cfg.patience:
                metric_name = "ic" if train_cfg.cross_sectional else "lift"
                log(f"early stop at epoch {epoch} (best {best_epoch}, {metric_name} {best_metric:+.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainResult(
        model=model,
        best_epoch=best_epoch,
        val_loss=best_val_loss,
        val_accuracy={h: m["acc"] for h, m in best_val_metrics.items()},
        return_scale=return_scale,
        class_weights=class_weights,
        history=history,
    )


# =============================================================
# Persistence
# =============================================================


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


async def save_model_version(
    pool,
    result: TrainResult,
    model_cfg: PatchTSTConfig,
    train_cfg: TrainConfig,
    splits: dict[str, list[Sample]],
    holdout_metrics: dict[str, dict[str, float]],
    n_tickers_trained: int,
) -> str:
    """Write the .pt weights and a candidate model_versions row. Returns its id."""
    settings = get_settings()
    settings.models_dir.mkdir(parents=True, exist_ok=True)

    model_version_id = await pool.fetchval("select gen_random_uuid()")
    weights_path = settings.models_dir / f"{model_version_id}.pt"
    torch.save(result.model.state_dict(), weights_path)
    sha = _sha256(weights_path)

    def window(split: str) -> tuple[date, date]:
        ds = [s.sample_end for s in splits[split]]
        return (min(ds), max(ds))

    train_start, train_end = window("train")
    holdout_start, holdout_end = window("holdout") if splits["holdout"] else (train_end, train_end)

    config = {
        "model": {**asdict(model_cfg), "horizons": list(model_cfg.horizons)},
        "train": {**asdict(train_cfg)},
        "feature_dim": model_cfg.n_features,
        "seq_len": model_cfg.seq_len,
        "return_scale": result.return_scale,    # per-horizon std used to scale reg residuals
        "class_weights": result.class_weights,  # per-horizon [w_down, w_up] used in CE
    }

    await pool.execute(
        """
        insert into model_versions (
            model_version_id, training_window_start, training_window_end,
            holdout_window_start, holdout_window_end, weights_path, weights_sha256,
            n_params, n_tickers_trained, val_loss, directional_accuracy,
            holdout_metrics, status, config
        ) values ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,'candidate',$13)
        """,
        model_version_id, train_start, train_end, holdout_start, holdout_end,
        str(weights_path), sha, result.model.num_parameters(), n_tickers_trained,
        result.val_loss,
        json.dumps(_sanitize_json(result.val_accuracy)),
        json.dumps(_sanitize_json({h: m for h, m in holdout_metrics.items()})),
        json.dumps(_sanitize_json(config)),
    )
    return str(model_version_id)


# =============================================================
# CLI shell
# =============================================================


def _parse_window(window: str | None) -> tuple[date | None, date | None]:
    if not window:
        return None, None
    start_s, _, end_s = window.partition(":")
    start = date.fromisoformat(start_s) if start_s else None
    end = date.fromisoformat(end_s) if end_s else None
    return start, end


def _clip(samples: list[Sample], start: date | None, end: date | None) -> list[Sample]:
    return [
        s for s in samples
        if (start is None or s.sample_end >= start) and (end is None or s.sample_end <= end)
    ]


async def run(args) -> None:
    model_cfg = PatchTSTConfig()
    train_cfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
        class_weight_power=args.class_weight_power, shuffle_labels=args.shuffle_labels,
        cross_sectional=args.cross_sectional,
    )
    split_cfg = SplitConfig()
    start, end = _parse_window(args.window)

    async with pool_context() as pool:
        frames = await load_frames_cached(pool, symbols=args.symbols, refresh=args.refresh_cache)
        if not frames:
            raise SystemExit("no active tickers / frames loaded")
        num_tickers = (
            max(f.embedding_idx for f in frames) + 1
            if not args.symbols
            else await pool.fetchval("select max(embedding_idx) + 1 from tickers")
        )

        if args.calendar_aligned:
            grid = build_calendar_grid(frames)
            print(f"loaded {len(frames)} tickers; assembling calendar-aligned samples ({len(grid)} months) ...")
            samples = assemble_calendar_aligned(frames, grid)
        else:
            print(f"loaded {len(frames)} tickers; assembling samples (stride={args.stride}) ...")
            samples = assemble_samples(frames, stride=args.stride)
        if args.cross_sectional:
            print("cross-sectional: relabeling vs per-date universe median ...")
            samples = relabel_cross_sectional(samples, cross_sectional_medians(frames))
        samples = _clip(samples, start, end)
        T = end or max(s.sample_end for s in samples)
        splits = split_samples(samples, split_cfg, T=T)
        print(
            f"samples: train={len(splits['train'])} val={len(splits['val'])} "
            f"holdout={len(splits['holdout'])} (T={T})"
        )

        result = train(splits, num_tickers=num_tickers, model_cfg=model_cfg, train_cfg=train_cfg)

        holdout_metrics: dict[str, dict[str, float]] = {}
        if splits["holdout"]:
            _, holdout_metrics = _evaluate_samples(
                result.model, splits["holdout"], pick_device(), train_cfg.batch_size,
                result.return_scale, train_cfg.reg_loss_weight, result.class_weights,
            )
        print(f"best epoch {result.best_epoch}  val_acc {result.val_accuracy}")
        for h in HORIZONS:
            m = holdout_metrics.get(h)
            if m:
                print(
                    f"  holdout {h}: acc={m['acc']:.3f} lift={m['lift']:+.3f} "
                    f"ic={m['ic']:+.4f}(d={m['ic_n']}) base={m['base_rate']:.3f} "
                    f"ret_rmse={m['ret_rmse']:.4f} n={m['n']}"
                )

        if args.dry_run:
            print("--dry-run: skipping model_versions write")
            return

        n_trained = len({s.embedding_idx for s in splits["train"]})
        mvid = await save_model_version(
            pool, result, model_cfg, train_cfg, splits, holdout_metrics, n_trained
        )
        print(f"saved candidate model_version {mvid}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train the PatchTST trend model")
    p.add_argument("--symbols", nargs="*", help="restrict to these symbols (single-ticker sanity)")
    p.add_argument("--refresh-cache", action="store_true",
                   help="re-pull frames from Supabase and overwrite the local frame cache")
    p.add_argument("--window", help="START:END (ISO dates) to clip sample-end range")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--stride", type=int, default=21, help="subsample end dates (thins overlap)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--class-weight-power", type=float, default=1.0,
                   help="class-weight strength: 1.0=inverse-frequency, 0.0=uniform/off")
    p.add_argument("--shuffle-labels", action="store_true",
                   help="permutation control: randomize labels to measure the no-signal floor")
    p.add_argument("--cross-sectional", action="store_true",
                   help="relabel vs per-date universe median; select on rank IC")
    p.add_argument("--calendar-aligned", action="store_true",
                   help="one sample per month-end per ticker; thick cross-sections for walk-forward IC")
    p.add_argument("--dry-run", action="store_true", help="train but do not write model_versions")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
