"""Training-loop tests — pure, CPU, no DB.

Covers the masked multi-horizon loss, the metrics, and a small end-to-end run
of `train()` on synthetic data to prove the loop (optimizer, LR schedule,
early-stop bookkeeping, ticker embedding) actually fits a learnable signal.
"""

from __future__ import annotations

from datetime import date, timedelta

import torch

from backend.ml.dataset import TickerFrame, assemble_ticker_samples
from backend.ml.model import HORIZONS, PatchTSTConfig
from backend.ml.train import (
    TrainConfig,
    horizon_metrics,
    masked_loss,
    mean_accuracy,
    train,
)


def _logits_pointing_at(y: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build per-horizon logits whose argmax equals the labels (confident)."""
    out = {}
    for j, h in enumerate(HORIZONS):
        lg = torch.full((y.shape[0], 2), -3.0)
        lg[torch.arange(y.shape[0]), y[:, j]] = 3.0
        out[h] = lg
    return out


# =============================================================
# masked_loss
# =============================================================


_NO_SCALE = {h: 1.0 for h in HORIZONS}


def test_masked_loss_skips_fully_masked_horizon():
    torch.manual_seed(0)
    b = 4
    logits = {h: torch.randn(b, 2) for h in HORIZONS}
    y = torch.randint(0, 2, (b, len(HORIZONS)))
    r = torch.zeros(b, len(HORIZONS))
    mask = torch.ones(b, len(HORIZONS))
    mask[:, 3] = 0.0  # mask 1Y entirely
    weights = {h: 1.0 for h in HORIZONS}

    # returns=None -> classification only; sum of per-horizon mean CE over 3 horizons.
    got = masked_loss(logits, None, y, r, mask, weights, _NO_SCALE, 0.0)
    expected = sum(
        torch.nn.functional.cross_entropy(logits[h], y[:, j])
        for j, h in enumerate(HORIZONS[:3])
    )
    assert torch.allclose(got, expected, atol=1e-6)


def test_masked_loss_respects_per_sample_mask_denominator():
    b = 4
    logits = {h: torch.zeros(b, 2) for h in HORIZONS}  # uniform -> CE = ln2 each
    y = torch.zeros(b, len(HORIZONS), dtype=torch.long)
    r = torch.zeros(b, len(HORIZONS))
    mask = torch.zeros(b, len(HORIZONS))
    mask[0, 0] = 1.0  # only one valid entry, in 1M
    weights = {h: 1.0 for h in HORIZONS}
    got = masked_loss(logits, None, y, r, mask, weights, _NO_SCALE, 0.0).item()
    assert abs(got - torch.log(torch.tensor(2.0)).item()) < 1e-6


def test_masked_loss_adds_regression_term():
    b = 4
    logits = {h: torch.zeros(b, 2) for h in HORIZONS}
    y = torch.zeros(b, len(HORIZONS), dtype=torch.long)
    r = torch.full((b, len(HORIZONS)), 0.5)        # target log-return 0.5
    returns = {h: torch.zeros(b) for h in HORIZONS}  # predict 0 -> residual -0.5
    mask = torch.ones(b, len(HORIZONS))
    w = {h: 1.0 for h in HORIZONS}
    cls_only = masked_loss(logits, None, y, r, mask, w, _NO_SCALE, 1.0).item()
    with_reg = masked_loss(logits, returns, y, r, mask, w, _NO_SCALE, 1.0).item()
    # Huber(0.5) = 0.5 * 0.5^2 = 0.125 per horizon, x4 horizons = 0.5.
    assert abs((with_reg - cls_only) - 0.5) < 1e-5


# =============================================================
# metrics
# =============================================================


def test_horizon_metrics_perfect_predictions():
    b = 8
    y = torch.randint(0, 2, (b, len(HORIZONS)))
    logits = _logits_pointing_at(y)
    r = torch.zeros(b, len(HORIZONS))
    mask = torch.ones(b, len(HORIZONS))
    m = horizon_metrics(logits, None, y, r, mask)
    for j, h in enumerate(HORIZONS):
        assert m[h]["acc"] == 1.0
        assert m[h]["n"] == b
        assert m[h]["brier"] < 0.01
        # base_rate is the up-fraction; lift = acc - majority-class baseline.
        assert abs(m[h]["base_rate"] - y[:, j].float().mean().item()) < 1e-9
        br = m[h]["base_rate"]
        assert abs(m[h]["lift"] - (m[h]["acc"] - max(br, 1.0 - br))) < 1e-9
    assert abs(mean_accuracy(m) - 1.0) < 1e-9


def test_horizon_lift_is_zero_for_majority_class_predictor():
    # All labels up, model always predicts up -> acc == base_rate -> lift 0.
    b = 10
    y = torch.ones(b, len(HORIZONS), dtype=torch.long)
    logits = {h: torch.tensor([[-2.0, 2.0]]).repeat(b, 1) for h in HORIZONS}
    r = torch.zeros(b, len(HORIZONS))
    mask = torch.ones(b, len(HORIZONS))
    m = horizon_metrics(logits, None, y, r, mask)
    for h in HORIZONS:
        assert m[h]["acc"] == 1.0
        assert m[h]["base_rate"] == 1.0
        assert abs(m[h]["lift"]) < 1e-9   # no skill above the base rate


def test_horizon_metrics_regression_rmse_and_mae():
    b = 6
    y = torch.zeros(b, len(HORIZONS), dtype=torch.long)
    logits = {h: torch.zeros(b, 2) for h in HORIZONS}
    r = torch.full((b, len(HORIZONS)), 0.2)
    returns = {h: torch.full((b,), 0.1) for h in HORIZONS}  # residual -0.1
    mask = torch.ones(b, len(HORIZONS))
    m = horizon_metrics(logits, returns, y, r, mask)
    for h in HORIZONS:
        assert abs(m[h]["ret_rmse"] - 0.1) < 1e-5
        assert abs(m[h]["ret_mae"] - 0.1) < 1e-5


def test_horizon_metrics_handles_all_masked():
    b = 4
    y = torch.zeros(b, len(HORIZONS), dtype=torch.long)
    logits = {h: torch.randn(b, 2) for h in HORIZONS}
    r = torch.zeros(b, len(HORIZONS))
    mask = torch.zeros(b, len(HORIZONS))
    m = horizon_metrics(logits, None, y, r, mask)
    assert all(m[h]["n"] == 0 for h in HORIZONS)
    # nan != nan -> confirms base_rate/lift are NaN when nothing is valid.
    assert all(m[h]["base_rate"] != m[h]["base_rate"] for h in HORIZONS)
    assert all(m[h]["lift"] != m[h]["lift"] for h in HORIZONS)


# =============================================================
# end-to-end train() on synthetic signal
# =============================================================


def _trend_frame(n_days: int, daily_drift: float, tid: int, emb: int) -> TickerFrame:
    # Multiplicative so prices stay positive (features.py rejects non-positive
    # adj_close in the log-return, as real prices never go <= 0).
    d0 = date(2022, 1, 3)
    prices = [
        {"trade_date": d0 + timedelta(days=i), "adj_close": 100.0 * (1.0 + daily_drift) ** i,
         "volume": 1_000_000}
        for i in range(n_days)
    ]
    return TickerFrame(tid, emb, f"T{tid}", prices, [], [])


def test_train_loop_fits_synthetic_signal():
    # Two tickers with opposite trends: ticker 1 always rises (labels 1),
    # ticker 2 always falls (labels 0). The model must learn to separate them
    # via the ticker embedding. Reduced config so it runs fast on CPU.
    up = assemble_ticker_samples(_trend_frame(520, +0.002, tid=1, emb=1), stride=10)
    down = assemble_ticker_samples(_trend_frame(520, -0.002, tid=2, emb=2), stride=10)
    samples = up + down
    assert samples

    # Manual split (time-split logic is tested separately); hold a few out for val.
    val = samples[::6]
    val_ids = {id(s) for s in val}
    train_s = [s for s in samples if id(s) not in val_ids]
    splits = {"train": train_s, "val": val, "holdout": []}

    cfg = PatchTSTConfig(d_model=32, n_heads=4, d_ff=64, n_layers=2,
                         dropout=0.0, drop_path=0.0, gate_hidden=16)
    tcfg = TrainConfig(epochs=80, batch_size=16, lr=1e-3, patience=80, seed=0)

    result = train(splits, num_tickers=3, model_cfg=cfg, train_cfg=tcfg,
                   device="cpu", log=lambda *_: None)

    assert result.history
    assert set(result.val_accuracy) == set(HORIZONS)
    # Loss should fall meaningfully and the separable signal should be learned well.
    assert result.history[-1]["train_loss"] < result.history[0]["train_loss"]
    assert result.history[-1]["val_mean_acc"] > 0.8
