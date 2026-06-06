"""Train today's LightGBM ranker bundle and write relative-rank predictions.

This is the practical production path while the transformer is shelved/reworked:
fit one shallow GBDT per horizon on all labeled historical S&P rows, score the
latest active-ticker cross-section, and upsert rank-like projections to
`predictions`.

The current schema only has `direction_prob` as a required bounded score. For a
rank-target GBDT, we store the clipped predicted percentile rank there and leave
`predicted_return` null. Dashboard copy should label it as a relative-rank score,
not a calibrated probability.
"""

from __future__ import annotations

import os

# libomp coexistence: this process loads torch (via dataset -> model) AND
# lightgbm (when models are fit, or unpickled in the --score-ticker path). Each
# ships its own OpenMP runtime, and a multi-threaded team from one alongside the
# other segfaults on macOS — exactly what crashed the reloaded-model predict path
# (SIGSEGV at load_bundle). Forcing single-threaded OpenMP BEFORE those imports
# sidesteps it. Fit already uses n_jobs=1; this extends the same guard to the
# unpickle/predict path. setdefault so an explicit override still wins.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import asyncio
import hashlib
import json
import pickle
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np

from backend.config import get_settings
from backend.ingestion.db import pool_context
from backend.ml.dataset import (
    build_calendar_grid,
    load_frames,
    load_frames_cached,
    write_frames_cache,
)
from backend.ml.gbm_baseline import (
    FEATURE_COLS,
    PRODUCTION_HORIZON_SPECS,
    HorizonSpec,
    LGBMConfig,
    blend_gbdt_linear,
    fit_linear_model,
    fit_lgbm_model,
    prepare_panel,
)
from backend.ml.model import HORIZONS

DEFAULT_INFERENCE_HORIZONS = ("3M", "6M", "1Y")


def _target_col(horizon: str, target_mode: str) -> str:
    return f"r_{horizon}" if target_mode == "return" else f"y_{horizon}_{target_mode}"


def _spec_feature_cols(spec: HorizonSpec) -> list[str]:
    """Per-horizon feature list — falls back to production FEATURE_COLS."""
    return spec.feature_cols if spec.feature_cols is not None else list(FEATURE_COLS)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fit_horizon_models(
    panel,
    specs: dict[str, HorizonSpec],
    seed: int,
    as_of: date,
    n_seeds: int = 1,
    exclude_ids: set[int] | None = None,
):
    """Fit one LightGBM model ensemble per horizon, each with its own spec.

    Returns `(models, train_windows, trained_ids, linear_models)`:
      - `models[h]` = `[model_seed_0, ...]` — always a list even when n_seeds=1,
        so `score_current_cross_section` can average uniformly.
      - `linear_models[h]` = `(ridge_model, blend_weight)` only for horizons whose
        spec sets `linear_blend > 0`; absent otherwise (pure GBDT).

    `exclude_ids` drops those ticker_ids from the training rows (used to keep
    user-added off-index names out of the model — they are still scored, just
    never trained on). Default None preserves the original behavior exactly.
    """
    models: dict[str, list] = {}
    linear_models: dict[str, tuple] = {}
    train_windows: dict[str, dict] = {}
    trained_ids: set[int] = set()
    for i, (h, spec) in enumerate(specs.items()):
        m_col = f"mask_{h}"
        t_col = _target_col(h, spec.target_mode)
        cols = _spec_feature_cols(spec)
        train = panel[
            (panel["date"] < as_of)
            & panel[m_col].astype(bool)
            & panel[t_col].notna()
        ]
        if exclude_ids:
            train = train[~train["ticker_id"].isin(exclude_ids)]
        if train.empty:
            raise ValueError(
                f"no labeled training rows for {h} target={spec.target_mode}"
            )
        models[h] = [
            fit_lgbm_model(
                train, t_col, spec.lgb_cfg, seed=seed + i + s * 997, shuffle=False, feature_cols=cols
            )
            for s in range(max(1, n_seeds))
        ]
        if getattr(spec, "linear_blend", 0.0) > 0:
            linear_models[h] = (
                fit_linear_model(
                    train, t_col, feature_cols=cols, alpha=spec.ridge_alpha, seed=seed + i
                ),
                float(spec.linear_blend),
            )
        train_windows[h] = {
            "start": min(train["date"]),
            "end": max(train["date"]),
            "rows": int(train.shape[0]),
            "tickers": int(train["ticker_id"].nunique()),
        }
        trained_ids.update(int(v) for v in train["ticker_id"].unique())
    return models, train_windows, trained_ids, linear_models


def score_current_cross_section(
    panel,
    models: dict,
    specs: dict[str, HorizonSpec] | tuple[str, ...],
    as_of: date,
    active_ids: set[int],
    linear_models: dict | None = None,
):
    """Return prediction dicts for active tickers present in the `as_of` cross-section.

    `specs` accepts either the new per-horizon `HorizonSpec` mapping (so each
    model predicts on its own `feature_cols`) or — for backward compatibility
    with the existing test — a plain horizons tuple, in which case all horizons
    predict on the default `FEATURE_COLS`.

    `linear_models` (optional) maps a horizon to `(ridge_model, blend_weight)`;
    when present the GBDT and ridge predictions are rank-blended before the final
    rank-transform, matching the walk-forward stack that validated the weight.
    """
    linear_models = linear_models or {}
    current = panel[(panel["date"] == as_of) & panel["ticker_id"].isin(active_ids)].copy()
    if current.empty:
        raise ValueError(f"no active ticker rows available for as_of={as_of}")

    if isinstance(specs, dict):
        iter_pairs = [(h, _spec_feature_cols(s)) for h, s in specs.items()]
    else:
        iter_pairs = [(h, list(FEATURE_COLS)) for h in specs]

    import pandas as pd

    rows: list[dict] = []
    n = len(current)
    for h, cols in iter_pairs:
        # Predictions are mapped to within-cross-section percentile rank in [0, 1]
        # before storage. The previous "clip to [0,1]" path worked for `rank`-mode
        # training where preds were already roughly in that range, but a
        # regression target like `beta_resid` outputs log returns (~[-0.05, +0.1])
        # which would mostly clip to 0 and destroy the ranking. Rank-transforming
        # makes the dashboard semantics ("relative rank") consistent across all
        # target modes — pandas average-rank handles ties; single-name
        # cross-sections collapse to 0.5.
        model_or_list = models[h]
        if isinstance(model_or_list, list):
            preds = np.mean(
                [np.asarray(m.predict(current[cols]), dtype=float) for m in model_or_list], axis=0
            )
        else:
            preds = np.asarray(model_or_list.predict(current[cols]), dtype=float)
        if h in linear_models:
            ridge, weight = linear_models[h]
            lin_pred = np.asarray(ridge.predict(current[cols].to_numpy(dtype=float)), dtype=float)
            preds = blend_gbdt_linear(preds, lin_pred, weight)
        if n > 1:
            ranks_raw = pd.Series(preds).rank(method="average").to_numpy()
            ranks = (ranks_raw - 1.0) / (n - 1)
        else:
            ranks = np.full_like(preds, 0.5, dtype=float)
        for ticker_id, rank in zip(current["ticker_id"].to_numpy(), ranks, strict=True):
            rows.append({
                "ticker_id": int(ticker_id),
                "horizon": h,
                "relative_rank": float(rank),
            })
    return rows


def apply_cross_horizon_shrink(
    prediction_rows: list[dict],
    source: str = "6M",
    target: str = "1Y",
    weight: float = 0.0,
) -> list[dict]:
    """Shrink the `target` horizon's percentile rank toward the `source` horizon's.

    James-Stein-style strength borrowing: the long horizon is power-limited
    (annual overlapping labels ⇒ few independent blocks) and its standalone rank
    is noisy, while 6M is the reliable signal and the horizons are highly
    correlated. Blending the target rank toward the source rank trades a little
    bias for a meaningful variance reduction. `weight` in [0, 1]; 0 = no change.

    Mutates and returns `prediction_rows`. Target rows are re-ranked to a clean
    percentile after blending so stored `relative_rank` semantics are unchanged.
    Names absent from the source keep their original rank.
    """
    if weight <= 0:
        return prediction_rows
    src = {
        r["ticker_id"]: r["relative_rank"]
        for r in prediction_rows
        if r["horizon"] == source
    }
    tgt_rows = [r for r in prediction_rows if r["horizon"] == target]
    if not src or len(tgt_rows) < 2:
        return prediction_rows

    import pandas as pd

    blended = np.array([
        (1.0 - weight) * r["relative_rank"] + weight * src[r["ticker_id"]]
        if r["ticker_id"] in src else r["relative_rank"]
        for r in tgt_rows
    ], dtype=float)
    ranks = (pd.Series(blended).rank(method="average").to_numpy() - 1.0) / (len(tgt_rows) - 1)
    for r, rank in zip(tgt_rows, ranks, strict=True):
        r["relative_rank"] = float(rank)
    return prediction_rows


def apply_rank_smoothing(
    prediction_rows: list[dict],
    specs: dict,
    prior_ranks: dict[tuple[int, str], list[float]],
) -> list[dict]:
    """EWMA-smooth each name's percentile rank toward its most recent stored rank.

    For horizons whose `HorizonSpec.smooth_span > 0` (3M/1Y in production), blend the
    current `relative_rank` with the name's last stored (already-smoothed) rank using
    `alpha = 2/(span+1)`, then re-rank within the horizon to a clean percentile so the
    stored value stays a uniform [0,1] rank. This is the online one-step form of the
    walk-forward's `ewma_rank_by_ticker`: the previously stored rank is the carried
    state. Names with no prior keep their raw rank; single-name horizons are skipped.

    `prior_ranks` maps (ticker_id, horizon) -> [most-recent-first stored ranks]; only
    the newest (index 0) is used. Mutates and returns `prediction_rows`.
    """
    if not isinstance(specs, dict):
        return prediction_rows

    import pandas as pd

    for h, spec in specs.items():
        span = getattr(spec, "smooth_span", 0)
        if span <= 0:
            continue
        rows = [r for r in prediction_rows if r["horizon"] == h]
        if len(rows) < 2:
            continue
        alpha = 2.0 / (span + 1.0)
        blended = np.array([
            alpha * r["relative_rank"] + (1.0 - alpha) * prior_ranks[(r["ticker_id"], h)][0]
            if prior_ranks.get((r["ticker_id"], h)) else r["relative_rank"]
            for r in rows
        ], dtype=float)
        ranks = (pd.Series(blended).rank(method="average").to_numpy() - 1.0) / (len(rows) - 1)
        for r, rank in zip(rows, ranks, strict=True):
            r["relative_rank"] = float(rank)
    return prediction_rows


def rank_stability(ranks: list[float]) -> float | None:
    """Std of predicted percentile rank across the last few scoring dates.

    Low std => the model has been ranking this name consistently; high std => its
    relative view of the name has been moving around. Returns None when fewer than
    two scoring dates are available (stability undefined). This is stored in
    predictions.confidence in place of the old distance-from-median heuristic.
    Population std (ddof=0) so it stays defined for two points.
    """
    vals = [float(r) for r in ranks if r is not None]
    if len(vals) < 2:
        return None
    return float(np.std(np.asarray(vals, dtype=float)))


async def fetch_prior_ranks(
    pool, ticker_ids, horizons, before: date, n_prior: int = 2
) -> dict[tuple[int, str], list[float]]:
    """Most-recent prior predicted ranks per (ticker_id, horizon), newest first.

    One value per distinct prior scoring date (the latest model that scored that
    date), at most `n_prior` of them, all strictly before `before`.
    """
    rows = await pool.fetch(
        """
        select distinct on (ticker_id, horizon, as_of_date)
               ticker_id, horizon, as_of_date, direction_prob
          from predictions
         where horizon = any($1::text[]) and ticker_id = any($2::bigint[])
           and as_of_date < $3
         order by ticker_id, horizon, as_of_date desc, created_at desc
        """,
        list(horizons),
        [int(t) for t in ticker_ids],
        before,
    )
    out: dict[tuple[int, str], list[float]] = {}
    for r in rows:
        key = (int(r["ticker_id"]), r["horizon"])
        lst = out.setdefault(key, [])
        if len(lst) < n_prior and r["direction_prob"] is not None:
            lst.append(float(r["direction_prob"]))
    return out


def save_bundle(path: Path, bundle: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(bundle, f)
    return _sha256(path)


def load_bundle(path: Path) -> dict:
    """Reload a saved model artifact (the dict written by save_bundle)."""
    with path.open("rb") as f:
        return pickle.load(f)


def _specs_from_serialized(serialized: dict) -> dict[str, HorizonSpec]:
    """Rebuild per-horizon HorizonSpecs from a saved artifact's `specs` dict.

    Uses the spec stored WITH the model (its exact training feature_cols), not the
    live PRODUCTION_HORIZON_SPECS, so reloaded scoring can't drift if the code's
    feature packs change after the model was promoted.
    """
    out: dict[str, HorizonSpec] = {}
    for h, d in serialized.items():
        out[h] = HorizonSpec(
            target_mode=d["target_mode"],
            lgb_cfg=LGBMConfig(**d["lgb_cfg"]),
            feature_cols=d.get("feature_cols"),
            linear_blend=d.get("linear_blend", 0.0),
            ridge_alpha=d.get("ridge_alpha", 10.0),
            smooth_span=d.get("smooth_span", 0),
        )
    return out


async def upsert_predictions(
    pool, model_version_id, as_of: date, prediction_rows: list[dict]
) -> None:
    await pool.executemany(
        """
        insert into predictions (
            ticker_id, model_version_id, as_of_date, horizon, direction_prob,
            predicted_return, confidence, cold_start
        ) values ($1,$2,$3,$4,$5,$6,$7,false)
        on conflict (ticker_id, model_version_id, as_of_date, horizon) do update set
            direction_prob   = excluded.direction_prob,
            predicted_return = excluded.predicted_return,
            confidence       = excluded.confidence,
            cold_start       = excluded.cold_start,
            created_at       = now()
        """,
        [
            (
                r["ticker_id"],
                model_version_id,
                as_of,
                r["horizon"],
                r["relative_rank"],
                None,
                r["confidence"],
            )
            for r in prediction_rows
        ],
    )


def _serialize_spec(spec: HorizonSpec) -> dict:
    """JSON-safe view of a HorizonSpec for the model-version `config` column."""
    return {
        "target_mode": spec.target_mode,
        "lgb_cfg": asdict(spec.lgb_cfg),
        "feature_cols": _spec_feature_cols(spec),
        "linear_blend": getattr(spec, "linear_blend", 0.0),
        "ridge_alpha": getattr(spec, "ridge_alpha", 10.0),
        "smooth_span": getattr(spec, "smooth_span", 0),
    }


def build_specs_from_args(args) -> dict[str, HorizonSpec]:
    """Resolve per-horizon training specs from CLI args.

    Resolution order:
      1. Start from `PRODUCTION_HORIZON_SPECS` (the validated defaults).
      2. If `--target X` is given, override target_mode for every requested
         horizon (the legacy single-mode behavior, useful for ablations).
      3. Restrict to `args.horizons` — only those models get trained today.
    Per-horizon hyperparameter / feature overrides are not exposed on the CLI
    yet; edit `PRODUCTION_HORIZON_SPECS` directly when promoting a sweep result.
    """
    base: dict[str, HorizonSpec] = {}
    for h in args.horizons:
        spec = PRODUCTION_HORIZON_SPECS.get(h, HorizonSpec())
        if args.target is not None:
            spec = HorizonSpec(
                target_mode=args.target,
                lgb_cfg=spec.lgb_cfg,
                feature_cols=spec.feature_cols,
                linear_blend=spec.linear_blend,
                ridge_alpha=spec.ridge_alpha,
                smooth_span=spec.smooth_span,
            )
        base[h] = spec
    return base


async def run(args) -> None:
    horizons = tuple(args.horizons)
    invalid = sorted(set(horizons) - set(HORIZONS))
    if invalid:
        raise SystemExit(f"invalid horizon(s): {', '.join(invalid)}")

    specs = build_specs_from_args(args)
    # Bulk historical load_frames pull can exceed the default 60s command timeout
    # through the Supabase pooler under load; give the inference read path headroom.
    async with pool_context(command_timeout=300) as pool:
        frames = await load_frames(pool)
        if not frames:
            raise SystemExit("no ticker frames loaded")
        # Warm the on-disk frame cache with the freshly-pulled frames (no extra
        # egress) so the single-ticker add path (score_single_ticker) reads the
        # universe from disk instead of re-pulling the full history per add.
        write_frames_cache(frames)
        active_rows = await pool.fetch("select ticker_id, symbol from tickers where active = true")
        active_ids = {int(r["ticker_id"]) for r in active_rows}
        # User-added off-index names are scored (they're active) but must never
        # train the model — exclude them from the fit so the production model and
        # the S&P ranking stay exactly what they'd be on the index alone.
        user_added_rows = await pool.fetch("select ticker_id from tickers where user_added = true")
        user_added_ids = {int(r["ticker_id"]) for r in user_added_rows}
        grid = build_calendar_grid(frames)
        as_of = date.fromisoformat(args.as_of) if args.as_of else max(grid)
        grid = sorted(set(grid + [as_of]))

        print(f"loaded {len(frames)} tickers ({len(active_ids)} active); preparing panel ...")
        # Rank-normalize every feature any horizon uses (not just FEATURE_COLS), so
        # promoted packs (e.g. revenue_surprise on 6M/1Y) are normalized exactly as
        # they were during the walk-forward validation that promoted them.
        rank_cols = sorted({c for s in specs.values() for c in _spec_feature_cols(s)})
        panel = prepare_panel(frames, grid, n_buckets=args.n_buckets, rank_cols=rank_cols)
        if panel.empty:
            raise SystemExit("empty panel (not enough history?)")

        models, train_windows, trained_ids, linear_models = fit_horizon_models(
            panel, specs, args.seed, as_of, n_seeds=args.n_seeds, exclude_ids=user_added_ids
        )
        prediction_rows = score_current_cross_section(
            panel, models, specs, as_of, active_ids, linear_models=linear_models
        )
        # Cross-date smoothing: EWMA each name's rank toward its last stored rank
        # for horizons whose spec sets smooth_span>0 (3M/1Y in production). Fetch
        # priors once here and reuse them for the confidence/stability metric below.
        ticker_ids = sorted({r["ticker_id"] for r in prediction_rows})
        prior = await fetch_prior_ranks(pool, ticker_ids, horizons, as_of)
        prediction_rows = apply_rank_smoothing(prediction_rows, specs, prior)
        if args.shrink_1y_toward_6m > 0 and {"6M", "1Y"} <= set(horizons):
            prediction_rows = apply_cross_horizon_shrink(
                prediction_rows, source="6M", target="1Y",
                weight=args.shrink_1y_toward_6m,
            )

        targets_str = ",".join(f"{h}={specs[h].target_mode}" for h in horizons)
        print(
            f"as_of={as_of} targets={targets_str} predictions={len(prediction_rows)}"
        )
        for h in horizons:
            ranks = [r["relative_rank"] for r in prediction_rows if r["horizon"] == h]
            print(
                f"  {h}: target={specs[h].target_mode}  "
                f"train_rows={train_windows[h]['rows']}  "
                f"train_end={train_windows[h]['end']}  "
                f"rank_range=[{min(ranks):.3f}, {max(ranks):.3f}]"
            )

        if args.dry_run:
            print("--dry-run: skipping model_versions and predictions writes")
            return

        settings = get_settings()
        model_version_id = await pool.fetchval("select gen_random_uuid()")
        weights_path = settings.models_dir / f"{model_version_id}_gbdt.pkl"
        train_windows_json = {
            h: {**v, "start": v["start"].isoformat(), "end": v["end"].isoformat()}
            for h, v in train_windows.items()
        }
        serialized_specs = {h: _serialize_spec(s) for h, s in specs.items()}
        artifact = {
            "model_type": "lightgbm_gbdt_ranker",
            "models": models,
            "linear_models": linear_models,      # {h: (ridge, weight)} when blended
            "horizons": horizons,
            "specs": serialized_specs,           # per-horizon target / cfg / features
            "train_windows": train_windows_json,
            "as_of": as_of.isoformat(),
        }
        config = {
            "model_type": "lightgbm_gbdt_ranker",
            "score_semantics": (
                "direction_prob stores within-cross-section percentile rank of the "
                "model's prediction (rank-transformed at score time so semantics are "
                "consistent across target modes)"
            ),
            "confidence_semantics": (
                "rank stability: std of predicted rank over the last up-to-3 scoring "
                "dates (lower = steadier; null if <2 dates available)"
            ),
            "horizons": list(horizons),
            "specs": serialized_specs,
            "train_windows": train_windows_json,
            "as_of": as_of.isoformat(),
        }
        # Rank stability: std of each name's predicted rank across the last
        # up-to-3 scoring dates (this run + the two most recent prior dates).
        # `prior` was fetched above (before smoothing) and is reused here; the
        # series now ends on the smoothed rank, which is what we store.
        for r in prediction_rows:
            series = prior.get((r["ticker_id"], r["horizon"]), []) + [r["relative_rank"]]
            r["confidence"] = rank_stability(series)

        sha = save_bundle(weights_path, artifact)
        await insert_model_version_with_id(
            pool,
            model_version_id,
            weights_path,
            sha,
            config,
            train_windows,
            as_of,
            len(trained_ids),
        )
        await upsert_predictions(pool, model_version_id, as_of, prediction_rows)
        print(f"saved candidate GBDT model_version {model_version_id}; upserted predictions")


async def insert_model_version_with_id(
    pool,
    model_version_id,
    weights_path: Path,
    weights_sha256: str,
    config: dict,
    train_windows: dict,
    as_of: date,
    n_tickers_trained: int,
):
    starts = [v["start"] for v in train_windows.values()]
    ends = [v["end"] for v in train_windows.values()]
    await pool.execute(
        """
        insert into model_versions (
            model_version_id, training_window_start, training_window_end,
            holdout_window_start, holdout_window_end, weights_path, weights_sha256,
            n_params, n_tickers_trained, val_loss, directional_accuracy,
            holdout_metrics, status, config
        ) values ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,'candidate',$13)
        """,
        model_version_id,
        min(starts),
        max(ends),
        as_of,
        as_of,
        str(weights_path),
        weights_sha256,
        None,
        n_tickers_trained,
        None,
        json.dumps({}),
        json.dumps({}),
        json.dumps(config),
    )


async def _resolve_production_model(pool) -> tuple[str, str]:
    """Return (model_version_id, weights_path) for the production model.

    Falls back to the most recently created candidate when no model is promoted
    yet (mirrors the dashboard's _resolve_active_model), so a fresh setup still
    scores against whatever model wrote the current cross-section.
    """
    row = await pool.fetchrow(
        "select model_version_id, weights_path from model_versions "
        "where status = 'production' order by promoted_at desc nulls last limit 1"
    )
    if row is None:
        row = await pool.fetchrow(
            "select model_version_id, weights_path from model_versions "
            "order by created_at desc limit 1"
        )
    if row is None:
        raise SystemExit("no model_versions row — train a model before scoring tickers")
    return str(row["model_version_id"]), str(row["weights_path"])


async def score_single_ticker(pool, symbol: str) -> dict:
    """Score ONE already-ingested ticker against the current S&P cross-section.

    Reloads the production model (no retraining), ranks the ticker against the
    full cross-section ∪ {this ticker} at the model's latest stored prediction
    date, and writes ONLY this ticker's rows under the existing production
    model_version_id. Existing S&P prediction rows are never read or written, so
    the S&P ranking is untouched.

    Returns a structured outcome dict (no DB-status side effects — the orchestrator
    owns the ingestion_runs row): {"status": "scored", "symbol", "as_of", "ranks"}
    or {"status": "insufficient_history", "symbol"}.
    """
    model_version_id, weights_path = await _resolve_production_model(pool)
    as_of = await pool.fetchval(
        "select max(as_of_date) from predictions where model_version_id = $1",
        model_version_id,
    )
    if as_of is None:
        raise SystemExit(
            f"production model {model_version_id} has no predictions yet — "
            "run a full inference pass before scoring individual tickers"
        )

    trow = await pool.fetchrow(
        "select ticker_id from tickers where upper(symbol) = upper($1)", symbol
    )
    if trow is None:
        raise SystemExit(f"ticker not found (ingest it first): {symbol}")
    new_id = int(trow["ticker_id"])

    path = Path(weights_path)
    if not path.exists():
        raise SystemExit(f"model artifact missing on disk: {path}")
    bundle = load_bundle(path)
    models = bundle["models"]
    linear_models = bundle.get("linear_models", {})
    specs = _specs_from_serialized(bundle["specs"])

    # DB-efficient load: reuse the cached universe (no full-universe re-pull) and
    # fetch only the new ticker fresh, then splice it in (dropping any stale copy
    # already in the cache).
    universe = await load_frames_cached(pool)
    new_frames = await load_frames(pool, symbols=[symbol])
    if not new_frames:
        raise SystemExit(f"no frame data for {symbol} — ingestion incomplete")
    frames = [f for f in universe if f.ticker_id != new_id] + new_frames

    grid = build_calendar_grid(frames)
    grid = sorted(set(grid + [as_of]))
    rank_cols = sorted({c for s in specs.values() for c in _spec_feature_cols(s)})
    panel = prepare_panel(frames, grid, n_buckets=5, rank_cols=rank_cols)

    # Rank against the S&P cross-section ONLY (+ this ticker), so the percentile
    # means "vs the S&P" — not vs other off-index names a user happens to have added.
    index_rows = await pool.fetch(
        "select ticker_id from tickers where active = true and user_added = false"
    )
    active_ids = {int(r["ticker_id"]) for r in index_rows} | {new_id}

    prediction_rows = score_current_cross_section(
        panel, models, specs, as_of, active_ids, linear_models=linear_models
    )
    mine = [r for r in prediction_rows if r["ticker_id"] == new_id]
    if not mine:
        # build_ticker_rows dropped it: <252 trading days of history at as_of.
        return {"status": "insufficient_history", "symbol": symbol}

    horizons = list(specs.keys())
    # Smooth against the full (reconstructed) cross-section so the new name's rank
    # is re-ranked among the smoothed S&P, consistent with a full inference pass.
    # Mutates `mine`'s rows in place (same dict objects). Needs priors for all
    # scored ids; one indexed query on a user-initiated add — cost is negligible.
    all_ids = sorted({r["ticker_id"] for r in prediction_rows})
    prior = await fetch_prior_ranks(pool, all_ids, horizons, as_of)
    apply_rank_smoothing(prediction_rows, specs, prior)
    for r in mine:
        series = prior.get((new_id, r["horizon"]), []) + [r["relative_rank"]]
        r["confidence"] = rank_stability(series)

    await upsert_predictions(pool, model_version_id, as_of, mine)
    return {
        "status": "scored",
        "symbol": symbol,
        "as_of": as_of.isoformat(),
        "model_version_id": model_version_id,
        "ranks": {r["horizon"]: r["relative_rank"] for r in mine},
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Train GBDT rankers and write current predictions")
    p.add_argument("--as-of", help="score this date (default = latest month/grid date)")
    p.add_argument("--horizons", nargs="+", default=list(DEFAULT_INFERENCE_HORIZONS),
                   choices=list(HORIZONS))
    p.add_argument(
        "--target", default=None,
        choices=["return", "rank", "quantile", "sector_return", "beta_resid",
                 "beta_sector_resid"],
        help="legacy global target override — sets the training target for EVERY "
             "horizon in --horizons to this mode. By default each horizon uses its "
             "own validated target from PRODUCTION_HORIZON_SPECS in gbm_baseline.py "
             "(currently 3M/6M/1Y=sector_return, 1M=rank).",
    )
    p.add_argument("--n-buckets", type=int, default=5)
    p.add_argument("--shrink-1y-toward-6m", type=float, default=0.0, metavar="W",
                   help="James-Stein cross-horizon shrink: blend the 1Y rank toward "
                        "the 6M rank at this weight (0..1) to borrow strength from "
                        "the reliable horizon. 0 = off (default). Requires both "
                        "horizons in --horizons")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--n-seeds", type=int, default=8,
                   help="seed-ensemble size: fit this many models per horizon and "
                        "average their raw predictions before rank-transforming "
                        "(default 8; reduces seed variance at low compute cost)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--score-ticker", metavar="SYMBOL",
                   help="score ONE already-ingested ticker against the current S&P "
                        "cross-section using the existing production model (no "
                        "retraining, writes only that ticker's rows). Skips the "
                        "training/scoring/upsert pass entirely.")
    args = p.parse_args()
    if args.score_ticker:
        asyncio.run(_run_score_ticker(args.score_ticker))
        return
    asyncio.run(run(args))


async def _run_score_ticker(symbol: str) -> None:
    async with pool_context(command_timeout=300) as pool:
        result = await score_single_ticker(pool, symbol)
    # Single machine-readable line the add_ticker orchestrator parses from stdout.
    print(json.dumps(result))
    if result["status"] == "scored":
        print(f"scored {symbol} as_of={result['as_of']} ranks={result['ranks']}")
    else:
        print(f"{symbol}: {result['status']}")


if __name__ == "__main__":
    main()
