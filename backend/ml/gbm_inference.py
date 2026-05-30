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
from backend.ml.dataset import build_calendar_grid, load_frames
from backend.ml.gbm_baseline import (
    FEATURE_COLS,
    PRODUCTION_HORIZON_SPECS,
    HorizonSpec,
    LGBMConfig,
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
):
    """Fit one LightGBM model ensemble per horizon, each with its own spec.

    Returns `{horizon: [model_seed_0, model_seed_1, ...]}` — always a list even
    when n_seeds=1, so `score_current_cross_section` can average uniformly.
    """
    models: dict[str, list] = {}
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
        train_windows[h] = {
            "start": min(train["date"]),
            "end": max(train["date"]),
            "rows": int(train.shape[0]),
            "tickers": int(train["ticker_id"].nunique()),
        }
        trained_ids.update(int(v) for v in train["ticker_id"].unique())
    return models, train_windows, trained_ids


def score_current_cross_section(
    panel,
    models: dict,
    specs: dict[str, HorizonSpec] | tuple[str, ...],
    as_of: date,
    active_ids: set[int],
):
    """Return prediction dicts for active tickers present in the `as_of` cross-section.

    `specs` accepts either the new per-horizon `HorizonSpec` mapping (so each
    model predicts on its own `feature_cols`) or — for backward compatibility
    with the existing test — a plain horizons tuple, in which case all horizons
    predict on the default `FEATURE_COLS`.
    """
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
            )
        base[h] = spec
    return base


async def run(args) -> None:
    horizons = tuple(args.horizons)
    invalid = sorted(set(horizons) - set(HORIZONS))
    if invalid:
        raise SystemExit(f"invalid horizon(s): {', '.join(invalid)}")

    specs = build_specs_from_args(args)
    async with pool_context() as pool:
        frames = await load_frames(pool)
        if not frames:
            raise SystemExit("no ticker frames loaded")
        active_rows = await pool.fetch("select ticker_id, symbol from tickers where active = true")
        active_ids = {int(r["ticker_id"]) for r in active_rows}
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

        models, train_windows, trained_ids = fit_horizon_models(
            panel, specs, args.seed, as_of, n_seeds=args.n_seeds
        )
        prediction_rows = score_current_cross_section(panel, models, specs, as_of, active_ids)

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
        ticker_ids = sorted({r["ticker_id"] for r in prediction_rows})
        prior = await fetch_prior_ranks(pool, ticker_ids, horizons, as_of)
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


def main() -> None:
    p = argparse.ArgumentParser(description="Train GBDT rankers and write current predictions")
    p.add_argument("--as-of", help="score this date (default = latest month/grid date)")
    p.add_argument("--horizons", nargs="+", default=list(DEFAULT_INFERENCE_HORIZONS),
                   choices=list(HORIZONS))
    p.add_argument(
        "--target", default=None,
        choices=["return", "rank", "quantile", "sector_return", "beta_resid"],
        help="legacy global target override — sets the training target for EVERY "
             "horizon in --horizons to this mode. By default each horizon uses its "
             "own validated target from PRODUCTION_HORIZON_SPECS in gbm_baseline.py "
             "(currently 1M/3M/6M=rank, 1Y=beta_resid).",
    )
    p.add_argument("--n-buckets", type=int, default=5)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--n-seeds", type=int, default=8,
                   help="seed-ensemble size: fit this many models per horizon and "
                        "average their raw predictions before rank-transforming "
                        "(default 8; reduces seed variance at low compute cost)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
