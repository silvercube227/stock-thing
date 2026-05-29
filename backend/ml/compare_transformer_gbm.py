"""Head-to-head: PatchTST vs the LightGBM baseline on one identical split.

The expanding walk-forward (gbm_baseline) refits monthly — too expensive for the
~1M-param transformer. So this does the fair, feasible comparison instead: a single
wide holdout, both models fit on the SAME data boundary and scored on the SAME
holdout cross-sections, by the SAME metric (mean per-date Spearman rank-IC of the
predicted relative return vs the realized universe-demeaned return).

Split (T = latest month-end):
    fit:     sample_end <  T - (holdout+gap) months   <- both models fit here
    gap/val: [T-(holdout+gap), T-holdout)              <- transformer selection only; GBDT ignores
    holdout: sample_end >= T - holdout months          <- scored here
The gap doubles as a clean embargo so a fit-period 6M/1Y label can't reach holdout.
This fixes the original sin of the first PatchTST holdout (6 months => d=2-7 dates):
a multi-year holdout gives tens of monthly cross-sections per horizon.

Run: python -m backend.ml.compare_transformer_gbm --holdout-months 36 --epochs 80
"""

from __future__ import annotations

import argparse
import asyncio

from backend.ingestion.db import pool_context
from backend.ml.dataset import (
    SplitConfig,
    assemble_calendar_aligned,
    build_calendar_grid,
    cross_sectional_medians,
    load_frames_cached,
    months_before,
    relabel_cross_sectional,
    split_samples,
)
from backend.ml.gbm_baseline import LGBMConfig, prepare_panel, single_split_ic
from backend.ml.model import HORIZONS, PatchTSTConfig
from backend.ml.train import TrainConfig, _evaluate_samples, pick_device, train


async def run(args) -> None:
    async with pool_context() as pool:
        frames = await load_frames_cached(pool, refresh=args.refresh_cache)
        if not frames:
            raise SystemExit("no active tickers / frames loaded")
        num_tickers = max(f.embedding_idx for f in frames) + 1
        grid = build_calendar_grid(frames)
        T = max(grid)
        holdout_start = months_before(T, args.holdout_months)
        fit_cutoff = months_before(T, args.holdout_months + args.gap_months)
        cfg = SplitConfig(holdout_months=args.holdout_months,
                          val_months=args.holdout_months + args.gap_months)
        print(f"frames={len(frames)} grid={len(grid)} months  T={T}")
        print(f"fit(<)={fit_cutoff}  val/gap=[{fit_cutoff},{holdout_start})  holdout(>=)={holdout_start}")

        # ---------------- transformer (single fit) ----------------
        medians = cross_sectional_medians(frames)
        samples = relabel_cross_sectional(assemble_calendar_aligned(frames, grid), medians)
        splits = split_samples(samples, cfg, T=T)
        print(f"transformer samples: train={len(splits['train'])} "
              f"val={len(splits['val'])} holdout={len(splits['holdout'])}")
        result = train(
            splits, num_tickers=num_tickers, model_cfg=PatchTSTConfig(),
            train_cfg=TrainConfig(epochs=args.epochs, cross_sectional=True, seed=args.seed),
        )
        _, tf = _evaluate_samples(
            result.model, splits["holdout"], pick_device(), 256,
            result.return_scale, 1.0, result.class_weights,
        )

        # ---------------- GBDT (same boundary, single fit) ----------------
        panel = prepare_panel(frames, grid, n_buckets=args.n_buckets)
        gbdt = {
            h: single_split_ic(panel, h, fit_cutoff, holdout_start,
                               LGBMConfig(), target_mode=args.target, seed=args.seed)["summary"]
            for h in HORIZONS
        }

        # ---------------- report ----------------
        print(f"\n=== PatchTST vs LightGBM — holdout cross-sectional rank-IC "
              f"(single split, gbdt target={args.target}) ===")
        print(f"{'H':<4}{'TF_ic':>9}{'TF_d':>6}   {'GBDT_ic':>9}{'GBDT_d':>7}{'GBDT_t':>8}{'GBDT_hit':>9}   winner")
        for h in HORIZONS:
            tf_ic, tf_d = tf[h]["ic"], tf[h]["ic_n"]
            g = gbdt[h]
            tf_s = f"{tf_ic:>+9.4f}" if tf_ic == tf_ic else f"{'nan':>9}"
            win = "—"
            if tf_ic == tf_ic and g["n_folds"] > 0:
                win = "TF" if tf_ic > g["mean_ic"] else "GBDT"
            print(f"{h:<4}{tf_s}{tf_d:>6}   {g['mean_ic']:>+9.4f}{g['n_folds']:>7}"
                  f"{g['t_stat']:>+8.2f}{g['hit_rate']:>9.3f}   {win}")


def main() -> None:
    p = argparse.ArgumentParser(description="PatchTST vs LightGBM head-to-head on one split")
    p.add_argument("--holdout-months", type=int, default=36, help="months scored as holdout")
    p.add_argument("--gap-months", type=int, default=12, help="embargo/selection band before holdout")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--target", default="return", choices=["return", "rank", "quantile"],
                   help="GBDT training target transform")
    p.add_argument("--n-buckets", type=int, default=5)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--refresh-cache", action="store_true",
                   help="re-pull frames from Supabase and overwrite the local frame cache")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
