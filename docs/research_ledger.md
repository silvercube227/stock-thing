# Research ledger

Every experiment run against the panel, with its acceptance criterion written down
**before** the run. This is the trial count any deflated statistic needs (Gençay 2026,
arXiv:2608.27734, certified 0 of ~100 mined strategies on 453 US large caps once trials
were counted). It is deliberately a ledger and not an FDR hurdle: Chen 2025
(arXiv:2311.10685) shows Harvey–Liu–Zhu-style hurdles reject nearly every genuine
out-of-sample performer, and that empirical-Bayes shrinkage of the estimate beats
raising the bar.

## How to read this

- **SECB** = mean within-GICS-sector rank-IC, moving-block bootstrapped. The headline.
- Selection folds only (`--max-test-date` = 3M 2023-09-30, 6M 2023-06-30, 1Y 2022-12-31).
  The 2024-01+ holdout is touched **once** per finalized config.
- Detection floors on this panel are ~0.026 (3M) / ~0.050 (6M) / ~0.062 (1Y) while the
  honest large-cap effect is ~0.01–0.02. **Any delta below the floor is not evidence.**
  Sub-floor moves are recorded here for direction and consistency only.
- Iterate single-seed L2; 8-seed / LambdaRank only to confirm, matched seeds for A/Bs.

## Entries

### 2026-09-05 — Phase 0.5: NaN-native missing data (`--` no flag; changes the panel)

**Change.** Features whose upstream source has no observation as of the row date now
emit NaN instead of the sentinel `0.0`: fundamentals/valuation/quality/earnings-reaction
gated on `fund_available`, the analyst packs on a new `est_available`, `log_market_cap`
when no share count exists, `short_ratio` with no FINRA row, and the quarterly surprises
when no quarter has been reported. `rank_normalize_features` keeps NaN out of the
ranking; LightGBM routes it natively. Motivation: Bryzgalova–Lerner–Lettau–Pelger
(RFS 2025) and Freyberger et al. (RFS 2025) — characteristic missingness is systematic,
and zero/mean imputation biases cross-sectional estimates.

**Missingness actually present** (production book, membership-filtered panel, 84,041 rows):
estimate pack 11.6% overall but **100% in 2011 and 97.6% in 2012**, ~2% from 2013;
`revenue_surprise` 4.4%; fundamentals 1.9%; `log_market_cap` 0.52% (all pre-2017).
So the pre-2013 rows were previously ranked on sentinel zeros across the whole
estimate pack — a coverage/size proxy, exactly as suspected.

**Criterion (pre-written).** Correctness change; accept unless SECB falls by more than
0.003 at any horizon.

**Result.** Matched folds, matched seeds, `sector_return`/L2, production packs,
expanding window, no overlays. Measured against the pre-change commit (`8aa08d1`) in a
stashed working tree, same command both sides.

| Horizon | folds | SECB before | SECB after | Δ | size-neutral before → after |
|---------|-------|-------------|------------|-----|------------------------------|
| 3M | 114 | −0.0016 | +0.0025 | +0.0041 | −0.0043 → −0.0010 |
| 6M | 108 | −0.0001 | +0.0028 | +0.0029 | −0.0018 → −0.0012 |
| 1Y |  96 | −0.0069 | −0.0073 | −0.0004 | −0.0037 → −0.0042 |

**Verdict: PASS.** No horizon fell. **Not a signal result** — every delta is far below
its detection floor (0.026 / 0.050 / 0.062). Recorded as a correctness fix whose sign
happened to be favourable at 3M/6M.

### 2026-09-05 — Phase 0.2/0.3/0.4/0.6: rig corrections (no criterion; bugs)

Run in the same batch as 0.5, so the A/B above is their joint effect at `n_seeds=1`
(where the ensemble change is inert by construction).

- **0.4 seed ensemble now rank-averages** (`_fit_predict`, `score_current_cross_section`).
  Raw-score averaging is meaningless across independently-seeded LambdaRank models.
  Inert at `n_seeds=1`; an 8-seed A/B at 6M/1Y is still outstanding.
- **0.2 `walk_forward_ic` honors `HorizonSpec.max_train_months`** and counts *labeled*
  training dates, matching `fit_horizon_models`. Previously a 6M re-validation that
  forgot `--max-train-months 60` silently measured a different model than production.
- **0.2 shuffle null is seed-matched** (was always `n_seeds=1` against an 8-seed real).
- **0.2 off-index names no longer define the cross-section.** 14 user-added tickers —
  including SQQQ (−3x), GBTC, SMH, FDT — sat inside production's per-date rank
  normalization but never inside the walk-forward's. They are still scored, now placed
  against the member distribution.
- **0.3 production smoothing made byte-identical to `ewma_rank_by_ticker`** (migration
  014 stores the raw EWMA state) and anchored to the monthly grid. Production blended
  against the re-ranked percentile and advanced on every Friday run, so it smoothed
  harder and ~4-5x more often than the folds that promoted spans 3 (3M) and 4 (1Y).
- **0.6 symbol-reuse guard.** 16 delisted tickers had another company's bars under the
  old symbol (SE→Sea Limited, EMC, CA, APC, INFO, STI, SPLS, PCL, CAM, POM, TE, ADT,
  SBNY, CCE, CSRA, NFX). Excluded from cross-sections by the membership filter, but they
  fed `build_universe_return_map` (the market series behind beta / residual momentum /
  the `beta_resid` target) and `cross_sectional_medians` (the demean baseline for every
  label). Guard keys on the *gap*, so continuously-trading ex-members (FOSL, GME, AA,
  RIG) — the de-survivorship data the panel needs — are untouched.
  **Not yet reflected in any measured number**: it needs a frame-cache refresh.

## Outstanding before Phase 1

- Re-baseline with `--refresh-cache` so the reuse guard and `removed_at` are in the panel.
- 8-seed confirmation of the rank-averaged ensemble at 6M/1Y.
- Production inference has not been re-run since 2026-07-31; the deployed model
  (`48d1749f`, promoted 2026-08-01) still predates the 2026-08-21 audit fixes.
