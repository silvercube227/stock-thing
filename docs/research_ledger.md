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

### 2026-09-05 — PHASE 0 REFERENCE BASELINE (grade every later A/B against this)

Refreshed frame cache (symbol-reuse guard active, prices current to 2026-08-31),
selection folds only, single-seed L2 on `sector_return`, production per-horizon packs,
expanding window, no overlays. This is the reference row for Phases 1-5.

| Horizon | folds | SECB IC | t_block | p_block | min_detect\|IC\| | size-neutral SECB | raw turnover |
|---------|-------|---------|---------|---------|----------------|-------------------|--------------|
| 3M | 114 | +0.0067 | +0.40 | 0.61 | 0.0259 | +0.0031 | 0.160 |
| 6M | 108 | +0.0018 | +0.06 | 0.94 | 0.0491 | −0.0019 | 0.133 |
| 1Y |  96 | −0.0081 | −0.20 | 0.80 | 0.0596 | −0.0048 | 0.113 |

Pre-Phase-0 the same command gave −0.0016 / −0.0001 / −0.0069. **Nothing here is
significant and every value sits below its own detection floor**; the panel still
cannot distinguish "no signal" from "normal honest signal". Phase 0 bought correctness,
not edge.

The symbol-reuse guard contributed the difference between this table and the
same-config run on the stale cache (3M +0.0025, 6M +0.0028, 1Y −0.0073): it does not
change the filtered panel (those rows were already outside every cross-section) but it
does change `build_universe_return_map` and `cross_sectional_medians`, which set the
demean baseline for every label. Effect is mixed and sub-floor, as expected for ~10.7k
of 1.9M price rows.

### 2026-09-06 — Phase 1: PIT sector target, terminal labels, cohort consistency

**Change.** Three label/metric corrections landed together (see the Phase 1 commit):
each panel row carries the GICS sector as of ITS OWN date from `sector_history`;
a series that ENDED gets a hold-to-last-trade exit instead of being masked; a row whose
sector group is missing or thinner than `sector_min_group_size` gets a NaN
sector-relative target and is dropped at fit time rather than falling through to the
universe-demeaned return.

**Criterion (pre-written).** None — these are corrections, accepted on correctness.
Recorded for direction only.

**Result.** Matched A/B on the SAME refreshed frame cache (identical folds, identical
panel rows, only the code differs; Phase-0 side run from a git worktree at `aafc2cf`):

| Horizon | folds | SECB Phase 0 | SECB Phase 1 | Δ | min_detect | size-neutral P0 → P1 |
|---------|-------|--------------|--------------|-----|------------|----------------------|
| 3M | 114 | +0.0010 | +0.0037 | +0.0027 | 0.0231 | −0.0021 → −0.0000 |
| 6M | 108 | +0.0030 | −0.0034 | −0.0064 | 0.0461 | −0.0012 → −0.0100 |
| 1Y |  96 | −0.0107 | −0.0111 | −0.0004 | 0.0585 | −0.0082 → −0.0133 |

Every delta is a fraction of its own detection floor — noise at this power. The 6M
decline is **not** evidence of harm: the prior number was computed by demeaning
against, and scoring inside, peer groups that partly did not exist at the time
(Communication Services post-dates Sept 2018, Real Estate Sept 2016), so removing that
artifact can move IC in either direction.

Scale of what was corrected: 6.1% of member-months carried a different sector than
today's label; 0.29% of member-months are newly dropped for a thin sector group;
~15 price series terminate early and now receive a label instead of being masked.

### 2026-09-06 — PHASE 1 REFERENCE BASELINE (supersedes the Phase 0 row)

Refreshed cache, prices to 2026-09-04, LSEG caught up (278k estimate + 34k surprise
rows), selection folds, single-seed L2 `sector_return`, production packs, expanding
window, no overlays.

| Horizon | folds | SECB IC | t_block | min_detect\|IC\| | size-neutral SECB |
|---------|-------|---------|---------|----------------|-------------------|
| 3M | 114 | +0.0037 | +0.23 | 0.0231 | −0.0000 |
| 6M | 108 | −0.0034 | −0.11 | 0.0461 | −0.0100 |
| 1Y |  96 | −0.0111 | −0.27 | 0.0585 | −0.0133 |

Still nothing significant, still below every floor.

## Outstanding

- 8-seed confirmation of the rank-averaged ensemble at 6M/1Y (no criterion attached; it
  is the correct estimator either way, and inert at the `n_seeds=1` used for every A/B
  above).
- **Deferred, needs its own measured run:** recomputing the universe demean over
  membership-filtered frames only. It changes the label baseline at every horizon and
  would have been unattributable inside the Phase 1 batch.
- The top of the ranked list is still unmeasured (plan item 2.3). The 2026-09-04
  production cross-section puts 8 Information Technology names in the 3M top 8 —
  `direction_prob` is a UNIVERSE percentile while the model is trained on a
  within-sector target, so sector concentration at the top is expected and invisible
  to SECB.
- 5 tickers return empty yfinance responses (AVB, SATS, EA, EQR, BK) and are stale;
  all are from the known "active but absent from the live index list" cohort.
- LSEG has no RIC for BF-B / BRK-A / BRK-B (pre-existing share-class gap).
