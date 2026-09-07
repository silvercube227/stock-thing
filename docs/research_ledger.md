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

### 2026-09-06 — Phase 2: diagnostics and honest benchmarks (measurement only)

No model change. Four things the harness previously could not measure at all. All
runs: selection folds, single-seed, production per-horizon config, refreshed cache.

**2.1 The model is massively over-fit.** Train IC was never computed, so the
in-sample/out-of-sample gap was unobservable. Scored on the headline metric (SECB),
in-sample over the last 12 training cross-sections:

| Horizon | in-sample SECB | out-of-sample SECB | gap |
|---------|----------------|--------------------|-----|
| 3M | +0.2985 | +0.0037 | +0.295 |
| 6M | +0.2373 | −0.0086 | +0.246 |
| 1Y | +0.2422 | −0.0088 | +0.251 |

`LGBMConfig`'s docstring claims it is "deliberately shallow + regularized ... should
resist memorizing the train cross-sections". It does not. The IC-vs-trees curve (read
off the same fits via `num_iteration`, no extra training) peaks well below the
configured 300 at every horizon: **3M at 200, 6M at 50, 1Y at 100**. The differences
are sub-floor, so this is not yet a promotion case — it is a specification finding
that makes Phase 4.1 and 4.3 concrete.

**2.2 Ridge beats the GBDT at every horizon.** Same folds, same target, same features;
only the estimator differs.

| Horizon | GBDT | ridge | forecast combination |
|---------|------|-------|----------------------|
| 3M | +0.0037 | **+0.0139** | −0.0239 |
| 6M | −0.0034 | **+0.0074** | −0.0210 |
| 1Y | −0.0111 | **+0.0041** | −0.0729 |

Ridge is the only estimator positive at all three horizons, and at 3M it is the first
thing measured on this panel to land inside the literature's honest large-cap range
(0.01–0.02) — though still below the 0.0231 detection floor, so **not significant**.
Consistent with Han-He-Rapach-Zhou (RoF 2024) and with the over-fitting above.

**The pre-written criterion was about the COMBINATION, and the GBDT passes it 3/3.**
The ridge result is a post-hoc comparison and is explicitly *not* grounds for
promotion; it earns its own pre-registered test in Phase 4. The equal-weight
combination fails badly everywhere — one vote per feature over 20-23 columns that are
largely collinear or noise dilutes rather than diversifies.

**2.3 The top of the list — what the product actually ships — is weak or negative.**
SECB is a full-list average over eleven sectors; the dashboard serves
`order by direction_prob desc`. Per fold, on the demeaned return:

| Horizon (production config) | decile spread | t_block | top-decile return | hit | precision@50 |
|---|---|---|---|---|---|
| 3M (knife 0.20 + smooth 3) | +0.0051 | +0.43 | −0.0057 | 0.46 | 0.225 |
| 6M (lambdarank, rolling-60) | −0.0003 | −0.01 | −0.0111 | 0.46 | 0.268 |
| 1Y (lambdarank, smooth 4)   | −0.0061 | −0.15 | −0.0057 | 0.47 | 0.270 |

**The top decile has underperformed the cross-sectional median at every horizon**, and
no decile spread is distinguishable from zero. Precision@50 is mildly above the 0.20
chance rate everywhere, so the ranking finds slightly more winners than chance while
losing on magnitude — the falling-knife shape.

**The 3M overlays are vindicated on exactly the metric that motivated them, and it was
invisible to SECB.** Same fold set, overlays off vs on: decile spread −0.0014 → +0.0051,
top-decile return −0.0135 → −0.0057, hit 0.32 → 0.46. SECB *falls* slightly under the
overlays, which is why this could never have been seen before.

**2.3b Per-sector heterogeneity dwarfs the headline.** 3M SECB +0.0037 is the mean of
eleven sector ICs spanning **+0.045 (Consumer Staples) to −0.043 (Industrials)**, seven
positive and four negative. `Communication Services` (n=60) and `Real Estate` (n=81)
have fewer folds than the rest (n=114) — correct PIT behaviour, since those sectors did
not exist for the whole panel.

**Incidental, and uncomfortable:** the highest gain-importance feature at 3M is
`fund_available` (0.094), ahead of `vol_120d` and `log_market_cap`. A missingness
indicator is the single most-used input in the book — and it is the same feature the
2026-08 audit flagged as a survivorship-availability proxy. Worth its own ablation.

### 2026-09-06 — Phase 2.4: 3M rank target — REJECTED at its pre-written criterion

**Hypothesis.** Cakici & Zaremba (SSRN 6615698, 2025): target preprocessing dominates
feature preprocessing; rank targets roughly triple predictive accuracy versus raw
returns, **and the edge is specifically a large-cap phenomenon** that reverses in
micro caps. 3M is the one horizon still fitting L2 on a raw relative return, so it is
where the prior should bite. New `sector_rank` target = per-date percentile of
`sector_return` (the continuous sibling of `sector_grade`, keeps an L2 fit and avoids
the ranker's 14x compute).

**Criterion (pre-written).** Promote if SECB ICIR improves ≥10% AND mean SECB IC is
not lower by more than 0.003.

**Result.** Matched folds and seeds, production 3M pack, expanding window, no overlays:

| 3M target | folds | SECB IC | ICIR | t_block | verdict |
|-----------|-------|---------|------|---------|---------|
| `sector_return` / L2 (production) | 114 | +0.0037 | +0.038 | +0.23 | — |
| `sector_rank` / L2 | 114 | −0.0008 | −0.008 | −0.05 | **FAIL** (ΔIC −0.0045) |
| `sector_grade` / LambdaRank | 114 | +0.0008 | +0.009 | +0.06 | **FAIL** (ΔIC −0.0029) |

**Verdict: REJECTED.** Production 3M keeps `sector_return`/L2. Both alternatives are
worse on both halves of the criterion. All three sit far below the 0.0231 floor, so
the honest reading is "no difference is detectable here and neither alternative earns
the change" rather than "ranking is proven harmful".

Why the prior may not transfer: the Cakici-Zaremba result is a broad multi-market
panel against a RAW return target, whereas `sector_return` is already a
within-(date, sector) demean — a normalization of its own. Ranking on top of that
discards the magnitude information that survives it, and the 2026-07 E3 result on
this panel (winsorizing the 3M label degraded it monotonically) already said 3M's
fat-tailed labels carry signal rather than noise. This is the third transform of the
3M label to be rejected, which is itself evidence about the horizon.

### 2026-09-06 — Phase 4.1 + 4.3: capacity and overlap-aware regularization — BOTH PROMOTED

Run after Phase 2 measured an in-sample/out-of-sample SECB gap of +0.25 to +0.30 — the
first direct evidence that the model was memorizing its training cross-sections.

**4.1 Tree count.** The IC-vs-trees curve peaked at **200 / 50 / 100** by horizon
against a configured 300. Pooled by mean Δ versus 300 across horizons, **150** is the
peak — and the only candidate whose *worst* horizon is non-negative, which is exactly
the criterion. (A fixed learning rate and no early stopping mean the first N trees of a
300-tree fit ARE the N-tree fit, so the curve and a refit agree; the A/B confirms it.)

*Criterion (pre-written): mean SECB IC ≥ baseline at EVERY horizon.*

| Horizon | folds | SECB @300 | SECB @150 | Δ |
|---------|-------|-----------|-----------|-----|
| 3M | 114 | +0.0037 | +0.0037 | +0.0000 |
| 6M | 108 | −0.0034 | +0.0034 | +0.0068 |
| 1Y |  96 | −0.0111 | −0.0089 | +0.0022 |

**PASS.**

**4.3 Overlap-aware leaf minimum.** A 6M label spans six months, so consecutive
monthly rows share five-sixths of their outcome window: the effective sample is
≈ rows/H. `min_child_samples = 50` counted RAW rows, so the fit was regularized against
a sample it does not have — the overlap was corrected in the metric (block bootstrap)
but never in the fit. Scaled to 50 × horizon-months = **150 / 300 / 600**
(`overlap_aware_cfg`), on top of 150 trees.

| Horizon | folds | SECB @mcs 50 | SECB @mcs eff | Δ | ICIR |
|---------|-------|--------------|---------------|-----|------|
| 3M | 114 | +0.0037 | +0.0069 | +0.0032 | +0.037 → +0.069 |
| 6M | 108 | +0.0034 | +0.0037 | +0.0003 | +0.026 → +0.027 |
| 1Y |  96 | −0.0089 | −0.0047 | +0.0042 | −0.080 → −0.046 |

**PASS.**

**Mechanism confirmed, not just the outcome.** At 3M the in-sample/out-of-sample gap
fell from **+0.2948 to +0.2183** — in-sample IC dropped 0.299 → 0.225 while
out-of-sample rose 0.0037 → 0.0069. Less memorization, slightly better generalization,
which is the stated reason for the change.

**Combined, versus the Phase 1 reference:** 3M +0.0037 → **+0.0069**, 6M −0.0034 →
**+0.0037**, 1Y −0.0111 → **−0.0047**. Two of three horizons now positive. On the
production LambdaRank config 6M also improves (−0.0086 → −0.0075), so the change is
not an artifact of the L2 reference configuration. **Everything is still far below the
detection floors (0.023 / 0.046 / 0.059) — this bought a better-specified model, not
edge.**

**Parity bug found and fixed while promoting this.** The research CLI built `lgb_cfg`
from scratch instead of the horizon's production spec, so the promoted per-horizon
`min_child_samples` would have been invisible to every future sweep — the identical
failure to the 6M rolling-60 window in Phase 0.2. `run()` now starts from
`PRODUCTION_HORIZON_SPECS[h].lgb_cfg`, explicit flags override, and the resolved config
is printed. `--objective` / `--lambdarank-truncation` default to the spec; a
return-like `--target` with a spec-supplied ranking objective falls back to L2 with a
printed note rather than erroring.

### 2026-09-06 — POST-PHASE-4 REFERENCE (supersedes the Phase 1 row)

Each horizon on **its own production config** (3M `sector_return`/L2 + revision
momentum; 6M `sector_grade`/LambdaRank + rolling-60; 1Y `sector_grade`/LambdaRank +
smooth 4), selection folds, single seed, refreshed cache:

| Horizon | folds | SECB IC | t_block | min_detect | vs pre-Phase-4 |
|---------|-------|---------|---------|------------|----------------|
| 3M | 114 | **+0.0069** | +0.74 | 0.0231 | +0.0037 → +0.0069 |
| 6M | 108 | −0.0075 | −0.70 | 0.0461 | −0.0086 → −0.0075 |
| 1Y |  96 | **+0.0024** | +0.14 | 0.0585 | −0.0088 → +0.0024 |

Nothing is significant; everything sits below its own floor.

**Flagged for a future pre-registered test — the 6M LambdaRank promotion may not
survive the fixed panel.** At 6M on the same folds and capacity settings, plain
L2 on `sector_return` now scores **+0.0037** while the promoted LambdaRank on
`sector_grade` scores **−0.0075**. LambdaRank was promoted in 2026-07 on the
pre-2026-08-21 contaminated panel (CLAUDE.md already carries a provenance warning on
those statistics). This is a post-hoc observation from a run made for another purpose,
so it is **not** grounds to demote anything — it earns its own criterion and its own
run, alongside the Phase 2 ridge finding.

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
