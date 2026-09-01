## Stock Trend Predictor

Personal long-only stock and ETF trend prediction app. Not for active trading — for directional awareness over 3M, 6M, and 1Y horizons.

### Stack
- **Frontend:** Next.js 15 App Router + TypeScript + Tailwind (deploys to Vercel)
- **Backend:** FastAPI (asyncpg, runs locally on M4 Mac)
- **Database:** Postgres (Supabase)
- **ML:** LightGBM cross-sectional ranker (production); PatchTST transformer (built, shelved — underperformed on 1M horizon which was the only labeled horizon available at the time)
- **Sentiment:** FinBERT running locally on MPS, daily pipeline pushes 7/14-day rolling scores to Supabase

### ML model status

**Production: LightGBM GBDT cross-sectional ranker** (`backend/ml/gbm_inference.py`)
- One shallow model per horizon (3M, 6M, 1Y) — 1M has no detectable signal, skip it
- Per-horizon training target (`PRODUCTION_HORIZON_SPECS` in gbm_baseline.py): **3M trains on `sector_return`** (within-(date,sector)-relative return) with L2 regression; **6M/1Y train on `sector_grade`** (per-date qcut of sector_return into 5 ordinal grades) with a **LambdaRank objective** (`LGBMRanker`, promoted 2026-07-06 — see the LambdaRank bullet). Both target the within-sector selection bar directly. 1M stays `rank` (dead horizon, not scored). Scored by cross-sectional rank-IC; promotions are graded on the block-bootstrapped WITHIN-SECTOR IC (SECB).
- 19 base features: 4 momentum windows, log_market_cap (PIT: raw close x as-reported shares), 3 volatility windows, 52w high/low distances, 2 MA gaps, vol_trend, 5 EDGAR fundamentals, `fund_available` (binary: has SEC filing as-of date). **FinBERT sentiment (`sentiment_7d`/`sentiment_14d`) was DEMOTED to an opt-in `--with-sentiment` pack 2026-08-21**: yfinance serves only ~30 days of headlines and there is no backfill, so the columns were ~98% zero across the training panel yet non-zero at inference — a train/serve skew where the model never had a chance to learn them. Still computed on every row; turn back on when a headline archive exists. **Per-horizon promoted packs (`PRODUCTION_HORIZON_SPECS`): 3M adds `revision_momentum` (forward-EPS estimate revisions + analyst-coverage / PT-estimate counts); 6M + 1Y add LSEG `revenue_surprise` (now computed from QUARTERLY `earnings_surprises`, was annual).** Opt-in `--with-*` packs built but NOT promoted: `eps_surprise` (PEAD — ablated & rejected at every horizon, even on quarterly data: negative standalone), `linear_blend` (GBDT+ridge stack), analyst revisions, forward valuation, `microstructure` (price/volume higher-moment + liquidity: ret_skew_120d/downside_vol_ratio_120d/amihud_illiq_60d/turnover_60d/efficiency_ratio_120d — rejected at 6M: only amihud carries within-sector signal and it's ≈ the size factor already in `log_market_cap`), `eps_dispersion` (DMS analyst disagreement = eps_std_dev/|eps_mean| — awaiting diagnostics gate), `short_interest` (FINRA Reg SHO days-to-cover — awaiting backfill), `seasonality` (Heston-Sadka same-calendar-month persistence: seasonal_same/other/gap_5y — **rejected 2026-07-06 at diagnostics: zero within-sector signal at every horizon**, sec_p ≫ 0.10), `insider` (Cohen-Malloy-Pomorski Form 4: insider_net_buy_6m/insider_buyers_90d/insider_net_ratio_12m — infra built + backfilled 994k rows 2010-2026, migration 010; **rejected 2026-07-06 at ablation: only net_ratio_12m passed the standalone p-gate (6M 0.034/1Y 0.073) but it's regime-inconsistent + 0.55 corr with `fund_available` (a filer-availability proxy), and the 8-seed net-of-book ablation HURT mean IC −6–8% at both horizons** — the CMP opportunistic-vs-routine filter, deferred to v2, is likely needed to isolate the ~82bp signal from routine-trade noise). Opt-in experiment knobs (not features): `--winsorize-target` (per-date label clip — **rejected 2026-07-06: 3M degrades monotonically, 6M/1Y sub-threshold**), `--industry-relative` (**rejected 2026-07-06: hurts all horizons even under sector_return**). Candidate features are vetted before any fit by `feature_diagnostics()` / `--feature-diagnostics` (decorrelation vs the book + standalone within-sector block-IC + a Kaufman-efficiency-ratio tertile split that flags sideways/consolidation-only signal).
- `n_jobs=1` required (MPS + multiprocessing conflict on M4)
- Production inference: 8-seed ensemble per horizon (predictions averaged before rank-transform), reduces seed variance.
- ⚠️ **PROVENANCE WARNING — every promotion statistic in the four bullets below (smoothing, knife overlay, rolling-60, LambdaRank) was measured on the PRE-2026-08-21 panel, which carried a contaminated `log_market_cap` and a look-ahead index universe.** They are kept as an accurate record of what was decided and why, NOT as current evidence. The post-remediation walk-forward table further down is the live result; on the fixed panel none of these levers has a significant base signal left to improve. The overlays' turnover/risk effects still reproduce (they are mechanical re-rankings); their IC/ICIR deltas do not.
- **Cross-date rank smoothing** (`HorizonSpec.smooth_span`, promoted 2026-06-05): EWMA each name's percentile rank toward its last stored rank (`alpha=2/(span+1)`), then re-rank — averages out per-date estimation noise in a persistent signal. Promoted at the noisiest horizons: **3M span 3, 1Y span 4** (8-seed SECB walk-forward — 3M ICIR +0.42→+0.49, t +2.93→+3.41, turnover −50%; 1Y ICIR +0.45→+0.53, t +1.44→+1.70, turnover −54%). **6M left unsmoothed**: already the most stable horizon, smoothing was ~flat on IC/ICIR (only a turnover trade). Applied in `gbm_inference.apply_rank_smoothing` (prod, online one-step recursion off the stored rank) and `walk_forward_ic(smooth_span=)` (eval, post-hoc `ewma_rank_by_ticker` — sweep with `--smooth-span`). Stability levers chosen over raw-IC ceiling because our IC is already at SOTA (Qlib LightGBM/Alpha158 ≈ 0.045); see [[ml-experiment-discipline]].
- **Falling-knife output overlay** (`HorizonSpec.knife_lambda`, promoted 2026-06-06): re-rank each cross-section by `final = rank((1−λ)·model_rank − λ·knife_rank)` where `knife = vol_p × downtrend_p` (high realized vol AND below-200d-MA / near-52w-low, from rank-normalized `vol_120d`/`ma_gap_200`/`dist_low_252`). Demotes "falling knife" picks out of the top while sparing high-vol UPtrenders. Promoted **3M only, λ=0.20** (8-seed SECB walk-forward, composed with smooth_span=3): top-decile knife score −47%, realized downside −7%, top-decile mean return still positive, for SECB IC +0.0575→+0.0563 (t_block +3.41→+2.89, p 0.0005→0.0020 — still strongly significant). **6M/1Y NOT promoted**: 6M erodes SECB ~6% per 0.10 λ for negligible downside; 1Y is power-limited and drops below its detection floor. **NOT a churn lever** — turnover was ~flat under the overlay (smoothing already owns turnover; the original "stabilize churn" goal is served by `smooth_span`). Applied in `gbm_inference.score_current_cross_section` (prod, before smoothing) and `walk_forward_ic(knife_lambda=)` (eval, no-refit sweep `--knife-sweep` / single `--knife-lambda`); shared math in `gbm_baseline.knife_overlay_ranks`. Metric: `top_decile_risk` (ex-ante knife/vol/downtrend + ex-post downside semi-deviation/tail of the top decile). See [[knife-overlay-falling-knife]].
- **Rolling-60 training window at 6M** (`HorizonSpec.max_train_months=60`, promoted 2026-06-09): limits training to the most recent 60 monthly grid dates (~5 years), discarding structurally stale 2010–2016 data. 8-seed SECB walk-forward: SECB IC +0.0474→+0.0556, ICIR 0.443→0.493 (+11.3%), t_block +2.12→+2.36, p 0.0025→0.0015, turnover flat. **3M/1Y not promoted**: 3M +4.1% below bar; 1Y −14.7% (long-horizon signal needs full regime history). Applied in `gbm_inference.fit_horizon_models` (trims `train` to last N dates). `HorizonSpec` now carries `max_train_months`; serialized in `_serialize_spec` / `_specs_from_serialized`.
- **LambdaRank ranking objective at 6M/1Y** (`LGBMConfig.objective="lambdarank"` + `target_mode="sector_grade"`, promoted 2026-07-06): the fit was pointwise L2 regress-then-rank, but scoring is pure rank-IC — a known misalignment (Poh et al. 2020; LambdaRankIC 2026). Switch the two most stable horizons to an `LGBMRanker` trained on ordinal `sector_grade` (per-date `qcut(sector_return, 5)`) with a **per-date query group** and **linear `label_gain`** (Spearman rewards monotone ordering through the whole list, not the default 2^i−1 top-heavy gain). `lambdarank_truncation_level=100` (top-100 focus = what the product ships; full-list 500 was ~3 hr/horizon in the walk-forward for negligible extra signal — one ranker fit is 14× a regressor at trunc 100, 34× at 500). 8-seed SECB walk-forward vs the same-seed regression baseline: **6M** SECB IC +0.0560→+0.0563, ICIR 0.494→0.575 (+16.4%), t_block 2.36→2.75, p 0.0020→0.0005, turnover 0.116→0.105; **1Y** SECB IC +0.0579→+0.0690 (+19.2%), ICIR 0.747→1.109 (+48.5%), t_block 2.41→3.58, hit 0.82→0.89, turnover 0.042→0.038. **3M NOT promoted**: ICIR +8.9% (below the +10% bar) AND mean IC −9% (net loss). The overlays (6M rolling-60, 1Y smooth_span=4) compose unchanged — they act on the ranker's percentile ranks (monotone-invariant). Inference needs ZERO changes: `fit_lgbm_model` branches on `lgb_cfg.objective` to `LGBMRanker` (same `.predict()` signature); `sector_grade` flows via `apply_target_modes`; config round-trips through `asdict`. Sweep with `--objective lambdarank --target sector_grade [--lambdarank-truncation N] [--rank-grades K]`.
- Walk-forward stats (8-seed ensemble, de-survivorshipped universe, 716 tickers incl. removed-from-index), production per-horizon target (3M `sector_return`/L2; 6M+1Y `sector_grade`/LambdaRank), PRODUCTION per-horizon feature packs, graded on SECB:

  | Horizon | mean_IC | t_block | p_block | SEC_IC | SECB_t | SECB_p | Verdict |
  |---------|---------|---------|---------|--------|--------|--------|---------|
  | 3M (+revmom, smooth=3, knife=0.20) | +0.006 | +0.29 | 0.735 | +0.016 | +0.83 | 0.329 | **NOT significant** |
  | 6M (+rev, rolling-60, lambdarank) | −0.009 | −0.32 | 0.681 | +0.001 | +0.05 | 0.942 | **NOT significant** (flat) |
  | 1Y (+rev, smooth=4, lambdarank)   | −0.025 | −0.77 | 0.325 | −0.010 | −0.40 | 0.537 | **NOT significant** (negative) |

  t_block/SECB_t = moving-block bootstrap (block=horizon months, 2000 reps). SEC/SECB = mean within-GICS-sector IC (min 10 names/sector), also block-corrected. t_naive (not shown) is inflated by overlapping labels. Folds: 144/138/126. min_detect|IC| = 0.032/0.035/0.030 — every observed effect is BELOW its own detection floor.
- **THE 2026-08-21 AUDIT REMEDIATION KILLED THE MEASURED SIGNAL AT ALL THREE HORIZONS.** Before the fixes the table read 3M SEC +0.027 (p=0.011), 6M +0.056 (p=0.0005), 1Y +0.069 (p=0.0005). Attribution at 6M (single-seed L2, one fix removed at a time, same frames/code path):

  | Variant | SEC IC | SEC t |
  |---|---|---|
  | old panel (no fixes) | +0.0622 | +7.52 |
  | − implementation lag | +0.0603 | +7.13 |
  | − **PIT market cap** | +0.0310 | +3.07 |
  | − **membership filter** (= all fixes on) | **−0.0033** | −0.29 |

  Two roughly equal construction errors carried essentially the entire measured signal: (1) **`log_market_cap` was future information twice over** — back-adjusted close makes a high-dividend payer look smaller in hindsight, and today's share count applied to 2012 makes a heavy-buyback firm look smaller in hindsight; both "looks smaller" conditions correlate with subsequent returns, so the size feature was a disguised future-information signal (≈49% of the IC). (2) **look-ahead index inclusion** — names promoted into the index later sat in earlier cross-sections (the remainder; −64% on its own). The implementation lag was worth only −3%. The old-panel baseline reproduces the audit's independently measured universe IC (+0.0732 vs +0.0743), so the harness is faithful. **This refines the audit's §0.2**: the signal was not load-bearing on the size *premium*, it was load-bearing on the *contamination inside* size. Post-fix the size-neutral IC ≈ the raw within-sector IC at every horizon (3M +0.011 vs +0.016), i.e. what little remains is no longer a size tilt.
- **NOT a measurement artifact** (checked): PIT market caps verified realistic (AAPL $4.5T, MSFT $3.45T, JPM $943B, KO $377B); filtered cross-sections stay 440–500 names/date post-2017 with the same 11 sectors clearing min_group_size=10; fold counts barely move (138 vs 139); two independent code paths agree to 4 decimals. 8-seed LambdaRank gained nothing over single-seed L2, so the collapse is not an artifact of a weaker eval config.
- **The one reproducible non-null finding — volatility-regime dependence** (`--regime-report`). Mean IC by tertile of the cross-sectional median of raw realized vol:

  | Horizon | low_vol | mid_vol | high_vol |
  |---------|---------|---------|----------|
  | 3M | +0.058 | +0.034 | −0.045 |
  | 6M | +0.054 | +0.007 | −0.057 |
  | 1Y | +0.028 | +0.007 | −0.064 |

  (sector IC shown.) The pattern holds across two objectives (L2 and LambdaRank) and all three horizons: whatever the book picks up works in calm markets and **inverts under stress** (3M sector IC 2020 −0.12, 2022 −0.030, 2025 −0.067). This was invisible in the pooled mean and is the audit's open item #11. ⚠️ **These tertiles are IN-SAMPLE** (breakpoints cut over the whole fold set), so the table describes the panel; it is not evidence for a tradeable rule. `regime_report` now also returns `by_vol_regime_pit` — the same cut against an expanding window of strictly prior folds (24-fold burn-in) — and that is the one a regime claim has to survive.
- **Status: no horizon currently clears the promotion bar.** `PRODUCTION_HORIZON_SPECS` is UNCHANGED (the overlays still behave as designed — 3M turnover 0.154→0.077, 1Y 0.090→0.041), but the specs are no longer backed by a significant result. Do not describe this model as having demonstrated edge until something clears SECB again.
- Signal is cross-sectional (relative ranking), not absolute direction — absolute direction has no detectable edge

### Post-audit research program (2026-08-31)

**Stage 1 — panel breadth.** The pre-2016 panel was only ~184 names wide against ~500 post-2016: 320 tickers had price history floored at exactly `2016-01-04`, a fossil of an old `backfill_prices --years 5` run. Backfilling that cohort to 2010 **roughly doubled the 2010–2015 cross-section** (184 → 356 names at 2010-06-30, 380 by 2012, 404 by 2014). Separately, 15 names carried a membership stretch we could not see at all — our sources only know a name's CURRENT index entry date, so AMD (member until 2013-09-23, back 2017-03-20), DD, DOW, EQT, PCG, TMUS, KDP, FSLR, JBL, LDOS, CEG, DELL, SNDK, TER and Q sat out every cross-section in between despite full price history. Recovered via fja05680 (see `scripts/seed_index_membership.py` source 4 for why that file is used but not trusted).

  **The removal-side survivorship gap is NOT closable on free data — measured, not assumed.** 102 index members that exited before 2016 have no row in `tickers`; sampling 30 of them against yfinance, **25 return zero bars** (BNI, CEPH, MOLX, SLE, EK…). Only names that still trade come back. Pre-2016 folds remain 100% survivor-composed; Stage 1 raised N, it did not reduce that bias. Coverage by year (PIT members vs priceable): 2010 500/356, 2014 535/404, 2016 545/446, 2020 523/477, 2026 505/505.

- **Post-Stage-1 baseline** (single-seed L2, `sector_return`, production per-horizon packs, no overlays, SELECTION folds only — labels fully realized before the 2024-01 holdout):

  | Horizon | folds | SEC IC | t_block | p_block | min_detect | PIT low_vol | PIT high_vol |
  |---------|-------|--------|---------|---------|------------|-------------|--------------|
  | 3M (≤2023-09-30) | 114 | −0.0016 | −0.09 | 0.913 | 0.0279 | **+0.0245** | **−0.0296** |
  | 6M (≤2023-06-30) | 108 | −0.0033 | −0.11 | 0.898 | 0.0497 | **+0.0223** | **−0.0587** |
  | 1Y (≤2022-12-31) |  96 | −0.0069 | −0.17 | 0.834 | 0.0616 | **+0.0800** | **−0.0622** |

  Pooled IC is indistinguishable from zero at every horizon, as it was before. **The frozen holdout costs power**: truncating the folds pushed 6M/1Y min-detect from ~0.035/0.030 up to 0.0497/0.0616 (1Y has only 8 effective blocks). 3M improved (0.032 → 0.0279) because the wider panel outweighs the lost folds. Worth being explicit about — honest measurement is not free.

- **The vol-regime inversion REPLICATES under point-in-time tertiles** — the Stage 0.2 criterion (high-vol sector IC negative at all three horizons AND low-vol positive at all three) passes, so it is no longer resting on in-sample breakpoints. Caveat: PIT buckets are badly unbalanced (3M 11/27/52 low/mid/high) because realized vol drifted up after the ultra-calm 2014–2017 stretch that forms the early history, so "low_vol" is thin.

- **Stage 2 — vol-regime gate: REJECTED at its pre-written criterion.** Two findings, both about the rule as specified:

  1. **Shrinking ranks toward 0.5 cannot move rank-IC.** It is a monotone transform of the cross-section, so per-date Spearman is *exactly* unchanged; no shrink factor moves IC toward zero (full shrink makes it undefined, not 0). The gate bites only through `ewma_rank_by_ticker`, where halving today's spread halves its weight against the prior rank. **6M is therefore inert by construction** (`smooth_span=0`), which `test_vol_gate_cannot_change_ic_without_smoothing` pins.
  2. Measured at the two horizons where it *can* act (matched seeds, selection folds): 3M PIT high-vol sector IC −0.0380 → **−0.0388 (worse)**; 1Y −0.0603 → −0.0594 (better by 0.0009 — 1.5% of its own 0.0616 detection floor). The criterion required improvement at both. Pooled IC moved in the 4th decimal at both.

  Also: the gate **fired on 29–33% of folds, not the ≤20% the rule assumed "by construction"** — an expanding 80th percentile over an upward-drifting vol series fires far more often than 20%, the same drift that skews the PIT tertiles. So the display-side variant (2b) fails its own firing-rate criterion too, and per protocol the percentile is not tuned to fix it. The one real side effect is ~13% lower rank turnover (3M 0.0798→0.0696, 1Y 0.0491→0.0439), a mechanical consequence of leaning on the prior rank — but smoothing already owns turnover, and turnover is a stability proxy, not a portfolio quantity. **Mechanism kept and tested (`vol_gate_flags` / `vol_gate_ranks` / `--vol-gate`, `HorizonSpec.vol_gate`); `PRODUCTION_HORIZON_SPECS` unchanged.**

### Research protocol (locked 2026-08-31)

Adopted after the audit remediation showed that two construction errors had carried the entire measured signal. The point is to make it impossible to promote on a sub-floor result again.

- **Frozen holdout, purge-aware.** The holdout is calendar **2024-01 onward** and is touched ONCE per finalized config. Selection/research folds must have labels fully realized before it opens: `--max-test-date` = **3M 2023-09-30, 6M 2023-06-30, 1Y 2022-12-31**. The holdout run passes `--min-test-date 2024-01-01`. Both are new flags on `walk_forward_ic` (they filter the fold list; the per-fold training window is untouched) and are threaded through every call site in `run()` — main fit, target-blend fit, reg-sweep, shuffle-null — so A/Bs and nulls stay matched. There is no default: truncation is always explicit.
- **The holdout is a sign/sanity check, not a significance test.** With data through 2026-08 it yields roughly 28 / 25 / 19 folds at 3M / 6M / 1Y (a label must also be realized *inside* the holdout), i.e. 2–3 effective 1Y blocks. Always report it next to its own min-detect IC.
- **The honest benchmark is IC ≈ 0.01–0.02**, not the old "Qlib SOTA 0.045" comparison, which was cross-market (retail-dominated China) and cross-horizon. Our detection floors are 0.030–0.035, i.e. **the floor sits above the realistic effect**: the panel currently cannot distinguish "no signal" from "normal honest signal", and anything that DOES clear the bar is disproportionately likely to be contamination.
- **Sanity ceiling: any large-cap monthly IC > ~0.035 is presumptively a bug or a leak.** Audit before celebrating — the 2026-08 audit is the existence proof.
- **No sub-floor significance hunts.** An experiment must either target |ΔIC| ≥ ~0.02, raise N_eff (universe breadth qualifies; single features almost never do), or be judged against a criterion written down numerically BEFORE the run.
- **Compute discipline.** Iterate single-seed L2; 8-seed / LambdaRank only to confirm, with matched seeds for any A/B.
- **Production stays frozen** while research runs. Only a lever that passes its own pre-written criterion ships.
- **Honesty note on the vol-regime hypothesis:** it was motivated by an *in-sample* observation on these same folds. It rests on a literature prior (Barroso–Santa-Clara 2015 own-vol scaling; Daniel–Moskowitz 2016 momentum crashes), not on in-panel significance — with ~5 stress episodes in the sample it cannot be validated in-panel, and generic factor-vol-timing is refuted (Cederburg et al. 2020). Any rule built on it must be near-parameter-free and never swept.

**Shelved: PatchTST transformer** (`backend/ml/model.py`, `train.py`, `dataset.py`)
- 4-layer encoder, FeatureGate variable-selection, multi-horizon heads, ~1M params
- Failed holdout because it was evaluated on 1M (the dead horizon); re-evaluation on 3M/6M may show signal but not prioritized

### Architecture

```
yfinance / SEC EDGAR / LSEG Workspace
        |
backend/ingestion/          prices.py  fundamentals.py  headlines.py  estimates.py
        |                   (asyncpg upserts to Supabase)
        v
Supabase Postgres           price_history  fundamentals  headlines  sentiment_daily  analyst_estimates  earnings_surprises
        |
backend/ml/                 gbm_inference.py  (reads frames, trains, writes predictions)
        |
        v
Supabase Postgres           predictions  model_versions
        |
backend/api/                FastAPI  (JWT auth via Supabase JWKS, asyncpg pool)
        |
frontend/                   Next.js dashboard  (Supabase JS client + FastAPI)
```

Daily pipeline orchestrated by `backend/jobs/daily_pipeline.py`, scheduled via macOS launchd (`deploy/launchd/com.stockthing.daily-pipeline.plist`), fires Mon–Fri at 17:30 local time.

### Repo layout

```
backend/
  config.py               pydantic-settings; loads .env; Settings singleton
  ingestion/
    db.py                 asyncpg pool helpers (pool_context, asyncpg_dsn)
    calendar.py           NYSE calendar: is_trading_day, trading_days_between, HORIZON_TRADING_DAYS
    prices.py             yfinance incremental ingest + drift detection; entry: ingest_recent()
    fundamentals.py       SEC EDGAR companyfacts parser + upsert; entry: ingest_fundamentals()
    headlines.py          yfinance news fetch, FinBERT scoring, sentiment_daily recompute; entry: ingest_sentiment()
    estimates.py          LSEG (lseg.data) analyst estimates → analyst_estimates (monthly) + earnings_surprises (quarterly FQ0 grid); entry: ingest_estimates()
    short_interest.py     FINRA Reg SHO bimonthly → short_interest; entry: ingest_short_interest() (NOT in daily pipeline yet)
    insiders.py           SEC Form 3/4/5 → insider_transactions; DERA quarterly bulk (parse_insider_dataset) + per-CIK Form 4 XML (parse_form4_xml); entries: ingest_insider_backfill() / ingest_insider_transactions() (feature REJECTED — infra kept; NOT in daily pipeline)
  jobs/
    daily_pipeline.py     Orchestrator: prices → sentiment → fundamentals (Fri) → inference (Fri+month-start)
    promote_model.py      Promote a candidate model_version → production (retires the old one, atomic)
  ml/
    features.py           build_sample(): 12-feature point-in-time assembly, seq_len=252 (transformer path)
    dataset.py            TickerFrame, load_frames(+_cached experiment cache), train/val/holdout split
    factors/              Feature layer (extracted from gbm_baseline 2026-06-07; gbm_baseline re-exports for back-compat):
      constants.py          all *_FEATURES column catalogs + FEATURE_COLS / EXPERIMENTAL_FEATURES
      price.py              _price_features (momentum/vol/MA/52w + lottery + MICROSTRUCTURE pack) + _seasonality_asof (Heston-Sadka, REJECTED) + _short_interest_asof
      fundamentals.py       _ttm_net_income_asof, _fundamental_context_asof (EDGAR TTM/valuation/quality)
      estimates.py          _estimates_context_asof, _earnings_reaction_asof (LSEG + filing drift/surprise)
      insiders.py           _insider_context_asof (Form 4 net-buy / cluster-buy / net-ratio, PIT on filing_date — REJECTED)
      assembly.py           build_ticker_rows + universe/market-horizon return maps
      util.py               _log_ratio, _safe_ratio
    gbm_baseline.py       Walk-forward LightGBM, panel prep, rank-IC scoring, PRODUCTION_HORIZON_SPECS; opt-in packs via --with-*; experiment tooling: walk_forward_ic, knife/smooth/target_blend/regularization sweeps, feature_diagnostics
    gbm_inference.py      Production: fit per-horizon GBDTs, score cross-section, upsert predictions
    model.py              PatchTST transformer (shelved)
    train.py              Transformer training harness (shelved)
  api/
    main.py               FastAPI app, lifespan pool, CORS to localhost:3000
    auth.py               JWT verification via Supabase JWKS
    deps.py               get_pool dependency
    schemas.py            Pydantic request/response models
    quotes.py             yfinance intraday quotes router
    routers/
      portfolio.py        Portfolio holdings CRUD
      rankings.py         Prediction/ranking endpoints
      tickers.py          Ticker add/search
  db/
    schema.sql            13 tables: tickers, price_history, fundamentals, headlines,
                          sentiment_daily, analyst_estimates, earnings_surprises, short_interest, insider_transactions, portfolio_holdings, model_versions, predictions, ingestion_runs
    rls.sql               RLS on portfolio_holdings (user_id scoped)
    seed_tickers.sql      35-ticker starter seed (live universe is ~716: 507 active + 209 removed-from-index)
    migrations/
      001_headlines_score_date.sql
      002_analyst_estimates.sql
      003_ingestion_runs_skipped_status.sql
      004_analyst_estimates_eps.sql
      005_quarterly_surprises_and_counts.sql   earnings_surprises table + num_analysts/pt_num_estimates cols
      006_normalize_sectors.sql
      006_user_added_tickers.sql
      007_predictions_risk_flag.sql            predictions.risk_flag (falling-knife tag: none/elevated/high)
      008_eps_dispersion.sql                   analyst_estimates.eps_std_dev + eps_num_inc_estimates
      009_short_interest.sql                   short_interest table (FINRA Reg SHO bimonthly)
      010_insider_transactions.sql             insider_transactions table (SEC Form 3/4/5, filing_date PIT key)
      011_fundamentals_shares_outstanding.sql  fundamentals.shares_outstanding (PIT cover-page share count)
      012_index_membership.sql                 index_membership table (PIT S&P 500 intervals)
      013_normalize_sectors_again.sql          'Technology' -> 'Information Technology' (6 mega-caps had
                                               formed their own sub-min_group_size peer group)
  tests/                  ~134 tests (prices drift, features no-lookahead, dataset splits,
                          model arch, GBM baseline, API, sentiment, fundamentals parser, frame cache,
                          estimate ingestion + estimate-feature PIT)

frontend/
  src/app/
    page.tsx              Dashboard (portfolio table + rank gauges)
    login/page.tsx        Supabase email auth
    screener/page.tsx     (stub)
    ticker/[symbol]/      Ticker detail: price chart, fundamentals panel, rank gauges
  src/components/
    PortfolioTable.tsx    Holdings with intraday quotes + P&L
    RankGauge.tsx         Percentile rank arc gauge per horizon
    SentimentGauge.tsx    Rolling FinBERT score gauge
    PriceChart.tsx        60-day price sparkline
    FundamentalsPanel.tsx TTM revenue/margins/FCF from EDGAR
    AddTickerControl.tsx  Search + add to portfolio
    NetValueHeader.tsx    Total portfolio value header
    SharesEditor.tsx      Inline shares editing
    AppHeader.tsx         Nav + user menu
    AuthProvider.tsx      Supabase session context
  src/hooks/
    usePortfolio.ts       Portfolio holdings + live quotes
    useQuotes.ts          yfinance intraday batch quotes
    useTickerDetail.ts    Per-ticker predictions + fundamentals
  src/lib/
    api.ts                FastAPI client with JWT injection
    supabase.ts           Supabase JS client
    format.ts             Number/currency formatters

scripts/                  One-off backfill + seed utilities (run with python -m scripts.<name>)
  seed_sp500.py           Bootstrap tickers table
  seed_sp500_historical.py  Historical price seed
  backfill_prices.py      Price history backfill
  backfill_fundamentals.py  EDGAR fundamentals backfill
  backfill_sentiment.py   Historical sentiment backfill
  backfill_estimates.py   LSEG analyst-estimate backfill (--missing-only, --symbols); needs Workspace running
  backfill_insiders.py    SEC DERA insider-transaction backfill (--start/--end quarter, 2006+); needs CIKs populated
  backfill_ciks.py        Populate CIKs from ticker symbols (SEC map -> DERA removal-quarter fallback -> overrides CSV)
  backfill_sectors.py     GICS sector/industry for removed names via Wikipedia revision snapshots
  seed_index_membership.py  Build PIT index_membership intervals ("Date added" + tickers.removed_at)
  data/removed_ticker_overrides.csv  Hand-curated CIK/sector for names no automated source resolves
  _inspect_prices.py      Debug utility
  _inspect_fundamentals.py  Debug utility
  _probe_lseg.py          One-off LSEG field/PIT-history probe
  measure_egress.py       Supabase egress attribution (read-only)

deploy/
  launchd/
    com.stockthing.daily-pipeline.plist  macOS LaunchAgent — install to ~/Library/LaunchAgents/
```

### Daily pipeline stages

| Stage | Frequency | Module |
|---|---|---|
| prices_daily | every trading day | `ingestion/prices.py` — `ingest_recent()` |
| sentiment | every trading day | `ingestion/headlines.py` — `ingest_sentiment()` |
| fundamentals | Fridays | `ingestion/fundamentals.py` — `ingest_fundamentals()` |
| estimates | first trading day of month, only when an LSEG session is reachable | `ingestion/estimates.py` — `ingest_estimates()`; logs a `skipped` run when Workspace is down |
| gbm_inference | Fridays + first trading day of month | `ml/gbm_inference.py` — trains + writes `predictions` |

Each stage logs start/finish/status to `ingestion_runs`. The orchestrator adds a top-level `daily_pipeline` row. Non-trading days exit 0 without touching the DB.

### Key invariants

- **No lookahead**: feature joins are point-in-time as of the sample date — fundamentals on `filed_at` (SEC receipt), LSEG monthly estimates on `as_of_date` (observation date), quarterly surprises on `report_date` (announcement date, in `earnings_surprises`), each looked up independently per field; never `period_end`. The quarterly consensus stored in `earnings_surprises` is the pre-report `Period=FQ0` value (LSEG-probe-verified: matches the last pre-report monthly snapshot, not the post-announcement revision).
- **Implementation lag** (`compute_targets`, 2026-08-21): features come from the close of bar `pos`, so the position is entered at bar `pos+1` — the label is `log(adj_close[pos+1+H] / adj_close[pos+1])`. Previously the return started at the *same* close the signal was computed from, which assumes observing a close and trading at it simultaneously. `cross_sectional_medians` (shift −1 → −(H+1)) and `build_market_horizon_returns` (window starts at `idx(g)+2`, since `daily[i]` is the return INTO bar `i`) are aligned to the same convention.
- **Point-in-time market cap** (2026-08-21): `log_market_cap = log(raw close × as-reported shares on file)`. `adj_close` is back-adjusted, so its LEVEL embeds every split/dividend AFTER the bar — future information — and `tickers.shares_outstanding` is a single CURRENT scalar. Shares now come from the filing cover page (`dei:EntityCommonStockSharesOutstanding`, migration 011) joined on `filed_at`; multi-class issuers fall back to weighted-average basic shares. Ratios of `adj_close` (returns/momentum) are unaffected and still use it. Fallback chain in `factors/assembly.py::_market_cap_at`.
- **Point-in-time index membership** (`index_membership`, migration 012): training and evaluation cross-sections contain only names that were index members on that date. Without it a name promoted into the index in 2023 sits in the 2017 cross-section — look-ahead universe bias, the mirror image of survivorship. Applied in `prepare_panel(membership_filter=True)` BEFORE normalization so the per-date feature ranks span the real universe; user-added tickers are exempt. Research CLI raises on missing membership data; inference degrades with a warning. Measure the bias with `--no-membership-filter`.
- **Survivorship**: the universe includes removed-from-index names (`active=false`, `removed_at` set) with their price history, so the cross-sectional panel isn't survivor-only (de-survivorshipping deflated older ICIRs to honest levels). The removed cohort now has CIK (208/210), sector (209/209) and EDGAR fundamentals (205/209) — before the 2026-08-21 backfill it had none of these, which made `fund_available` a proxy for future index removal. Still missing: sentiment for removed names; delisting returns (a name that stops trading is simply masked, never marked −100%).
- **Cross-sectional scoring**: `direction_prob` in `predictions` stores the clipped predicted percentile rank (0–1), not a calibrated probability. Dashboard copy should say "relative rank."
- **Rank stability** (`predictions.confidence`): std of predicted rank across the last ≤3 scoring dates. Lower = steadier model view of that name. Null if fewer than 2 dates available.
- **Horizon trading days**: 1M=21, 3M=63, 6M=126, 1Y=252. Defined in `calendar.py:HORIZON_TRADING_DAYS`.
- **1M horizon**: no detectable cross-sectional signal (t=0.59 in walk-forward). Skip in inference.

### Data sources

- Price/volume: yfinance (incremental, drift-corrected for splits/dividends)
- Fundamentals: SEC EDGAR companyfacts XBRL API (10-K + 10-Q, TTM where applicable). Also yields point-in-time shares outstanding from the filing cover page (`dei:EntityCommonStockSharesOutstanding`, unit `shares`, fallbacks `us-gaap:CommonStockSharesOutstanding` then weighted-average basic/diluted for multi-class issuers whose cover-page fact is dimensioned per share class) -> `fundamentals.shares_outstanding`, 98.5% fill.
- Sentiment: yfinance news (~30 days lookback) scored by FinBERT (`ProsusAI/finbert`)
- Analyst estimates: LSEG Workspace via `lseg.data` desktop session. Two grains: (1) MONTHLY point-in-time snapshots → `analyst_estimates` (recommendation/price-target consensus, forward EPS consensus, forward P/E & EV/EBITDA, analyst-coverage + PT-estimate counts); (2) QUARTERLY fiscal-period grid (`Period=FQ0,Frq=FQ`) → `earnings_surprises` (pre-report EPS/revenue consensus + actual + report date), which drives the `revenue_surprise` (promoted 6M/1Y) and `eps_surprise` (rejected) features. Note: this license has no recommendation-bucket counts (strong-buy/buy) — only aggregate `RecMean`. Conditional month-start pipeline stage (runs only when the desktop session is reachable, else logs a `skipped` run); manual `backfill_estimates.py` still available.
- Insider transactions: SEC Form 3/4/5 via DERA "Insider Transactions Data Sets" (quarterly bulk TSVs, 2006+) for backfill + per-CIK submissions JSON / Form 4 XML for incremental. PIT key = filing_date (SEC receipt). Backfilled 2010-2026 (994k rows, 504 issuers). **Feature rejected** (see rejected-packs note) — data + ingestion + `_insider_context_asof` kept for a possible v2 with the CMP opportunistic-vs-routine filter.
- Macro: deliberately excluded (scope decision)

### Scope

- Options support deferred (requires Monte Carlo pricer)
- Personal tool, not a hosted service
- Screener page is a stub
- Candidate → production promotion is manual (`backend/jobs/promote_model.py`); auto-promotion not implemented
- De-survivorship: removed-from-index names now have CIK/sector/fundamentals (2026-08-21 backfill); sentiment for them is still a gap, and delisting returns are absent (needs CRSP).
- Sector labels are TODAY's GICS applied retroactively (no PIT sector history) — they define both the training target and the headline SECB metric. Known limitation; a PIT sector history needs a paid source.
- No transaction-cost model, portfolio construction, or backtest: the project produces rankings, not positions. `rank_turnover` is a stability proxy, NOT portfolio turnover.
- No untouched final holdout: promoted configs were selected on the same walk-forward folds they are reported on, with no multiple-testing correction. `single_split_ic()` exists but is not wired into the sweep CLI.
- `scripts/seed_sp500_historical.py` removal detection is BROKEN upstream: Wikipedia deleted the "Selected changes" table from the S&P 500 page (2026). `scripts/seed_index_membership.py` works around it using the constituents table's "Date added" column plus `tickers.removed_at`.

1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

State your assumptions explicitly. If uncertain, ask.
If multiple interpretations exist, present them - don't pick silently.
If a simpler approach exists, say so. Push back when warranted.
If something is unclear, stop. Name what's confusing. Ask.
2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

No features beyond what was asked.
No abstractions for single-use code.
No "flexibility" or "configurability" that wasn't requested.
No error handling for impossible scenarios.
If you write 200 lines and it could be 50, rewrite it.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

3. Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:

Don't "improve" adjacent code, comments, or formatting.
Don't refactor things that aren't broken.
Match existing style, even if you'd do it differently.
If you notice unrelated dead code, mention it - don't delete it.
When your changes create orphans:

Remove imports/variables/functions that YOUR changes made unused.
Don't remove pre-existing dead code unless asked.
The test: Every changed line should trace directly to the user's request.

4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

"Add validation" → "Write tests for invalid inputs, then make them pass"
"Fix the bug" → "Write a test that reproduces it, then make it pass"
"Refactor X" → "Ensure tests pass before and after"
For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.