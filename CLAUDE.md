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
- Per-horizon training target (`PRODUCTION_HORIZON_SPECS` in gbm_baseline.py): 3M/6M/1Y all train on `sector_return` (within-(date,sector)-relative return), which directly optimizes within-sector stock selection — the success bar. 1M stays `rank` (dead horizon, not scored). Scored by cross-sectional rank-IC; promotions are graded on the block-bootstrapped WITHIN-SECTOR IC (SECB).
- 21 base features: 4 momentum windows, log_market_cap, 3 volatility windows, 52w high/low distances, 2 MA gaps, vol_trend, 5 EDGAR fundamentals, `fund_available` (binary: has SEC filing as-of date), 2 FinBERT sentiment rolling averages. **Per-horizon promoted packs (`PRODUCTION_HORIZON_SPECS`): 3M adds `revision_momentum` (forward-EPS estimate revisions + analyst-coverage / PT-estimate counts); 6M + 1Y add LSEG `revenue_surprise` (now computed from QUARTERLY `earnings_surprises`, was annual).** Opt-in `--with-*` packs built but NOT promoted: `eps_surprise` (PEAD — ablated & rejected at every horizon, even on quarterly data: negative standalone), `linear_blend` (GBDT+ridge stack), analyst revisions, forward valuation, `microstructure` (price/volume higher-moment + liquidity: ret_skew_120d/downside_vol_ratio_120d/amihud_illiq_60d/turnover_60d/efficiency_ratio_120d — rejected at 6M: only amihud carries within-sector signal and it's ≈ the size factor already in `log_market_cap`), `eps_dispersion` (DMS analyst disagreement = eps_std_dev/|eps_mean| — awaiting diagnostics gate), `short_interest` (FINRA Reg SHO days-to-cover — awaiting backfill). Candidate features are vetted before any fit by `feature_diagnostics()` / `--feature-diagnostics` (decorrelation vs the book + standalone within-sector block-IC + a Kaufman-efficiency-ratio tertile split that flags sideways/consolidation-only signal).
- `n_jobs=1` required (MPS + multiprocessing conflict on M4)
- Production inference: 8-seed ensemble per horizon (predictions averaged before rank-transform), reduces seed variance.
- **Cross-date rank smoothing** (`HorizonSpec.smooth_span`, promoted 2026-06-05): EWMA each name's percentile rank toward its last stored rank (`alpha=2/(span+1)`), then re-rank — averages out per-date estimation noise in a persistent signal. Promoted at the noisiest horizons: **3M span 3, 1Y span 4** (8-seed SECB walk-forward — 3M ICIR +0.42→+0.49, t +2.93→+3.41, turnover −50%; 1Y ICIR +0.45→+0.53, t +1.44→+1.70, turnover −54%). **6M left unsmoothed**: already the most stable horizon, smoothing was ~flat on IC/ICIR (only a turnover trade). Applied in `gbm_inference.apply_rank_smoothing` (prod, online one-step recursion off the stored rank) and `walk_forward_ic(smooth_span=)` (eval, post-hoc `ewma_rank_by_ticker` — sweep with `--smooth-span`). Stability levers chosen over raw-IC ceiling because our IC is already at SOTA (Qlib LightGBM/Alpha158 ≈ 0.045); see [[ml-experiment-discipline]].
- **Falling-knife output overlay** (`HorizonSpec.knife_lambda`, promoted 2026-06-06): re-rank each cross-section by `final = rank((1−λ)·model_rank − λ·knife_rank)` where `knife = vol_p × downtrend_p` (high realized vol AND below-200d-MA / near-52w-low, from rank-normalized `vol_120d`/`ma_gap_200`/`dist_low_252`). Demotes "falling knife" picks out of the top while sparing high-vol UPtrenders. Promoted **3M only, λ=0.20** (8-seed SECB walk-forward, composed with smooth_span=3): top-decile knife score −47%, realized downside −7%, top-decile mean return still positive, for SECB IC +0.0575→+0.0563 (t_block +3.41→+2.89, p 0.0005→0.0020 — still strongly significant). **6M/1Y NOT promoted**: 6M erodes SECB ~6% per 0.10 λ for negligible downside; 1Y is power-limited and drops below its detection floor. **NOT a churn lever** — turnover was ~flat under the overlay (smoothing already owns turnover; the original "stabilize churn" goal is served by `smooth_span`). Applied in `gbm_inference.score_current_cross_section` (prod, before smoothing) and `walk_forward_ic(knife_lambda=)` (eval, no-refit sweep `--knife-sweep` / single `--knife-lambda`); shared math in `gbm_baseline.knife_overlay_ranks`. Metric: `top_decile_risk` (ex-ante knife/vol/downtrend + ex-post downside semi-deviation/tail of the top decile). See [[knife-overlay-falling-knife]].
- **Rolling-60 training window at 6M** (`HorizonSpec.max_train_months=60`, promoted 2026-06-09): limits training to the most recent 60 monthly grid dates (~5 years), discarding structurally stale 2010–2016 data. 8-seed SECB walk-forward: SECB IC +0.0474→+0.0556, ICIR 0.443→0.493 (+11.3%), t_block +2.12→+2.36, p 0.0025→0.0015, turnover flat. **3M/1Y not promoted**: 3M +4.1% below bar; 1Y −14.7% (long-horizon signal needs full regime history). Applied in `gbm_inference.fit_horizon_models` (trims `train` to last N dates). `HorizonSpec` now carries `max_train_months`; serialized in `_serialize_spec` / `_specs_from_serialized`.
- Walk-forward stats (8-seed ensemble, de-survivorshipped universe, 716 tickers incl. removed-from-index), `sector_return` target, PRODUCTION per-horizon feature packs, graded on SECB:

  | Horizon | mean_IC | t_block | p_block | SEC_IC | SECB_t | SECB_p | Verdict |
  |---------|---------|---------|---------|--------|--------|--------|---------|
  | 3M (+revmom, smooth=3, knife=0.20) | +0.047 | 2.89 | 0.002 | +0.027 | 1.96 | 0.011 | Block-significant |
  | 6M (+rev, rolling-60) | +0.056 | 2.36 | 0.002 | +0.042 | — | — | Block-significant; rolling-60 promoted 2026-06-09 |
  | 1Y (+rev, smooth=4)   | +0.061 | 1.51 | 0.040 | +0.050 | 1.96 | 0.003 | **Block-significant within-sector selection** (hit 0.80) |

  t_block/SECB_t = moving-block bootstrap (block=horizon months, 2000 reps). SEC/SECB = mean within-GICS-sector IC (min 10 names/sector), also block-corrected. The `sector_return` target traded a little universe IC for genuine within-sector selection at every horizon; 1Y went from failing under `beta_resid` (SECB p=0.266) to clearly passing under `sector_return`. t_naive (not shown) is inflated by overlapping labels.
- **All three scored horizons now show block-significant within-sector selection** (SECB p ≤ 0.011). The quarterly-surprise rebuild (2026-05-31) strengthened `revenue_surprise` at 6M and made it a wash at 1Y (base ≈ +rev — left on as a no-op, candidate parsimony cleanup); the 3M `revision_momentum` pack lifted 3M off the borderline. Power note: 1Y is block-limited (~8 effective blocks) so its `min_detect|IC|`≈0.032 — the sweep CLI prints this.
- Signal is cross-sectional (relative ranking), not absolute direction — absolute direction has no detectable edge

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
  jobs/
    daily_pipeline.py     Orchestrator: prices → sentiment → fundamentals (Fri) → inference (Fri+month-start)
    promote_model.py      Promote a candidate model_version → production (retires the old one, atomic)
  ml/
    features.py           build_sample(): 12-feature point-in-time assembly, seq_len=252 (transformer path)
    dataset.py            TickerFrame, load_frames(+_cached experiment cache), train/val/holdout split
    factors/              Feature layer (extracted from gbm_baseline 2026-06-07; gbm_baseline re-exports for back-compat):
      constants.py          all *_FEATURES column catalogs + FEATURE_COLS / EXPERIMENTAL_FEATURES
      price.py              _price_features (momentum/vol/MA/52w + lottery + MICROSTRUCTURE pack)
      fundamentals.py       _ttm_net_income_asof, _fundamental_context_asof (EDGAR TTM/valuation/quality)
      estimates.py          _estimates_context_asof, _earnings_reaction_asof (LSEG + filing drift/surprise)
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
    schema.sql            12 tables: tickers, price_history, fundamentals, headlines,
                          sentiment_daily, analyst_estimates, earnings_surprises, short_interest, portfolio_holdings, model_versions, predictions, ingestion_runs
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
  backfill_ciks.py        Populate CIKs from ticker symbols
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
- **Survivorship**: the universe includes removed-from-index names (`active=false`, `removed_at` set) with their price history, so the cross-sectional panel isn't survivor-only (de-survivorshipping deflated older ICIRs to honest levels). Estimates now cover removed names; fundamentals/sentiment for them are still a gap.
- **Cross-sectional scoring**: `direction_prob` in `predictions` stores the clipped predicted percentile rank (0–1), not a calibrated probability. Dashboard copy should say "relative rank."
- **Rank stability** (`predictions.confidence`): std of predicted rank across the last ≤3 scoring dates. Lower = steadier model view of that name. Null if fewer than 2 dates available.
- **Horizon trading days**: 1M=21, 3M=63, 6M=126, 1Y=252. Defined in `calendar.py:HORIZON_TRADING_DAYS`.
- **1M horizon**: no detectable cross-sectional signal (t=0.59 in walk-forward). Skip in inference.

### Data sources

- Price/volume: yfinance (incremental, drift-corrected for splits/dividends)
- Fundamentals: SEC EDGAR companyfacts XBRL API (10-K + 10-Q, TTM where applicable)
- Sentiment: yfinance news (~30 days lookback) scored by FinBERT (`ProsusAI/finbert`)
- Analyst estimates: LSEG Workspace via `lseg.data` desktop session. Two grains: (1) MONTHLY point-in-time snapshots → `analyst_estimates` (recommendation/price-target consensus, forward EPS consensus, forward P/E & EV/EBITDA, analyst-coverage + PT-estimate counts); (2) QUARTERLY fiscal-period grid (`Period=FQ0,Frq=FQ`) → `earnings_surprises` (pre-report EPS/revenue consensus + actual + report date), which drives the `revenue_surprise` (promoted 6M/1Y) and `eps_surprise` (rejected) features. Note: this license has no recommendation-bucket counts (strong-buy/buy) — only aggregate `RecMean`. Conditional month-start pipeline stage (runs only when the desktop session is reachable, else logs a `skipped` run); manual `backfill_estimates.py` still available.
- Macro: deliberately excluded (scope decision)

### Scope

- Options support deferred (requires Monte Carlo pricer)
- Personal tool, not a hosted service
- Screener page is a stub
- Candidate → production promotion is manual (`backend/jobs/promote_model.py`); auto-promotion not implemented
- Whole-pipeline de-survivorship incomplete: removed-from-index names lack fundamentals/sentiment

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