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
- Per-horizon training target (`PRODUCTION_HORIZON_SPECS` in gbm_baseline.py): 3M/6M = rank-normalized cross-sectional return; 1Y = beta-residualized return (strips market beta toward idiosyncratic alpha). Scored by cross-sectional rank-IC.
- 20 base features: 4 momentum windows, log_market_cap, 3 volatility windows, 52w high/low distances, 2 MA gaps, vol_trend, 5 EDGAR fundamentals, 2 FinBERT sentiment rolling averages. 6M + 1Y also include LSEG `revenue_surprise` (promoted via walk-forward ablation). Other LSEG packs (analyst revisions, forward valuation) exist as opt-in `--with-*` flags but didn't beat the null.
- `n_jobs=1` required (MPS + multiprocessing conflict on M4)
- Walk-forward validated ICIR (de-survivorshipped universe, incl. removed-from-index names): 3M=0.41/t=5.5, 6M=0.44/t=5.9 (+revenue_surprise), 1Y=0.45/t=6.0 (+revenue_surprise). NOTE: earlier survivor-only ICIRs were higher (e.g. 6M 0.64) but inflated by survivorship bias.
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
Supabase Postgres           price_history  fundamentals  headlines  sentiment_daily  analyst_estimates
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
    estimates.py          LSEG (lseg.data) analyst estimates → analyst_estimates; entry: ingest_estimates()
  jobs/
    daily_pipeline.py     Orchestrator: prices → sentiment → fundamentals (Fri) → inference (Fri+month-start)
    promote_model.py      Promote a candidate model_version → production (retires the old one, atomic)
  ml/
    features.py           build_sample(): 12-feature point-in-time assembly, seq_len=252 (transformer path)
    dataset.py            TickerFrame, load_frames(+_cached experiment cache), train/val/holdout split
    gbm_baseline.py       Walk-forward LightGBM, FEATURE_COLS (20) + opt-in packs (valuation/quality/LSEG estimate), rank-IC scoring, PRODUCTION_HORIZON_SPECS
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
    schema.sql            10 tables: tickers, price_history, fundamentals, headlines,
                          sentiment_daily, analyst_estimates, portfolio_holdings, model_versions, predictions, ingestion_runs
    rls.sql               RLS on portfolio_holdings (user_id scoped)
    seed_tickers.sql      35-ticker starter seed (live universe is ~716: 507 active + 209 removed-from-index)
    migrations/
      001_headlines_score_date.sql
      002_analyst_estimates.sql
  tests/                  12 modules, ~126 tests (prices drift, features no-lookahead, dataset splits,
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
| gbm_inference | Fridays + first trading day of month | `ml/gbm_inference.py` — trains + writes `predictions` |

Each stage logs start/finish/status to `ingestion_runs`. The orchestrator adds a top-level `daily_pipeline` row. Non-trading days exit 0 without touching the DB.

### Key invariants

- **No lookahead**: feature joins are point-in-time as of the sample date — fundamentals on `filed_at` (SEC receipt), LSEG estimates on `as_of_date` (observation date), each looked up independently per field; never `period_end`.
- **Survivorship**: the universe includes removed-from-index names (`active=false`, `removed_at` set) with their price history, so the cross-sectional panel isn't survivor-only (de-survivorshipping deflated older ICIRs to honest levels). Estimates now cover removed names; fundamentals/sentiment for them are still a gap.
- **Cross-sectional scoring**: `direction_prob` in `predictions` stores the clipped predicted percentile rank (0–1), not a calibrated probability. Dashboard copy should say "relative rank."
- **Rank stability** (`predictions.confidence`): std of predicted rank across the last ≤3 scoring dates. Lower = steadier model view of that name. Null if fewer than 2 dates available.
- **Horizon trading days**: 1M=21, 3M=63, 6M=126, 1Y=252. Defined in `calendar.py:HORIZON_TRADING_DAYS`.
- **1M horizon**: no detectable cross-sectional signal (t=0.59 in walk-forward). Skip in inference.

### Data sources

- Price/volume: yfinance (incremental, drift-corrected for splits/dividends)
- Fundamentals: SEC EDGAR companyfacts XBRL API (10-K + 10-Q, TTM where applicable)
- Sentiment: yfinance news (~30 days lookback) scored by FinBERT (`ProsusAI/finbert`)
- Analyst estimates: LSEG Workspace via `lseg.data` desktop session (recommendation/price-target consensus, revenue estimates + actuals, forward P/E & EV/EBITDA); monthly point-in-time history. Manual backfill only — not a daily pipeline stage (the desktop session needs Workspace running locally), so estimates go stale unless re-backfilled.
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