# Cardinal Ranker

A cross-sectional stock ranker for the S&P 500. It ranks each stock against peers in its GICS sector, using public fundamentals, analyst estimates, insider transactions, short interest, and news sentiment.

**Live app:** https://cardinal-ranker.vercel.app. The backend runs on a free tier, so the first request can take about a minute to wake up. [Demo login, if applicable: …]

<!-- Add 1–2 screenshots here: rankings view + a single-ticker view -->

## What it does and why

The question: can public, point-in-time information rank stocks by their forward returns *relative to their sector peers*? I framed it as cross-sectional ranking, not price prediction. The goal is to identify which names outperform their peers, not to forecast where the market goes.

## Data: point-in-time and survivorship-bias-free

Most of the work is in the dataset. Backtests on financial data can leak future information in subtle ways.

- **Scale:** 22-table Postgres database (Supabase), about 4.3M rows, 16.7 years of daily data (2010–present), 878 tickers.
  - Includes current *and* delisted S&P 500 members; 875 point-in-time index-membership intervals.
- **Sources:** daily prices; SEC XBRL accounting facts; analyst estimates (LSEG); insider transactions; short interest; earnings surprises; headline sentiment (FinBERT).
- **As-of joins:** every feature is joined on the date it became public (filing, report, or as-of date), never the period it describes.
- **No back-adjusted values:** market cap uses the as-reported share count from filings at that date, not today's share count or back-adjusted prices.
- **Censored names kept:** acquired and delisted names are labeled at their last-trade exit price instead of dropped, so the model doesn't train only on survivors.
- **Missing data stays missing:** missing source data stays `NaN` (never `0`), with explicit availability flags such as `fund_available` and `est_available`.
- **Idempotent ingestion:** `INSERT ... ON CONFLICT` upserts; split/dividend drift detection triggers a re-pull; symbol-reuse detection; an `ingestion_runs` audit log.

## Model

**LightGBM**, sector-relative ranking. The features are tabular, mixed in type, and often missing. LightGBM handles missing values natively, and the availability flags let it tell "no data" apart from a real value.

## Validation

- **Walk-forward cross-validation:** the model always trains on the past and is evaluated on unseen future periods.
- **Block bootstrap:** checks whether the ranking signal is distinguishable from noise, given autocorrelated returns.
- **Result:** [backtested IC, horizon, and a simple baseline such as 12-month momentum]

## Architecture

```
backend/   FastAPI + asyncpg (raw SQL) + ingestion pipelines + LightGBM; FinBERT sentiment (PyTorch)
frontend/  Next.js 15 App Router + TypeScript + Tailwind (Vercel)
scripts/   One-off backfills, ticker seeding
deploy/    launchd schedule for the daily pipeline
```

**Daily pipeline** (runs after the NYSE close; skips market holidays):
prices (daily) → FinBERT sentiment (daily) → EDGAR fundamentals (Fridays) → GBM re-score (Fridays + month-starts).

---

## Running it locally

> Requires an LSEG Workspace session for analyst-estimate ingestion.

```bash
# Backend
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[ingestion,ml,dev]"
cp .env.example .env  # fill in Supabase creds
python -m backend.api.main  # http://localhost:8000/health

# Frontend
cd frontend
cp .env.example .env.local  # fill in NEXT_PUBLIC_SUPABASE_* keys
npm install
npm run dev  # http://localhost:3000
```

### Database setup (Supabase)

1. Create a Supabase project.
2. Apply the schema: `backend/db/schema.sql`, then `backend/db/rls.sql`, then `backend/db/seed_tickers.sql`.
3. Fill `.env` with `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_KEY`, `SUPABASE_JWT_SECRET`, and `DATABASE_URL`. Use the pooled connection string, in `postgresql+asyncpg://` form.
4. Backfill SEC CIKs: `python -m scripts.backfill_ciks`

### Scheduling the daily pipeline (macOS)

```bash
cp deploy/launchd/com.stockthing.daily-pipeline.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.stockthing.daily-pipeline.plist
python -m backend.jobs.daily_pipeline   # run manually
```