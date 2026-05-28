# stock-thing

Personal long-only stock and ETF trend prediction app. 

## Repo layout

```
backend/   FastAPI + PyTorch + ingestion (runs locally on M4)
frontend/  Next.js 15 App Router + TS + Tailwind (deploys to Vercel)
models/    Local .pt weights, gitignored
scripts/   One-off backfills, ticker seeding
```

## Quickstart

```bash
# Backend
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[ingestion,ml,dev]"
cp .env.example .env  # then fill in Supabase creds
python -m backend.api.main  # http://localhost:8000/health

# Frontend
cd frontend
cp .env.example .env.local  # fill in NEXT_PUBLIC_SUPABASE_* keys
npm install
npm run dev  # http://localhost:3000
```

## Database setup (Supabase)

1. Create a new project at https://supabase.com/dashboard. Pick a region close to home.
2. From the Supabase **SQL Editor**, run the files in this order:
   ```
   backend/db/schema.sql        -- tables, indexes, sequences, triggers
   backend/db/rls.sql           -- RLS policies + grants
   backend/db/seed_tickers.sql  -- starter universe (edit symbols first if desired)
   ```
3. Fill `.env` with `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_KEY`, `SUPABASE_JWT_SECRET`, and `DATABASE_URL` (the **pooled** connection string from *Project Settings → Database → Connection pooling*, port 5432, in `postgresql+asyncpg://` form).
4. Backfill SEC CIKs (one-time, for the EDGAR fundamentals pipeline):
   ```bash
   python -m scripts.backfill_ciks
   ```
5. Verify in the SQL Editor:
   ```sql
   select count(*) from tickers where active = true;        -- expect 35
   select symbol, cik, embedding_idx from tickers
    where asset_type = 'equity' order by embedding_idx;     -- CIKs populated
   ```

## Daily pipeline (launchd)

The pipeline runs automatically after NYSE close via a macOS LaunchAgent.

```bash
# Activate (fires Mon–Fri at 17:30 local time)
cp deploy/launchd/com.stockthing.daily-pipeline.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.stockthing.daily-pipeline.plist

# Deactivate
launchctl unload ~/Library/LaunchAgents/com.stockthing.daily-pipeline.plist

# Run manually
python -m backend.jobs.daily_pipeline

# Watch logs
tail -f logs/daily_pipeline.log
```

Holidays are handled automatically — the script checks the NYSE calendar and exits cleanly on non-trading days. Stages: prices (daily) → sentiment/FinBERT (daily) → EDGAR fundamentals (Fridays) → GBM re-score (Fridays + month-starts).

## Build sequence

See `/Users/bennettye/.claude/plans/you-are-helping-me-dreamy-narwhal.md` for the full plan. Current status: step 2 (schema applied).
