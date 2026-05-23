# stock-thing

Personal long-only stock and ETF trend prediction app. See [CLAUDE.md](./CLAUDE.md) for high-level architecture.

## Repo layout

```
backend/   FastAPI + PyTorch + ingestion (runs locally on M4)
frontend/  Vite + React + TS (deploys to Vercel)
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
cp .env.example .env.local  # fill in VITE_SUPABASE_* keys
npm install
npm run dev
```

## Build sequence

See `/Users/bennettye/.claude/plans/you-are-helping-me-dreamy-narwhal.md` for the full plan. Current status: step 1 (scaffolding).
