# stock-thing — progress & resume notes

> Last updated: 2026-05-22. Resume this build from **step 5 (FinBERT sentiment)**.

The full architectural plan lives at `/Users/bennettye/.claude/plans/you-are-helping-me-dreamy-narwhal.md`. This file tracks execution state — what's done, what's next, and the non-obvious decisions made along the way.

---

## Status at a glance

| # | Step | Status | Notes |
|---|------|--------|-------|
| 1 | Repo scaffolding | ✅ done | Backend pyproject + Next.js 16 app + dirs |
| 2 | Supabase schema, RLS, seed | ✅ done | 35 tickers (4 ETFs + 31 equities), all CIKs backfilled |
| 3 | Price ingestion (yfinance) | ✅ done | 43,960 rows over 5y for the universe |
| 4 | Fundamentals (SEC EDGAR) | ✅ done | 2,014 filings (2009-Q2 → 2026-Q1); look-ahead-clean |
| 5 | FinBERT sentiment | ⏭️ **next** | Local-only (M4 MPS); pushes daily aggregates to Supabase |
| 6 | Feature builder | ⏳ | Point-in-time correct; look-ahead unit test before code |
| 7 | Transformer (PatchTST) | ⏳ | LSTM was rejected — see plan §4. PyTorch + MPS needed |
| 8 | Model registry + promotion | ⏳ | Promotion rule baked into plan §5 |
| 9 | Daily inference job | ⏳ | Writes to `predictions` |
| 10 | FastAPI endpoints | ⏳ | Smaller surface than originally planned — see "Architecture clarifications" |
| 11 | Next.js dashboard MVP | ⏳ | Direct Supabase reads via `@supabase/ssr` |
| 12 | SentimentGauge + HorizonChart | ⏳ | Recharts |
| 13 | launchd orchestrator | ⏳ | Single nightly cron |
| 14 | Live-accuracy + auto-rollback | ⏳ | After ≥1mo of realized predictions |

---

## Architecture clarifications made during the build

These came out of mid-build discussions and supersede earlier plan drafts:

1. **Model is a transformer, not an LSTM.** Specifically a PatchTST-style encoder (4 layers, 8 heads, `d_model=128`, 21-day patches → 12 patches per 252-day window, ~1M params). Justification: attention scales better as the universe grows. Plan §4 has the full spec.
2. **Next.js owns the dashboard read/write path; FastAPI is ML-only.** Next.js Server Components read Supabase directly via `@supabase/ssr`. FastAPI only handles retrain, promote, rollback, and the `POST /tickers` auto-add (which kicks off backfill of a new symbol). This was changed from the original plan after the user pushed back on routing everything through FastAPI.
3. **`POST /tickers` is the auto-add path.** When the user types a symbol in the dashboard's "add holding" form that doesn't exist in the universe yet, the API auto-creates the ticker, allocates the next `embedding_idx`, and kicks off a backfill. Predictions show `cold_start=true` until the next retrain.
4. **Supabase key terminology was updated** — `anon`/`service_role` are now called `publishable`/`secret` (and prefixed `sb_publishable_*` / `sb_secret_*`). JWTs are verified via JWKS (`<URL>/auth/v1/.well-known/jwks.json`), no shared secret.

---

## Environment / config state

**Python venv:** `stockproject/` at repo root, Python 3.14.3.
- Installed: `fastapi`, `uvicorn`, `pydantic`, `pydantic-settings`, `httpx`, `asyncpg`, `sqlalchemy`, `supabase`, `pyjwt[crypto]`, plus the `[ingestion,dev]` extras (`yfinance`, `pandas`, `pandas-market-calendars`, `sec-edgar-downloader`, `pytest`, `ruff`, etc.).
- **Not yet installed:** `[ml]` extra (torch, transformers, scikit-learn, numpy). Heads-up: **PyTorch may not have Python 3.14 wheels yet**. If install fails, fall back to a 3.12 venv just for ML work (data stays in Supabase, so the venv swap is cheap).

**Frontend:** Next.js 16.2.6 (App Router) + TS + Tailwind. `npm install` already run; `npm run build` clean. Replaced the earlier Vite scaffold after the user asked.

**`.env` (repo root):** has `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY`, `SUPABASE_SECRET_KEY`, `SEC_EDGAR_USER_AGENT`, and a working `DATABASE_URL` (Supabase session pooler form: `postgresql://postgres.<REF>:<PASSWORD>@<HOST>:5432/postgres`). Direct connection didn't resolve from this network — that's expected (IPv6-only on free plan).

**Supabase:** all 9 tables created, RLS enabled on `portfolio_holdings` only, 4 RLS policies (`portfolio_{select,insert,update,delete}_own`). All catalog/market-data tables granted `select` to the `authenticated` role; `headlines` is NOT granted (FastAPI mediates).

**`ticker_embedding_seq`:** allocated `embedding_idx` 1–35 to the seeded universe. **Never reuse, never recycle.** Even if a ticker is removed, set `active=false` rather than DELETE.

---

## Data currently in Supabase

```
tickers           35 rows (4 etf + 31 equity); all equities have cik populated
price_history     43,960 rows (35 tickers × ~1256 trading days, 2021-05-24 → 2026-05-22)
fundamentals      2,014 rows (10-K + 10-Q), 2009-Q2 → 2026-Q1
ingestion_runs    several rows; most recent = fundamentals success
headlines         empty (step 5)
sentiment_daily   empty (step 5)
portfolio_holdings  empty
model_versions    empty (step 7)
predictions       empty (step 9)
```

---

## Files created so far

```
pyproject.toml                       # backend deps, extras, ruff/pytest config
README.md                            # quickstart
CLAUDE.md                            # high-level architecture (now references transformer)
PROGRESS.md                          # this file
.env / .env.example                  # publishable/secret keys, DB URL, EDGAR UA
.gitignore                           # includes stockproject/, .next/, .vercel/, models/*.pt

backend/
  __init__.py
  config.py                          # pydantic-settings; computes supabase_jwks_url
  api/                               # (just main.py with /health for now)
    main.py
    routers/                         # empty
  db/
    schema.sql                       # 9 tables + sequence + triggers
    rls.sql                          # 4 policies + grants
    seed_tickers.sql                 # 35 ticker bootstrap
    migrations/                      # empty
  ingestion/
    db.py                            # asyncpg pool helper (statement_cache_size=0 for pgbouncer)
    calendar.py                      # NYSE schedule + HORIZON_TRADING_DAYS table
    prices.py                        # yfinance fetch, drift detection, NYSE gap log, upsert
    fundamentals.py                  # EDGAR companyfacts, two-pass parser, derived ratios
  ml/                                # empty (step 7+)
  jobs/                              # empty (step 13)
    tasks/                           # empty
  tests/
    test_prices_drift.py             # 8 tests — drift detection (pure)
    test_fundamentals_parser.py      # 12 tests — concept fallback, period filter, look-ahead

scripts/
  backfill_ciks.py                   # one-time SEC ticker→CIK fill
  backfill_prices.py                 # CLI for 5y price backfill (--years, --symbols)
  backfill_fundamentals.py           # CLI for fundamentals backfill (--symbols)
  _inspect_prices.py                 # health check utility
  _inspect_fundamentals.py           # health check utility

frontend/                            # Next.js 16 + App Router + TS + Tailwind
  src/{app,api,components,lib}/      # subdirs created, no code yet
  .env.example                       # NEXT_PUBLIC_* keys
  package.json, tsconfig.json, ...   # standard Next defaults
```

---

## Non-obvious decisions / sharp edges encountered

1. **Stack:** we use **asyncpg** for all Python ↔ Postgres access. No psycopg2, no sync SQLAlchemy. This was confirmed when the user ran into a `pg_config` build error on `pip install psycopg2`.
2. **pgbouncer compatibility:** the pool uses `statement_cache_size=0` (see `backend/ingestion/db.py`). Prepared statement caching breaks behind Supabase's transaction pooler. The session pooler is fine but we keep it off everywhere for consistency.
3. **Price URL DSN handling:** the `DATABASE_URL` is written with `postgresql+asyncpg://` for SQLAlchemy compatibility, but asyncpg itself wants `postgresql://`. `backend/ingestion/db.asyncpg_dsn()` strips the driver hint.
4. **yfinance options:** we call `Ticker.history(auto_adjust=False, actions=True)` so we get RAW OHLC + `Adj Close` separately, plus per-bar `Stock Splits` and `Dividends`. The schema stores both raw and adjusted — re-running ingestion when `adj_close` drifts triggers a `period="max"` re-pull (drift threshold is relative `1e-3`).
5. **EDGAR companyfacts parser pitfalls:**
   - A single 10-K's `accn` has restated prior-year entries. **Pick the entry whose `end` matches the filing's max `end`** (±5 days), not the first one in iteration order. This was a real bug — original parser pulled FY2020 revenue into the FY2022 row.
   - Forward-looking entries (`end > filed`) are dropped entirely.
   - Period filter: 340–380 days for `10-K`, 80–100 days for `10-Q`. Drops YTD cumulative entries that would otherwise be picked up alongside the standalone quarter.
   - Concept fallback lists are ordered by preference; in the smoke run, the preferred ASC 606 names matched for most tickers.
6. **Survivorship bias rule (will hurt if violated):** removing a ticker from the universe means `active=false`, never `DELETE`. Training data must include failures.
7. **Look-ahead:** **never** join fundamentals on `period_end`. Always use `filed_at`. The inspect script has a `filed_at < period_end` assertion that must stay at 0.
8. **NYSE close = 16:00 ET** for the sentiment day-bucketing rule (step 5 implements this).

---

## How to resume

1. **Activate the venv:**
   ```bash
   source stockproject/bin/activate
   ```
2. **Sanity-check the data:**
   ```bash
   python -m scripts._inspect_prices         # expect 43,960 rows, 35 tickers
   python -m scripts._inspect_fundamentals   # expect 2,014 rows, 0 leak rows
   ```
3. **Run tests:**
   ```bash
   python -m pytest backend/tests -v   # 20/20 pass as of last run
   ```
4. **Start step 5.** Open the plan at `/Users/bennettye/.claude/plans/you-are-helping-me-dreamy-narwhal.md` §3 (look for "Headlines + Sentiment") for the spec. Brief outline:
   - `backend/ingestion/headlines.py` — pull `yf.Ticker(sym).news` per ticker, dedupe by URL, batch-score via FinBERT on MPS (batch 32), upsert into `headlines`, recompute `sentiment_daily` rolling 7d/14d for the last 14 days.
   - Time-bucket rule: headlines with `published_at` after 16:00 ET roll into the NEXT trading day's score.
   - `scripts/backfill_sentiment.py` — initial bootstrap (yfinance only returns recent news per ticker, typically the last ~30-50 headlines; don't expect 5-year history here).
   - Add to `pyproject.toml [ml]` extra: `transformers`, ensure FinBERT model identifier is `ProsusAI/finbert`.
   - Unit test on the time-bucketing rule (synthetic 15:00 ET vs 17:00 ET headlines).

5. **Step 5 gotchas to plan for:**
   - FinBERT inference on MPS: first run will download ~440MB of weights. Set a cache dir under repo root or `~/.cache/huggingface/`.
   - yfinance news is sparse — for backfill purposes we only get recent items, so this isn't a 5-year backfill. Going forward the daily cron is what populates history.
   - The `headlines` table is NOT granted to `authenticated` in RLS — the dashboard reads aggregates from `sentiment_daily` instead. If you want article drill-downs, expose them via FastAPI rather than direct PostgREST.

---

## Quick recovery commands if something looks off

- Schema state: `psql ... -c "\dt"` or via Supabase SQL Editor:
  ```sql
  select table_name from information_schema.tables where table_schema='public' order by 1;
  ```
- Re-run a full ingest (idempotent):
  ```bash
  python -m scripts.backfill_prices --years 5
  python -m scripts.backfill_fundamentals
  ```
- Test suite:
  ```bash
  python -m pytest backend/tests -v
  ```

If `pip install -e ".[ml]"` fails on Python 3.14, the cleanest path is `pyenv install 3.12.7 && pyenv local 3.12.7` for the ML work — the data in Supabase is independent of the venv.
