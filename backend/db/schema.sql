-- stock-thing schema. Apply via Supabase SQL Editor or psql.
-- Idempotent where possible. Run as the project owner so auth.users is in scope.

-- =============================================================
-- Extensions
-- =============================================================
create extension if not exists pgcrypto;       -- gen_random_uuid()

-- =============================================================
-- Sequences
-- =============================================================
-- embedding_idx is what the LSTM sees. Never recycle, never reuse.
-- A dedicated sequence ensures append-only allocation even if rows
-- are ever deleted by accident.
create sequence if not exists ticker_embedding_seq start 1;

-- =============================================================
-- tickers
-- =============================================================
create table if not exists tickers (
    ticker_id       bigserial primary key,
    symbol          text not null,
    cik             text,                       -- nullable; filled by scripts/backfill_ciks.py
    name            text,
    asset_type      text not null check (asset_type in ('equity', 'etf')),
    exchange        text,
    sector          text,
    industry        text,
    active          boolean not null default true,
    added_at        timestamptz not null default now(),
    removed_at      timestamptz,
    embedding_idx   integer not null unique default nextval('ticker_embedding_seq'),
    created_at      timestamptz not null default now(),
    updated_at      timestamptz not null default now()
);

create unique index if not exists tickers_active_symbol_uniq
    on tickers (symbol) where active = true;
create index if not exists tickers_cik_idx on tickers (cik) where cik is not null;

-- =============================================================
-- price_history
-- =============================================================
create table if not exists price_history (
    ticker_id       bigint not null references tickers(ticker_id) on delete restrict,
    trade_date      date not null,
    open            numeric(18, 6),
    high            numeric(18, 6),
    low             numeric(18, 6),
    close           numeric(18, 6),
    adj_close       numeric(18, 6),
    volume          bigint,
    split_factor    numeric not null default 1.0,
    dividend        numeric not null default 0.0,
    source          text not null default 'yfinance',
    ingested_at     timestamptz not null default now(),
    primary key (ticker_id, trade_date)
);

create index if not exists price_history_trade_date_idx on price_history (trade_date);

-- =============================================================
-- fundamentals
-- =============================================================
create table if not exists fundamentals (
    ticker_id           bigint not null references tickers(ticker_id) on delete restrict,
    accession_number    text not null,           -- EDGAR filing identifier
    filing_type         text not null check (filing_type in ('10-K', '10-Q')),
    period_end          date not null,           -- the period the numbers describe
    filed_at            timestamptz not null,    -- when SEC received it; THIS is the join key
    revenue             numeric,
    net_income          numeric,
    gross_margin        numeric,
    operating_margin    numeric,
    total_debt          numeric,
    total_equity        numeric,
    fcf                 numeric,
    ingested_at         timestamptz not null default now(),
    primary key (ticker_id, accession_number)
);

-- Critical: point-in-time joins use filed_at, never period_end.
create index if not exists fundamentals_ticker_filed_idx
    on fundamentals (ticker_id, filed_at);

-- =============================================================
-- headlines
-- =============================================================
create table if not exists headlines (
    headline_id         bigserial primary key,
    ticker_id           bigint not null references tickers(ticker_id) on delete restrict,
    published_at        timestamptz not null,
    -- NYSE close-bucketed date: if published_at > 16:00 ET, score_date = ET date + 1.
    -- Computed in Python (bucket_score_date) and stored for efficient aggregation.
    score_date          date not null,
    source              text,
    url                 text not null unique,    -- dedupe key
    title               text,
    summary             text,
    sentiment_score     numeric,                 -- FinBERT signed score in [-1, 1]
    sentiment_label     text check (sentiment_label in ('pos', 'neg', 'neu')),
    finbert_version     text,
    scored_at           timestamptz,
    ingested_at         timestamptz not null default now()
);

create index if not exists headlines_ticker_published_idx
    on headlines (ticker_id, published_at);
create index if not exists headlines_ticker_score_date_idx
    on headlines (ticker_id, score_date);

-- =============================================================
-- sentiment_daily
-- =============================================================
-- score_date is bucketed against NYSE close (16:00 ET). Headlines published
-- after close go into the NEXT day. The aggregator (backend/ingestion/headlines.py)
-- owns that bucketing; do not trust the date portion of published_at directly.
create table if not exists sentiment_daily (
    ticker_id       bigint not null references tickers(ticker_id) on delete restrict,
    score_date      date not null,
    mean_score      numeric,
    headline_count  integer not null default 0,
    rolling_7d      numeric,                     -- trailing only, never centered
    rolling_14d     numeric,                     -- trailing only, never centered
    computed_at     timestamptz not null default now(),
    primary key (ticker_id, score_date)
);

-- =============================================================
-- portfolio_holdings  (the ONLY table with RLS)
-- =============================================================
create table if not exists portfolio_holdings (
    id              uuid primary key default gen_random_uuid(),
    user_id         uuid not null references auth.users(id) on delete cascade,
    ticker_id       bigint not null references tickers(ticker_id) on delete restrict,
    shares          numeric not null check (shares >= 0),
    cost_basis      numeric,
    acquired_at     date,
    notes           text,
    created_at      timestamptz not null default now(),
    updated_at      timestamptz not null default now()
);

create index if not exists portfolio_holdings_user_ticker_idx
    on portfolio_holdings (user_id, ticker_id);

-- =============================================================
-- model_versions
-- =============================================================
create table if not exists model_versions (
    model_version_id        uuid primary key default gen_random_uuid(),
    created_at              timestamptz not null default now(),
    training_window_start   date not null,
    training_window_end     date not null,
    holdout_window_start    date not null,
    holdout_window_end      date not null,
    weights_path            text not null,       -- local FS path on the M4
    weights_sha256          text not null,
    n_params                integer,
    n_tickers_trained       integer,
    val_loss                numeric,
    directional_accuracy    jsonb,               -- {"1M": 0.58, "3M": 0.61, ...}
    holdout_metrics         jsonb,
    status                  text not null check (status in ('candidate', 'production', 'retired', 'rolled_back')),
    promoted_at             timestamptz,
    retired_at              timestamptz,
    parent_version_id       uuid references model_versions(model_version_id),
    config                  jsonb not null,      -- hyperparams + feature list + RNG seeds
    check (training_window_end <= holdout_window_start)
);

-- Only one production model at a time.
create unique index if not exists model_versions_one_production
    on model_versions (status)
    where status = 'production';

create index if not exists model_versions_status_created_idx
    on model_versions (status, created_at desc);

-- =============================================================
-- predictions
-- =============================================================
create table if not exists predictions (
    ticker_id           bigint not null references tickers(ticker_id) on delete restrict,
    model_version_id    uuid not null references model_versions(model_version_id) on delete restrict,
    as_of_date          date not null,           -- date the inference ran against
    horizon             text not null check (horizon in ('1M', '3M', '6M', '1Y')),
    direction_prob      numeric not null check (direction_prob between 0 and 1),
    predicted_return    numeric,
    confidence          numeric,                 -- e.g. MC-dropout std or softmax margin
    cold_start          boolean not null default false,
    created_at          timestamptz not null default now(),
    primary key (ticker_id, model_version_id, as_of_date, horizon)
);

-- Dashboard hot path.
create index if not exists predictions_ticker_asof_idx
    on predictions (ticker_id, as_of_date desc);
create index if not exists predictions_asof_idx
    on predictions (as_of_date desc);

-- =============================================================
-- ingestion_runs (observability log for the daily orchestrator)
-- =============================================================
create table if not exists ingestion_runs (
    run_id          bigserial primary key,
    job_name        text not null,               -- e.g. 'prices', 'fundamentals', 'sentiment'
    started_at      timestamptz not null default now(),
    finished_at     timestamptz,
    status          text not null default 'running'
                    check (status in ('running', 'success', 'partial', 'failed')),
    rows_inserted   integer,
    rows_updated    integer,
    error_message   text,
    metadata        jsonb
);

create index if not exists ingestion_runs_job_started_idx
    on ingestion_runs (job_name, started_at desc);

-- =============================================================
-- updated_at triggers (tickers, portfolio_holdings)
-- =============================================================
create or replace function set_updated_at() returns trigger as $$
begin
    new.updated_at := now();
    return new;
end;
$$ language plpgsql;

drop trigger if exists tickers_set_updated_at on tickers;
create trigger tickers_set_updated_at
    before update on tickers
    for each row execute function set_updated_at();

drop trigger if exists portfolio_holdings_set_updated_at on portfolio_holdings;
create trigger portfolio_holdings_set_updated_at
    before update on portfolio_holdings
    for each row execute function set_updated_at();
