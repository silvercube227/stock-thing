-- Migration 002: add analyst_estimates (LSEG/I-B-E-S point-in-time consensus).
-- Apply via Supabase SQL Editor or psql before running scripts.backfill_estimates.
-- New table; no backfill of existing rows needed.

create table if not exists analyst_estimates (
    ticker_id           bigint not null references tickers(ticker_id) on delete restrict,
    as_of_date          date not null,           -- LSEG observation date; the PIT join key
    rec_mean            numeric,                 -- consensus recommendation (1=Strong Buy .. 5=Sell)
    price_target_mean   numeric,
    revenue_mean        numeric,                 -- forward consensus revenue (monthly)
    revenue_actual      numeric,                 -- reported revenue (report-date rows only)
    fwd_pe              numeric,                 -- forward P/E ratio
    fwd_ev_ebitda       numeric,                 -- forward EV/EBITDA ratio
    ingested_at         timestamptz not null default now(),
    primary key (ticker_id, as_of_date)
);

create index if not exists analyst_estimates_ticker_asof_idx
    on analyst_estimates (ticker_id, as_of_date);
