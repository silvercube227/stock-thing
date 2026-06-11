-- Migration 009: FINRA Reg SHO bimonthly short interest table.
-- One row per (ticker, settlement_date). The PIT join key is publication_date
-- (~14 calendar days after settlement) — the date FINRA actually publishes the
-- data. Using settlement_date as the join key would introduce a lookahead
-- (the positions are measured on settlement_date but unknowable until published).
--
-- Source: FINRA consolidated short interest files (Regulation SHO)
-- Frequency: bimonthly (~1st and ~15th settlement dates each month)
-- History: available from ~2009 via FINRA download site

create table if not exists short_interest (
    ticker_id           bigint not null references tickers(ticker_id) on delete restrict,
    settlement_date     date not null,           -- when the short positions were measured
    publication_date    date not null,           -- PIT join key: ~14 days after settlement
    short_interest      bigint,                  -- shares short
    avg_daily_volume    bigint,                  -- average daily share volume (FINRA-supplied)
    days_to_cover       numeric,                 -- short_interest / avg_daily_volume
    ingested_at         timestamptz not null default now(),
    primary key (ticker_id, settlement_date)
);

-- PIT lookups: find the most recent published snapshot as-of a given date.
create index if not exists short_interest_ticker_pub_idx
    on short_interest (ticker_id, publication_date);
