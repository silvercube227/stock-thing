-- Migration 015: point-in-time GICS sector history.
--
-- `tickers.sector` is a single CURRENT label applied retroactively to a name's whole
-- 16-year history, and it is load-bearing twice over: it defines the TRAINING TARGET
-- (`sector_return` / `sector_grade` demean the return within (date, sector)) and it
-- defines the HEADLINE METRIC (`within_sector_ic`, the SECB success bar). So every
-- pre-2018 number was computed against peer groups that did not exist at the time:
--
--   * Sept 2018 restructured Telecommunication Services into Communication Services
--     and moved GOOGL/META/NFLX/DIS out of Information Technology and Consumer
--     Discretionary into it.
--   * Sept 2016 split Real Estate out of Financials.
--   * Individual names get reclassified continuously (e.g. retailers between Consumer
--     Discretionary and Consumer Staples).
--
-- Wikipedia's "List of S&P 500 companies" carries a GICS Sector column in revisions
-- back to at least 2010-01-10 (sub-industry from ~2016), and scripts/backfill_sectors.py
-- already fetches and header-parses those revisions. seed_sector_history.py walks them
-- quarterly and merges consecutive equal labels into intervals.
--
-- Interval convention matches index_membership (migration 012): valid_from inclusive,
-- valid_to EXCLUSIVE, null valid_to = still current.
create table if not exists sector_history (
    ticker_id   bigint not null references tickers(ticker_id) on delete cascade,
    valid_from  date not null,
    valid_to    date,
    sector      text not null,
    industry    text,
    source      text not null,   -- wikipedia_snapshot | assumed_pre_first_snapshot |
                                 -- tickers_current | tickers_only
    ingested_at timestamptz not null default now(),
    primary key (ticker_id, valid_from),
    constraint sector_history_interval_ordered check (valid_to is null or valid_to > valid_from)
);

create index if not exists sector_history_lookup_idx
    on sector_history (ticker_id, valid_from);

comment on table sector_history is
    'Point-in-time GICS sector/industry intervals. Consumed by factors/assembly.py so '
    'each panel row carries the sector as of ITS OWN date, which is what the '
    'sector_return/sector_grade target and the within_sector_ic metric then use.';
