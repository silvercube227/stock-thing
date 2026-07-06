-- Migration 010: SEC Form 3/4/5 insider transactions (Cohen-Malloy-Pomorski 2012).
-- One row per (accession_number, transaction_idx) — a single non-derivative
-- transaction line inside an insider filing.
--
-- The PIT join key is filing_date (SEC receipt of the Form 4), NEVER transaction_date:
-- the trade itself is unknowable to the market until the filing is public (Form 4 is
-- due within 2 business days of the trade). Joining on transaction_date would leak the
-- ~2-day-ahead knowledge the filer had. acceptance_datetime is an optional finer PIT
-- key (intraday) when the source carries it.
--
-- Source: SEC DERA "Insider Transactions Data Sets" (quarterly bulk TSVs, 2006+) for
-- backfill; per-CIK submissions JSON + Form 4 XML for the daily incremental.
-- transaction_code: P=open-market buy, S=open-market sale, A=grant, M=option exercise,
-- etc. All codes are stored; the feature layer filters to P/S.

create table if not exists insider_transactions (
    ticker_id            bigint not null references tickers(ticker_id) on delete restrict,
    accession_number     text not null,          -- SEC filing accession (e.g. 0001209191-24-000123)
    transaction_idx      int not null,           -- 0-based line index within the filing's non-deriv table
    insider_cik          text,                   -- reporting owner CIK
    insider_name         text,
    is_officer           boolean not null default false,
    is_director          boolean not null default false,
    is_ten_pct_owner     boolean not null default false,
    filing_date          date not null,          -- PIT join key: SEC receipt date of the Form 4
    acceptance_datetime  timestamptz,            -- optional finer PIT key (intraday acceptance)
    transaction_date     date,                   -- when the trade occurred (NOT a join key)
    transaction_code     text,                   -- P/S/A/M/... (see header)
    shares               numeric,                -- transacted share count
    price_per_share      numeric,                -- transaction price
    value                numeric,                -- signed dollar value (+buy / -sell), shares*price
    ingested_at          timestamptz not null default now(),
    primary key (accession_number, transaction_idx)
);

-- PIT lookups: all of a ticker's insider lines visible as-of a given date.
create index if not exists insider_transactions_ticker_filing_idx
    on insider_transactions (ticker_id, filing_date);
