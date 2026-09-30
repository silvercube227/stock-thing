-- Migration 012: point-in-time S&P 500 index membership intervals.
--
-- De-survivorshipping added removed-from-index names to the panel, but INCLUSION
-- was still not point-in-time: a name added to the index in 2023 has price history
-- back to 2016 and appeared in the 2017 cross-sections, which selects on future
-- index promotion (the mirror image of survivorship bias). This table lets the
-- panel builder ask "was this ticker a member on this date?".
--
-- One row per contiguous membership interval; a name removed and later re-added
-- gets multiple rows. valid_to is EXCLUSIVE (membership holds for
-- valid_from <= d < valid_to); null valid_to = still a member.
--
-- Source: the "Selected changes" table on Wikipedia's S&P 500 constituents page
-- (both the added and removed columns). `source` records interval provenance:
--   'wikipedia_changes'  - both endpoints observed in the changes table
--   'assumed_start'      - removal observed, no addition in window -> opened at 2010-01-01
--   'assumed_pre_window' - current constituent with no change rows -> member throughout
--
-- Known gap: names removed before 2016 were never seeded into `tickers`, so
-- pre-2016 membership is incomplete on the removal side.

create table if not exists index_membership (
    ticker_id   bigint not null references tickers(ticker_id) on delete restrict,
    valid_from  date not null,              -- inclusive
    valid_to    date,                       -- exclusive; null = still a member
    source      text not null default 'wikipedia_changes',
    ingested_at timestamptz not null default now(),
    primary key (ticker_id, valid_from)
);

-- As-of lookups: all intervals still open at a given date.
create index if not exists index_membership_valid_to_idx
    on index_membership (valid_to);
