-- Migration 001: add score_date to headlines.
-- Apply via Supabase SQL Editor or psql before running step 5 ingestion.
-- The headlines table is empty at this point, so no backfill is needed.

alter table headlines
    add column if not exists score_date date;

-- Make it non-null only after backfilling (table is empty, so this is instant).
alter table headlines
    alter column score_date set not null;

-- Index for the sentiment_daily aggregation query.
create index if not exists headlines_ticker_score_date_idx
    on headlines (ticker_id, score_date);
