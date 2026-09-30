-- Preserve raw evidence while preventing reviewed wrong-security histories from
-- reaching features, market controls, labels or successor valuations.
alter table tickers add column if not exists price_source_exclusions jsonb
    not null default '{}'::jsonb check (jsonb_typeof(price_source_exclusions) = 'object');
comment on column tickers.price_source_exclusions is
    'Source -> documented exclusion evidence. Excluded raw prices are retained in price_history but not loaded into model frames.';
