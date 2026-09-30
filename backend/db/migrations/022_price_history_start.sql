-- Bound a stored security history on reviewed evidence without deleting the
-- vendor's linked predecessor rows. This is a price-history boundary, not a
-- claim about the issuer's incorporation date or trading on other venues.
alter table tickers add column if not exists price_history_start date;
alter table tickers add column if not exists price_history_start_source jsonb;
do $$ begin
    if not exists (select 1 from pg_constraint where conname='tickers_price_history_start_evidence'
                   and conrelid='tickers'::regclass) then
        alter table tickers add constraint tickers_price_history_start_evidence
        check ((price_history_start is null and price_history_start_source is null)
            or (price_history_start is not null and price_history_start_source is not null
                and jsonb_typeof(price_history_start_source)='object'));
    end if;
end $$;
comment on column tickers.price_history_start is
    'Inclusive reviewed start of this security in the stored price history; earlier predecessor rows remain raw evidence but cannot enter model frames.';
