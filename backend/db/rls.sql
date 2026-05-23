-- Row-level security for stock-thing.
-- Only portfolio_holdings has RLS — all other tables are catalog/market data,
-- read-anonymously by the dashboard. The service-role key bypasses RLS for
-- server-side ingestion and training jobs.

alter table portfolio_holdings enable row level security;

drop policy if exists portfolio_select_own on portfolio_holdings;
create policy portfolio_select_own
    on portfolio_holdings
    for select
    using (user_id = auth.uid());

drop policy if exists portfolio_insert_own on portfolio_holdings;
create policy portfolio_insert_own
    on portfolio_holdings
    for insert
    with check (user_id = auth.uid());

drop policy if exists portfolio_update_own on portfolio_holdings;
create policy portfolio_update_own
    on portfolio_holdings
    for update
    using (user_id = auth.uid())
    with check (user_id = auth.uid());

drop policy if exists portfolio_delete_own on portfolio_holdings;
create policy portfolio_delete_own
    on portfolio_holdings
    for delete
    using (user_id = auth.uid());

-- Catalog/market-data tables: explicitly grant read to authenticated users.
-- Supabase's anon role is intentionally NOT granted — the dashboard requires login.
grant select on tickers           to authenticated;
grant select on price_history     to authenticated;
grant select on fundamentals      to authenticated;
grant select on sentiment_daily   to authenticated;
grant select on predictions       to authenticated;
grant select on model_versions    to authenticated;

-- Headlines may be heavy and per-ticker; gate via the API rather than direct PostgREST.
-- (No grant to authenticated; the FastAPI service role will read these.)

-- portfolio_holdings is reachable by authenticated users only, gated by RLS above.
grant select, insert, update, delete on portfolio_holdings to authenticated;
