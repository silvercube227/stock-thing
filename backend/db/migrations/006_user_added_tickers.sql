-- User-added (off-index) tickers.
--
-- The seeded universe is the S&P 500 (active = true). This flag marks tickers a
-- user added themselves so they can be evaluated by the production model without
-- joining the index: they are `active = true` (so search / portfolio / detail work
-- and the daily ingestion + scheduled scoring keep them fresh) AND `user_added =
-- true`, which (a) excludes them from model TRAINING (gbm_inference.run drops them
-- from the fit), (b) keeps them OUT of the /rankings screener, and (c) drives the
-- UI accuracy disclaimer. They are never deactivated automatically.
alter table tickers add column if not exists user_added boolean not null default false;
