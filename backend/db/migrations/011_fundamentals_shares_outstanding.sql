-- Migration 011: point-in-time shares outstanding on the fundamentals row.
--
-- `tickers.shares_outstanding` is a single CURRENT scalar, so
-- `log(price * tickers.shares_outstanding)` applies today's share count to a
-- 2012 bar and misstates historical market cap by the whole intervening
-- buyback/issuance history. The as-reported cover-page count is contemporaneous
-- with the filing, so it belongs on the fundamentals row and inherits the
-- existing PIT join key (filed_at) and index (ticker_id, filed_at).
--
-- Source: EDGAR companyfacts cover-page fact dei:EntityCommonStockSharesOutstanding
-- (unit "shares"), falling back to us-gaap:CommonStockSharesOutstanding. The fact's
-- own `end` is the cover measurement date (typically weeks AFTER period_end), so the
-- parser takes the latest end <= filed_at rather than period-matching it.
--
-- Caveat: for multi-class issuers the non-dimensional companyfacts fact may carry a
-- single class's count.

alter table fundamentals add column if not exists shares_outstanding bigint;
