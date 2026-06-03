-- 006_normalize_sectors.sql
-- Some tickers were stored with raw yfinance sector labels instead of the GICS
-- canonical form used everywhere else (the _YAHOO_TO_GICS map in
-- backend/jobs/add_ticker.py normalizes new adds, but a few historical rows
-- leaked the raw label). Duplicate labels split a sector into two peer groups,
-- which breaks within-sector ranking. Merge them into the canonical GICS name.
update tickers set sector = 'Health Care'            where sector = 'Healthcare';
update tickers set sector = 'Information Technology' where sector = 'Technology';
