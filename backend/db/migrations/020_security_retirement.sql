-- Migration 020: authoritative security retirement date on tickers.
--
-- `tickers.removed_at` records leaving the index, which is not the end of the
-- security. TechnipFMC left the index in 2021 and still trades, so its later bars
-- are real. Starwood, Rockwell Collins and Harman stopped existing, yet bars
-- continue under their symbols for six months, two years and five years
-- respectively because another company later took the symbol.
--
-- `_drop_reused_symbol_bars` could not see this: it fires only when a series
-- begins after removal or resumes after a >180 day hole, and these tails are
-- contiguous with the original company's bars. Recording the vendor's retirement
-- date lets the loader truncate on evidence instead of on a gap heuristic.
--
-- Source: LSEG security lifecycle (RetireDate), resolved to tickers by issuer CIK
-- or class-preserving RIC stem with a lifecycle check
-- (scripts/map_security_identity.py). Null means the security is still listed.

alter table tickers add column if not exists security_retired_at date;
alter table tickers add column if not exists security_retired_source text;

comment on column tickers.security_retired_at is
  'Vendor-observed date the SECURITY stopped existing; distinct from removed_at (index exit).';
