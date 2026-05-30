-- EPS consensus + actual for the PEAD / earnings-surprise feature (Phase 2).
-- Mirrors revenue_mean / revenue_actual: we store RAW consensus and reported EPS
-- and COMPUTE the surprise downstream (this LSEG license does not expose a direct
-- EPSSurprise field — same compute-it-ourselves split as revenue_surprise).
alter table analyst_estimates add column if not exists eps_mean numeric;
alter table analyst_estimates add column if not exists eps_actual numeric;
