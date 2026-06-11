-- Migration 008: EPS estimate dispersion columns in analyst_estimates.
-- Adds the standard deviation and count of analyst EPS estimates (LSEG
-- TR.EPSStdDev / TR.EPSNumIncEstimates) needed to compute the
-- eps_dispersion = eps_std_dev / max(|eps_mean|, 0.01) feature.
-- PIT-safe: these are stored as raw values alongside eps_mean on each
-- as_of_date observation, not derived or lagged.

alter table analyst_estimates add column if not exists eps_std_dev numeric;
alter table analyst_estimates add column if not exists eps_num_inc_estimates numeric;
