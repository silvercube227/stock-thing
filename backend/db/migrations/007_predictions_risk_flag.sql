-- Falling-knife transparency tag on predictions.
--
-- The model already computes a vol×downtrend "falling-knife" score at inference time (it
-- powers the 3M output overlay). Rather than only silently re-ranking those names, we now
-- ALSO store a graded tag so the dashboard can show *why* a high-vol downtrender sits where
-- it does. Purely descriptive — it does not affect direction_prob. Values:
--   'high'     = top-vol AND clearly below-trend / near 52w low
--   'elevated' = elevated on both
--   'none'     = neither (or risk features unavailable)
alter table predictions
    add column if not exists risk_flag text
        check (risk_flag in ('none', 'elevated', 'high'));
