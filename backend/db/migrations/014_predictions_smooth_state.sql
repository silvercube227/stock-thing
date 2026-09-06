-- Migration 014: carry the EWMA smoothing STATE on predictions.
--
-- `HorizonSpec.smooth_span` was validated by walk_forward_ic -> ewma_rank_by_ticker,
-- whose recursion carries the *blended* value forward:
--
--     state[t] = alpha * rank_now + (1 - alpha) * state[t]
--
-- Production could not do that: it had nowhere to put `state`, so it blended against
-- `direction_prob` — the RE-RANKED percentile. Re-ranking restores full dispersion to
-- the prior term at every step, so the prior pulled harder in production than in the
-- walk-forward that promoted spans 3 (3M) and 4 (1Y): the shipped ranking was not the
-- one that was measured.
--
-- Storing the raw state makes the production recursion byte-identical to the eval one.
-- Nullable: rows written before this migration have no state, and a name whose state
-- is missing simply seeds from its current rank — exactly what ewma_rank_by_ticker
-- does on a ticker's first appearance.
alter table predictions add column if not exists smooth_state numeric;

comment on column predictions.smooth_state is
    'Raw (un-re-ranked) EWMA state behind direction_prob, advanced once per calendar '
    'month so the cadence matches the monthly walk-forward that validated smooth_span. '
    'Null = seed from the current rank.';
