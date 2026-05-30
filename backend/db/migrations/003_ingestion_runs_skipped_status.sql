-- Allow the daily orchestrator to record a stage that was scheduled but
-- deliberately not run as 'skipped', rather than a misleading 'success' or
-- 'failed'. Motivating case: the LSEG estimate stage, which can only run when the
-- Workspace desktop session is reachable — on days it's down we want an honest
-- 'skipped' row, not a failure that pages or a success that implies fresh data.
alter table ingestion_runs drop constraint if exists ingestion_runs_status_check;
alter table ingestion_runs add constraint ingestion_runs_status_check
    check (status in ('running', 'success', 'partial', 'failed', 'skipped'));
