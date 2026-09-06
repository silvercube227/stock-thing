-- Migration 016: add a 'queued' status to ingestion_runs.
--
-- POST /tickers spawns the add-ticker worker with subprocess.Popen. That works on the
-- local Mac, where the ML/ingestion stack and the model artifact live — but the hosted
-- read-API (Render, see render.yaml) installs only fastapi/asyncpg/yfinance and has no
-- models/ directory. There the child process dies immediately on ModuleNotFoundError,
-- and because Popen sends stdout/stderr to DEVNULL the failure is SILENT: the API still
-- returns 202 "queued", the ingestion_runs row stays 'running' forever, and the UI
-- polls until the 15-minute stale guard reports "Timed out".
--
-- So the hosted API now records the request as 'queued' and does not pretend to run it;
-- the local machine drains the queue (`python -m backend.jobs.add_ticker --drain`).
-- 'queued' is deliberately distinct from 'running': a queued job has not started, so
-- the staleness clock that catches a hard-killed worker must not apply to it.
alter table ingestion_runs drop constraint if exists ingestion_runs_status_check;
alter table ingestion_runs add constraint ingestion_runs_status_check
    check (status in ('queued', 'running', 'success', 'partial', 'failed', 'skipped'));
