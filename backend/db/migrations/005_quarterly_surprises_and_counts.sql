-- Quarterly earnings-surprise rebuild + analyst-coverage counts.
--
-- The original eps/revenue surprise was computed from analyst_estimates.{eps,revenue}_actual,
-- which LSEG returns at ANNUAL (FY) frequency by default — ~13 reports/ticker. PEAD is a
-- quarterly effect, so that signal was degraded. earnings_surprises holds one row per fiscal
-- quarter with the pre-report consensus, the reported actual, and the report_date (the PIT
-- anchor). Probe confirmed TR.*Mean(Period=FQ0) is the pre-report consensus, not a revision.
create table if not exists earnings_surprises (
    ticker_id     bigint not null references tickers(ticker_id) on delete restrict,
    period_end    date not null,            -- fiscal quarter end
    report_date   date not null,            -- announcement date; THE point-in-time join key
    eps_consensus numeric,                  -- pre-report consensus EPS for the quarter
    eps_actual    numeric,                  -- reported EPS
    rev_consensus numeric,                  -- pre-report consensus revenue
    rev_actual    numeric,                  -- reported revenue
    ingested_at   timestamptz not null default now(),
    primary key (ticker_id, period_end)
);

-- Surprise features look up the latest report with report_date <= sample date.
create index if not exists earnings_surprises_ticker_report_idx
    on earnings_surprises (ticker_id, report_date);

-- Revision-momentum pack inputs (monthly PIT snapshots, alongside the existing consensus
-- fields). This license has no recommendation-bucket counts, but exposes analyst coverage
-- and the price-target estimate count.
alter table analyst_estimates add column if not exists num_analysts numeric;
alter table analyst_estimates add column if not exists pt_num_estimates numeric;
