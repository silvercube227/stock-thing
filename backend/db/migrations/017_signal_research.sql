-- Research metadata. Existing rows deliberately remain unknown until verified/backfilled.
alter table tickers add column if not exists ric text;
alter table tickers add column if not exists research_only boolean not null default false;
alter table price_history add column if not exists price_basis text;
alter table fundamentals add column if not exists shares_measured_at date;
alter table fundamentals add column if not exists shares_concept text;
alter table fundamentals add column if not exists shares_kind text;
alter table fundamentals add column if not exists shares_basis text;
alter table index_membership add column if not exists index_id text not null default 'SPX';
alter table index_membership drop constraint if exists index_membership_pkey;
alter table index_membership add primary key (ticker_id, index_id, valid_from);

create table if not exists security_identifiers (
    ticker_id bigint not null references tickers(ticker_id),
    identifier_type text not null, identifier text not null,
    valid_from date not null, valid_to date, source text not null,
    primary key (ticker_id, identifier_type, identifier, valid_from),
    check (valid_to is null or valid_to > valid_from)
);
create table if not exists security_events (
    ticker_id bigint not null references tickers(ticker_id),
    effective_date date not null,
    event_type text not null check (event_type in ('acquisition', 'bankruptcy',
        'liquidation', 'security_replacement', 'trading_termination', 'unresolved_gap')),
    consideration_type text, cash_value double precision, proceeds_basis text,
    horizon_values jsonb, source text not null, verified boolean not null default false,
    primary key (ticker_id, effective_date)
);
-- Original filing spans, including YTD flows; later accessions never overwrite earlier ones.
create table if not exists accounting_facts (
    ticker_id bigint not null references tickers(ticker_id), accession_number text not null,
    filed_at date not null, period_start date, period_end date not null,
    metric text not null, value double precision not null, source_concept text not null,
    span_start date generated always as (coalesce(period_start, period_end)) stored,
    primary key (ticker_id, accession_number, metric, span_start, period_end, source_concept),
    check (period_end <= filed_at)
);
create table if not exists fixed_estimates (
    ticker_id bigint not null references tickers(ticker_id), as_of_date date not null,
    fiscal_period_end date not null, contributor_id text not null default 'consensus',
    eps double precision not null, adjustment_basis text not null,
    source text not null, verified boolean not null default false,
    primary key (ticker_id, as_of_date, fiscal_period_end, contributor_id)
);
create table if not exists macro_vintages (
    series_id text not null, obs_date date not null, available_from date not null,
    vintage_date date not null, value double precision, source text not null,
    verified boolean not null default false,
    primary key (series_id, obs_date, vintage_date),
    check (available_from >= obs_date)
);
create table if not exists research_news_daily (
    ticker_id bigint not null references tickers(ticker_id), score_date date not null,
    scorer_revision text not null, n_events integer not null,
    sentiment_sum double precision, n_negative integer not null,
    guidance_up integer not null, guidance_down integer not null,
    operating_n integer not null, operating_sentiment_sum double precision,
    primary key (ticker_id, score_date, scorer_revision)
);
create table if not exists research_news_coverage (
    ticker_id bigint not null references tickers(ticker_id),
    start_date date not null, end_date date not null, status text not null,
    source text not null, primary key (ticker_id, start_date, end_date),
    check (status in ('complete', 'empty', 'partial', 'error'))
);
