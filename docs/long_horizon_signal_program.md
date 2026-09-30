# Long-horizon signal research program

Implemented against the corrected plan requested on 2026-09-07. Production
feature specifications are unchanged. Correctness changes deliberately alter
newly assembled historical data; old caches must be refreshed.

The user's 2026-09-19 exploratory standard in `experiment_gate_status.md`
supersedes exhaustive source certification as a prerequisite to exploratory
experiments. Use `scripts.signal_research run --exploratory` with a frozen
exploratory reference. Target 95% usable coverage; 90% is acceptable. Known
problems stay excluded or missing, and chronology and paired samples remain
enforced. Incomplete source labels remain incomplete; exploratory results do
not establish supported improvement or authorize production promotion.

## Evaluation contract

The seven primary comparisons are analyst/3M, news/6M, accounting/1Y,
stress/6M, macro/6M, S&P 1500 training/6M, and combined/6M. No subset search or
secondary-horizon rescue is supported by the research runner.

Each comparison writes its configuration before fitting and refuses to overwrite
an existing run. Single-seed L2 screening preserves the production target, window,
and overlays. A paired mean improvement of 0.003 earns eight-seed confirmation
using the production objective. Confirmation must reuse the exact input snapshot
and feature list.

Paired evaluation checks dates, security identities, outcomes, sectors, size
controls, and execution/realization dates. It evaluates final overlay/smoothed
predictions, not raw model scores. It reports sector IC, size-neutral IC and
sector-balanced top-decile excess log return. Tied selections receive fractional
weights. This is a ranking diagnostic, not a cost-adjusted portfolio backtest.

Calendar moving-block bootstrap uses 3/6/12 months and repeats with twice those
lengths. Missing months are never joined. Fewer than five effective blocks, or
isolated segments shorter than the requested block, yield insufficient inference.
Absolute IC uncertainty and paired-difference uncertainty are separate outputs.
`approx_95pct_threshold_ic` replaces the misleading reporting label;
`min_detect_ic` remains a deprecated dictionary alias in the legacy evaluator.

Holm adjustment always accounts for seven comparisons, including p=1 for
unexecuted families. Supported historical improvement requires confirmed ΔIC at
least 0.003, a positive 95% paired interval, adjusted p<0.05, and nonnegative
size-neutral/top-decile changes. A positive doubled-block interval additionally
earns the strong-evidence label. These labels do not promote a model.

The selection cutoffs remain 2023-09-30 / 2023-06-30 / 2022-12-31. Actual
selection labels must finish before 2024-01-01, and training labels must finish
before each test date. The previously inspected 2024+ period is not an untouched
holdout and is not used by this runner.

## Data and feature contracts

- Prices: Yahoo close is split-adjusted; reconstruct as-traded close with explicit
  source/action metadata. Apply splits since the disclosed share measurement date.
  Current shares and weighted-average shares cannot substitute for dated shares.
  Unknown price/share bases and unverified historical price targets emit NaN.
- Terminal labels: require verified proceeds on the entry-price adjustment basis.
  Cash stays cash through the horizon; successor consideration needs its value at
  that horizon. Missing terminal information stays masked. Index removal is not a
  terminal event. Documented zero recovery uses log(1e-12) solely to provide finite
  model input and is separately tagged; it is not an estimate of recovery.
- Identity: security IDs remain distinct from issuer CIKs. Membership reconstruction
  runs backwards from current constituents and rejects unexplained events; it does
  not manufacture endpoints or merge share classes by issuer.
- News: raw versions, version-security mappings, scores and fetch coverage remain
  in SQLite. Exact version/text hashes deduplicate; later text never replaces earlier
  versions. Capped calls subdivide recursively. Archive ordering is `newToOld`;
  local aggregation restores chronological order. Midnight-only stamps are delayed
  through the UTC publication date because intraday precision is unverified.
- News scoring: dated issuer aliases and conservative clause attribution reject
  ambiguous multi-issuer clauses. FinBERT is pinned to an exact model revision;
  aliases and rule code enter the scorer fingerprint. Full scoring requires 200
  independently annotated locked validation examples from a 400-story sample,
  relevance precision ≥0.90, event precision ≥0.80 and sentiment macro-F1 ≥0.70.
  Human gold labels are never synthesized from the classifier's predictions.
- Analyst: revisions compare the same fiscal target and adjustment basis, not
  rolling FY1 targets. Persistence requires four distinct monthly snapshots.
  Optional breadth uses at least five continuing contributors. Missing/unresolved
  observations remain NaN; no coverage-count proxy is substituted.
- Accounting: original filing spans preserve both quarterly and YTD flows. The
  builder converts cumulative flows to contiguous quarters or uses a complete
  original annual span. Accruals/assets, gross profitability/assets and asset growth
  are unavailable for financials. Later restatements do not replace first-published
  observations in this registered pack.
- Macro: ALFRED observations preserve publication/vintage dates. With date-only
  availability, use the next calendar day conservatively. No latest-vintage CSV
  fallback. Rates/credit betas use 60 completed months and at least 36 observations;
  rank(beta) is multiplied by a causal standardized shock and never re-ranked.
  Macro inputs are included in the immutable snapshot.
  The 2026-09-08 user decision retains DGS10/BAA10Y for the initial comparison;
  extended longer-history sourcing is conditional on initial evidence. See
  `macro_research_decision.md`. This does not change the evaluation contract.
- Stress: expanding 80th percentile, 24-observation burn-in, multiplied by normalized
  3M and 12-minus-1M momentum after normalization. No threshold tuning.
- Universe: expanded training uses a fixed SPX market control. Predictions for the
  original SPX evaluation cohort are extracted from the expanded cross-section;
  original labels and size controls are retained. Expanded performance is separate.

Geographic revenue, PMI, FX/oil, and fundamental refinancing exposures remain
deferred. No current-sector historical fallback qualifies S&P 1500 for research.

## Commands and artifacts

Use the project's virtual environment. Migrations 017–019 were applied to the
configured database on 2026-09-07. Migration 017 passed a rollback rehearsal and
preserved all 728 membership rows. Migration 018 enables RLS on all seven new
tables and revokes PUBLIC, anon and authenticated grants. No browser-role
policies are provided: research tables are backend-only. The configured backend
role (`postgres`) successfully exercised add-ticker DB helpers with RLS enabled;
anonymous/authenticated reads and writes were denied. The integration check
rolled back its test rows. This is not a complete external-feed/model-scoring test.

The SEC backfill completed for 720 securities with CIKs, storing 397,625 facts
and processing 41,135 filings. Raw responses are archived by SHA256 under
`.research/backfill-verified/raw`; checkpoints are bound to the database.
38,024 filing rows have dated point-in-time share metadata; 2,476 weighted-average
rows remain unverified and unavailable to the corrected cap feature. Exact
source replay passed for AAPL, GOOGL, NVDA, JPM, GOOG, GE, COL and EVHC. Source
replay proves the stored values match the archived response, not that every
share-class attribution, filing availability time or adjustment basis is valid.

```sh
stockproject/bin/python -m scripts.probe_signal_sources --output .research/probes/new.json
stockproject/bin/python -m scripts.backfill_signal_data --source accounting
stockproject/bin/python -m scripts.rebuild_macro_vintages download --directory .research/macro-new
stockproject/bin/python -m scripts.rebuild_macro_vintages apply --directory .research/macro-new --output .research/verification/macro-new-rehearsal.json
stockproject/bin/python -m scripts.rebuild_macro_vintages apply --directory .research/macro-new --output .research/verification/macro-new-applied.json --commit --rehearsal .research/verification/macro-new-rehearsal.json
stockproject/bin/python -m scripts.backfill_signal_data --source news --symbols AAPL KO
stockproject/bin/python -m scripts.signal_research snapshot --output .research/reference --source-report docs/signal_source_readiness.json
stockproject/bin/python -m scripts.signal_research run --snapshot .research/reference --family stress --output .research/runs
stockproject/bin/python -m scripts.signal_research run --snapshot .research/reference --family stress --phase confirmation --output .research/runs
stockproject/bin/python -m scripts.signal_research report --output .research/runs
```

Backfill and snapshot commands use configured `DATABASE_URL`; use a dedicated
local research database for expanded-universe data, never production Supabase.
Set `FRAME_CACHE_DIR=.frame_cache_research` when changing databases. Checkpoints
must also be separate per database/backfill. Raw news stays under `.news_cache`.

`scripts.news_research` provides annotation export, validation, scoring and local
aggregate export. It requires dated reviewed aliases, an exact FinBERT revision,
and sector/removal strata. The aliases format is `{ticker_id: [{alias, valid_from,
valid_to}]}`; strata are `{ticker_id: {sector, cohort}}`, with cohort `surviving` or
`removed`. Exported gold fields remain blank for independent annotation.

The research runner rejects blocked readiness entries and all-missing candidates.
Do not change readiness to `verified` without recording the actual source audit
evidence. Snapshots are immutable local trusted pickles with input hashes, schema
version, source report and code identity. Existing generic frame caches are not
valid research snapshots. Prospective logging writes one immutable local file per
batch and rejects entries in the past or records already containing outcomes.

## Current source findings and remaining execution

See `signal_source_readiness.json`. Live read-only probes established:

- Workspace is reachable outside the sandbox.
- Monthly EPS includes absolute fiscal labels; endpoint/adjustment/contributor
  verification remains outstanding.
- The full SPX archive returned 751 join/leave events for 2010–2026 and 503 current
  RICs. A source-reviewed zero-duration predecessor resolves EVHC.N^L16's
  same-day pair, yielding 876 intervals over 862 RICs. Dated DB identity mapping
  remains incomplete (267 missing/mismatched and two ambiguous candidates);
  assumed DB intervals were not replaced. Existing intervals have no overlaps.
- BNI's dead RIC returned historical prices, but this does not verify adjustments
  or terminal proceeds.
- The historical GICS query returned one current sector with no date.
- The 2013 Reuters English sample returned 11 headlines with version fields;
  sample version timestamps were midnight-only.
- `FRED_API_KEY` is absent. Local PostgreSQL executables are absent.

No corrected full-universe reference, real-data ablation, confirmation, or
prospective candidate has been produced. Source semantics, security mappings,
terminal proceeds, historical sectors and independent news annotation remain
gates. No model improvement or statistical significance is claimed.

Implementation verification: 345 backend tests passed, including a real LightGBM
paired-run smoke test on synthetic inputs. Targeted static checks and
`git diff --check` passed. Migration and RLS checks ran on the configured live
PostgreSQL database. Local PostgreSQL for expanded-universe research is still absent.
The add-ticker scoring path automatically reloads incompatible frame caches from
the database; research callers still reject them. Frame cache version 3 carries
separate share-action metadata. Live rollback checks also exercise the actual
price upsert, retaining classifications for unchanged adjustments and clearing
them when the source factor changes. Production feature specs remain frozen.

The foundation follow-up classified 179 adjustments (44 pricing-only), left 131
unknown, measured accounting feature coverage, and inserted Envision's documented
$46 cash acquisition after SEC/security/price-basis checks. See
`docs/signal_live_verification.md`. New snapshots include a hashed source-code
archive; older snapshots retain their original input hashes and limitations.

Before data execution can close the program, complete full identity/membership
reconstruction, adjustment parity and terminal-event review; complete
annotation/vintage requirements; then
freeze and run the reference. Unsupported paths remain explicitly blocked.

ALFRED API reference: https://fred.stlouisfed.org/docs/api/fred/series_observations.html
