# Experiment gates — 2026-09-19

## User-authorized exploratory standard — 2026-09-19

The user explicitly accepts incomplete data from the available free sources and
prioritizes running experiments over exhaustive source certification. This
supersedes the requirement to finish every historical repair before exploratory
runs. The older certification checklist below remains a record of limitations,
not a requirement to achieve perfect data before learning from experiments.

- Target 95% usable coverage; 90% is acceptable for exploratory comparisons.
  Measure coverage against the fixed intended evaluation cohort before exclusions,
  disclose exclusions by period and surviving/removed cohort, and do not equate
  coverage with a measured percentage of correctness.
- Exclude or mask known-wrong identities, unresolved price/action observations and
  unknown terminal payoffs. Keep missing sectors and features missing. Do not
  spend indefinitely recovering the last 5–10% or select exclusions using results.
- Preserve chronological availability, no-look-ahead labels, and identical paired
  baseline/candidate samples. These are requirements for an interpretable experiment.
- Start with stress/6M on the usable corrected reference. Other families' gaps do
  not block it. Run additional families as their own usable inputs become available.
- Full source certification, exhaustive corporate-action reconstruction, full news
  annotation acceptance and prior supported individual-pack results are not required
  to explore hypotheses. Unvalidated news and combined runs must be explicitly
  exploratory; no fabricated human labels or unsupported validation claims.
- Freeze inputs and exclusions, report coverage and known limitations with results,
  and retain the existing production configuration. Exploratory results do not
  establish supported improvement or authorize production promotion.

This policy records the user's revised acceptance criteria, reiterated on
2026-09-19. The runner now has a distinct `--exploratory` mode requiring a frozen
intended cohort, exclusions, limitations and measured usable evaluation coverage.
The 90–95% target is reported, not used to indefinitely postpone a first
exploratory fit. Below-target results remain labelled below target; they do not
establish broad-cohort validity or clear source-certification gates.
Sources remain incomplete. Exploratory outputs are stored separately and cannot
receive supported-improvement or promotion verdicts. Chronology and matched-panel
checks still run. The intended-cohort denominator includes unmapped and unpriced
members before exclusion, using the archived reconstructed SPX membership.

Five exploratory screens and analyst's eight-seed confirmation have completed;
results are below. Confirmation is inconclusive; news and expanded-universe remain unexecuted.
Do not resume exhaustive source certification
as a prerequisite to the next usable experiment.

The initial preflight revision `2026-09-19-v1` used
`.research/exploratory-reference-2026-09-19-v1` (878 frames, 102 provisional
historical additions, 23 recorded identity/history exclusions). The input hash is
`91716fa41f69ef31eea92acc1e84bf71fdc0ce9351bebad8e0fd308c65149a0b`.
Its stress and macro attempts stopped before fitting: the legacy database
membership denominator counted as many as 565 SPX members and reported 85.09%
usable coverage. Revision v2 uses the already-reconstructed SPX archive for both
membership filtering and the fixed intended cohort, and reports below-target
coverage without declaring it adequate or prohibiting exploratory fits.
The next comparisons are macro/6M and accounting/1Y using their available inputs.
The initial exploratory combined pack is fixed to **stress + macro** before
inspecting any results; inclusion is not selected from the winning individual run.
Analyst/news/universe may proceed when their usable inputs exist, without reopening
exhaustive certification. No performance result is asserted by this launch record.

### Exploratory results — revision v2

Stress/6M screening completed on the frozen v2 reference, input hash
`dd5c443d45d244e946f230f957380fe945e8fbbccf1593f325bb70abfc910eda`.
All 107 monthly prediction pairs / 46,927 rows have matching identities, entry
dates and realization dates; entries follow feature dates and selection labels
finish before 2024. The baseline mean sector IC is -0.03131; adding stress gives
-0.03686, a paired difference of **-0.00555** (95% calendar-block interval
[-0.02301, 0.00716]). Size-neutral IC and top-decile changes are also negative.
This screen does not earn confirmation; no stress threshold or feature is tuned
in response.

Usable coverage is **86.96%** of 53,965 intended security-months. Coverage is
95.11% for the fixed surviving cohort and 42.31% for the removed cohort, so these
results have a material survivorship limitation. Below-target coverage remains
visible and is not relabelled adequate. Result and replayable predictions are in
`.research/experiments-2026-09-19-v2/exploratory/stress/screening/`.
Macro/6M completed with paired ΔIC +0.00167 (95% interval [-0.01036, +0.01517]).
The preselected stress + macro run completed with ΔIC +0.00019 (95% interval
[-0.01954, +0.01769]). Neither earns confirmation. All three 6M baseline prediction
files match exactly, and each pair's identities, labels and controls were checked.
Accounting/1Y completed with ΔIC -0.01634 (95% interval [-0.05176, +0.00524]),
95 folds / 41,672 matched rows and 86.93% usable coverage. It does not earn
confirmation. Its candidate-column comparison bug was fixed before fitting;
data and feature definitions are unchanged. All four screens are complete and
none advances to confirmation. See the concise
`exploratory_results_2026_09_19.md` report.

### Analyst / 3M — screening and confirmation complete

Monthly FY1 and FY2 consensus was archived for all 829 mapped intended-cohort
RICs, January 2010–December 2023. The local snapshot contains 204,062 usable
observations across 796 securities; 48,378 missing/invalid and 15,082
unavailable/expired records were excluded, with no conflicting duplicate keys.
The snapshot is `.research/exploratory-analyst-2026-09-19-v1`, input hash
`0f1ce8c6f5e93c361643b64cc2bd1765328bc0e63439d202dbfffc1acbf7c94d`.
Screening completed after its paired-panel check passed: 49,588 usable rows
out of 56,977 intended security-months (87.03%). Paired sector-IC change is
+0.00717 (95% interval [+0.00139, +0.01345]), clearing the unchanged +0.003
screening threshold. Eight-seed confirmation completed with the same input and
code hashes: paired IC change +0.00460, 95% interval [+0.00046, +0.00881].
It is inconclusive under the full criteria: Holm-adjusted p=0.22748 and top-decile
change -0.00041. No promotion follows. Saved predictions for both phases pass
paired controls and chronology checks; no process remains running.
Returned absolute fiscal targets identify the
same-period revision series; CalcDate establishes availability, never the earlier
last-update date. Inputs remain provisional and local, with `verified=False`.
The exploratory runner explicitly admits these rows; default production feature
assembly continues to exclude them. Missing, expired and conflicting observations
are excluded using fixed rules before fitting. The registered 30/90-day revision
and persistence features, 0.01 EPS cutoff, 3M horizon and screen threshold remain
unchanged. No contributor breadth is available. Vendor fiscal ends and adjusted
USD basis remain disclosed limitations, not prerequisites for another source audit.

News still lacks a broad usable panel: the annotation archive has 105 retained
versions and only 15 complete three-month security windows (13 populated, two
empty), across three selected August cross-sections. Full human annotation
acceptance is not an exploratory prerequisite. Expanded-universe inputs remain
absent from the initialized local research database. Neither gap blocks another
family's runnable experiment.

## Certification requirements (not exploratory blockers)

The actual runner (`scripts/signal_research.py`, `run`) requires all three
source-report entries below to have status `verified` and evidence. They are
currently `incomplete`. Changing the status alone would not verify the data.

| Gate | Work required before verification |
|---|---|
| `price_share_basis` | Review and apply the historical price candidates; establish adjustment/security provenance; explicitly mask unresolved distributions and gaps; disclose resulting coverage. The 46 unresolved share adjustments may remain NaN under the plan. |
| `membership` | Attach reconstructed membership to verified security identities and integrate the separate historical securities into the research reference. Resolve predecessor splices, the VAL conflict, and parent/sector inconsistencies. Of the original 147 registered securities, 146 now have applied exact-RIC membership; MRP remains blocked. Praxair was applied separately. Broader identity and parent/sector reconciliation remains incomplete. |
| `terminal_outcomes` | Complete classification/exclusion coverage for the reconstructed universe and original HOT/COL/HAR/CA securities. Praxair's separate terminal handling is now applied after an exact rollback rehearsal. The original six acquisitions are handled as five valued outcomes and Andeavor with an explicitly unknown payoff; exact Andeavor proration is not required to proceed. |

After application, freeze a corrected reference snapshot containing all inputs,
verify chronology and coverage, and rerun the production reference configuration.
The version-7 source-exclusion cache is superseded by version 8, which enforces
the reviewed inclusive price-history boundary and preserves unknown historical
sectors. Research-only additions remain excluded from the default production
cache. Neither an audit snapshot nor a refreshed cache certifies the reference.

Historical sectors are explicitly checked as a separate source by the universe
experiment. They also remain a substantive reference-quality issue for matched
sector evaluation, membership reconciliation and accounting's financial-sector
exclusion; source status alone must not conceal current-sector backfilling.

## Family-specific certification requirements

| Experiment | Additional prerequisite |
|---|---|
| Stress | None beyond the shared reference. First experiment to unlock. |
| Analyst | Verify fixed fiscal-period endpoints and split-adjustment semantics. Contributor history is optional; the registered consensus-only subset is allowed. |
| News | Verify exact-version timing and archive coverage; complete the 400-story annotation acceptance gate. |
| Accounting | Verify original filing availability, cumulative-flow handling and historical sector treatment. Incomplete observations may remain NaN. |
| Macro | BAA10Y retained by user decision on 2026-09-08. Actual vintages rebuilt/applied and independently checked. Finish usable paired coverage and freeze the corrected comparison; legacy calendar-backdated values do not qualify. |
| S&P 1500 | Corrected S&P 500 reference, reliable historical sectors, and separate local PostgreSQL/caches. Dedicated Docker PostgreSQL15.15 is initialized on localhost:55432 with separate caches; universe ingestion remains pending. |
| Combined | Supported individual 6M packs. The current runner requires at least two supported packs before executing the combined comparison. |

Migrations, existing research-table RLS, add-ticker database compatibility and the
paired-evaluation implementation are not the currently reported blockers.
Family-specific sources do not need to be finished before running stress.

Detailed evidence is in `signal_source_readiness.json` and
`signal_live_verification.md`. Some older narrative entries in those files retain
historical counts; their later dated receipts and structured updates supersede
those counts. This document summarizes dependencies, not certification.

## Latest gate work — 2026-09-08

- **Applied-reference paired coverage audited:** baseline/candidate controls and
  labels match for both stress and macro on 48,673 test rows across 107 folds.
  All stress features are observed on those rows; both macro inputs are observed
  on 45,512 rows across 102 folds. There are 51 test rows without sector and 2,300
  without size; those limitations remain visible to matched evaluation. June
  2023 labels fall outside the allowed realization window. These are diagnostic
  counts on an uncertified reference, not predictive-performance evidence.
- **Praxair scoped repair applied:** 2,223 native predecessor prices, separate
  membership and 252 native-successor valuations passed exact rollback rehearsal
  and transaction readback. All 4,194 raw Linde rows remain stored; only the
  1,971 post-boundary rows reach its frame. No sector intervals were applied.
  The four cash discrepancies are issuer-corroborated; the two remaining residuals
  are reproduced inside Yahoo's own derived adjusted series. Native return
  reconstruction passes without changing either tolerance. Historical sector
  gaps now remain unknown instead of borrowing today's classifications.
  The earlier 48,673-row offline paired result included staged Materials history
  and cannot substitute for the new applied-reference audit, even though the
  total test count happens to match. Independent verification preserves all 876
  unrelated securities and matches the refreshed version-8 production cache to
  its snapshot (730 frames, 2,394,143 prices). See `reference_gate_progress.md`.

- **Macro vintage correction applied:** 13,936 actual value versions archived,
  replayed and applied after an exact rollback rehearsal. Fourteen independent
  historical snapshots match. All 13,918 old rows are archived for recovery;
  1,869 noncolliding legacy rows remain stored but unverified. BAA10Y stays in
  the initial comparison; longer-history sourcing is conditional on evidence.
  This repairs ingestion, not the shared reference or paired evaluation gates.
  Diagnostic credit features start January 2014; both macro inputs are observed
  in 109 months through June 2023. October 2016–February 2017 remains missing
  under the existing staleness rule. These are not certified paired samples.

- **Wrong-security prices contained:** native-history comparison disproved all
  1,558 overlapping COL prices and all 1,044 overlapping HAR prices, including
  dates before retirement. Migration 021 and source-specific exclusions were
  applied after an exact rollback rehearsal. The loader now excludes 2,604
  previously loaded bars; all 4,340 raw database rows remain recoverable and
  unchanged. Retirement truncation alone had not fixed these histories.
- **Historical normalization advanced:** declared special cash explains 14
  coincident vendor adjustment steps under an exact-factor rule (relative
  tolerance 1e-5). Counting that cash once increases the candidate from 133 to
  **140 securities / 212,061 rows**. Seven remain excluded: ASIX/WRK interior
  gaps, CCEP/FERG/WIN unresolved coincident events, FHN stock distributions and
  NBR warrants. Candidates remain unapplied and uncertified.
- **Original histories recovered:** native and adjusted archives now contain
  7,969 observations for CA/COL/HAR/HOT. Separate recovery candidates preserve
  these identities. HOT's four 2014 special-cash adjustments are explained, but
  its September 21, 2016 cash dividend disagrees between sources; terminal
  consideration must also account for any merger-related dividend entitlement.
  None of these four terminal payoffs has been applied.

Receipts: `.research/verification/price-source-exclusions-applied.json`,
`.research/verification/original-security-price-parity.json`,
`.research/verification/historical-embedded-cash-review.json`,
`.research/normalized-historical-candidate-v3/report.json`, and
`.research/original-security-normalized-candidate-v2/report.json`.
Snapshot/cache verification: `.research/verification/frame-cache-v7-exclusions.json`.

**Macro decision accepted — 2026-09-08:** retain BAA10Y for the initial registered
comparison. Limit longer-history research to an availability check; defer extended
sourcing unless initial evidence warrants it. The decision and stopping policy are
recorded in `macro_research_decision.md`. This does not block stress or relax any
data/inference gate. No shared source gate has been certified and no registered
experiment has run.

## Scoped membership application — 2026-09-18

- Independently replayed the archived parent join/leave stream and checked all
  147 registered RIC/ISIN identities against exact security rows. Applied **143
  membership intervals** after an exact rollback rehearsal and post-commit
  readback. All **729 existing membership rows** and registered ticker attributes
  were preserved. The additions remain research-only; no prices, sectors,
  issuer CIKs or terminal outcomes were written.
- Four securities remain unapplied: CVH (2013-05-08), DTV (2015-07-28),
  Goodrich/GR (2012-07-30), and MRP (11 sessions in January–February 2025).
  Those sessions lie outside the archived lifecycle. The full interval stays
  blocked; endpoints were not guessed or clipped.
- Corrected the identity auditor: one result per requested RIC, blank-RIC issuer
  context excluded from security matching, no issuer-only mapping, explicit
  share classes and cross-RIC collisions. The 862-RIC audit has 768 identity
  candidates (216 without raw prices), 60 symbol/issuer discrepancies, eight
  issuer-only review cases, 12 early-history conflicts, 11 late-history conflicts,
  two colliding RIC mappings and one absent exact security (VAL). Candidates
  remain subject to dated continuity review. Raw-price diagnostics include rows
  already excluded by the loader; they do not invalidate those scoped repairs.
- Independent transformation replay passes for **140 histories / 212,061 rows**,
  maximum relative daily-return error 4.44e-16. The candidate still needs
  distribution/security completeness review before price application.
- Accounting now requires a known nonfinancial GICS sector; unknown sectors
  emit NaN. The full backend suite passed **484 tests** after reinstalling the
  existing PyTorch 2.12.0 version to restore its missing CPU library.

Receipts: `.research/verification/registered-membership-applied-2026-09-18-v1.json`,
`registered-membership-rehearsal-2026-09-18-v1.json`,
`security-mapping-exact-2026-09-18-v2.json`, and
`historical-normalization-parity-2026-09-18-v1.json` in the same directory.

All shared source gates remain **incomplete**. Next: certify and apply historical
price candidates with explicit exclusions, resolve the remaining MRP endpoint
and remaining security/sector conflicts, finish terminal classification/exclusion
coverage, then freeze and rerun the corrected production reference. No registered
feature experiment has run.

## Lifecycle repairs, local database and news sample — 2026-09-19

- Applied three independently reviewed index intervals and terminal classifications
  for Coventry, DIRECTV and Goodrich after exact rollback rehearsal. This brings
  the registered membership total to **146 of 147**, and the database to **875
  membership intervals / 11 terminal events**. Goodrich has documented cash
  consideration; Coventry and DIRECTV payoffs remain masked pending successor
  valuation. All 872 prior intervals and eight prior events were preserved; no
  candidate price rows were applied. The prior frozen snapshot is now stale.
- MRP remains unresolved: short-window events reproduce January 21–February 10,
  2025 membership, but seven direct dated constituent snapshots omit it. Explicit
  historical-chain requests returned contemporary-looking sets and do not resolve
  the contradiction. The [S&P notice](https://press.spglobal.com/2025-02-05-Millrose-Properties-Set-to-Join-S-P-SmallCap-600)
  documents February 10 SmallCap600 addition. Do not infer an SPX start date.
- A dedicated PostgreSQL15.15 container at localhost:55432 has the full research
  schema and separate frame/news caches. It contains no securities or production
  user data. Corrected reference and expanded-universe ingestion are still needed.
- The frozen news retrieval plan spans 60 era/sector/cohort windows. Thirteen
  imported, two were empty and 45 failed timestamp-order validation. Preserved
  rejected responses contain 2,384 rows, including 581 reversed timestamps (170
  cross-date). These are quarantined, not retimed or accepted for features.
- Blinded human worksheets now contain **400 distinct stories**, split **200/200**
  between development and locked validation, spanning 11 sectors, three eras and
  58 available strata. Gold labels remain blank; no scorer predictions were
  generated. Raw quarantined text is included for text review only; sectors and
  mappings remain provisional sampling metadata. The package is under
  `.news_cache/annotation-review/review-2026-09-19-v1/`. Annotation does not certify
  source timing or archive coverage.

Receipts: `membership-lifecycle-applied-2026-09-18-v1.json`,
`local-research-database-2026-09-18-v1.json`,
`mrp-membership-events-2026-09-19-v1.json`, and
`news-review-audit-2026-09-19-v1.json` under `.research/verification/`.
Shared gates remain incomplete; no registered experiment has run.

### Additional source checks — 2026-09-19

The analyst probe archived 41 rows with separate CalcDate, update Date, absolute
FY label and vendor PeriodEndDate. All five rolling AAPL snapshots match their
separately queried absolute targets. In 24 rows, the update date differs from the
snapshot date; backdating the snapshot would misstate availability. Three split
windows are compatible with a common adjusted basis. Standardized period-end
semantics, split/revision conventions and the 0.01 EPS eligibility threshold still
need resolution before fixed-estimate backfill. See
`.research/verification/fixed-estimate-semantics-2026-09-19-v1.json`.

CA and Harman cash outcomes are source-verified and staged against their native
recovery candidates. CA pays $44.50 and Harman $112.00 per share, supported by their
[CA completion filing](https://www.sec.gov/Archives/edgar/data/356028/000119312518317918/d631849d8k.htm)
and [Harman completion filing](https://www.sec.gov/Archives/edgar/data/800459/000119312517078968/d184457d8k.htm).
CA's filing documents a halt before November 5 opening, explaining the missing
November 5 price despite a November 6 administrative suspension notice. Each
candidate passes 21/63/126/252 terminal-crossing label checks. These outcomes were subsequently applied with 4,035 recovered native rows after
cash-return corroboration, exact rollback rehearsal and postcommit readback.
The 2,896 replaced legacy rows are preserved in the compressed preimage archive.
The database now contains 13 terminal events. See
`.research/verification/original-cash-applied-2026-09-19-v1.json`.
The production cache and reference snapshot still need rebuilding after the
remaining shared repairs; no global gate was promoted.
See `.research/verification/original-cash-candidate-2026-09-19-v1.json`.

The news sampling and ingestion tests passed 32 checks, including duplicate-story
rejection, balanced locked splits and raw archival without partial ingestion on
timestamp failure. New scripts pass Ruff; existing unrelated style findings in
news_lseg remain. No shared gate was promoted and no registered model was fitted.

### Historical cash-return audit — 2026-09-19

The reproducible, hash-bound audit covers all 140 candidates / 212,061 price rows
and compares 211,864 ordinary daily returns. Its results are 102 histories with
cash-return agreement, 34 requiring capital-event or missing-session review, and
four with cash disagreements: BNI, KTB and both former Fox share classes.
Sixteen extra source rows have stale observation dates and no native price on
their calculation date. They are disclosed and excluded, never retimed. Seven
histories have 16 missing native sessions; 11 gap-crossing comparisons are skipped.
Agreement between two LSEG retrieval paths does not certify distribution completeness.

Fresh Workspace queries reproduce the cash inventories. The
[Kontoor issuer release](https://www.kontoorbrands.com/news-media/press-release/35/kontoor-brands-announces-third-quarter-2019-results)
confirms the disputed $0.56 dividend. The
[News Corporation filing](https://www.sec.gov/Archives/edgar/data/1308161/000119312510106827/d10q.htm)
confirms $0.075 on both classes for the March 2010 record date. Both sources are
archived by content hash. These findings support retaining documented cash while
the conflicting total-return records are investigated; the remaining Fox dates
are not thereby certified. BNI's conditional dividend has due-bill terms in the
[exchange notice](https://www.cmegroup.com/tools-information/lookups/advisories/clearing/files/Chadv10-62.pdf)
and must be reconciled with merger proceeds. Its direct archival download timed
out; no ordinary ex-date or dividend override was applied.

Receipts under `.research/verification/`:
`historical-total-return-audit-2026-09-19-v1.json`,
`cash-disagreement-lseg-archive-2026-09-19-v1.json`, and
`cash-disagreement-issuer-sources-2026-09-19-v1.json`.
The audit, price-component, coverage, normalization and terminal tests passed
**41 checks**. New audit files pass Ruff. No historical candidate prices were
applied by this audit, no source gate was promoted, and no feature experiment ran.

### Capital-event corroboration — 2026-09-19

The full candidate action audit separates 46 observed capital-adjustment dates:
six declared split returns corroborate within the unchanged 1e-6 tolerance,
14 special-cash dates retain convention review, and 26 distribution dates retain
event-specific review. Sixteen action records have no price step on either
reported event date. This catches declarations ordinary return parity cannot
certify. Missing steps are not silently interpreted as no action.

Issuer filings archived by content hash corroborate split ratios for Peabody,
Dean Foods, HollyFrontier, Lorillard, ODP and SUPERVALU. These six event components
are resolved; full historical security continuity and distribution completeness
remain separate requirements. In particular, ODP's split coincided with a holding
company reorganization. See `docs/historical_split_component_reviews.json` and
`.research/verification/historical-split-corroboration-2026-09-19-v1.json`.

Special-cash diagnostics distinguish return conventions without choosing whichever
formula fits: twelve dates agree numerically with a special-cash price adjustment
plus ordinary cash reinvestment; AVP and MWV resemble counting special cash through
both paths. These observations do not establish authoritative TR field semantics.
Stored price normalization and tolerances are unchanged.

The event audit and related normalization/return tests passed **34 checks**.
New files pass Ruff. No database writes or gate promotions resulted.
