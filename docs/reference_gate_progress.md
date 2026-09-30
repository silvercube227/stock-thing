# Reference gates and paired coverage — 2026-09-08

No gate was promoted and no model was fit. Source/identity completeness remains
the blocker; a numerically valid panel is not a certified historical reference.

## Actual paired-sample audit

The new immutable audit snapshot includes 878 security frames and the corrected
macro vintages. Of these, 713 have S&P 500 membership, 165 have none, and 255
have no loaded prices. The counts overlap. All 148 separately registered research
securities are included in the snapshot, but registration alone does not attach
membership or prices.

`select_walk_forward_samples` is shared by the real runner and the pre-fit audit.
It preserves embargoes, the last 60 **labeled** training months for 6M, training
labels realized strictly before the test date, next-session entry and selection
labels ending strictly before 2024. No new feature-completeness filter was added.

Baseline and macro panels matched exactly on all original columns, including
security/date identity, features, sectors, size, outcomes and execution dates.
On the current uncorrected reference:

- 107 eligible folds, July 2014–May 2023; 48,686 paired test rows.
- Both macro features are observed on 45,549 test rows across 102 folds.
- Five credit-gap months remain in the registered runner with NaN credit values;
  observed-feature months form contiguous segments of 27 and 75 months.
- June 2023 has no eligible tests because its 126-session labels finish in 2024.
- July 2014 has observable credit test features but no credit features in its
  eligible training history. This is disclosed, not silently dropped.
- 2,339 test rows lack a usable size control. A model can handle missing inputs,
  but size-neutral evaluation coverage must be reported separately.

The observed-feature calendar meets the structural minimum for both 6- and
12-month blocks (17 and 8.5 effective blocks). This does **not** establish finite
ICs, statistical power, significance or source validity. The original 109-month
feature-only count was not an actual fold count.

Artifacts:

- `.research/reference-gate-audit-2026-09-08-v2/`
- `.research/paired-coverage-macro-2026-09-08-v1/report.json`
- The corresponding hashed `panel.pkl.gz` retains the audited panel for review.

## Praxair/Linde reference candidate

The [Praxair completion filing](https://www.sec.gov/Archives/edgar/data/884905/000119312518313154/d646523d8k.htm)
documents the October 31, 2018 exchange into one Linde plc share and Praxair's
last trading session on October 30. Its full archived document is hash-checked.

The offline reference candidate now:

- Replays 2,223 native Praxair price rows and all 35 parsed cash-dividend dates;
  checks the complete NYSE session calendar against the prior normalized artifact.
- Reconstructs the separate predecessor membership interval from the original
  parent-index archive, with a strict October 31 boundary and no bridged gaps.
- Removes those 2,223 predecessor bars from the candidate Linde frame so they
  cannot become Linde's own price-feature history. No raw database rows are deleted.
- Preserves separate issuer filings; stages the reviewed Materials intervals
  without making the wider historical sector reconstruction certified.
- Corroborates the final Praxair close (164.50) and first successor ownership
  close (165.47), then computes 252 exact-date successor valuations.
- Restores 21/63/126/252 terminal-crossing labels for 1M/3M/6M/1Y within the
  checked final-history window. These are candidate labels, not promoted outcomes.

Artifacts: `.research/reference-praxair-candidate-2026-09-08-v1/` and its
`praxair-review.json`. Original database contents and production caches are
unchanged. The candidate retains incomplete source statuses and cannot pass the
research runner's gates.

Independent verification confirmed that all 876 unrelated frames, macro values,
issuer attributes and gate statuses are unchanged. All 2,223 nominal closes
match the independently stored predecessor prices (maximum relative error
5.47e-8). Four cash discrepancies are 0.7875 versus 0.788 in 2017; the issuer's
[quarterly financial summary](https://assets.linde.com/-/media/global/corporate/corporate/documents/investors/archive-praxair-quarterly-earnings/praxair-4q17-earnings-release-tables.pdf)
reports 0.7875 for each quarter. The exact native amount is retained, not rounded
to match the legacy feed. This resolves those four cash amounts, not all source
completeness. Six daily adjusted-return ratios exceeded the original 1e-6 parity
tolerance: four coincide with these cash discrepancies, and two small residuals
on 2010-06-07 and 2010-09-02 remain disclosed (about 1.2e-6). No tolerance was
relaxed. Receipts: `.research/verification/praxair-reference-verification-v1.json`
and `docs/praxair_dividend_review.json`. The issuer PDF is archived by content
hash in `.research/verification/praxair-dividend-source-archived.json`.

The paired audit was rerun from scratch on the separated candidate. All original
baseline/candidate columns still match exactly. It has **48,673 paired test rows**
across the same 107 eligible folds; **45,512 rows across 102 folds** have both
macro features, and 2,300 test rows lack size. Separating a predecessor does not
guarantee more eligible rows: Linde can no longer borrow Praxair's price-feature
history. No complete-macro filter was introduced. The candidate has 714 frames
with membership, 164 without membership and 254 without loaded prices.
Receipt: `.research/paired-coverage-praxair-macro-2026-09-08-v1/report.json`.

Those were the remaining prerequisites at the offline-candidate stage. The
subsequent scoped application below supersedes that application status, but not
the incomplete wider reference gates.

## Scoped database repair applied (subsequent work)

The bounded source review now covers all 2,223 predecessor sessions, 35 cash
dates and its archived capital-action inventory. The single archived buyback
is not a holder distribution. The two residual adjusted-return discrepancies
also occur against Yahoo's **own** nominal closes and recorded dividends:
1.1038e-6 on 2010-06-07 and 1.2431e-6 on 2010-09-02. Native reconstruction passes
its formula check. The source-reconstructed series is retained; no tolerance was
widened and no share ratio was inferred from a price ratio. Cross-feed agreement
is corroboration, not a guarantee against every possible source omission.

Successor verification covers all 253 NYSE sessions from 2018-10-31 through
2019-11-01, all four cash dates and both archived capital events (the documented
1:1 exchange and an issuer buyback). All nominal/cash components agree, with no
unresolved adjusted-series residual. All 252 terminal values now use this native
successor reconstruction, rather than the legacy Yahoo adjusted series. Maximum
relative difference from the old candidate is 5.88046e-7.

Migration 022 and `scripts/apply_praxair_reference.py` were applied only after an
exact rollback rehearsal. The repair adds 2,223 PX prices, splits membership at
2018-10-31 and stores the documented acquisition. Stored labels restore
21/63/126/252 terminal-crossing observations for 1M/3M/6M/1Y. Existing
NUMERIC(18,6) price storage is checked at its explicit half-micro-unit quantization;
that is separate from source-parity tolerances.

All 4,194 raw LIN prices are retained, while the inclusive reviewed boundary
allows only 1,971 successor bars into its model frame. Distinct issuer filings,
accounting facts and all sector records are preserved. No unverified Materials
intervals were applied. A successfully loaded but empty/gapped sector history now
produces unknown labels; it no longer falls back to today's sector or industry.
Unexpected sector-query errors propagate rather than enabling that fallback.
The missing-table legacy fallback remains explicit. Frame-cache version 8 rejects
older caches that lack these semantics.

Receipts and recovery evidence:

- `.research/verification/praxair-price-components-v1.json`
- `.research/praxair-successor-basis-v1/report.json`
- `.research/verification/praxair-reference-rehearsal-v2.json`
- `.research/verification/praxair-reference-applied-v1.json`
- `.research/verification/praxair-reference-preimage.json.gz`

The preimage preserves the exact pre-write rows for both affected identities;
its SHA256 is `8217b44e468ecd6718b88dce197ee116ddd223d89cca410e4552950e522cd98a`.
No raw data was deleted. The repair does not certify broader historical-sector,
security-identity, price/share or terminal-outcome coverage.

Next reference work is dated sector reconciliation (including the PX interval),
the other 147 registered historical securities and their reviewed/masked price
histories, VAL/predecessor/share-class conflicts, and complete terminal
classification/exclusion coverage. Stress is still the first experiment to unlock
after the shared reference is certified; macro retains BAA10Y under the existing
conditional sourcing decision.

Accounting-specific follow-up: `accounting_features` currently excludes named
financial sectors but does not yet mask an unknown sector. That source gate must
remain incomplete until unknown-sector treatment is covered by its acceptance
checks; the sector-loader repair alone does not certify the accounting pack.

## Post-commit verification and paired coverage

Independent snapshot verification passes for all 876 unrelated securities and
all macro values. The 156 previously empty sector histories are now explicitly
unknown. The version-8 production cache exactly matches its 730-frame snapshot
(2,394,143 loaded prices); the research snapshot has 878 frames. The repaired PX
frame remains research-only and does not enter the default production cache.
Receipt: `.research/verification/praxair-reference-postcommit-v1.json`.

Research input SHA256:
`1d519da136ca6945af02c2d2c09e1a97e86849249536481cd6fd58f517bf223b`.
Production input SHA256:
`9caefc7e02616868443a5a68395f160568a861e277e3a70e865ce32876747ebd`.

The applied-reference macro audit matches all baseline/candidate controls and
labels on **48,673 test rows / 107 eligible folds**. Both macro inputs are observed
on **45,512 rows / 102 folds**; 2,300 test rows lack size and **51 lack sector**.
The full assembled panel has 93 unknown-sector rows, all PX. The existing runner
excludes undefined sector-grade training labels; unknown-sector test rows can
still receive predictions but cannot contribute to within-sector scoring. The
test-row count therefore matches the old offline candidate by coincidence, not
because its staged Materials history was applied. No complete-macro filter or
new sample rule was introduced.

Receipt: `.research/paired-coverage-praxair-applied-macro-2026-09-08-v1/report.json`.
The 6-/12-month block calendars remain structurally sufficient; no IC, power or
performance claim follows from that result. July 2014 still has no observed
credit features in eligible training history; the five credit-gap months remain.

The stress audit independently rebuilt both panels and also matched every
baseline/candidate control and label: **48,673 test rows / 107 folds**, with all
stress inputs observed on every test row. It has the same 51 unknown-sector and
2,300 missing-size test rows. Both 6- and 12-month block calendars meet the
structural minimum, without claiming finite ICs or statistical power.
Receipt: `.research/paired-coverage-praxair-applied-stress-2026-09-08-v1/report.json`.

All **463 backend tests pass**; the new application, component audits, post-commit
verifier and their new test files pass targeted lint. No global source gate was
promoted and no registered model experiment was run.

## Decisions

No statistical thresholds or sample rules changed. Restricting the primary macro
comparison to complete-macro months would be a new registration decision; the
current implementation retains the registered missing-value handling and reports
observed-feature coverage separately. No paid data search or vendor coordination
has been initiated. Remaining source checks are factual work, not reasons to
lower the gates.

## Historical membership integrated — 2026-09-18

143 of the 147 registered securities now have exact-RIC membership applied.
`apply_registered_membership` independently replays the archived event stream,
checks the registration ISIN against the exact source security row, rejects
conflicting known issuer IDs, and leaves missing issuer mappings missing. It
checks actual trading sessions against the archived first-trade/retirement bounds.
CVH, DTV, GR and MRP remain blocked in full pending endpoint review.

The transaction preserved all 729 prior membership rows, including the separate
Praxair interval, and all affected ticker attributes. It passed exact rollback,
transaction readback and post-commit readback. No prices, sectors or terminal
records were written. The recovery preimage is
`.research/verification/registered-membership-preimage-2026-09-18-v1.json.gz`,
SHA256 of uncompressed bytes
`d363a4385316627c5cd11000f4e6a243142b71469745eb5a4a8d317c6ebd1821`.

The exact executed application source is retained as
`.research/verification/registered-membership-implementation-79ed02c7dcd0fa3962848b05bc057217fe8fd3b706eb7dc9758b3e346e96db17.py`.
The working script was subsequently formatted; execution receipts bind the
archived version, while future applications require their own matching rehearsal.

The older CIK-first mapping counts are superseded by the 862 exact-RIC results
in `security-mapping-exact-2026-09-18-v2.json`. This audit no longer calls a
unique issuer an approved security mapping or treats blank-RIC issuer rows as
separate securities. For VAL, the exact row identifies Valaris; the context row
cannot redirect it to another issuer. VAL still lacks a separately verified
database security and history. Sixty same-symbol issuer discrepancies need source
review rather than automatic re-registration.

The accounting unknown-sector implementation issue above is fixed and tested.
Source availability and sector certification remain incomplete. No model was fit.

The fresh audit snapshot
`.research/reference-membership-applied-research-2026-09-18-v1`
contains 878 frames and exactly 872 membership intervals: all 729 preexisting
intervals plus the 143 scoped additions. Independent verification against the
recovery preimage and archived membership confirms that all four blocked frames
remain without membership, and all 147 registered frames still lack loaded
prices, sector intervals and terminal events. The normalized-price candidate
remains separate. Shared gate statuses are unchanged; the production cache was
not refreshed and no model was fit.

Snapshot input SHA256:
`fae84cce2085d92d5d571d491b2a6054ad308f8c94b02d0833e0a86dc0f0426a`.
Receipt: `.research/verification/registered-membership-snapshot-verification-2026-09-18-v1.json`.
The full suite passed 484 tests; 35 targeted tests passed after final formatting.
The new membership and identity scripts/tests pass Ruff, and `git diff --check`
passes. Existing lint findings in unrelated parts of long-horizon features/tests
were not changed by this work.

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

The hash-bound 140-history audit compared 211,864 daily returns: 102 histories
match ordinary cash returns, 34 retain capital/gap review, and four retain cash
disagreements. Sixteen stale source rows are excluded without retiming; 16 missing
native sessions remain visible. Issuer evidence corroborates the disputed KTB
cash and the March 2010 dividend on both former Fox classes. No price overrides
or global gate promotions followed. See docs/experiment_gate_status.md and
.research/verification/historical-total-return-audit-2026-09-19-v1.json.
Targeted validation passed 41 tests; new audit files pass Ruff.
