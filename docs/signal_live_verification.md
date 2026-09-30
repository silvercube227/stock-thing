# Live verification — 2026-09-08

## Separate Praxair registration and source backfill

Registered `LSEG:PX.N^J18` as ticker 947, CIK 0000884905, inactive and research-only.
Live readback confirms those flags, 332 accounting facts and 40 fundamentals
filing rows, with zero price rows applied. Existing Linde identity and production
loader defaults remain unchanged. No schema or RLS changes were required.

Archived 2,223 raw prices and 2,223 matching capital-adjusted observations through
October 30, 2018. Dates align exactly, capital-adjusted and nominal closes agree,
and every expected exchange session from 2010 is present. Parsed 35 USD cash
dividends without source conflicts. The normalized candidate is 63,962 bytes
compressed at `.research/praxair-normalized-candidate/`. The corporate-action
query returned one buyback record and no reported split event; successful
retrieval alone is not a completeness certification. Share-action fields remain
unknown in the candidate.

Receipts: `.research/verification/praxair-registration-readback.json`,
`praxair-raw-history.json`, `praxair-adjusted-history.json`,
`praxair-dividends.json`, `praxair-corporate-actions.json` in the same directory,
and `.research/backfill-verified/accounting-947.json`. Remaining work includes
action/filing-availability review, terminal valuation and applying the separate
research history and membership after verification. No price or sector candidate
has been promoted.

## Praxair predecessor/successor candidate

Archived the SEC completion filing by SHA256 and independently probed native
Praxair/Linde prices across October 31, 2018. `PX.N^J18` ends October 30 at $164.50;
`LIN.OQ` returns the same three preceding Praxair bars and then the successor's
October 31 close of $165.47. This directly confirms vendor-linked predecessor
history. The filing documents one Linde ordinary share per Praxair share.

Implemented an explicit dated interval split that preserves membership episodes,
refuses overlaps and requires distinct predecessor/successor identities. The
research candidate separates the parent LIN interval into PX before October 31
and LIN thereafter, raising the parent interval count from 876 to 877. The same
split makes Materials reproduce both archived 2018 snapshots (24 securities each).
It does not assert that all later Linde identity changes are resolved.

Evidence: `.research/verification/praxair-source-archived.json`,
`praxair-price-transition-probe.json`, `praxair-separated-interval-candidate.json`
and `praxair-separated-interval-replay.json` in the same directory. No database
membership or price history has been changed. Separate predecessor registration,
price/filing backfill and terminal-value verification remain required. This
additional predecessor was hidden by the parent source's linked LIN history and
is not closure evidence for the earlier 147-security backfill. Full backend
suite: **409 passed**, with 123 existing correlation warnings.

## Host Hotels security-level resolution

Resolved `HST.N` and `HST.OQ` using their exact common-stock RIC rows, which
share ISIN `US44107P1049`, CUSIP `44107P104` and issuer CIK `0000314733`.
Additional source rows have blank security identifiers and are not mapped.
The issuer's [listing-transfer announcement](https://ir.hosthotels.com/news-releases/news-release-details/host-hotels-resorts-announces-transfer-nasdaq-stock-market)
documents the November 2, 2020 transfer. Reviewed alias duplicates within a
constituent snapshot are now counted once and disclosed in the audit. Repeated
exact RIC rows still fail; different share classes remain separate. The operation
does not deduplicate event streams or combine price/filing histories.

Real Estate now reproduces both 2018 checkpoints from the 2023 anchor. Nine
sectors match, Materials differs, and Consumer Discretionary retains the Booking
Holdings event conflict. Evidence:
`.research/verification/sector-host-alias-audit.json`. Full backend suite:
**406 passed**, with 123 existing correlation warnings.

Praxair's [completion filing](https://www.sec.gov/Archives/edgar/data/884905/000119312518313154/d646523d8k.htm)
documents one Linde share per Praxair share on October 31, 2018. The company's
pre-close notice identifies October 30 as Praxair's final trading date. These
terms explain a successor relationship, not an alias: the parent membership's
2010-onward LIN interval still spans distinct securities. The review at
`.research/verification/praxair-successor-review.json` records required separate
price/filing backfill, membership identity and terminal-label work. No correction
has been applied from unarchived source documents or unverified price bases.

## Reviewed venue alias and remaining security conflicts

The LSEG identity probe returns matching ISIN `US05722G1004`, CUSIP `05722G100`
and issuer CIK for `BKR.N` and `BKR.OQ`. Nasdaq's
[transfer notice](https://www.nasdaqtrader.com/TraderNews.aspx?id=dtn2021-48)
documents December 7, 2021 as the listing transfer date. A scoped review now
reconciles these index-source aliases to the identified share class; it does not
merge prices or claim either RIC was tradable outside its historical validity.
Energy consequently matches both 2018 checkpoints from the 2023 anchor, bringing
the matched sector count to eight. Two sectors still differ and one fails event
consistency. Evidence: `docs/sector_security_alias_reviews.json` and
`.research/verification/sector-selection-alias-audit.json`.

The new identity evidence confirms that `PX.N^J18` and `LIN.OQ` have different
ISINs and CIKs. Their parent/sector mismatch must not be erased as a rename.
Host Hotels returns matching fully identified stock rows but also an additional
issuer row with no security identifiers; its duplicate remains unresolved.
Booking Holdings is present in both March 19 and March 22, 2021 sector snapshots
despite the March 22 leaver record. This conflicting event remains unmodified.
Evidence: `.research/verification/sector-conflict-probe.json`.
The full backend suite passes **404 tests** with 123 existing correlation warnings;
the alias tests explicitly reject merging different securities of the same issuer.

## Historical anchors and sector continuity

Anchoring at October 1, 2018 instead of the latest endpoint allows all eleven
sector histories to reconstruct and reproduce their August 31, 2018 snapshots.
Industrials required one scoped review: Honeywell's September 1, 2016 same-day
join/leave pair preserves existing membership. Independently requested August 31
and September 1 constituent sets are identical (68 members, including Honeywell).
`docs/sector_event_reviews.json` pins both source artifacts by hash. The interval
builder permits reviewed continuity only for an exact event pair and an already
active security; unreviewed pairs and attempts to create absent members fail.

Evidence: `.research/verification/sector-2018-anchor-reviewed.json`. This is not
full coverage certification. Comparing its sector union with parent membership
on August 31, 2018 finds four missing parent RICs and five extra sector RICs.
Three pairs resemble venue aliases; Praxair/Linde and an extra Host Hotels
identifier require explicit security-identity resolution. No ticker-stem merge
has been applied. Earlier Real Estate overlap also requires taxonomy-date review.
The partition differences are saved in
`.research/verification/sector-parent-partition-audit.json`.

Single-index 2011/2014 snapshot probes remain unavailable; batch requests returned
empty rows even for dates available individually and cannot establish absence.
All eleven individual snapshots at the fixed September 29, 2023 selection anchor
were retrieved. Reconstruction from that anchor matches both 2018 checkpoints for
seven sectors. Energy, Materials and Real Estate differ; Consumer Discretionary
fails on the March 22, 2021 Booking Holdings leaver event. Evidence:
`.research/verification/sector-selection-anchor-probe.json` and
`sector-selection-anchor-audit.json`. No historical sector rows were applied.
The full backend suite passes **402 tests** with 123 existing correlation warnings.

## Andeavor classification and unresolved payoff

Applied ANDV's documented October 1, 2018 acquisition with
`consideration_type=unresolved_election`, null cash and null horizon values.
Verified means the termination classification is documented; the payoff remains
unknown. The September 28 final nominal close matches LSEG at $153.50. Live
transactional label checks agree with the full-history rollback rehearsal:
63 daily 3M, 126 daily 6M and 252 daily 1Y labels crossing the acquisition remain
unavailable. There is also one rejected post-terminal entry per horizon. The
receipt includes the year breakdown and 1M diagnostics. No approximate allocation
or optimal holder election was substituted for realized proceeds.

Evidence: `.research/verification/andv-unresolved-rehearsal.json` and
`.research/verification/andv-unresolved-applied.json`. The pre-application review
is archived by hash in `.research/terminal_sources/`. The first rehearsal lost
its idle database connection during calendar calculations; the successful rerun
kept the transaction responsive and rolled back before the separate application.

Fixed the post-terminal entry guard to cover verified termination classifications
with unresolved consideration, not only valued cash/successor events. A missing
data gap remains insufficient proof of termination. Cache version is now 6;
the full backend suite passes **399 tests** with 123 existing correlation warnings.
The original six-acquisition list is handled as five valued outcomes and one
explicitly unknown outcome. Expanded-universe terminal coverage remains open.

Refreshed the existing compressed production-universe frame cache from a
repeatable-read database transaction: 730 frames, version 6, 110,037,032 bytes.
Replay verified the committed ANDV classification and null payoff. Receipt:
`.research/verification/frame-cache-v6-refresh.json`. This replaces the existing
cache without another full snapshot; it does not certify the historical reference.

## Sector overlap follow-up

Three smaller event-window queries reproduce the long-window records exactly;
three current sector chains match their September 4 endpoints. The contradictions
therefore persist under this probe rather than being repaired by subdivision.
Evidence: `.research/verification/sector-overlap-probe.json` and
`sector-overlap-audit.json` in the same directory. No fabricated event or alias
mapping has been applied. This source-quality gate remains incomplete.

## Historical-sector source feasibility

Sector-index joiner/leaver fields and dated constituent requests provide a usable
new probe path. The 2018 pilot for Technology (`.SPLRCT`) and Communication
Services (`.SPLRCL`) reproduces the August 31 membership exactly when reconstructed
from October 1: 73 and three securities respectively. The event records move both
Alphabet share classes and Meta between the indices on September 24, consistent
with [S&P's account of the reclassification](https://www.indexologyblog.com/2019/09/24/happy-birthday-to-the-communication-services-sector/).
Evidence: `.research/probes/sector-index-history-live-2026-09-08.json` and
`sector-index-history-replay-2026-09-08.json` in the same directory.

This is not yet a certified history. Broader requests expose unavailable early
2010 snapshots, RIC aliases across venue changes, and inconsistencies between
long-window events and dated endpoint sets. For example, Energy's archived events
omit the Hess departure needed to reconcile its 2018 and 2026 sets. These require
source or identity resolution; no missing events have been invented. A proposed
Real Estate identifier `.SPLRCREC` was rejected because LSEG identifies it as the
REIT **industry group**, not the full sector. The archive script now requests
`.SPLRCR` for verification. Ordinary `TR.CommonName` names the index provider;
`TR.InstrumentDescription` supplies the index identity. The real-time display-name
request was unavailable under the session's entitlements.

New source archives are resumable, hash-checked and gzip compressed. All requests
are sequential and read-only; there are no new tables or RLS changes. Relevant
source tests: 29 passed. Historical-sector readiness remains incomplete.

The completed corrected archive verifies `.SPLRCR` as S&P 500 Real Estate
(Sector). Full-window reconstruction results: six sector histories fail event
consistency, three reconstruct but disagree with the 2018 snapshots, and two
(Financials and Utilities) match both 2018 snapshots. The latter still lack
independent early-window coverage proof. The 2026 endpoint match is an anchor,
not an independent validation. Reports:
`.research/verification/sector-history-archive-v2.json` and
`.research/verification/sector-history-audit-v2.json`. Compressed source archives
occupy 496 KiB on disk, including the retained rejected-identifier probe.

## Compressed historical-price candidate

Built `.research/normalized-historical-candidate-v2/` from the hashed raw and
capital-adjusted archives and the reviewed cash-dividend table. Its gzip artifact
is 4.7 MiB and contains 192,030 rows across 133 securities. Nominal closes and
cash amounts match source observations in every row; 191,897 adjacent adjusted
price ratios match the source capital-price/cash-adjustment convention. The
receipt is `source-parity.json`; this verifies transformation, not action-source
completeness or observed total returns. No candidate prices were written to the
database and no reference snapshot was certified.

Fourteen securities remain excluded: ten have coincident cash and vendor capital
adjustments requiring distribution review, two (ASIX and WRK) have interior
missing-price records, and FHN/NBR have unresolved dividend-source records.
SIAL's unpriced 2015-11-18 terminal boundary is explicitly recorded as unavailable;
its preceding 1,480 valid observations are retained without filling the boundary.
The boundary handler rejects interior gaps, asymmetric source availability and
cash events on unavailable dates. Share ratios remain unknown in this candidate.

Full backend suite: **397 passed**, with 123 existing constant-input correlation
warnings. Measured storage: `.research` 893 MiB and `.frame_cache` 109 MiB.
Both candidate versions are compressed; no additional full snapshot was created.
The Andeavor tax-basis PDF download returned HTTP 403 and supplied no verified
proration evidence. Its terminal outcome remains unresolved.

## Vendor adjustment reconciliation

Retrieved capital-action-adjusted histories for all 147 new RICs using explicit
`CCH,CRE,RPO,RTS` adjustments: 233,108 observations. All dates align with their
raw histories. Reconciliation finds 57 price-adjustment steps across 36 securities;
these are price ratios, never inferred share-count ratios. FHN has steps on all
four 2010 missing-cash dates. NBR has no step for the 2021 warrant distribution,
so its vendor-adjusted series does not establish complete total returns. COV's
purported 100-for-1 terms record has no corresponding price-adjustment step and
must not be applied automatically.

Evidence: `.research/verification/missing-prices-adjusted.json` and
`.research/verification/vendor-adjustment-reconciliation.json`. These archives
add approximately 30 MB, not another full snapshot. Full backend suite: 389 passed.

## Dividend normalization review

The cash-dividend parser now handles numeric-format duplicates, retains additive
ordinary/special distributions, rejects conflicting same-type events and masks
dates with unknown amounts/currencies. Normalization rejects negative/nonfinite
cash dividends instead of propagating corrupt adjusted prices. The archive review
finds 145 histories without parsing conflicts and two with five unresolved events:
FHN (four 2010 dates) and NBR (2021-06-03). This is parsing evidence, not proof of
dividend completeness. `.research/verification/historical-dividend-review.json`
records the details without creating another full dataset copy.

Nabors' missing cash amount corresponds to a
[warrant distribution](https://www.nabors.com/nabors-announces-distribution-warrants-purch/),
with its June 3 effective date corroborated by the
[exchange notice](https://www.miaxglobal.com/alerts/2021/06/02/miax-exchange-group-options-markets-corporate-action-alert-nabors-industries-ltd).
It must not become a zero cash dividend; valuation remains unresolved.

## New-history session and dividend coverage

Exact native-RIC membership/session audit finds 131 of 147 raw histories cover
every reconstructed membership session. Sixteen histories miss 30 sessions,
mostly around terminal dates; MRP accounts for 11 sessions before the archived
price start. Reentries and exclusive removal boundaries are preserved. Missing
prices and membership semantics remain review items, not filled observations.
Evidence: `.research/verification/historical-price-coverage.json`.

The dividend probe succeeded and the sequential 147-RIC archive returned 2,131
rows, including empty source placeholders. Queries explicitly filter on ex-date
(`DateType=XD`), since payment-date filtering could omit the needed event. Amounts,
currency, dividend type and date validity are retained for adjustment review.
Retrieval does not certify absence of dividends or action completeness. Evidence:
`.research/verification/registered-historical-dividends.json`.
Full backend suite: 382 passed. Audit artifacts remain compact; no extra snapshot
or cache copy was created during these checks.

## Storage and historical source follow-up

Per the user's storage constraint, new snapshots and frame caches use gzip
compression at their existing paths, and loaders detect both gzip and legacy
plain files. Existing snapshots/caches were compressed one at a time, verifying
the original and decompressed SHA256 before atomic replacement. Manifests and
logical data hashes remain unchanged; no source evidence was deleted. Older
archived code that reads pickle files directly requires transport decompression.
Conversion freed 3,579,506,019 bytes; `.research` is now 868 MiB and `.frame_cache`
116 MiB. Evidence: `.research/verification/storage-compression.json`.

The newly registered historical securities' SEC backfill completed for 124:
32,221 accounting facts and 3,290 filings. Eighteen lack an unambiguous source CIK;
five have unavailable SEC companyfacts. The 147-security LSEG action pull returned
241 records without failed batches. This confirms retrieval, not completeness,
effective-date semantics or dividend coverage. Evidence:
`.research/verification/historical-accounting-backfill.json` and
`.research/verification/registered-historical-actions.json`.

## Historical pricing service and missing-security backfill

LSEG's historical-pricing service with `adjustments='unadjusted'` supplies bars
where the TR reference-data price fields returned null/error. AET, TWX and BMS
final nominal closes agree with stored prices. All three successor deals are now
applied, with 252 horizon dates each and exact stored-label replay in the same
transaction. Total documented events: six including EVHC. Andeavor remains
unapplied because the disclosed final non-election proration is approximate.
Evidence: `.research/verification/historical-pricing-service.json` and
`.research/verification/successor-events-three-applied.json`.

The same probed historical-pricing path retrieved all 147 missing securities:
233,108 raw price observations, with hash-addressed per-request archives under
`.research/missing-price-archive-live/`. These are venue-level bars; volumes and
intraday extrema can differ from consolidated Yahoo observations. Actions,
dividends, lifecycle coverage and sectors are not yet certified. The initial
sandbox-denied attempt is preserved separately and is not a source-content failure.

147 separate inactive research-only ticker rows are now registered using exact
RIC namespace symbols. No current ticker was reused, no share classes merged,
and no production membership changed. Raw ISIN/issuer metadata is preserved in
the registration artifact without inventing historical identifier intervals.
The VAL.N^H20 conflict remains excluded. Evidence:
`.research/verification/missing-prices-backfill-live.json` and
`.research/verification/historical-registration-applied.json`.

The last frame cache/snapshot predates the three newly applied terminal deals;
refresh it before evaluation. No experiment is authorized by these partial gates.
RLS policies are unchanged.

## Successor-label application

SCANA and Express Scripts successor events are applied. Each has 252 exact
horizon-date values, with merger cash held without reinvestment and stock
distributions represented through successor total returns. Independent LSEG
nominal final/anchor prices agree within 1e-6 relative tolerance. The stored
events reproduce the tested labels inside the same transaction. Price-series
hashes and archived filing/probe hashes accompany the provenance.

Source review corrected SCANA's legal completion to 2019-01-01, distinct from
the January 2 exchange session. Four target histories contain zero-volume
placeholder bars after source-documented trading suspension. Explicit reviewed
last-trade dates now permit entries on the final trading session and reject
later placeholder entries. Express Scripts' successor anchor uses the new
Cigna's first session, 2018-12-21, not the predecessor's December 20 bar.

Frozen-data replay produced the full affected horizon values for AET, ESRX,
SCG, BMS and TWX. This is not independent source certification: AET/BMS/TWX
native-RIC price queries failed and those events remain unapplied. Andeavor's
final non-election allocation is disclosed as approximately 87% stock and 13%
cash; the precision limitation remains explicit, without selecting an election
using future returns. Three total events are now documented, including EVHC.

Evidence: `.research/verification/terminal-price-bases.json`,
`.research/verification/successor-label-preview.json`, and
`.research/verification/successor-events-applied.json`. No RLS policy changed.

The follow-up snapshot is `.research/audit-snapshot-successor-events/` and the
production frame cache was refreshed from the same database read at version 5.
The reference remains uncertified. Full backend suite: 376 passed.
Exact-RIC grouping of the previous 162 absent identity rows yields 147 wholly
unmatched securities and one conflicting RIC (VAL.N^H20); see
`.research/verification/missing-security-inventory.json`. This inventory prevents
duplicate source rows becoming duplicate securities; no ticker rows were created.

## Independent foundation audit

The live DB audit confirms 264 classified adjustments and one stored terminal
event. The claim that only six deals and 162 missing tickers remain is not
supported: historical sectors, original filing availability, security identity,
and broader price/source coverage are still uncertified.

Two implementation defects were corrected: retirement filtering allowed a
90-day tail despite a recorded retirement date, and the directional action-date
rule accepted later dates. The loader now stops at the recorded date and frame
cache version 4 rejects stale inputs. All 85 newly applied share ratios replay
under the corrected rule. Three provenance offsets need the earlier ex-date
instead of the same-day effective date (LRCX -1, NKTR -4, AMCR -1); ratios agree.
No source rows or historical reports were overwritten.

Issuer-only identity matches remain candidates, even when the newer mapping
artifact labels them resolved. A lifecycle-compatible span is not proof of
security identity. Removing reused bars also does not resolve the original
company's terminal payoff; its acquisition must remain in the outcome census.

Time Warner's completion 8-K is located and its terms recorded in
`docs/terminal_event_research.json`; all six reviewed deal sources are now
hash-archived in `.research/verification/terminal-research-archived.json`. New successor
valuation logic keeps merger cash fixed and values the stock leg using successor
total returns, with explicit conversion to target adjusted units. It rejects
missing ownership closes, reused target tails and unknown adjustment bases.
It is unit tested but has not yet written any live successor labels.

Full backend suite: 375 passed. A fresh audit snapshot is saved at
`.research/audit-snapshot-goal-review/`; it captures corrected retirement filtering
and preserves open readiness gates. This does not certify the reference or replace
the production scoring cache. Legacy version-3 frame caches are rejected and the
add-ticker compatibility path reloads them. No schema or RLS policies changed.

## Security lifecycle follow-up

Sequential LSEG retrieval completed for all 862 native RICs, including exchange
ticker, first-trade and retirement fields, with no failed batches. Blank historical
symbols usually remain blank in the alternate exchange-ticker field. Class-preserving
separator normalization identifies BF-B and BRK-B candidates without merging classes.
The updated inventory has 595 unique candidates, 265 missing/mismatched and two
ambiguous securities. Missing/mismatched cases comprise 162 absent database issuers,
83 missing source symbols, 13 missing source issuer IDs and seven other mismatches.
Issuer-only matches are separate review hints and never become security matches.

Old and current PLD have different ISINs despite shared ticker/issuer metadata.
The audit flags price tails beyond source retirement for CAM, HOT, PCL and the
ambiguous PLD predecessor. CAM and PCL have prices through 2026-09-04. These are
review conflicts, not proof of terminal proceeds or sufficient grounds to truncate
prices automatically. No dated identity mapping or membership was written.

Evidence is `.research/verification/identity-lifecycle-archive.json` and
`.research/verification/security-mapping-lifecycle-v2.json`. The latter includes
a repeatable-read database ticker/price-span inventory, source hash, failed-query
list, review reasons and explicit price-history conflicts. Existing research RLS
and backend add-ticker access are unchanged. The four added audit regression tests
cover share classes, ticker reuse, missing source fields and stale price tails.

## Foundation follow-up

Migration 019 is applied. `price_history.share_split_factor` is separate from
the source price adjustment ratio: spin-offs must not inflate parent shares.
179 of 310 recorded adjustments are classified, including 44 pricing-only
events. The remaining 131 yield unknown shares across the event. Future price
refreshes invalidate a classification if the source price factor changes.
Frame cache version 3 rejects old inputs; add-ticker reloads them automatically.

As-traded reconstruction matched LSEG's explicitly unadjusted prices on all 27
sampled AAPL/NVDA/GE dates (maximum relative error below 6e-8). Nevertheless, the
old share calculation overstated GE's January 2023 count by 28.1% and April 2024
count by 25.3%. These were pricing-only distributions, not share splits. This
distinction follows the [LSEG action-field documentation](https://developers.lseg.com/en/article-catalog/article/workspace-corporate-actions-content-set-guide).
Original and corrected values are in `.research/verification/database-actions-final.json`.

The Envision membership mismatch is resolved at RIC/session level by a specific,
evidenced zero-duration predecessor review. [S&P's notice](https://www.spglobal.com/spdji/en/documents/indexnews/announcements/20161129-444551/444551_midamericaam5pzza4wing6.pdf?force_download=true)
places the AmSurg/successor change after December 1's close. Other same-day
contradictions still fail validation. Reconstruction yields 876 intervals over
862 RICs. The identity archive contains 593 unique symbol-and-issuer candidates,
267 missing/mismatched histories and two ambiguous cases. No candidates have been
silently promoted into dated security mappings or used to replace DB membership.

The first terminal outcome is stored: EVHC.N^J18, issuer CIK 1678531, acquired
October 11, 2018 for $46/share cash. The [SEC 8-K](https://www.sec.gov/Archives/edgar/data/1678531/000119312518297466/d637722d8k.htm)
is archived under `.research/terminal_sources/`. Identity and the final unit price
basis were verified, and live matured-label checks passed in the insertion
transaction. Entries after a confirmed terminal event are now masked, preventing
reused price tails from becoming new investments. Ten other flagged stale tails
remain under investigation; this is not a full terminal-event census.

Accounting monthly coverage was measured without fitting against returns. On
the existing uncertified 2022 nonfinancial panel, accruals/assets are available
for 80.25% of observations, asset growth for 98.79%, gross profitability/assets
for 38.62%. Financial observations remain unavailable. Filing availability and
security/sector provenance still require certification.

Full backend suite: **345 passed**. RLS/add-ticker database integration passed
again after migration 019; browser-role read/write attempts remain denied. The
actual price-upsert SQL was exercised too, including automatic invalidation of
share-action metadata when its source factor changes. Audit JSON now preserves
PostgreSQL decimal precision and serializes before creating exclusive files,
avoiding partial reports on serialization failure.
The follow-up audit snapshot is `.research/audit-snapshot-actions-2026-09-07/`:
730 frames, a 790 MiB input payload and a 365 KiB source-code archive. Both hashes
were independently verified. It retains incomplete readiness gates and is not
the certified reference for fitting. The earlier snapshot remains unchanged.
No real-data model comparison or promotion has run. Earlier audit results below
describe the snapshot before this follow-up and remain preserved for comparison.

## Initial migrated audit snapshot

Migrations 017 and 018 are applied to the configured database. All 728 membership
rows survived migration. Seven research tables have RLS enabled and all PUBLIC,
anon and authenticated table grants revoked. Backend ingestion uses the configured
direct database role; no service credentials are sent to browsers.

The rollback integration check exercised the actual add-ticker worker helpers:
insert/update ticker, write filing and share metadata, insert accounting facts,
load the ticker frame (including research tables), and write terminal job status.
All passed with RLS enabled. Anonymous and authenticated read/write attempts
failed. Test rows were rolled back; sequence allocations can leave harmless gaps.
The check does not exercise external feeds, HTTP authentication or model inference.

A separate cache regression was fixed: existing add-ticker installations with
legacy frame caches now fetch compatible data rather than failing the scoring job.
Research runs still reject incompatible caches. Full backend suite: **339 passed**.

The complete SEC backfill processed 720 CIK-bearing securities, 41,135 filings and
397,625 accounting facts. Two securities had no supported accounting facts.
697 securities have at least one dated point-in-time share observation. Weighted
averages are explicitly excluded. Gross-profit facts cover only 369 securities;
usable TTM feature coverage remains to be assessed.

Eight predefined examples replayed exactly from hashed raw SEC responses to DB
accounting values and share metadata: AAPL, GOOGL, NVDA, JPM, GOOG, GE, COL, EVHC.
This verifies ingestion fidelity, not independent economic correctness of SEC
share-class attribution or publication timing.

The LSEG archive retrieval returned 751 membership events and 503 current RICs.
Strict reconstruction rejects EVHC.N^L16's same-day join/leave on 2016-12-02;
dated security mapping and effective-date semantics require resolution. Existing
membership contains assumed starts, and no terminal-event records are populated.
Eleven inactive securities have old price tails requiring investigation; index
removal and missing prices are not treated as proof of delisting.

No corrected reference certification, real-data ablation, confirmation or
prospective model has been completed. Readiness entries distinguish incomplete
work from unavailable dependencies. FRED credentials, historical sectors and
independent news annotation remain outstanding.

An immutable **audit snapshot**, not a certified reference, was saved at
`.research/audit-snapshot-2026-09-07/`. It contains 730 security frames plus macro
inputs and the readiness report, captured in a repeatable-read transaction. The
765 MiB payload's SHA256 was independently rechecked:
`710ca8b22a90962121c5f436897ef9134948422657fae46a82a4c71a8be3ec09`.

Local evidence (ignored data artifacts):

- `.research/verification/migration-017-rehearsal.json`
- `.research/verification/migration-017-applied.json`
- `.research/verification/migration-018-applied.json`
- `.research/verification/add-ticker-rls.json`
- `.research/verification/database-post-backfill.json`
- `.research/verification/membership-archive.json`
- `.research/backfill-verified/` (database-bound checkpoints and raw SEC payloads)

## Archive storage (2026-09-07)

The 707 archived SEC companyfacts responses under
`.research/backfill-verified/raw/` are stored gzip-compressed as
`<sha256>.json.gz` (2.38 GB -> 0.175 GB). The file name remains the SHA256 of the
UNCOMPRESSED canonical bytes, so replay must `gzip.decompress` before hashing.
Every file was verified before and after compression: 707 digests matched on
read, and each round-trip was re-hashed before the plaintext copy was removed.
No archived response was modified or discarded.

Remaining large local artifacts, neither certified nor required to reproduce a
future certified reference: `.research/audit-snapshot-2026-09-07/` (765 MB, no
source archive) and `.research/audit-snapshot-actions-2026-09-07/` (791 MB, the
later of the two). The older snapshot is superseded; deleting it is safe but is
left to the operator because both are cited above as audit evidence.

## Second identity resolution pass (2026-09-07)

`scripts/map_security_identity.py` re-resolves the archived security lifecycle
(969 records) against the database. It orders two rules and records which one
produced each mapping: unique issuer CIK, then a class-preserving RIC stem when
the source carries no CIK. Every single-candidate mapping is then checked against
the security's own lifetime, using the 90-day grace of the existing symbol-reuse
guard. Share classes are never merged by issuer.

| status | n | meaning |
|---|---|---|
| resolved | 662 | one candidate; stored price span lies inside the security lifetime |
| resolved_no_history | 87 | one candidate; the ticker has no stored bars |
| reuse_conflict | 30 | bars continue >90d past retirement: another company under the symbol |
| history_conflict | 10 | bars predate the security's first trade: a spliced predecessor |
| ambiguous | 18 | several securities share the issuer (share classes, merger successors) |
| absent_from_database | 162 | no ticker row for the security |

This resolves the 88 securities that were unmatched only because a retired RIC
carries an empty source symbol (AABA/YHOO, AET, ABMD, AGN and 84 others): the
issuer CIK identifies exactly one ticker for each.

Two conflict classes are new correctness findings, not mapping bookkeeping:

- **Reuse (30)** — FTI, HAR, HOT, INFO, SBNY and others carry bars long after the
  mapped security retired. CLAUDE.md records 16 such symbols; the lifecycle check
  finds more because it tests every security against its own retirement date
  rather than a gap heuristic.
- **Predecessor splice (10)** — AA, AAL, AMCR, CRH, FTI, LYB, MTCH, SMCI, SW and
  VST hold bars from before the mapped security first traded. One `ticker_id`
  spans two companies across a merger or separation boundary, so momentum,
  volatility and fundamentals cross an issuer change. `AA` before 2016-10-18 is
  Alcoa Inc, not Alcoa Corp; `AAL` before 2013-12-09 is AMR.

The 18 ambiguous cases are share classes (GOOG/GOOGL, FOX/FOXA, NWS/NWSA,
UA/UAA, BRK) and merger successors (CB/ACE, DD/DWDP, JCI/TYC, KDP/DPS, CHK/EXE,
WBD/DISCA/DISCK, BNY/BK). Several of them also own unclassified price
adjustments, so identity and price basis are one problem, not two.

Nothing here is approved as a dated mapping and nothing was written to the
database. `membership` stays `incomplete`.

## Why 131 price adjustments remain unclassified (2026-09-07)

Decomposed against the archived corporate-action pull. The counts below are the
work list for `price_share_basis`, not a classification: nothing was applied.

| n | cause | what closing it needs |
|---|---|---|
| 35 | only `LSEG Pricing Only` records on the exact date | a stated tolerance rule for spin-offs |
| 9 | exact term-ratio match, LSEG date 1-4 days earlier | a bounded date-offset rule |
| 7 | mixed or non-matching capital change on the date | individual review |
| 60 | no source record within four days | a wider corporate-action pull |
| 20 | no RIC coverage in the identity archive | share-class and delisted RIC resolution |

The first group is the interesting one. For a spin-off the parent's share count
does not change, so the share ratio is 1.0 and only the price is adjusted, which
is exactly how the reviewed GE demergers were recorded by hand. The classifier
rejects them because it requires Yahoo's price factor to match the inverse of
LSEG's adjustment factor to 5e-4, and the two disagree by 0.1-1.2%: they value
the distributed stub against different reference prices. Verizon/Frontier 2010
shows 1.0662 against 1.0698, Abbott/AbbVie 2013 shows 2.0842 against 2.0900.
Widening that tolerance is a semantic claim about the source and needs its own
recorded rule, so it is left unapplied here.

The 20 uncovered securities and several of the 60 missing records belong to the
share-class and successor cases listed in the identity section above, including
GOOG/GOOGL 2014, UA/UAA 2016, DD 2019 and MTCH 2020.

## Macro source now ingesting, with two source findings (2026-09-07)

**Superseded by the 2026-09-08 correction below.** This historical entry describes
the invalid calendar-backdating importer. Its verification claims, generalization
that Treasury yields are never revised, and inference that BAA10Y necessarily
strengthens the prior are not accepted. The observed calendar dates and bounded
revision-audit counts alone did not establish point-in-time values.

`FRED_API_KEY` is configured. `backend/ingestion/macro.py` was rewritten: it reads
FRED's published vintage calendar and dates each observation by the first vintage
released strictly after it, so a Friday observation becomes available on the
following Monday. No release lag is assumed anywhere. 13,918 rows were ingested.

**FRED withdrew the ICE BofA high-yield history.** Every BAML option-adjusted
spread series now begins 2023-09-05, so `BAMLH0A0HYM2` covers none of the research
window and the registered credit factor had no data at all. The credit series is
re-sourced to `BAA10Y`, Moody's Baa corporate yield minus the 10-year Treasury,
daily from 1986. That is the classical default spread of Fama-French (1989) and
Welch-Goyal (2008), so the substitution strengthens rather than weakens the prior.
`add_macro_features` was pointed at the new series.

**The two series do not behave the same way under revision.** A bounded vintage
audit gives:

| series | window | observations | revised |
|---|---|---|---|
| DGS10 | 2015-16 | 521 | 0 |
| DGS10 | 2020 | 261 | 1 (a late fill of a missing value, not a restatement) |
| DGS10 | 2024 | 261 | 0 |
| BAA10Y | 2015-16 | 461 | 1 (late fill) |
| BAA10Y | 2020 | 261 | 8 genuine restatements |
| BAA10Y | 2024 | 261 | 1 genuine restatement |

Treasury yields are published once. The Baa spread is restated, by roughly 0.02
to 0.05 points on a spread near 3. That rules out the convenient assumption that
a current-vintage value can stand in for a published one before FRED's archive
begins. Observations older than the archive are stored with
`source='fred_pre_archive'` and `verified=false`; `add_macro_features` already
filters on `verified`, so they cannot reach a model.

The consequence is a real limitation, not a formality: **verified credit history
starts 2014-01-27**, so the macro family can only be evaluated from 2014 onward
unless a longer vintage source is found. That shortens its window against the
selection cutoffs and should be registered before the comparison is run.

ALFRED also refuses any request spanning more than 2000 vintage dates. DGS10 has
5,103, so the previous implementation would have failed outright on its first
call; ingestion now reads the vintage calendar separately.

## Price basis, retirement and the terminal census (2026-09-07)

**Share ratios.** `scripts/classify_share_actions.py` classifies observed price
adjustments under four named rules, recording which applied to each event: exact
date and terms, pricing-only distribution (the parent share count is unchanged,
so the ratio is 1.0), declared terms matching exactly where the source dates the
term change one to four days early, and one share-changing action combined with
price-only distributions. Every ratio comes from declared terms or the adjustment
type; the observed price is only a bounded sanity check. That took corroborated
adjustments from 179 to **264 of 310**. Maximum price disagreement was 1.2% on
pricing-only events and 2.9% on mixed ones.

Two defects were found while doing it, both of which silently destroyed
corroboration rather than raising: near-duplicate source records differing only
in numeric formatting ("1.0/9" against "1.0/9.0") squared the adjustment factor,
and one ticker scoped to the same RIC twice doubled every event on a date. The
remaining 46 emit NaN and are never guessed; their only cost is a NaN market cap
until the next as-reported share filing, a median of 60 days.

**Security retirement (migration 020).** `removed_at` records leaving the index,
which is not the end of a listing, so the reuse guard keyed on it plus a
180-day gap. Three securities defeat that entirely, because the next company's
bars are contiguous with the original's:

| ticker | security retired | bars ran to | dropped |
|---|---|---|---|
| HOT (Starwood) | 2016-09-23 | 2017-03-20 | 58 bars |
| COL (Rockwell Collins) | 2018-11-27 | 2020-11-30 | 438 bars |
| HAR (Harman) | 2017-03-13 | 2022-03-02 | 1,178 bars |

`tickers.security_retired_at` now carries the vendor date for 97 tickers and
`_drop_reused_symbol_bars` truncates on it. TechnipFMC is the case that proves the
rule needs care: FMC Technologies retired in 2017, but the same ticker carries the
live TechnipFMC listing, so any ticker that also maps to a security with no
retirement date is left alone.

**Terminal census.** Seventeen series end more than 45 days before the panel does:
one documented, four reuse tails, five stale ingestion in 2026, and six genuine
acquisitions still needing proceeds (TWX, ANDV, AET, ESRX, SCG, BMS). All six are
cash-and-stock or all-stock, so each needs its successor's value at every horizon
end. Their labels stay masked until then, which drops the acquisition premium for
six large caps: conservative, but a stated bias, not a neutral omission.

## Original-security containment and normalization — 2026-09-08

Full native-history comparison found that retirement truncation had retained
wrong-price histories: all 1,558 overlapping COL dates and 1,044 overlapping HAR
dates disagree with the exact historical RIC. This is not a tail-only problem.
HOT's 380 overlapping nominal prices match, and CA has no overlapping original
history after retirement filtering. Raw and adjusted native archives now contain
7,969 observations for these four securities.

Migration 021 adds per-source exclusion evidence on tickers. A repeatable-read
rollback rehearsal reproduced the audit, checked database rows against the
compressed preimage, exercised the real frame loader, and verified that no raw
price changed. Application reproduced the same result: COL/HAR no longer load
the 2,604 bars previously surviving retirement filtering. All 4,340 raw rows
are retained. Exclusions are source-specific; future Yahoo reingestion cannot
silently re-enable them. The loader carries the evidence into snapshots. Cache
version 7 rejects older caches, and add-ticker's existing reload path handles
that incompatibility. Production feature specifications were not changed.

Separately, [the embedded cash rule](embedded_cash_review_rule.md) explains 14
special-cash adjustment events in the original 147-security recovery cohort.
The normalized candidate now contains 140 securities / 212,061 rows, up from
133 / 192,030. The remaining seven are ASIX/WRK (interior price gaps),
CCEP/FERG/WIN (unresolved coincident adjustments), FHN (stock distributions),
and NBR (warrants). HOT's four 2014 special cash events are also explained;
all four original-security recovery histories normalize to 7,969 candidate
rows. HOT's stored September 21, 2016 dividend is absent from the native archive,
so it is not certified merely because nominal prices agree. Merger-related
dividend entitlement also requires review before terminal valuation.

Independent forward-ratio replay passed every candidate row without calling the
normalizer. Synthetic tests cover double counting, mixed ordinary/special cash,
duplicate source formatting, coincident demergers, unsupported split explanations,
malformed exclusion evidence and exclusion from market controls. All 423 backend
tests passed after the price-normalization change. No registered experiment ran,
and all three shared source statuses remain incomplete.

Receipts:

- `.research/verification/original-security-price-parity.json`
- `.research/verification/price-source-exclusions-rehearsal.json`
- `.research/verification/price-source-exclusions-applied.json`
- `.research/verification/historical-embedded-cash-review.json`
- `.research/normalized-historical-candidate-v3/source-parity.json`
- `.research/original-security-normalized-candidate-v2/source-parity.json`

The production frame cache was refreshed to version 7 and independently compared
with `.research/audit-snapshot-exclusions-2026-09-08`: all 730 frames agree,
covering 2,396,366 price rows, with COL/HAR excluded and provenance retained.
The input hash is
`f2bce6cbdb09e68a81c9a1af7d95da1d83db6adf43dcce4ee46f502804e43d10`.
`.research/verification/frame-cache-v7-exclusions.json` verifies the snapshot
payload and source-archive hashes, full cache equality and continued rejection
of all three shared source gates. This is an audit snapshot, not a corrected
full-universe reference; research-only historical additions remain excluded.

## BAA10Y decision and actual vintage recovery (2026-09-08)

The user retained DGS10/BAA10Y for the initial comparison and deferred extended
longer-history sourcing unless initial evidence warrants it. The complete policy
is in `macro_research_decision.md`; no screen, confirmation or selection cutoff
changed. The limited source check found an ICE historical-data lead, not a
verified replacement; no vendor contact, trial or purchase occurred.

The prior importer used current revised observations and calendar-derived dates.
It has been replaced with explicit annual real-time queries, complete pagination,
raw response hashes and exact offline source replay. Annual-boundary duplicates
are suppressed, but genuine value changes and missing-value revisions survive.
Features require `source='alfred_realtime_v2'` and next-day vintage availability;
legacy source labels do not qualify even if an old snapshot says `verified=true`.

Recovered and applied **13,936 actual value versions**: 6,963 DGS10 and 6,973
BAA10Y, each covering 6,959 distinct observation dates. Four DGS10 and fourteen
BAA10Y observation dates have revisions (including late missing-value fills).
For example, DGS10's 2014-01-14 value changed from 2.77 to 2.88; BAA10Y's
2020-05-26 value changed from 3.12 to 3.14 in the 2020-07-21 vintage. The old
claim that Treasury yields are never revised was disproved by full retrieval.

Application reproduced the rollback rehearsal exactly and verified every stored
corrected row. All 13,918 old rows are recoverable from a hashed gzip preimage;
12,049 colliding rows were replaced, and the 1,869 noncolliding legacy rows remain
in the database with `verified=false`. The table now has 15,805 rows for these
series. The legacy backfill command fails before accessing the database and
directs callers to the audited download/rehearsal/application workflow.

Eight independent single-vintage FRED queries matched the reconstructed as-of
values exactly: each series' first vintage, 2020-07-20, 2020-07-21 and 2024-12-03.
The checks cover the first archive and both sides of a known credit revision.
Raw query responses contain no API keys.

Receipts:

- `.research/macro-realtime-candidate-v1/candidate.json` and `summary.json`
- `.research/verification/macro-realtime-v2-rehearsal.json`
- `.research/verification/macro-realtime-v2-applied.json`
- `.research/verification/macro-realtime-v2-spotcheck.json`

Candidate SHA256: `cb0431927aa01c201c3baee0bc5e9760ce5ee760697faf8d4c47bd934fb3d3e1`.
Preimage SHA256: `4e73276470ad99216caeb338924025d3b24a97f9e22bc5f812e8625a69471a66`.
All 439 backend tests passed, including vintage pagination, revision/withdrawal,
legacy-source rejection, archive integrity and overwrite guards. Macro readiness
remains incomplete pending usable paired coverage and a corrected frozen reference;
the three shared gates remain incomplete and no registered experiment has run.

Diagnostic member-feature coverage on the existing 730-frame audit snapshot
starts 2013-02-28 for rates and 2014-01-31 for credit, with both observed in 109
months through June 2023. October 2016–February 2017 credit features remain NaN:
the last then-archived observation is 2016-10-06, and the later observations first
appear in the 2017-03-22 vintage. The seven-day staleness limit is unchanged.
This audit excludes neither missing labels nor incomplete baseline controls, so
it is not final paired coverage and cannot certify the family or reference.
Receipt: `.research/verification/macro-realtime-v2-coverage.json`.

Six further independent snapshots (both series on 2016-10-31, 2017-02-28 and
2017-03-22) matched exactly, confirming the credit archive gap rather than an
annual-window retrieval bug. Receipt:
`.research/verification/macro-realtime-v2-gap-spotcheck.json`.

## Shared reference and paired samples (2026-09-08, subsequent work)

`reference_gate_progress.md` records the fresh 878-frame audit snapshot, exact
pre-fit paired coverage and offline Praxair/Linde reference repair. The runner's
sample-selection helper is now shared with the coverage audit; 443 backend tests
passed. On the current reference, baseline/candidate controls and labels match on
48,686 test rows in 107 folds, of which 102 have both macro inputs observed.
No model was fitted and no source status promoted.

The separate Praxair candidate stages 2,223 prices and its membership interval,
removes predecessor bars from Linde's own feature history, and exercises 252 exact
successor valuations. It preserves the original database and production cache;
distribution completeness, successor-price certification and a rollback rehearsal
remain necessary before application.

Independent replay verified preservation of all 876 unrelated frames, macro
values, issuer attributes and gate statuses. All 2,223 nominal closes agree across
feeds. The four 2017 dividend discrepancies are resolved in favor of the issuer's
exact 0.7875 amount; the report is archived in
`.research/verification/praxair-dividend-source-archived.json`. Two small daily
adjustment residuals remain disclosed rather than widening the parity tolerance.
The full paired audit on the separated candidate matches on 48,673 test rows in
107 folds, with both macro inputs observed on 45,512 rows in 102 folds. Details
and remaining application prerequisites are in `reference_gate_progress.md`.

## Praxair scoped application and sector-gap correction (2026-09-08)

The scoped PX/LIN repair is now applied after exact rollback rehearsal v2.
It adds 2,223 native PX rows, separate membership and 252 native-successor
terminal values. All 4,194 raw LIN bars remain stored; its loaded history starts
2018-10-31 and contains 1,971 bars. Distinct issuer filings/accounting facts and
all sector records are preserved. No staged Materials intervals were applied.
Native successor checks cover 253 sessions and four cash dates. The two residual
PX return discrepancies are reproduced within Yahoo's own derived adjusted
series; source-reconstructed values are retained without tolerance changes.

The application and rollback receipts are:
`.research/verification/praxair-reference-applied-v1.json` and
`.research/verification/praxair-reference-rehearsal-v2.json`.
Exact pre-write rows are archived at
`.research/verification/praxair-reference-preimage.json.gz`; no raw rows were deleted.

Migration 022 enforces a reviewed inclusive start boundary at load time.
Successfully loaded but empty/gapped historical sectors now remain unknown;
unexpected sector-query failures propagate. Version-8 cache/snapshots replace
version 7. The backend suite passes **463 tests** (123 existing constant-input
correlation warnings). New application/audit/verifier files pass their
targeted lint checks. All shared source gates remain incomplete; no model fit or
registered experiment has run. The earlier offline-candidate paired count cannot
be reused because that candidate staged unverified Materials intervals.

Post-commit verification subsequently passed: all 876 unrelated securities and
all macro values preserved; 156 empty sector histories explicitly unknown;
production cache exactly matches its 730-frame snapshot (2,394,143 prices), while
the research snapshot contains 878 frames. Receipt:
`.research/verification/praxair-reference-postcommit-v1.json`.

Fresh stress and macro paired audits both match on 48,673 test rows / 107 folds.
Stress inputs are observed on all those rows; both macro inputs are observed on
45,512 rows / 102 folds. There are 51 unknown-sector and 2,300 missing-size test
rows. Undefined sector-grade training labels remain excluded by the runner;
unknown-sector predictions do not contribute to within-sector scoring.
Receipts: `.research/paired-coverage-praxair-applied-stress-2026-09-08-v1/report.json`
and `.research/paired-coverage-praxair-applied-macro-2026-09-08-v1/report.json`.
These are pre-fit coverage checks, not certification or performance results.

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

### Historical cash-return audit — 2026-09-19

The hash-bound 140-history audit compared 211,864 daily returns: 102 histories
match ordinary cash returns, 34 retain capital/gap review, and four retain cash
disagreements. Sixteen stale source rows are excluded without retiming; 16 missing
native sessions remain visible. Issuer evidence corroborates the disputed KTB
cash and the March 2010 dividend on both former Fox classes. No price overrides
or global gate promotions followed. See docs/experiment_gate_status.md and
.research/verification/historical-total-return-audit-2026-09-19-v1.json.
Targeted validation passed 41 tests; new audit files pass Ruff.
