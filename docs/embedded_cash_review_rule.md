# Embedded special cash review — 2026-09-08

Capital-adjusted vendor prices can already include special cash distributions.
Applying every dividend again would double count those events. LSEG's own
[worked example](https://community.developers.lseg.com/discussion/39737/dividend-adjusted-historical-prices-for-us-stocks/p1)
demonstrates the issue, but explicitly does not certify every event type. The
[corporate-action guide](https://developers.lseg.com/en/article-catalog/article/workspace-corporate-actions-content-set-guide)
also distinguishes price-only adjustments from changes in shares outstanding.
Neither source is treated as blanket certification of these histories.

The candidate transformation now accepts a narrowly reviewed cash component:

1. Raw and adjusted histories must have exact-date source records, archived by
   hash. USD cash amount and ex-date must be explicitly reported; missing or
   conflicting dividend records fail review.
2. Sum declared `Special`/`Extra` amounts on that date, deduplicating identical
   typed events. Do not infer the amount from the observed price step or search
   subsets of events for a better match.
3. Reject reported coincident non-buyback capital actions. The observed change
   in adjusted/raw scale must match `previous nominal close / (previous nominal
   close - declared special cash)` within relative tolerance 1e-5.
4. Retain the entire declared cash dividend in the normalized row. Subtract
   only the reviewed embedded portion when adjusting the capital series again.
   Recheck the factor inside the normalizer; a review cannot bypass it.
5. Do not infer share-count ratios. Unknown actions, source completeness,
   identities and terminal outcomes remain separate gates.

Fourteen events explain seven previously blocked historical candidates. Four
additional events explain HOT's 2014 special distributions. Independent forward
return replay agrees with the backward normalizer across 212,061 historical
candidate rows and 7,969 original-security recovery rows; maximum relative
errors are 4.45e-16 and 2.23e-16 respectively. This establishes transformation
parity only. HOT still has a September 2016 cash-source discrepancy.

Evidence is in `.research/verification/historical-embedded-cash-review.json`,
`.research/verification/original-security-embedded-cash-review.json`, and the
`source-parity.json` receipts beside the version-3 historical and version-2
original-security candidate artifacts. No candidate prices were applied.
