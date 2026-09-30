# Macro research decision — 2026-09-08

User decision: keep BAA10Y for the initial experiments while checking for a longer
history source; avoid extended source work if the experiments do not show useful
evidence. This resolves the earlier choice between accepting a shorter usable
window and deferring macro entirely.

## Working comparison and stopping policy

- Keep DGS10 and BAA10Y. BAA10Y is Moody's Baa corporate yield minus the 10-year
  Treasury yield, not the originally proposed high-yield option-adjusted spread.
  No post-result series switching or alternative-horizon rescue.
- First repair actual vintage retrieval, audit feature and paired-cohort coverage,
  and pass the shared reference gates. Then freeze inputs and run the registered
  macro/6M screen. Do not fit against the known-invalid legacy macro values.
- Preserve the registered screen threshold (paired mean ΔIC >= 0.003), followed
  by eight-seed confirmation and the existing seven-family Holm criteria. A
  promising screen can justify revisiting sourcing; it is not significant evidence
  on its own. There is no new p-value cutoff chosen after seeing results.
- If the initial comparison fails to justify further work, park extended source
  research. Report inadequate coverage or insufficient effective blocks as
  inconclusive, not proof of no macro signal. Disclose dates, paired observations,
  uncertainty and the existing doubled-block sensitivity.
- Do not buy data, contact vendors or change the registered credit definition
  without a further decision. Preserve the original pre-2024 label cutoff.

## Bounded source check

[ICE's index platform](https://www.ice.com/fixed-income-data-services/fixed-income/index-trial)
advertises current and historical index-level data. This is a lead, not a verified
replacement: the exact high-yield OAS series, earliest coverage, historical value
versions, entitlements and cost have not been established. No trial or contact was
initiated. Extended investigation is deferred under the policy above.

## Vintage correction and diagnostic coverage

The old importer fetched current observations and assigned dates using the vintage
calendar. This does not recover the values actually known at those dates. BAA10Y
revisions have been observed, so the old `verified` flags cannot certify these rows.
The [FRED observation API](https://fred.stlouisfed.org/docs/api/fred/series_observations.html)
defaults both real-time endpoints to today; corrected retrieval must explicitly
request historical real-time periods and preserve individual value versions.

Corrected rows use `alfred_realtime_v2`; the feature builder rejects legacy source
labels even if their old verification flag is true. Date-only vintage availability
is delayed to the next calendar day. Missing-value revisions must supersede prior
values. Raw responses must be archived without credentials before application.

The observed vintage calendars begin 2005-06-28 for DGS10 and 2014-01-27 for
BAA10Y. These are not yet certified feature start dates. An initial archived
vintage may contain older observations usable for estimator history **from that
vintage onward**, never for backdating earlier predictions. Audit the actual
available history and estimator burn-in before declaring the usable window.

The corrected retrieval has now been archived, replayed, applied after a rollback
rehearsal and independently spot-checked. The current **uncertified** reference
produces credit features from 2014-01-31, with both macro features observed in
109 months through 2023-06-30. Five months (October 2016–February 2017) remain
missing because the latest credit observation available in those archives is
2016-10-06, exceeding the registered seven-day staleness limit. Later observations
first appear in the 2017-03-22 vintage; they cannot repair earlier predictions.
This is member-feature availability, not final matched-label/model coverage or
evidence of predictive value. Paired coverage must be re-audited on the corrected
reference. Receipt: `.research/verification/macro-realtime-v2-coverage.json`.
