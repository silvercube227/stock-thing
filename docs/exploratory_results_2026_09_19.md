# Exploratory signal experiments — 2026-09-19

Five screening experiments completed on the available data with frozen exclusions. These are
single-seed screens, not production-promotion evidence. No further exhaustive
source audit is required before examining these results.

| Comparison | Paired change in sector IC | 95% calendar-block interval | Next step |
|---|---:|---|---|
| Stress / 6M | -0.00555 | [-0.02301, +0.00716] | Does not earn confirmation |
| Macro / 6M | +0.00167 | [-0.01036, +0.01517] | Below +0.003 confirmation threshold |
| Accounting / 1Y | -0.01634 | [-0.05176, +0.00524] | Does not earn confirmation |
| Stress + macro / 6M | +0.00019 | [-0.01954, +0.01769] | Below +0.003 confirmation threshold |
| Fixed-target analyst / 3M | +0.00717 | [+0.00139, +0.01345] | Eight-seed confirmation completed; inconclusive |

All three 6M screens share exactly the same baseline predictions and 107 monthly
evaluation folds, July 2014–May 2023. Baseline mean sector IC is -0.03131.
Predictions, controls, entry dates and realization dates match across each pair;
entries follow feature dates and all selection labels finish before 2024.

Usable coverage is 46,927 / 53,965 intended security-months (**86.96%**), below
the 90–95% target. Coverage is **95.11% for surviving securities** and **42.31%
for removed securities**, using a fixed September 29, 2023 cohort anchor. This
survivorship limitation restricts generalization; it is disclosed rather than
used to postpone exploratory fits. Missing source inputs remain missing.

Accounting covers 95 monthly folds / 41,672 paired rows, with **86.93%** usable
coverage. Its baseline mean sector IC is -0.02630 versus -0.04264 with the pack.
Accounting values remain sparse: full-panel availability is 59.16% for accruals,
80.81% for asset growth and 33.31% for gross profitability. Both sides retain the
same sample; missing candidate features are handled as missing.

**The analyst screen earned confirmation; its confirmation is inconclusive.
The first four screens did not earn confirmation.**
No thresholds, horizons or feature subsets were changed in response. News and
expanded-universe comparisons remain unexecuted; incomplete certification alone will not block
their exploration when usable inputs exist.

Analyst uses 204,062 provisional monthly fixed-target observations across 796
securities added to the same reference. Its snapshot is
`.research/exploratory-analyst-2026-09-19-v1`, hash
`0f1ce8c6f5e93c361643b64cc2bd1765328bc0e63439d202dbfffc1acbf7c94d`.
The 3M screen covers 113 folds, April 2014–August 2023, and 49,588 matched rows
out of 56,977 intended security-months (**87.03%**). Baseline mean sector IC is
-0.01308 versus -0.00592 with analyst features. Size-neutral IC improves +0.00753
and top-decile excess return improves +0.00160. Full-panel availability is 79.84%
for 30-day revisions, 79.82% for 90-day revisions and 56.38% for persistence.
Availability follows CalcDate; revisions compare the same absolute fiscal target.
Vendor fiscal ends, current adjusted USD basis and the fixed 0.01 EPS cutoff
remain limitations. No estimate row is marked verified. Eight-seed confirmation
used the unchanged snapshot and implementation. Its paired IC change is
**+0.00460**, with 95% three-month block interval **[+0.00046, +0.00881]**.
The six-month sensitivity interval is also positive [+0.00036, +0.00884].
However, the seven-comparison Holm-adjusted p-value is **0.22748**, and top-decile
excess return changes **-0.00041**. It therefore does not pass the full confirmation
criteria. Size-neutral IC improves +0.00415. Absolute baseline/candidate mean
sector IC remains negative (-0.01333 / -0.00873). No configuration is promoted.
Both phases' saved predictions pass paired controls and chronology checks.
Analyst evaluation coverage is 95.09% for surviving securities and 42.90% for
removed securities, so the survivorship limitation remains material.

News has only 15 complete three-month security windows in its annotation archive,
including two empty windows; expanded-universe data is not yet populated locally.
Those input gaps remain. They do not reopen exhaustive certification requirements
for runnable families. No experiment process is left running from this batch.

The frozen reference is `.research/exploratory-reference-2026-09-19-v2`, hash
`dd5c443d45d244e946f230f957380fe945e8fbbccf1593f325bb70abfc910eda`.
Its reconstructed SPX cohort includes unmapped and unpriced securities in the
coverage denominator. Results, manifests, coverage and replayable predictions
are under `.research/experiments-2026-09-19-v2/exploratory/<family>/screening/`.
The initial v1 attempts stopped before fitting and are retained as failed
preflights; v2 corrects their legacy-membership denominator.
The updated content-hashed results summary is
`.research/experiments-2026-09-19-v2/exploratory/completed_screens-v2.json`.

Accounting's first attempt stopped before fitting because the newly added paired
check compared unused raw accounting columns with normalized candidate columns.
The repaired check excludes only candidate-only transformations and continues to
require identical shared features, labels and controls. The failed preflight is
retained, and the accounting data and feature definitions are unchanged.
