from datetime import date

import numpy as np

from backend.ml.factors.long_horizon import fixed_estimate_features
from scripts.prepare_exploratory_analyst import deduplicate, parse_records


def raw(calc, target, eps, update=None):
    return {"Instrument": "A.N", "Calc Date": calc, "Date": update or calc,
            "Period End Date": target, "Financial Period Absolute": "FY2021",
            "Earnings Per Share - Mean": str(eps)}


def test_exploratory_estimates_preserve_availability_and_default_exclusion():
    rows, rejected = parse_records([
        raw("2020-03-31", "2021-12-31", 2, "2020-03-15"),
        raw("2020-04-30", "2021-12-31", 3, "2020-04-20"),
        raw("2020-04-30", "2021-12-31", 99, "2020-05-01"),
    ], {"A.N": 1})
    assert rows[0]["as_of_date"] == date(2020, 3, 31)
    assert rejected["unavailable_or_expired"] == 1
    assert all(not r["verified"] for r in rows)
    d = [date(2020, 4, 30)]
    assert np.isnan(fixed_estimate_features(rows, d)[0]["fixed_eps_rev_30d"])
    assert fixed_estimate_features(rows, d, allow_provisional=True)[0]["fixed_eps_rev_30d"] == 0.5
    rows[-1]["fiscal_period_end"] = date(2022, 12, 31)
    result = fixed_estimate_features(rows, d, allow_provisional=True)[0]
    assert np.isnan(result["fixed_eps_rev_30d"])


def test_conflicting_snapshots_excluded_and_duplicate_snapshots_count_once():
    a, _ = parse_records([raw("2020-03-31", "2021-12-31", 2)], {"A.N": 1})
    assert deduplicate(a + a) == (a, 0)
    assert deduplicate(a + [dict(a[0], eps=3)]) == ([], 1)
