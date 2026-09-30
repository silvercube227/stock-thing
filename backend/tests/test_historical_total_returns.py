from copy import deepcopy
from datetime import date

import pytest

from scripts.audit_historical_total_returns import compare


def fixture():
    prices = [
        dict(trade_date=date(2020, 1, 2), close=100, dividend=0, split_factor=1),
        dict(trade_date=date(2020, 1, 3), close=99, dividend=2, split_factor=1),
    ]
    returns = [
        dict(
            Instrument="X.N",
            Date=str(p["trade_date"]),
            **{"Calc Date": str(p["trade_date"]), "Daily Total Return": r},
        )
        for p, r in zip(prices, ["<NA>", "1"], strict=True)
    ]
    return prices, returns


def test_missing_cash_is_disagreement_and_no_distribution_is_inferred():
    prices, returns = fixture()
    assert compare("X.N", prices, returns)["compared_returns"] == 1
    prices[1]["dividend"] = 0
    result = compare("X.N", prices, returns)
    assert len(result["disagreements"]) == 1
    assert prices[1]["dividend"] == 0


def test_stale_row_outside_native_calendar_is_disclosed_never_retimed():
    prices, returns = fixture()
    extra = deepcopy(returns[-1])
    extra["Calc Date"] = "2020-01-06"
    result = compare("X.N", prices, returns + [extra])
    assert result["compared_returns"] == 1
    assert len(result["excluded_source_rows"]) == 1
    returns[-1]["Date"] = "2020-01-02"
    assert compare("X.N", prices, returns)["stale_native_dates"] == ["2020-01-03"]


def test_gap_and_capital_changes_are_not_claimed_as_daily_matches():
    prices, returns = fixture()
    prices[-1]["split_factor"] = 2
    result = compare("X.N", prices, returns)
    assert result["capital_dates_not_compared"] == ["2020-01-03"]
    assert result["compared_returns"] == 0
    prices[-1]["trade_date"] = date(2020, 1, 6)
    returns[-1].update(Date="2020-01-06", **{"Calc Date": "2020-01-06"})
    result = compare("X.N", prices, returns)
    assert result["gap_crossings_not_compared"] == ["2020-01-06"]
    assert result["missing_native_sessions"] == ["2020-01-03"]


@pytest.mark.parametrize("problem", ["duplicate", "security", "nonfinite", "missing"])
def test_bad_source_cannot_pass(problem):
    prices, returns = fixture()
    if problem == "duplicate":
        with pytest.raises(ValueError, match="Duplicate"):
            compare("X.N", prices, returns + returns[-1:])
    elif problem == "security":
        returns[-1]["Instrument"] = "OTHER.N"
        with pytest.raises(ValueError, match="security mismatch"):
            compare("X.N", prices, returns)
    else:
        if problem == "missing":
            returns.pop()
        else:
            returns[-1]["Daily Total Return"] = "nan"
        assert compare("X.N", prices, returns)["status"] == "source_disagreement_requires_review"
