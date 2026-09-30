from datetime import date

import pytest

from scripts.audit_cash_recovery_components import compare


def fixture():
    prices = [
        dict(trade_date=date(2020, 1, 2), close=100.0, dividend=0.0, split_factor=1.0),
        dict(trade_date=date(2020, 1, 3), close=99.0, dividend=2.0, split_factor=1.0),
    ]
    returns = [
        dict(
            Date=str(p["trade_date"]),
            **{"Calc Date": str(p["trade_date"]), "Daily Total Return": v},
        )
        for p, v in zip(prices, ["0", "1"], strict=True)
    ]
    return prices, returns


def test_cash_component_check_detects_missing_distribution():
    prices, returns = fixture()
    assert compare(prices, returns)["compared_returns"] == 1
    prices[1]["dividend"] = 0
    with pytest.raises(ValueError, match="Cash-return disagreement"):
        compare(prices, returns)


def test_cash_component_check_does_not_align_stale_or_duplicate_returns():
    prices, returns = fixture()
    with pytest.raises(ValueError, match="Duplicate or stale"):
        compare(prices, returns + [returns[-1]])
    returns[-1]["Date"] = "2020-01-02"
    with pytest.raises(ValueError, match="Duplicate or stale"):
        compare(prices, returns)


def test_cash_component_check_refuses_capital_adjustments_or_missing_sessions():
    prices, returns = fixture()
    prices[-1]["split_factor"] = 2
    with pytest.raises(ValueError, match="capital adjustments"):
        compare(prices, returns)
    with pytest.raises(ValueError, match="session coverage differs"):
        compare(prices, returns[:1])
