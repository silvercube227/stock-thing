from datetime import date

from scripts.audit_historical_action_components import audit


def fixture():
    prices = [
        dict(trade_date=date(2020, 1, 2), close=100, dividend=0, split_factor=1),
        dict(trade_date=date(2020, 1, 3), close=51, dividend=0, split_factor=2),
    ]
    action = {
        "Instrument": "X.N",
        "Corporate Change Event Type": "Share Split",
        "Capital Change Is Rescinded": "False",
        "Capital Change Ex Date": "2020-01-03",
        "Capital Change Effective Date": "2020-01-03",
        "Terms New Shares": "2",
        "Terms Old Shares": "1",
        "Adjustment Type": "Capital Change Type",
    }
    returns = [
        dict(
            Instrument="X.N",
            Date="2020-01-03",
            **{"Calc Date": "2020-01-03", "Daily Total Return": "2"},
        )
    ]
    return prices, action, returns


def test_split_requires_declared_terms_and_return_parity():
    prices, action, returns = fixture()
    r = audit("X.N", prices, [action], returns, {})
    assert r["events"][0]["status"] == "declared_split_return_corroborated"
    assert r["events"][0]["share_ratio"] == 2
    action["Terms New Shares"] = "3"
    assert (
        audit("X.N", prices, [action], returns, {})["events"][0]["status"]
        == "capital_distribution_requires_review"
    )
    action["Terms New Shares"] = "2"
    returns[0]["Daily Total Return"] = "3"
    assert (
        audit("X.N", prices, [action], returns, {})["events"][0]["status"]
        == "declared_split_return_disagrees"
    )


def test_mixed_distribution_is_not_certified_by_a_matching_split():
    prices, action, returns = fixture()
    demerger = {**action, "Corporate Change Event Type": "Demerger"}
    r = audit("X.N", prices, [action, demerger], returns, {})
    assert r["events"][0]["status"] == "capital_distribution_requires_review"


def test_declared_distribution_without_price_step_remains_visible():
    prices, action, returns = fixture()
    prices[1]["split_factor"] = 1
    r = audit("X.N", prices, [action], returns, {})
    assert r["declared_actions_without_exact_price_step"] == [action]
    assert r["source_certified"] is False


def test_special_cash_hypothesis_match_does_not_select_a_convention():
    prices, _, returns = fixture()
    prices[1].update(close=99, dividend=1, split_factor=100 / 99)
    returns[0]["Daily Total Return"] = "0"
    r = audit("X.N", prices, [], returns, {"2020-01-03": {"amount": "1"}})
    event = r["events"][0]
    assert event["status"] == "special_cash_convention_requires_review"
    assert event["hypothesis_errors"]["special_cash_price_adjustment"] == 0
    assert "share_ratio" not in event
