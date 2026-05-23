"""Unit tests for the EDGAR companyfacts parser. No network or DB."""

from __future__ import annotations

from datetime import date

from backend.ingestion.fundamentals import parse_companyfacts


def _facts(concepts: dict) -> dict:
    """Build a minimal companyfacts JSON given {concept_name: [entry_dicts]}."""
    return {
        "facts": {
            "us-gaap": {
                name: {"units": {"USD": entries}} for name, entries in concepts.items()
            }
        }
    }


def _annual(accn: str, start: str, end: str, val: float, filed: str) -> dict:
    return {
        "start": start, "end": end, "val": val, "accn": accn,
        "fy": int(end[:4]), "fp": "FY", "form": "10-K", "filed": filed,
    }


def _quarter(accn: str, start: str, end: str, val: float, filed: str) -> dict:
    return {
        "start": start, "end": end, "val": val, "accn": accn,
        "fy": int(end[:4]), "fp": "Q1", "form": "10-Q", "filed": filed,
    }


def _instant(accn: str, end: str, val: float, filed: str, form: str = "10-K") -> dict:
    """Balance-sheet item — no `start`."""
    return {
        "end": end, "val": val, "accn": accn,
        "fy": int(end[:4]), "fp": "FY", "form": form, "filed": filed,
    }


# =============================================================
# Concept fallback
# =============================================================


def test_prefers_first_concept_in_fallback_list() -> None:
    # RevenueFromContractWithCustomerExcludingAssessedTax (preferred) and
    # Revenues (fallback) both present — preferred wins.
    facts = _facts(
        {
            "RevenueFromContractWithCustomerExcludingAssessedTax": [
                _annual("ACC-1", "2023-01-01", "2023-12-31", 1_000_000_000, "2024-02-01")
            ],
            "Revenues": [
                _annual("ACC-1", "2023-01-01", "2023-12-31", 999_000_000, "2024-02-01")
            ],
        }
    )
    rows = parse_companyfacts(facts, ticker_id=1)
    assert len(rows) == 1
    assert rows[0]["revenue"] == 1_000_000_000


def test_falls_back_when_preferred_missing() -> None:
    facts = _facts(
        {
            "Revenues": [
                _annual("ACC-1", "2023-01-01", "2023-12-31", 500_000_000, "2024-02-01")
            ],
        }
    )
    rows = parse_companyfacts(facts, 1)
    assert rows[0]["revenue"] == 500_000_000


# =============================================================
# Period filtering — keeps natural span, drops YTD cumulative
# =============================================================


def test_drops_ytd_quarterly_entries() -> None:
    # 10-Q filing should keep only the 3-month standalone, not the 9-month YTD.
    facts = _facts(
        {
            "NetIncomeLoss": [
                _quarter("ACC-Q3", "2023-07-01", "2023-09-30", 50, "2023-11-01"),  # Q3 quarter
                {  # YTD 9 months — must be filtered out
                    "start": "2023-01-01", "end": "2023-09-30", "val": 150,
                    "accn": "ACC-Q3", "fy": 2023, "fp": "Q3", "form": "10-Q",
                    "filed": "2023-11-01",
                },
            ]
        }
    )
    rows = parse_companyfacts(facts, 1)
    assert len(rows) == 1
    assert rows[0]["net_income"] == 50


def test_keeps_balance_sheet_instant_values() -> None:
    """Balance-sheet items have no `start`. They should pass through."""
    facts = _facts(
        {
            "StockholdersEquity": [_instant("ACC-1", "2023-12-31", 5_000_000, "2024-02-01")],
        }
    )
    rows = parse_companyfacts(facts, 1)
    assert rows[0]["total_equity"] == 5_000_000


# =============================================================
# Derived ratios
# =============================================================


def test_gross_margin_from_gross_profit() -> None:
    facts = _facts(
        {
            "Revenues": [_annual("A", "2023-01-01", "2023-12-31", 1000, "2024-02-01")],
            "GrossProfit": [_annual("A", "2023-01-01", "2023-12-31", 400, "2024-02-01")],
        }
    )
    rows = parse_companyfacts(facts, 1)
    assert rows[0]["gross_margin"] == 0.4


def test_gross_margin_falls_back_to_revenue_minus_cost() -> None:
    facts = _facts(
        {
            "Revenues": [_annual("A", "2023-01-01", "2023-12-31", 1000, "2024-02-01")],
            "CostOfRevenue": [_annual("A", "2023-01-01", "2023-12-31", 600, "2024-02-01")],
        }
    )
    rows = parse_companyfacts(facts, 1)
    assert rows[0]["gross_margin"] == 0.4  # (1000 - 600) / 1000


def test_operating_margin() -> None:
    facts = _facts(
        {
            "Revenues": [_annual("A", "2023-01-01", "2023-12-31", 1000, "2024-02-01")],
            "OperatingIncomeLoss": [
                _annual("A", "2023-01-01", "2023-12-31", 250, "2024-02-01")
            ],
        }
    )
    rows = parse_companyfacts(facts, 1)
    assert rows[0]["operating_margin"] == 0.25


def test_total_debt_sums_components() -> None:
    facts = _facts(
        {
            "LongTermDebtNoncurrent": [_instant("A", "2023-12-31", 300, "2024-02-01")],
            "LongTermDebtCurrent": [_instant("A", "2023-12-31", 100, "2024-02-01")],
            "ShortTermBorrowings": [_instant("A", "2023-12-31", 50, "2024-02-01")],
        }
    )
    rows = parse_companyfacts(facts, 1)
    assert rows[0]["total_debt"] == 450


def test_fcf_is_ocf_minus_capex() -> None:
    facts = _facts(
        {
            "NetCashProvidedByUsedInOperatingActivities": [
                _annual("A", "2023-01-01", "2023-12-31", 500, "2024-02-01")
            ],
            "PaymentsToAcquirePropertyPlantAndEquipment": [
                _annual("A", "2023-01-01", "2023-12-31", 120, "2024-02-01")
            ],
        }
    )
    rows = parse_companyfacts(facts, 1)
    assert rows[0]["fcf"] == 380


# =============================================================
# Look-ahead protection
# =============================================================


def test_restated_prior_year_does_not_displace_current_year() -> None:
    """A 10-K filed in 2024 for FY2023 may include restated FY2022 and FY2021
    revenue entries (same accn). The parser must pick the CURRENT year's value,
    not an older one whose `end` doesn't match the filing's reporting period.

    Regression test for the bug found in the AAPL FY2022 row.
    """
    facts = _facts(
        {
            "Revenues": [
                # Restated FY2021 inside the FY2023 10-K (don't pick this).
                _annual("ACC-2023K", "2020-09-27", "2021-09-25", 365_817_000_000, "2023-11-01"),
                # Restated FY2022 inside the same filing (don't pick this either).
                _annual("ACC-2023K", "2021-09-26", "2022-09-24", 394_328_000_000, "2023-11-01"),
                # FY2023 — the current reporting period; this is what we want.
                _annual("ACC-2023K", "2022-09-25", "2023-09-30", 383_285_000_000, "2023-11-01"),
            ]
        }
    )
    rows = parse_companyfacts(facts, 1)
    assert len(rows) == 1
    assert rows[0]["period_end"] == date(2023, 9, 30)
    assert rows[0]["revenue"] == 383_285_000_000


def test_drops_forward_looking_entry_outside_filing_date() -> None:
    """If a filing includes a period whose end is AFTER the filing date, drop
    that entry — it's forward-looking and would create a fake look-ahead row."""
    facts = _facts(
        {
            "Revenues": [
                # Valid current-year entry.
                _annual("ACC-1", "2022-01-01", "2022-12-31", 1000, "2023-02-15"),
                # Forward-looking — end is AFTER filed date.
                _annual("ACC-1", "2023-01-01", "2023-12-31", 999, "2023-02-15"),
            ]
        }
    )
    rows = parse_companyfacts(facts, 1)
    assert len(rows) == 1
    # period_end must be the legitimate one, not the forward-looking one.
    assert rows[0]["period_end"] == date(2022, 12, 31)
    assert rows[0]["revenue"] == 1000


def test_filed_at_is_filing_date_not_period_end() -> None:
    """The dataclass field powering point-in-time joins must be `filed_at`,
    populated from EDGAR's `filed`, NOT from the fiscal `period_end`."""
    facts = _facts(
        {
            "Revenues": [
                # FY2023 reported, but the 10-K wasn't actually filed until Feb 2024.
                _annual("ACC-1", "2023-01-01", "2023-12-31", 100, "2024-02-15"),
            ]
        }
    )
    rows = parse_companyfacts(facts, 1)
    assert rows[0]["filed_at"] == date(2024, 2, 15)
    assert rows[0]["period_end"] == date(2023, 12, 31)
    # The whole point: filed_at is LATER than period_end. Joins that use
    # period_end as the time gate would leak this data into Jan 2024 samples.
    assert rows[0]["filed_at"] > rows[0]["period_end"]
