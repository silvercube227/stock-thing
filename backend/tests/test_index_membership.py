"""Unit tests for point-in-time index-membership interval construction. No network/DB."""

from __future__ import annotations

from datetime import date

from scripts.seed_index_membership import (
    _parse_added_date,
    build_intervals,
    parse_constituents_with_dates,
)

WINDOW = date(2010, 1, 1)


def test_current_member_uses_observed_addition_date():
    """The whole point of the table: a name added in 2023 must NOT be a member of
    the 2017 cross-sections."""
    intervals, _ = build_intervals(
        added_dates={"NEW": date(2023, 3, 15)},
        removed_at={},
        known_symbols={"NEW"},
    )
    assert intervals["NEW"] == [
        {"valid_from": date(2023, 3, 15), "valid_to": None,
         "source": "wikipedia_date_added"}
    ]


def test_member_added_before_the_window_spans_it():
    intervals, _ = build_intervals(
        added_dates={"OLD": date(1998, 5, 1)},
        removed_at={},
        known_symbols={"OLD"},
    )
    assert intervals["OLD"][0]["valid_from"] == WINDOW
    assert intervals["OLD"][0]["valid_to"] is None
    assert intervals["OLD"][0]["source"] == "assumed_pre_window"


def test_member_with_unparseable_date_spans_the_window():
    intervals, _ = build_intervals(
        added_dates={"HUH": None}, removed_at={}, known_symbols={"HUH"},
    )
    assert intervals["HUH"][0]["source"] == "assumed_pre_window"


def test_removed_name_gets_an_interval_closed_at_removal():
    intervals, _ = build_intervals(
        added_dates={},
        removed_at={"GONE": date(2019, 6, 11)},
        known_symbols={"GONE"},
    )
    assert intervals["GONE"] == [
        {"valid_from": WINDOW, "valid_to": date(2019, 6, 11), "source": "assumed_start"}
    ]


def test_readded_name_gets_two_disjoint_intervals():
    """Removed in 2015, back in the index since 2021: the gap between must be
    excluded, so the name needs two intervals rather than one."""
    intervals, _ = build_intervals(
        added_dates={"BACK": date(2021, 4, 1)},
        removed_at={"BACK": date(2015, 8, 1)},
        known_symbols={"BACK"},
    )
    assert intervals["BACK"] == [
        {"valid_from": WINDOW, "valid_to": date(2015, 8, 1), "source": "assumed_start"},
        {"valid_from": date(2021, 4, 1), "valid_to": None,
         "source": "wikipedia_date_added"},
    ]


def test_unknown_symbols_are_ignored():
    intervals, _ = build_intervals(
        added_dates={"NOTOURS": date(2020, 1, 1)},
        removed_at={"ALSONOT": date(2020, 1, 1)},
        known_symbols=set(),
    )
    assert intervals == {}


def test_active_but_absent_from_index_list_is_closed_at_observation():
    """A removal we never recorded (the changes table is gone). Dropping the name
    would erase years of real membership, so close it at the observation date."""
    intervals, anomalies = build_intervals(
        added_dates={},
        removed_at={},
        known_symbols={"STALE"},
        still_active={"STALE"},
        observed_on=date(2026, 8, 21),
    )
    assert intervals["STALE"] == [
        {"valid_from": WINDOW, "valid_to": date(2026, 8, 21),
         "source": "observed_absent"}
    ]
    assert any("STALE" in a for a in anomalies)


def test_current_member_is_not_overridden_by_the_active_fallback():
    intervals, anomalies = build_intervals(
        added_dates={"LIVE": date(2015, 1, 5)},
        removed_at={},
        known_symbols={"LIVE"},
        still_active={"LIVE"},
        observed_on=date(2026, 8, 21),
    )
    assert intervals["LIVE"][0]["valid_to"] is None
    assert not anomalies


def test_intervals_never_overlap_for_a_symbol():
    intervals, _ = build_intervals(
        added_dates={"BACK": date(2021, 4, 1)},
        removed_at={"BACK": date(2015, 8, 1)},
        known_symbols={"BACK"},
    )
    rows = sorted(intervals["BACK"], key=lambda r: r["valid_from"])
    for earlier, later in zip(rows, rows[1:]):
        assert earlier["valid_to"] is not None
        assert earlier["valid_to"] <= later["valid_from"]


# =============================================================
# Constituents table parsing
# =============================================================


def test_parse_added_date_handles_wikipedia_formats():
    assert _parse_added_date("2019-06-11") == date(2019, 6, 11)
    assert _parse_added_date("June 11, 2019") == date(2019, 6, 11)
    assert _parse_added_date("1957") == date(1957, 1, 1)
    assert _parse_added_date("2019-06-11 (a note)") == date(2019, 6, 11)
    assert _parse_added_date("") is None
    assert _parse_added_date("unknown") is None


def test_parse_constituents_locates_columns_by_header_not_position():
    """The table has gained/lost columns over the years, so a positional parse
    silently reads the wrong field on older revisions."""
    html = """
    <table class="wikitable" id="constituents">
      <tr><th>Symbol</th><th>Security</th><th>GICS Sector</th>
          <th>Headquarters</th><th>Date added</th><th>CIK</th></tr>
      <tr><td>BRK.B</td><td>Berkshire</td><td>Financials</td>
          <td>Omaha</td><td>2010-02-16</td><td>0001067983</td></tr>
      <tr><td>AAPL</td><td>Apple</td><td>Information Technology</td>
          <td>Cupertino</td><td>1982-11-30</td><td>0000320193</td></tr>
    </table>
    """
    out = parse_constituents_with_dates(html)
    assert out["BRK-B"] == date(2010, 2, 16)   # '.' normalized to '-'
    assert out["AAPL"] == date(1982, 11, 30)
