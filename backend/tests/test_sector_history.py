"""Interval construction for the point-in-time GICS sector history.

`tickers.sector` is one current label applied to a name's whole history, and it sets
both the training target (`sector_return` / `sector_grade`) and the headline metric
(`within_sector_ic`). These tests pin the merge that turns quarterly Wikipedia
snapshots into intervals — the piece that decides which peer group a 2012 row is
demeaned against.
"""

from __future__ import annotations

from datetime import date

from scripts.seed_sector_history import build_intervals, quarterly_dates

FLOOR = date(2010, 1, 1)


def test_unchanged_label_collapses_to_one_interval():
    obs = [(date(2012, 1, 1), "Information Technology", "Software"),
           (date(2015, 1, 1), "Information Technology", "Software"),
           (date(2020, 1, 1), "Information Technology", "Software")]
    out = build_intervals(obs, FLOOR, ("Information Technology", "Software"))
    assert len(out) == 1
    assert out[0]["valid_from"] == FLOOR          # back-filled to the panel floor
    assert out[0]["valid_to"] is None             # still current
    assert out[0]["source"] == "assumed_pre_first_snapshot"


def test_2018_communication_services_reshuffle_splits_the_history():
    """The concrete case this table exists for: GOOGL/META/NFLX/DIS moved into
    Communication Services in Sept 2018, so a 2012 row must NOT be demeaned against
    their post-2018 peers."""
    obs = [(date(2012, 1, 1), "Information Technology", None),
           (date(2018, 1, 1), "Information Technology", None),
           (date(2019, 1, 1), "Communication Services", None),
           (date(2024, 1, 1), "Communication Services", None)]
    out = build_intervals(obs, FLOOR, ("Communication Services", None))
    assert [iv["sector"] for iv in out] == ["Information Technology", "Communication Services"]
    assert out[0]["valid_from"] == FLOOR
    # Exclusive upper bound meets the next interval's lower bound exactly — no gap,
    # no overlap, so an as-of lookup is unambiguous on the boundary date.
    assert out[0]["valid_to"] == out[1]["valid_from"] == date(2019, 1, 1)
    assert out[1]["valid_to"] is None


def test_never_observed_name_falls_back_to_the_tickers_row():
    out = build_intervals([], FLOOR, ("Health Care", "Biotech"))
    assert len(out) == 1
    assert out[0]["source"] == "tickers_only"
    assert out[0]["sector"] == "Health Care"
    assert out[0]["valid_from"] == FLOOR and out[0]["valid_to"] is None


def test_no_sector_anywhere_yields_no_intervals():
    # Better to write nothing than to invent a peer group.
    assert build_intervals([], FLOOR, (None, None)) == []


def test_live_tickers_row_wins_after_the_last_snapshot():
    """A reclassification made after our newest snapshot still has to show up."""
    obs = [(date(2012, 1, 1), "Consumer Discretionary", None),
           (date(2024, 1, 1), "Consumer Discretionary", None)]
    out = build_intervals(obs, FLOOR, ("Consumer Staples", None))
    assert [iv["sector"] for iv in out] == ["Consumer Discretionary", "Consumer Staples"]
    assert out[-1]["source"] == "tickers_current"
    assert out[-1]["valid_from"] > date(2024, 1, 1)
    assert out[0]["valid_to"] == out[1]["valid_from"]


def test_intervals_never_overlap_and_stay_ordered():
    obs = [(date(2011, 1, 1), "Financials", None),
           (date(2016, 1, 1), "Real Estate", None),      # the 2016 Real Estate split
           (date(2021, 1, 1), "Financials", None)]
    out = build_intervals(obs, FLOOR, ("Financials", None))
    for a, b in zip(out, out[1:], strict=False):
        assert a["valid_to"] == b["valid_from"], "intervals must abut exactly"
        assert a["valid_from"] < a["valid_to"]
    assert out[-1]["valid_to"] is None


def test_quarterly_dates_span_the_panel_and_end_today():
    ds = quarterly_dates(date(2010, 1, 1), date(2026, 9, 6))
    assert ds[0] == date(2010, 1, 1)
    assert ds[-1] == date(2026, 9, 6)          # always includes the end date itself
    assert all(a < b for a, b in zip(ds, ds[1:], strict=False))
    assert 60 < len(ds) < 80                    # ~4/yr over ~17 years
