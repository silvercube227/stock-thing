"""Point-in-time / no-lookahead tests for the LSEG estimate feature computation.

_estimates_context_asof must (a) look up each field independently at its most
recent non-null observation <= the grid date (LSEG fields land on different
dates), (b) never see a snapshot dated after the grid date, and (c) compute
revenue surprise from the consensus BEFORE the report date.
"""
from __future__ import annotations

from datetime import date

import pytest

from backend.ml.gbm_baseline import _estimates_context_asof


def _rows() -> list[dict]:
    # Each field deliberately lands on its own date (the real sparsity pattern).
    return [
        {"as_of_date": date(2020, 1, 15), "rec_mean": 3.0, "price_target_mean": None,
         "revenue_mean": None, "revenue_actual": None, "fwd_pe": None, "fwd_ev_ebitda": None},
        {"as_of_date": date(2020, 2, 15), "rec_mean": None, "price_target_mean": 100.0,
         "revenue_mean": None, "revenue_actual": None, "fwd_pe": None, "fwd_ev_ebitda": None},
        {"as_of_date": date(2020, 3, 15), "rec_mean": 2.0, "price_target_mean": None,
         "revenue_mean": 1000.0, "revenue_actual": None, "fwd_pe": 20.0, "fwd_ev_ebitda": 10.0},
        {"as_of_date": date(2020, 3, 20), "rec_mean": None, "price_target_mean": None,
         "revenue_mean": None, "revenue_actual": 1100.0, "fwd_pe": None, "fwd_ev_ebitda": None},
        {"as_of_date": date(2020, 6, 15), "rec_mean": 2.5, "price_target_mean": 120.0,
         "revenue_mean": None, "revenue_actual": None, "fwd_pe": None, "fwd_ev_ebitda": None},
    ]


def test_per_field_asof_and_revisions():
    ctx = _estimates_context_asof(_rows(), [date(2020, 4, 30)])
    # rec: latest <= 4/30 is the 3/15 value (2.0); 90d-ago (~1/31) is the 1/15 value (3.0).
    assert ctx["rec_mean_level"][0] == 2.0
    assert ctx["rec_rev_30d"][0] == 0.0
    assert ctx["rec_rev_90d"][0] == 1.0                    # 3.0 - 2.0 (upgrade reads positive)
    assert ctx["forward_earnings_yield"][0] == pytest.approx(0.05)   # 1/20
    assert ctx["forward_ebitda_yield"][0] == pytest.approx(0.10)     # 1/10
    # surprise: report 3/20 actual 1100 vs pre-report consensus 1000 (3/15).
    assert ctx["revenue_surprise"][0] == pytest.approx(0.10)


def test_no_lookahead():
    # As of 2/1: must NOT see the 2/15 target, 3/15 rec, or 3/20 report.
    ctx = _estimates_context_asof(_rows(), [date(2020, 2, 1)])
    assert ctx["rec_mean_level"][0] == 3.0                 # 1/15, not the future 3/15
    assert ctx["price_target_mean"][0] == 0.0              # 2/15 target unseen
    assert ctx["forward_earnings_yield"][0] == 0.0
    assert ctx["revenue_surprise"][0] == 0.0


def test_empty_estimates_all_zero():
    ctx = _estimates_context_asof([], [date(2020, 4, 30), date(2021, 1, 1)])
    assert all(v == 0.0 for vals in ctx.values() for v in vals)
    assert all(len(vals) == 2 for vals in ctx.values())
