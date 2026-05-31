"""Point-in-time / no-lookahead tests for the LSEG estimate feature computation.

_estimates_context_asof must (a) look up each monthly field independently at its
most recent non-null observation <= the grid date (LSEG fields land on different
dates), (b) never see a snapshot dated after the grid date, and (c) compute
quarterly surprises from earnings_surprises anchored on report_date, never before.
"""
from __future__ import annotations

from datetime import date

import pytest

from backend.ml.gbm_baseline import _estimates_context_asof


def _rows() -> list[dict]:
    # Each monthly field deliberately lands on its own date (the real sparsity
    # pattern). eps_mean/num_analysts/pt_num_estimates feed the revision-momentum
    # pack; eps_mean rises 4.0 -> 5.0 and coverage 10 -> 12 across the window.
    return [
        {"as_of_date": date(2020, 1, 15), "rec_mean": 3.0, "price_target_mean": None,
         "eps_mean": 4.0, "fwd_pe": None, "fwd_ev_ebitda": None,
         "num_analysts": 10.0, "pt_num_estimates": None},
        {"as_of_date": date(2020, 2, 15), "rec_mean": None, "price_target_mean": 100.0,
         "eps_mean": None, "fwd_pe": None, "fwd_ev_ebitda": None,
         "num_analysts": None, "pt_num_estimates": None},
        {"as_of_date": date(2020, 3, 15), "rec_mean": 2.0, "price_target_mean": None,
         "eps_mean": 5.0, "fwd_pe": 20.0, "fwd_ev_ebitda": 10.0,
         "num_analysts": 12.0, "pt_num_estimates": 8.0},
        {"as_of_date": date(2020, 6, 15), "rec_mean": 2.5, "price_target_mean": 120.0,
         "eps_mean": None, "fwd_pe": None, "fwd_ev_ebitda": None,
         "num_analysts": None, "pt_num_estimates": None},
    ]


def _surprises() -> list[dict]:
    # One fiscal quarter, reported 3/20: EPS 5.5 vs pre-report consensus 5.0 and
    # revenue 1100 vs 1000 -> both +10% surprises, anchored on report_date.
    return [
        {"report_date": date(2020, 3, 20), "period_end": date(2020, 2, 29),
         "eps_consensus": 5.0, "eps_actual": 5.5,
         "rev_consensus": 1000.0, "rev_actual": 1100.0},
    ]


def test_per_field_asof_and_revisions():
    ctx = _estimates_context_asof(_rows(), _surprises(), [date(2020, 4, 30)])
    # rec: latest <= 4/30 is the 3/15 value (2.0); 90d-ago (~1/31) is the 1/15 value (3.0).
    assert ctx["rec_mean_level"][0] == 2.0
    assert ctx["rec_rev_30d"][0] == 0.0
    assert ctx["rec_rev_90d"][0] == 1.0                    # 3.0 - 2.0 (upgrade reads positive)
    assert ctx["forward_earnings_yield"][0] == pytest.approx(0.05)   # 1/20
    assert ctx["forward_ebitda_yield"][0] == pytest.approx(0.10)     # 1/10
    # quarterly surprise: report 3/20 actual vs stored pre-report consensus.
    assert ctx["revenue_surprise"][0] == pytest.approx(0.10)
    assert ctx["eps_surprise"][0] == pytest.approx(0.10)
    # revision momentum: eps_mean 4.0 (1/15, ~90d ago) -> 5.0 (3/15, current) = +25%.
    assert ctx["eps_est_rev_90d"][0] == pytest.approx(0.25)
    assert ctx["eps_est_rev_30d"][0] == 0.0                # 5.0 vs 5.0 (both 3/15)
    assert ctx["coverage_chg_90d"][0] == pytest.approx(2.0)  # 12 - 10
    assert ctx["pt_num_estimates"][0] == 8.0


def test_no_lookahead():
    # As of 2/1: must NOT see the 2/15 target, 3/15 rec/eps, or 3/20 report.
    ctx = _estimates_context_asof(_rows(), _surprises(), [date(2020, 2, 1)])
    assert ctx["rec_mean_level"][0] == 3.0                 # 1/15, not the future 3/15
    assert ctx["price_target_mean"][0] == 0.0              # 2/15 target unseen
    assert ctx["forward_earnings_yield"][0] == 0.0
    assert ctx["revenue_surprise"][0] == 0.0
    assert ctx["eps_surprise"][0] == 0.0                   # 3/20 report unseen
    assert ctx["eps_est_rev_90d"][0] == 0.0                # only one eps_mean seen (4.0)
    assert ctx["coverage_chg_90d"][0] == 0.0               # only one coverage point seen


def test_empty_all_zero():
    ctx = _estimates_context_asof([], [], [date(2020, 4, 30), date(2021, 1, 1)])
    assert all(v == 0.0 for vals in ctx.values() for v in vals)
    assert all(len(vals) == 2 for vals in ctx.values())
