"""Point-in-time LSEG analyst-estimate features + filing-reaction (drift/surprise)."""

from __future__ import annotations

import bisect
import math
from datetime import timedelta

import numpy as np

from backend.ml.dataset import _as_date


def _estimates_context_asof(
    est_rows: list[dict], surprise_rows: list[dict], as_of_dates: list
) -> dict[str, list[float]]:
    """Point-in-time LSEG analyst-estimate features for each as-of date.

    LSEG fields land on different dates (sparse rows), so each field is looked up
    INDEPENDENTLY: the most recent non-null observation with as_of_date <= d.
    Revisions compare against the value ~30/90 calendar days earlier. Forward
    multiples become yields (inverse); non-positive ratios -> 0. Gaps -> 0.

    Surprises come from the QUARTERLY `surprise_rows` (earnings_surprises), anchored
    on report_date: each fiscal quarter's (actual - pre-report consensus)/|consensus|
    carried forward from its report_date — proper quarterly PEAD, not annual.

    Revision-momentum pack: `eps_est_rev_*` is the %Δ in the monthly forward EPS
    consensus (analysts revising estimates); `coverage_chg_90d` the Δ in analyst
    count; `pt_num_estimates` the price-target estimate count level.

    `price_target_mean` is returned as an intermediate (build_ticker_rows turns it
    into price_target_upside with the as-of price); it is not a feature itself.
    """
    keys = ("rec_mean_level", "rec_rev_30d", "rec_rev_90d", "price_target_mean",
            "price_target_rev_90d", "forward_earnings_yield", "forward_ebitda_yield",
            "revenue_surprise", "eps_surprise",
            "eps_est_rev_30d", "eps_est_rev_90d", "coverage_chg_90d", "pt_num_estimates",
            "eps_dispersion")

    snap_fields = ("rec_mean", "price_target_mean", "eps_mean",
                   "fwd_pe", "fwd_ev_ebitda", "num_analysts", "pt_num_estimates",
                   "eps_std_dev")
    series: dict[str, tuple[list, list]] = {f: ([], []) for f in snap_fields}
    for r in sorted(est_rows or [], key=lambda r: _as_date(r["as_of_date"])):
        d = _as_date(r["as_of_date"])
        for f in snap_fields:
            v = r.get(f)
            if v is not None:
                series[f][0].append(d)
                series[f][1].append(float(v))

    # Quarterly surprise per metric: (report_date, surprise) carried forward.
    surp: dict[str, tuple[list, list]] = {"eps": ([], []), "revenue": ([], [])}
    for r in sorted(surprise_rows or [], key=lambda r: _as_date(r["report_date"])):
        rd = _as_date(r["report_date"])
        for metric, acol, ccol in (("eps", "eps_actual", "eps_consensus"),
                                    ("revenue", "rev_actual", "rev_consensus")):
            act, cons = r.get(acol), r.get(ccol)
            if act is not None and cons not in (None, 0):
                surp[metric][0].append(rd)
                surp[metric][1].append((float(act) - float(cons)) / abs(float(cons)))

    def asof(field: str, target) -> float | None:
        dates, vals = series[field]
        i = bisect.bisect_right(dates, target) - 1
        return vals[i] if i >= 0 else None

    def surprise_asof(metric: str, target) -> float | None:
        dates, vals = surp[metric]
        i = bisect.bisect_right(dates, target) - 1
        return vals[i] if i >= 0 else None

    def pct_rev(field: str, d, days: int) -> float:
        cur, prev = asof(field, d), asof(field, d - timedelta(days=days))
        return (cur - prev) / abs(prev) if cur is not None and prev not in (None, 0) else 0.0

    out: dict[str, list[float]] = {k: [] for k in keys}
    for d in as_of_dates:
        d = _as_date(d)
        rec = asof("rec_mean", d)
        rec30 = asof("rec_mean", d - timedelta(days=30))
        rec90 = asof("rec_mean", d - timedelta(days=90))
        pt = asof("price_target_mean", d)
        pt90 = asof("price_target_mean", d - timedelta(days=90))
        pe = asof("fwd_pe", d)
        ev = asof("fwd_ev_ebitda", d)
        na, na90 = asof("num_analysts", d), asof("num_analysts", d - timedelta(days=90))
        ptn = asof("pt_num_estimates", d)

        out["rec_mean_level"].append(rec if rec is not None else 0.0)
        out["rec_rev_30d"].append((rec30 - rec) if rec is not None and rec30 is not None else 0.0)
        out["rec_rev_90d"].append((rec90 - rec) if rec is not None and rec90 is not None else 0.0)
        out["price_target_mean"].append(pt if pt is not None else 0.0)
        out["price_target_rev_90d"].append(
            (pt - pt90) / abs(pt90) if pt is not None and pt90 not in (None, 0) else 0.0)
        out["forward_earnings_yield"].append(1.0 / pe if pe is not None and pe > 0 else 0.0)
        out["forward_ebitda_yield"].append(1.0 / ev if ev is not None and ev > 0 else 0.0)
        out["revenue_surprise"].append(surprise_asof("revenue", d) or 0.0)
        out["eps_surprise"].append(surprise_asof("eps", d) or 0.0)
        out["eps_est_rev_30d"].append(pct_rev("eps_mean", d, 30))
        out["eps_est_rev_90d"].append(pct_rev("eps_mean", d, 90))
        out["coverage_chg_90d"].append((na - na90) if na is not None and na90 is not None else 0.0)
        out["pt_num_estimates"].append(ptn if ptn is not None else 0.0)
        # DMS dispersion: eps_std_dev / max(|eps_mean|, 0.01). Near-zero eps_mean
        # makes the ratio unstable; we zero the feature rather than extrapolate.
        eps_m = asof("eps_mean", d)
        eps_s = asof("eps_std_dev", d)
        if eps_s is not None and eps_m is not None and abs(eps_m) >= 0.01:
            out["eps_dispersion"].append(float(eps_s) / abs(float(eps_m)))
        else:
            out["eps_dispersion"].append(0.0)
    return out


def _earnings_reaction_asof(
    fund_rows: list[dict],
    bar_positions: list[int],
    bar_dates: list,
    trade_dates: list,
    adj_close: list[float | None],
    market_returns: dict | None,
) -> dict[str, list[float]]:
    """Per-grid-date filing-reaction features built from prices + filed_at only.

    For each grid date we look back to the most recent filing on/before that date
    and summarize a few aspects of its market reaction:
      filing_surprise_3d  — abnormal return over [filed_at-1, filed_at+1] trading
                            days (proxy for what the market thought of the print).
      filing_drift_30d    — abnormal return from the trading day after filed_at
                            out to +30 trading days (or up to the grid date, if
                            fewer days have elapsed). Post-earnings drift signal.
      filings_recency_days — calendar days since the latest filing.
      filings_in_90d      — count of filings within the trailing 90 calendar days.

    PIT-safe by construction: we only ever index `adj_close` up to `pos` (the
    grid-date bar position).
    """
    n = len(bar_dates)
    out: dict[str, list[float]] = {
        "filing_drift_30d": [0.0] * n,
        "filing_surprise_3d": [0.0] * n,
        "filings_recency_days": [0.0] * n,
        "filings_in_90d": [0.0] * n,
    }
    if not fund_rows or not trade_dates:
        return out

    filings_sorted = sorted(fund_rows, key=lambda r: _as_date(r["filed_at"]))
    filed_ats = [_as_date(r["filed_at"]) for r in filings_sorted]

    def _sum_mkt(start_idx: int, end_idx: int) -> float:
        if market_returns is None or end_idx <= start_idx:
            return 0.0
        total = 0.0
        for k in range(start_idx + 1, end_idx + 1):
            r = market_returns.get(trade_dates[k])
            if r is not None and np.isfinite(r):
                total += float(r)
        return total

    for j, g in enumerate(bar_dates):
        pos = bar_positions[j]
        idx = bisect.bisect_right(filed_ats, g) - 1
        if idx < 0:
            continue
        filed_at = filed_ats[idx]
        # First trading day on or after filed_at (bisect_left returns the next
        # bar when filed_at falls on a weekend/holiday; if equal to a trading
        # day, that day itself is selected). Bound by pos so we never peek past
        # the grid-date bar — PIT guard.
        file_pos = bisect.bisect_left(trade_dates, filed_at)
        if file_pos > pos:
            # Filing recorded ahead of the price series for this grid date — can't
            # measure reaction yet; recency + count are still valid.
            out["filings_recency_days"][j] = float((g - filed_at).days)
            cutoff = g - timedelta(days=90)
            lo = bisect.bisect_left(filed_ats, cutoff)
            out["filings_in_90d"][j] = float(idx + 1 - lo)
            continue

        out["filings_recency_days"][j] = float((g - filed_at).days)
        cutoff = g - timedelta(days=90)
        lo = bisect.bisect_left(filed_ats, cutoff)
        out["filings_in_90d"][j] = float(idx + 1 - lo)

        # 3-day surprise window: log return [file_pos-1, file_pos+1], minus market.
        start_s = max(file_pos - 1, 0)
        end_s = min(file_pos + 1, pos)
        if end_s > start_s:
            p_a = adj_close[start_s]
            p_b = adj_close[end_s]
            if p_a and p_b and p_a > 0 and p_b > 0:
                out["filing_surprise_3d"][j] = float(
                    math.log(p_b / p_a) - _sum_mkt(start_s, end_s)
                )

        # Post-filing drift: [file_pos+1, file_pos+31] or shorter if too fresh.
        start_d = file_pos + 1
        end_d = min(start_d + 30, pos)
        # Require ≥ 5 trading days of post-filing data so the value isn't noise.
        if start_d < len(adj_close) and end_d - start_d >= 5:
            p_a = adj_close[start_d]
            p_b = adj_close[end_d]
            if p_a and p_b and p_a > 0 and p_b > 0:
                out["filing_drift_30d"][j] = float(
                    math.log(p_b / p_a) - _sum_mkt(start_d, end_d)
                )

    return out
