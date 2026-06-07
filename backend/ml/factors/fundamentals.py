"""Point-in-time SEC-EDGAR fundamental snapshots (TTM roll-ups + valuation/quality)."""

from __future__ import annotations

import bisect

import numpy as np

from backend.ml.dataset import _as_date
from backend.ml.features import _annotate_fundamentals


def _ttm_net_income_asof(fund_rows: list[dict], as_of_dates: list) -> list[float]:
    """Most recent point-in-time TTM net income for each as-of date.

    10-K rows contribute their annual `net_income` directly. 10-Q rows use the
    trailing four quarterly `net_income` values when available; otherwise we
    fall back to the most recent annual filing already on file.
    """
    rows_sorted = sorted(fund_rows, key=lambda r: _as_date(r["filed_at"]))
    annotated: list[dict] = []
    quarterly_history: list[tuple] = []
    annual_history: list[tuple] = []

    for row in rows_sorted:
        filed_at = _as_date(row["filed_at"])
        period_end = _as_date(row["period_end"])
        try:
            net_income = float(row["net_income"]) if row.get("net_income") is not None else None
        except (TypeError, ValueError):
            net_income = None

        filing_type = row.get("filing_type")
        ttm_net_income: float | None = None

        if filing_type == "10-Q" and net_income is not None:
            quarterly_history.append((period_end, net_income))
            recent: list[tuple] = []
            seen_periods: set = set()
            for pe, ni in reversed(quarterly_history):
                if pe in seen_periods:
                    continue
                seen_periods.add(pe)
                recent.append((pe, ni))
                if len(recent) == 4:
                    break
            if len(recent) == 4 and (period_end - recent[-1][0]).days <= 380:
                ttm_net_income = float(sum(ni for _pe, ni in recent))
        elif filing_type == "10-K" and net_income is not None:
            annual_history.append((period_end, net_income))
            ttm_net_income = net_income

        if ttm_net_income is None:
            for pe, ni in reversed(annual_history):
                if abs((period_end - pe).days) <= 380:
                    ttm_net_income = float(ni)
                    break

        annotated.append({"filed_at": filed_at, "ttm_net_income": float(ttm_net_income or 0.0)})

    filed_ats = [r["filed_at"] for r in annotated]
    out: list[float] = []
    for d in as_of_dates:
        idx = bisect.bisect_right(filed_ats, d) - 1
        out.append(float(annotated[idx]["ttm_net_income"]) if idx >= 0 else 0.0)
    return out


def _fundamental_context_asof(fund_rows: list[dict], as_of_dates: list) -> dict[str, list[float]]:
    """Point-in-time valuation + quality snapshots for each as-of date."""
    rows_sorted = sorted(fund_rows, key=lambda r: _as_date(r["filed_at"]))
    if not rows_sorted:
        return {
            "ttm_revenue": [0.0] * len(as_of_dates),
            "ttm_net_income": [0.0] * len(as_of_dates),
            "ttm_fcf": [0.0] * len(as_of_dates),
            "total_equity": [0.0] * len(as_of_dates),
            "gross_margin": [0.0] * len(as_of_dates),
            "operating_margin": [0.0] * len(as_of_dates),
            "revenue_growth": [0.0] * len(as_of_dates),
            "gross_margin_stability_4q": [0.0] * len(as_of_dates),
            "operating_margin_stability_4q": [0.0] * len(as_of_dates),
            "revenue_growth_stability_4q": [0.0] * len(as_of_dates),
        }

    annotated_core = _annotate_fundamentals(rows_sorted)
    annotated: list[dict] = []
    quarter_hist: dict[str, list[tuple]] = {"revenue": [], "net_income": [], "fcf": []}
    annual_hist: dict[str, list[tuple]] = {"revenue": [], "net_income": [], "fcf": []}

    def _safe_num(val) -> float | None:
        try:
            return float(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    def _compute_ttm(metric: str, period_end, filing_type: str) -> float:
        q_hist = quarter_hist[metric]
        a_hist = annual_hist[metric]
        ttm_val: float | None = None
        if filing_type == "10-Q":
            recent: list[tuple] = []
            seen_periods: set = set()
            for pe, v in reversed(q_hist):
                if pe in seen_periods:
                    continue
                seen_periods.add(pe)
                recent.append((pe, v))
                if len(recent) == 4:
                    break
            if len(recent) == 4 and (period_end - recent[-1][0]).days <= 380:
                ttm_val = float(sum(v for _pe, v in recent))
        elif filing_type == "10-K" and a_hist:
            ttm_val = float(a_hist[-1][1])

        if ttm_val is None:
            for pe, v in reversed(a_hist):
                if abs((period_end - pe).days) <= 380:
                    ttm_val = float(v)
                    break
        return float(ttm_val or 0.0)

    def _stability(field: str) -> float:
        vals = [float(r[field]) for r in annotated[-4:] if field in r]
        return float(-np.std(np.asarray(vals, dtype=float))) if len(vals) >= 2 else 0.0

    for row, core in zip(rows_sorted, annotated_core, strict=False):
        period_end = _as_date(row["period_end"])
        filing_type = str(row.get("filing_type") or "")
        for metric in ("revenue", "net_income", "fcf"):
            val = _safe_num(row.get(metric))
            if val is None:
                continue
            if filing_type == "10-Q":
                quarter_hist[metric].append((period_end, val))
            elif filing_type == "10-K":
                annual_hist[metric].append((period_end, val))

        current = {
            "filed_at": _as_date(row["filed_at"]),
            "ttm_revenue": _compute_ttm("revenue", period_end, filing_type),
            "ttm_net_income": _compute_ttm("net_income", period_end, filing_type),
            "ttm_fcf": _compute_ttm("fcf", period_end, filing_type),
            "total_equity": float(_safe_num(row.get("total_equity")) or 0.0),
            "gross_margin": float(core["gross_margin"]),
            "operating_margin": float(core["operating_margin"]),
            "revenue_growth": float(core["revenue_growth"]),
        }
        annotated.append(current)
        current["gross_margin_stability_4q"] = _stability("gross_margin")
        current["operating_margin_stability_4q"] = _stability("operating_margin")
        current["revenue_growth_stability_4q"] = _stability("revenue_growth")

    filed_ats = [r["filed_at"] for r in annotated]
    out = {
        "ttm_revenue": [],
        "ttm_net_income": [],
        "ttm_fcf": [],
        "total_equity": [],
        "gross_margin": [],
        "operating_margin": [],
        "revenue_growth": [],
        "gross_margin_stability_4q": [],
        "operating_margin_stability_4q": [],
        "revenue_growth_stability_4q": [],
    }
    for d in as_of_dates:
        idx = bisect.bisect_right(filed_ats, d) - 1
        snap = annotated[idx] if idx >= 0 else None
        for k in out:
            out[k].append(float(snap[k]) if snap is not None else 0.0)
    return out
