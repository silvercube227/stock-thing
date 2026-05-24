"""Point-in-time-correct feature assembly for the PatchTST transformer.

`build_sample` is the public API. It takes raw DB rows (as plain dicts) and
produces a (SEQUENCE_LENGTH, FEATURE_DIM) float32 array. No I/O, no DB calls —
those belong in the training and inference scripts.

Feature layout (12 columns):
  0  log_return_z      — log(adj_close_t / adj_close_{t-1}), 252d z-score
  1  log_volume_z      — log(volume), 252d z-score
  2  volatility_20d    — rolling 20d std of log returns
  3  sentiment_7d      — rolling_7d from sentiment_daily, fill-forward ≤3 days else 0
  4  sentiment_14d     — rolling_14d, same fill rule
  5  revenue_growth    — YoY revenue growth from most recent eligible filing
  6  gross_margin      — from most recent filing with filed_at <= day
  7  operating_margin  — same
  8  debt_equity       — total_debt / total_equity
  9  fcf_revenue       — fcf / revenue
 10  day_sin           — sin(2π · yday / 365.25)
 11  day_cos           — cos(2π · yday / 365.25)

CRITICAL: fundamentals are joined on filed_at <= sample_date, NEVER period_end.
The look-ahead tests in test_features_no_lookahead.py enforce this invariant.
"""

from __future__ import annotations

import bisect
import math
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np

FEATURE_DIM: int = 12
SEQUENCE_LENGTH: int = 252       # one year of trading days
SENTIMENT_MAX_FILL_DAYS: int = 3  # zero-fill beyond this many calendar days


# =============================================================
# Internal helpers
# =============================================================


def _to_date(val: Any) -> date:
    """Coerce date or datetime to date."""
    if isinstance(val, datetime):
        return val.date()
    return val


def _zscore(arr: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalization. Returns zeros if std is near 0."""
    std = arr.std()
    if std < 1e-9:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# =============================================================
# Fundamentals helpers
# =============================================================


def _prior_year_revenue(
    sorted_filings: list[dict], current_period_end: date, filing_type: str
) -> float | None:
    """Find revenue from a filing of the same type whose period_end is ≈ 1 year prior."""
    target = date(
        current_period_end.year - 1,
        current_period_end.month,
        current_period_end.day,
    )
    best: dict | None = None
    best_delta = 999
    for f in sorted_filings:
        if f.get("filing_type") != filing_type:
            continue
        delta = abs((_to_date(f["period_end"]) - target).days)
        if delta <= 30 and delta < best_delta:
            best = f
            best_delta = delta
    if best is None or best.get("revenue") is None:
        return None
    return float(best["revenue"])


def _annotate_fundamentals(fund_rows: list[dict]) -> list[dict]:
    """Pre-compute derived features for each filing row.

    Returns a new list sorted ascending by filed_at. Each dict has:
      filed_at (date), revenue_growth, gross_margin, operating_margin,
      debt_equity, fcf_revenue.
    """
    rows_sorted = sorted(fund_rows, key=lambda r: _to_date(r["filed_at"]))

    annotated: list[dict] = []
    for row in rows_sorted:
        revenue = _safe_float(row.get("revenue"), default=0.0) or None
        period_end = _to_date(row["period_end"])

        prior_rev = _prior_year_revenue(rows_sorted, period_end, row.get("filing_type", ""))
        if revenue and prior_rev and abs(prior_rev) > 1e-9:
            rev_growth = (revenue - prior_rev) / abs(prior_rev)
        else:
            rev_growth = 0.0

        equity = _safe_float(row.get("total_equity"), default=0.0)
        debt = _safe_float(row.get("total_debt"), default=0.0)
        debt_equity = (debt / equity) if abs(equity) > 1e-9 else 0.0

        fcf = _safe_float(row.get("fcf"), default=0.0)
        fcf_revenue = (fcf / revenue) if (revenue and abs(revenue) > 1e-9) else 0.0

        annotated.append(
            {
                "filed_at": _to_date(row["filed_at"]),
                "revenue_growth": rev_growth,
                "gross_margin": _safe_float(row.get("gross_margin")),
                "operating_margin": _safe_float(row.get("operating_margin")),
                "debt_equity": debt_equity,
                "fcf_revenue": fcf_revenue,
            }
        )
    return annotated


def _build_fundamental_series(
    window_dates: list[date], fund_rows: list[dict]
) -> np.ndarray:
    """Shape: (len(window_dates), 5).

    For each window date, finds the most recent filing with filed_at <= date
    and returns its feature values. Rows with no eligible filing get 0.
    """
    annotated = _annotate_fundamentals(fund_rows)
    # filed_at keys for bisect
    filed_ats = [f["filed_at"] for f in annotated]

    n = len(window_dates)
    out = np.zeros((n, 5), dtype=np.float32)
    for i, d in enumerate(window_dates):
        # bisect_right gives insertion point after any equal date
        idx = bisect.bisect_right(filed_ats, d) - 1
        if idx < 0:
            continue
        f = annotated[idx]
        out[i, 0] = f["revenue_growth"]
        out[i, 1] = f["gross_margin"]
        out[i, 2] = f["operating_margin"]
        out[i, 3] = f["debt_equity"]
        out[i, 4] = f["fcf_revenue"]
    return out


# =============================================================
# Sentiment helpers
# =============================================================


def _build_sentiment_series(
    window_dates: list[date],
    sentiment_rows: list[dict],
    max_fill_days: int = SENTIMENT_MAX_FILL_DAYS,
) -> np.ndarray:
    """Shape: (len(window_dates), 2) — columns [rolling_7d, rolling_14d].

    For each window date d, looks back up to max_fill_days calendar days for
    the most recent sentiment record. If none found within the window, 0.
    """
    sent_map: dict[date, tuple[float, float]] = {}
    for r in sentiment_rows:
        sd = _to_date(r["score_date"])
        sent_map[sd] = (
            _safe_float(r.get("rolling_7d")),
            _safe_float(r.get("rolling_14d")),
        )

    n = len(window_dates)
    out = np.zeros((n, 2), dtype=np.float32)
    for i, d in enumerate(window_dates):
        for delta in range(max_fill_days + 1):
            key = d - timedelta(days=delta)
            if key in sent_map:
                out[i, 0], out[i, 1] = sent_map[key]
                break
    return out


# =============================================================
# Public API
# =============================================================


def build_sample(
    ticker_id: int,
    sample_end: date,
    price_rows: list[dict],
    fundamental_rows: list[dict],
    sentiment_rows: list[dict],
) -> np.ndarray | None:
    """Build a (SEQUENCE_LENGTH, FEATURE_DIM) float32 feature matrix.

    Args:
        ticker_id:        Ticker identifier (unused in computation; reserved for
                          future per-ticker dispatch).
        sample_end:       Last date of the 252-day input window.
        price_rows:       All available price rows for this ticker, each with
                          keys: trade_date (date), adj_close (float), volume (int).
                          Need at least SEQUENCE_LENGTH + 1 rows ending at sample_end.
        fundamental_rows: All filing rows for this ticker. Rows with
                          filed_at > sample_end are silently excluded (point-in-time).
        sentiment_rows:   sentiment_daily rows for this ticker. Forward-filled
                          ≤ SENTIMENT_MAX_FILL_DAYS calendar days, else 0.

    Returns:
        float32 ndarray of shape (252, 12), or None if there are not enough
        price rows ending at sample_end.
    """
    # --- Price window ---
    prices = sorted(price_rows, key=lambda r: _to_date(r["trade_date"]))
    dates = [_to_date(r["trade_date"]) for r in prices]

    try:
        end_idx = dates.index(sample_end)
    except ValueError:
        return None

    # Need end_idx >= SEQUENCE_LENGTH so we have one row before the window
    # for the first log-return calculation.
    if end_idx < SEQUENCE_LENGTH:
        return None

    # The window is prices[end_idx - SEQUENCE_LENGTH + 1 : end_idx + 1]
    # The previous close (for the first return) is prices[end_idx - SEQUENCE_LENGTH]
    prev_row = prices[end_idx - SEQUENCE_LENGTH]
    window = prices[end_idx - SEQUENCE_LENGTH + 1 : end_idx + 1]
    window_dates = [_to_date(r["trade_date"]) for r in window]

    # --- Log returns and log volume ---
    log_returns = np.empty(SEQUENCE_LENGTH, dtype=np.float64)
    log_volumes = np.empty(SEQUENCE_LENGTH, dtype=np.float64)

    prev_close = _safe_float(prev_row["adj_close"], default=1.0)
    for i, row in enumerate(window):
        cur = _safe_float(row["adj_close"], default=prev_close)
        log_returns[i] = math.log(cur / prev_close) if prev_close > 0 else 0.0
        prev_close = cur

        vol = _safe_float(row.get("volume"), default=1.0)
        log_volumes[i] = math.log(max(vol, 1.0))

    log_returns_z = _zscore(log_returns).astype(np.float32)
    log_volumes_z = _zscore(log_volumes).astype(np.float32)

    # --- 20-day realized volatility ---
    volatility = np.array(
        [np.std(log_returns[max(0, i - 19) : i + 1]) for i in range(SEQUENCE_LENGTH)],
        dtype=np.float32,
    )

    # --- Sentiment (forward-fill ≤3 calendar days) ---
    sentiment = _build_sentiment_series(window_dates, sentiment_rows)

    # --- Fundamentals (point-in-time: only filed_at <= each window date) ---
    fundamentals = _build_fundamental_series(window_dates, fundamental_rows)

    # --- Seasonal ---
    day_sin = np.array(
        [math.sin(2 * math.pi * d.timetuple().tm_yday / 365.25) for d in window_dates],
        dtype=np.float32,
    )
    day_cos = np.array(
        [math.cos(2 * math.pi * d.timetuple().tm_yday / 365.25) for d in window_dates],
        dtype=np.float32,
    )

    # --- Stack (252, 12) ---
    return np.stack(
        [
            log_returns_z,          # 0
            log_volumes_z,          # 1
            volatility,             # 2
            sentiment[:, 0],        # 3  rolling_7d
            sentiment[:, 1],        # 4  rolling_14d
            fundamentals[:, 0],     # 5  revenue_growth
            fundamentals[:, 1],     # 6  gross_margin
            fundamentals[:, 2],     # 7  operating_margin
            fundamentals[:, 3],     # 8  debt_equity
            fundamentals[:, 4],     # 9  fcf_revenue
            day_sin,                # 10
            day_cos,                # 11
        ],
        axis=1,
    ).astype(np.float32)
