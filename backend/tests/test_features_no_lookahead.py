"""Look-ahead invariant tests for the feature builder.

These tests must pass before features.py is considered correct.
The critical invariant: fundamentals are joined on filed_at <= sample_date,
NEVER on period_end. Violating this leaks future earnings into past samples.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from backend.ml.features import (
    FEATURE_DIM,
    SEQUENCE_LENGTH,
    SENTIMENT_MAX_FILL_DAYS,
    build_sample,
)

# Column indices in the (252, 12) output matrix
COL_LOG_RETURN = 0
COL_LOG_VOLUME = 1
COL_VOLATILITY = 2
COL_SENT_7D = 3
COL_SENT_14D = 4
COL_REV_GROWTH = 5
COL_GROSS_MARGIN = 6
COL_OP_MARGIN = 7
COL_DEBT_EQUITY = 8
COL_FCF_REVENUE = 9
COL_DAY_SIN = 10
COL_DAY_COS = 11


# =============================================================
# Helpers
# =============================================================


def _trading_dates(start: date, n: int) -> list[date]:
    """Generate n consecutive weekday dates from start."""
    dates: list[date] = []
    d = start
    while len(dates) < n:
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=1)
    return dates


def _make_prices(dates: list[date], base: float = 100.0) -> list[dict]:
    """Deterministic ascending prices (0.1% daily return)."""
    rows = []
    price = base
    for d in dates:
        price *= 1.001
        rows.append(
            {
                "trade_date": d,
                "adj_close": round(price, 6),
                "volume": 1_000_000,
            }
        )
    return rows


def _fund_row(
    ticker_id: int,
    filed_at: date,
    period_end: date,
    gross_margin: float = 0.3,
    operating_margin: float = 0.15,
    revenue: float = 1e9,
    net_income: float = 1e8,
    total_debt: float = 5e8,
    total_equity: float = 2e9,
    fcf: float = 2e8,
) -> dict:
    return {
        "ticker_id": ticker_id,
        "accession_number": f"acc-{filed_at}",
        "filing_type": "10-K",
        "period_end": period_end,
        "filed_at": filed_at,
        "revenue": revenue,
        "net_income": net_income,
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "total_debt": total_debt,
        "total_equity": total_equity,
        "fcf": fcf,
    }


def _sent_row(ticker_id: int, score_date: date, val7: float, val14: float) -> dict:
    return {
        "ticker_id": ticker_id,
        "score_date": score_date,
        "rolling_7d": val7,
        "rolling_14d": val14,
    }


# =============================================================
# Look-ahead: fundamentals filed AFTER sample_end must be excluded
# =============================================================


def test_future_filing_gross_margin_not_in_features() -> None:
    """A filing filed after sample_end must not contribute any feature values."""
    # 253 trading days = 252-day window + 1 day for the first log-return
    start = date(2022, 1, 3)
    all_dates = _trading_dates(start, SEQUENCE_LENGTH + 1)
    sample_end = all_dates[-1]
    prices = _make_prices(all_dates)

    past_filing = _fund_row(
        1,
        filed_at=sample_end - timedelta(days=90),
        period_end=sample_end - timedelta(days=180),
        gross_margin=0.30,
    )
    future_filing = _fund_row(
        1,
        filed_at=sample_end + timedelta(days=10),  # future!
        period_end=sample_end + timedelta(days=1),
        gross_margin=0.99,  # distinctive value — must never appear
    )

    result = build_sample(1, sample_end, prices, [past_filing, future_filing], [])
    assert result is not None
    # The future filing's gross_margin (0.99) must not appear anywhere
    assert not np.any(np.isclose(result[:, COL_GROSS_MARGIN], 0.99, atol=1e-4))
    # The past filing's gross_margin (0.30) should appear for days after filed_at
    filed_idx = len(all_dates) - 1 - 90 // 1  # approximate
    # At minimum the last row should reflect the past filing
    assert np.isclose(result[-1, COL_GROSS_MARGIN], 0.30, atol=1e-4)


def test_future_filing_operating_margin_not_in_features() -> None:
    """Variant: checks operating_margin column specifically."""
    start = date(2022, 1, 3)
    all_dates = _trading_dates(start, SEQUENCE_LENGTH + 1)
    sample_end = all_dates[-1]
    prices = _make_prices(all_dates)

    past = _fund_row(1, filed_at=sample_end - timedelta(days=60),
                     period_end=sample_end - timedelta(days=120), operating_margin=0.12)
    future = _fund_row(1, filed_at=sample_end + timedelta(days=5),
                       period_end=sample_end, operating_margin=0.77)

    result = build_sample(1, sample_end, prices, [past, future], [])
    assert result is not None
    assert not np.any(np.isclose(result[:, COL_OP_MARGIN], 0.77, atol=1e-4))
    assert np.isclose(result[-1, COL_OP_MARGIN], 0.12, atol=1e-4)


def test_filing_on_sample_end_is_included() -> None:
    """A filing filed exactly on sample_end is available that day — include it."""
    start = date(2022, 1, 3)
    all_dates = _trading_dates(start, SEQUENCE_LENGTH + 1)
    sample_end = all_dates[-1]
    prices = _make_prices(all_dates)

    same_day_filing = _fund_row(
        1,
        filed_at=sample_end,
        period_end=sample_end - timedelta(days=90),
        gross_margin=0.55,
    )

    result = build_sample(1, sample_end, prices, [same_day_filing], [])
    assert result is not None
    assert np.isclose(result[-1, COL_GROSS_MARGIN], 0.55, atol=1e-4)


def test_only_future_filings_yields_zero_fundamentals() -> None:
    """If all filings are in the future, fundamental columns must be 0 throughout."""
    start = date(2022, 1, 3)
    all_dates = _trading_dates(start, SEQUENCE_LENGTH + 1)
    sample_end = all_dates[-1]
    prices = _make_prices(all_dates)

    future = _fund_row(1, filed_at=sample_end + timedelta(days=1),
                       period_end=sample_end, gross_margin=0.5)

    result = build_sample(1, sample_end, prices, [future], [])
    assert result is not None
    assert np.all(result[:, COL_GROSS_MARGIN] == 0.0)
    assert np.all(result[:, COL_OP_MARGIN] == 0.0)


# =============================================================
# Point-in-time: fundamentals forward-fill correctly between filings
# =============================================================


def test_fundamental_values_forward_fill_between_filings() -> None:
    """Days between two filings should hold the earlier filing's values."""
    start = date(2022, 1, 3)
    all_dates = _trading_dates(start, SEQUENCE_LENGTH + 1)
    sample_end = all_dates[-1]
    prices = _make_prices(all_dates)

    early = _fund_row(1, filed_at=all_dates[50], period_end=all_dates[30],
                      gross_margin=0.20)
    late = _fund_row(1, filed_at=all_dates[180], period_end=all_dates[160],
                     gross_margin=0.40)

    result = build_sample(1, sample_end, prices, [early, late], [])
    assert result is not None

    # Before early filing (days 0-49 in the window = indices 1-49 of all_dates):
    # the window starts at all_dates[1] (index 0 of the window)
    window_start_idx_in_all = 1  # all_dates[1] is window[0]
    early_filed_window_idx = 50 - window_start_idx_in_all  # index in window

    # Days before early filing: 0
    for i in range(early_filed_window_idx):
        assert result[i, COL_GROSS_MARGIN] == 0.0, f"Day {i} should be 0 (before first filing)"

    # Days from early to late filing: 0.20
    late_filed_window_idx = 180 - window_start_idx_in_all
    for i in range(early_filed_window_idx, late_filed_window_idx):
        assert np.isclose(result[i, COL_GROSS_MARGIN], 0.20, atol=1e-4), (
            f"Day {i} should be 0.20 (early filing)"
        )

    # Days from late filing onwards: 0.40
    for i in range(late_filed_window_idx, SEQUENCE_LENGTH):
        assert np.isclose(result[i, COL_GROSS_MARGIN], 0.40, atol=1e-4), (
            f"Day {i} should be 0.40 (late filing)"
        )


# =============================================================
# Sentiment forward-fill rule
# =============================================================


def test_sentiment_forward_fills_within_limit() -> None:
    """Sentiment within SENTIMENT_MAX_FILL_DAYS should propagate forward."""
    start = date(2022, 1, 3)
    all_dates = _trading_dates(start, SEQUENCE_LENGTH + 1)
    sample_end = all_dates[-1]
    prices = _make_prices(all_dates)

    # Put sentiment 3 calendar days before the last window day
    sentiment_date = sample_end - timedelta(days=3)
    sent = [_sent_row(1, sentiment_date, val7=0.42, val14=0.35)]

    result = build_sample(1, sample_end, prices, [], sent)
    assert result is not None

    # The last window day is exactly 3 days after sentiment_date — should fill
    assert np.isclose(result[-1, COL_SENT_7D], 0.42, atol=1e-4)
    assert np.isclose(result[-1, COL_SENT_14D], 0.35, atol=1e-4)


def test_sentiment_zeros_out_beyond_fill_limit() -> None:
    """Sentiment older than SENTIMENT_MAX_FILL_DAYS days becomes 0."""
    start = date(2022, 1, 3)
    all_dates = _trading_dates(start, SEQUENCE_LENGTH + 1)
    sample_end = all_dates[-1]
    prices = _make_prices(all_dates)

    # 4 calendar days before last window day — one past the fill limit
    sentiment_date = sample_end - timedelta(days=SENTIMENT_MAX_FILL_DAYS + 1)
    sent = [_sent_row(1, sentiment_date, val7=0.99, val14=0.88)]

    result = build_sample(1, sample_end, prices, [], sent)
    assert result is not None
    assert result[-1, COL_SENT_7D] == 0.0
    assert result[-1, COL_SENT_14D] == 0.0


def test_no_sentiment_yields_zeros() -> None:
    start = date(2022, 1, 3)
    all_dates = _trading_dates(start, SEQUENCE_LENGTH + 1)
    sample_end = all_dates[-1]
    prices = _make_prices(all_dates)

    result = build_sample(1, sample_end, prices, [], [])
    assert result is not None
    assert np.all(result[:, COL_SENT_7D] == 0.0)
    assert np.all(result[:, COL_SENT_14D] == 0.0)


# =============================================================
# Output shape and basic properties
# =============================================================


def test_output_shape() -> None:
    start = date(2022, 1, 3)
    all_dates = _trading_dates(start, SEQUENCE_LENGTH + 1)
    sample_end = all_dates[-1]
    prices = _make_prices(all_dates)

    result = build_sample(1, sample_end, prices, [], [])
    assert result is not None
    assert result.shape == (SEQUENCE_LENGTH, FEATURE_DIM)
    assert result.dtype == np.float32


def test_returns_none_if_insufficient_history() -> None:
    """Fewer than SEQUENCE_LENGTH + 1 price rows → None."""
    start = date(2022, 1, 3)
    all_dates = _trading_dates(start, SEQUENCE_LENGTH)  # one short
    sample_end = all_dates[-1]
    prices = _make_prices(all_dates)

    result = build_sample(1, sample_end, prices, [], [])
    assert result is None


def test_seasonal_features_in_range() -> None:
    """sin/cos columns must be in [-1, 1]."""
    start = date(2022, 1, 3)
    all_dates = _trading_dates(start, SEQUENCE_LENGTH + 1)
    sample_end = all_dates[-1]
    prices = _make_prices(all_dates)

    result = build_sample(1, sample_end, prices, [], [])
    assert result is not None
    assert np.all(result[:, COL_DAY_SIN] >= -1.0) and np.all(result[:, COL_DAY_SIN] <= 1.0)
    assert np.all(result[:, COL_DAY_COS] >= -1.0) and np.all(result[:, COL_DAY_COS] <= 1.0)


def test_zscored_returns_have_near_zero_mean() -> None:
    """Z-scored log-returns across the 252-day window should have ~0 mean."""
    start = date(2022, 1, 3)
    all_dates = _trading_dates(start, SEQUENCE_LENGTH + 1)
    sample_end = all_dates[-1]
    prices = _make_prices(all_dates)

    result = build_sample(1, sample_end, prices, [], [])
    assert result is not None
    assert abs(float(np.mean(result[:, COL_LOG_RETURN]))) < 0.01
