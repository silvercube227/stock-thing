"""Pure-parser tests for LSEG estimate ingestion (no Workspace / no DB).

parse_history maps a get_history frame (columns named by LSEG display labels, as
confirmed by scripts/_probe_lseg.py) into raw upsert rows. We pin the column
mapping, NaN->None coercion, and all-null-row skipping.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from backend.ingestion.estimates import DB_FIELDS, parse_history

# Exact display labels the probe returned, including trailing qualifiers.
_LABELS = {
    "rec_mean": "Recommendation - Mean (1-5)",
    "price_target_mean": "Price Target - Mean",
    "revenue_mean": "Revenue - Mean",
    "revenue_actual": "Revenue - Actual",
    "fwd_pe": "Forward P/E (Daily Time Series Ratio)",
    "fwd_ev_ebitda": "Forward Enterprise Value To EBITDA (Daily Time Series Ratio)",
}


def test_parse_history_maps_known_columns_and_dates():
    idx = pd.to_datetime(["2020-01-31", "2020-02-29"])
    df = pd.DataFrame({
        _LABELS["rec_mean"]: [2.1, 2.0],
        _LABELS["price_target_mean"]: [150.0, 155.0],
        _LABELS["revenue_mean"]: [1000.0, 1010.0],
        _LABELS["revenue_actual"]: [float("nan"), 1005.0],  # sparse report-date col
        _LABELS["fwd_pe"]: [25.0, 26.0],
        _LABELS["fwd_ev_ebitda"]: [18.0, 19.0],
    }, index=idx)

    rows = parse_history(df, ticker_id=7)

    assert len(rows) == 2
    assert set(rows[0]) == {"ticker_id", "as_of_date", *DB_FIELDS}
    assert rows[0]["ticker_id"] == 7
    assert rows[0]["as_of_date"] == date(2020, 1, 31)
    assert rows[0]["rec_mean"] == 2.1
    assert rows[0]["fwd_ev_ebitda"] == 18.0
    assert rows[0]["fwd_pe"] == 25.0
    assert rows[0]["revenue_actual"] is None       # NaN -> None
    assert rows[1]["revenue_actual"] == 1005.0      # populated on the report-date row


def test_parse_history_skips_all_null_rows():
    idx = pd.to_datetime(["2020-01-31", "2020-02-29"])
    nan = float("nan")
    df = pd.DataFrame({
        _LABELS["rec_mean"]: [2.1, nan],
        _LABELS["price_target_mean"]: [150.0, nan],
        _LABELS["revenue_mean"]: [1000.0, nan],
        _LABELS["revenue_actual"]: [nan, nan],
        _LABELS["fwd_pe"]: [25.0, nan],
        _LABELS["fwd_ev_ebitda"]: [18.0, nan],
    }, index=idx)

    rows = parse_history(df, 1)

    assert len(rows) == 1                            # the all-null second row is dropped
    assert rows[0]["as_of_date"] == date(2020, 1, 31)


def test_parse_history_empty_and_unmapped():
    assert parse_history(None, 1) == []
    assert parse_history(pd.DataFrame(), 1) == []

    idx = pd.to_datetime(["2021-06-30"])
    df = pd.DataFrame({"Some Unrelated Field": [1.0], _LABELS["revenue_mean"]: [500.0]}, index=idx)
    rows = parse_history(df, 3)

    assert len(rows) == 1
    assert rows[0]["revenue_mean"] == 500.0          # mapped column kept
    assert all(rows[0][f] is None for f in DB_FIELDS if f != "revenue_mean")  # unmapped ignored
