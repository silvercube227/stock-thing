"""Dataset assembly tests — pure, synthetic frames, no DB.

Focus on the things that are easy to get silently wrong: label direction,
horizon masking at the tail, the time-based split boundaries, and the
calendar-free month arithmetic.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from backend.ml.dataset import (
    Sample,
    SplitConfig,
    TickerFrame,
    assemble_ticker_samples,
    compute_targets,
    months_before,
    split_samples,
    to_arrays,
)
from backend.ml.features import FEATURE_DIM, SEQUENCE_LENGTH
from backend.ml.model import HORIZONS


def make_frame(n_days: int, trend: float = 1.0, tid: int = 1, emb: int = 1) -> TickerFrame:
    d0 = date(2022, 1, 3)
    prices = [
        {
            "trade_date": d0 + timedelta(days=i),
            "adj_close": 100.0 + trend * i,
            "volume": 1_000_000 + i,
        }
        for i in range(n_days)
    ]
    return TickerFrame(tid, emb, "TST", prices, [], [])


# =============================================================
# months_before
# =============================================================


def test_months_before_basic():
    assert months_before(date(2026, 5, 23), 6) == date(2025, 11, 23)
    assert months_before(date(2026, 5, 23), 18) == date(2024, 11, 23)


def test_months_before_year_rollover():
    assert months_before(date(2026, 1, 15), 1) == date(2025, 12, 15)


def test_months_before_day_clamp_and_leap():
    assert months_before(date(2026, 3, 31), 1) == date(2026, 2, 28)
    assert months_before(date(2024, 3, 31), 1) == date(2024, 2, 29)  # leap


# =============================================================
# compute_labels
# =============================================================


def test_labels_up_when_price_rises():
    adj = [float(i + 1) for i in range(300)]   # strictly increasing, all positive
    labels, returns, mask = compute_targets(adj, end_idx=0)
    for h in HORIZONS:
        assert mask[h] is True
        assert labels[h] == 1
        assert returns[h] > 0


def test_labels_down_when_price_falls():
    adj = [float(300 - i) for i in range(300)]  # strictly decreasing
    labels, returns, mask = compute_targets(adj, end_idx=0)
    for h in HORIZONS:
        assert labels[h] == 0
        assert returns[h] < 0


def test_return_target_is_log_ratio():
    import math
    adj = [2.0] * 300
    adj[21] = 2.0 * math.e          # 1M horizon (21 bars) -> log(e) = 1.0
    labels, returns, mask = compute_targets(adj, end_idx=0)
    assert mask["1M"] and abs(returns["1M"] - 1.0) < 1e-9 and labels["1M"] == 1
    assert returns["3M"] == 0.0 and labels["3M"] == 0   # flat -> non-positive


def test_labels_mask_beyond_available_bars():
    adj = [float(i + 1) for i in range(300)]
    # end_idx=100: 1Y (252) lands at 352 >= 300 -> masked; shorter horizons fine.
    labels, returns, mask = compute_targets(adj, end_idx=100)
    assert mask["1M"] and mask["3M"] and mask["6M"]
    assert mask["1Y"] is False
    assert returns["1Y"] == 0.0


# =============================================================
# assemble_ticker_samples
# =============================================================


def test_assemble_shapes_and_label_direction():
    frame = make_frame(n_days=520, trend=1.0)   # rising -> labels 1 where available
    samples = assemble_ticker_samples(frame, stride=20)
    assert samples, "expected at least one sample"
    for s in samples:
        assert s.features.shape == (SEQUENCE_LENGTH, FEATURE_DIM)
        assert s.features.dtype == np.float32

    earliest = min(samples, key=lambda s: s.sample_end)
    # First valid window has full forward history -> all 4 horizons present & up.
    assert all(earliest.mask[h] for h in HORIZONS)
    assert all(earliest.labels[h] == 1 for h in HORIZONS)
    assert all(earliest.returns[h] > 0 for h in HORIZONS)


def test_assemble_skips_when_too_little_history():
    frame = make_frame(n_days=SEQUENCE_LENGTH)   # exactly 252 -> no full window
    assert assemble_ticker_samples(frame) == []


def test_assemble_drops_samples_with_no_available_label():
    # 1M horizon is 21 bars; the last ~21 end dates have no label at all.
    frame = make_frame(n_days=520)
    samples = assemble_ticker_samples(frame, stride=1)
    last_end_idx = max(range(SEQUENCE_LENGTH, 520))  # 519
    # The very last emitted sample must still have at least one valid horizon.
    latest = max(samples, key=lambda s: s.sample_end)
    assert any(latest.mask[h] for h in HORIZONS)
    # And it cannot be the final bar (which has no forward label).
    assert latest.sample_end < frame.prices[last_end_idx]["trade_date"]


# =============================================================
# split
# =============================================================


def _dummy_sample(d: date) -> Sample:
    return Sample(1, 1, d, np.zeros((SEQUENCE_LENGTH, FEATURE_DIM), np.float32),
                  {h: 1 for h in HORIZONS}, {h: 0.05 for h in HORIZONS},
                  {h: True for h in HORIZONS})


def test_split_boundaries():
    T = date(2026, 5, 22)
    cfg = SplitConfig(holdout_months=6, val_months=18)
    holdout_start = months_before(T, 6)   # 2025-11-22
    val_start = months_before(T, 18)      # 2024-11-22

    samples = [
        _dummy_sample(date(2023, 1, 1)),   # train
        _dummy_sample(val_start),          # val (boundary inclusive)
        _dummy_sample(date(2025, 6, 1)),   # val
        _dummy_sample(holdout_start),      # holdout (boundary inclusive)
        _dummy_sample(T),                  # holdout
    ]
    out = split_samples(samples, cfg, T=T)
    assert [s.sample_end for s in out["train"]] == [date(2023, 1, 1)]
    assert {s.sample_end for s in out["val"]} == {val_start, date(2025, 6, 1)}
    assert {s.sample_end for s in out["holdout"]} == {holdout_start, T}


# =============================================================
# to_arrays
# =============================================================


def test_to_arrays_shapes_and_dtypes():
    frame = make_frame(n_days=520)
    samples = assemble_ticker_samples(frame, stride=40)
    arr = to_arrays(samples)
    n = len(samples)
    assert arr["x"].shape == (n, SEQUENCE_LENGTH, FEATURE_DIM)
    assert arr["x"].dtype == np.float32
    assert arr["ticker_idx"].shape == (n,) and arr["ticker_idx"].dtype == np.int64
    assert arr["y"].shape == (n, 4) and arr["y"].dtype == np.int64
    assert arr["r"].shape == (n, 4) and arr["r"].dtype == np.float32
    assert arr["mask"].shape == (n, 4) and arr["mask"].dtype == np.float32
    assert set(np.unique(arr["mask"])).issubset({0.0, 1.0})
