"""LightGBM walk-forward baseline tests — pure, synthetic, no DB.

Covers the things that are easy to get silently wrong: point-in-time feature
assembly, the cross-sectional rank transform, median-demeaning, the embargo/fold
index math (leakage guard), and an end-to-end planted-signal recovery that the
shuffle null must NOT reproduce.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from backend.ml.dataset import TickerFrame, build_calendar_grid
from backend.ml.gbm_baseline import (
    FEATURE_COLS,
    LGBMConfig,
    WalkForwardConfig,
    block_bootstrap_summary,
    build_ticker_rows,
    demean_cross_sectional,
    prepare_panel,
    rank_normalize_features,
    summarize,
    walk_forward_folds,
    walk_forward_ic,
)
from backend.ml.gbm_inference import score_current_cross_section
from backend.ml.model import HORIZONS

# =============================================================
# Synthetic frames
# =============================================================


def make_frame(n_days: int, trend: float, tid: int, vol_seed: int = 0) -> TickerFrame:
    """A daily price series starting 2018-01-02 with drift `trend` and mild noise."""
    rng = np.random.default_rng(vol_seed)
    d0 = date(2018, 1, 2)
    price = 100.0
    prices = []
    for i in range(n_days):
        price *= 1.0 + trend + rng.normal(0, 0.01)  # geometric drift + daily noise
        prices.append({
            "trade_date": d0 + timedelta(days=i),
            "adj_close": max(price, 1.0),
            "volume": 1_000_000 + 1000 * i,
        })
    return TickerFrame(tid, tid, f"T{tid}", prices, [], [])


# =============================================================
# Feature assembly
# =============================================================


def test_build_ticker_rows_shapes_and_columns():
    frame = make_frame(n_days=900, trend=0.0005, tid=1)
    grid = build_calendar_grid([frame])
    rows = build_ticker_rows(frame, grid)
    assert rows, "expected at least one row"
    cols = set(rows[0])
    for c in FEATURE_COLS:
        assert c in cols, f"missing feature {c}"
    for h in HORIZONS:
        assert f"r_{h}" in cols and f"mask_{h}" in cols
    # No feature should be NaN/inf (the model can't ingest those).
    for r in rows:
        for c in FEATURE_COLS:
            assert np.isfinite(r[c]), f"{c} not finite"


def test_rising_series_has_positive_momentum():
    frame = make_frame(n_days=900, trend=0.001, tid=1)  # steady uptrend
    rows = build_ticker_rows(frame, build_calendar_grid([frame]))
    # On a persistent uptrend most rows show positive 6M momentum.
    pos_frac = np.mean([r["mom_6m"] > 0 for r in rows])
    assert pos_frac > 0.8


def test_too_little_history_yields_no_rows():
    frame = make_frame(n_days=200, trend=0.0, tid=1)  # < SEQUENCE_LENGTH+ usable
    assert build_ticker_rows(frame, build_calendar_grid([frame])) == []


def test_build_ticker_rows_skips_stale_grid_dates():
    frame = make_frame(n_days=360, trend=0.0005, tid=1)
    grid = build_calendar_grid([frame])
    grid.append(date(2022, 1, 31))  # long after this synthetic ticker stopped trading
    rows = build_ticker_rows(frame, grid, max_stale_days=7)
    assert rows
    assert max(r["date"] for r in rows) < date(2022, 1, 31)


# =============================================================
# Cross-sectional transforms
# =============================================================


def test_rank_normalize_in_range_and_centered():
    df = pd.DataFrame({
        "date": [date(2020, 1, 31)] * 5,
        "mom_1m": [5.0, 1.0, 3.0, 2.0, 4.0],
    })
    for c in FEATURE_COLS:
        if c not in df:
            df[c] = 0.0
    out = rank_normalize_features(df, ["mom_1m"])
    vals = out["mom_1m"].to_numpy()
    assert vals.min() >= -1.0 and vals.max() <= 1.0
    assert abs(vals.mean()) < 1e-9          # symmetric ranks center at 0
    # Order is preserved (largest input -> largest normalized rank).
    assert np.argmax(vals) == 0 and np.argmin(vals) == 1


def test_demean_subtracts_median_and_masks_missing():
    df = pd.DataFrame({
        "date": [date(2020, 1, 31), date(2020, 1, 31), date(2020, 2, 28)],
        "r_1M": [0.10, 0.00, 0.05],
        "mask_1M": [True, True, True],
    })
    for h in HORIZONS:
        if f"r_{h}" not in df:
            df[f"r_{h}"] = 0.0
        if f"mask_{h}" not in df:
            df[f"mask_{h}"] = False
    df["mask_1M"] = [True, True, True]
    df["r_1M"] = [0.10, 0.00, 0.05]
    medians = {h: {} for h in HORIZONS}
    medians["1M"] = {date(2020, 1, 31): 0.04}  # Feb has no median -> masked
    out = demean_cross_sectional(df, medians)
    jan = out[out["date"] == date(2020, 1, 31)]
    assert np.allclose(sorted(jan["r_1M"]), sorted([0.10 - 0.04, 0.00 - 0.04]))
    feb = out[out["date"] == date(2020, 2, 28)]
    assert bool(feb["mask_1M"].iloc[0]) is False  # no median -> masked off


# =============================================================
# Fold / embargo math (leakage guard)
# =============================================================


def test_walk_forward_folds_respect_embargo():
    grid = [date(2020, m, 28) for m in range(1, 13)]  # 12 month-ends
    folds = walk_forward_folds(grid, min_train_months=3, embargo_steps=2)
    # First testable index = min_train(3) + embargo(2) = 5.
    assert folds[0][0] == grid[5]
    for test_date, cutoff in folds:
        i = grid.index(test_date)
        assert cutoff == grid[i - 2]          # cutoff is exactly embargo behind
        assert cutoff < test_date


def test_summarize_basic():
    s = summarize([0.1, 0.1, 0.1, 0.1])
    assert s["n_folds"] == 4
    assert abs(s["mean_ic"] - 0.1) < 1e-12
    assert s["hit_rate"] == 1.0
    assert summarize([])["n_folds"] == 0


def test_block_bootstrap_summary_reports_ci_and_overlap_adjusted_t():
    s = block_bootstrap_summary(
        [0.10, 0.08, 0.12, 0.09, 0.11, 0.07, 0.13, 0.10],
        block_size=2,
        reps=200,
        seed=1,
    )
    assert s["n_folds"] == 8
    assert s["block_size"] == 2
    assert s["ci_low"] < s["mean_ic"] < s["ci_high"]
    assert s["effective_blocks"] == 4
    assert s["t_block"] > 0
    assert 0 <= s["p_value"] <= 1


def test_score_current_cross_section_clips_rank_predictions():
    class DummyModel:
        def predict(self, X):
            return np.array([-0.2, 1.3])

    as_of = date(2026, 5, 29)
    df = pd.DataFrame({
        "date": [as_of, as_of, as_of],
        "ticker_id": [1, 2, 3],
    })
    for c in FEATURE_COLS:
        df[c] = 0.0

    rows = score_current_cross_section(
        df, {"3M": DummyModel()}, ("3M",), as_of=as_of, active_ids={1, 3}
    )

    assert [r["ticker_id"] for r in rows] == [1, 3]
    assert [r["relative_rank"] for r in rows] == [0.0, 1.0]
    assert [r["confidence"] for r in rows] == [1.0, 1.0]


# =============================================================
# End-to-end: planted signal recovered, shuffle null is not
# =============================================================


def _planted_panel(n_dates: int = 48, n_names: int = 60, beta: float = 1.0, seed: int = 0):
    """Panel where the 1M relative return is driven by mom_1m plus noise.

    Built directly (bypassing price synthesis) so we test the harness + LightGBM +
    IC end-to-end with a known cross-sectional signal of controllable strength.
    """
    rng = np.random.default_rng(seed)
    dates = [date(2018, 1, 1) + timedelta(days=28 * k) for k in range(n_dates)]
    rows = []
    for d in dates:
        signal = rng.normal(0, 1, n_names)
        rel = beta * signal + rng.normal(0, 1.0, n_names)  # noisy but real
        rel -= rel.mean()                                   # demeaned target
        for j in range(n_names):
            row = {"date": d, "ticker_id": j}
            for c in FEATURE_COLS:
                row[c] = 0.0
            row["mom_1m"] = float(signal[j])
            for h in HORIZONS:
                row[f"r_{h}"] = 0.0
                row[f"mask_{h}"] = False
            row["r_1M"] = float(rel[j])
            row["mask_1M"] = True
            rows.append(row)
    return rank_normalize_features(pd.DataFrame(rows))


def test_walk_forward_recovers_planted_signal_and_null_does_not():
    panel = _planted_panel(n_dates=48, n_names=60, beta=1.0, seed=7)
    wf = WalkForwardConfig(min_train_months=12, min_names=20)
    cfg = LGBMConfig(n_estimators=150)

    real = walk_forward_ic(panel, "1M", cfg, wf, seed=1, shuffle=False)["summary"]
    null = walk_forward_ic(panel, "1M", cfg, wf, seed=1, shuffle=True)["summary"]

    assert real["n_folds"] > 20
    assert real["mean_ic"] > 0.20, f"expected to recover signal, got {real['mean_ic']}"
    assert real["mean_ic"] > null["mean_ic"] + 0.15
    assert abs(null["mean_ic"]) < 0.10, f"shuffle null should be ~0, got {null['mean_ic']}"


def test_prepare_panel_end_to_end_on_frames():
    frames = [make_frame(n_days=800, trend=0.0003 * (k + 1), tid=k, vol_seed=k) for k in range(6)]
    grid = build_calendar_grid(frames)
    panel = prepare_panel(frames, grid)
    assert not panel.empty
    # Rank-normalized features stay in range.
    for c in FEATURE_COLS:
        assert panel[c].min() >= -1.0001 and panel[c].max() <= 1.0001
