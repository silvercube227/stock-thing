"""Dataset assembly tests — pure, synthetic frames, no DB.

Focus on the things that are easy to get silently wrong: label direction,
horizon masking at the tail, the time-based split boundaries, and the
calendar-free month arithmetic.
"""

from __future__ import annotations

import math

from datetime import date, timedelta

import numpy as np
import pytest

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
    # Entry is the bar AFTER end_idx (implementation lag), so the 1M exit bar is
    # 1 + 21 = 22, not 21.
    adj[22] = 2.0 * math.e          # 1M horizon (21 bars) -> log(e) = 1.0
    labels, returns, mask = compute_targets(adj, end_idx=0)
    assert mask["1M"] and abs(returns["1M"] - 1.0) < 1e-9 and labels["1M"] == 1
    assert returns["3M"] == 0.0 and labels["3M"] == 0   # flat -> non-positive


def test_label_ignores_the_signal_bar_close():
    """The entry price is the NEXT bar, so bar `end_idx` cannot enter the return.

    Pins the one-bar implementation lag: without it, signal and entry share a close.
    """
    adj = [1.0] * 300
    baseline = compute_targets(adj, end_idx=0)[1]
    adj[0] = 100.0                  # wild price on the signal bar only
    shifted = compute_targets(adj, end_idx=0)[1]
    assert shifted == baseline


def test_label_masked_when_entry_bar_missing():
    """No bar after `end_idx` means the position could never be opened."""
    adj = [float(i + 1) for i in range(300)]
    labels, returns, mask = compute_targets(adj, end_idx=299)
    for h in HORIZONS:
        assert mask[h] is False
        assert returns[h] == 0.0


def test_labels_mask_beyond_available_bars():
    adj = [float(i + 1) for i in range(300)]
    # end_idx=100: entry at 101, 1Y (252) exits at 353 >= 300 -> masked;
    # shorter horizons fine.
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


# =============================================================
# Symbol reuse: a delisted ticker's symbol reassigned to another company
# =============================================================


def _bars(start, n, step_days=1):
    from datetime import timedelta
    return [{"trade_date": start + timedelta(days=i * step_days)} for i in range(n)]


def test_reuse_guard_leaves_a_continuously_trading_ex_member_alone():
    """De-survivorship depends on these rows: a name dropped from the index that
    keeps trading (FOSL, GME, AA, RIG) must keep every bar."""
    from backend.ml.dataset import _drop_reused_symbol_bars

    rows = _bars(date(2014, 1, 1), 3000)
    assert _drop_reused_symbol_bars(rows, date(2016, 1, 5)) == rows
    # A name still in the index has no removal date and is never touched.
    assert _drop_reused_symbol_bars(rows, None) == rows


def test_reuse_guard_drops_a_series_that_begins_after_removal():
    """SE/EMC/CA/APC pattern: every bar we hold belongs to the new issuer."""
    from backend.ml.dataset import _drop_reused_symbol_bars

    rows = _bars(date(2023, 12, 13), 600)
    assert _drop_reused_symbol_bars(rows, date(2018, 11, 6)) == []


def test_reuse_guard_truncates_at_a_multi_year_hole():
    """CSRA/NFX pattern: the real company's bars stop at the acquisition and an
    unrelated listing resumes under the symbol years later."""
    from backend.ml.dataset import _drop_reused_symbol_bars

    original = _bars(date(2015, 1, 1), 800)
    rows = original + _bars(date(2026, 7, 7), 40)
    assert _drop_reused_symbol_bars(rows, date(2019, 2, 15)) == original


def test_reuse_guard_ignores_a_short_trading_halt():
    """A gap of a few months after removal is a halt or a thin tape, not a new
    issuer — the threshold has to be wide enough not to eat those."""
    from backend.ml.dataset import _drop_reused_symbol_bars

    rows = _bars(date(2014, 1, 1), 900) + _bars(date(2016, 8, 1), 500)
    assert _drop_reused_symbol_bars(rows, date(2016, 1, 5)) == rows


# =============================================================
# Terminal (delisting) labels vs right-censoring
# =============================================================


def test_terminal_flag_cannot_invent_last_trade_proceeds():
    """Only documented_targets can establish an explicit terminal outcome."""
    from backend.ml.dataset import compute_targets

    # 30 bars: entry at index 1, but 1M (21 bars) would need index 22 — present;
    # 3M (63 bars) is past the end.
    prices = [100.0 + i for i in range(30)]
    _lab, ret, mask = compute_targets(prices, end_idx=0, terminal=True)
    assert mask["3M"] is False
    assert ret["3M"] == 0.0
    # A price collapse alone does not establish the subsequent recovery value.
    crash = [100.0] * 25 + [20.0]
    _l2, ret2, mask2 = compute_targets(crash, end_idx=0, terminal=True)
    assert mask2["1Y"] is False and ret2["1Y"] == 0.0


def test_right_censored_series_is_still_masked():
    """The panel simply ending is NOT a delisting: the future has not happened yet,
    so those horizons must stay masked rather than inventing an exit."""
    from backend.ml.dataset import compute_targets

    prices = [100.0 + i for i in range(30)]
    _lab, ret, mask = compute_targets(prices, end_idx=0, terminal=False)
    assert mask["1M"] is True          # 21 bars ahead exists
    assert mask["3M"] is False         # 63 bars ahead does not
    assert ret["3M"] == 0.0


def test_terminal_needs_a_bar_after_entry():
    from backend.ml.dataset import compute_targets

    # entry_idx = 1 is the last bar, so there is no exit bar even holding to the end.
    _lab, _ret, mask = compute_targets([100.0, 101.0], end_idx=0, terminal=True)
    assert all(v is False for v in mask.values())


# =============================================================
# Point-in-time sector lookup
# =============================================================


def test_sector_on_returns_the_label_as_of_the_row_date():
    from backend.ml.factors.assembly import _sector_on

    history = [
        {"valid_from": date(2010, 1, 1), "valid_to": date(2018, 10, 1),
         "sector": "Consumer Discretionary", "industry": "Media"},
        {"valid_from": date(2018, 10, 1), "valid_to": None,
         "sector": "Communication Services", "industry": "Entertainment"},
    ]
    fallback = ("Communication Services", "Entertainment")
    # Before the 2018 GICS reshuffle DIS was Consumer Discretionary, and a 2012 row
    # must be demeaned against THAT peer group.
    assert _sector_on(history, date(2012, 6, 30), fallback)[0] == "Consumer Discretionary"
    # valid_to is exclusive, so the boundary date itself belongs to the new interval.
    assert _sector_on(history, date(2018, 9, 30), fallback)[0] == "Consumer Discretionary"
    assert _sector_on(history, date(2018, 10, 1), fallback)[0] == "Communication Services"
    assert _sector_on(history, date(2025, 1, 1), fallback)[0] == "Communication Services"


def test_sector_on_falls_back_only_when_history_was_not_loaded():
    from backend.ml.factors.assembly import _sector_on

    fallback = ("Health Care", "Biotech")
    # Table not loaded (pre-migration cache) -> static tickers label, i.e. the old
    # behaviour exactly.
    assert _sector_on(None, date(2015, 1, 1), fallback) == fallback
    assert _sector_on([], date(2015, 1, 1), fallback) == (None, None)
    # A known coverage gap must not borrow the present-day classification.
    history = [{"valid_from": date(2020, 1, 1), "valid_to": None,
                "sector": "Financials", "industry": None}]
    assert _sector_on(history, date(2015, 1, 1), fallback) == (None, None)
    assert _sector_on(history, date(2021, 1, 1), fallback)[0] == "Financials"


def test_ticker_frame_accepts_every_kwarg_load_frames_passes():
    """Guard against a field being dropped from TickerFrame.

    load_frames is the only caller that passes the full kwarg set, and it needs a
    live DB — so an editing slip that removed the `membership` field passed the whole
    suite and only surfaced on a real cache refresh. This pins the constructor
    contract without needing a database.
    """
    from backend.ml.dataset import TickerFrame

    frame = TickerFrame(
        ticker_id=1, embedding_idx=1, symbol="T1",
        prices=[], fundamentals=[], sentiment=[],
        shares_outstanding=100, sector="Information Technology", industry="Software",
        estimates=[], surprises=[], short_interest=[], insiders=[],
        membership=[{"valid_from": date(2015, 1, 1), "valid_to": None}],
        sector_history=[{"valid_from": date(2015, 1, 1), "valid_to": None,
                         "sector": "Information Technology", "industry": "Software"}],
        removed_at=None,
    )
    assert frame.membership and frame.sector_history
    assert frame.removed_at is None


def test_retirement_truncates_contiguous_reused_bars():
    """Starwood, Rockwell Collins and Harman keep trading under a new owner.

    The gap rule cannot see these: the next company's bars are contiguous with
    the original's, so only the vendor retirement date separates them.
    """
    from datetime import date as _date, timedelta as _td

    from backend.ml.dataset import _drop_reused_symbol_bars

    start = _date(2016, 1, 4)
    rows = [{"trade_date": start + _td(days=i)} for i in range(0, 500, 7)]
    kept = _drop_reused_symbol_bars(rows, None, _date(2016, 9, 23))
    assert kept, "the security's own bars must survive"
    assert max(r["trade_date"] for r in kept) <= _date(2016, 9, 23)
    assert len(kept) < len(rows)


def test_index_removal_alone_never_truncates_a_live_listing():
    """TechnipFMC left the index in 2021 and still trades; its bars are real."""
    from datetime import date as _date, timedelta as _td

    from backend.ml.dataset import _drop_reused_symbol_bars

    rows = [{"trade_date": _date(2021, 1, 4) + _td(days=i)} for i in range(0, 900, 7)]
    assert _drop_reused_symbol_bars(rows, _date(2021, 2, 12), None) == rows


def test_retirement_and_gap_rules_compose():
    from datetime import date as _date, timedelta as _td

    from backend.ml.dataset import _drop_reused_symbol_bars

    rows = [{"trade_date": _date(2018, 1, 3) + _td(days=i)} for i in range(0, 200, 7)]
    rows += [{"trade_date": _date(2020, 6, 1) + _td(days=i)} for i in range(0, 100, 7)]
    kept = _drop_reused_symbol_bars(rows, _date(2018, 12, 3), _date(2018, 11, 27))
    assert all(r["trade_date"] <= _date(2018, 11, 27) for r in kept)


def test_no_retirement_date_leaves_the_series_untouched():
    from datetime import date as _date, timedelta as _td

    from backend.ml.dataset import _drop_reused_symbol_bars

    rows = [{"trade_date": _date(2020, 1, 2) + _td(days=i)} for i in range(0, 300, 7)]
    assert _drop_reused_symbol_bars(rows, None, None) == rows
