"""load_frames_cached: disk cache cuts repeat Supabase pulls for experiments.

The cache is what stops every local walk-forward / sweep / backtest from
re-pulling full history through the Supabase pooler (the dominant egress
source). These tests pin the hit / miss / refresh behavior with a stubbed
load_frames so no DB is touched.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import backend.ml.dataset as ds
from backend.ml.dataset import TickerFrame, load_frames_cached


def _frame(tid: int) -> TickerFrame:
    return TickerFrame(
        ticker_id=tid, embedding_idx=tid, symbol=f"T{tid}",
        prices=[{"trade_date": "2020-01-01", "adj_close": 1.0, "volume": 10}],
        fundamentals=[], sentiment=[],
    )


def _patch(monkeypatch, tmp_path):
    """Point the cache at tmp_path and count how often the real pull runs."""
    calls = {"n": 0}

    async def fake_load_frames(pool, symbols=None):
        calls["n"] += 1
        return [_frame(1), _frame(2)]

    monkeypatch.setattr(ds, "load_frames", fake_load_frames)
    monkeypatch.setattr(ds, "get_settings",
                        lambda: SimpleNamespace(frame_cache_dir=tmp_path))
    return calls


def test_miss_then_hit(monkeypatch, tmp_path):
    calls = _patch(monkeypatch, tmp_path)

    first = asyncio.run(load_frames_cached(pool=None))
    assert calls["n"] == 1                      # miss -> pulled
    assert (tmp_path / "frames_all.pkl").exists()

    second = asyncio.run(load_frames_cached(pool=None))
    assert calls["n"] == 1                      # hit -> no second pull
    assert [f.ticker_id for f in second] == [f.ticker_id for f in first]


def test_refresh_forces_repull(monkeypatch, tmp_path):
    calls = _patch(monkeypatch, tmp_path)

    asyncio.run(load_frames_cached(pool=None))
    asyncio.run(load_frames_cached(pool=None, refresh=True))
    assert calls["n"] == 2                       # refresh re-pulls and overwrites


def test_symbol_sets_use_separate_cache_files(monkeypatch, tmp_path):
    calls = _patch(monkeypatch, tmp_path)

    asyncio.run(load_frames_cached(pool=None))                       # "all"
    asyncio.run(load_frames_cached(pool=None, symbols=["AAPL"]))     # distinct key
    assert calls["n"] == 2
    assert (tmp_path / "frames_all.pkl").exists()
    assert len(list(tmp_path.glob("frames_sym-*.pkl"))) == 1
