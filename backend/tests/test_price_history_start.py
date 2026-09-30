import json
from datetime import date

import pytest

from backend.ml.dataset import _price_history_start, _trim_predecessor_prices, load_frames
from backend.ml.factors.assembly import _sector_on, build_universe_return_map

EVIDENCE = dict(reason="Reviewed predecessor separation", artifact="source.json", sha256="a" * 64)


def test_known_sector_gaps_do_not_backfill_current_labels():
    fallback = ("Materials", "Current industry")
    assert _sector_on(None, date(2018, 1, 1), fallback) == fallback
    history = [
        dict(
            valid_from=date(2018, 2, 1),
            valid_to=date(2018, 3, 1),
            sector="Industrials",
            industry=None,
        )
    ]
    assert _sector_on(history, date(2018, 1, 31), fallback) == (None, None)
    assert _sector_on(history, date(2018, 2, 1), fallback) == ("Industrials", None)
    assert _sector_on(history, date(2018, 3, 1), fallback) == (None, None)


def test_start_is_inclusive_and_preserves_raw_rows():
    start, evidence = _price_history_start("2018-10-31", json.dumps(EVIDENCE))
    raw = [dict(trade_date=date(2018, 10, d)) for d in (30, 31)]
    assert _trim_predecessor_prices(raw, start) == raw[1:]
    assert len(raw) == 2 and evidence == EVIDENCE
    assert _price_history_start(None, None) == (None, None)


@pytest.mark.parametrize(
    "start,source",
    [
        (None, EVIDENCE),
        ("2018-10-31", None),
        ("2018-10-31", {}),
        ("bad", EVIDENCE),
        ("2018-10-31", "{}"),
    ],
)
def test_start_without_valid_provenance_fails(start, source):
    with pytest.raises(ValueError):
        _price_history_start(start, source)


@pytest.mark.asyncio
async def test_loader_never_uses_predecessor_bar_for_first_successor_return():
    class Pool:
        async def fetch(self, sql, *args):
            if "from tickers" in sql:
                return [
                    dict(
                        ticker_id=1,
                        symbol="LIN",
                        embedding_idx=1,
                        shares_outstanding=None,
                        sector="Materials",
                        industry=None,
                        removed_at=None,
                        security_retired_at=None,
                        price_source_exclusions=None,
                        price_history_start="2018-10-31",
                        price_history_start_source=json.dumps(EVIDENCE),
                    )
                ]
            if "from price_history" in sql:
                return [
                    dict(
                        ticker_id=1, trade_date=date(2018, 10, 30), adj_close=100, source="yfinance"
                    ),
                    dict(
                        ticker_id=1, trade_date=date(2018, 10, 31), adj_close=150, source="yfinance"
                    ),
                ]
            return []

    frames = await load_frames(Pool())
    assert len(frames[0].prices) == 1
    assert frames[0].price_history_start == date(2018, 10, 31)
    assert build_universe_return_map(frames) == {}
    assert frames[0].sector_history == []
    assert _sector_on(frames[0].sector_history, date(2018, 10, 31), ("Materials", None)) == (
        None,
        None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_table", [True, False])
async def test_sector_query_errors_cannot_silently_enable_current_sector(missing_table):
    import asyncpg

    class Pool:
        async def fetch(self, sql, *args):
            if "from tickers" in sql:
                return [
                    dict(
                        ticker_id=1,
                        symbol="PX",
                        embedding_idx=1,
                        shares_outstanding=None,
                        sector="Materials",
                        industry=None,
                        removed_at=None,
                        security_retired_at=None,
                        price_source_exclusions=None,
                    )
                ]
            if "from sector_history" in sql:
                if missing_table:
                    raise asyncpg.UndefinedTableError("pre-migration fixture")
                raise RuntimeError("query failure")
            return []

    if missing_table:
        frames = await load_frames(Pool())
        assert frames[0].sector_history is None
    else:
        with pytest.raises(RuntimeError, match="query failure"):
            await load_frames(Pool())
