"""Intraday quote endpoint.

The stored `price_history` only has daily bars (yfinance daily cron), so "live"
prices come from yfinance `fast_info` (delayed, not real-time). Results are cached
per symbol for a short TTL and fetched with bounded concurrency so a polling
dashboard doesn't hammer yfinance. When a live fetch yields nothing we fall back
to the latest stored close and flag the quote `stale`.

yfinance is synchronous; blocking calls are wrapped in `asyncio.to_thread`, the
same pattern used by `backend/ingestion/prices.py`.
"""

from __future__ import annotations

import asyncio
import logging
import time

import asyncpg
import yfinance as yf
from fastapi import APIRouter, Depends, Query

from backend.api.auth import get_current_user
from backend.api.deps import get_pool
from backend.api.schemas import Quote

log = logging.getLogger(__name__)

# Short TTL: the dashboard polls ~every 45s; 30s keeps it fresh without thrash.
QUOTE_TTL_SECONDS: float = 30.0
_MAX_CONCURRENCY = 6

# symbol -> (monotonic_ts, Quote)
_cache: dict[str, tuple[float, Quote]] = {}
_sem = asyncio.Semaphore(_MAX_CONCURRENCY)

router = APIRouter(tags=["quotes"])


def _fi_get(fi, *keys: str) -> float | None:
    """First non-None value among `keys` from a yfinance FastInfo.

    yfinance has used both camelCase ('lastPrice') and snake_case ('last_price')
    across versions, so we try each.
    """
    for k in keys:
        v = fi.get(k) if hasattr(fi, "get") else getattr(fi, k, None)
        if v is not None:
            return float(v)
    return None


def _fetch_one_sync(symbol: str) -> tuple[float | None, float | None]:
    """Return (last_price, previous_close) from yfinance fast_info, or (None, None)."""
    try:
        fi = yf.Ticker(symbol).fast_info
        last = _fi_get(fi, "lastPrice", "last_price")
        prev = _fi_get(fi, "previousClose", "previous_close")
        return last, prev
    except Exception as exc:  # noqa: BLE001 — yfinance throws a variety of errors
        log.warning("quote fetch failed for %s: %s", symbol, exc)
        return (None, None)


def _build_quote(symbol: str, last: float | None, prev: float | None, *, stale: bool) -> Quote:
    change = change_pct = None
    if last is not None and prev not in (None, 0):
        change = last - prev
        change_pct = change / prev * 100.0
    return Quote(
        symbol=symbol,
        price=last,
        prev_close=prev,
        change=change,
        change_pct=change_pct,
        stale=stale,
    )


async def _fetch_live(symbol: str) -> Quote | None:
    async with _sem:
        last, prev = await asyncio.to_thread(_fetch_one_sync, symbol)
    if last is None:
        return None
    return _build_quote(symbol, last, prev, stale=False)


async def _stored_closes(pool: asyncpg.Pool, symbols: list[str]) -> dict[str, float]:
    """Latest stored close per symbol, for the fallback path."""
    if not symbols:
        return {}
    rows = await pool.fetch(
        """
        select t.symbol,
               (select close from price_history
                 where ticker_id = t.ticker_id
                 order by trade_date desc limit 1) as close
          from tickers t
         where upper(t.symbol) = any($1::text[])
        """,
        symbols,
    )
    return {r["symbol"].upper(): float(r["close"]) for r in rows if r["close"] is not None}


async def fetch_quotes(symbols: list[str], pool: asyncpg.Pool) -> dict[str, Quote]:
    """Resolve quotes for `symbols`, using cache, then live fetch, then stored close."""
    now = time.monotonic()
    wanted = {s.strip().upper() for s in symbols if s.strip()}
    result: dict[str, Quote] = {}
    to_fetch: list[str] = []

    for sym in wanted:
        cached = _cache.get(sym)
        if cached and (now - cached[0]) < QUOTE_TTL_SECONDS:
            result[sym] = cached[1]
        else:
            to_fetch.append(sym)

    if to_fetch:
        live = await asyncio.gather(*(_fetch_live(s) for s in to_fetch))
        missing: list[str] = []
        for sym, quote in zip(to_fetch, live, strict=True):
            if quote is not None:
                _cache[sym] = (now, quote)
                result[sym] = quote
            else:
                missing.append(sym)

        if missing:
            closes = await _stored_closes(pool, missing)
            for sym in missing:
                close = closes.get(sym)
                quote = _build_quote(sym, close, close, stale=True)
                # Cache the fallback briefly too, to avoid re-fetching a dud symbol.
                _cache[sym] = (now, quote)
                result[sym] = quote

    return result


@router.get("/quotes", response_model=dict[str, Quote])
async def get_quotes(
    symbols: str = Query(..., description="Comma-separated symbols, e.g. AAPL,MSFT"),
    pool: asyncpg.Pool = Depends(get_pool),
    _user: str = Depends(get_current_user),
) -> dict[str, Quote]:
    parsed = [s for s in symbols.split(",") if s.strip()]
    return await fetch_quotes(parsed, pool)
