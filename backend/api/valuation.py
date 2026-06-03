"""Live valuation-multiples endpoint.

P/E, P/S and EBITDA are price-dependent and not stored in Postgres (the
`fundamentals` table holds point-in-time EDGAR filings). They come from yfinance
`.info`, which is a heavier call than `fast_info`, so results are cached per
symbol with a longer TTL and fetched with bounded concurrency. A failed/empty
fetch returns an all-null snapshot rather than erroring, so the ticker page just
omits the multiples.

Same blocking-call-in-thread pattern as `backend/api/quotes.py`.
"""

from __future__ import annotations

import asyncio
import logging
import time

import yfinance as yf
from fastapi import APIRouter, Depends

from backend.api.auth import get_current_user
from backend.api.schemas import ValuationSnapshot

log = logging.getLogger(__name__)

# Multiples move only with the (delayed) price and quarterly filings, so a longer
# TTL than the quote cache is fine and keeps the heavy .info call infrequent.
VALUATION_TTL_SECONDS: float = 900.0  # 15 min
_MAX_CONCURRENCY = 4

# symbol -> (monotonic_ts, ValuationSnapshot)
_cache: dict[str, tuple[float, ValuationSnapshot]] = {}
_sem = asyncio.Semaphore(_MAX_CONCURRENCY)

router = APIRouter(prefix="/tickers", tags=["valuation"])


def _num(info: dict, key: str) -> float | None:
    v = info.get(key)
    if isinstance(v, (int, float)) and v == v:  # reject None and NaN
        return float(v)
    return None


def _fetch_info_sync(symbol: str) -> dict:
    try:
        return yf.Ticker(symbol).info or {}
    except Exception as exc:  # noqa: BLE001 — yfinance throws a variety of errors
        log.warning("valuation fetch failed for %s: %s", symbol, exc)
        return {}


async def fetch_valuation(symbol: str) -> ValuationSnapshot:
    sym = symbol.strip().upper()
    now = time.monotonic()
    cached = _cache.get(sym)
    if cached and (now - cached[0]) < VALUATION_TTL_SECONDS:
        return cached[1]

    async with _sem:
        info = await asyncio.to_thread(_fetch_info_sync, sym)
    snap = ValuationSnapshot(
        symbol=sym,
        trailing_pe=_num(info, "trailingPE"),
        forward_pe=_num(info, "forwardPE"),
        price_to_sales=_num(info, "priceToSalesTrailing12Months"),
        ebitda=_num(info, "ebitda"),
        revenue=_num(info, "totalRevenue"),
        net_income=_num(info, "netIncomeToCommon"),
        gross_margin=_num(info, "grossMargins"),
        operating_margin=_num(info, "operatingMargins"),
        fcf=_num(info, "freeCashflow"),
    )
    _cache[sym] = (now, snap)
    return snap


@router.get("/{symbol}/valuation", response_model=ValuationSnapshot)
async def get_valuation(
    symbol: str,
    _user: str = Depends(get_current_user),
) -> ValuationSnapshot:
    return await fetch_valuation(symbol)
