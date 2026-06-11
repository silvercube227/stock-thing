"""FINRA Regulation SHO consolidated short interest ingestion.

Uses the FINRA Open API (no auth required):
    https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest

Each record has `settlementDate` (bimonthly ~1st and ~15th) and
`currentShortPositionQuantity` (shares short). PIT-safe: we store
`publication_date = settlement_date + 14 days` as the join key — the date
FINRA actually publishes the data.

The API returns all historical records paginated (5000 max per page); there is
no reliable server-side date filter, so callers must page through and filter
client-side. ~3.8M total records (all exchanges, ~5000 symbols per date, 2009-present).

Usage (from the venv, Supabase reachable):
    python -m backend.ingestion.short_interest   # scores latest available date

For backfill: scripts/backfill_short_interest.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
from datetime import date, timedelta
from typing import Iterator

import asyncpg

from backend.ingestion.db import pool_context

log = logging.getLogger(__name__)

_API_URL = (
    "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
)
_PAGE_SIZE = 5000

# FINRA typically publishes ~14 calendar days after the settlement date.
PUBLICATION_LAG_DAYS = 14


def _publication_date(settlement: date) -> date:
    return settlement + timedelta(days=PUBLICATION_LAG_DAYS)


def _fetch_page(offset: int) -> list[dict]:
    """Fetch one page of FINRA short interest records."""
    url = f"{_API_URL}?limit={_PAGE_SIZE}&offset={offset}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _iter_all_records(max_pages: int | None = None) -> Iterator[dict]:
    """Yield all FINRA short interest records by paging through the API."""
    offset = 0
    page = 0
    while True:
        records = _fetch_page(offset)
        if not records:
            break
        yield from records
        offset += len(records)
        page += 1
        if len(records) < _PAGE_SIZE:
            break  # last page
        if max_pages is not None and page >= max_pages:
            break


def _parse_record(rec: dict) -> dict | None:
    """Convert a FINRA API record to a DB row dict. Returns None if symbol is missing."""
    symbol = rec.get("symbolCode", "").strip().upper()
    if not symbol:
        return None
    sd_raw = rec.get("settlementDate")
    if not sd_raw:
        return None
    try:
        sd = date.fromisoformat(str(sd_raw)[:10])
    except ValueError:
        return None
    si = rec.get("currentShortPositionQuantity")
    adv = rec.get("averageDailyVolumeQuantity")
    dtc = rec.get("daysToCoverQuantity")
    return {
        "symbol": symbol,
        "settlement_date": sd,
        "publication_date": _publication_date(sd),
        "short_interest": int(si) if si is not None else None,
        "avg_daily_volume": int(adv) if adv is not None else None,
        "days_to_cover": float(dtc) if dtc is not None else None,
    }


async def _resolve_ticker_ids(
    conn: asyncpg.Connection, symbols: list[str]
) -> dict[str, int]:
    if not symbols:
        return {}
    records = await conn.fetch(
        "select symbol, ticker_id from tickers where symbol = any($1::text[])",
        symbols,
    )
    return {r["symbol"]: r["ticker_id"] for r in records}


_UPSERT_SQL = """
insert into short_interest (
    ticker_id, settlement_date, publication_date,
    short_interest, avg_daily_volume, days_to_cover, ingested_at
) values ($1, $2, $3, $4, $5, $6, now())
on conflict (ticker_id, settlement_date) do update set
    publication_date  = excluded.publication_date,
    short_interest    = excluded.short_interest,
    avg_daily_volume  = excluded.avg_daily_volume,
    days_to_cover     = excluded.days_to_cover,
    ingested_at       = now()
"""


async def _upsert_rows(conn: asyncpg.Connection, rows: list[dict]) -> int:
    if not rows:
        return 0
    payload = [
        (r["ticker_id"], r["settlement_date"], r["publication_date"],
         r["short_interest"], r["avg_daily_volume"], r["days_to_cover"])
        for r in rows
    ]
    async with conn.transaction():
        await conn.executemany(_UPSERT_SQL, payload)
    return len(payload)


async def ingest_short_interest_full(
    pool: asyncpg.Pool,
    after_date: date | None = None,
    max_pages: int | None = None,
) -> dict:
    """Download + upsert ALL FINRA short interest records, optionally filtered.

    `after_date`: skip records whose settlement_date <= after_date (for incremental).
    `max_pages`: cap pages for testing.
    Returns {"fetched": int, "upserted": int}.
    """
    fetched = 0
    upserted = 0
    batch: list[dict] = []
    BATCH = 5000

    async with pool.acquire() as conn:
        known_symbols: dict[str, int] = {}

        for raw_rec in _iter_all_records(max_pages=max_pages):
            parsed = _parse_record(raw_rec)
            if parsed is None:
                continue
            if after_date is not None and parsed["settlement_date"] <= after_date:
                continue
            fetched += 1
            batch.append(parsed)
            if len(batch) >= BATCH:
                new_syms = [r["symbol"] for r in batch if r["symbol"] not in known_symbols]
                if new_syms:
                    extra = await _resolve_ticker_ids(conn, new_syms)
                    known_symbols.update(extra)
                db_rows = [
                    {**r, "ticker_id": known_symbols[r["symbol"]]}
                    for r in batch if r["symbol"] in known_symbols
                ]
                upserted += await _upsert_rows(conn, db_rows)
                log.info("batch upserted=%d (total fetched=%d upserted=%d)", len(db_rows), fetched, upserted)
                batch = []

        # Flush final partial batch
        if batch:
            new_syms = [r["symbol"] for r in batch if r["symbol"] not in known_symbols]
            if new_syms:
                extra = await _resolve_ticker_ids(conn, new_syms)
                known_symbols.update(extra)
            db_rows = [
                {**r, "ticker_id": known_symbols[r["symbol"]]}
                for r in batch if r["symbol"] in known_symbols
            ]
            upserted += await _upsert_rows(conn, db_rows)

    return {"fetched": fetched, "upserted": upserted}


# =============================================================
# CLI entry point
# =============================================================

async def _main() -> None:
    async with pool_context() as pool:
        # Pull latest available date only (last page of the API = most recent)
        result = await ingest_short_interest_full(pool, max_pages=2)
        print(result)


if __name__ == "__main__":
    asyncio.run(_main())
