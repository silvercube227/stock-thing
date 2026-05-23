"""Backfill SEC CIK identifiers for active equity tickers.

SEC publishes a canonical ticker -> CIK mapping at
https://www.sec.gov/files/company_tickers.json. We fetch it once and update
the cik column for any active equity ticker that doesn't yet have one.

Usage:
    python -m scripts.backfill_ciks

Requires SUPABASE_URL, SUPABASE_SECRET_KEY (or DATABASE_URL) in .env.
Run AFTER seed_tickers.sql has been applied.

ETFs are skipped — CIK is only meaningful for issuers that file 10-K/10-Q
with the SEC, which is what our fundamentals pipeline targets.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import asyncpg
import httpx

from backend.config import get_settings

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


async def fetch_sec_mapping(user_agent: str) -> dict[str, str]:
    """Return {symbol -> zero-padded 10-digit CIK string}."""
    headers = {"User-Agent": user_agent, "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        resp = await client.get(SEC_TICKERS_URL)
        resp.raise_for_status()
        data: dict[str, dict[str, Any]] = resp.json()

    mapping: dict[str, str] = {}
    for entry in data.values():
        symbol = str(entry["ticker"]).upper()
        cik = str(entry["cik_str"]).zfill(10)
        mapping[symbol] = cik
    return mapping


def _asyncpg_dsn(database_url: str) -> str:
    """Strip the SQLAlchemy driver hint (postgresql+asyncpg://) for asyncpg."""
    return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def amain() -> int:
    settings = get_settings()
    if not settings.database_url:
        print("DATABASE_URL is not set in .env", file=sys.stderr)
        return 1

    print(f"Fetching SEC ticker mapping with UA: {settings.sec_edgar_user_agent!r}")
    mapping = await fetch_sec_mapping(settings.sec_edgar_user_agent)
    print(f"  -> {len(mapping):,} symbols available from SEC")

    conn = await asyncpg.connect(_asyncpg_dsn(settings.database_url))
    try:
        rows = await conn.fetch(
            """
            select ticker_id, symbol
              from tickers
             where active = true
               and asset_type = 'equity'
               and cik is null
            """
        )

        updated = 0
        missing: list[str] = []
        async with conn.transaction():
            for row in rows:
                ticker_id, symbol = row["ticker_id"], row["symbol"]
                cik = mapping.get(symbol.upper())
                if cik is None:
                    missing.append(symbol)
                    continue
                await conn.execute(
                    "update tickers set cik = $1 where ticker_id = $2",
                    cik,
                    ticker_id,
                )
                updated += 1
    finally:
        await conn.close()

    print(f"Updated CIK for {updated} ticker(s).")
    if missing:
        print(f"No SEC mapping for: {', '.join(missing)}")
        print("  (delisted, foreign issuer, or wrong symbol — investigate before training)")
    return 0


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())
