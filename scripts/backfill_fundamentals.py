"""Backfill SEC EDGAR fundamentals for active equity tickers.

Usage:
    python -m scripts.backfill_fundamentals                  # all active equities
    python -m scripts.backfill_fundamentals --symbols AAPL MSFT

Requires SEC_EDGAR_USER_AGENT in .env (SEC rejects unidentified requests).
Run AFTER scripts.backfill_ciks (needs CIK populated).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from backend.ingestion.db import pool_context
from backend.ingestion.fundamentals import ingest_fundamentals


async def amain(symbols: list[str] | None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    async with pool_context() as pool:
        tickers = None
        if symbols:
            symbols_upper = [s.upper() for s in symbols]
            rows = await pool.fetch(
                """
                select ticker_id, symbol, cik
                  from tickers
                 where active = true
                   and asset_type = 'equity'
                   and symbol = any($1::text[])
                 order by ticker_id
                """,
                symbols_upper,
            )
            tickers = [(r["ticker_id"], r["symbol"], r["cik"]) for r in rows]
            if not tickers:
                print(
                    f"No matching active equities for: {', '.join(symbols_upper)}",
                    file=sys.stderr,
                )
                return 1

        result = await ingest_fundamentals(pool, tickers=tickers)

    print()
    print(f"Status:         {result.status}")
    print(f"Rows inserted:  {result.rows_inserted:,}")
    print(f"Tickers:        {len(result.per_ticker)}")
    if result.skipped_no_cik:
        print(f"Skipped (no CIK): {', '.join(result.skipped_no_cik)}")
    if result.failed_tickers:
        print(f"Failed:         {', '.join(result.failed_tickers)}")
        for tr in result.per_ticker:
            if tr.error and tr.error != "no CIK":
                print(f"  - {tr.symbol}: {tr.error}")
    return 0 if result.status != "failed" else 2


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill fundamentals from SEC EDGAR.")
    p.add_argument(
        "--symbols", nargs="*", default=None,
        help="optional subset of symbols; default = all active equities",
    )
    args = p.parse_args()
    return asyncio.run(amain(symbols=args.symbols))


if __name__ == "__main__":
    raise SystemExit(main())
