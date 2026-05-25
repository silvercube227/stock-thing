"""5-year price backfill for the active universe.

Usage:
    python -m scripts.backfill_prices              # default 5 years
    python -m scripts.backfill_prices --years 10   # longer history
    python -m scripts.backfill_prices --symbols AAPL MSFT  # subset

Run AFTER seed_tickers.sql and backfill_ciks.py. Writes to price_history and
logs a row in ingestion_runs.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date, timedelta

from backend.ingestion.db import pool_context
from backend.ingestion.prices import ingest_full_history

log = logging.getLogger(__name__)


async def amain(start_date: date, symbols: list[str] | None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    log.info("pulling price history from %s", start_date.isoformat())

    async with pool_context() as pool:
        tickers: list[tuple[int, str]] | None = None
        if symbols:
            symbols_upper = [s.upper() for s in symbols]
            rows = await pool.fetch(
                """
                select ticker_id, symbol
                  from tickers
                 where active = true
                   and symbol = any($1::text[])
                 order by ticker_id
                """,
                symbols_upper,
            )
            tickers = [(r["ticker_id"], r["symbol"]) for r in rows]
            if not tickers:
                print(f"No matching active tickers for: {', '.join(symbols_upper)}", file=sys.stderr)
                return 1

        result = await ingest_full_history(pool, tickers=tickers, start_date=start_date)

    print()
    print(f"Status:         {result.status}")
    print(f"Rows inserted:  {result.rows_inserted:,}")
    print(f"Tickers:        {len(result.per_ticker)}")
    if result.failed_tickers:
        print(f"Failed:         {', '.join(result.failed_tickers)}")
        for tr in result.per_ticker:
            if tr.error:
                print(f"  - {tr.symbol}: {tr.error}")
    return 0 if result.status != "failed" else 2


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill price_history from yfinance.")
    p.add_argument("--years", type=int, default=5, help="years of history to pull (default: 5)")
    p.add_argument(
        "--start",
        default=None,
        help="explicit start date YYYY-MM-DD (overrides --years)",
    )
    p.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="optional subset of symbols; default = all active tickers",
    )
    args = p.parse_args()
    start_date = (
        date.fromisoformat(args.start)
        if args.start
        else date.today() - timedelta(days=365 * args.years)
    )
    return asyncio.run(amain(start_date=start_date, symbols=args.symbols))


if __name__ == "__main__":
    raise SystemExit(main())
