"""Backfill LSEG analyst estimates for active equity tickers.

Usage:
    python -m scripts.backfill_estimates                  # all active equities
    python -m scripts.backfill_estimates --symbols AAPL MSFT

Requires Workspace running locally and LSEG_APP_KEY in .env. Start small
(--symbols AAPL MSFT) to confirm RIC resolution + field mapping before the
full universe.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from backend.ingestion.db import pool_context
from backend.ingestion.estimates import ingest_estimates


async def amain(symbols: list[str] | None, missing_only: bool = False) -> int:
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
                select ticker_id, symbol
                  from tickers
                 where asset_type = 'equity'
                   and symbol = any($1::text[])
                 order by ticker_id
                """,
                symbols_upper,
            )
            tickers = [(r["ticker_id"], r["symbol"]) for r in rows]
            if not tickers:
                print(
                    f"No matching equities for: {', '.join(symbols_upper)}",
                    file=sys.stderr,
                )
                return 1
        elif missing_only:
            rows = await pool.fetch(
                """
                select distinct t.ticker_id, t.symbol
                  from tickers t
                  join price_history p using (ticker_id)
                  left join analyst_estimates e on e.ticker_id = t.ticker_id
                 where t.asset_type = 'equity' and e.ticker_id is null
                 order by t.symbol
                """
            )
            tickers = [(r["ticker_id"], r["symbol"]) for r in rows]
            if not tickers:
                print("No equities with price history are missing estimates — nothing to do.")
                return 0
            print(f"--missing-only: pulling estimates for {len(tickers)} uncovered equities.")

        result = await ingest_estimates(pool, tickers=tickers)

    print()
    print(f"Status:           {result.status}")
    print(f"Rows inserted:    {result.rows_inserted:,}")
    print(f"Surprise rows:    {result.surprise_rows_inserted:,}")
    print(f"Tickers:          {len(result.per_ticker)}")
    if result.skipped_no_ric:
        print(f"Skipped (no RIC): {', '.join(result.skipped_no_ric)}")
    if result.failed_tickers:
        print(f"Failed:           {', '.join(result.failed_tickers)}")
        for tr in result.per_ticker:
            if tr.error and tr.error != "no RIC":
                print(f"  - {tr.symbol}: {tr.error}")
    return 0 if result.status != "failed" else 2


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill analyst estimates from LSEG.")
    p.add_argument(
        "--symbols", nargs="*", default=None,
        help="optional subset of symbols; default = all panel equities (active + removed)",
    )
    p.add_argument(
        "--missing-only", action="store_true",
        help="only pull equities (with price history) that have no estimates yet — "
             "e.g. removed-from-index names skipped earlier",
    )
    args = p.parse_args()
    return asyncio.run(amain(symbols=args.symbols, missing_only=args.missing_only))


if __name__ == "__main__":
    raise SystemExit(main())
