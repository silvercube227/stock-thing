"""One-off sanity check after the price backfill. Prints summary stats."""
from __future__ import annotations

import asyncio

from backend.ingestion.db import pool_context


async def amain() -> None:
    async with pool_context() as pool:
        total = await pool.fetchval("select count(*) from price_history")
        per_ticker = await pool.fetch(
            """
            select t.symbol, count(*) as bars,
                   min(trade_date) as first_bar, max(trade_date) as last_bar
              from price_history p
              join tickers t using (ticker_id)
             group by t.symbol
             order by bars desc
            """
        )
        runs = await pool.fetch(
            """
            select run_id, job_name, status, rows_inserted,
                   finished_at - started_at as duration, metadata
              from ingestion_runs
             order by run_id desc
             limit 5
            """
        )

    print(f"\nTotal price_history rows: {total:,}\n")
    print("Per-ticker bar counts:")
    for row in per_ticker:
        print(f"  {row['symbol']:6s} {row['bars']:>5,}  ({row['first_bar']} → {row['last_bar']})")
    print("\nRecent ingestion_runs:")
    for r in runs:
        print(f"  #{r['run_id']:>3} {r['job_name']:<14} {r['status']:<8} rows={r['rows_inserted']!s:<6} {r['duration']}")


if __name__ == "__main__":
    asyncio.run(amain())
