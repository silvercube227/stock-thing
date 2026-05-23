"""Sanity check after the fundamentals backfill."""
from __future__ import annotations

import asyncio

from backend.ingestion.db import pool_context


async def amain() -> None:
    async with pool_context() as pool:
        total = await pool.fetchval("select count(*) from fundamentals")
        per_ticker = await pool.fetch(
            """
            select t.symbol,
                   count(*) as filings,
                   count(*) filter (where filing_type = '10-K') as annual,
                   count(*) filter (where filing_type = '10-Q') as quarterly,
                   min(period_end) as earliest, max(period_end) as latest
              from fundamentals f join tickers t using (ticker_id)
             group by t.symbol order by t.symbol
            """
        )
        sample = await pool.fetch(
            """
            select t.symbol, filing_type, period_end, filed_at, revenue,
                   gross_margin, operating_margin, total_debt, fcf
              from fundamentals f join tickers t using (ticker_id)
             where t.symbol = 'AAPL' and filing_type = '10-K'
             order by period_end desc limit 5
            """
        )
        leak_check = await pool.fetchval(
            "select count(*) from fundamentals where filed_at < period_end"
        )

    print(f"\nTotal fundamentals rows: {total:,}\n")
    print(f"{'symbol':<6} {'filings':>7} {'10-K':>5} {'10-Q':>5}  earliest    latest")
    for r in per_ticker:
        print(
            f"{r['symbol']:<6} {r['filings']:>7} {r['annual']:>5} {r['quarterly']:>5}"
            f"  {r['earliest']}  {r['latest']}"
        )

    print("\nAAPL 10-K sample (most recent 5):")
    print(f"  {'period_end':<12} {'filed_at':<12} {'revenue':>16} {'gm':>6} {'om':>6} {'debt':>14} {'fcf':>14}")
    for r in sample:
        gm = f"{r['gross_margin']:.3f}" if r["gross_margin"] is not None else "—"
        om = f"{r['operating_margin']:.3f}" if r["operating_margin"] is not None else "—"
        debt = f"{r['total_debt']:>14,.0f}" if r["total_debt"] is not None else "—" * 14
        fcf = f"{r['fcf']:>14,.0f}" if r["fcf"] is not None else "—" * 14
        rev = f"{r['revenue']:>16,.0f}" if r["revenue"] is not None else "—" * 16
        print(f"  {r['period_end']!s:<12} {r['filed_at']!s:<12} {rev} {gm:>6} {om:>6} {debt} {fcf}")

    print(f"\nLook-ahead leakage check (filed_at < period_end): {leak_check} rows")
    print("(should be 0 — would indicate a filing claiming to be received BEFORE its reporting period closed)")


if __name__ == "__main__":
    asyncio.run(amain())
