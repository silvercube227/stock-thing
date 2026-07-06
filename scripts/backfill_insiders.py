"""Backfill SEC insider transactions from the DERA quarterly datasets.

Downloads one ZIP per calendar quarter (SEC DERA "Insider Transactions Data Sets",
2006+), parses SUBMISSION / REPORTINGOWNER / NONDERIV_TRANS, keeps only Form 3/4/5
filings whose issuer CIK is in our universe, and upserts to `insider_transactions`.

Prereq: `python -m scripts.backfill_ciks` so `tickers.cik` is populated (the DERA
issuer CIK is the join key).

Run the full history:
    python -m scripts.backfill_insiders                 # 2006q1 .. current

Or a window (fewer downloads while validating):
    python -m scripts.backfill_insiders --start 2010q1 --end 2026q2
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date

from backend.ingestion.db import pool_context
from backend.ingestion.insiders import ingest_insider_backfill, _quarter_list

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _parse_quarter(s: str) -> tuple[int, int]:
    y, q = s.lower().split("q")
    return int(y), int(q)


def _default_end() -> tuple[int, int]:
    # Most recently *completed* quarter (DERA publishes ~1 month after quarter end).
    today = date.today()
    q = (today.month - 1) // 3 + 1
    y = today.year
    q -= 1
    if q < 1:
        q, y = 4, y - 1
    return y, q


async def main(args) -> None:
    sy, sq = _parse_quarter(args.start)
    ey, eq = _parse_quarter(args.end) if args.end else _default_end()
    quarters = _quarter_list(sy, sq, ey, eq)
    log.info("Insider backfill: %s .. %s (%d quarters)",
             f"{sy}q{sq}", f"{ey}q{eq}", len(quarters))
    async with pool_context() as pool:
        result = await ingest_insider_backfill(pool, quarters, concurrency=args.concurrency)
    log.info("Backfill complete: quarters=%d rows=%d",
             result["quarters"], result["rows"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Backfill SEC insider transactions (DERA)")
    p.add_argument("--start", default="2006q1", help="first quarter, e.g. 2010q1")
    p.add_argument("--end", default=None,
                   help="last quarter, e.g. 2026q2 (default: most recent completed)")
    p.add_argument("--concurrency", type=int, default=4,
                   help="parallel DERA ZIP downloads (SEC fair-access: keep low)")
    asyncio.run(main(p.parse_args()))
