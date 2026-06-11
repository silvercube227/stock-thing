"""Backfill FINRA short interest (all available history) via the FINRA Open API.

The FINRA API returns ~3.8M records across all bimonthly settlement dates from
2009-present. This script pages through the full dataset and upserts to the
`short_interest` table.

Run once to populate the historical data:
    python -m scripts.backfill_short_interest

For incremental updates (after a new settlement date):
    python -m scripts.backfill_short_interest --after 2026-05-15

Note: the full backfill downloads ~3.8M records in ~764 API pages. Each page
is an HTTP request to api.finra.org; expect ~15–30 minutes.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date

from backend.ingestion.db import pool_context
from backend.ingestion.short_interest import ingest_short_interest_full

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


async def main(args) -> None:
    after = date.fromisoformat(args.after) if args.after else None
    log.info("Starting FINRA short interest backfill (after=%s)", after)

    async with pool_context() as pool:
        result = await ingest_short_interest_full(pool, after_date=after)

    log.info("Backfill complete: fetched=%d upserted=%d",
             result["fetched"], result["upserted"])


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Backfill FINRA short interest")
    p.add_argument("--after", default=None,
                   help="skip records on or before this settlement date YYYY-MM-DD "
                        "(for incremental; omit for full history)")
    asyncio.run(main(p.parse_args()))
