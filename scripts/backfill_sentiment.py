"""Bootstrap the sentiment pipeline for all active tickers.

yfinance returns ~30-50 recent headlines per ticker (last ~30 days), so this
is NOT a 5-year backfill — it's an initial seed. Going forward, the daily cron
populates the rolling history.

Usage:
    python -m scripts.backfill_sentiment [--symbols AAPL MSFT ...]

Options:
    --symbols   Only process these symbols (space-separated). Default: all active tickers.
    --dry-run   Fetch and score but do not write to DB.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from backend.ingestion.db import pool_context
from backend.ingestion.headlines import SentimentResult, ingest_sentiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--symbols",
        nargs="+",
        metavar="SYM",
        help="Restrict to these symbols (default: all active tickers)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and score without writing to the database",
    )
    return p.parse_args()


async def _main(args: argparse.Namespace) -> None:
    async with pool_context() as pool:
        tickers: list[tuple[int, str]] | None = None

        if args.symbols:
            rows = await pool.fetch(
                """
                select ticker_id, symbol
                  from tickers
                 where symbol = any($1::text[])
                   and active = true
                 order by symbol
                """,
                args.symbols,
            )
            if not rows:
                log.error("No active tickers found for symbols: %s", args.symbols)
                sys.exit(1)
            tickers = [(r["ticker_id"], r["symbol"]) for r in rows]
            log.info("Restricting to %d symbol(s): %s", len(tickers), [s for _, s in tickers])

        if args.dry_run:
            log.info("DRY RUN — fetching + scoring only, no DB writes")
            from backend.ingestion.headlines import fetch_news_yf, score_texts

            sym_list = [s for _, s in tickers] if tickers else []
            if not sym_list:
                rows = await pool.fetch(
                    "select symbol from tickers where active = true order by symbol"
                )
                sym_list = [r["symbol"] for r in rows]

            total_headlines = 0
            for sym in sym_list:
                items = fetch_news_yf(sym)
                log.info("%s: %d headlines fetched", sym, len(items))
                total_headlines += len(items)
            log.info("Total headlines fetched: %d (dry run, no scoring/writes)", total_headlines)
            return

        result: SentimentResult = await ingest_sentiment(pool, tickers=tickers)

    log.info("--- Backfill sentiment complete ---")
    log.info("Status              : %s", result.status)
    log.info("Headlines inserted  : %d", result.headlines_inserted)
    log.info("Sentiment days upserted: %d", result.sentiment_days_upserted)
    log.info("Tickers processed   : %d", len(result.per_ticker))
    if result.failed_tickers:
        log.warning("Failed tickers (%d): %s", len(result.failed_tickers), result.failed_tickers)

    for tr in result.per_ticker:
        status = f"fetched={tr.headlines_fetched} inserted={tr.headlines_inserted} days={tr.sentiment_days_updated}"
        if tr.error:
            log.warning("  %-8s  ERROR: %s", tr.symbol, tr.error)
        else:
            log.info("  %-8s  %s", tr.symbol, status)


def main() -> None:
    args = _parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
