"""Seed historical S&P 500 constituents that have since been removed from the index.

Parses the 'Selected changes' table on the Wikipedia S&P 500 page to find
tickers removed since --since (default 2016-01-01), inserts them as
active=false with removed_at set, then backfills price history via yfinance.

Complements seed_sp500.py which seeds current constituents. Together they
eliminate the forward-looking survivorship bias from training only on today's
S&P 500 winners.

Run order:
    seed_sp500 -> seed_sp500_historical -> backfill_ciks -> backfill_prices -> ...

Usage:
    python -m scripts.seed_sp500_historical               # since 2016-01-01
    python -m scripts.seed_sp500_historical --since 2010  # further back
    python -m scripts.seed_sp500_historical --dry-run     # parse only, no DB writes
    python -m scripts.seed_sp500_historical --no-prices   # insert tickers but skip price backfill
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, datetime

import bs4
import httpx

from backend.config import get_settings
from backend.ingestion.db import pool_context
from backend.ingestion.prices import DEFAULT_CONCURRENCY, ingest_full_history

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def normalize_symbol(raw: str) -> str:
    return raw.strip().upper().replace(".", "-")


_EMPTY_CELL = ("", "—", "-", "–")


def parse_change_rows(html: str) -> list[dict]:
    """Every index change event, keeping BOTH sides of each row.

    The changes table is the second wikitable on the S&P 500 page. Rows have
    six columns: Date | Added Ticker | Added Name | Removed Ticker | Removed Name | Reason.
    Either side may be empty (an addition-only or removal-only event).

    Returns [{date, added_symbol, added_name, removed_symbol, removed_name}] sorted
    ascending by date. The additions are what `scripts/seed_index_membership.py`
    needs to reconstruct point-in-time membership; this seed script only consumes
    the removals (see `parse_removed_tickers`).
    """
    soup = bs4.BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table", class_="wikitable")
    if len(tables) < 2:
        raise ValueError("could not find historical changes table (expected >=2 wikitables)")
    changes_table = tables[1]

    def cell_text(cells, idx: str | int) -> str | None:
        if idx >= len(cells):
            return None
        text = cells[idx].get_text(strip=True)
        return None if not text or text in _EMPTY_CELL else text

    out: list[dict] = []
    for tr in changes_table.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 4:
            continue  # header or malformed row

        raw_date = cells[0].get_text(strip=True)
        change_date: date | None = None
        for fmt in ("%B %d, %Y", "%Y-%m-%d", "%b %d, %Y"):
            try:
                change_date = datetime.strptime(raw_date, fmt).date()
                break
            except ValueError:
                continue
        if change_date is None:
            continue

        added_raw = cell_text(cells, 1)
        removed_raw = cell_text(cells, 3)
        out.append({
            "date": change_date,
            "added_symbol": normalize_symbol(added_raw) if added_raw else None,
            "added_name": cell_text(cells, 2),
            "removed_symbol": normalize_symbol(removed_raw) if removed_raw else None,
            "removed_name": cell_text(cells, 4),
        })

    out.sort(key=lambda r: r["date"])
    return out


def parse_removed_tickers(html: str, since: date) -> list[dict]:
    """Extract [{symbol, name, removed_at}] for removals on or after `since`."""
    return [
        {
            "symbol": row["removed_symbol"],
            "name": row["removed_name"],
            "removed_at": row["date"],
        }
        for row in parse_change_rows(html)
        if row["removed_symbol"] and row["date"] >= since
    ]


async def fetch_wiki_html() -> str:
    settings = get_settings()
    headers = {"User-Agent": f"stock-thing-seed/1.0 ({settings.sec_edgar_user_agent})"}
    async with httpx.AsyncClient(timeout=30.0, headers=headers, follow_redirects=True) as client:
        resp = await client.get(WIKI_URL)
        resp.raise_for_status()
    return resp.text


# Insert only if the symbol doesn't already exist in any state (active or inactive).
# We avoid inserting a second row for a ticker that's already in the DB (e.g., it was
# removed, re-added, and we seeded it as active=true via seed_sp500.py).
_INSERT_SQL = """
insert into tickers (symbol, name, asset_type, active, removed_at)
select $1, $2, 'equity', false, $3
where not exists (select 1 from tickers where symbol = $1)
"""


async def amain(since: date, dry_run: bool, no_prices: bool, concurrency: int) -> int:
    print(f"Fetching Wikipedia S&P 500 page ...")
    html = await fetch_wiki_html()

    removed = parse_removed_tickers(html, since=since)
    print(f"Found {len(removed)} removals since {since}.")
    if not removed:
        print("Nothing to insert.", file=sys.stderr)
        return 0

    for r in removed[:10]:
        print(f"  {r['symbol']} ({r['removed_at']}): {r['name']}")
    if len(removed) > 10:
        print(f"  ... and {len(removed) - 10} more")

    if dry_run:
        print("--dry-run: no DB writes.")
        return 0

    async with pool_context() as pool:
        before = await pool.fetchval("select count(*) from tickers")

        await pool.executemany(
            _INSERT_SQL,
            [(r["symbol"], r["name"], r["removed_at"]) for r in removed],
        )

        after = await pool.fetchval("select count(*) from tickers")
        inserted = after - before
        print(f"Inserted {inserted} new historical ticker(s) ({len(removed) - inserted} already present).")

        if inserted == 0:
            print("No new tickers; skipping price backfill.")
            return 0

        if no_prices:
            print("--no-prices: skipping price backfill.")
            return 0

        # Fetch the newly inserted tickers so we can pass them to the backfill.
        new_rows = await pool.fetch(
            "select ticker_id, symbol from tickers "
            "where active = false and symbol = any($1::text[])",
            [r["symbol"] for r in removed],
        )
        new_tickers = [(r["ticker_id"], r["symbol"]) for r in new_rows]
        print(f"Backfilling price history for {len(new_tickers)} historical tickers ...")

        result = await ingest_full_history(
            pool,
            tickers=new_tickers,
            start_date=since,
            concurrency=concurrency,
        )
        print(
            f"Price backfill complete: {result.rows_inserted} rows inserted, "
            f"{len(result.failed_tickers)} failed."
        )
        if result.failed_tickers:
            print(f"  Failed: {', '.join(result.failed_tickers)}")
            print("  (failures are normal for acquired/delisted tickers with no yfinance data)")

    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Seed historical S&P 500 removals")
    p.add_argument(
        "--since", default="2016-01-01",
        help="include removals on or after this date (default: 2016-01-01)",
    )
    p.add_argument("--dry-run", action="store_true", help="parse and print, no DB writes")
    p.add_argument("--no-prices", action="store_true", help="insert tickers but skip price backfill")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                   help="concurrent yfinance fetches (default: 4)")
    args = p.parse_args()

    try:
        since = date.fromisoformat(args.since)
    except ValueError:
        # Accept bare year: "2016" -> 2016-01-01
        since = date(int(args.since), 1, 1)

    return asyncio.run(amain(since, args.dry_run, args.no_prices, args.concurrency))


if __name__ == "__main__":
    raise SystemExit(main())
