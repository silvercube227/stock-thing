"""Seed the current S&P 500 constituents into the `tickers` table.

Fetches the maintained constituent list from Wikipedia, normalizes symbols for
yfinance/SEC (BRK.B -> BRK-B), and upserts into `tickers` as equities.

Idempotent: ON CONFLICT DO NOTHING, so already-seeded tickers keep their
permanent `embedding_idx` (removing a name later means active=false, never a
delete — survivorship bias). CIK is left NULL and filled afterward by
scripts/backfill_ciks.py.

Usage:
    python -m scripts.seed_sp500              # fetch + insert
    python -m scripts.seed_sp500 --dry-run    # parse + report, no DB writes

Run order for a full universe expansion:
    seed_sp500  ->  backfill_ciks  ->  backfill_prices  ->  backfill_fundamentals
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import bs4
import httpx

from backend.config import get_settings
from backend.ingestion.db import pool_context

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def normalize_symbol(raw: str) -> str:
    """Wikipedia uses dotted class shares (BRK.B); yfinance/SEC use dashes."""
    return raw.strip().upper().replace(".", "-")


def parse_constituents(html: str) -> list[dict]:
    """Extract [{symbol, name, sector, industry}] from the Wikipedia table.

    Targets the `#constituents` wikitable; falls back to the first wikitable.
    Uses the stdlib html.parser so no lxml/html5lib dependency is required.
    """
    soup = bs4.BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="constituents") or soup.find("table", class_="wikitable")
    if table is None:
        raise ValueError("could not locate the constituents table on the page")

    out: list[dict] = []
    seen: set[str] = set()
    for tr in table.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 4:
            continue  # header / malformed row
        symbol = normalize_symbol(cells[0].get_text())
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(
            {
                "symbol": symbol,
                "name": cells[1].get_text(strip=True) or None,
                "sector": cells[2].get_text(strip=True) or None,
                "industry": cells[3].get_text(strip=True) or None,
            }
        )
    return out


async def fetch_constituents() -> list[dict]:
    settings = get_settings()
    # Wikipedia rejects requests without a descriptive User-Agent.
    headers = {"User-Agent": f"stock-thing-seed/1.0 ({settings.sec_edgar_user_agent})"}
    async with httpx.AsyncClient(timeout=30.0, headers=headers, follow_redirects=True) as client:
        resp = await client.get(WIKI_URL)
        resp.raise_for_status()
    return parse_constituents(resp.text)


_INSERT_SQL = """
insert into tickers (symbol, name, asset_type, sector, industry)
values ($1, $2, 'equity', $3, $4)
on conflict do nothing
"""


async def amain(dry_run: bool) -> int:
    rows = await fetch_constituents()
    print(f"Parsed {len(rows)} constituents from Wikipedia.")
    if not rows:
        print("No rows parsed — page layout may have changed.", file=sys.stderr)
        return 1

    sample = ", ".join(r["symbol"] for r in rows[:8])
    print(f"  sample: {sample} ...")

    if dry_run:
        print("--dry-run: no DB writes.")
        return 0

    async with pool_context() as pool:
        before = await pool.fetchval("select count(*) from tickers")
        await pool.executemany(
            _INSERT_SQL,
            [(r["symbol"], r["name"], r["sector"], r["industry"]) for r in rows],
        )
        after = await pool.fetchval("select count(*) from tickers")
        active = await pool.fetchval("select count(*) from tickers where active")

    print(f"Inserted {after - before} new ticker(s) ({len(rows) - (after - before)} already present).")
    print(f"Active tickers now: {active}")
    print("Next: python -m scripts.backfill_ciks")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Seed S&P 500 constituents from Wikipedia.")
    p.add_argument("--dry-run", action="store_true", help="parse and report without writing to the DB")
    args = p.parse_args()
    return asyncio.run(amain(dry_run=args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
