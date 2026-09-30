"""Backfill GICS sector/industry for the removed-from-index cohort.

`scripts/seed_sp500_historical.py` inserts removed names with only symbol/name/
removed_at, so the whole cohort has a NULL sector. That has two costs in the ML
layer: `within_sector_ic` (the headline SECB metric) drops null-sector rows, so
the cohort is invisible to the metric it should be stressing; and
`apply_target_modes` silently falls back to a universe-demeaned target for them.

Wikipedia's current constituents page only lists CURRENT members, so we read the
page REVISION as it stood just before each name's removal — that snapshot still
lists the name with its GICS sector. Revisions are resolved via the MediaWiki API
and deduped to month granularity so a few dozen fetches cover the cohort.

Fallbacks, in order: Wikipedia snapshot -> yfinance .info (via the existing
_YAHOO_TO_GICS map) -> scripts/data/removed_ticker_overrides.csv.

A sector is only written if it matches a label already used in `tickers` — a
wrong peer group is worse than a null one (same rule as backend/jobs/add_ticker.py).

Usage:
    python -m scripts.backfill_sectors [--dry-run] [--no-yfinance]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from datetime import date, timedelta
from pathlib import Path

import bs4
import httpx

from backend.config import get_settings
from backend.ingestion.db import pool_context
from backend.jobs.add_ticker import _YAHOO_TO_GICS
from scripts.backfill_ciks import OVERRIDES_CSV
from scripts.seed_sp500 import WIKI_URL, normalize_symbol

MEDIAWIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_OLDID_URL = "https://en.wikipedia.org/w/index.php?oldid={revid}"
WIKI_PAGE_TITLE = "List of S&P 500 companies"

# Sector labels used by older revisions of the page, mapped to today's GICS names.
# Consistent with migration 006_normalize_sectors.sql.
_LEGACY_SECTOR_ALIASES = {
    "Telecommunication Services": "Communication Services",
    "Telecommunications Services": "Communication Services",
    "Healthcare": "Health Care",
    "Technology": "Information Technology",
    "Consumer Discretionary ": "Consumer Discretionary",
}


def canonical_sector(raw: str | None) -> str | None:
    if not raw:
        return None
    s = " ".join(raw.split())
    return _LEGACY_SECTOR_ALIASES.get(s, s)


def parse_constituents_by_header(html: str) -> dict[str, dict]:
    """{symbol -> {name, sector, industry}} from any revision of the page.

    Column positions moved over the years (older revisions carry an extra
    "SEC filings" column), so locate the GICS columns by HEADER TEXT rather than
    by index — parsing by position silently yields the wrong field on old revisions.
    """
    soup = bs4.BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="constituents") or soup.find("table", class_="wikitable")
    if table is None:
        raise ValueError("could not locate the constituents table")

    header_cells = None
    for tr in table.find_all("tr"):
        ths = tr.find_all("th")
        if len(ths) >= 3:
            header_cells = [" ".join(th.get_text(strip=True).split()).lower() for th in ths]
            break
    if not header_cells:
        raise ValueError("could not locate the constituents table header")

    def find_col(*needles: str) -> int | None:
        for i, h in enumerate(header_cells):
            if any(n in h for n in needles):
                return i
        return None

    i_symbol = find_col("symbol", "ticker")
    i_name = find_col("security", "company")
    i_sector = find_col("gics sector", "sector")
    i_industry = find_col("sub-industry", "sub industry", "industry")
    if i_symbol is None or i_sector is None:
        raise ValueError(f"unexpected header layout: {header_cells}")

    def cell(cells, idx):
        if idx is None or idx >= len(cells):
            return None
        return cells[idx].get_text(strip=True) or None

    out: dict[str, dict] = {}
    for tr in table.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) <= max(i for i in (i_symbol, i_sector) if i is not None):
            continue
        symbol = normalize_symbol(cell(cells, i_symbol) or "")
        if not symbol or symbol in out:
            continue
        out[symbol] = {
            "name": cell(cells, i_name),
            "sector": canonical_sector(cell(cells, i_sector)),
            "industry": cell(cells, i_industry),
        }
    return out


async def revision_as_of(client: httpx.AsyncClient, when: date) -> int | None:
    """Newest revision id of the constituents page at or before `when`."""
    params = {
        "action": "query", "format": "json", "prop": "revisions",
        "titles": WIKI_PAGE_TITLE, "rvlimit": 1, "rvdir": "older",
        "rvstart": f"{when.isoformat()}T00:00:00Z", "rvprop": "ids|timestamp",
    }
    resp = await client.get(MEDIAWIKI_API, params=params)
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        revs = page.get("revisions") or []
        if revs:
            return int(revs[0]["revid"])
    return None


async def snapshot_for(client: httpx.AsyncClient, when: date) -> dict[str, dict]:
    revid = await revision_as_of(client, when)
    if revid is None:
        return {}
    resp = await client.get(WIKI_OLDID_URL.format(revid=revid))
    resp.raise_for_status()
    try:
        return parse_constituents_by_header(resp.text)
    except ValueError as exc:
        print(f"  {when}: revision {revid} unparsed ({exc})")
        return {}


def load_sector_overrides(path: Path = OVERRIDES_CSV) -> dict[str, dict]:
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            symbol = (row.get("symbol") or "").strip().upper()
            sector = (row.get("sector") or "").strip()
            if symbol and sector:
                out[symbol] = {"sector": canonical_sector(sector),
                               "industry": (row.get("industry") or "").strip() or None}
    return out


def _yf_sector(symbol: str) -> tuple[str | None, str | None]:
    """(sector, industry) from yfinance .info; often empty for delisted names."""
    try:
        import yfinance as yf

        info = yf.Ticker(symbol).info or {}
    except Exception:
        return None, None
    return _YAHOO_TO_GICS.get(info.get("sector")), info.get("industry") or None


async def amain(args: argparse.Namespace) -> int:
    settings = get_settings()
    headers = {"User-Agent": f"stock-thing-seed/1.0 ({settings.sec_edgar_user_agent})"}

    async with pool_context() as pool:
        rows = await pool.fetch(
            """
            select ticker_id, symbol, removed_at
              from tickers
             where asset_type = 'equity'
               and sector is null
             order by ticker_id
            """
        )
        known_sectors = {
            r["sector"] for r in
            await pool.fetch("select distinct sector from tickers where sector is not null")
        }
        print(f"{len(rows)} equity ticker(s) without a sector; "
              f"{len(known_sectors)} known sector labels")
        if not rows:
            return 0

        # Group the cohort by the month before removal so each revision is fetched once.
        by_month: dict[date, list] = {}
        undated: list = []
        for row in rows:
            removed_at = row["removed_at"]
            if removed_at is None:
                undated.append(row)
                continue
            target = removed_at.date() - timedelta(days=7)
            by_month.setdefault(date(target.year, target.month, 1), []).append(row)

        resolved: dict[int, dict] = {}
        overrides = load_sector_overrides()

        async with httpx.AsyncClient(timeout=60.0, headers=headers,
                                     follow_redirects=True) as client:
            print(f"Fetching {len(by_month)} Wikipedia revision(s)...")
            for month, month_rows in sorted(by_month.items()):
                snapshot = await snapshot_for(client, month)
                if not snapshot:
                    continue
                hits = 0
                for row in month_rows:
                    entry = snapshot.get(row["symbol"].upper())
                    if entry and entry.get("sector"):
                        resolved[row["ticker_id"]] = entry
                        hits += 1
                print(f"  {month}: {hits}/{len(month_rows)} resolved "
                      f"({len(snapshot)} names in snapshot)")

        missing = [r for r in rows if r["ticker_id"] not in resolved]
        print(f"Wikipedia snapshots resolved {len(resolved)}/{len(rows)}")

        # Fallback 2: yfinance (works for names that still trade off-index).
        if missing and not args.no_yfinance:
            print(f"yfinance fallback for {len(missing)} name(s)...")
            hits = 0
            for row in missing:
                sector, industry = await asyncio.to_thread(_yf_sector, row["symbol"])
                if sector:
                    resolved[row["ticker_id"]] = {"sector": sector, "industry": industry}
                    hits += 1
            print(f"  resolved {hits}")

        # Fallback 3: manual overrides CSV.
        for row in rows:
            if row["ticker_id"] in resolved:
                continue
            entry = overrides.get(row["symbol"].upper())
            if entry:
                resolved[row["ticker_id"]] = entry

        # Only write labels that already exist in the table — never invent a peer group.
        writes, rejected = [], []
        for row in rows:
            entry = resolved.get(row["ticker_id"])
            if not entry or not entry.get("sector"):
                continue
            if entry["sector"] not in known_sectors:
                rejected.append((row["symbol"], entry["sector"]))
                continue
            writes.append((row["ticker_id"], entry["sector"], entry.get("industry")))

        if rejected:
            print(f"Rejected {len(rejected)} unknown sector label(s): "
                  f"{sorted({s for _, s in rejected})}")

        if args.dry_run:
            print(f"[dry-run] would set sector for {len(writes)} ticker(s)")
        else:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    for ticker_id, sector, industry in writes:
                        await conn.execute(
                            """
                            update tickers
                               set sector = coalesce(sector, $2),
                                   industry = coalesce(industry, $3)
                             where ticker_id = $1
                            """,
                            ticker_id, sector, industry,
                        )
            print(f"Updated sector for {len(writes)} ticker(s).")

        still = [r["symbol"] for r in rows if r["ticker_id"] not in
                 {w[0] for w in writes} and r["ticker_id"] not in resolved]
        if still:
            print(f"\nNo sector for {len(still)}: {', '.join(sorted(still))}")
            print(f"  Add them to {OVERRIDES_CSV}.")
        if undated:
            print(f"({len(undated)} had no removed_at and were snapshot-skipped)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true", help="report without writing")
    p.add_argument("--no-yfinance", action="store_true",
                   help="skip the yfinance fallback (offline / faster)")
    return asyncio.run(amain(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
