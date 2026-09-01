"""Build point-in-time S&P 500 membership intervals into `index_membership`.

De-survivorshipping added removed-from-index names to the panel, but INCLUSION was
still not point-in-time: a name added to the index in 2023 has price history back to
2016 and appeared in the 2017 cross-sections. That selects on future index promotion
— the mirror image of survivorship bias — and it is arguably the larger of the two,
because promotion into the index follows a run of good performance.

Sources, in order of authority:
  1. The constituents table's "Date added" column — a real, per-name addition date
     for every CURRENT member. This is the inclusion-side fix.
  2. `tickers.removed_at` — the removal date captured when the historical seed ran.
  3. The "Selected changes" wikitable, when present. Wikipedia removed it from the
     page (2026), so this is treated as optional: absent, we fall back to 1 + 2.

Interval rules (valid_to is EXCLUSIVE; null = still a member):
  * current member with a known "Date added"
                             -> [date_added, null)         source 'wikipedia_date_added'
  * current member added before the window (or with an unparseable date)
                             -> [2010-01-01, null)         source 'assumed_pre_window'
  * removed name             -> [2010-01-01, removed_at)   source 'assumed_start'
                                (no addition date survives for these)
  * a change-table row that adds a name back after a removal opens a second
    interval, which is why the PK is (ticker_id, valid_from)

Full refresh: deletes and rebuilds every row, so it is safe to re-run.

Known gaps, both conservative (a name is absent from the panel rather than wrongly
present): names removed before the historical seed's window were never inserted into
`tickers`, so pre-2016 membership is incomplete on the removal side; and removed
names carry an assumed rather than observed start.

Usage:
    python -m scripts.seed_index_membership [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from datetime import date, datetime

import bs4

from backend.ingestion.db import pool_context
from backend.ingestion.prices import HISTORY_START
from scripts.seed_sp500 import normalize_symbol
from scripts.seed_sp500_historical import fetch_wiki_html, parse_change_rows

# Intervals cannot start before the panel does; anything earlier is unobservable.
WINDOW_START: date = HISTORY_START

_DATE_FORMATS = ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%Y")


def _parse_added_date(raw: str | None) -> date | None:
    if not raw:
        return None
    text = " ".join(raw.split()).split("(")[0].strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_constituents_with_dates(html: str) -> dict[str, date | None]:
    """{symbol -> index addition date} from the constituents table.

    Columns are located by header text, not position — the table has gained and
    lost columns over the years (it now carries CIK and Founded as well).
    """
    soup = bs4.BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="constituents") or soup.find("table", class_="wikitable")
    if table is None:
        raise ValueError("could not locate the constituents table")

    header_cells: list[str] = []
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
    i_added = find_col("date added", "date first added")
    if i_symbol is None:
        raise ValueError(f"unexpected header layout: {header_cells}")

    out: dict[str, date | None] = {}
    for tr in table.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) <= i_symbol:
            continue
        symbol = normalize_symbol(cells[i_symbol].get_text(strip=True))
        if not symbol or symbol in out:
            continue
        raw = (
            cells[i_added].get_text(strip=True)
            if i_added is not None and i_added < len(cells)
            else None
        )
        out[symbol] = _parse_added_date(raw)
    return out


def build_intervals(
    added_dates: dict[str, date | None],
    removed_at: dict[str, date],
    known_symbols: set[str],
    change_rows: list[dict] | None = None,
    window_start: date = WINDOW_START,
    still_active: set[str] | None = None,
    observed_on: date | None = None,
) -> tuple[dict[str, list[dict]], list[str]]:
    """{symbol -> [{valid_from, valid_to, source}]} plus a list of anomalies.

    Pure function so the interval logic is testable without network or DB.
    `added_dates` covers current members, `removed_at` the removed cohort, and
    `known_symbols` is the set present in `tickers` (everything else is ignored).
    `change_rows` is optional extra history from the "Selected changes" table.

    `still_active` names are flagged active in our own table. Any of them missing
    from today's constituent list left the index at a date we never recorded (the
    changes table that used to surface removals is gone). Dropping them outright
    would erase years of legitimate membership, so they get an interval closed at
    `observed_on` — "a member until we noticed it was gone" — and are reported.
    """
    intervals: dict[str, list[dict]] = {}
    anomalies: list[str] = []

    # Current members: open interval from the observed addition date.
    for symbol, added in added_dates.items():
        if symbol not in known_symbols:
            continue
        if added is not None and added > window_start:
            intervals[symbol] = [{"valid_from": added, "valid_to": None,
                                  "source": "wikipedia_date_added"}]
        else:
            # Added before the panel starts (or undated): a member throughout.
            intervals[symbol] = [{"valid_from": window_start, "valid_to": None,
                                  "source": "assumed_pre_window"}]

    # Removed names: closed interval ending at the observed removal date. No
    # addition date survives for these, so the start is assumed.
    for symbol, removed in removed_at.items():
        if symbol not in known_symbols or symbol in intervals:
            continue
        if removed <= window_start:
            anomalies.append(f"{symbol} (removed {removed} before window)")
            continue
        intervals[symbol] = [{"valid_from": window_start, "valid_to": removed,
                              "source": "assumed_start"}]

    # A name that was removed and later re-added shows up in BOTH maps. The
    # current-member interval above already covers the live stretch; add the
    # earlier closed stretch so the gap between them is correctly excluded.
    for symbol, removed in removed_at.items():
        if symbol not in known_symbols:
            continue
        existing = intervals.get(symbol) or []
        live = [iv for iv in existing if iv["valid_to"] is None]
        if not live or removed <= window_start:
            continue
        start = live[0]["valid_from"]
        if removed < start:
            intervals[symbol] = [
                {"valid_from": window_start, "valid_to": removed,
                 "source": "assumed_start"},
                *existing,
            ]

    # Flagged active here, absent from the live constituent list: an unrecorded
    # removal. Close at the observation date rather than dropping the name.
    if still_active and observed_on is not None:
        for symbol in sorted(still_active):
            if symbol not in known_symbols or symbol in intervals:
                continue
            intervals[symbol] = [{"valid_from": window_start, "valid_to": observed_on,
                                  "source": "observed_absent"}]
            anomalies.append(f"{symbol} (active here, absent from index list)")

    for row in change_rows or []:
        symbol = row.get("added_symbol")
        if symbol and symbol in known_symbols and symbol not in intervals:
            anomalies.append(f"{symbol} (in changes table, not in current list)")

    return {s: rows for s, rows in intervals.items() if rows}, anomalies


async def amain(args: argparse.Namespace) -> int:
    html = await fetch_wiki_html()
    added_dates = parse_constituents_with_dates(html)
    dated = sum(1 for d in added_dates.values() if d is not None)
    print(f"{len(added_dates)} current constituent(s), {dated} with an addition date")

    # Optional: Wikipedia removed this table from the page in 2026.
    try:
        change_rows = parse_change_rows(html)
        print(f"{len(change_rows)} change row(s) from the Selected-changes table")
    except ValueError:
        change_rows = []
        print("Selected-changes table not present; using date-added + removed_at only")

    async with pool_context() as pool:
        rows = await pool.fetch(
            """
            select ticker_id, symbol, removed_at, active, user_added
              from tickers
             where asset_type = 'equity'
            """
        )
        by_symbol = {r["symbol"].upper(): r["ticker_id"] for r in rows}
        removed_at = {
            r["symbol"].upper(): r["removed_at"].date()
            for r in rows if r["removed_at"] is not None
        }
        still_active = {
            r["symbol"].upper() for r in rows
            if r["active"] and not r["user_added"]
        }
        print(f"{len(removed_at)} removed-from-index ticker(s) with a removal date")

        intervals, anomalies = build_intervals(
            added_dates, removed_at, set(by_symbol), change_rows,
            still_active=still_active, observed_on=date.today(),
        )
        total = sum(len(v) for v in intervals.values())
        multi = sum(1 for v in intervals.values() if len(v) > 1)
        sources: dict[str, int] = defaultdict(int)
        for ivs in intervals.values():
            for iv in ivs:
                sources[iv["source"]] += 1

        print(f"{total} interval(s) across {len(intervals)} ticker(s) "
              f"({multi} with more than one)")
        for source, n in sorted(sources.items()):
            print(f"  {source}: {n}")
        missing = sorted(set(by_symbol) - set(intervals))
        print(f"{len(missing)} ticker(s) with no interval "
              f"(never in the index during the window)")
        if anomalies:
            print(f"Anomalies ({len(anomalies)}): {'; '.join(sorted(anomalies)[:15])}")

        if args.dry_run:
            print("[dry-run] no rows written")
            return 0

        payload = [
            (by_symbol[symbol], iv["valid_from"], iv["valid_to"], iv["source"])
            for symbol, ivs in intervals.items()
            for iv in ivs
        ]
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("delete from index_membership")
                await conn.executemany(
                    """
                    insert into index_membership (ticker_id, valid_from, valid_to, source)
                    values ($1, $2, $3, $4)
                    """,
                    payload,
                )
        print(f"Wrote {len(payload)} membership interval(s).")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true", help="report without writing")
    return asyncio.run(amain(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
