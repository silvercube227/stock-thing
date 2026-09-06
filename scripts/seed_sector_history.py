"""Build point-in-time GICS sector intervals from Wikipedia page revisions.

`tickers.sector` is one CURRENT label stretched over a name's entire history, and it
drives both the training target (`sector_return` / `sector_grade`) and the headline
metric (`within_sector_ic`). Applying today's GICS to 2012 means the model was trained
against, and scored on, peer groups that did not exist — most visibly across the Sept
2018 Communication Services restructuring and the Sept 2016 Real Estate split.

This walks the "List of S&P 500 companies" page quarterly from 2010, reading the GICS
column out of each revision (present back to at least 2010-01-10), and merges runs of
equal labels into intervals in `sector_history`.

Reuses `scripts/backfill_sectors.py` for the network + parsing layer: `snapshot_for`
resolves a revision id and header-parses the constituents table (column positions moved
over the years, so parsing is by header text, not index), and `canonical_sector` folds
legacy labels ("Telecommunication Services", "Technology") onto today's GICS names.

Provenance, recorded per interval in `source`:
  wikipedia_snapshot          observed in a revision at that date
  assumed_pre_first_snapshot  back-fill of the earliest observed label to the floor
  tickers_current             the live tickers.sector, when it disagrees with the last
                              observed label (the page's current state is authoritative)
  tickers_only                never seen in any snapshot; one interval from tickers

Full refresh, in a transaction: this table is derived, never hand-edited.

Usage:
    python -m scripts.seed_sector_history [--start 2010-01-01] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import date, timedelta

import httpx

from backend.config import get_settings
from backend.ingestion.db import pool_context
from scripts.backfill_sectors import canonical_sector, snapshot_for

HISTORY_FLOOR = date(2010, 1, 1)


def quarterly_dates(start: date, end: date) -> list[date]:
    """First-of-quarter dates in [start, end], plus `end` itself."""
    out: list[date] = []
    y, q = start.year, (start.month - 1) // 3
    while True:
        d = date(y, q * 3 + 1, 1)
        if d > end:
            break
        if d >= start:
            out.append(d)
        q += 1
        if q == 4:
            q, y = 0, y + 1
    if not out or out[-1] != end:
        out.append(end)
    return out


def build_intervals(
    observations: list[tuple[date, str | None, str | None]],
    floor: date,
    current: tuple[str | None, str | None],
) -> list[dict]:
    """Merge per-snapshot (date, sector, industry) observations into intervals.

    `observations` must be ascending by date and may contain gaps (a name absent from
    a snapshot simply has no observation there — being out of the index does not change
    its sector). Consecutive equal SECTOR labels merge; the industry of the run's first
    observation is carried, since sub-industry is missing from pre-2016 revisions.
    """
    seen = [(d, s, i) for d, s, i in observations if s]
    if not seen:
        sector, industry = current
        return ([{"valid_from": floor, "valid_to": None, "sector": sector,
                  "industry": industry, "source": "tickers_only"}]
                if sector else [])

    runs: list[dict] = []
    for d, sector, industry in seen:
        if runs and runs[-1]["sector"] == sector:
            continue
        runs.append({"valid_from": d, "valid_to": None, "sector": sector,
                     "industry": industry, "source": "wikipedia_snapshot"})

    # The first observed label is assumed to have held back to the panel floor: the
    # alternative is leaving early rows with no sector at all, which drops them from
    # the SECB metric entirely.
    if runs[0]["valid_from"] > floor:
        runs[0] = {**runs[0], "valid_from": floor, "source": "assumed_pre_first_snapshot"}

    # The live tickers row wins at the tail — it reflects reclassifications made after
    # our last snapshot (and for still-listed names it is simply more current).
    cur_sector, cur_industry = current
    if cur_sector and cur_sector != runs[-1]["sector"]:
        boundary = max(seen[-1][0] + timedelta(days=1), runs[-1]["valid_from"] + timedelta(days=1))
        runs.append({"valid_from": boundary, "valid_to": None, "sector": cur_sector,
                     "industry": cur_industry, "source": "tickers_current"})

    for a, b in zip(runs, runs[1:], strict=False):
        a["valid_to"] = b["valid_from"]
    return runs


async def amain(args: argparse.Namespace) -> int:
    settings = get_settings()
    headers = {"User-Agent": f"stock-thing-seed/1.0 ({settings.sec_edgar_user_agent})"}
    snapshots = quarterly_dates(args.start, date.today())
    print(f"fetching {len(snapshots)} quarterly snapshots {snapshots[0]} .. {snapshots[-1]}")

    per_date: dict[date, dict[str, dict]] = {}
    async with httpx.AsyncClient(headers=headers, timeout=30.0,
                                 follow_redirects=True) as client:
        for when in snapshots:
            snap = await snapshot_for(client, when)
            if snap:
                per_date[when] = snap
            print(f"  {when}: {len(snap)} constituents")
            await asyncio.sleep(0.3)   # be polite to the MediaWiki API

    if not per_date:
        print("no snapshots parsed — aborting rather than writing an empty table")
        return 1

    async with pool_context() as pool:
        tickers = await pool.fetch(
            "select ticker_id, symbol, sector, industry from tickers "
            " where asset_type = 'equity' order by ticker_id"
        )
        known = {r["sector"] for r in tickers if r["sector"]}
        by_symbol = {r["symbol"]: r for r in tickers}

        rows: list[tuple] = []
        stats = {"observed": 0, "tickers_only": 0, "no_sector": 0, "multi_interval": 0}
        for r in tickers:
            obs: list[tuple[date, str | None, str | None]] = []
            for when in sorted(per_date):
                hit = per_date[when].get(r["symbol"])
                if hit and hit.get("sector"):
                    sector = canonical_sector(hit["sector"])
                    # Never introduce a label the panel has never seen — a wrong peer
                    # group is worse than falling back (same rule as add_ticker.py).
                    if sector in known:
                        obs.append((when, sector, hit.get("industry")))
            intervals = build_intervals(
                obs, args.start, (r["sector"], r["industry"])
            )
            if not intervals:
                stats["no_sector"] += 1
                continue
            stats["observed" if obs else "tickers_only"] += 1
            if len(intervals) > 1:
                stats["multi_interval"] += 1
            for iv in intervals:
                rows.append((r["ticker_id"], iv["valid_from"], iv["valid_to"],
                             iv["sector"], iv["industry"], iv["source"]))

        print(f"\n{len(rows)} intervals for {stats['observed'] + stats['tickers_only']} tickers "
              f"({stats['observed']} observed, {stats['tickers_only']} from tickers only, "
              f"{stats['no_sector']} with no sector at all); "
              f"{stats['multi_interval']} tickers changed sector at least once")

        changed = [s for s, r in by_symbol.items()
                   if sum(1 for row in rows if row[0] == r["ticker_id"]) > 1]
        print("reclassified names (first 25):", sorted(changed)[:25])

        if args.dry_run:
            print("--dry-run: not writing")
            return 0

        async with pool.acquire() as conn, conn.transaction():
            await conn.execute("delete from sector_history")
            await conn.executemany(
                "insert into sector_history "
                "(ticker_id, valid_from, valid_to, sector, industry, source) "
                "values ($1,$2,$3,$4,$5,$6)",
                rows,
            )
        print(f"wrote {len(rows)} sector_history rows")

        # Validation: intervals must not overlap for a ticker.
        bad = await pool.fetch(
            """
            select a.ticker_id, count(*) n
              from sector_history a join sector_history b
                on a.ticker_id = b.ticker_id and a.valid_from < b.valid_from
             where coalesce(a.valid_to, date '9999-12-31') > b.valid_from
             group by 1
            """
        )
        print("overlapping-interval tickers:", len(bad), "(must be 0)")
        return 1 if bad else 0


def main() -> int:
    p = argparse.ArgumentParser(description="Build point-in-time GICS sector intervals.")
    p.add_argument("--start", type=date.fromisoformat, default=HISTORY_FLOOR,
                   help="panel floor; the first observed label is back-filled to it")
    p.add_argument("--dry-run", action="store_true")
    return asyncio.run(amain(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
