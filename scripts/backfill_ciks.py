"""Backfill SEC CIK identifiers for equity tickers (active AND removed-from-index).

SEC publishes a canonical ticker -> CIK mapping at
https://www.sec.gov/files/company_tickers.json. We fetch it once and update
the cik column for any equity ticker that doesn't yet have one.

That file only lists CURRENT filers with a live ticker, so the removed-from-index
cohort (active=false, seeded by scripts/seed_sp500_historical.py with no CIK)
mostly misses. Without a CIK those names never get EDGAR fundamentals, which made
`fund_available` a proxy for future index removal — a survivorship leak. Hence two
fallbacks:

  --dera-fallback  For each still-unmapped symbol, read the SEC DERA form345
                   SUBMISSION.tsv for the quarter CONTEMPORANEOUS with the ticker's
                   removed_at and map ISSUERTRADINGSYMBOL -> ISSUERCIK. Using the
                   removal-quarter file (rather than a recent one) avoids matching a
                   symbol that was later reassigned to a different issuer.

  overrides CSV    scripts/data/removed_ticker_overrides.csv — hand-curated last
                   resort (find a CIK via EDGAR company search by company name).
                   Rows with a blank cik are the permanent record of names that are
                   genuinely unmappable; they are skipped, not retried.

Usage:
    python -m scripts.backfill_ciks [--dera-fallback] [--dry-run]

Requires SUPABASE_URL, SUPABASE_SECRET_KEY (or DATABASE_URL) in .env.

ETFs are skipped — CIK is only meaningful for issuers that file 10-K/10-Q
with the SEC, which is what our fundamentals pipeline targets.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import sys
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

import asyncpg
import httpx

from backend.config import get_settings
from backend.ingestion.insiders import DERA_URL, _iter_tsv, _to_int

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
OVERRIDES_CSV = Path(__file__).parent / "data" / "removed_ticker_overrides.csv"


async def fetch_sec_mapping(user_agent: str) -> dict[str, str]:
    """Return {symbol -> zero-padded 10-digit CIK string}."""
    headers = {"User-Agent": user_agent, "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        resp = await client.get(SEC_TICKERS_URL)
        resp.raise_for_status()
        data: dict[str, dict[str, Any]] = resp.json()

    mapping: dict[str, str] = {}
    for entry in data.values():
        symbol = str(entry["ticker"]).upper()
        cik = str(entry["cik_str"]).zfill(10)
        mapping[symbol] = cik
    return mapping


def load_overrides(path: Path = OVERRIDES_CSV) -> dict[str, str]:
    """Return {symbol -> cik} from the manual overrides CSV (blank ciks skipped).

    A row with a blank cik documents a name we investigated and could not map;
    it stays in the file as the record, and is not treated as a mapping.
    """
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            symbol = (row.get("symbol") or "").strip().upper()
            cik = (row.get("cik") or "").strip()
            if symbol and cik:
                out[symbol] = cik.zfill(10)
    return out


def _quarter_of(d: date) -> tuple[int, int]:
    return d.year, (d.month - 1) // 3 + 1


def _prev_quarter(year: int, q: int) -> tuple[int, int]:
    return (year - 1, 4) if q == 1 else (year, q - 1)


def parse_dera_symbol_map(zip_bytes: bytes) -> dict[str, str]:
    """{trading symbol -> zero-padded CIK} from one DERA quarter's SUBMISSION.tsv."""
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    out: dict[str, str] = {}
    for r in _iter_tsv(z, "SUBMISSION.tsv"):
        symbol = (r.get("ISSUERTRADINGSYMBOL") or "").strip().upper()
        cik = _to_int(r.get("ISSUERCIK"))
        if symbol and cik is not None:
            # Symbols are stored with '.' normalized to '-' in our tickers table.
            out[symbol.replace(".", "-")] = str(cik).zfill(10)
    return out


async def dera_lookup(
    client: httpx.AsyncClient, pending: list[tuple[str, date | None]]
) -> dict[str, str]:
    """Resolve {symbol -> cik} via the DERA quarter contemporaneous with removal.

    Symbols are grouped by removal quarter so each quarterly ZIP is downloaded
    once. If a quarter is missing (404) we retry the preceding one, which covers
    removals dated just after a quarter boundary.
    """
    by_quarter: dict[tuple[int, int], list[str]] = {}
    for symbol, removed_at in pending:
        if removed_at is None:
            continue
        by_quarter.setdefault(_quarter_of(removed_at), []).append(symbol)

    resolved: dict[str, str] = {}
    for (year, q), symbols in sorted(by_quarter.items()):
        for attempt_year, attempt_q in ((year, q), _prev_quarter(year, q)):
            url = DERA_URL.format(year=attempt_year, q=attempt_q)
            try:
                resp = await client.get(url)
            except httpx.HTTPError as exc:
                print(f"  {attempt_year}q{attempt_q}: fetch failed ({exc})")
                continue
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            symbol_map = parse_dera_symbol_map(resp.content)
            hits = [s for s in symbols if s in symbol_map]
            for s in hits:
                resolved[s] = symbol_map[s]
            print(f"  {attempt_year}q{attempt_q}: {len(hits)}/{len(symbols)} resolved")
            break
    return resolved


async def amain(args: argparse.Namespace) -> int:
    settings = get_settings()
    if not settings.database_url:
        print("DATABASE_URL is not set in .env", file=sys.stderr)
        return 1

    print(f"Fetching SEC ticker mapping with UA: {settings.sec_edgar_user_agent!r}")
    mapping = await fetch_sec_mapping(settings.sec_edgar_user_agent)
    print(f"  -> {len(mapping):,} symbols available from SEC")

    overrides = load_overrides()
    if overrides:
        print(f"  -> {len(overrides)} manual override(s) from {OVERRIDES_CSV.name}")

    conn = await asyncpg.connect(
        settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    )
    try:
        # No `active` filter: the removed-from-index cohort is exactly the set that
        # needs CIKs. `cik is null` keeps this idempotent.
        rows = await conn.fetch(
            """
            select ticker_id, symbol, active, removed_at
              from tickers
             where asset_type = 'equity'
               and cik is null
             order by ticker_id
            """
        )
        print(f"{len(rows)} equity ticker(s) without a CIK")

        found: dict[str, str] = {}
        pending: list[tuple[str, date | None]] = []
        for row in rows:
            symbol = row["symbol"].upper()
            cik = overrides.get(symbol) or mapping.get(symbol)
            if cik is not None:
                found[symbol] = cik
            else:
                removed_at = row["removed_at"]
                pending.append((symbol, removed_at.date() if removed_at else None))

        print(f"  resolved by SEC mapping/overrides: {len(found)}")
        print(f"  still unresolved: {len(pending)}")

        if pending and args.dera_fallback:
            print("DERA fallback (form345 SUBMISSION.tsv, removal quarter):")
            headers = {"User-Agent": settings.sec_edgar_user_agent,
                       "Accept-Encoding": "gzip, deflate"}
            async with httpx.AsyncClient(timeout=180.0, headers=headers,
                                         follow_redirects=True) as client:
                found.update(await dera_lookup(client, pending))

        missing = [s for s, _ in pending if s not in found]

        if args.dry_run:
            print(f"[dry-run] would set CIK for {len(found)} ticker(s)")
        else:
            updated = 0
            async with conn.transaction():
                for row in rows:
                    cik = found.get(row["symbol"].upper())
                    if cik is None:
                        continue
                    await conn.execute(
                        "update tickers set cik = $1 where ticker_id = $2",
                        cik, row["ticker_id"],
                    )
                    updated += 1
            print(f"Updated CIK for {updated} ticker(s).")
    finally:
        await conn.close()

    if missing:
        print(f"\nNo CIK for {len(missing)} symbol(s): {', '.join(sorted(missing))}")
        print(f"  Add them to {OVERRIDES_CSV} (blank cik = confirmed unmappable).")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dera-fallback", action="store_true",
                   help="resolve leftovers via the DERA form345 quarter matching removed_at")
    p.add_argument("--dry-run", action="store_true", help="report without writing")
    return asyncio.run(amain(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
