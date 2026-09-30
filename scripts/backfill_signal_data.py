"""Resumable research-only backfills; explicit source selection, no promotion.

Accounting/ALFRED records go to the configured research database. News raw text
goes only to local SQLite. Migrations must already have been applied.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

from backend.config import get_settings
from backend.ingestion.db import pool_context
from backend.ingestion.fundamentals import _upsert_filings, fetch_companyfacts, parse_companyfacts
from backend.ingestion.research_fundamentals import parse_accounting_facts


async def save_rows(conn, table, rows, columns):
    if not rows:
        return
    sql = f"insert into {table} ({','.join(columns)}) values ({','.join('$' + str(i + 1) for i in range(len(columns)))}) on conflict do nothing"
    await conn.executemany(sql, [tuple(r[c] for c in columns) for r in rows])


async def backfill(args):
    if args.source == 'macro':
        raise ValueError('Macro requires archived source replay and a rollback rehearsal; '
                         'use scripts.rebuild_macro_vintages download/apply')
    settings = get_settings()
    checkpoint = Path(args.checkpoint)
    checkpoint.mkdir(parents=True, exist_ok=True)
    # Bind resume markers to this database without storing credentials.
    identity = hashlib.sha256(settings.database_url.encode()).hexdigest()
    binding = checkpoint / 'database.sha256'
    if binding.exists() and binding.read_text() != identity:
        raise ValueError('Checkpoint belongs to another database')
    if not binding.exists():
        binding.write_text(identity)
    async with pool_context(command_timeout=300) as pool:
        tickers = await pool.fetch(
            "select ticker_id,symbol,cik, to_jsonb(tickers)->>'ric' as ric from tickers order by ticker_id"
        )
        if args.symbols:
            tickers = [r for r in tickers if r["symbol"] in args.symbols]
        if args.source == "news":
            from backend.ingestion.estimates import _open_session
            from backend.ingestion.news_lseg import ingest_window, open_store

            settings.news_cache_dir.mkdir(parents=True, exist_ok=True)
            conn = open_store(settings.news_cache_dir / "news_lseg.sqlite")
            ld = await asyncio.to_thread(_open_session, settings.lseg_app_key)
            try:
                for ticker in tickers:
                    if not ticker["ric"]:
                        continue
                    for year in range(args.start_year, args.end_year + 1):
                        start = datetime(year, 1, 1, tzinfo=timezone.utc)
                        end = min(
                            datetime(year + 1, 1, 1, tzinfo=timezone.utc),
                            datetime.now(timezone.utc),
                        )
                        if start >= end:
                            continue
                        status = ingest_window(
                            ld, conn, ticker["ticker_id"], ticker["ric"], start, end
                        )
                        print(ticker["ticker_id"], year, status, flush=True)
                        await asyncio.sleep(0.5)
            finally:
                ld.close_session()
                conn.close()
            return
        async with httpx.AsyncClient(
            timeout=60, headers={"User-Agent": settings.sec_edgar_user_agent}
        ) as client:
            for ticker in tickers:
                tid = ticker["ticker_id"]
                marker = checkpoint / f"accounting-{tid}.json"
                if marker.exists() or not ticker["cik"]:
                    continue
                payload = await fetch_companyfacts(client, str(ticker["cik"]).zfill(10))
                if payload is None:
                    print(tid, "blocked: SEC facts unavailable", flush=True)
                    continue
                raw = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
                digest = hashlib.sha256(raw).hexdigest()
                raw_dir = checkpoint / 'raw'
                raw_dir.mkdir(exist_ok=True)
                raw_path = raw_dir / f'{digest}.json'
                if not raw_path.exists():
                    raw_path.write_bytes(raw)
                facts = parse_accounting_facts(payload, tid)
                filings = parse_companyfacts(payload, tid)
                async with pool.acquire() as conn, conn.transaction():
                    await _upsert_filings(conn, filings)
                    await save_rows(
                        conn,
                        "accounting_facts",
                        facts,
                        (
                            "ticker_id",
                            "accession_number",
                            "filed_at",
                            "period_start",
                            "period_end",
                            "metric",
                            "value",
                            "source_concept",
                        ),
                    )
                from backend.ml.research import write_json_new

                write_json_new(
                    marker, {"ticker_id": tid, "facts": len(facts), "filings": len(filings),
                             "source_sha256": digest, "fetched_at": datetime.now(timezone.utc).isoformat()}
                )
                print(tid, len(facts), "facts", flush=True)
                await asyncio.sleep(0.2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=("accounting", "macro", "news"), required=True)
    p.add_argument("--symbols", nargs="+")
    p.add_argument("--checkpoint", default=".research/backfill")
    p.add_argument("--start-year", type=int, default=2012)
    p.add_argument("--end-year", type=int, default=2026)
    args = p.parse_args()
    asyncio.run(backfill(args))


if __name__ == "__main__":
    main()
