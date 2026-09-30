"""Replace scoped CA/HAR histories and apply cash outcomes after exact rollback.

The complete prior rows remain recoverable in a hashed compressed preimage.
Membership, sectors and issuer accounting records are preserved. No global gate
or production cache is promoted by this scoped repair.
"""

import argparse
import asyncio
import gzip
import json
from collections import Counter
from datetime import date
from pathlib import Path

from backend.ingestion.calendar import HORIZON_TRADING_DAYS
from backend.ingestion.db import pool_context
from backend.ingestion.prices_lseg import ingest_verified_prices
from backend.ml.dataset import load_frames
from backend.ml.factors.pit import documented_targets
from backend.ml.research import write_json_new
from scripts.apply_praxair_reference import verify_stored_prices
from scripts.apply_registered_membership import canonical, digest
from scripts.audit_cash_recovery_components import prepare

IDS = [712, 749]
TABLES = {
    "tickers": "ticker_id",
    "price_history": "ticker_id,trade_date",
    "index_membership": "ticker_id,index_id,valid_from",
    "security_events": "ticker_id,effective_date",
    "sector_history": "ticker_id,valid_from",
    "fundamentals": "ticker_id,accession_number",
    "accounting_facts": "ticker_id,accession_number,metric,span_start,period_end,source_concept",
}


async def capture(conn):
    return {
        table: [
            dict(r)
            for r in await conn.fetch(
                f"select * from {table} where ticker_id=any($1::bigint[]) order by {order}", IDS
            )
        ]
        for table, order in TABLES.items()
    }


async def verify(conn, before, prices, outcomes, provenance):
    after = await capture(conn)
    for table in ("index_membership", "sector_history", "fundamentals", "accounting_facts"):
        if canonical(before[table]) != canonical(after[table]):
            raise ValueError(f"Unexpected {table} mutation")
    expected_tickers = []
    by_id = {o["event"]["ticker_id"]: o for o in outcomes}
    for ticker in before["tickers"]:
        event = by_id[ticker["ticker_id"]]["event"]
        expected_tickers.append(
            dict(
                ticker,
                security_retired_at=date.fromisoformat(event["effective_date"]),
                security_retired_source=provenance,
            )
        )
    touched_at = await conn.fetchval("select transaction_timestamp()")
    for ticker in expected_tickers:
        ticker["updated_at"] = touched_at
    if canonical(expected_tickers) != canonical(after["tickers"]):
        raise ValueError("Unexpected ticker mutation")
    for outcome in outcomes:
        tid, ric = outcome["event"]["ticker_id"], outcome["ric"]
        actual = [r for r in after["price_history"] if r["ticker_id"] == tid]
        verify_stored_prices(prices[ric], actual)
        events = [r for r in after["security_events"] if r["ticker_id"] == tid]
        if len(events) != 1:
            raise ValueError("Wrong terminal event count")
        for key, value in outcome["event"].items():
            expected = date.fromisoformat(value) if key == "effective_date" else value
            if events[0][key] != expected and canonical(events[0][key]) != canonical(expected):
                raise ValueError("Cash event readback differs")
    frames = await load_frames(conn, symbols=["CA", "HAR"])
    if {f.ticker_id for f in frames} != set(IDS):
        raise ValueError("Recovered loader frames missing")
    census = {}
    for frame in frames:
        ric = by_id[frame.ticker_id]["ric"]
        required = {
            "trade_date",
            "close",
            "adj_close",
            "dividend",
            "split_factor",
            "source",
            "price_basis",
            "share_split_factor",
            "share_action_source",
        }
        if not frame.prices or not required <= frame.prices[0].keys():
            raise ValueError("Loader omitted required price/basis fields")
        loaded_keys = frame.prices[0].keys()
        verify_stored_prices(
            [{k: v for k, v in p.items() if k in loaded_keys} for p in prices[ric]], frame.prices
        )
        counts = Counter()
        dates = [p["trade_date"] for p in frame.prices]
        for pos in range(len(frame.prices) - 254, len(frame.prices) - 1):
            labels = documented_targets(
                frame.prices, pos, frame.security_events, date(2020, 1, 1), dates
            )
            for horizon, label in labels.items():
                if label[4] == "acquisition":
                    if not label[1]:
                        raise ValueError("Live cash outcome masked")
                    counts[horizon] += 1
        if dict(counts) != HORIZON_TRADING_DAYS:
            raise ValueError("Live terminal-crossing census differs")
        census[ric] = dict(counts)
    return after, census


async def run(args):
    if Path(args.output).exists():
        raise ValueError("Receipt exists")
    prices, outcomes, components, hashes = prepare()
    hashes[str(Path(__file__))] = digest(Path(__file__).read_bytes())
    for path in [
        "backend/ingestion/prices_lseg.py",
        "backend/ml/dataset.py",
        "backend/ml/factors/pit.py",
        "scripts/apply_praxair_reference.py",
    ]:
        hashes[path] = digest(Path(path).read_bytes())
    proposal = digest(canonical(dict(outcomes=outcomes, components=components, hashes=hashes)))
    provenance = "original_cash_recovery:" + proposal
    rehearsal = json.loads(Path(args.rehearsal).read_bytes()) if args.rehearsal else None
    if args.apply and (
        not rehearsal
        or rehearsal["applied"]
        or not rehearsal["rollback_verified"]
        or rehearsal["proposal_sha256"] != proposal
    ):
        raise ValueError("Matching exact rollback rehearsal required")
    result = dict(
        applied=args.apply,
        proposal_sha256=proposal,
        source_hashes=hashes,
        components=components,
        prices_written=sum(len(prices[o["ric"]]) for o in outcomes),
        terminal_events=2,
        database_scope=IDS,
        global_gates_promoted=0,
        production_cache_refreshed=False,
    )
    async with pool_context(max_size=1) as pool, pool.acquire() as conn:
        tx = conn.transaction(isolation="repeatable_read")
        await tx.start()
        try:
            await conn.execute("set local lock_timeout='10s'")
            await conn.execute(
                "lock table price_history,security_events in share row exclusive mode"
            )
            await conn.fetch(
                "select ticker_id from tickers where ticker_id=any($1::bigint[]) for update", IDS
            )
            before = await capture(conn)
            unrelated_before = {
                table: await conn.fetchval(
                    f"select count(*) from {table} where not(ticker_id=any($1::bigint[]))", IDS
                )
                for table in TABLES
            }
            if before["security_events"]:
                raise ValueError("Existing scoped terminal events require explicit reconciliation")
            expected = {712: ("CA", 356028), 749: ("HAR", 800459)}
            if len(before["tickers"]) != 2 or any(
                (t["symbol"], int(t["cik"])) != expected[t["ticker_id"]] for t in before["tickers"]
            ):
                raise ValueError("Database issuer/security identity changed")
            preimage = canonical(before)
            preimage_hash = digest(preimage)
            if args.apply and preimage_hash != rehearsal["preimage_sha256"]:
                raise ValueError("Database changed after rehearsal")
            path = Path(args.preimage)
            if path.exists():
                if gzip.decompress(path.read_bytes()) != preimage:
                    raise ValueError("Preimage differs")
            else:
                with path.open("xb") as stream:
                    stream.write(gzip.compress(preimage, mtime=0))
            if gzip.decompress(path.read_bytes()) != preimage:
                raise ValueError("Preimage archival verification failed")
            # Exact two-security replacement; raw legacy rows survive in preimage.
            await conn.execute("delete from price_history where ticker_id=any($1::bigint[])", IDS)
            for outcome in outcomes:
                rows, event = prices[outcome["ric"]], outcome["event"]
                await ingest_verified_prices(conn, rows)
                when = date.fromisoformat(event["effective_date"])
                await conn.execute(
                    "update tickers set security_retired_at=$2, "
                    "security_retired_source=$3 where ticker_id=$1",
                    event["ticker_id"],
                    when,
                    provenance,
                )
                await conn.execute(
                    """insert into security_events
                    (ticker_id,effective_date,event_type,consideration_type,cash_value,
                     proceeds_basis,source,verified)
                    values($1,$2,$3,$4,$5,$6,$7,$8)""",
                    event["ticker_id"],
                    when,
                    event["event_type"],
                    event["consideration_type"],
                    event["cash_value"],
                    event["proceeds_basis"],
                    event["source"],
                    event["verified"],
                )
            after, census = await verify(conn, before, prices, outcomes, provenance)
            unrelated_after = {
                table: await conn.fetchval(
                    f"select count(*) from {table} where not(ticker_id=any($1::bigint[]))", IDS
                )
                for table in TABLES
            }
            if unrelated_after != unrelated_before:
                raise ValueError("Unrelated row counts changed")
            result.update(
                preimage=str(path),
                preimage_sha256=preimage_hash,
                legacy_prices_archived=len(before["price_history"]),
                preserved_unrelated_row_counts=unrelated_before,
                live_terminal_label_census=census,
                transaction_readback_verified=True,
            )
            if args.apply:
                await tx.commit()
            else:
                await tx.rollback()
        except BaseException:
            await tx.rollback()
            raise
        post = await capture(conn)
        if canonical(post) != canonical(after if args.apply else before):
            raise ValueError("Posttransaction exact readback differs")
        result["postcommit_verified" if args.apply else "rollback_verified"] = True
    write_json_new(args.output, result)
    print({k: v for k, v in result.items() if k not in ("source_hashes", "components")})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--preimage", required=True)
    p.add_argument("--rehearsal")
    p.add_argument("--apply", action="store_true")
    asyncio.run(run(p.parse_args()))
