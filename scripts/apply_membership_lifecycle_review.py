"""Apply publisher-confirmed membership retention with separate terminal events."""

import argparse
import asyncio
import gzip
import json
import pickle
from collections import Counter
from datetime import date
from pathlib import Path

from backend.ingestion.calendar import HORIZON_TRADING_DAYS
from backend.ingestion.db import pool_context
from backend.ml.factors.pit import documented_targets
from backend.ml.research import write_json_new
from scripts.apply_registered_membership import canonical, digest, member_values
from scripts.apply_registered_membership import prepare as registered

REVIEW = Path("docs/membership_lifecycle_reviews.json")
ARCHIVE = Path(".research/verification/membership-lifecycle-sources-2026-09-18-v1.json")
CANDIDATE = Path(".research/normalized-historical-candidate-v3")


def prepare():
    review_raw, archive_raw = REVIEW.read_bytes(), ARCHIVE.read_bytes()
    review, archive = json.loads(review_raw), json.loads(archive_raw)
    if archive["review_sha256"] != digest(review_raw):
        raise ValueError("Source archive does not bind the reviewed decisions")
    _, blocked, hashes = registered()
    blocked = {r["ric"]: r for r in blocked}
    hashes.update({str(REVIEW): digest(review_raw), str(ARCHIVE): digest(archive_raw)})
    sources = {}
    for source in archive["events"]:
        content = Path(source["archived_source"]).read_bytes()
        if digest(content) != source["source_sha256"]:
            raise ValueError("Primary source changed")
        hashes[source["archived_source"]] = source["source_sha256"]
        sources[source["symbol"]] = source
    registration = json.loads(
        Path(".research/verification/historical-registration-applied.json").read_text()
    )
    identities = {r["ric"]: r for r in registration["securities"]}
    candidate_report = (CANDIDATE / "report.json").read_bytes()
    raw_prices = gzip.decompress((CANDIDATE / "prices.pkl.gz").read_bytes())
    if digest(raw_prices) != json.loads(candidate_report)["input_sha256"]:
        raise ValueError("Native price candidate changed")
    prices = pickle.loads(raw_prices)
    hashes[str(CANDIDATE / "report.json")] = digest(candidate_report)
    hashes[str(CANDIDATE / "prices.pkl.gz")] = digest((CANDIDATE / "prices.pkl.gz").read_bytes())
    results = []
    for row in review["reviews"]:
        b, identity = blocked[row["ric"]], identities[row["ric"]]
        if (
            identity["ticker_id"] != row["ticker_id"]
            or identity["source_isin"] != row["isin"]
            or int(identity["cik"]) != int(row["cik"])
            or b["retire_date"].isoformat() != row["vendor_retire_date"]
            or [str(d) for d in b["dates"]] != row["reviewed_lifecycle_exception_dates"]
        ):
            raise ValueError("Reviewed security or lifecycle discrepancies changed")
        if len(b["intervals"]) != 1 or any(
            b["intervals"][0][k] != row[k] for k in ("valid_from", "valid_to")
        ):
            raise ValueError("Publisher-reviewed membership interval changed")
        evidence = [sources[key] for key in row["evidence_keys"]]
        if len(evidence) != 2:
            raise ValueError("Both publisher and completion-filing evidence required")
        final = prices[row["ric"]][-1]
        if str(final["trade_date"]) != row["last_trade_date"]:
            raise ValueError("Native final session does not match reviewed trading cessation")
        if final["price_basis"] != "as_traded" or final["close"] != final["adj_close"]:
            raise ValueError("Final nominal-to-adjusted cash basis is not established")
        source = dict(
            row,
            evidence=evidence,
            review_sha256=digest(review_raw),
            normalized_price_sha256=digest(raw_prices),
            policy="Membership retention does not create tradable prices after last_trade_date.",
        )
        event = dict(
            ticker_id=row["ticker_id"],
            effective_date=date.fromisoformat(row["effective_date"]),
            event_type="acquisition",
            verified=True,
            proceeds_basis="adj_close",
            consideration_type="cash"
            if row["successor_shares_per_share"] == 0
            else "pending_successor_valuation",
            cash_value=row["cash_per_share"] if row["successor_shares_per_share"] == 0 else None,
            source=json.dumps(source, sort_keys=True),
        )
        counts = Counter()
        p = prices[row["ric"]]
        dates = [r["trade_date"] for r in p]
        # Validate the actual candidate's final entry and all terminal-crossing
        # horizons without applying its still-uncertified historical prices.
        for pos in range(max(0, len(p) - 254), len(p) - 1):
            values = documented_targets(p, pos, [event], date(2020, 1, 1), dates)
            for horizon, label in values.items():
                if label[4] == "acquisition":
                    if bool(label[1]) != (event["consideration_type"] == "cash"):
                        raise ValueError("Unknown successor payoff leaked into a valid label")
                    counts[(horizon, bool(label[1]))] += 1
        for horizon, steps in HORIZON_TRADING_DAYS.items():
            if sum(n for (h, _), n in counts.items() if h == horizon) != steps:
                raise ValueError("Incomplete terminal-crossing label census")
        results.append(
            dict(
                identity=identity,
                membership=dict(
                    ticker_id=row["ticker_id"],
                    index_id="SPX",
                    valid_from=date.fromisoformat(row["valid_from"]),
                    valid_to=date.fromisoformat(row["valid_to"]),
                ),
                event=event,
                label_census=[
                    dict(horizon=h, observed=observed, rows=n)
                    for (h, observed), n in sorted(counts.items())
                ],
            )
        )
    return results, hashes


async def capture(conn, ids):
    return dict(
        tickers=[
            dict(r)
            for r in await conn.fetch(
                "select * from tickers where ticker_id=any($1::bigint[]) order by ticker_id", ids
            )
        ],
        membership=[
            dict(r)
            for r in await conn.fetch(
                "select * from index_membership order by ticker_id,index_id,valid_from"
            )
        ],
        events=[
            dict(r)
            for r in await conn.fetch(
                "select * from security_events order by ticker_id,effective_date"
            )
        ],
        prices=await conn.fetchval(
            "select count(*) from price_history where ticker_id=any($1::bigint[])", ids
        ),
    )


async def run(args):
    if Path(args.output).exists():
        raise ValueError("Receipt already exists")
    changes, hashes = prepare()
    hashes["scripts/apply_membership_lifecycle_review.py"] = digest(Path(__file__).read_bytes())
    proposal_hash = digest(canonical(dict(changes=changes, source_hashes=hashes)))
    ids = [r["identity"]["ticker_id"] for r in changes]
    expected = [
        dict(r["membership"], source="lseg_reviewed_retention:" + proposal_hash) for r in changes
    ]
    rehearsal = json.loads(Path(args.rehearsal).read_bytes()) if args.rehearsal else None
    if args.apply and (
        not rehearsal
        or rehearsal["applied"]
        or not rehearsal["rollback_verified"]
        or rehearsal["proposal_sha256"] != proposal_hash
    ):
        raise ValueError("Matching successful rehearsal required")
    result = dict(
        applied=args.apply,
        proposal_sha256=proposal_hash,
        source_hashes=hashes,
        securities=ids,
        membership_intervals=3,
        terminal_events=3,
        cash_outcomes=1,
        successor_outcomes_masked=2,
        prices_written=0,
        gates_promoted=0,
        label_census={r["identity"]["ric"]: r["label_census"] for r in changes},
    )
    async with pool_context(max_size=1) as pool, pool.acquire() as conn:
        tx = conn.transaction(isolation="repeatable_read")
        await tx.start()
        try:
            await conn.execute("set local lock_timeout='10s'")
            await conn.execute(
                "lock table index_membership,security_events in share row exclusive mode"
            )
            await conn.fetch(
                "select ticker_id from tickers where ticker_id=any($1::bigint[]) for update", ids
            )
            before = await capture(conn, ids)
            live = {r["ticker_id"]: r for r in before["tickers"]}
            for change in changes:
                identity = change["identity"]
                t = live[identity["ticker_id"]]
                if (
                    any(t[k] != identity[k] for k in ("symbol", "ric", "cik"))
                    or t["active"]
                    or not t["research_only"]
                ):
                    raise ValueError("Research registration changed")
            if before["prices"] or any(
                r["ticker_id"] in ids for r in before["membership"] + before["events"]
            ):
                raise ValueError("Scoped securities already have reference data")
            preimage_hash = digest(canonical(before))
            if args.apply and rehearsal["preimage_sha256"] != preimage_hash:
                raise ValueError("Database changed since rehearsal")
            preimage = Path(args.preimage)
            if preimage.exists():
                if digest(gzip.decompress(preimage.read_bytes())) != preimage_hash:
                    raise ValueError("Recovery preimage changed")
            else:
                with preimage.open("xb") as out:
                    out.write(gzip.compress(canonical(before), mtime=0))
            await conn.executemany(
                """insert into index_membership
                (ticker_id,index_id,valid_from,valid_to,source) values($1,$2,$3,$4,$5)""",
                [
                    tuple(
                        r[k] for k in ("ticker_id", "index_id", "valid_from", "valid_to", "source")
                    )
                    for r in expected
                ],
            )
            columns = (
                "ticker_id",
                "effective_date",
                "event_type",
                "consideration_type",
                "cash_value",
                "proceeds_basis",
                "source",
                "verified",
            )
            await conn.executemany(
                """insert into security_events
                (ticker_id,effective_date,event_type,consideration_type,cash_value,proceeds_basis,source,verified)
                values($1,$2,$3,$4,$5,$6,$7,$8)""",
                [tuple(r["event"][k] for k in columns) for r in changes],
            )
            after = await capture(conn, ids)
            if canonical(before["tickers"]) != canonical(after["tickers"]) or after["prices"]:
                raise ValueError("Unexpected ticker or price change")
            if member_values(after["membership"]) != member_values(before["membership"] + expected):
                raise ValueError("Membership readback mismatch")
            unrelated = [r for r in after["events"] if r["ticker_id"] not in ids]
            if canonical(unrelated) != canonical(before["events"]):
                raise ValueError("Unrelated terminal events changed")
            actual = {r["ticker_id"]: r for r in after["events"] if r["ticker_id"] in ids}
            for r in changes:
                if any(actual[r["identity"]["ticker_id"]][k] != r["event"][k] for k in columns):
                    raise ValueError("Terminal event readback mismatch")
            result.update(
                preimage=str(preimage),
                preimage_sha256=preimage_hash,
                existing_membership_preserved=len(before["membership"]),
                existing_events_preserved=len(before["events"]),
                transaction_readback_verified=True,
            )
            if args.apply:
                await tx.commit()
            else:
                await tx.rollback()
        except BaseException:
            await tx.rollback()
            raise
        post = await capture(conn, ids)
        if canonical(post) != canonical(after if args.apply else before):
            raise ValueError("Posttransaction verification failed")
        result["postcommit_verified" if args.apply else "rollback_verified"] = True
    write_json_new(args.output, result)
    print({k: v for k, v in result.items() if k not in ("source_hashes", "label_census")})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--preimage", required=True)
    p.add_argument("--rehearsal")
    p.add_argument("--apply", action="store_true")
    asyncio.run(run(p.parse_args()))
