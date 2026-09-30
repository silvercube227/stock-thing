"""Apply exact-RIC membership for registered research securities after rollback replay.

Prices, issuer CIKs, sectors and terminal payoffs are outside this repair. Securities
with sessions outside their archived lifecycle remain blocked, without guessed
endpoints. The complete reference and experimental source gates remain incomplete.
"""

import argparse
import asyncio
import gzip
import hashlib
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from backend.ingestion.calendar import trading_days_between
from backend.ingestion.db import pool_context
from backend.ingestion.index_lseg import build_jl_intervals
from backend.ml.research import json_value, write_json_new


def canonical(value):
    return json.dumps(json_value(value), sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def proposed_memberships(registrations, identities, intervals, observed_on):
    """Require exact security identity; issuer-context rows cannot change it."""
    exact, membership = defaultdict(list), defaultdict(list)
    for row in identities:
        if row.get("RIC") == row["Instrument"]:
            exact[row["RIC"]].append(row)
    for row in intervals:
        membership[row["security_id"]].append(row)
    if len({r["ric"] for r in registrations}) != len(registrations) or len(
        {r["ticker_id"] for r in registrations}
    ) != len(registrations):
        raise ValueError("Duplicate registered security")
    sessions = trading_days_between(date(2010, 1, 1), observed_on)
    proposed, blocked = [], []
    for registration in registrations:
        ric = registration["ric"]
        rows = exact[ric]
        if len(rows) != 1 or rows[0].get("ISIN") != registration["source_isin"]:
            raise ValueError("Registered ISIN no longer matches exact source security")
        source = rows[0]
        # Missing issuer mapping stays missing. A security identity does not
        # authorize substituting an issuer or copying its filings.
        if (
            registration.get("cik")
            and source.get("CIK Number")
            and int(registration["cik"]) != int(source["CIK Number"])
        ):
            raise ValueError("Registered issuer conflicts with exact source security")
        if not membership[ric]:
            raise ValueError("Registered security has no exact-RIC membership")
        dated = sorted(membership[ric], key=lambda r: r["valid_from"])
        for i, row in enumerate(dated):
            start, end = row["valid_from"], row["valid_to"]
            if (end is not None and end <= start) or (
                i and (dated[i - 1]["valid_to"] is None or dated[i - 1]["valid_to"] > start)
            ):
                raise ValueError("Invalid or overlapping membership")
        first, retired = source.get("First Trade Date"), source.get("RetireDate")
        first = date.fromisoformat(first) if first not in (None, "", "NaT") else None
        retired = date.fromisoformat(retired) if retired not in (None, "", "NaT") else None
        expected = [
            d
            for d in sessions
            if any(
                r["valid_from"] <= str(d) and (r["valid_to"] is None or str(d) < r["valid_to"])
                for r in dated
            )
        ]
        invalid = [d for d in expected if (first and d < first) or (retired and d > retired)]
        if invalid:
            blocked.append(
                dict(
                    ric=ric,
                    ticker_id=registration["ticker_id"],
                    status="membership_lifecycle_requires_review",
                    dates=invalid,
                    first_trade_date=first,
                    retire_date=retired,
                    intervals=dated,
                )
            )
            continue
        proposed.append(
            dict(
                registration,
                membership=[
                    dict(
                        index_id="SPX",
                        valid_from=date.fromisoformat(r["valid_from"]),
                        valid_to=date.fromisoformat(r["valid_to"]) if r["valid_to"] else None,
                    )
                    for r in dated
                ],
                membership_sessions=len(expected),
            )
        )
    return proposed, blocked


def prepare():
    paths = {
        key: Path(path)
        for key, path in dict(
            registration=".research/verification/historical-registration-applied.json",
            identity=".research/verification/identity-lifecycle-archive.json",
            archive=".research/verification/membership-archive.json",
            reconstructed=".research/verification/membership-reconciled.json",
            reviews="docs/membership_event_reviews.json",
        ).items()
    }
    raw = {k: p.read_bytes() for k, p in paths.items()}
    data = {k: json.loads(v) for k, v in raw.items()}
    report, review = data["reconstructed"], data["reviews"]
    if report["archive_sha256"] != digest(raw["archive"]) or report["reviews_sha256"] != digest(
        raw["reviews"]
    ):
        raise ValueError("Membership source or review changed")
    source = data["archive"]["sources"]["membership_archive"]
    intervals = build_jl_intervals(
        [r["RIC"] for r in source["current"]],
        [
            dict(date=r["Date"], security_id=r["Constituent RIC"], change=r["Change"])
            for r in source["events"]
        ],
        review["window_start"],
        review["observed_on"],
        reviewed_zero_duration=review["zero_duration"],
    )
    if canonical(intervals) != canonical(report["intervals"]):
        raise ValueError("Archived membership does not independently replay")
    if not data["registration"]["applied"]:
        raise ValueError("Historical securities are not registered")
    proposed, blocked = proposed_memberships(
        data["registration"]["securities"],
        data["identity"]["sources"]["identity_archive"]["records"],
        json_value(intervals),
        date.fromisoformat(review["observed_on"]),
    )
    hashes = {str(paths[k]): digest(v) for k, v in raw.items()}
    hashes.update(
        {
            p: digest(Path(p).read_bytes())
            for p in (
                "scripts/apply_registered_membership.py",
                "backend/ingestion/index_lseg.py",
                "backend/ingestion/calendar.py",
            )
        }
    )
    return proposed, blocked, hashes


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
    )


def member_values(rows):
    keys = ("ticker_id", "index_id", "valid_from", "valid_to", "source")
    return sorted(
        ({k: r[k] for k in keys} for r in rows),
        key=lambda r: (r["ticker_id"], r["index_id"], r["valid_from"]),
    )


async def run(args):
    if Path(args.output).exists():
        raise ValueError("Receipt already exists")
    proposed, blocked, hashes = prepare()
    proposal = dict(securities=proposed, blocked=blocked, source_hashes=hashes)
    proposal_hash = digest(canonical(proposal))
    provenance = "lseg_exact_ric_membership:" + proposal_hash
    ids = [r["ticker_id"] for r in proposed]
    expected = [
        dict(row, ticker_id=r["ticker_id"], source=provenance)
        for r in proposed
        for row in r["membership"]
    ]
    receipt = dict(
        applied=args.apply,
        proposal_sha256=proposal_hash,
        source_hashes=hashes,
        securities=len(proposed),
        intervals=len(expected),
        blocked=blocked,
        prices_written=0,
        sectors_written=0,
        issuer_ids_changed=0,
        gates_promoted=0,
    )
    rehearsal = json.loads(Path(args.rehearsal).read_text()) if args.rehearsal else None
    if args.apply and (
        not rehearsal
        or rehearsal.get("applied")
        or not rehearsal.get("rollback_verified")
        or rehearsal.get("proposal_sha256") != proposal_hash
    ):
        raise ValueError("Matching successful rollback rehearsal required")
    async with pool_context(max_size=1) as pool, pool.acquire() as conn:
        tx = conn.transaction(isolation="repeatable_read")
        await tx.start()
        try:
            await conn.execute("set local lock_timeout='10s'")
            await conn.execute(
                "select pg_advisory_xact_lock(hashtext('apply_registered_membership'))"
            )
            await conn.execute("lock table index_membership in share row exclusive mode")
            await conn.fetch(
                "select ticker_id from tickers where ticker_id=any($1::bigint[]) for update",
                ids,
            )
            before = await capture(conn, ids)
            live = {r["ticker_id"]: r for r in before["tickers"]}
            for r in proposed:
                t = live[r["ticker_id"]]
                if (
                    any(t[k] != r[k] for k in ("symbol", "ric", "cik"))
                    or not t["research_only"]
                    or t["active"]
                ):
                    raise ValueError("Registered research identity changed")
            if any(r["ticker_id"] in ids for r in before["membership"]):
                raise ValueError("Research membership already populated; review required")
            preimage_hash = digest(canonical(before))
            if args.apply and rehearsal["preimage_sha256"] != preimage_hash:
                raise ValueError("Database changed since rehearsal")
            archive = Path(args.preimage)
            if archive.exists():
                if digest(gzip.decompress(archive.read_bytes())) != preimage_hash:
                    raise ValueError("Preimage changed")
            else:
                with archive.open("xb") as out:
                    out.write(gzip.compress(canonical(before), mtime=0))
            await conn.executemany(
                """insert into index_membership
                    (ticker_id,index_id,valid_from,valid_to,source) values($1,$2,$3,$4,$5)""",
                [
                    (r["ticker_id"], r["index_id"], r["valid_from"], r["valid_to"], r["source"])
                    for r in expected
                ],
            )
            after = await capture(conn, ids)
            if canonical(before["tickers"]) != canonical(after["tickers"]):
                raise ValueError("Ticker attributes changed")
            if member_values(after["membership"]) != member_values(
                before["membership"] + expected
            ):
                raise ValueError("Exact membership readback failed")
            unrelated = [r for r in after["membership"] if r["ticker_id"] not in ids]
            if canonical(unrelated) != canonical(before["membership"]):
                raise ValueError("Unrelated membership changed")
            receipt.update(
                preimage=str(archive),
                preimage_sha256=preimage_hash,
                unrelated_membership_rows=len(unrelated),
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
        if not args.apply:
            if canonical(post) != canonical(before):
                raise ValueError("Rollback did not restore exact preimage")
            receipt["rollback_verified"] = True
        else:
            if canonical(post) != canonical(after):
                raise ValueError("Postcommit readback differs")
            receipt["postcommit_verified"] = True
    write_json_new(args.output, receipt)
    print({k: v for k, v in receipt.items() if k not in ("source_hashes", "blocked")})
    print("Blocked securities:", [r["ric"] for r in blocked])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--preimage", required=True)
    p.add_argument("--rehearsal")
    p.add_argument("--apply", action="store_true")
    asyncio.run(run(p.parse_args()))
