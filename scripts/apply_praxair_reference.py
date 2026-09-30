"""Scoped PX/LIN repair with exact input replay, preimage and rollback rehearsal.

Does not apply staged sector intervals, alter raw LIN prices, or certify global
source gates. Applying requires the same deterministic receipt as a dry run.
"""

import argparse
import asyncio
import gzip
import hashlib
import json
import pickle
from dataclasses import asdict
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from backend.ingestion.calendar import HORIZON_TRADING_DAYS, trading_days_between
from backend.ingestion.db import pool_context
from backend.ingestion.dividend_review import review_dividends
from backend.ingestion.prices_lseg import ingest_verified_prices, normalize_vendor_pair
from backend.ingestion.security_intervals import split_predecessor_intervals
from backend.ml.dataset import load_frames
from backend.ml.factors.pit import day, documented_targets
from backend.ml.research import json_value, load_snapshot, write_json_new
from backend.ml.terminal import successor_horizon_values
from scripts.audit_praxair_price_basis import component_parity
from scripts.audit_vendor_adjustments import changes, load_rows

FINAL, BOUNDARY = date(2018, 10, 30), date(2018, 10, 31)
IDS = [360, 947]


def canonical(value):
    return json.dumps(json_value(value), sort_keys=True, separators=(",", ":")).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(parent, candidate, successor_basis):
    original, parent_manifest = load_snapshot(parent)
    staged, manifest = load_snapshot(candidate)
    if manifest["metadata"]["parent_input_sha256"] != parent_manifest["input_sha256"]:
        raise ValueError("Wrong reference parent")
    hashes = dict(manifest["metadata"]["source_hashes"])
    basis_path = Path(successor_basis) / "report.json"
    basis = json.loads(basis_path.read_bytes())
    hashes[str(basis_path)] = sha(basis_path.read_bytes())
    hashes.update(basis["source_hashes"])
    for path, digest in hashes.items():
        if sha(Path(path).read_bytes()) != digest:
            raise ValueError(f"Changed archived source: {path}")
    if basis["input_sha256"] != manifest["input_sha256"]:
        raise ValueError("Successor review is for another candidate")
    before = {f.ticker_id: f for f in original["frames"]}
    after = {f.ticker_id: f for f in staged["frames"]}
    target = after[947]

    def read(path):
        raw = Path(path).read_bytes()
        hashes[path] = sha(raw)
        return json.loads(raw)

    base = ".research/verification/"
    source = read(base + "praxair-source-archived.json")["events"][0]
    if (
        sha(Path(source["archived_source"]).read_bytes()) != source["source_sha256"]
        or source["last_trade_date"] != str(FINAL)
        or source["effective_date"] != str(BOUNDARY)
        or source["cik"] != "884905"
        or source["terms"] != dict(cash_per_share=0, successor_shares_per_share=1)
    ):
        raise ValueError("Merger review changed")
    prices_by_id = {}
    for tid, ric, stem, kinds, start, end in [
        (
            947,
            "PX.N^J18",
            "praxair-",
            ("raw-history", "adjusted-history"),
            "2010-01-01",
            "2018-10-30",
        ),
        (
            360,
            "LIN.OQ",
            "praxair-successor-",
            ("native-prices", "adjusted-prices"),
            "2018-10-31",
            "2019-11-01",
        ),
    ]:
        raw, adjusted = [
            load_rows(read(base + stem + kind + ".json")["securities"][0]) for kind in kinds
        ]
        if (
            raw["request"]["ric"] != ric
            or adjusted["request"]["ric"] != ric
            or raw["request"]["adjustments"] != "unadjusted"
            or adjusted["request"]["adjustments"] != ["CCH", "CRE", "RPO", "RTS"]
        ):
            raise ValueError("Wrong security or adjustment convention")
        dividends = read(base + stem + "dividends.json")
        if dividends["requested_rics"] != [ric] or dividends["parameters"]["DateType"] != "XD":
            raise ValueError("Wrong dividend security or date convention")
        cash = review_dividends(dividends["records"], [r["Date"][:10] for r in raw["records"]])
        if (
            cash["issues"]
            or cash["empty_rows"]
            or cash["outside_price_history"]
            or changes(raw["records"], adjusted["records"])["events"]
        ):
            raise ValueError("Incomplete or unreviewed distributions")
        native = normalize_vendor_pair(
            tid, raw["records"], adjusted["records"], cash["cash_by_ex_date"]
        )
        if [p["trade_date"] for p in native] != list(trading_days_between(day(start), day(end))):
            raise ValueError("Native price calendar changed")
        prices_by_id[tid] = native
    if prices_by_id[947] != target.prices:
        raise ValueError("Predecessor normalization no longer reproduces")
    raw_native = gzip.decompress((Path(successor_basis) / "native-prices.pkl.gz").read_bytes())
    if (
        sha(raw_native) != basis["native_prices_sha256"]
        or pickle.loads(raw_native) != prices_by_id[360]
    ):
        raise ValueError("Successor normalization no longer reproduces")
    parity = component_parity(prices_by_id[947], before[360].prices)
    cash_review = read(base + "praxair-dividend-source-archived.json")["events"][0]
    if sha(Path(cash_review["archived_source"]).read_bytes()) != cash_review["source_sha256"]:
        raise ValueError("Issuer cash evidence changed")
    expected_cash = [
        dict(
            date=day(d),
            native=float(cash_review["cash_per_quarter"]),
            comparison=float(cash_review["legacy_amount"]),
        )
        for d in sorted(cash_review["ex_dates_agree_between_price_feeds"])
    ]
    if (
        parity["nominal_errors"]
        or parity["native_formula_errors"]
        or parity["cash_differences"] != expected_cash
    ):
        raise ValueError("Native predecessor component verification failed")
    successor_parity = component_parity(prices_by_id[360], before[360].prices)
    if any(
        successor_parity[k]
        for k in (
            "nominal_errors",
            "native_formula_errors",
            "cash_differences",
            "vendor_derived_residuals",
        )
    ):
        raise ValueError("Native successor component verification failed")
    # This inventory is bounded to this security and window; never a generic
    # rule that unknown actions with factor 1 are safe.
    actions = read(base + "praxair-corporate-actions.json")["sources"]["corporate_actions"]
    if (
        actions["failures"]
        or actions["requested_rics"] != ["PX.N^J18"]
        or actions["window"] != ["2010-01-01", "2018-10-31"]
        or len(actions["records"]) != 1
    ):
        raise ValueError("Predecessor action inventory changed")
    action = actions["records"][0]
    expected_action = {
        "Instrument": "PX.N^J18",
        "Corporate Change Event Type": "Buyback",
        "Capital Change Effective Date": "2010-07-28",
        "Adjustment Factor": "1",
        "Capital Change Is Rescinded": "False",
        "Corporate Action Description": "Total Amount Returned USD 1500000000.",
    }
    if any(action[k] != v for k, v in expected_action.items()):
        raise ValueError("Unreviewed predecessor action")
    if basis["reviewed_actions"] != [
        dict(type="Exchange Offer", effective_date="2018-10-31", treatment="ownership_boundary"),
        dict(
            type="Buyback",
            effective_date="2019-01-21",
            treatment="issuer_buyback_not_holder_distribution",
        ),
    ]:
        raise ValueError("Successor action review changed")
    # Hashes above bind the independently reviewed successor action payload.
    parent_intervals = read(base + "membership-reconciled.json")["intervals"]
    split = split_predecessor_intervals(
        parent_intervals,
        predecessor="PX.N^J18",
        successor="LIN.OQ",
        effective_date=BOUNDARY,
        evidence=source["source_sha256"],
    )
    memberships = {
        tid: [
            dict(
                valid_from=day(r["valid_from"]),
                valid_to=day(r["valid_to"]) if r["valid_to"] else None,
                index_id="SPX",
            )
            for r in split
            if r["security_id"] == ric
        ]
        for tid, ric in [(947, "PX.N^J18"), (360, "LIN.OQ")]
    }
    if memberships[947] != [dict(valid_from=date(2010, 1, 1), valid_to=BOUNDARY, index_id="SPX")]:
        raise ValueError("Predecessor membership changed")
    if memberships[360] != [dict(valid_from=BOUNDARY, valid_to=None, index_id="SPX")]:
        raise ValueError("Successor membership changed")
    valuation = successor_horizon_values(
        target_prices=target.prices,
        successor_prices=prices_by_id[360],
        target_last_trade=FINAL,
        ownership_close=BOUNDARY,
        horizon_dates=target.security_events[0]["horizon_values"],
        cash_per_share=0,
        exchange_ratio=1,
    )
    if (
        valuation != basis["valuation"]
        or valuation["missing_horizons"]
        or len(valuation["horizon_values"]) != 252
    ):
        raise ValueError("Native terminal valuation does not reproduce")
    evidence = dict(
        artifact=source["archived_source"],
        sha256=source["source_sha256"],
        reason="US price history before 2018-10-31 belongs to separate predecessor Praxair",
    )
    provenance = dict(
        source,
        source_hashes=hashes,
        native_successor_prices_sha256=basis["native_prices_sha256"],
        price_policy=basis["basis_policy"],
        sector_policy="No sector intervals applied; missing evidence remains unknown",
    )
    return before, prices_by_id[947], memberships, valuation, evidence, provenance, parity


async def capture(conn):
    result = {}
    for table, order in [
        ("tickers", "ticker_id"),
        ("price_history", "ticker_id,trade_date"),
        ("index_membership", "ticker_id,index_id,valid_from"),
        ("security_events", "ticker_id,effective_date"),
        ("sector_history", "ticker_id,valid_from"),
        ("fundamentals", "ticker_id,accession_number"),
        (
            "accounting_facts",
            "ticker_id,accession_number,metric,span_start,period_end,source_concept",
        ),
    ]:
        result[table] = [
            dict(r)
            for r in await conn.fetch(
                f"select * from {table} where ticker_id=any($1::bigint[]) "
                f"order by {order} for update",
                IDS,
            )
        ]
    return result


def verify_stored_prices(expected, stored):
    if len(expected) != len(stored):
        raise ValueError("Stored price count mismatch")
    for p, q in zip(expected, stored, strict=True):
        for key, value in p.items():
            if key in ("open", "high", "low", "close", "adj_close") and value is not None:
                # Existing NUMERIC(18,6) schema. This is storage quantization,
                # not a change to the independent source-parity tolerance.
                if abs(Decimal(str(value)) - Decimal(str(q[key]))) > Decimal("0.000000500001"):
                    raise ValueError("Stored price exceeds schema quantization")
            elif isinstance(value, (float, Decimal)) and isinstance(q[key], (float, Decimal)):
                if abs(Decimal(str(value)) - Decimal(str(q[key]))) > Decimal("1e-12"):
                    raise ValueError(f"Stored numeric field mismatch: {key}")
            elif value != q[key] and canonical(value) != canonical(q[key]):
                raise ValueError(f"Stored price field mismatch: {key}")


async def run(a):
    if Path(a.output).exists():
        raise ValueError("Receipt already exists")
    before, prices, membership, valuation, evidence, provenance, parity = prepare(
        a.parent, a.candidate, a.successor_basis
    )
    migration = Path("backend/db/migrations/022_price_history_start.sql").read_bytes()
    result = dict(
        applied=a.apply,
        migration_sha256=sha(migration),
        implementation_hashes={
            path: sha(Path(path).read_bytes())
            for path in (
                "scripts/apply_praxair_reference.py",
                "backend/ml/dataset.py",
                "backend/ml/factors/pit.py",
                "backend/ml/factors/assembly.py",
                "backend/ml/terminal.py",
                "backend/ingestion/prices_lseg.py",
            )
        },
        input_sha256=sha(canonical(provenance)),
        sectors_applied=0,
        raw_prices_deleted=0,
        global_gates_promoted=0,
        vendor_derived_residuals=parity["vendor_derived_residuals"],
    )
    async with pool_context(max_size=1, command_timeout=300) as pool:  # noqa: SIM117
        async with pool.acquire() as conn:
            tx = conn.transaction(isolation="repeatable_read")
            await tx.start()
            try:
                await conn.execute("set local lock_timeout='10s'")
                await conn.execute(migration.decode())
                await conn.fetch(
                    "select ticker_id from tickers where ticker_id=any($1::bigint[]) for update",
                    IDS,
                )
                preimage = await capture(conn)
                tickers = {r["ticker_id"]: r for r in preimage["tickers"]}
                for tid, symbol, cik in [(947, "LSEG:PX.N^J18", "884905"), (360, "LIN", "1707925")]:
                    r = tickers[tid]
                    if (
                        r["symbol"] != symbol
                        or int(r["cik"]) != int(cik)
                        or r["price_history_start"] is not None
                    ):
                        raise ValueError("Ticker identity or boundary changed")
                if not tickers[947]["research_only"] or tickers[947]["active"]:
                    raise ValueError("Expected inactive research-only predecessor")
                if any(
                    r["ticker_id"] == 947
                    for table in (
                        "price_history",
                        "index_membership",
                        "security_events",
                        "sector_history",
                    )
                    for r in preimage[table]
                ):
                    raise ValueError(
                        "Predecessor already has reference data; explicit review required"
                    )
                live = {
                    f.ticker_id: f
                    for f in await load_frames(
                        conn, symbols=["LIN", "LSEG:PX.N^J18"], include_research=True
                    )
                }
                for tid in IDS:
                    # Empty historical sectors now mean unknown; no static fallback.
                    left, right = asdict(before[tid]), asdict(live[tid])
                    left["sector_history"] = left["sector_history"] or []
                    if canonical(left) != canonical(right):
                        raise ValueError(f"Live frame changed since parent audit: {tid}")
                old_membership = preimage["index_membership"]
                if (
                    len(old_membership) != 1
                    or old_membership[0]["index_id"] != "SPX"
                    or old_membership[0]["valid_to"] is not None
                ):
                    raise ValueError("Unexpected prior membership")
                if old_membership[0]["valid_from"] >= BOUNDARY:
                    raise ValueError("No predecessor membership to split")
                raw_preimage = canonical(preimage)
                preimage_path = Path(a.preimage)
                if preimage_path.exists():
                    if gzip.decompress(preimage_path.read_bytes()) != raw_preimage:
                        raise ValueError("Archived preimage differs from current database")
                else:
                    preimage_path.parent.mkdir(parents=True, exist_ok=True)
                    with gzip.open(preimage_path, "xb") as f:
                        f.write(raw_preimage)
                result.update(preimage=str(preimage_path), preimage_sha256=sha(raw_preimage))
                await ingest_verified_prices(conn, prices)
                await conn.execute(
                    "update tickers set price_history_start=$2,"
                    "price_history_start_source=$3::jsonb "
                    "where ticker_id=$1",
                    360,
                    BOUNDARY,
                    json.dumps(evidence, sort_keys=True),
                )
                await conn.execute(
                    "update tickers set security_retired_at=$2,"
                    "security_retired_source=$3,removed_at=$4 "
                    "where ticker_id=$1",
                    947,
                    FINAL,
                    json.dumps(provenance, sort_keys=True),
                    day(BOUNDARY),
                )
                source = json.dumps(provenance, sort_keys=True)
                await conn.execute(
                    "update index_membership set valid_from=$2,source=$3 where ticker_id=$1",
                    360,
                    BOUNDARY,
                    source,
                )
                await conn.execute(
                    "insert into index_membership(ticker_id,index_id,valid_from,valid_to,source) "
                    "values(947,'SPX',$1,$2,$3)",
                    date(2010, 1, 1),
                    BOUNDARY,
                    source,
                )
                await conn.execute(
                    """insert into security_events(ticker_id,effective_date,event_type,
                    consideration_type,proceeds_basis,horizon_values,source,verified)
                    values(947,$1,'acquisition','successor','adj_close',$2::jsonb,$3,true)""",
                    BOUNDARY,
                    json.dumps(valuation["horizon_values"], sort_keys=True),
                    source,
                )
                after = await capture(conn)
                for table in ("sector_history", "fundamentals", "accounting_facts"):
                    if preimage[table] != after[table]:
                        raise ValueError("Unrelated issuer or sector data changed")
                if [r for r in preimage["price_history"] if r["ticker_id"] == 360] != [
                    r for r in after["price_history"] if r["ticker_id"] == 360
                ]:
                    raise ValueError("Raw successor evidence changed")
                verify_stored_prices(
                    prices, [r for r in after["price_history"] if r["ticker_id"] == 947]
                )
                frames = {
                    f.ticker_id: f
                    for f in await load_frames(
                        conn, symbols=["LIN", "LSEG:PX.N^J18"], include_research=True
                    )
                }
                allowed = {
                    "prices",
                    "membership",
                    "security_events",
                    "removed_at",
                    "price_history_start",
                    "price_history_start_source",
                }
                for tid in IDS:
                    original_fields = {
                        k: v for k, v in asdict(live[tid]).items() if k not in allowed
                    }
                    actual_fields = {
                        k: v for k, v in asdict(frames[tid]).items() if k not in allowed
                    }
                    if canonical(original_fields) != canonical(actual_fields):
                        raise ValueError("Unrelated frame fields changed")
                if frames[360].prices != [
                    p for p in before[360].prices if p["trade_date"] >= BOUNDARY
                ]:
                    raise ValueError("Predecessor leaked into successor frame")
                if frames[947].sector_history != []:
                    raise ValueError("Unknown predecessor sector was fabricated")
                for tid in IDS:
                    actual = [
                        {k: v for k, v in r.items() if k != "ticker_id"}
                        for r in frames[tid].membership
                    ]
                    if actual != membership[tid]:
                        raise ValueError("Stored membership failed replay")
                restored = {h: 0 for h in HORIZON_TRADING_DAYS}
                target = frames[947]
                verify_stored_prices(
                    [{k: v for k, v in p.items() if k != "open"} for p in prices], target.prices
                )
                event = target.security_events[0]
                for i in range(len(prices) - 1):
                    if prices[i]["trade_date"] < FINAL - timedelta(days=400):
                        continue
                    labels = documented_targets(target.prices, i, [event], date(2019, 11, 1))
                    for h, v in labels.items():
                        restored[h] += int(v[1] and v[4] == "acquisition")
                if restored != HORIZON_TRADING_DAYS:
                    raise ValueError(
                        "Stored terminal label coverage differs from reviewed horizons"
                    )
                result.update(
                    predecessor_rows=len(target.prices),
                    successor_rows=len(frames[360].prices),
                    retained_raw_successor_rows=sum(
                        r["ticker_id"] == 360 for r in after["price_history"]
                    ),
                    terminal_horizons=len(valuation["horizon_values"]),
                    restored_terminal_labels=restored,
                    stored_frames_sha256=sha(canonical([asdict(frames[tid]) for tid in IDS])),
                )
                if a.apply:
                    if not a.rehearsal or json.loads(Path(a.rehearsal).read_text()) != json.loads(
                        canonical(dict(result, applied=False))
                    ):
                        raise ValueError(
                            "Application requires an exactly matching rollback rehearsal"
                        )
                    await tx.commit()
                else:
                    await tx.rollback()
            except BaseException:
                if not conn.is_closed():
                    await tx.rollback()
                raise
    write_json_new(a.output, result)
    print(json.dumps(json_value(result), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", default=".research/reference-gate-audit-2026-09-08-v2")
    p.add_argument("--candidate", default=".research/reference-praxair-candidate-2026-09-08-v1")
    p.add_argument("--successor-basis", default=".research/praxair-successor-basis-v1")
    p.add_argument(
        "--preimage", default=".research/verification/praxair-reference-preimage.json.gz"
    )
    p.add_argument("--output", required=True)
    p.add_argument("--rehearsal")
    p.add_argument("--apply", action="store_true")
    asyncio.run(run(p.parse_args()))
