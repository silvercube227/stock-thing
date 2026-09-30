"""Classify observed price adjustments against archived corporate actions.

Four ordered rules, each recorded per event so a reader can see which one applied:

  exact          the strict rule: event date and terms both corroborate
  pricing_only   every action on the date is `LSEG Pricing Only`, so the parent
                 share count is unchanged and the ratio is 1.0
  terms_offset   one split/consolidation within four days whose declared terms
                 equal the observed factor exactly
  mixed          one share-changing action plus price-only distributions on the
                 same date; the ratio comes only from the share-changing record

Share ratios always come from declared terms or from the adjustment type, never
from the observed price. `--apply` re-derives every classification from the
archives inside the write transaction and refuses if the stored price factor has
changed, so the report cannot assert something the source no longer supports.
"""
import argparse
import asyncio
import hashlib
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

from backend.ingestion.db import pool_context
from backend.ingestion.share_actions import (
    classify_mixed_capital_change,
    classify_pricing_only,
    classify_share_action,
    classify_terms_offset,
    dedupe_actions,
)
from backend.ml.research import write_json_new

RULES = ("exact", "pricing_only", "terms_offset", "mixed")


def load_sources(paths):
    records, digests = [], {}
    for p in paths:
        raw = Path(p).read_bytes()
        digests[str(p)] = hashlib.sha256(raw).hexdigest()
        records += json.loads(raw)["sources"]["corporate_actions"]["records"]
    return dedupe_actions(records), digests


def classify(observed_date, price_factor, records):
    hit = classify_share_action(observed_date, price_factor, records)
    if hit is not None:
        return "exact", hit[1], [hit[0]], 0.0
    for name, fn in (("pricing_only", classify_pricing_only),
                     ("terms_offset", classify_terms_offset),
                     ("mixed", classify_mixed_capital_change)):
        hit = fn(observed_date, price_factor, records)
        if hit is None:
            continue
        indices, ratio, extra = hit
        return name, ratio, ([indices] if isinstance(indices, int) else list(indices)), extra
    return None


async def run(sources, mapping_path, output, apply):
    records, digests = load_sources(sources)
    mapping_raw = Path(mapping_path).read_bytes()
    mapping = json.loads(mapping_raw)
    by_ticker = defaultdict(list)
    for m in mapping["mappings"]:
        for tid in m["candidate_ticker_ids"]:
            # One RIC can appear in several archived identity records; scoping a
            # ticker to the same RIC twice would double every event on a date and
            # square the adjustment factor.
            if m["ric"] not in by_ticker[tid]:
                by_ticker[tid].append(m["ric"])
    by_instrument = defaultdict(list)
    for r in records:
        by_instrument[r["Instrument"]].append(r)

    report = {
        "sources": digests,
        "mapping_artifact": mapping_path,
        "mapping_sha256": hashlib.sha256(mapping_raw).hexdigest(),
        "rules": list(RULES),
        "applied": apply,
        "events": [],
        "unresolved": [],
    }
    async with pool_context(max_size=1) as pool:
        async with pool.acquire() as conn:
            tx = conn.transaction()
            await tx.start()
            try:
                rows = await conn.fetch(
                    """select p.ticker_id, t.symbol, p.trade_date, p.split_factor
                         from price_history p join tickers t using (ticker_id)
                        where p.split_factor <> 1 and p.share_split_factor is null
                        order by p.trade_date for update of p"""
                )
                for row in rows:
                    scoped, rics = [], by_ticker.get(row["ticker_id"], [])
                    for ric in rics:
                        scoped += by_instrument.get(ric, [])
                    scoped = dedupe_actions(scoped)
                    factor, observed = float(row["split_factor"]), row["trade_date"]
                    entry = {"ticker_id": row["ticker_id"], "symbol": row["symbol"],
                             "trade_date": str(observed), "observed_price_factor": factor,
                             "rics": rics}
                    if not scoped:
                        report["unresolved"].append({**entry, "reason": "no archived action for any mapped RIC"})
                        continue
                    hit = classify(observed, factor, scoped)
                    if hit is None:
                        report["unresolved"].append({**entry, "reason": "no rule corroborates the observed adjustment"})
                        continue
                    rule, ratio, indices, extra = hit
                    if not (math.isfinite(ratio) and ratio > 0):
                        report["unresolved"].append({**entry, "reason": "non-finite share ratio"})
                        continue
                    report["events"].append({
                        **entry, "rule": rule, "share_split_factor": ratio,
                        "price_disagreement": float(extra) if rule != "terms_offset" else None,
                        "date_offset_days": int(extra) if rule == "terms_offset" else None,
                        "source_records": [scoped[i] for i in indices],
                    })
                provenance = "classified_share_action:" + hashlib.sha256(
                    json.dumps(report["events"], sort_keys=True, default=str).encode()
                ).hexdigest()
                report["provenance"] = provenance
                if apply:
                    for e in report["events"]:
                        await conn.execute(
                            """update price_history set share_split_factor=$3, share_action_source=$4
                                where ticker_id=$1 and trade_date=$2""",
                            e["ticker_id"], date.fromisoformat(e["trade_date"]),
                            e["share_split_factor"], provenance,
                        )
                    await tx.commit()
                else:
                    await tx.rollback()
            except BaseException:
                await tx.rollback()
                raise
    write_json_new(output, report)
    counts = defaultdict(int)
    for e in report["events"]:
        counts[e["rule"]] += 1
    print({"applied": apply, "classified": len(report["events"]),
           "unresolved": len(report["unresolved"]), **counts})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--mapping", default=".research/verification/security-mapping-resolved.json")
    p.add_argument("--output", required=True)
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()
    asyncio.run(run(a.sources, a.mapping, a.output, a.apply))
