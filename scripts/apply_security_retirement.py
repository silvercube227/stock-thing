"""Record vendor retirement dates so reused symbols truncate on evidence.

Only single-candidate mappings whose lifecycle check passed or which failed ONLY
by carrying bars past retirement are used. Ambiguous issuers (share classes,
merger successors) and securities that predate their mapped ticker are skipped:
those need dated mappings, not a single retirement stamp.
"""
import argparse
import asyncio
import hashlib
import json
from datetime import date
from pathlib import Path

from backend.ingestion.db import pool_context
from backend.ml.research import write_json_new

USABLE = ("resolved", "resolved_no_history", "reuse_conflict")


async def run(mapping_path, output, apply):
    raw = Path(mapping_path).read_bytes()
    mapping = json.loads(raw)
    provenance = "security_lifecycle:" + hashlib.sha256(raw).hexdigest()
    proposed = {}
    # A ticker that also maps to a security with no retirement date still holds a
    # live listing: FMC Technologies retired in 2017 but the same ticker carries
    # TechnipFMC to today. Stamping those would truncate nine years of real bars.
    live = {
        tid
        for m in mapping["mappings"]
        if not m["retire_date"]
        for tid in m["candidate_ticker_ids"]
    }
    for m in mapping["mappings"]:
        if m["status"] not in USABLE or len(m["candidate_ticker_ids"]) != 1:
            continue
        if not m["retire_date"] or m["candidate_ticker_ids"][0] in live:
            continue
        tid = m["candidate_ticker_ids"][0]
        retired = date.fromisoformat(m["retire_date"])
        # A ticker can map to several retired securities (a predecessor and its
        # successor share a CIK). The LAST retirement is the only one that could
        # bound the stored series; earlier ones would truncate legitimate bars.
        if tid not in proposed or retired > proposed[tid]["retired"]:
            proposed[tid] = {"retired": retired, "ric": m["ric"], "status": m["status"]}
    report = {"mapping_artifact": mapping_path, "mapping_sha256": hashlib.sha256(raw).hexdigest(),
              "provenance": provenance, "applied": apply, "updates": [], "skipped": []}
    async with pool_context(max_size=1) as pool:
        async with pool.acquire() as conn:
            tx = conn.transaction()
            await tx.start()
            try:
                for tid, info in sorted(proposed.items()):
                    row = await conn.fetchrow(
                        """select t.symbol, t.active, t.removed_at::date rm,
                                  (select max(p.trade_date) from price_history p
                                    where p.ticker_id = t.ticker_id) last_px
                             from tickers t where t.ticker_id = $1 for update""", tid)
                    if row is None:
                        continue
                    # A still-listed security must never be stamped as retired.
                    if row["active"] and row["last_px"] and row["last_px"] > info["retired"]:
                        report["skipped"].append({
                            "ticker_id": tid, "symbol": row["symbol"], "ric": info["ric"],
                            "retired": str(info["retired"]), "last_px": str(row["last_px"]),
                            "reason": "active ticker still trading after the mapped retirement"})
                        continue
                    report["updates"].append({
                        "ticker_id": tid, "symbol": row["symbol"], "ric": info["ric"],
                        "status": info["status"], "retired": str(info["retired"]),
                        "removed_at": str(row["rm"]) if row["rm"] else None,
                        "last_px": str(row["last_px"]) if row["last_px"] else None,
                        "bars_after_retirement": bool(row["last_px"] and row["last_px"] > info["retired"]),
                    })
                    if apply:
                        await conn.execute(
                            """update tickers set security_retired_at=$2, security_retired_source=$3
                                where ticker_id=$1""", tid, info["retired"], provenance)
                if apply:
                    await tx.commit()
                else:
                    await tx.rollback()
            except BaseException:
                await tx.rollback()
                raise
    write_json_new(output, report)
    late = [u for u in report["updates"] if u["bars_after_retirement"]]
    print({"applied": apply, "updates": len(report["updates"]),
           "with_bars_after_retirement": len(late), "skipped": len(report["skipped"])})
    for u in late[:12]:
        print(f"   {u['symbol']:<6} retired {u['retired']} but bars to {u['last_px']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mapping", default=".research/verification/security-mapping-resolved.json")
    p.add_argument("--output", required=True)
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()
    asyncio.run(run(a.mapping, a.output, a.apply))
