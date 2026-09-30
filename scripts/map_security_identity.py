"""Audit exact security rows without promoting issuer matches into dated mappings.

Each requested RIC produces one result. Blank-RIC issuer-context rows are retained
as diagnostics but cannot supply a security's CIK, ISIN or lifecycle. Exact stored
RICs or a class-preserving symbol AND matching issuer produce candidates only;
source lifecycles and price spans remain separate checks, not certification.
"""

import argparse
import asyncio
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

from backend.ingestion.db import pool_context
from backend.ml.research import write_json_new

REUSE_GRACE = timedelta(days=90)  # matches the load_frames symbol-reuse guard


def ric_stem(ric):
    """Class-preserving symbol from a RIC: strip the retirement and exchange suffix."""
    s = re.sub(r"\^[A-Z]\d{2}$", "", ric)
    s = re.sub(r"\.(N|OQ|O|A|P|K|TH|B|DE|L)$", "", s)
    return s.replace("b", "-B") if s.endswith("b") else s


def _d(value):
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def history_check(ticker, first_trade, retire):
    """Does the ticker's stored price span fit inside the security's lifetime?"""
    first_px, last_px = ticker["first_px"], ticker["last_px"]
    if last_px is None:
        return "no_price_history", None
    if retire and last_px > retire + REUSE_GRACE:
        return "reuse_conflict", f"bars to {last_px} after retirement {retire}"
    if first_trade and first_px and first_px < first_trade - REUSE_GRACE:
        return "predates_security", f"bars from {first_px} before first trade {first_trade}"
    return "consistent", None


def _value(value):
    text = str(value).strip() if value is not None else ""
    return "" if text in ("", "<NA>", "None", "nan", "NaT") else text


def _cik(value):
    text = _value(value)
    return str(int(text)) if text.isdigit() else None


def build_mappings(data, tickers):
    from scripts.audit_security_mapping import normalize_symbol

    grouped = defaultdict(list)
    for row in data["records"]:
        grouped[row["Instrument"]].append(row)
    if len(set(data["requested_rics"])) != len(data["requested_rics"]):
        raise ValueError("Duplicate requested security")
    mappings = []
    for ric in data["requested_rics"]:
        rows = grouped[ric]
        exact = [r for r in rows if r.get("RIC") == ric]
        entry = dict(
            ric=ric,
            candidate_ticker_ids=[],
            candidate_symbols=[],
            rule=None,
            cik=None,
            isin=None,
            first_trade_date=None,
            retire_date=None,
            issuer_context_rows=sum(not _value(r.get("RIC")) for r in rows),
        )
        if len(exact) != 1:
            mappings.append(dict(entry, status="blocked_source_security_rows"))
            continue
        rec = exact[0]
        cik, isin = _cik(rec.get("CIK Number")), _value(rec.get("ISIN"))
        first, retire = _d(rec.get("First Trade Date")), _d(rec.get("RetireDate"))
        entry.update(
            cik=cik,
            isin=isin or None,
            first_trade_date=str(first) if first else None,
            retire_date=str(retire) if retire else None,
        )
        entry["issuer_only_review_ids"] = sorted(
            t["ticker_id"] for t in tickers if cik and _cik(t.get("cik")) == cik
        )
        if not isin:
            mappings.append(dict(entry, status="blocked_missing_security_identifier"))
            continue
        symbols = {
            normalize_symbol(_value(rec.get(k)))
            for k in ("Ticker Symbol", "Exchange Ticker")
            if _value(rec.get(k))
        }
        # A retired RIC often omits the ticker fields. The stem still preserves
        # the class; an issuer match alone is never enough to select it.
        symbols.add(normalize_symbol(ric_stem(ric)))
        native = [t for t in tickers if t.get("ric") == ric]
        if native:
            cands, rule = native, "exact_stored_ric"
            if any(cik and _cik(t.get("cik")) and _cik(t.get("cik")) != cik for t in native):
                entry.update(
                    candidate_ticker_ids=[t["ticker_id"] for t in native],
                    candidate_symbols=[t["symbol"] for t in native],
                    rule=rule,
                )
                mappings.append(dict(entry, status="blocked_stored_issuer_conflict"))
                continue
        else:
            cands = [
                t
                for t in tickers
                if cik and _cik(t.get("cik")) == cik and normalize_symbol(t["symbol"]) in symbols
            ]
            rule = "class_symbol_and_issuer" if cands else None
        entry["symbol_only_review_ids"] = sorted(
            t["ticker_id"] for t in tickers if normalize_symbol(t["symbol"]) in symbols
        )
        entry.update(
            candidate_ticker_ids=[t["ticker_id"] for t in cands],
            candidate_symbols=[t["symbol"] for t in cands],
            rule=rule,
        )
        if len(cands) == 1:
            state, note = history_check(cands[0], first, retire)
            entry.update(
                history_check=state,
                history_note=note,
                status={
                    "consistent": "identity_candidate_requires_history_review",
                    "no_price_history": "identity_candidate_no_history",
                    "reuse_conflict": "reuse_conflict",
                    "predates_security": "history_conflict",
                }[state],
            )
        else:
            entry["status"] = (
                "ambiguous"
                if cands
                else (
                    "issuer_only_requires_security_review"
                    if entry["issuer_only_review_ids"]
                    else "symbol_only_requires_issuer_review"
                    if entry["symbol_only_review_ids"]
                    else "absent_from_database"
                )
            )
        mappings.append(entry)
    reverse = defaultdict(list)
    for row in mappings:
        if len(row["candidate_ticker_ids"]) == 1:
            reverse[row["candidate_ticker_ids"][0]].append(row)
    for rows in reverse.values():
        if len(rows) > 1:
            for row in rows:
                row["status_before_collision"] = row["status"]
                row["status"] = "blocked_multiple_ric_candidates"
                row["colliding_rics"] = sorted(r["ric"] for r in rows)
    return mappings


async def run(archive_path, output):
    raw = Path(archive_path).read_bytes()
    data = json.loads(raw)["sources"]["identity_archive"]
    async with (
        pool_context(max_size=1) as pool,
        pool.acquire() as conn,
        conn.transaction(isolation="repeatable_read", readonly=True),
    ):
        rows = [
            dict(r)
            for r in await conn.fetch(
                """select t.ticker_id, t.symbol, t.ric, t.cik, t.active, t.research_only,
                          t.removed_at, min(p.trade_date) first_px, max(p.trade_date) last_px
                     from tickers t left join price_history p using (ticker_id)
                    where t.asset_type='equity' group by t.ticker_id"""
            )
        ]
    mappings = build_mappings(data, rows)
    report = dict(
        source_artifact=str(archive_path),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        rule=(
            "Exact-RIC security rows; stored RIC or class symbol plus issuer. "
            "No issuer-only mapping."
        ),
        reuse_grace_days=REUSE_GRACE.days,
        counts=dict(Counter(r["status"] for r in mappings)),
        mappings=mappings,
        ticker_inventory=rows,
        source_failures=data.get("failures", []),
        database_writes=0,
        status="identity_audit_not_certified",
        limitations=[
            "Price spans use raw stored rows, including documented loader exclusions.",
            "The 90-day lifecycle diagnostic is not evidence of security continuity.",
            "Listing first-trade dates may describe venue changes; review before truncation.",
            "Candidates do not authorize membership, sector, price or retirement application.",
        ],
    )
    write_json_new(output, report)
    print(json.dumps(report["counts"], indent=1, sort_keys=True))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", default=".research/verification/identity-lifecycle-archive.json")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    asyncio.run(run(a.archive, a.output))
