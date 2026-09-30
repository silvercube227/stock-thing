"""Add archived monthly fixed-target consensus to a local exploratory snapshot."""

import argparse
import copy
import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from backend.ml.research import create_snapshot, load_snapshot


def parse_records(records, ric_to_id):
    """Use CalcDate availability and absolute targets; never join rolling FY labels."""
    parsed, counts = [], Counter()
    for raw in records:
        ric = raw.get("Instrument")
        if ric not in ric_to_id:
            counts["unmapped"] += 1
            continue
        try:
            as_of = date.fromisoformat(raw["Calc Date"][:10])
            period_end = date.fromisoformat(raw["Period End Date"][:10])
            update = date.fromisoformat(raw["Date"][:10])
            eps = float(raw["Earnings Per Share - Mean"])
        except (KeyError, TypeError, ValueError):
            counts["missing_or_invalid"] += 1
            continue
        fiscal = raw.get("Financial Period Absolute", "")
        if not math.isfinite(eps) or not re.fullmatch(r"FY\d{4}", fiscal):
            counts["missing_or_invalid"] += 1
            continue
        if update > as_of or period_end <= as_of or as_of >= date(2024, 1, 1):
            counts["unavailable_or_expired"] += 1
            continue
        parsed.append(dict(
            ticker_id=ric_to_id[ric], as_of_date=as_of,
            fiscal_period_end=period_end, fiscal_period_absolute=fiscal,
            contributor_id="consensus", eps=eps,
            adjustment_basis="lseg_current_adjusted_usd_exploratory_2026-09-19",
            source="lseg_monthly_calcdate_fixed_absolute_target",
            verified=False, exploratory_usable=True,
        ))
    return parsed, counts


def deduplicate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["ticker_id"], row["as_of_date"], row["fiscal_period_end"])].append(row)
    accepted, conflicts = [], 0
    for values in grouped.values():
        if len({(v["eps"], v["fiscal_period_absolute"]) for v in values}) != 1:
            conflicts += 1
        else:
            accepted.append(values[0])
    return sorted(accepted, key=lambda r: (r["ticker_id"], r["as_of_date"],
                                           r["fiscal_period_end"])), conflicts


def run(snapshot, source_manifest, output):
    inputs, manifest = load_snapshot(snapshot)
    metadata = copy.deepcopy(manifest["metadata"])
    cohort = metadata["exploratory"]["intended_cohort"]
    mapping = {r["ric"]: r["ticker_id"] for r in cohort if r["mapped"]}
    source_bytes = Path(source_manifest).read_bytes()
    archive = json.loads(source_bytes)
    rows, counts = [], Counter()
    for entry in archive["entries"]:
        blob = gzip.decompress(Path(entry["artifact"]).read_bytes())
        if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
            raise ValueError("Estimate archive hash mismatch")
        payload = json.loads(blob)
        if payload["request"] != entry["request"]:
            raise ValueError("Estimate archive request mismatch")
        parsed, rejected = parse_records(payload["records"], mapping)
        rows.extend(parsed)
        counts.update(rejected)
    rows, conflicts = deduplicate(rows)
    by_id = defaultdict(list)
    for row in rows:
        by_id[row["ticker_id"]].append(row)
    for frame in inputs["frames"]:
        if frame.fixed_estimates:
            raise ValueError("This preparation must not replace existing fixed estimates")
        frame.fixed_estimates = by_id.get(frame.ticker_id, [])
    evidence = dict(
        source_manifest=str(source_manifest),
        source_manifest_sha256=hashlib.sha256(source_bytes).hexdigest(),
        parent_snapshot_sha256=manifest["input_sha256"],
        observations=len(rows), securities=len(by_id), rejected=dict(counts),
        conflicting_keys_excluded=conflicts, verified=False, database_writes=0,
    )
    metadata["exploratory"]["analyst"] = evidence
    metadata["exploratory"]["limitations"] += [
        "Provisional analyst history uses monthly CalcDate and vendor absolute FY targets.",
        "Vendor fiscal ends and current adjusted USD basis are not independently certified.",
        "The unchanged 0.01 EPS cutoff is basis dependent; contributor breadth is unavailable.",
        "Nontrading-day snapshots become usable on the next available feature date.",
    ]
    result = create_snapshot(output, inputs["frames"], inputs["macro"], metadata)
    print(json.dumps(dict(evidence, input_sha256=result["input_sha256"]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.snapshot, args.source_manifest, args.output)
