"""Freeze available SPX inputs and pre-result exclusions for exploration."""

import argparse
import copy
import gzip
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from backend.ml.research import create_snapshot, load_snapshot


def run(snapshot, output):
    inputs, manifest = load_snapshot(snapshot)
    frames = copy.deepcopy(inputs["frames"])
    observation_end = max(p["trade_date"] for f in frames for p in f.prices)
    hashes = {}

    def read(path):
        blob = Path(path).read_bytes()
        hashes[str(path)] = hashlib.sha256(blob).hexdigest()
        return blob

    mapping = json.loads(read(".research/verification/security-mapping-exact-2026-09-18-v2.json"))
    membership = json.loads(read(".research/verification/membership-reconciled.json"))
    grouped = defaultdict(list)
    for row in membership["intervals"]:
        grouped[row["security_id"]].append(
            dict(
                index_id="SPX",
                valid_from=date.fromisoformat(row["valid_from"]),
                valid_to=date.fromisoformat(row["valid_to"]) if row["valid_to"] else None,
            )
        )
    candidates = {}
    for row in mapping["mappings"]:
        ids = row["candidate_ticker_ids"] or row["symbol_only_review_ids"]
        if len(ids) == 1 and row["status"] != "blocked_multiple_ric_candidates":
            candidates[row["ric"]] = ids[0]
    multiplicity = Counter(candidates.values())
    by_id = {f.ticker_id: f for f in frames}
    for frame in frames:
        frame.membership = []
    intended = []
    for index, (ric, intervals) in enumerate(sorted(grouped.items())):
        tid = candidates.get(ric)
        mapped = tid in by_id and multiplicity[tid] == 1
        # Negative IDs identify unmapped denominator entries only. They are never
        # created as securities or passed into the model.
        intended.append(
            dict(
                ticker_id=tid if mapped else -index - 1,
                ric=ric,
                mapped=bool(mapped),
                membership=intervals,
            )
        )
        if mapped:
            by_id[tid].membership = intervals
    root = Path(".research/normalized-historical-candidate-v3")
    report = json.loads(read(root / "report.json"))
    raw = gzip.decompress(read(root / "prices.pkl.gz"))
    if hashlib.sha256(raw).hexdigest() != report["input_sha256"]:
        raise ValueError("Historical candidate hash changed")
    prices = pickle.loads(raw)
    returns = json.loads(
        read(".research/verification/historical-total-return-audit-2026-09-19-v1.json")
    )
    actions = json.loads(
        read(".research/verification/historical-action-components-2026-09-19-v1.json")
    )
    # Accept incomplete certification, not identified disagreements. This fixed
    # source rule is applied before any model results exist.
    eligible = {
        r["ric"]
        for r in returns["securities"]
        if r["status"] == "cash_returns_match_source_completeness_not_certified"
    }
    for row in actions["securities"]:
        if row["declared_actions_without_exact_price_step"]:
            eligible.discard(row["ric"])
        elif row["events"] and all(
            e["status"] == "declared_split_return_corroborated" for e in row["events"]
        ):
            eligible.add(row["ric"])
    by_id = {f.ticker_id: f for f in frames}
    restored = []
    for ric in sorted(eligible):
        rows = prices[ric]
        tid = rows[0]["ticker_id"]
        frame = by_id[tid]
        if frame.prices:
            raise ValueError("Historical augmentation would replace existing prices")
        frame.prices = rows
        restored.append(dict(ric=ric, ticker_id=tid, rows=len(rows)))
    exclusions = []
    # These archived conflicts include known reused symbols; native CA/HAR
    # recoveries supersede their old raw-price flags. Other conflicts remain out.
    repaired = {712, 749}
    for row in mapping["mappings"]:
        if row["status"] not in (
            "history_conflict",
            "reuse_conflict",
            "blocked_multiple_ric_candidates",
        ):
            continue
        for tid in row["candidate_ticker_ids"]:
            if tid in repaired or tid not in by_id:
                continue
            frame = by_id[tid]
            if row["status"] == "history_conflict" and row["first_trade_date"]:
                start = date.fromisoformat(row["first_trade_date"])
                removed = sum(p["trade_date"] < start for p in frame.prices)
                frame.prices = [p for p in frame.prices if p["trade_date"] >= start]
                exclusions.append(
                    dict(
                        ticker_id=tid,
                        ric=row["ric"],
                        reason="unresolved_pre_listing_history",
                        before=str(start),
                        price_rows=removed,
                    )
                )
            else:
                exclusions.append(
                    dict(
                        ticker_id=tid,
                        ric=row["ric"],
                        reason=row["status"],
                        price_rows=len(frame.prices),
                    )
                )
                frame.prices = []
    for frame in frames:
        if frame.sector_history is None:
            frame.sector_history = []  # Never introduce static-sector fallback.
    metadata = copy.deepcopy(manifest["metadata"])
    metadata["universe"] = "sp500_exploratory"
    metadata["exploratory"] = dict(
        intended_cohort=intended,
        observation_end=observation_end,
        exclusions=exclusions,
        restored_provisional_histories=restored,
        source_hashes=hashes,
        parent_snapshot_sha256=manifest["input_sha256"],
        policy="docs/experiment_gate_status.md: user-authorized exploratory standard 2026-09-19",
        limitations=[
            "Source certification is incomplete; this is exploratory, not promotion evidence.",
            "SPX membership uses the reconstructed archive; mappings remain provisional.",
            "Identity candidates are provisional; known reuse and ambiguity are excluded.",
            "Early-history truncation may conservatively omit venue-continuous observations.",
            "Missing inputs remain missing; unknown terminal payoffs stay masked.",
            "Historical augmentation uses matched paths from one vendor, not independent vendors.",
            "2024 onward is excluded from model selection and is not an untouched holdout.",
        ],
    )
    result = create_snapshot(output, frames, inputs["macro"], metadata)
    print(
        dict(
            frames=len(frames),
            restored_histories=len(restored),
            exclusions=len(exclusions),
            snapshot_sha256=result["input_sha256"],
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.snapshot, args.output)
