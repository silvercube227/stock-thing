"""Check declared capital events without inferring share ratios from prices."""

import argparse
import gzip
import hashlib
import json
import math
import pickle
from collections import Counter
from contextlib import suppress
from datetime import date
from pathlib import Path

from backend.ingestion.calendar import trading_days_between
from backend.ingestion.share_actions import classify_share_action, dedupe_actions
from backend.ml.research import write_json_new


def event_dates(row):
    result = set()
    for key in ("Capital Change Effective Date", "Capital Change Ex Date"):
        with suppress(TypeError, ValueError):
            result.add(date.fromisoformat(row.get(key, "")))
    return result


def audit(ric, prices, actions, returns, embedded):
    actions = dedupe_actions([r for r in actions if r["Instrument"] == ric])
    source = {}
    for row in returns:
        if row["Instrument"] != ric or row["Calc Date"] in source:
            raise ValueError("Wrong security or duplicate return date")
        source[row["Calc Date"]] = row
    steps = {p["trade_date"] for p in prices if p["split_factor"] != 1}
    results = []
    for prior, current in zip(prices[:-1], prices[1:], strict=True):
        day = current["trade_date"]
        if day not in steps:
            continue
        ds = str(day)
        on_date = [
            a
            for a in actions
            if day in event_dates(a)
            and a["Capital Change Is Rescinded"] == "False"
            and a["Corporate Change Event Type"] not in ("", "Buyback")
        ]
        out = dict(
            date=ds,
            price_factor=current["split_factor"],
            actions=on_date,
            status="capital_distribution_requires_review",
        )
        row = source.get(ds)
        if row is None or row["Date"] != ds:
            out["status"] = "missing_or_stale_return"
        elif len(trading_days_between(prior["trade_date"], day)) != 2:
            out["status"] = "gap_crossing_not_compared"
        else:
            reference = float(row["Daily Total Return"]) / 100
            if not math.isfinite(reference):
                raise ValueError("Invalid event return")
            out["source_return"] = reference
            cash = embedded.get(ds)
            hit = classify_share_action(day, current["split_factor"], on_date)
            if cash and not on_date:
                special = float(cash["amount"])
                total = current["dividend"]
                if not 0 < special <= total < prior["close"]:
                    raise ValueError("Invalid declared cash basis")
                # These diagnostics distinguish conventions. A best-fitting
                # formula is NOT selected as proof of vendor semantics.
                hypotheses = {
                    "cash_reinvested_at_close": (current["close"] + total) / prior["close"] - 1,
                    "special_cash_price_adjustment": (current["close"] + total - special)
                    / (prior["close"] - special)
                    - 1,
                    "special_cash_in_price_and_cash": (current["close"] + total)
                    / (prior["close"] - special)
                    - 1,
                }
                out.update(
                    status="special_cash_convention_requires_review",
                    declared_special_cash=special,
                    hypothesis_errors={k: v - reference for k, v in hypotheses.items()},
                )
            elif len(on_date) == 1 and hit is not None and current["dividend"] == 0:
                action = on_date[hit[0]]
                if action["Corporate Change Event Type"] in ("Share Split", "Share Consolidation"):
                    ratio = hit[1]  # Declared new/old terms, never observed factor.
                    expected = current["close"] * ratio / prior["close"] - 1
                    error = abs(expected - reference)
                    out.update(
                        share_ratio=ratio,
                        expected_return=expected,
                        absolute_error=error,
                        status="declared_split_return_corroborated"
                        if error <= 1e-6
                        else "declared_split_return_disagrees",
                    )
        results.append(out)
    missing_steps = []
    for action in actions:
        if action["Corporate Change Event Type"] in ("", "Buyback"):
            continue
        if action["Capital Change Is Rescinded"] != "False":
            continue
        dates = {
            d
            for d in event_dates(action)
            if prices[0]["trade_date"] <= d <= prices[-1]["trade_date"]
        }
        if dates and not dates.intersection(steps):
            missing_steps.append(action)
    return dict(
        ric=ric,
        events=results,
        declared_actions_without_exact_price_step=missing_steps,
        source_certified=False,
    )


def run(candidate, output):
    hashes = {}

    def read(path):
        blob = Path(path).read_bytes()
        hashes[str(path)] = hashlib.sha256(blob).hexdigest()
        return blob

    root = Path(candidate)
    report = json.loads(read(root / "report.json"))
    payload = gzip.decompress(read(root / "prices.pkl.gz"))
    if hashlib.sha256(payload).hexdigest() != report["input_sha256"]:
        raise ValueError("Candidate changed")
    prices = pickle.loads(payload)
    reviews = {r["ric"]: r for r in report["securities"]}
    actions_path = ".research/verification/registered-historical-actions.json"
    actions = json.loads(read(actions_path))["sources"]["corporate_actions"]
    if actions["failures"]:
        raise ValueError("Incomplete action archive")
    # Replay every original embedded-cash input hash, not just the report entry.
    embedded_path = ".research/verification/historical-embedded-cash-review.json"
    embedded = json.loads(read(embedded_path))
    if hashes[embedded_path] != report["source_hashes"]["embedded_cash_review"]:
        raise ValueError("Embedded cash review differs from candidate")
    for path, expected in embedded["source_hashes"].items():
        if hashlib.sha256(read(path)).hexdigest() != expected:
            raise ValueError("Embedded cash evidence changed")
    manifest = json.loads(
        read(".research/verification/historical-total-return-archive-2026-09-19-v1.json")
    )
    returns = {}
    for entry in manifest["entries"]:
        raw = gzip.decompress(read(entry["artifact"]))
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("Return archive changed")
        payload = json.loads(raw)
        if payload["request"] != entry["request"]:
            raise ValueError("Return request changed")
        ric = entry["request"]["universe"]
        if ric in returns or entry["request"]["parameters"].get("Curn") != "USD":
            raise ValueError("Duplicate security or unsupported currency")
        returns[ric] = payload["records"]
    if returns.keys() != prices.keys():
        raise ValueError("Return/candidate scope differs")
    results = [
        audit(
            ric,
            rows,
            actions["records"],
            returns[ric],
            reviews[ric].get("embedded_cash_review", {}),
        )
        for ric, rows in prices.items()
    ]
    read(__file__)
    read("backend/ingestion/share_actions.py")
    counts = dict(Counter(event["status"] for r in results for event in r["events"]))
    write_json_new(
        output,
        dict(
            status="action_components_reviewed_not_source_certified",
            counts=counts,
            securities=results,
            source_hashes=hashes,
            database_writes=0,
            gates_promoted=0,
        ),
    )
    print(counts)
    print(
        "Actions without exact price step:",
        sum(len(r["declared_actions_without_exact_price_step"]) for r in results),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default=".research/normalized-historical-candidate-v3")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.candidate, args.output)
