"""Full-session cash-return corroboration for the scoped CA/HAR native recovery."""

import argparse
import gzip
import json
import math
import pickle
from pathlib import Path

from backend.ingestion.calendar import trading_days_between
from backend.ml.research import write_json_new
from scripts.apply_registered_membership import digest

CANDIDATE = Path(".research/original-security-normalized-candidate-v2")
RETURNS = Path(".research/verification/total-return-archive-2026-09-19-v1.json")
ACTIONS = Path(".research/verification/original-cash-actions-2026-09-19-v1.json")
CASH = Path(".research/verification/original-cash-candidate-2026-09-19-v1.json")


def compare(prices, returns):
    by_date = {}
    for row in returns:
        d = row["Calc Date"]
        if d in by_date or row["Date"] != d:
            raise ValueError("Duplicate or stale total-return dates")
        by_date[d] = float(row["Daily Total Return"]) / 100
    if set(by_date) != {str(r["trade_date"]) for r in prices}:
        raise ValueError("Total-return and native-price session coverage differs")
    expected_sessions = trading_days_between(prices[0]["trade_date"], prices[-1]["trade_date"])
    if tuple(r["trade_date"] for r in prices) != expected_sessions:
        raise ValueError("Native history has missing or duplicate exchange sessions")
    errors = []
    for prior, current in zip(prices[:-1], prices[1:], strict=True):
        if current["split_factor"] != 1:
            raise ValueError("This scoped cash-only check cannot certify capital adjustments")
        # Total return reinvests cash at the ex-date close. The stored Yahoo-style
        # back-adjusted price uses a different dividend convention; do not compare
        # that price ratio directly or alter its frozen normalization formula.
        actual = (current["close"] + current["dividend"]) / prior["close"] - 1
        reference = by_date[str(current["trade_date"])]
        error = abs(actual - reference)
        if not math.isfinite(reference) or error > 1e-6:
            raise ValueError(f"Cash-return disagreement on {current['trade_date']}")
        errors.append(error)
    return dict(
        native_rows=len(prices),
        compared_returns=len(errors),
        max_absolute_return_error=max(errors),
        cash_dates=sum(r["dividend"] > 0 for r in prices),
        missing_sessions=0,
    )


def prepare():
    hashes = {}

    def read(path):
        blob = Path(path).read_bytes()
        hashes[str(path)] = digest(blob)
        return blob

    cash = json.loads(read(CASH))
    for path, expected in cash["source_hashes"].items():
        if digest(read(path)) != expected:
            raise ValueError("Staged cash evidence changed")
    report = json.loads(read(CANDIDATE / "report.json"))
    raw = gzip.decompress(read(CANDIDATE / "prices.pkl.gz"))
    if digest(raw) != report["input_sha256"]:
        raise ValueError("Price candidate changed")
    prices = pickle.loads(raw)

    def archive(path):
        out = {}
        for entry in json.loads(read(path))["entries"]:
            raw = gzip.decompress(read(entry["artifact"]))
            if digest(raw) != entry["sha256"]:
                raise ValueError("LSEG source archive changed")
            payload = json.loads(raw)
            if payload["request"] != entry["request"]:
                raise ValueError("Source request mismatch")
            out[entry["request"]["universe"]] = payload
        return out

    returns, actions = archive(RETURNS), archive(ACTIONS)
    results = []
    for outcome in cash["outcomes"]:
        ric = outcome["ric"]
        if ric not in ("CA.OQ^K18", "HAR.N^C17"):
            raise ValueError("Unreviewed security in scoped cash recovery")
        for row in actions[ric]["records"]:
            if row["Instrument"] != ric or row["Corporate Change Event Type"] not in (
                "",
                "Buyback",
            ):
                raise ValueError("Capital-action inventory needs additional review")
            if (
                row["Corporate Change Event Type"] == "Buyback"
                and float(row["Adjustment Factor"]) != 1
            ):
                raise ValueError("Unexpected buyback adjustment")
        if returns[ric]["request"]["parameters"].get("Curn") != "USD":
            raise ValueError("Unexpected return currency")
        results.append(
            dict(
                ric=ric,
                **compare(prices[ric], returns[ric]["records"]),
                corporate_action_records=actions[ric]["records"],
            )
        )
    read(__file__)
    return prices, cash["outcomes"], results, hashes


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    _, _, results, hashes = prepare()
    write_json_new(
        p.parse_args().output,
        dict(
            status="scoped_cash_recovery_components_corroborated",
            securities=results,
            source_hashes=hashes,
            database_writes=0,
            gates_promoted=0,
            limitations=[
                "Separate vendor total-return and price/action paths are not independent vendors.",
                "CA/HAR scope only; known noncash distributions elsewhere stay unresolved.",
                "No adjustment ratio or missing cash amount was inferred from a price ratio.",
            ],
        ),
    )
    print(results)
