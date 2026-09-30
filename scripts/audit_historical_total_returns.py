"""Corroborate native cash returns; retain gaps and capital events for review."""

import argparse
import gzip
import hashlib
import json
import math
import pickle
from collections import Counter
from datetime import date
from pathlib import Path

from backend.ingestion.calendar import trading_days_between
from backend.ml.research import write_json_new


def compare(ric, prices, records):
    days = [row["trade_date"] for row in prices]
    if not days or days != sorted(set(days)):
        raise ValueError("Native dates must be unique and ordered")
    sessions = trading_days_between(days[0], days[-1])
    positions = {day: i for i, day in enumerate(sessions)}
    if any(day not in positions for day in days):
        raise ValueError("Native prices contain a non-session date")
    native = set(days)
    source, excluded = {}, []
    for row in records:
        if row["Instrument"] != ric:
            raise ValueError("Return security mismatch")
        day = date.fromisoformat(row["Calc Date"])
        if day in source:
            raise ValueError("Duplicate total-return calculation date")
        source[day] = row
        if day not in native:
            excluded.append(
                dict(calc_date=str(day), source_date=row["Date"], reason="no_native_price_not_used")
            )
    missing, stale, invalid, gaps, capital, disagreements = [], [], [], [], [], []
    errors = []
    for i, current in enumerate(prices):
        day = current["trade_date"]
        row = source.get(day)
        if row is None:
            missing.append(str(day))
            continue
        if row["Date"] != str(day):
            stale.append(str(day))
            continue
        if i == 0:
            continue  # No preceding native close; no return is claimed.
        prior = prices[i - 1]
        if positions[day] != positions[prior["trade_date"]] + 1:
            gaps.append(str(day))
            continue  # A multi-session price move cannot test a one-day return.
        if current["split_factor"] != 1:
            capital.append(str(day))
            continue  # Spin-offs and splits need event-specific corroboration.
        try:
            reference = float(row["Daily Total Return"]) / 100
            actual = (current["close"] + current["dividend"]) / prior["close"] - 1
            if not all(math.isfinite(x) for x in (reference, actual)):
                raise ValueError("Nonfinite return")
        except (ValueError, TypeError, ZeroDivisionError):
            invalid.append(str(day))
            continue
        error = abs(actual - reference)
        errors.append(error)
        if error > 1e-6:
            disagreements.append(
                dict(
                    date=str(day),
                    candidate_cash=current["dividend"],
                    candidate_return=actual,
                    source_return=reference,
                    absolute_error=error,
                )
            )
    missing_sessions = sorted(set(sessions) - native)
    flags = dict(
        missing_returns=missing,
        stale_native_dates=stale,
        invalid_returns=invalid,
        gap_crossings_not_compared=gaps,
        capital_dates_not_compared=capital,
        missing_native_sessions=[str(day) for day in missing_sessions],
        disagreements=disagreements,
    )
    if missing or stale or invalid or disagreements:
        status = "source_disagreement_requires_review"
    elif missing_sessions or capital:
        status = "ordinary_cash_returns_match_other_components_require_review"
    else:
        status = "cash_returns_match_source_completeness_not_certified"
    return dict(
        ric=ric,
        status=status,
        native_rows=len(prices),
        compared_returns=len(errors),
        max_absolute_error=max(errors, default=None),
        excluded_source_rows=excluded,
        **flags,
    )


def run(candidate, manifest, output):
    hashes = {}

    def read(path):
        blob = Path(path).read_bytes()
        hashes[str(path)] = hashlib.sha256(blob).hexdigest()
        return blob

    root = Path(candidate)
    report = json.loads(read(root / "report.json"))
    raw = gzip.decompress(read(root / "prices.pkl.gz"))
    if hashlib.sha256(raw).hexdigest() != report["input_sha256"]:
        raise ValueError("Candidate changed")
    prices = pickle.loads(raw)
    entries = json.loads(read(manifest))["entries"]
    if len(entries) != len(prices):
        raise ValueError("Source and candidate scope differ")
    results, seen = [], set()
    for entry in entries:
        request = entry["request"]
        ric = request["universe"]
        if ric in seen or ric not in prices:
            raise ValueError("Duplicate or unexpected source security")
        seen.add(ric)
        rows = prices[ric]
        expected = dict(
            universe=ric,
            fields=["TR.TotalReturn1D", "TR.TotalReturn1D.date", "TR.TotalReturn1D.calcdate"],
            parameters=dict(
                Curn="USD",
                Frq="D",
                SDate=str(rows[0]["trade_date"]),
                EDate=str(rows[-1]["trade_date"]),
            ),
        )
        raw = gzip.decompress(read(entry["artifact"]))
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("Source archive changed")
        payload = json.loads(raw)
        if request != expected or payload["request"] != request:
            raise ValueError("Unexpected return request")
        if len(payload["records"]) != entry["rows"]:
            raise ValueError("Source row count changed")
        results.append(compare(ric, rows, payload["records"]))
    read(__file__)
    receipt = dict(
        status="component_audit_not_source_certification",
        source_hashes=hashes,
        counts=dict(Counter(row["status"] for row in results)),
        securities=results,
        native_rows=sum(row["native_rows"] for row in results),
        database_writes=0,
        gates_promoted=0,
        limitations=[
            "Different LSEG retrieval paths are not independent vendors.",
            "Agreement cannot prove missing-distribution completeness.",
            "No stale values were retimed or forward-filled.",
            "Gaps and capital dates remain explicitly untested.",
        ],
    )
    write_json_new(output, receipt)
    print(receipt["counts"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default=".research/normalized-historical-candidate-v3")
    parser.add_argument(
        "--manifest",
        default=".research/verification/historical-total-return-archive-2026-09-19-v1.json",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.candidate, args.manifest, args.output)
