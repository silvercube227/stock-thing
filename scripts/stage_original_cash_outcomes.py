"""Verify original-security cash outcomes against native recovery candidates only."""

import argparse
import gzip
import hashlib
import json
import pickle
from collections import Counter
from datetime import date
from pathlib import Path

from backend.ingestion.calendar import HORIZON_TRADING_DAYS, shift_trading_days
from backend.ml.factors.pit import documented_targets
from backend.ml.research import write_json_new


def run(output):
    hashes = {}

    def read(path):
        blob = Path(path).read_bytes()
        hashes[str(path)] = hashlib.sha256(blob).hexdigest()
        return blob

    archive = json.loads(read(".research/verification/original-cash-sources-2026-09-19-v1.json"))
    review = read("docs/original_cash_outcome_reviews.json")
    if hashlib.sha256(review).hexdigest() != archive["review_sha256"]:
        raise ValueError("Cash review changed after primary source archival")
    identities = json.loads(read(".research/verification/identity-lifecycle-archive.json"))[
        "sources"
    ]["identity_archive"]["records"]
    candidate = Path(".research/original-security-normalized-candidate-v2")
    report = json.loads(read(candidate / "report.json"))
    raw = gzip.decompress(read(candidate / "prices.pkl.gz"))
    if hashlib.sha256(raw).hexdigest() != report["input_sha256"]:
        raise ValueError("Recovery candidate changed")
    for path, expected in report["source_hashes"].items():
        if hashlib.sha256(read(path)).hexdigest() != expected:
            raise ValueError("Recovery candidate source changed")
    prices = pickle.loads(raw)
    outcomes = []
    for row in archive["events"]:
        if hashlib.sha256(read(row["archived_source"])).hexdigest() != row["source_sha256"]:
            raise ValueError("Primary completion source changed")
        exact = [r for r in identities if r.get("RIC") == row["ric"]]
        if (
            len(exact) != 1
            or exact[0]["ISIN"] != row["isin"]
            or int(exact[0]["CIK Number"]) != int(row["cik"])
        ):
            raise ValueError("Exact issuer/security identity mismatch")
        history = prices[row["ric"]]
        final = history[-1]
        if (
            str(final["trade_date"]) != row["last_trade_date"]
            or final["price_basis"] != "as_traded"
            or final["close"] != final["adj_close"]
            or final["ticker_id"] != row["ticker_id"]
        ):
            raise ValueError("Native final price/unit basis mismatch")
        if shift_trading_days(final["trade_date"], 1).isoformat() != row["effective_date"]:
            raise ValueError("Terminal session does not follow documented final session")
        event = dict(
            ticker_id=row["ticker_id"],
            effective_date=date.fromisoformat(row["effective_date"]),
            event_type="acquisition",
            consideration_type="cash",
            proceeds_basis="adj_close",
            cash_value=row["cash_per_share"],
            verified=True,
            source=json.dumps(row, sort_keys=True),
        )
        counts = Counter()
        dates = [r["trade_date"] for r in history]
        for pos in range(max(0, len(history) - 254), len(history) - 1):
            labels = documented_targets(history, pos, [event], date(2020, 1, 1), dates)
            for horizon, label in labels.items():
                if label[4] == "acquisition":
                    if not label[1]:
                        raise ValueError("Documented cash label unavailable")
                    counts[horizon] += 1
        if dict(counts) != HORIZON_TRADING_DAYS:
            raise ValueError("Terminal label census incomplete")
        outcomes.append(
            dict(
                ric=row["ric"],
                event=event,
                final_native_price=final["close"],
                terminal_crossing_candidate_labels=dict(counts),
                source_and_final_unit_basis_verified=True,
                historical_price_completeness_certified=False,
            )
        )
    read(__file__)
    write_json_new(
        output,
        dict(
            status="cash_outcomes_staged_native_histories_unapplied",
            outcomes=outcomes,
            source_hashes=hashes,
            database_writes=0,
            gates_promoted=0,
            pending="Certify and apply recovered price histories with exact "
            "rollback and verify loader price basis before applying outcomes.",
        ),
    )
    print(dict(cash_outcomes=len(outcomes), database_writes=0, output=output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
