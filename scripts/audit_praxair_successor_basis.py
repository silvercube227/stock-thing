"""Audit every native successor session used by the Praxair terminal candidate."""

import argparse
import gzip
import hashlib
import json
import pickle
from datetime import date
from pathlib import Path

from backend.ingestion.calendar import trading_days_between
from backend.ingestion.dividend_review import review_dividends
from backend.ingestion.prices_lseg import normalize_vendor_pair
from backend.ml.research import load_snapshot, write_json_new
from backend.ml.terminal import successor_horizon_values
from scripts.audit_praxair_price_basis import component_parity
from scripts.audit_vendor_adjustments import changes, load_rows


def run(snapshot, output):
    inputs, manifest = load_snapshot(snapshot)
    frames = {f.ticker_id: f for f in inputs["frames"]}
    target, successor = frames[947], frames[360]
    hashes = {}

    def read(path):
        raw = Path(path).read_bytes()
        hashes[path] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    base = ".research/verification/praxair-successor-"
    raw, adjusted = [
        load_rows(read(base + kind + "-prices.json")["securities"][0])
        for kind in ("native", "adjusted")
    ]
    for payload in (raw, adjusted):
        request = payload["request"]
        if (
            request["ric"] != "LIN.OQ"
            or request["start"] != "2018-10-31"
            or request["end"] != "2019-11-01"
        ):
            raise ValueError("Unexpected successor source window or identity")
    if raw["request"]["adjustments"] != "unadjusted" or adjusted["request"]["adjustments"] != [
        "CCH",
        "CRE",
        "RPO",
        "RTS",
    ]:
        raise ValueError("Unexpected successor price convention")
    capital = changes(raw["records"], adjusted["records"])
    dividends = read(base + "dividends.json")
    if dividends["requested_rics"] != ["LIN.OQ"] or dividends["parameters"] != {
        "SDate": "2018-10-31",
        "EDate": "2019-11-01",
        "DateType": "XD",
    }:
        raise ValueError("Dividend query scope changed")
    cash = review_dividends(dividends["records"], [r["Date"][:10] for r in raw["records"]])
    if cash["issues"] or cash["empty_rows"] or capital["events"] or cash["outside_price_history"]:
        raise ValueError("Unresolved successor distributions or capital adjustments")
    actions = read(base + "actions.json")["sources"]["corporate_actions"]
    if (
        actions["failures"]
        or actions["requested_rics"] != ["LIN.OQ"]
        or actions["window"] != ["2018-10-31", "2019-11-01"]
    ):
        raise ValueError("Incomplete corporate-action request")
    reviewed_actions = []
    for action in actions["records"]:
        kind = action["Corporate Change Event Type"]
        if (
            action["Instrument"] != "LIN.OQ"
            or action["Capital Change Is Rescinded"] != "False"
            or float(action["Adjustment Factor"]) != 1
        ):
            raise ValueError("Unreviewed successor capital action")
        if kind == "Exchange Offer":
            if (
                action["Capital Change Effective Date"] != "2018-10-31"
                or float(action["Terms New Shares"]) != 1
                or float(action["Terms Old Shares"]) != 1
                or action["Corporate Action Description"]
                != (
                    "One Ordinary Share of Linde plc (New), "
                    "for each Praxair, Inc. Common Stock held."
                )
            ):
                raise ValueError("Exchange differs from the documented ownership transition")
        elif kind != "Buyback":
            raise ValueError("Unknown successor distribution type")
        reviewed_actions.append(
            dict(
                type=kind,
                effective_date=action["Capital Change Effective Date"],
                treatment="ownership_boundary"
                if kind == "Exchange Offer"
                else "issuer_buyback_not_holder_distribution",
            )
        )
    if sorted(r["type"] for r in reviewed_actions) != ["Buyback", "Exchange Offer"]:
        raise ValueError("Unexpected corporate-action inventory")
    native = normalize_vendor_pair(
        360, raw["records"], adjusted["records"], cash["cash_by_ex_date"]
    )
    expected = list(trading_days_between(date(2018, 10, 31), date(2019, 11, 1)))
    if [p["trade_date"] for p in native] != expected:
        raise ValueError("Missing exact successor sessions")
    parity = component_parity(native, successor.prices)
    if parity["nominal_errors"] or parity["cash_differences"] or parity["native_formula_errors"]:
        raise ValueError("Successor source components or normalization disagree")
    old_values = target.security_events[0]["horizon_values"]
    if isinstance(old_values, str):
        old_values = json.loads(old_values)
    valuation = successor_horizon_values(
        target_prices=target.prices,
        successor_prices=native,
        target_last_trade=date(2018, 10, 30),
        ownership_close=date(2018, 10, 31),
        horizon_dates=old_values.keys(),
        cash_per_share=0,
        exchange_ratio=1,
    )
    if valuation["missing_horizons"] or len(valuation["horizon_values"]) != 252:
        raise ValueError("Incomplete native successor valuation")
    differences = {
        d: abs(value / float(old_values[d]) - 1) for d, value in valuation["horizon_values"].items()
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    payload = pickle.dumps(native, protocol=5)
    with gzip.open(output / "native-prices.pkl.gz", "wb") as f:
        f.write(payload)
    report = dict(
        status="successor_component_review_complete_reference_application_pending",
        input_sha256=manifest["input_sha256"],
        native_prices_sha256=hashlib.sha256(payload).hexdigest(),
        source_hashes=hashes,
        reviewed_actions=reviewed_actions,
        dividend_review=cash,
        parity=parity,
        valuation=valuation,
        compared_horizons=len(differences),
        max_horizon_relative_difference=max(differences.values()),
        database_writes=0,
        basis_policy="Use native nominal prices and corroborated cash events "
        "to reconstruct returns; "
        "Yahoo adjusted prices are an independent diagnostic, not the source of terminal values.",
    )
    write_json_new(output / "report.json", report)
    print(
        dict(
            rows=len(native),
            cash_dates=len(cash["cash_by_ex_date"]),
            compared_horizons=len(differences),
            max_horizon_relative_difference=max(differences.values()),
            vendor_derived_residuals=parity["vendor_derived_residuals"],
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run(a.snapshot, a.output)
