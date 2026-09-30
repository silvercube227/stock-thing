"""Separate source-component parity from a vendor's derived adjusted-price ratio."""

import argparse
import gzip
import hashlib
import json
import math
import pickle
from datetime import date
from pathlib import Path

from backend.ml.factors.pit import as_traded_closes
from backend.ml.research import load_snapshot, write_json_new


def component_parity(native, comparison, tolerance=1e-6):
    """No-capital-action histories only; never explain a share action with cash."""
    dates = [p["trade_date"] for p in native]
    if dates != sorted(set(dates)) or not dates:
        raise ValueError("Native dates must be nonempty, sorted and unique")
    other_dates = [p["trade_date"] for p in comparison]
    if other_dates != sorted(set(other_dates)):
        raise ValueError("Comparison dates must be sorted and unique")
    other = {
        p["trade_date"]: (p, nominal)
        for p, nominal in zip(comparison, as_traded_closes(comparison), strict=True)
    }
    result = dict(
        rows=len(native),
        nominal_errors=[],
        cash_differences=[],
        native_formula_errors=[],
        vendor_derived_residuals=[],
        vendor_raw_ratio_differences=[],
    )
    for i, p in enumerate(native):
        d = p["trade_date"]
        if d not in other:
            raise ValueError("Comparison is missing a native observation")
        q, nominal = other[d]
        if p.get("price_basis") != "as_traded" or any(
            float(r["split_factor"]) != 1 for r in (p, q)
        ):
            raise ValueError("Component audit requires explicit nominal basis and no capital steps")
        values = [float(p["close"]), nominal, float(p["adj_close"]), float(q["adj_close"])]
        cash = [float(p["dividend"]), float(q["dividend"])]
        if any(not math.isfinite(v) or v <= 0 for v in values) or any(
            not math.isfinite(v) or v < 0 for v in cash
        ):
            raise ValueError("Invalid price or distribution input")
        nominal_error = abs(float(p["close"]) / nominal - 1)
        if nominal_error > tolerance:
            result["nominal_errors"].append(dict(date=d, relative_error=nominal_error))
        if not math.isclose(cash[0], cash[1], rel_tol=0, abs_tol=1e-6):
            result["cash_differences"].append(dict(date=d, native=cash[0], comparison=cash[1]))
        if not i:
            continue
        prev = native[i - 1]
        old_prev, nominal_prev = other[prev["trade_date"]]
        if cash[0] >= float(prev["close"]) or cash[1] >= nominal_prev:
            raise ValueError("Cash distribution exhausts the prior close")
        native_expected = float(p["close"]) / (float(prev["close"]) - cash[0])
        vendor_expected = nominal / (nominal_prev - cash[1])
        native_observed = float(p["adj_close"]) / float(prev["adj_close"])
        vendor_observed = float(q["adj_close"]) / float(old_prev["adj_close"])
        native_error = abs(native_observed / native_expected - 1)
        if native_error > 1e-12:
            result["native_formula_errors"].append(dict(date=d, relative_error=native_error))
        # Uses the comparison feed's OWN closes/dividends. This is diagnostic,
        # not permission to relabel missing distributions as numerical noise.
        residual = abs(vendor_observed / vendor_expected - 1)
        if residual > tolerance:
            result["vendor_derived_residuals"].append(
                dict(
                    date=d,
                    relative_error=residual,
                    expected_from_own_components=vendor_expected,
                    observed_adjusted_ratio=vendor_observed,
                )
            )
        ratio_error = abs(native_observed / vendor_observed - 1)
        if ratio_error > tolerance:
            result["vendor_raw_ratio_differences"].append(dict(date=d, relative_error=ratio_error))
    return result


def run(snapshot, output):
    inputs, manifest = load_snapshot(snapshot)
    legacy = next(f for f in inputs["frames"] if f.ticker_id == 360).prices
    root = Path(".research/praxair-normalized-candidate")
    report = json.loads((root / "report.json").read_text())
    raw = gzip.decompress((root / "prices.pkl.gz").read_bytes())
    if hashlib.sha256(raw).hexdigest() != report["input_sha256"]:
        raise ValueError("Native candidate changed")
    native = pickle.loads(raw)
    audit = component_parity(native, legacy)
    review_path = Path(".research/verification/praxair-dividend-source-archived.json")
    review_raw = review_path.read_bytes()
    review = json.loads(review_raw)
    if len(review["events"]) != 1:
        raise ValueError("Unexpected dividend source review")
    source = review["events"][0]
    if (
        hashlib.sha256(Path(source["archived_source"]).read_bytes()).hexdigest()
        != source["source_sha256"]
    ):
        raise ValueError("Issuer dividend source changed")
    expected_dates = {date.fromisoformat(d) for d in source["ex_dates_agree_between_price_feeds"]}
    differences = audit["cash_differences"]
    if {r["date"] for r in differences} != expected_dates or any(
        r["native"] != float(source["cash_per_quarter"])
        or r["comparison"] != float(source["legacy_amount"])
        for r in differences
    ):
        raise ValueError("Cash discrepancies differ from individually reviewed issuer amounts")
    audit.update(
        status="component_audit_not_global_certification",
        parent_input_sha256=manifest["input_sha256"],
        native_input_sha256=report["input_sha256"],
        dividend_review_sha256=hashlib.sha256(review_raw).hexdigest(),
        nominal_and_cash_components_supported=not audit["nominal_errors"],
        native_formula_verified=not audit["native_formula_errors"],
        tolerance_unchanged=1e-6,
        database_writes=0,
        conclusion="The residual test uses Yahoo own nominal closes and recorded dividends. "
        "A residual here belongs to its derived adjusted series, not a missing native cash event. "
        "This does not establish completeness of every source action or certify the successor.",
    )
    write_json_new(output, audit)
    print(
        {
            k: v
            for k, v in audit.items()
            if k not in ("vendor_raw_ratio_differences", "cash_differences")
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run(a.snapshot, a.output)
