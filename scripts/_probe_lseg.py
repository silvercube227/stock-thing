"""Phase-0 spike: does LSEG give us point-in-time HISTORY for the estimate fields?

This gates the whole analyst-estimate feature plan. The model trains on a multi-year
walk-forward panel, so each estimate (recommendation consensus, price target, revenue
estimate/actual, forward P/E, forward EV/EBITDA) is only usable as a TRAINING feature
if LSEG returns it as a dated time series reaching back years — not just today's value.

This script opens a Workspace desktop session and, for a few tickers, pulls each
candidate field as monthly history over ~10 years, then reports per group:
  earliest date, latest date, non-null point count  ->  TRAINABLE / THIN / UNAVAILABLE

Run with Workspace RUNNING and LSEG_APP_KEY set in .env:
    python -m scripts._probe_lseg

Read-only probe. Writes nothing. Field codes below are best-guesses to confirm/adjust;
the point is to learn which codes actually return history on this license.
"""
from __future__ import annotations

import sys
import warnings
from datetime import date

from backend.config import get_settings

# lseg.data emits pandas-2.3 "downcasting" FutureWarnings from inside its own code.
warnings.filterwarnings("ignore", category=FutureWarning)

# RICs for a handful of liquid, well-covered names across sectors.
PROBE_RICS = ["AAPL.O", "MSFT.O", "JPM.N", "XOM.N", "JNJ.N"]
START = "2014-01-01"
END = date.today().isoformat()

# Candidate field codes per feature group. Multiple guesses per group: get_history
# returns a column only for codes the license resolves, so we can see which land.
FIELD_GROUPS: dict[str, list[str]] = {
    "recommendations": [
        "TR.RecMean", "TR.NumberOfRecommendations",
        "TR.RecLabel", "TR.RecEstNumberOfStrongBuy", "TR.RecEstNumberOfBuy",
    ],
    "price_target": [
        "TR.PriceTargetMean", "TR.PriceTargetNumberOfEstimates",
    ],
    "revenue_est_vs_actual": [
        "TR.RevenueMean", "TR.RevenueActValue", "TR.RevenueSurprise",
    ],
    "eps_est_vs_actual": [
        "TR.EPSMean", "TR.EPSActValue", "TR.EPSSurprise",
    ],
    # TR.FwdPE = forward (anchor); TR.EVToEBITDA = trailing (contrast). The rest are
    # forward EV/EBITDA candidates — keep whichever returns a "Forward"/NTM-labeled
    # series. TR.EBITDAMean is forward consensus EBITDA as a compute-it-ourselves fallback.
    "forward_valuation": [
        "TR.FwdPE", "TR.EVToEBITDA", "TR.PE",
        "TR.FwdEVToEBITDA", "TR.EVToEBITDAFwd", "TR.EVToEBITDANTM",
        "TR.EVToEBITDASmartEst", "TR.EBITDAMean",
    ],
}

# A group is "TRAINABLE" only if at least one field has history this far back.
TRAINABLE_BEFORE = date(2018, 1, 1)  # need a few years of test folds after the 36mo embargo


def _import_lseg():
    try:
        import lseg.data as ld  # noqa: PLC0415
        return ld
    except ModuleNotFoundError:
        sys.exit(
            "lseg.data not installed. In the venv:\n"
            "    pip install lseg-data\n"
            "(it's in pyproject.toml's [ingestion] extra)."
        )


def _to_iso(ts) -> str:
    return ts.date().isoformat() if hasattr(ts, "date") else str(ts)


def _field_label(col) -> str:
    """Multi-ticker get_history returns (instrument, field) column tuples; we report
    by field, so collapse to the field part."""
    return str(col[-1]) if isinstance(col, tuple) else str(col)


def _probe_group(ld, name: str, fields: list[str]) -> None:
    print(f"\n--- {name} ---")
    try:
        df = ld.get_history(
            universe=PROBE_RICS, fields=fields,
            start=START, end=END, interval="monthly",
        )
    except Exception as exc:  # noqa: BLE001 — probe: any failure is a data point
        print(f"  get_history FAILED: {type(exc).__name__}: {exc}")
        print("  verdict: UNAVAILABLE (call errored)")
        return

    if df is None or df.empty:
        print("  returned no data")
        print("  verdict: UNAVAILABLE (empty)")
        return

    # Columns are per-ticker; aggregate each field across the probe tickers so we
    # get one history-depth verdict per field rather than one per (ticker, field).
    by_field: dict[str, list] = {}
    for col in df.columns:
        s = df[col].dropna()
        if not s.empty:
            by_field.setdefault(_field_label(col), []).append(s)

    earliest_overall: date | None = None
    for label in sorted(by_field):
        series = by_field[label]
        n = sum(len(s) for s in series)
        lo = date.fromisoformat(_to_iso(min(s.index.min() for s in series)))
        hi = date.fromisoformat(_to_iso(max(s.index.max() for s in series)))
        if earliest_overall is None or lo < earliest_overall:
            earliest_overall = lo
        flag = "  <- reaches training window" if lo <= TRAINABLE_BEFORE else ""
        print(f"  {label:<34} points={n:<5} range={lo}..{hi}{flag}")

    if not by_field:
        verdict = "SNAPSHOT-ONLY or UNAVAILABLE (columns present but no dated points)"
    elif earliest_overall and earliest_overall <= TRAINABLE_BEFORE:
        verdict = f"TRAINABLE (history back to {earliest_overall})"
    else:
        verdict = f"THIN (earliest {earliest_overall} — too short for full walk-forward)"
    print(f"  verdict: {verdict}")


def main() -> None:
    app_key = get_settings().lseg_app_key
    if not app_key:
        sys.exit("LSEG_APP_KEY is not set in .env — generate one in Workspace's App Key Generator.")

    ld = _import_lseg()
    print("Opening LSEG desktop session (Workspace must be running) ...")
    try:
        ld.open_session(app_key=app_key)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Could not open session: {type(exc).__name__}: {exc}\n"
                 "Is Workspace running and logged in on this machine?")

    try:
        print(f"Probing {len(PROBE_RICS)} tickers, monthly history {START}..{END}")
        for name, fields in FIELD_GROUPS.items():
            _probe_group(ld, name, fields)
        print(
            "\nDecision: groups marked TRAINABLE become GBDT training features; "
            "SNAPSHOT-ONLY/THIN ones are deferred to display-only. The earliest "
            "TRAINABLE date sets how far back the panel can use these columns."
        )
    finally:
        ld.close_session()


if __name__ == "__main__":
    main()
