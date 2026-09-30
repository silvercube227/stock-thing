"""Archive LSEG corporate actions for an explicit RIC list. Read-only, no DB writes."""
import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from backend.config import get_settings
from backend.ingestion.estimates import _open_session
from backend.ml.research import write_json_new

FIELDS = [
    "TR.CACorpActEventType", "TR.CACorpActDesc", "TR.CAExDate", "TR.CAEffectiveDate",
    "TR.CAAdjustmentFactor", "TR.CAAdjustmentType", "TR.CATermsOldShares",
    "TR.CATermsNewShares", "TR.CAIsRescinded",
]


def run(rics, output, start, end, batch_size):
    ld = _open_session(get_settings().lseg_app_key)
    records, failures = [], []
    try:
        for i in range(0, len(rics), batch_size):
            batch = rics[i:i + batch_size]
            try:
                df = ld.get_data(batch, FIELDS, {"SDate": start, "EDate": end})
                records.extend(df.astype(str).to_dict("records"))
            except Exception as exc:  # noqa: BLE001 - one bad RIC must not lose the pull
                failures.append({"rics": batch, "reason": type(exc).__name__, "detail": str(exc)[:300]})
            print(f"{min(i + batch_size, len(rics))}/{len(rics)} securities queried", flush=True)
    finally:
        ld.close_session()
    write_json_new(output, {"sources": {"corporate_actions": {
        "requested_rics": rics, "records": records, "failures": failures,
        "window": [start, end],
        "status": "partial" if failures else "observed_requires_event_review"}}})
    print({"rics": len(rics), "records": len(records), "failed_batches": len(failures)})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rics", required=True, help="JSON file holding a list of RICs")
    p.add_argument("--output", required=True)
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default="2026-09-07")
    p.add_argument("--batch-size", type=int, default=20)
    a = p.parse_args()
    run(json.loads(Path(a.rics).read_text()), a.output, a.start, a.end, a.batch_size)
