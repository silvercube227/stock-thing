"""Original SEC flow spans for the registered accounting pack."""

from datetime import date

CONCEPTS = {
    "assets": ("Assets",),
    "net_income": ("NetIncomeLoss",),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "gross_profit": ("GrossProfit",),
}


def parse_accounting_facts(payload, ticker_id):
    out = {}
    facts = payload.get("facts", {}).get("us-gaap", {})
    for metric, concepts in CONCEPTS.items():
        for concept in concepts:
            for r in facts.get(concept, {}).get("units", {}).get("USD", []):
                if r.get("form") not in ("10-K", "10-Q"):
                    continue
                if not all(r.get(k) for k in ("accn", "filed", "end")) or r.get("val") is None:
                    continue
                if r["end"] > r["filed"]:
                    continue
                if metric != "assets" and not r.get("start"):
                    continue
                start = date.fromisoformat(r["start"]) if r.get("start") else None
                end, filed = date.fromisoformat(r["end"]), date.fromisoformat(r["filed"])
                if start and not 70 <= (end - start).days <= 380:
                    continue
                key = (r["accn"], metric, start, end)
                record = dict(
                    ticker_id=ticker_id,
                    accession_number=r["accn"],
                    filed_at=filed,
                    period_start=start,
                    period_end=end,
                    metric=metric,
                    value=float(r["val"]),
                    source_concept=concept,
                )
                if key in out and out[key]["value"] != record["value"]:
                    raise ValueError("ambiguous accounting fact within original accession")
                out[key] = record
    return list(out.values())
