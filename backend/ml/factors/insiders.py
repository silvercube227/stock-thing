"""Point-in-time insider-transaction features (Cohen-Malloy-Pomorski 2012).

Open-market insider BUYING (transaction code P) predicts positive abnormal returns;
selling (S) is noisier (diversification/liquidity driven). All windows join on
`filing_date <= as_of` — never the transaction date — so a filing that reached the
SEC after the as-of date cannot leak (this also handles late / amended filings for
free). Values are pre-signed in the table (+buy / -sell), so a plain window sum is a
net dollar flow.
"""

from __future__ import annotations

import bisect

_BUY_CODE = "P"   # open-market purchase
_SELL_CODE = "S"  # open-market sale


def _insider_context_asof(
    ins_rows: list[dict], as_of_dates: list
) -> dict[str, list[float]]:
    """Per-as-of-date insider features from one ticker's `insider_transactions` rows.

    Returns three aligned lists (0.0 / 0 fallbacks when no filings are visible):
      net_buy_value_6m     — signed P/S dollar flow filed in the last 183 days (raw
                             dollars; assembly divides by market cap)
      insider_buyers_90d   — distinct insiders with an open-market BUY in 90 days
                             (cluster-buy breadth)
      insider_net_ratio_12m— scale-free (buy−sell)/(buy+sell) dollar value over 365d
    """
    from backend.ml.dataset import _as_date

    keys = ("net_buy_value_6m", "insider_buyers_90d", "insider_net_ratio_12m")
    out: dict[str, list[float]] = {k: [] for k in keys}
    if not ins_rows:
        for _ in as_of_dates:
            out["net_buy_value_6m"].append(0.0)
            out["insider_buyers_90d"].append(0.0)
            out["insider_net_ratio_12m"].append(0.0)
        return out

    rows = sorted(ins_rows, key=lambda r: _as_date(r["filing_date"]))
    fdates = [_as_date(r["filing_date"]) for r in rows]
    codes = [(r.get("transaction_code") or "") for r in rows]
    values = [r.get("value") for r in rows]
    ciks = [r.get("insider_cik") for r in rows]

    from datetime import timedelta

    for d in as_of_dates:
        d = _as_date(d)
        hi = bisect.bisect_right(fdates, d)

        lo6 = bisect.bisect_left(fdates, d - timedelta(days=183))
        net6 = 0.0
        for i in range(lo6, hi):
            if codes[i] in (_BUY_CODE, _SELL_CODE) and values[i] is not None:
                net6 += float(values[i])
        out["net_buy_value_6m"].append(net6)

        lo90 = bisect.bisect_left(fdates, d - timedelta(days=90))
        buyers = {ciks[i] for i in range(lo90, hi)
                  if codes[i] == _BUY_CODE and ciks[i]}
        out["insider_buyers_90d"].append(float(len(buyers)))

        lo12 = bisect.bisect_left(fdates, d - timedelta(days=365))
        buy_val = sell_val = 0.0
        for i in range(lo12, hi):
            if values[i] is None:
                continue
            if codes[i] == _BUY_CODE:
                buy_val += float(values[i])          # +signed
            elif codes[i] == _SELL_CODE:
                sell_val += -float(values[i])         # magnitude of the -signed sale
        denom = buy_val + sell_val
        out["insider_net_ratio_12m"].append(
            (buy_val - sell_val) / denom if denom > 0 else 0.0
        )
    return out
