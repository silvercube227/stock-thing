"""Normalize VERIFIED as-traded LSEG bars and complete corporate actions."""

from __future__ import annotations

import numpy as np


def normalize_vendor_pair(ticker_id, raw_rows, adjusted_rows, cash_dividends, *, embedded_cash=None):
    """Build a research candidate from exact-date nominal and capital-adjusted bars.

    Ordinary cash dividends are applied separately. Coincident vendor adjustment
    steps and cash dividends require review to avoid counting a distribution twice.
    A reviewed embedded_cash entry identifies declared cash already represented
    by the capital-adjusted series. Its factor must reproduce exactly; only the
    remaining ordinary cash is applied separately. This does not certify action
    coverage or derive share ratios from price ratios.
    """
    from datetime import date
    import math

    def indexed(rows):
        out={}
        for row in rows:
            d=row['Date'][:10]
            if d in out:raise ValueError('duplicate vendor date')
            out[d]=row
        return out

    raw,adjusted=indexed(raw_rows),indexed(adjusted_rows)
    embedded_cash = embedded_cash or {}
    if raw.keys()!=adjusted.keys():raise ValueError('raw/adjusted date mismatch')
    if set(cash_dividends)-raw.keys():raise ValueError('cash event missing price date')
    if set(embedded_cash)-set(cash_dividends):raise ValueError('embedded cash missing declared dividend')
    prices=[]
    scales=[]
    capital_closes=[]
    remaining_cash=[]
    for d in sorted(raw):
        nominal=float(raw[d]['TRDPRC_1'])
        capital=float(adjusted[d]['TRDPRC_1'])
        if not all(math.isfinite(x) and x>0 for x in (nominal,capital)):
            raise ValueError('invalid vendor close')
        scale=capital/nominal
        step=scale/scales[-1] if scales else 1.0
        if math.isclose(step,1.0,rel_tol=1e-5):step=1.0
        dividend=float(cash_dividends.get(d,0))
        if not math.isfinite(dividend) or dividend<0:raise ValueError('invalid dividend')
        included = 0.0
        if d in embedded_cash:
            review = embedded_cash[d]
            included = float(review['amount'])
            if (not review.get('source') or not math.isfinite(included) or included <= 0
                    or included > dividend or not prices or step == 1):
                raise ValueError('invalid embedded cash review')
            prior = prices[-1]['close']
            if included >= prior or not math.isclose(step, prior/(prior-included), rel_tol=1e-5):
                raise ValueError('embedded cash does not explain vendor adjustment')
        if dividend and step!=1 and not included:
            raise ValueError(f'cash dividend coincides with vendor adjustment: {d}')
        row=dict(ticker_id=ticker_id,trade_date=date.fromisoformat(d),close=nominal,
                 adj_close=capital,dividend=dividend,split_factor=step,
                 share_split_factor=None,share_action_source=None,
                 source='lseg',price_basis='as_traded')
        for field,key in [('open','OPEN_PRC'),('high','HIGH_1'),('low','LOW_1'),('volume','ACVOL_UNS')]:
            try:v=float(raw[d][key])
            except (KeyError,ValueError,TypeError):v=float('nan')
            row[field]=int(v) if field=='volume' and math.isfinite(v) and v>=0 else (
                v if field!='volume' and math.isfinite(v) and v>0 else None)
        prices.append(row)
        scales.append(scale)
        capital_closes.append(capital)
        remaining_cash.append(dividend-included)
    factor=1.0
    for i in range(len(prices)-1,-1,-1):
        prices[i]['adj_close']=capital_closes[i]*factor
        div=remaining_cash[i]*scales[i]
        if div:
            if i==0 or div>=capital_closes[i-1]:raise ValueError('unresolved dividend basis')
            factor*=1-div/capital_closes[i-1]
    return prices


def to_price_rows(ticker_id, bars, dividends, splits, *, verified=False):
    """Return dictionaries in Yahoo-compatible split/dividend adjustment convention.

    bars: canonical trade_date/open/high/low/close/volume from a verified field map.
    Actions use effective/ex dates and new-shares/old-shares split ratios.
    A caller must verify full action coverage before supplying verified=True.
    """
    if not verified:
        raise ValueError("LSEG raw-price/action conventions have not been verified")
    bars = sorted(bars, key=lambda r: r["trade_date"])
    if len({r["trade_date"] for r in bars}) != len(bars):
        raise ValueError("duplicate price date")
    if any(not np.isfinite(v) or v <= 0 for v in splits.values()):
        raise ValueError("invalid split ratio")
    if any(not np.isfinite(v) or v < 0 for v in dividends.values()):
        raise ValueError("invalid cash dividend")
    dates = {r["trade_date"] for r in bars}
    if (set(dividends) | set(splits)) - dates:
        raise ValueError("corporate action without a price bar")
    suffix, out = 1.0, []
    for b in reversed(bars):
        d = b["trade_date"]
        close = b.get("close")
        if close is None or not np.isfinite(close) or close <= 0:
            raise ValueError("invalid as-traded close")
        row = dict(
            ticker_id=ticker_id,
            trade_date=d,
            source="lseg",
            price_basis="split_adjusted",
            split_factor=float(splits.get(d, 1.0)),
            share_split_factor=float(splits.get(d, 1.0)),
            share_action_source='verified_lseg_split_history',
            dividend=float(dividends.get(d, 0.0)) / suffix,
        )
        for field in ("open", "high", "low", "close"):
            v = b.get(field)
            row[field] = float(v) / suffix if v is not None and np.isfinite(v) else None
        volume = b.get("volume")
        row["volume"] = (
            int(round(volume * suffix)) if volume is not None and np.isfinite(volume) else None
        )
        out.append(row)
        suffix *= row["split_factor"]
    out.reverse()
    factor = 1.0
    for i in range(len(out) - 1, -1, -1):
        out[i]["adj_close"] = out[i]["close"] * factor
        if out[i]["dividend"]:
            if not i or out[i]["dividend"] >= out[i - 1]["close"]:
                raise ValueError("unresolved dividend adjustment")
            factor *= 1 - out[i]["dividend"] / out[i - 1]["close"]
    return out


async def ingest_verified_prices(conn, rows):
    if not rows:
        return 0
    ids = sorted({r["ticker_id"] for r in rows})
    if await conn.fetchval(
        "select exists(select 1 from price_history where ticker_id=any($1::bigint[]) and source<>'lseg')",
        ids,
    ):
        raise ValueError("refusing mixed-source security history")
    columns = (
        "ticker_id",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "split_factor",
        "dividend",
        "source",
        "price_basis",
        "share_split_factor",
        "share_action_source",
    )
    sql = (
        f"insert into price_history ({','.join(columns)}) values "
        f"({','.join('$' + str(i + 1) for i in range(len(columns)))}) "
        "on conflict (ticker_id,trade_date) do nothing"
    )
    await conn.executemany(sql, [tuple(r.get(c) for c in columns) for r in rows])
    return len(rows)
