"""Value documented stock/cash consideration on the target's adjusted-price basis.

Callers must verify security identities, the target's final trading session, and
the first successor close at which the shares are owned. This module never resolves
symbols or election/proration terms. The stock leg follows the adjusted total-return
series; merger cash stays cash without interest or reinvestment.
"""
import math

from backend.ml.factors.pit import as_traded_closes, day


def successor_horizon_values(*, target_prices, successor_prices, target_last_trade,
                             ownership_close, horizon_dates, cash_per_share,
                             exchange_ratio):
    cash, ratio = float(cash_per_share), float(exchange_ratio)
    if not all(math.isfinite(x) and x >= 0 for x in (cash, ratio)) or ratio == 0:
        raise ValueError('Stock consideration needs finite nonnegative cash and positive shares')
    final, anchor = day(target_last_trade), day(ownership_close)
    if final > anchor:
        raise ValueError('Target final trade follows successor ownership close')

    def index_prices(rows):
        dates = [day(r['trade_date']) for r in rows]
        if dates != sorted(set(dates)):
            raise ValueError('Price dates must be sorted and unique')
        return {d: i for i, d in enumerate(dates)}

    target_index, successor_index = index_prices(target_prices), index_prices(successor_prices)
    if not target_prices or day(target_prices[-1]['trade_date']) != final:
        raise ValueError('Target history must end at its reviewed final trading session')
    if anchor not in successor_index:
        raise ValueError('Missing exact successor ownership close; no nearest-date substitution')
    ti, si = target_index[final], successor_index[anchor]
    target_raw = as_traded_closes(target_prices)[ti]
    successor_raw = as_traded_closes(successor_prices)[si]
    target_adj = target_prices[ti].get('adj_close')
    successor_adj = successor_prices[si].get('adj_close')
    bases = (target_raw, successor_raw, target_adj, successor_adj)
    if any(x is None or not math.isfinite(float(x)) or float(x) <= 0 for x in bases):
        raise ValueError('Unverified or invalid price adjustment basis')
    target_scale = float(target_adj) / target_raw
    stock_at_anchor = ratio * successor_raw
    values, missing = {}, []
    for end in sorted({day(d) for d in horizon_dates}):
        if end < anchor:
            raise ValueError('Horizon precedes successor ownership close')
        idx = successor_index.get(end)
        adj = None if idx is None else successor_prices[idx].get('adj_close')
        if adj is None or not math.isfinite(float(adj)) or float(adj) <= 0:
            missing.append(end.isoformat())
            continue
        stock_value = stock_at_anchor * float(adj) / float(successor_adj)
        values[end.isoformat()] = target_scale * (cash + stock_value)
    return dict(horizon_values=values, missing_horizons=missing,
                target_adjusted_units_per_nominal=target_scale,
                successor_nominal_price_at_ownership=successor_raw,
                stock_distribution_policy='successor_adjusted_total_return',
                merger_cash_policy='held_without_reinvestment')
