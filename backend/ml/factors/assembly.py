"""Per-ticker row assembly: stitch the factor builders into feature+target rows."""

from __future__ import annotations

import bisect
import math

import numpy as np

from backend.ingestion.calendar import HORIZON_TRADING_DAYS
from backend.ml.dataset import TickerFrame, _as_date, compute_targets
from backend.ml.factors.constants import EARNINGS_REACTION_FEATURES, FUNDAMENTAL_FEATURES
from backend.ml.factors.estimates import _earnings_reaction_asof, _estimates_context_asof
from backend.ml.factors.fundamentals import _fundamental_context_asof
from backend.ml.factors.price import _price_features, _short_interest_asof
from backend.ml.factors.util import _safe_ratio
from backend.ml.features import (
    SEQUENCE_LENGTH,
    _build_fundamental_series,
    _build_sentiment_series,
    _fund_filing_mask,
)
from backend.ml.model import HORIZONS


def build_universe_return_map(frames: list[TickerFrame]) -> dict:
    """Equal-weight universe daily log return by trade date."""
    by_date: dict = {}
    for frame in frames:
        prices = sorted(frame.prices, key=lambda r: _as_date(r["trade_date"]))
        prev: float | None = None
        for row in prices:
            cur = float(row["adj_close"]) if row["adj_close"] is not None else None
            if prev is not None and cur is not None and prev > 0 and cur > 0:
                d = _as_date(row["trade_date"])
                by_date.setdefault(d, []).append(math.log(cur / prev))
            prev = cur
    return {d: float(np.mean(vals)) for d, vals in by_date.items() if vals}


def build_market_horizon_returns(
    market_returns: dict,
    grid: list,
    horizons: tuple[str, ...] = HORIZONS,
) -> dict[str, dict]:
    """For each (horizon, grid_date), the universe log return over the next H
    trading days starting at the first trade_date on or after grid_date.

    Used by the `beta_resid` target (`y_h = r_h - beta_252 * market_r_h`) — the
    market-return leg must be sampled on the same calendar window the ticker's
    horizon return spans. Cumsum over the sorted daily series and slice by
    bisect index, so this is O(len(grid)) per horizon after a one-time sort.

    Returns: `{horizon: {grid_date: float}}`. Grid dates whose forward window
    runs past the last trade_date are dropped (NaN downstream where used).
    """
    out: dict[str, dict] = {h: {} for h in horizons}
    if not market_returns:
        return out

    sorted_dates = sorted(market_returns.keys())
    daily = np.asarray([float(market_returns[d]) for d in sorted_dates], dtype=float)
    # cumsum[i] = sum of daily[0..i-1]; cumsum[end] - cumsum[start] = window log return.
    cumsum = np.concatenate(([0.0], np.cumsum(daily)))
    n = len(sorted_dates)

    for h in horizons:
        H = HORIZON_TRADING_DAYS[h]
        bucket = out[h]
        for g in grid:
            # First trade_date on or after g — beta_resid is computed forward from
            # the grid date, just like the ticker's forward return.
            start_idx = bisect.bisect_left(sorted_dates, g)
            end_idx = start_idx + H
            if end_idx >= n + 1:
                continue
            bucket[g] = float(cumsum[end_idx] - cumsum[start_idx])
    return out


def build_ticker_rows(
    frame: TickerFrame,
    grid: list,
    max_stale_days: int = 7,
    market_returns: dict | None = None,
) -> list[dict]:
    """One feature+target row per grid date for a single ticker (raw, pre-demean).

    Mirrors dataset.assemble_ticker_samples_aligned: use the ticker's last bar at
    or before each grid date, set the row's date to the grid date so all tickers
    on the same month-end share a cross-section. Forward returns/masks come from
    the shared `compute_targets` (same label definition as the transformer).

    `max_stale_days` prevents delisted/paused tickers from being repeated forever
    on later month-end grid dates after their final available bar.
    """
    prices = sorted(frame.prices, key=lambda r: _as_date(r["trade_date"]))
    if len(prices) <= SEQUENCE_LENGTH:
        return []
    trade_dates = [_as_date(r["trade_date"]) for r in prices]
    adj_close = [float(r["adj_close"]) if r["adj_close"] is not None else None for r in prices]
    volume = [float(r.get("volume") or 0.0) for r in prices]
    shares = frame.shares_outstanding

    entries = []  # (grid_date, pos, bar_date)
    for g in grid:
        pos = bisect.bisect_right(trade_dates, g) - 1
        if pos < SEQUENCE_LENGTH:
            continue
        if (g - trade_dates[pos]).days > max_stale_days:
            continue
        if adj_close[pos] is None or adj_close[pos] <= 0:
            continue
        entries.append((g, pos, trade_dates[pos]))
    if not entries:
        return []

    bar_dates = [e[2] for e in entries]
    bar_positions = [e[1] for e in entries]
    fund = _build_fundamental_series(bar_dates, frame.fundamentals)  # (k, 5)
    fund_avail = _fund_filing_mask(bar_dates, frame.fundamentals)    # (k,) bool
    sent = _build_sentiment_series(bar_dates, frame.sentiment)       # (k, 2)
    fund_ctx = _fundamental_context_asof(frame.fundamentals, bar_dates)
    reaction = _earnings_reaction_asof(
        frame.fundamentals,
        bar_positions,
        bar_dates,
        trade_dates,
        adj_close,
        market_returns,
    )
    est_ctx = _estimates_context_asof(frame.estimates or [], frame.surprises or [], bar_dates)
    si_ctx = _short_interest_asof(frame.short_interest or [], bar_dates)

    rows: list[dict] = []
    for j, (g, pos, _bd) in enumerate(entries):
        feats = _price_features(
            adj_close,
            volume,
            trade_dates,
            pos,
            shares_outstanding=shares,
            market_returns=market_returns,
        )
        for i, name in enumerate(FUNDAMENTAL_FEATURES):
            feats[name] = float(fund[j, i])
        feats["fund_available"] = 1.0 if fund_avail[j] else 0.0
        # Keep experimental factors on the row for quick ablations, but don't
        # feed them into the default production baseline unless they win.
        price = adj_close[pos]
        market_cap = (
            float(price * shares)
            if price is not None and shares is not None and price > 0 and shares > 0
            else 0.0
        )
        feats["earnings_yield"] = _safe_ratio(fund_ctx["ttm_net_income"][j], market_cap)
        feats["book_to_market"] = _safe_ratio(fund_ctx["total_equity"][j], market_cap)
        feats["sales_to_price"] = _safe_ratio(fund_ctx["ttm_revenue"][j], market_cap)
        feats["fcf_yield"] = _safe_ratio(fund_ctx["ttm_fcf"][j], market_cap)
        feats["roe_ttm"] = _safe_ratio(fund_ctx["ttm_net_income"][j], fund_ctx["total_equity"][j])
        feats["net_margin_ttm"] = _safe_ratio(fund_ctx["ttm_net_income"][j], fund_ctx["ttm_revenue"][j])
        feats["fcf_margin_ttm"] = _safe_ratio(fund_ctx["ttm_fcf"][j], fund_ctx["ttm_revenue"][j])
        feats["gross_margin_stability_4q"] = fund_ctx["gross_margin_stability_4q"][j]
        feats["operating_margin_stability_4q"] = fund_ctx["operating_margin_stability_4q"][j]
        feats["revenue_growth_stability_4q"] = fund_ctx["revenue_growth_stability_4q"][j]
        # Earnings-reaction features are precomputed once per ticker above.
        for name in EARNINGS_REACTION_FEATURES:
            feats[name] = reaction[name][j]
        # LSEG analyst-estimate features (precomputed per ticker in est_ctx).
        feats["rec_mean_level"] = est_ctx["rec_mean_level"][j]
        feats["rec_rev_30d"] = est_ctx["rec_rev_30d"][j]
        feats["rec_rev_90d"] = est_ctx["rec_rev_90d"][j]
        feats["price_target_rev_90d"] = est_ctx["price_target_rev_90d"][j]
        feats["forward_earnings_yield"] = est_ctx["forward_earnings_yield"][j]
        feats["forward_ebitda_yield"] = est_ctx["forward_ebitda_yield"][j]
        feats["revenue_surprise"] = est_ctx["revenue_surprise"][j]
        feats["eps_surprise"] = est_ctx["eps_surprise"][j]
        feats["eps_est_rev_30d"] = est_ctx["eps_est_rev_30d"][j]
        feats["eps_est_rev_90d"] = est_ctx["eps_est_rev_90d"][j]
        feats["coverage_chg_90d"] = est_ctx["coverage_chg_90d"][j]
        feats["pt_num_estimates"] = est_ctx["pt_num_estimates"][j]
        pt = est_ctx["price_target_mean"][j]
        feats["price_target_upside"] = ((pt - price) / price) if price and price > 0 and pt else 0.0
        # Panel-level demean overwrites this in `prepare_panel`; until then leave
        # it equal to mom_12_1 so single-ticker callers see a finite value.
        feats["industry_neutral_mom_12_1"] = feats["mom_12_1"]
        feats["sentiment_7d"] = float(sent[j, 0])
        feats["sentiment_14d"] = float(sent[j, 1])
        feats["eps_dispersion"] = est_ctx["eps_dispersion"][j]
        feats["short_ratio"] = si_ctx["short_ratio"][j]
        _labels, returns, mask = compute_targets(adj_close, pos)
        row = {
            "date": g,
            "ticker_id": frame.ticker_id,
            "sector": frame.sector,
            "industry": frame.industry,
            **feats,
        }
        for h in HORIZONS:
            row[f"r_{h}"] = returns[h]
            row[f"mask_{h}"] = mask[h]
        rows.append(row)
    return rows
