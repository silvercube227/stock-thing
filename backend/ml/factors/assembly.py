"""Per-ticker row assembly: stitch the factor builders into feature+target rows."""

from __future__ import annotations

import bisect
import math
from datetime import timedelta

import numpy as np

from backend.ingestion.calendar import HORIZON_TRADING_DAYS
from backend.ml.dataset import TickerFrame, _as_date
from backend.ml.factors.constants import (
    EARNINGS_REACTION_FEATURES,
    ESTIMATE_SOURCED_FEATURES,
    FUNDAMENTAL_FEATURES,
    FUNDAMENTAL_SOURCED_FEATURES,
)
from backend.ml.factors.estimates import _earnings_reaction_asof, _estimates_context_asof
from backend.ml.factors.fundamentals import (
    _fundamental_context_asof,
    _shares_outstanding_asof,
)
from backend.ml.factors.insiders import _insider_context_asof
from backend.ml.factors.pit import aligned_shares, as_traded_closes, documented_targets
from backend.ml.factors.long_horizon import (
    accounting_features, fixed_estimate_features, news_features,
)
from backend.ingestion.index_lseg import index_on
from backend.ml.factors.price import (
    _dividend_yield_ttm,
    _price_features,
    _range_vol,
    _seasonality_asof,
    _short_interest_asof,
)
from backend.ml.factors.util import _safe_ratio
from backend.ml.features import (
    SEQUENCE_LENGTH,
    _build_fundamental_series,
    _build_sentiment_series,
    _fund_filing_mask,
)
from backend.ml.model import HORIZONS


def build_universe_return_map(frames: list[TickerFrame], membership_filter: bool = False) -> dict:
    """Equal-weight universe daily log return by trade date."""
    by_date: dict = {}
    for frame in frames:
        prices = sorted(frame.prices, key=lambda r: _as_date(r["trade_date"]))
        prev: float | None = None
        for row in prices:
            cur = float(row["adj_close"]) if row["adj_close"] is not None else None
            if prev is not None and cur is not None and prev > 0 and cur > 0:
                d = _as_date(row["trade_date"])
                if not membership_filter or _in_index_on(getattr(frame, "membership", None), d):
                    by_date.setdefault(d, []).append(math.log(cur / prev))
            prev = cur
    return {d: float(np.mean(vals)) for d, vals in by_date.items() if vals}


def _sector_on(history: list[dict] | None, when, fallback: tuple):
    """Dated labels; legacy fallback only when history was not loaded (None).

    Intervals are [valid_from, valid_to) and non-overlapping (enforced by
    seed_sector_history.py), so at most one matches. Without this every row carried
    TODAY's GICS label — and since that label sets both the training target
    (`sector_return` / `sector_grade`) and the headline metric (`within_sector_ic`),
    pre-2018 rows were demeaned against, and scored inside, peer groups that did not
    exist: Communication Services was created in Sept 2018 and Real Estate in Sept
    2016. Missing intervals and missing dated industries remain unknown.
    """
    if history is None:
        return fallback
    for iv in history:
        vf = _as_date(iv["valid_from"])
        if vf > when:
            break                      # ordered by valid_from; no later one can match
        vt = iv["valid_to"]
        if vt is None or when < _as_date(vt):
            return iv["sector"], iv.get("industry")
    return None, None


def _in_index_on(membership: list[dict] | None, when) -> bool | None:
    """Was this ticker an index member on `when`?

    `valid_to` is exclusive, and None means the interval is still open. Returns
    None when membership was never loaded (a cache predating migration 012), which
    the panel filter must treat as "unknown" rather than "not a member".
    """
    if membership is None:
        return None
    for interval in membership:
        start = _as_date(interval["valid_from"])
        if when < start:
            continue
        end = interval.get("valid_to")
        if end is None or when < _as_date(end):
            return True
    return False


def _market_cap_at(
    raw_close: list[float | None],
    adj_close: list[float | None],
    pos: int,
    pit_shares: float | None,
    static_shares: int | None,
) -> float:
    """As-traded close times dated, split-aligned shares; unknown is NaN.

    Legacy arguments remain for call compatibility but cannot supply a fallback.
    """
    price = raw_close[pos] if pos < len(raw_close) else None
    if price is not None and price > 0 and pit_shares is not None and pit_shares > 0:
        return float(price * pit_shares)

    return float("nan")


def build_market_horizon_returns(
    market_returns: dict,
    grid: list,
    horizons: tuple[str, ...] = HORIZONS,
) -> dict[str, dict]:
    """For each (horizon, grid_date), the universe log return over the H trading
    days the ticker's own horizon return spans.

    Used by the `beta_resid` target (`y_h = r_h - beta_252 * market_r_h`) — the
    market-return leg must be sampled on the same calendar window the ticker's
    horizon return spans. Cumsum over the sorted daily series and slice by
    bisect index, so this is O(len(grid)) per horizon after a one-time sort.

    Index alignment: `market_returns[d]` is the return INTO bar `d`, so a close-to-
    close return from bar `a` to bar `b` is `sum(daily[a+1 .. b])`. `compute_targets`
    enters at the bar AFTER the grid date (one-bar implementation lag) and exits H
    bars later, so the window starts at `idx(grid_date) + 2`.

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
            # Entry is the bar after g; daily[i] is the return into bar i, so the
            # first daily return inside the holding window is two slots past g.
            start_idx = bisect.bisect_right(sorted_dates, g) + 1
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
    allow_provisional_estimates: bool = False,
    allow_provisional_news: bool = False,
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
    # Raw as-traded close, for market cap only. Absent from frames cached before the
    # shares/close backfill, hence `.get`.
    raw_close = as_traded_closes(prices)
    volume = [float(r.get("volume") or 0.0) for r in prices]
    # high/low/dividend have been stored since the first ingest and were never
    # selected until now; `.get` tolerates frames cached before they were added.
    high = [float(r["high"]) if r.get("high") is not None else None for r in prices]
    low = [float(r["low"]) if r.get("low") is not None else None for r in prices]
    dividends = [float(r.get("dividend") or 0.0) for r in prices]
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
    pit_shares = aligned_shares(frame.fundamentals, prices, bar_dates)
    # Same PIT lookup a year earlier — the denominator for net share issuance. Using
    # the as-of helper (rather than differencing filings) means the comparison is
    # "what was on file then" vs "what is on file now", which is what an investor
    # could actually have observed.
    pit_shares_1y = _shares_outstanding_asof(
        frame.fundamentals, [d - timedelta(days=365) for d in bar_dates],
        with_filed_at=True,
    )
    pit_shares_dated = _shares_outstanding_asof(
        frame.fundamentals, bar_dates, with_filed_at=True
    )
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
    season_ctx = _seasonality_asof(adj_close, trade_dates, bar_positions)
    ins_ctx = _insider_context_asof(getattr(frame, "insiders", None) or [], bar_dates)

    sectors = [_sector_on(getattr(frame, "sector_history", None), d,
                         (frame.sector, frame.industry))[0] for d in bar_dates]
    accounting = accounting_features(getattr(frame, "accounting_facts", None) or [], bar_dates, sectors)
    fixed = fixed_estimate_features(
        getattr(frame, "fixed_estimates", None) or [], bar_dates,
        allow_provisional=allow_provisional_estimates,
    )
    news = news_features(getattr(frame, "news", None) or [],
                         getattr(frame, "news_coverage", None) or [], bar_dates,
                         exploratory=allow_provisional_news)

    rows: list[dict] = []
    for j, (g, pos, _bd) in enumerate(entries):
        market_cap = _market_cap_at(raw_close, adj_close, pos, pit_shares[j], shares)
        feats = _price_features(
            adj_close,
            volume,
            trade_dates,
            pos,
            shares_outstanding=shares,
            market_returns=market_returns,
            market_cap=market_cap,
        )
        for i, name in enumerate(FUNDAMENTAL_FEATURES):
            feats[name] = float(fund[j, i])
        feats["fund_available"] = 1.0 if fund_avail[j] else 0.0
        # Keep experimental factors on the row for quick ablations, but don't
        # feed them into the default production baseline unless they win.
        # `market_cap` is the PIT cap computed above and shared with log_market_cap.
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
        # An analyst price target is a NOMINAL as-traded price, so it must be
        # compared against the raw close. Against adj_close the ratio silently
        # embeds every post-date split/dividend adjustment.
        pt_ref = raw_close[pos]
        # Historical PT basis is unverified on the legacy feed. Do not guess.
        pt_verified = any(r.get("price_target_basis") == "as_traded"
                          and _as_date(r["as_of_date"]) <= g for r in frame.estimates or [])
        feats["price_target_upside"] = (
            ((pt - pt_ref) / pt_ref) if pt_verified and pt_ref > 0 and pt else float("nan")
        )
        # Panel-level demean overwrites this in `prepare_panel`; until then leave
        # it equal to mom_12_1 so single-ticker callers see a finite value.
        feats["industry_neutral_mom_12_1"] = feats["mom_12_1"]
        feats["sentiment_7d"] = float(sent[j, 0])
        feats["sentiment_14d"] = float(sent[j, 1])
        feats["eps_dispersion"] = est_ctx["eps_dispersion"][j]
        feats["short_ratio"] = si_ctx["short_ratio"][j]
        feats["seasonal_same_month_5y"] = season_ctx["seasonal_same_month_5y"][j]
        feats["seasonal_other_month_5y"] = season_ctx["seasonal_other_month_5y"][j]
        feats["seasonal_gap_5y"] = season_ctx["seasonal_gap_5y"][j]
        # Insider net-buy dollars scaled by market cap (valuation-feature convention);
        # the breadth count and net ratio are already scale-free.
        feats["insider_net_buy_6m"] = _safe_ratio(ins_ctx["net_buy_value_6m"][j], market_cap)
        feats["insider_buyers_90d"] = ins_ctx["insider_buyers_90d"][j]
        feats["insider_net_ratio_12m"] = ins_ctx["insider_net_ratio_12m"][j]
        # Availability of the LSEG feed, the analyst-side analogue of fund_available.
        # Buybacks read negative, secondary offerings positive. NaN unless BOTH share
        # counts are on file: "no prior filing" is not "no issuance".
        (now_sh, now_filed), (prior_sh, prior_filed) = pit_shares_dated[j], pit_shares_1y[j]
        feats["net_issuance"] = (
            (now_sh / prior_sh - 1.0)
            if now_sh and prior_sh and now_sh > 0 and prior_sh > 0
            # Both lookups landing on the SAME filing is not an observation of a year
            # of issuance — it just means nothing new has been filed. Returning 0.0
            # there would put a fabricated exact-zero cluster into the cross-sectional
            # ranking for every sparse filer.
            and now_filed != prior_filed
            else float("nan")
        )
        feats["range_vol_20d"] = _range_vol(high, low, pos, 20)
        feats["range_vol_60d"] = _range_vol(high, low, pos, 60)
        feats["dividend_yield_ttm"] = _dividend_yield_ttm(dividends, adj_close[pos], pos)
        for _name in ("revenue_est_rev_30d", "revenue_est_rev_90d",
                      "eps_num_est_chg_90d", "coverage_level", "coverage_drop_90d"):
            feats[_name] = est_ctx[_name][j]
        feats["est_available"] = est_ctx["est_available"][j]
        feats["est_staleness_days"] = est_ctx["est_staleness_days"][j]
        # --- source-absence mask ------------------------------------------------
        # Where the upstream source has nothing as of this date, the derived columns
        # are UNDEFINED, not zero. Emitting NaN keeps them out of the cross-sectional
        # ranking and lets LightGBM route them natively; the availability flags above
        # stay finite so the model keeps an explicit handle on the missingness.
        if not fund_avail[j]:
            for _name in FUNDAMENTAL_SOURCED_FEATURES:
                feats[_name] = float("nan")
        if not feats["est_available"]:
            for _name in ESTIMATE_SOURCED_FEATURES:
                feats[_name] = float("nan")
        targets = documented_targets(prices, pos, getattr(frame, "security_events", None),
                                     max(grid), dates=trade_dates)
        feats.update(accounting[j])
        feats.update(fixed[j])
        feats.update(news[j])
        sector, industry = _sector_on(
            getattr(frame, "sector_history", None), g, (frame.sector, frame.industry)
        )
        row = {
            "date": g,
            "ticker_id": frame.ticker_id,
            # Audit-only cohort; never part of a feature catalog.
            "removed_cohort": getattr(frame, "removed_at", None) is not None,
            # Sector AS OF this row's date, not today's label — it drives both the
            # sector-relative target and the within-sector metric.
            "sector": sector,
            "industry": industry,
            # Tagged per row, filtered in prepare_panel: a name added to the index
            # in 2023 must not sit in the 2017 cross-section. None = unknown.
            "in_index": _in_index_on(getattr(frame, "membership", None), g),
            "index_id": index_on(getattr(frame, "membership", None), g),
            **feats,
        }
        for h in HORIZONS:
            ret, valid, entry_date, label_end, status = targets[h]
            row[f"r_{h}"] = ret
            row[f"mask_{h}"] = valid
            row[f"entry_{h}"] = entry_date
            row[f"label_end_{h}"] = label_end
            row[f"outcome_{h}"] = status
        rows.append(row)
    return rows
