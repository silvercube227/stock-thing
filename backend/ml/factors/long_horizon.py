"""Registered feature definitions. No source fallback or return-guided tuning."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from backend.ml.factors.pit import day

ANALYST_FEATURES = ["fixed_eps_rev_30d", "fixed_eps_rev_90d", "fixed_eps_persistence"]
BREADTH_FEATURES = ["fixed_eps_revision_breadth"]
NEWS_FEATURES = [
    "news_sent_90d",
    "news_neg_share_90d",
    "news_guidance_balance_90d",
    "news_operating_sent_90d",
]
ACCOUNTING_FEATURES = ["accruals_assets", "gross_profit_assets", "asset_growth_yoy"]
STRESS_FEATURES = ["stress_x_mom_3m", "stress_x_mom_12_1"]
MACRO_FEATURES = ["beta_rates_x_shock", "beta_credit_x_shock"]
PACKS = {
    "analyst": ANALYST_FEATURES,
    "news": NEWS_FEATURES,
    "accounting": ACCOUNTING_FEATURES,
    "stress": STRESS_FEATURES,
    "macro": MACRO_FEATURES,
}
SOURCED_FEATURES = ANALYST_FEATURES + BREADTH_FEATURES + NEWS_FEATURES + ACCOUNTING_FEATURES
TRANSFORMS = {c: "post_rank_product" for c in STRESS_FEATURES + MACRO_FEATURES}


def fixed_estimate_features(rows, dates, *, allow_provisional=False):
    rows = sorted(
        (r for r in rows if r.get("verified")
         or (allow_provisional and r.get("exploratory_usable"))),
        key=lambda r: day(r["as_of_date"]),
    )
    out = []
    for d in dates:
        d = day(d)
        result = dict.fromkeys(ANALYST_FEATURES + BREADTH_FEATURES, float("nan"))
        known = [r for r in rows if day(r["as_of_date"]) <= d]
        consensus = [
            r
            for r in known
            if r.get("contributor_id", "consensus") == "consensus"
            and day(r["fiscal_period_end"]) > d
        ]
        if not consensus:
            out.append(result)
            continue
        current = max(
            consensus,
            key=lambda r: (day(r["as_of_date"]), -day(r["fiscal_period_end"]).toordinal()),
        )
        period, basis = current["fiscal_period_end"], current["adjustment_basis"]
        series = [r for r in consensus if r["fiscal_period_end"] == period]

        def at(when, source=series, basis=basis):
            eligible = [r for r in source if day(r["as_of_date"]) <= when]
            if not eligible:
                return None
            row = eligible[-1]
            # Monthly snapshots must not silently carry stale values across gaps.
            if (when - day(row["as_of_date"])).days > 35 or row["adjustment_basis"] != basis:
                return None
            return row

        cur = at(d)
        if cur:
            for span in (30, 90):
                prior = at(d - timedelta(days=span))
                if prior and abs(prior["eps"]) >= 0.01:
                    result[f"fixed_eps_rev_{span}d"] = (cur["eps"] - prior["eps"]) / abs(
                        prior["eps"]
                    )
            monthly = [at(d)] + [
                at((pd.Period(d, freq="M") - i).end_time.date()) for i in range(1, 4)
            ]
            if (
                all(r is not None for r in monthly)
                and len({day(r["as_of_date"]) for r in monthly}) == 4
            ):
                result["fixed_eps_persistence"] = float(
                    np.mean([monthly[i]["eps"] > monthly[i + 1]["eps"] for i in range(3)])
                )
            changes = []
            contributors = {r.get("contributor_id") for r in known} - {None, "consensus"}
            for contributor in contributors:
                cr = [
                    r
                    for r in known
                    if r.get("contributor_id") == contributor and r["fiscal_period_end"] == period
                ]
                now, before = at(d, cr), at(d - timedelta(days=90), cr)
                if now and before:
                    changes.append(np.sign(now["eps"] - before["eps"]))
            if len(changes) >= 5:
                result["fixed_eps_revision_breadth"] = float(np.mean(changes))
        out.append(result)
    return out


def accounting_features(facts, dates, sectors):
    """Convert original cumulative spans to contiguous quarters, then form TTM."""
    nonfinancial_sectors = {
        "Communication Services", "Consumer Discretionary", "Consumer Staples",
        "Energy", "Health Care", "Industrials", "Information Technology",
        "Materials", "Real Estate", "Utilities",
    }
    out = []
    for d, sector in zip(dates, sectors, strict=True):
        result = dict.fromkeys(ACCOUNTING_FEATURES, float("nan"))
        # An unknown dated sector cannot establish eligibility for this pack.
        if not isinstance(sector, str) or sector not in nonfinancial_sectors:
            out.append(result)
            continue
        known = sorted(
            (r for r in facts if day(r["filed_at"]) <= day(d)),
            key=lambda r: (day(r["filed_at"]), r["accession_number"]),
        )
        # First-published observation for each exact span; later restatements do
        # not rewrite it. This choice is frozen for the registered experiment.
        spans = {}
        for r in known:
            key = (
                r["metric"],
                day(r["period_start"]) if r.get("period_start") else None,
                day(r["period_end"]),
            )
            spans.setdefault(key, float(r["value"]))
        assets = sorted(
            (end, v)
            for (metric, start, end), v in spans.items()
            if metric == "assets" and start is None and v > 0
        )
        if not assets:
            out.append(result)
            continue
        end, asset = assets[-1]
        previous = [(e, v) for e, v in assets if 340 <= (end - e).days <= 390]
        if not previous or (day(d) - end).days > 200:
            out.append(result)
            continue
        previous_end, previous_asset = previous[-1]
        avg_assets = (asset + previous_asset) / 2
        result["asset_growth_yoy"] = asset / previous_asset - 1
        totals = {}
        for metric in ("net_income", "operating_cash_flow", "gross_profit"):
            flows = sorted(
                (start, finish, v)
                for (m, start, finish), v in spans.items()
                if m == metric and start is not None and finish <= end
            )
            quarters = {}
            for start, finish, v in flows:
                if 70 <= (finish - start).days <= 105:
                    quarters[(start, finish)] = v
                elif 150 <= (finish - start).days <= 380:
                    prior = [
                        (e, pv)
                        for s, e, pv in flows
                        if s == start and 70 <= (finish - e).days <= 105
                    ]
                    if prior:
                        prior_end, prior_value = max(prior)
                        quarters.setdefault(
                            (prior_end + timedelta(days=1), finish), v - prior_value
                        )
            recent = sorted((s, e, v) for (s, e), v in quarters.items() if previous_end < e <= end)
            if (
                len(recent) == 4
                and recent[-1][1] == end
                and recent[0][0] == previous_end + timedelta(days=1)
                and all(recent[i][0] == recent[i - 1][1] + timedelta(days=1) for i in range(1, 4))
            ):
                totals[metric] = sum(v for _, _, v in recent)
            # An original annual span is also a complete trailing year.
            annual = [v for s, e, v in flows if e == end and s == previous_end + timedelta(days=1)]
            if annual:
                totals[metric] = annual[-1]
        if "net_income" in totals and "operating_cash_flow" in totals:
            result["accruals_assets"] = (
                totals["net_income"] - totals["operating_cash_flow"]
            ) / avg_assets
        if "gross_profit" in totals:
            result["gross_profit_assets"] = totals["gross_profit"] / avg_assets
        out.append(result)
    return out


def news_features(rows, coverage, dates, *, exploratory=False):
    rows = [r for r in rows if exploratory
            or not str(r["scorer_revision"]).startswith("exploratory:")]
    out = []
    for d in dates:
        d = day(d)
        start = d - timedelta(days=89)
        result = dict.fromkeys(NEWS_FEATURES, float("nan"))
        intervals = sorted(
            (day(r["start_date"]), day(r["end_date"]))
            for r in coverage
            if r["status"] in ("complete", "empty")
            or (exploratory and r["status"] == "partial")
        )
        cursor = start
        for a, b in intervals:
            if a <= cursor <= b:
                cursor = b + timedelta(days=1)
        if cursor <= d:
            out.append(result)
            continue
        window = [r for r in rows if start <= day(r["score_date"]) <= d]
        if len({r["scorer_revision"] for r in window}) > 1:
            raise ValueError("mixed scorer revisions in news window")
        n = sum(r["n_events"] for r in window)
        if n:
            result["news_sent_90d"] = sum(r["sentiment_sum"] for r in window) / n
            result["news_neg_share_90d"] = sum(r["n_negative"] for r in window) / n
        up, down = (sum(r[k] for r in window) for k in ("guidance_up", "guidance_down"))
        if up + down:
            result["news_guidance_balance_90d"] = (up - down) / (up + down)
        operating = sum(r["operating_n"] for r in window)
        if operating:
            result["news_operating_sent_90d"] = (
                sum(r["operating_sentiment_sum"] for r in window) / operating
            )
        out.append(result)
    return out


def add_macro_features(panel, frames, macro, market_returns):
    """Monthly market-controlled betas and causal magnitude-preserving shocks."""
    from backend.ingestion.macro import VERIFIED_SOURCE

    out = panel.copy()
    for c in MACRO_FEATURES:
        out[c] = np.nan
    market = pd.Series(market_returns, dtype=float)
    if market.empty:
        return out
    market.index = pd.to_datetime(market.index)
    mret = market.groupby(market.index.to_period("M")).sum()
    stocks = {}
    for frame in frames:
        s = pd.Series(
            {
                pd.Timestamp(p["trade_date"]): float(p["adj_close"])
                for p in frame.prices
                if p.get("adj_close") and p["adj_close"] > 0
            }
        )
        if not s.empty:
            s = s.sort_index().groupby(s.index.to_period("M")).last()
            s = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"))
            stocks[frame.ticker_id] = np.log(s).diff()
    for d, group in out.groupby("date"):
        d = day(d)
        month = pd.Period(d, freq="M")
        # BAA10Y retained by the 2026-09-08 research decision. The available
        # ICE OAS history does not overlap the registered selection window.
        for series, feature in (("DGS10", MACRO_FEATURES[0]), ("BAA10Y", MACRO_FEATURES[1])):
            rows = sorted(
                (
                    r
                    for r in macro
                    if r["series_id"] == series
                    and r.get("verified")
                    and r.get('source') == VERIFIED_SOURCE
                    and day(r["available_from"]) <= d
                    and day(r["vintage_date"]) <= d
                ),
                key=lambda r: day(r["vintage_date"]),
            )
            # Resolve versions before filtering missing values: a later explicit
            # withdrawal must erase the earlier value rather than revive it.
            latest = {pd.Timestamp(r['obs_date']): r['value'] for r in rows}
            values = {d: float(v) for d,v in latest.items() if v is not None and np.isfinite(v)}
            if not values:
                continue
            daily = pd.Series(values).sort_index()
            levels = daily.groupby(daily.index.to_period("M")).last()
            levels = levels.reindex(pd.period_range(levels.index.min(), month, freq="M"))
            history = levels[levels.index < month]
            changes = history.diff(3).dropna()
            if len(changes) < 36 or not np.isfinite(changes.std()) or changes.std() <= 0:
                continue
            # Last observable level and a level observed by d-3 calendar months.
            prior_date = pd.Timestamp(d) - pd.DateOffset(months=3)
            prior_levels = daily[daily.index <= prior_date]
            if prior_levels.empty or (pd.Timestamp(d) - daily.index[-1]).days > 7:
                continue
            shock = float(
                np.clip(
                    (daily.iloc[-1] - prior_levels.iloc[-1] - changes.mean()) / changes.std(), -3, 3
                )
            )
            factor = history.diff()
            betas = {}
            for idx, row in group.iterrows():
                stock = stocks.get(row.ticker_id)
                if stock is None:
                    continue
                sample = pd.concat(
                    [stock.rename("stock"), mret.rename("market"), factor.rename("factor")], axis=1
                )
                sample = sample[(sample.index < month) & (sample.index >= month - 60)].dropna()
                if len(sample) < 36:
                    continue
                x = np.column_stack([np.ones(len(sample)), sample.market, sample.factor])
                if np.linalg.matrix_rank(x) == 3:
                    betas[idx] = np.linalg.lstsq(x, sample.stock, rcond=None)[0][2]
            if betas:
                b = pd.Series(betas)
                ranks = 2 * (b.rank() - 1) / (len(b) - 1) - 1 if len(b) > 1 else b * 0
                out.loc[ranks.index, feature] = ranks * shock
    return out
