"""Cross-sectional LightGBM walk-forward baseline (the model to beat).

Why this exists: the PatchTST run showed a val rank-IC that did NOT survive to a
clean holdout, and a single 6-month split is too thin (7/5/2 monthly
cross-sections) to confirm or deny skill. Both the TSFM-in-finance literature and
this project's own results point to gradient-boosted trees on cross-sectional
factor features as the honest baseline for relative equity ranking. This module
answers the gating question — *is there ANY out-of-sample cross-sectional
signal?* — with a proper walk-forward and a shuffle null, cheaply.

Design (mirrors train.py's split: pure core + thin DB shell):
  - Feature/panel assembly + walk-forward are pure functions over `TickerFrame`s
    or a prepared panel DataFrame, so they unit-test on synthetic frames (no DB).
  - `run()` / `main()` load frames from Supabase and print the report.

Method:
  - One row per (ticker, month-end) on the common calendar grid → full ~500-name
    cross-sections (same grid as the transformer's --calendar-aligned mode).
  - Features: classic point-in-time factors (momentum, vol, MA gaps, liquidity)
    + the fundamentals/sentiment columns, reusing features.py's PIT helpers so we
    inherit its no-look-ahead guarantees.
  - Target: the universe-demeaned (relative) H-day log return — "did this ticker
    beat the universe?" in continuous form, the natural rank-IC pairing.
  - Features are cross-sectionally rank-normalized per date (robust to outliers;
    this is also where the "cross-sectional rank" factor lives).
  - Walk-forward: expanding window, refit every month-end, embargo = one horizon
    so a training label window can't overlap the test prediction window. Score
    Spearman rank-IC on each test cross-section, then report mean IC, ICIR
    (mean/std), the across-fold t-stat, and hit rate.
  - Shuffle null: same harness with train labels permuted, repeated, to get the
    no-signal band the real mean IC must clear.
"""

from __future__ import annotations

import argparse
import asyncio
import math
from dataclasses import dataclass, field

import numpy as np

from backend.ingestion.calendar import HORIZON_TRADING_DAYS
from backend.ingestion.db import pool_context
from backend.ml.dataset import (
    TickerFrame,
    _as_date,
    build_calendar_grid,
    compute_targets,
    cross_sectional_medians,
    load_frames_cached,
)
from backend.ml.features import (
    SEQUENCE_LENGTH,
    _annotate_fundamentals,
    _build_fundamental_series,
    _build_sentiment_series,
)
from backend.ml.model import HORIZONS

# Tabular factor columns the model trains on (order is informational only).
PRICE_FEATURES = [
    "mom_1m", "mom_3m", "mom_6m", "mom_12_1",   # momentum (12_1 skips the last month)
    "log_market_cap",                           # log(adj_close × shares_outstanding)
    "vol_20d", "vol_60d", "vol_120d",           # realized vol
    "dist_high_252", "dist_low_252",            # distance to 52w extremes
    "ma_gap_50", "ma_gap_200",                  # gap vs moving averages
    "vol_trend",                                # 20d vs 120d dollar/volume trend
]
FUNDAMENTAL_FEATURES = [
    "revenue_growth", "gross_margin", "operating_margin", "debt_equity", "fcf_revenue",
]
VALUATION_FEATURES = [
    "earnings_yield", "book_to_market", "sales_to_price", "fcf_yield",
]
QUALITY_FEATURES = [
    "roe_ttm", "net_margin_ttm", "fcf_margin_ttm",
    "gross_margin_stability_4q", "operating_margin_stability_4q",
    "revenue_growth_stability_4q",
]
# Test-4 phase-1 experimental packs (opt-in via CLI; not in production FEATURE_COLS).
# resid_mom_*  : momentum after stripping out beta_252 * market move (structural mom)
# mom_accel_3_6: 3M vs 6M momentum — captures inflection vs decay
# mom_consistency_6m: fraction of last 6 monthly returns positive (smoothness)
# industry_neutral_mom_12_1: mom_12_1 minus within-(date, industry) median (panel-level)
RESIDUAL_MOM_FEATURES = [
    "resid_mom_12_1", "resid_mom_6m", "mom_accel_3_6",
    "mom_consistency_6m", "industry_neutral_mom_12_1",
]
# Filing-drift / surprise reaction features, derived purely from prices + filed_at.
EARNINGS_REACTION_FEATURES = [
    "filing_drift_30d", "filing_surprise_3d",
    "filings_recency_days", "filings_in_90d",
]
# LSEG/I-B-E-S analyst-estimate packs (opt-in; require analyst_estimates ingested).
# rec_mean is the consensus rating (1=Strong Buy .. 5=Sell), so a DROP = upgrades;
# rec_rev_* are (prior - current) so "net upgrades" reads positive.
ANALYST_REVISION_FEATURES = [
    "rec_mean_level", "rec_rev_30d", "rec_rev_90d", "price_target_rev_90d",
]
ESTIMATE_SURPRISE_FEATURES = [
    "revenue_surprise",
]
# Forward valuation stored as yields (inverse multiples) so ranking is monotonic
# and negative/near-zero denominators don't blow up — mirrors earnings_yield.
FORWARD_VALUATION_FEATURES = [
    "forward_earnings_yield", "forward_ebitda_yield", "price_target_upside",
]
SENTIMENT_FEATURES = ["sentiment_7d", "sentiment_14d"]
FEATURE_COLS = PRICE_FEATURES + FUNDAMENTAL_FEATURES + SENTIMENT_FEATURES
EXPERIMENTAL_FEATURES = (
    VALUATION_FEATURES + QUALITY_FEATURES + RESIDUAL_MOM_FEATURES + EARNINGS_REACTION_FEATURES
    + ANALYST_REVISION_FEATURES + ESTIMATE_SURPRISE_FEATURES + FORWARD_VALUATION_FEATURES
)
# The industry-relative *normalization* sweep (which hurt in test 3); residual /
# earnings-reaction features already adjust for market or filing context so they
# stay out of this list — double-grouping would re-shrink whatever signal they
# carry.
INDUSTRY_RELATIVE_FEATURES = (
    PRICE_FEATURES + FUNDAMENTAL_FEATURES + VALUATION_FEATURES + QUALITY_FEATURES
)


# =============================================================
# Config
# =============================================================


@dataclass
class LGBMConfig:
    """Deliberately shallow + regularized: cross-sectional return signal is weak,
    so the baseline should resist memorizing the train cross-sections."""

    n_estimators: int = 300
    learning_rate: float = 0.03
    num_leaves: int = 15
    max_depth: int = 4
    min_child_samples: int = 50
    subsample: float = 0.8          # row bagging
    colsample_bytree: float = 0.8   # feature bagging
    reg_lambda: float = 1.0
    # n_jobs=1 is REQUIRED, not a perf choice: this process also loads torch
    # (dataset.py -> model.py), whose bundled libomp.dylib is a second LLVM
    # OpenMP runtime. LightGBM spawning its own OpenMP thread team alongside it
    # segfaults on macOS. Single-threaded sidesteps it; shallow trees on ~50k
    # rows are fast enough that it doesn't matter.
    n_jobs: int = 1


@dataclass
class WalkForwardConfig:
    min_train_months: int = 36           # don't test until this much history exists
    max_train_months: int | None = None  # None = expanding window; int = rolling
    min_names: int = 30                  # skip a test cross-section thinner than this


@dataclass
class HorizonSpec:
    """Per-horizon training spec — target, hyperparameters, optional feature override.

    Each horizon trains its own LightGBM model (one regressor per H), so the spec
    is what makes "tune each horizon separately" a real workflow. The defaults
    here are deliberately the universe-relative baseline; production overrides
    live in `PRODUCTION_HORIZON_SPECS` below, which is the single source of truth
    consumed by the inference path.

    `feature_cols=None` means "use the production FEATURE_COLS at fit and predict
    time" — keeping per-horizon feature overrides optional so we only carry them
    once an experiment promotes a non-default pack for a specific H.
    """

    target_mode: str = "return"
    lgb_cfg: LGBMConfig = field(default_factory=LGBMConfig)
    feature_cols: list[str] | None = None


# Per-horizon production training defaults. Update this dict — and only this dict
# — when a sweep promotes a new target / hyperparameter / feature pack. The
# inference path reads it as its starting config; the walk-forward sweep tool
# (this file's CLI) tests *one* horizon at a time and is unaffected.
#
# Source of current values:
#   - 1M / 3M / 6M: `rank` — the historical inference default. NOT directly
#     compared against `return`/`sector_return`/`beta_resid` in any sweep, so we
#     preserve it rather than silently switch to a sweep-validated target. To
#     change, sweep against `rank` baseline at these horizons first.
#   - 1Y: `beta_resid` — test-4 phase-4 sweep: ICIR 0.552 → 0.685, Δ +0.133;
#     hit rate 0.742 → 0.863. Stripping market beta from the training target
#     forces the model away from cheap "high beta = high 1Y return" and toward
#     idiosyncratic alpha. The asymmetry is structural: beta noise dominates 1Y
#     return variance but is smaller relative to 3M dispersion, and beta_252 is
#     noisier relative to short-horizon market moves.
#   - 6M / 1Y feature_cols: + ESTIMATE_SURPRISE_FEATURES (LSEG revenue surprise).
#     De-survivorshipped walk-forward ablation (2026-05-29): 6M ICIR 0.409→0.442
#     (+0.033, bootstrap p=0.003, null z=6.5), 1Y 0.378→0.454 (+0.076, null z=9.7).
#     The other LSEG packs (analyst revisions, forward valuation) did not beat the
#     null net of surprise, and 3M saw no estimate signal — so only surprise is
#     promoted, and only at 6M/1Y. (Baselines here are below CLAUDE.md's older
#     survivor-only ICIRs because the universe now includes removed-from-index names.)
_BASELINE_PLUS_SURPRISE = FEATURE_COLS + ESTIMATE_SURPRISE_FEATURES
PRODUCTION_HORIZON_SPECS: dict[str, HorizonSpec] = {
    "1M": HorizonSpec(target_mode="rank"),
    "3M": HorizonSpec(target_mode="rank"),
    "6M": HorizonSpec(target_mode="rank", feature_cols=_BASELINE_PLUS_SURPRISE),
    "1Y": HorizonSpec(target_mode="beta_resid", feature_cols=_BASELINE_PLUS_SURPRISE),
}


# =============================================================
# Per-ticker price-derived factors (point-in-time)
# =============================================================


def _log_ratio(a: float | None, b: float | None) -> float:
    """log(a/b), or 0.0 if either price is missing/non-positive."""
    if a is None or b is None or a <= 0 or b <= 0:
        return 0.0
    return math.log(a / b)


def _safe_ratio(num: float | None, den: float | None) -> float:
    if num is None or den is None or abs(den) <= 1e-9:
        return 0.0
    return float(num / den)


def _price_features(
    adj_close: list[float | None], volume: list[float], trade_dates: list, pos: int,
    shares_outstanding: int | None = None,
    market_returns: dict | None = None,
) -> dict[str, float]:
    """Factor features computed from the ticker's own series up to bar `pos`.

    Requires pos >= SEQUENCE_LENGTH (252) so the 12-1 momentum and 52-week window
    have full lookback — the same minimum the aligned assembler enforces.
    """
    P = adj_close[pos]
    log_mcap = (
        math.log(P * shares_outstanding)
        if P and shares_outstanding and P > 0 and shares_outstanding > 0
        else 0.0
    )
    feats = {
        "mom_1m": _log_ratio(P, adj_close[pos - 21]),
        "mom_3m": _log_ratio(P, adj_close[pos - 63]),
        "mom_6m": _log_ratio(P, adj_close[pos - 126]),
        "mom_12_1": _log_ratio(adj_close[pos - 21], adj_close[pos - 252]),
        "log_market_cap": log_mcap,
    }

    stock_daily: list[float] = []
    market_daily: list[float] = []
    if market_returns:
        for i in range(pos - 251 + 1, pos + 1):
            r_stock = _log_ratio(adj_close[i], adj_close[i - 1])
            r_mkt = market_returns.get(trade_dates[i])
            if r_mkt is None or not np.isfinite(r_mkt):
                continue
            stock_daily.append(r_stock)
            market_daily.append(float(r_mkt))
    if len(stock_daily) >= 60:
        x = np.asarray(stock_daily, dtype=float)
        y = np.asarray(market_daily, dtype=float)
        var_y = float(y.var())
        feats["beta_252d"] = float(np.cov(x, y, ddof=0)[0, 1] / var_y) if var_y > 1e-12 else 0.0
    else:
        feats["beta_252d"] = 0.0

    window = np.array(
        [np.nan if v is None or v <= 0 else v for v in adj_close[pos - 251 : pos + 1]],
        dtype=float,
    )
    logp = np.log(window)
    daily = np.diff(logp)  # length 251

    def _std(x: np.ndarray) -> float:
        x = x[~np.isnan(x)]
        return float(x.std()) if x.size > 1 else 0.0

    feats["vol_20d"] = _std(daily[-20:])
    feats["vol_60d"] = _std(daily[-60:])
    feats["vol_120d"] = _std(daily[-120:])

    hi, lo = np.nanmax(window), np.nanmin(window)
    feats["dist_high_252"] = _log_ratio(P, hi)
    feats["dist_low_252"] = _log_ratio(P, lo)
    feats["ma_gap_50"] = _log_ratio(P, float(np.nanmean(window[-50:])))
    feats["ma_gap_200"] = _log_ratio(P, float(np.nanmean(window[-200:])))

    v = np.array(volume[pos - 119 : pos + 1], dtype=float)
    v_recent = float(v[-20:].mean()) if v[-20:].size else 0.0
    v_long = float(v.mean()) if v.size else 0.0
    feats["vol_trend"] = math.log(v_recent / v_long) if v_recent > 0 and v_long > 0 else 0.0

    # --- Test-4 phase-1: residual / structural momentum (opt-in pack) ---
    # market_mom_* is the universe cumulative log return over the same trailing
    # window as the matching mom_* feature. When market_returns is missing we
    # fall back to zero, which makes resid_mom_* collapse to mom_* — the test
    # in test_gbm_baseline.py exercises that path.
    def _sum_mkt(start_idx: int, end_idx: int) -> float:
        """Sum of universe log returns over (start_idx, end_idx] in trade_dates."""
        if market_returns is None or end_idx <= start_idx:
            return 0.0
        total = 0.0
        for k in range(start_idx + 1, end_idx + 1):
            r = market_returns.get(trade_dates[k])
            if r is not None and np.isfinite(r):
                total += float(r)
        return total

    mkt_12_1 = _sum_mkt(pos - 252, pos - 21)
    mkt_6m = _sum_mkt(pos - 126, pos)
    beta = feats["beta_252d"]
    feats["resid_mom_12_1"] = feats["mom_12_1"] - beta * mkt_12_1
    feats["resid_mom_6m"] = feats["mom_6m"] - beta * mkt_6m
    feats["mom_accel_3_6"] = feats["mom_3m"] - feats["mom_6m"]

    monthly_pos = 0
    n_monthly = 0
    for i in range(1, 7):
        p_end = adj_close[pos - 21 * (i - 1)]
        p_start = adj_close[pos - 21 * i]
        if p_end and p_start and p_end > 0 and p_start > 0:
            if math.log(p_end / p_start) > 0:
                monthly_pos += 1
            n_monthly += 1
    feats["mom_consistency_6m"] = float(monthly_pos / n_monthly) if n_monthly else 0.0
    return feats


def _ttm_net_income_asof(fund_rows: list[dict], as_of_dates: list) -> list[float]:
    """Most recent point-in-time TTM net income for each as-of date.

    10-K rows contribute their annual `net_income` directly. 10-Q rows use the
    trailing four quarterly `net_income` values when available; otherwise we
    fall back to the most recent annual filing already on file.
    """
    import bisect

    rows_sorted = sorted(fund_rows, key=lambda r: _as_date(r["filed_at"]))
    annotated: list[dict] = []
    quarterly_history: list[tuple] = []
    annual_history: list[tuple] = []

    for row in rows_sorted:
        filed_at = _as_date(row["filed_at"])
        period_end = _as_date(row["period_end"])
        try:
            net_income = float(row["net_income"]) if row.get("net_income") is not None else None
        except (TypeError, ValueError):
            net_income = None

        filing_type = row.get("filing_type")
        ttm_net_income: float | None = None

        if filing_type == "10-Q" and net_income is not None:
            quarterly_history.append((period_end, net_income))
            recent: list[tuple] = []
            seen_periods: set = set()
            for pe, ni in reversed(quarterly_history):
                if pe in seen_periods:
                    continue
                seen_periods.add(pe)
                recent.append((pe, ni))
                if len(recent) == 4:
                    break
            if len(recent) == 4 and (period_end - recent[-1][0]).days <= 380:
                ttm_net_income = float(sum(ni for _pe, ni in recent))
        elif filing_type == "10-K" and net_income is not None:
            annual_history.append((period_end, net_income))
            ttm_net_income = net_income

        if ttm_net_income is None:
            for pe, ni in reversed(annual_history):
                if abs((period_end - pe).days) <= 380:
                    ttm_net_income = float(ni)
                    break

        annotated.append({"filed_at": filed_at, "ttm_net_income": float(ttm_net_income or 0.0)})

    filed_ats = [r["filed_at"] for r in annotated]
    out: list[float] = []
    for d in as_of_dates:
        idx = bisect.bisect_right(filed_ats, d) - 1
        out.append(float(annotated[idx]["ttm_net_income"]) if idx >= 0 else 0.0)
    return out


def _fundamental_context_asof(fund_rows: list[dict], as_of_dates: list) -> dict[str, list[float]]:
    """Point-in-time valuation + quality snapshots for each as-of date."""
    import bisect

    rows_sorted = sorted(fund_rows, key=lambda r: _as_date(r["filed_at"]))
    if not rows_sorted:
        return {
            "ttm_revenue": [0.0] * len(as_of_dates),
            "ttm_net_income": [0.0] * len(as_of_dates),
            "ttm_fcf": [0.0] * len(as_of_dates),
            "total_equity": [0.0] * len(as_of_dates),
            "gross_margin": [0.0] * len(as_of_dates),
            "operating_margin": [0.0] * len(as_of_dates),
            "revenue_growth": [0.0] * len(as_of_dates),
            "gross_margin_stability_4q": [0.0] * len(as_of_dates),
            "operating_margin_stability_4q": [0.0] * len(as_of_dates),
            "revenue_growth_stability_4q": [0.0] * len(as_of_dates),
        }

    annotated_core = _annotate_fundamentals(rows_sorted)
    annotated: list[dict] = []
    quarter_hist: dict[str, list[tuple]] = {"revenue": [], "net_income": [], "fcf": []}
    annual_hist: dict[str, list[tuple]] = {"revenue": [], "net_income": [], "fcf": []}

    def _safe_num(val) -> float | None:
        try:
            return float(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    def _compute_ttm(metric: str, period_end, filing_type: str) -> float:
        q_hist = quarter_hist[metric]
        a_hist = annual_hist[metric]
        ttm_val: float | None = None
        if filing_type == "10-Q":
            recent: list[tuple] = []
            seen_periods: set = set()
            for pe, v in reversed(q_hist):
                if pe in seen_periods:
                    continue
                seen_periods.add(pe)
                recent.append((pe, v))
                if len(recent) == 4:
                    break
            if len(recent) == 4 and (period_end - recent[-1][0]).days <= 380:
                ttm_val = float(sum(v for _pe, v in recent))
        elif filing_type == "10-K" and a_hist:
            ttm_val = float(a_hist[-1][1])

        if ttm_val is None:
            for pe, v in reversed(a_hist):
                if abs((period_end - pe).days) <= 380:
                    ttm_val = float(v)
                    break
        return float(ttm_val or 0.0)

    def _stability(field: str) -> float:
        vals = [float(r[field]) for r in annotated[-4:] if field in r]
        return float(-np.std(np.asarray(vals, dtype=float))) if len(vals) >= 2 else 0.0

    for row, core in zip(rows_sorted, annotated_core, strict=False):
        period_end = _as_date(row["period_end"])
        filing_type = str(row.get("filing_type") or "")
        for metric in ("revenue", "net_income", "fcf"):
            val = _safe_num(row.get(metric))
            if val is None:
                continue
            if filing_type == "10-Q":
                quarter_hist[metric].append((period_end, val))
            elif filing_type == "10-K":
                annual_hist[metric].append((period_end, val))

        current = {
            "filed_at": _as_date(row["filed_at"]),
            "ttm_revenue": _compute_ttm("revenue", period_end, filing_type),
            "ttm_net_income": _compute_ttm("net_income", period_end, filing_type),
            "ttm_fcf": _compute_ttm("fcf", period_end, filing_type),
            "total_equity": float(_safe_num(row.get("total_equity")) or 0.0),
            "gross_margin": float(core["gross_margin"]),
            "operating_margin": float(core["operating_margin"]),
            "revenue_growth": float(core["revenue_growth"]),
        }
        annotated.append(current)
        current["gross_margin_stability_4q"] = _stability("gross_margin")
        current["operating_margin_stability_4q"] = _stability("operating_margin")
        current["revenue_growth_stability_4q"] = _stability("revenue_growth")

    filed_ats = [r["filed_at"] for r in annotated]
    out = {
        "ttm_revenue": [],
        "ttm_net_income": [],
        "ttm_fcf": [],
        "total_equity": [],
        "gross_margin": [],
        "operating_margin": [],
        "revenue_growth": [],
        "gross_margin_stability_4q": [],
        "operating_margin_stability_4q": [],
        "revenue_growth_stability_4q": [],
    }
    for d in as_of_dates:
        idx = bisect.bisect_right(filed_ats, d) - 1
        snap = annotated[idx] if idx >= 0 else None
        for k in out:
            out[k].append(float(snap[k]) if snap is not None else 0.0)
    return out


def _estimates_context_asof(est_rows: list[dict], as_of_dates: list) -> dict[str, list[float]]:
    """Point-in-time LSEG analyst-estimate features for each as-of date.

    LSEG fields land on different dates (sparse rows), so each field is looked up
    INDEPENDENTLY: the most recent non-null observation with as_of_date <= d.
    Revisions compare against the value ~30/90 calendar days earlier. Revenue
    surprise carries forward the latest report's (actual - pre-report consensus).
    Forward multiples become yields (inverse); non-positive ratios -> 0. Gaps -> 0.

    `price_target_mean` is returned as an intermediate (build_ticker_rows turns it
    into price_target_upside with the as-of price); it is not a feature itself.
    """
    import bisect
    from datetime import timedelta

    keys = ("rec_mean_level", "rec_rev_30d", "rec_rev_90d", "price_target_mean",
            "price_target_rev_90d", "forward_earnings_yield", "forward_ebitda_yield",
            "revenue_surprise")
    if not est_rows:
        return {k: [0.0] * len(as_of_dates) for k in keys}

    rows_sorted = sorted(est_rows, key=lambda r: _as_date(r["as_of_date"]))
    snap_fields = ("rec_mean", "price_target_mean", "revenue_mean", "fwd_pe", "fwd_ev_ebitda")
    series: dict[str, tuple[list, list]] = {f: ([], []) for f in snap_fields}
    rev_dates: list = []
    rev_actuals: list[float] = []
    for r in rows_sorted:
        d = _as_date(r["as_of_date"])
        for f in snap_fields:
            v = r.get(f)
            if v is not None:
                series[f][0].append(d)
                series[f][1].append(float(v))
        ra = r.get("revenue_actual")
        if ra is not None:
            rev_dates.append(d)
            rev_actuals.append(float(ra))

    def asof(field: str, target) -> float | None:
        dates, vals = series[field]
        i = bisect.bisect_right(dates, target) - 1
        return vals[i] if i >= 0 else None

    def surprise_asof(target) -> float | None:
        i = bisect.bisect_right(rev_dates, target) - 1
        if i < 0:
            return None
        report_date, actual = rev_dates[i], rev_actuals[i]
        cons = asof("revenue_mean", report_date - timedelta(days=1))  # pre-report consensus
        if cons is None or cons == 0:
            return None
        return (actual - cons) / abs(cons)

    out: dict[str, list[float]] = {k: [] for k in keys}
    for d in as_of_dates:
        d = _as_date(d)
        rec = asof("rec_mean", d)
        rec30 = asof("rec_mean", d - timedelta(days=30))
        rec90 = asof("rec_mean", d - timedelta(days=90))
        pt = asof("price_target_mean", d)
        pt90 = asof("price_target_mean", d - timedelta(days=90))
        pe = asof("fwd_pe", d)
        ev = asof("fwd_ev_ebitda", d)
        sp = surprise_asof(d)

        out["rec_mean_level"].append(rec if rec is not None else 0.0)
        out["rec_rev_30d"].append((rec30 - rec) if rec is not None and rec30 is not None else 0.0)
        out["rec_rev_90d"].append((rec90 - rec) if rec is not None and rec90 is not None else 0.0)
        out["price_target_mean"].append(pt if pt is not None else 0.0)
        out["price_target_rev_90d"].append(
            (pt - pt90) / abs(pt90) if pt is not None and pt90 not in (None, 0) else 0.0)
        out["forward_earnings_yield"].append(1.0 / pe if pe is not None and pe > 0 else 0.0)
        out["forward_ebitda_yield"].append(1.0 / ev if ev is not None and ev > 0 else 0.0)
        out["revenue_surprise"].append(sp if sp is not None else 0.0)
    return out


def _earnings_reaction_asof(
    fund_rows: list[dict],
    bar_positions: list[int],
    bar_dates: list,
    trade_dates: list,
    adj_close: list[float | None],
    market_returns: dict | None,
) -> dict[str, list[float]]:
    """Per-grid-date filing-reaction features built from prices + filed_at only.

    For each grid date we look back to the most recent filing on/before that date
    and summarize a few aspects of its market reaction:
      filing_surprise_3d  — abnormal return over [filed_at-1, filed_at+1] trading
                            days (proxy for what the market thought of the print).
      filing_drift_30d    — abnormal return from the trading day after filed_at
                            out to +30 trading days (or up to the grid date, if
                            fewer days have elapsed). Post-earnings drift signal.
      filings_recency_days — calendar days since the latest filing.
      filings_in_90d      — count of filings within the trailing 90 calendar days.

    PIT-safe by construction: we only ever index `adj_close` up to `pos` (the
    grid-date bar position).
    """
    import bisect
    from datetime import timedelta

    n = len(bar_dates)
    out: dict[str, list[float]] = {
        "filing_drift_30d": [0.0] * n,
        "filing_surprise_3d": [0.0] * n,
        "filings_recency_days": [0.0] * n,
        "filings_in_90d": [0.0] * n,
    }
    if not fund_rows or not trade_dates:
        return out

    filings_sorted = sorted(fund_rows, key=lambda r: _as_date(r["filed_at"]))
    filed_ats = [_as_date(r["filed_at"]) for r in filings_sorted]

    def _sum_mkt(start_idx: int, end_idx: int) -> float:
        if market_returns is None or end_idx <= start_idx:
            return 0.0
        total = 0.0
        for k in range(start_idx + 1, end_idx + 1):
            r = market_returns.get(trade_dates[k])
            if r is not None and np.isfinite(r):
                total += float(r)
        return total

    for j, g in enumerate(bar_dates):
        pos = bar_positions[j]
        idx = bisect.bisect_right(filed_ats, g) - 1
        if idx < 0:
            continue
        filed_at = filed_ats[idx]
        # First trading day on or after filed_at (bisect_left returns the next
        # bar when filed_at falls on a weekend/holiday; if equal to a trading
        # day, that day itself is selected). Bound by pos so we never peek past
        # the grid-date bar — PIT guard.
        file_pos = bisect.bisect_left(trade_dates, filed_at)
        if file_pos > pos:
            # Filing recorded ahead of the price series for this grid date — can't
            # measure reaction yet; recency + count are still valid.
            out["filings_recency_days"][j] = float((g - filed_at).days)
            cutoff = g - timedelta(days=90)
            lo = bisect.bisect_left(filed_ats, cutoff)
            out["filings_in_90d"][j] = float(idx + 1 - lo)
            continue

        out["filings_recency_days"][j] = float((g - filed_at).days)
        cutoff = g - timedelta(days=90)
        lo = bisect.bisect_left(filed_ats, cutoff)
        out["filings_in_90d"][j] = float(idx + 1 - lo)

        # 3-day surprise window: log return [file_pos-1, file_pos+1], minus market.
        start_s = max(file_pos - 1, 0)
        end_s = min(file_pos + 1, pos)
        if end_s > start_s:
            p_a = adj_close[start_s]
            p_b = adj_close[end_s]
            if p_a and p_b and p_a > 0 and p_b > 0:
                out["filing_surprise_3d"][j] = float(
                    math.log(p_b / p_a) - _sum_mkt(start_s, end_s)
                )

        # Post-filing drift: [file_pos+1, file_pos+31] or shorter if too fresh.
        start_d = file_pos + 1
        end_d = min(start_d + 30, pos)
        # Require ≥ 5 trading days of post-filing data so the value isn't noise.
        if start_d < len(adj_close) and end_d - start_d >= 5:
            p_a = adj_close[start_d]
            p_b = adj_close[end_d]
            if p_a and p_b and p_a > 0 and p_b > 0:
                out["filing_drift_30d"][j] = float(
                    math.log(p_b / p_a) - _sum_mkt(start_d, end_d)
                )

    return out


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
    import bisect

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
    import bisect

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
    est_ctx = _estimates_context_asof(frame.estimates or [], bar_dates)

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
        pt = est_ctx["price_target_mean"][j]
        feats["price_target_upside"] = ((pt - price) / price) if price and price > 0 and pt else 0.0
        # Panel-level demean overwrites this in `prepare_panel`; until then leave
        # it equal to mom_12_1 so single-ticker callers see a finite value.
        feats["industry_neutral_mom_12_1"] = feats["mom_12_1"]
        feats["sentiment_7d"] = float(sent[j, 0])
        feats["sentiment_14d"] = float(sent[j, 1])
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


# =============================================================
# Panel assembly + cross-sectional transforms
# =============================================================


def assemble_panel(
    frames: list[TickerFrame],
    grid: list,
    max_stale_days: int = 7,
    market_returns: dict | None = None,
):
    """Stack every ticker's rows into one tidy panel DataFrame (raw features)."""
    import pandas as pd

    rows: list[dict] = []
    for frame in frames:
        rows.extend(
            build_ticker_rows(
                frame,
                grid,
                max_stale_days=max_stale_days,
                market_returns=market_returns,
            )
        )
    return pd.DataFrame(rows)


def demean_cross_sectional(panel, medians: dict[str, dict]):
    """Relative target: subtract the per-date universe-median forward return.

    A row whose date has no median for a horizon is masked off for that horizon
    (matches dataset.relabel_cross_sectional's behavior).
    """
    out = panel.copy()
    for h in HORIZONS:
        med = out["date"].map(medians[h])
        has = med.notna() & out[f"mask_{h}"].astype(bool)
        out[f"r_{h}"] = np.where(has, out[f"r_{h}"] - med.fillna(0.0), 0.0)
        out[f"mask_{h}"] = has
    return out


def add_industry_neutral_momentum(panel, min_group_size: int = 5):
    """Subtract within-(date, industry) median from mom_12_1.

    Run after `assemble_panel` and before `rank_normalize_features` so the
    industry-neutral momentum factor still gets per-date rank-normalized in the
    full universe — different failure mode than industry-relative *normalization*
    (which hurt at every horizon), because the rest of the feature set stays in
    universe space.
    """
    if panel.empty or "mom_12_1" not in panel.columns:
        return panel
    out = panel.copy()
    if "industry" not in out.columns:
        out["industry_neutral_mom_12_1"] = out["mom_12_1"].astype(float)
        return out
    grp = out.groupby(["date", "industry"], dropna=False)
    medians = grp["mom_12_1"].transform("median")
    counts = grp["mom_12_1"].transform("count")
    use_group = out["industry"].notna() & (counts >= min_group_size)
    base = out["mom_12_1"].astype(float)
    out["industry_neutral_mom_12_1"] = np.where(
        use_group, base - medians.fillna(0.0), base
    )
    return out


def rank_normalize_features(
    panel,
    cols: list[str] = FEATURE_COLS,
    *,
    industry_relative: bool = False,
    min_group_size: int = 5,
):
    """Map each feature to its within-date cross-sectional rank in [-1, 1].

    Point-in-time safe (only same-date rows) and robust to the heavy tails in raw
    factor values. Single-name (or empty) dates collapse to 0.
    """
    out = panel.copy()
    g = out.groupby("date")

    def _norm(rank_s, count_s):
        denom = (count_s - 1).clip(lower=1)
        return np.where(count_s > 1, (rank_s - 1) / denom * 2 - 1, 0.0)

    for c in cols:
        r = g[c].rank(method="average")
        n = g[c].transform("count")
        out[c] = _norm(r, n)
        if not industry_relative or c not in INDUSTRY_RELATIVE_FEATURES:
            continue

        sector_groups = out.groupby(["date", "sector"], dropna=False)
        sr = sector_groups[c].rank(method="average")
        sn = sector_groups[c].transform("count")
        sector_norm = _norm(sr, sn)
        use_sector = out["sector"].notna() & (sn >= min_group_size)
        out[c] = np.where(use_sector, sector_norm, out[c])

        industry_groups = out.groupby(["date", "industry"], dropna=False)
        ir = industry_groups[c].rank(method="average")
        inn = industry_groups[c].transform("count")
        industry_norm = _norm(ir, inn)
        use_industry = out["industry"].notna() & (inn >= min_group_size)
        out[c] = np.where(use_industry, industry_norm, out[c])
    return out


def apply_target_modes(
    panel,
    n_buckets: int = 5,
    market_horizon_returns: dict | None = None,
    sector_min_group_size: int = 5,
):
    """Add per-horizon training-target variants computed cross-sectionally per date.

    For each horizon the demeaned forward return `r_{h}` (still the SCORING target)
    gets five trainable transforms, all relabelings of the same future info (no
    feature leak, same class as the existing median-demean):
      y_{h}_return        = r_{h} (raw demeaned log return; outlier-heavy)
      y_{h}_rank          = within-date percentile of r_{h} in (0,1]
      y_{h}_quantile      = within-date equal-count bucket index 0..n_buckets-1
      y_{h}_sector_return = r_{h} minus within-(date, sector) median, with a
                            universe-demean fallback when the sector group has
                            fewer than `sector_min_group_size` names (test-4 §4).
      y_{h}_beta_resid    = r_{h} − beta_252 × market_r_h, the alpha-residual
                            target (test-4 §4). Falls back to NaN where either
                            beta or the horizon-aggregated market return is
                            missing — those rows are dropped at fit time.

    Scoring stays Spearman IC against the realized r_{h}, so target modes are
    apples-to-apples comparable: a sector-relative target trained model is judged
    on universe-relative ranking, the same metric.
    """
    import pandas as pd

    out = panel.copy()
    has_sector = "sector" in out.columns
    has_beta = "beta_252d" in out.columns
    mhr = market_horizon_returns or {}

    for h in HORIZONS:
        r, m = f"r_{h}", f"mask_{h}"
        valid = out[r].where(out[m].astype(bool))     # NaN where masked
        grp = valid.groupby(out["date"])
        out[f"y_{h}_return"] = out[r]
        out[f"y_{h}_rank"] = grp.rank(pct=True)

        def _bucket(s):
            if s.notna().sum() < n_buckets:
                return pd.Series(np.nan, index=s.index)
            return pd.qcut(s, n_buckets, labels=False, duplicates="drop").astype(float)

        out[f"y_{h}_quantile"] = grp.transform(_bucket)

        # --- Sector-relative target (test-4 phase 4) ---
        # Since `valid` is already universe-demeaned, subtracting the within-
        # (date, sector) median of `valid` is mathematically identical to
        # subtracting the within-sector median of the raw returns — the cancel
        # eats the universe median. Groups below the size threshold fall back
        # to the universe-demeaned target.
        if has_sector:
            sec_grp = valid.groupby([out["date"], out["sector"]])
            sec_med = sec_grp.transform("median")
            sec_count = sec_grp.transform("count")
            use_sector = out["sector"].notna() & (sec_count >= sector_min_group_size)
            out[f"y_{h}_sector_return"] = np.where(use_sector, valid - sec_med, valid)
        else:
            out[f"y_{h}_sector_return"] = valid

        # --- Beta-residual target (test-4 phase 4) ---
        # If beta or market_r_h is missing for a row we deliberately emit NaN
        # rather than passing through `valid` — silently substituting the
        # universe-demean target would corrupt the ICIR comparison the user
        # asked for. Fit-time row filtering drops those rows; if the dict is
        # entirely empty for this horizon, the trainer will surface that as an
        # empty-training-set error, which is the correct loud failure.
        if has_beta:
            mkt_r = out["date"].map(mhr.get(h, {})).astype(float)
            beta = out["beta_252d"].astype(float)
            out[f"y_{h}_beta_resid"] = valid - beta * mkt_r
        else:
            out[f"y_{h}_beta_resid"] = pd.Series(np.nan, index=out.index)
    return out


def prepare_panel(
    frames: list[TickerFrame],
    grid: list,
    n_buckets: int = 5,
    max_stale_days: int = 7,
    rank_cols: list[str] | None = None,
    industry_relative: bool = False,
    min_group_size: int = 5,
):
    """Full pipeline: assemble → demean target → rank-normalize features → targets."""
    market_returns = build_universe_return_map(frames)
    panel = assemble_panel(
        frames,
        grid,
        max_stale_days=max_stale_days,
        market_returns=market_returns,
    )
    if panel.empty:
        return panel
    medians = cross_sectional_medians(frames)
    panel = demean_cross_sectional(panel, medians)
    panel = add_industry_neutral_momentum(panel, min_group_size=min_group_size)
    panel = rank_normalize_features(
        panel,
        cols=rank_cols or FEATURE_COLS,
        industry_relative=industry_relative,
        min_group_size=min_group_size,
    )
    market_horizon_returns = build_market_horizon_returns(market_returns, grid)
    panel = apply_target_modes(
        panel, n_buckets, market_horizon_returns=market_horizon_returns
    )
    return panel


# =============================================================
# Walk-forward
# =============================================================


def walk_forward_folds(grid_dates: list, min_train_months: int, embargo_steps: int):
    """Yield (test_date, train_cutoff_date) for an expanding-window sweep.

    The embargo drops `embargo_steps` month-ends between the train cutoff and the
    test date so a training sample's H-day label window ends on/before the test
    date — it cannot overlap the [test, test+H] window being predicted.
    """
    folds = []
    for i in range(min_train_months + embargo_steps, len(grid_dates)):
        folds.append((grid_dates[i], grid_dates[i - embargo_steps]))
    return folds


def fit_lgbm_model(
    train_df,
    target_col: str,
    cfg: LGBMConfig,
    seed: int,
    shuffle: bool = False,
    feature_cols: list[str] | None = None,
):
    """Fit one LightGBM regressor on prepared panel rows and return the model.

    `feature_cols` defaults to the production FEATURE_COLS so existing callers
    (inference, compare_transformer_gbm) keep their behavior. Phase-1 packs flow
    in via the explicit list.
    """
    import lightgbm as lgb

    cols = feature_cols if feature_cols is not None else FEATURE_COLS
    # Pass DataFrames (not bare arrays) so feature names flow into LightGBM and
    # sklearn doesn't warn at predict time.
    X_tr = train_df[cols]
    y_tr = train_df[target_col].to_numpy(dtype=float)
    if shuffle:  # destroy feature->label link, preserve marginal => no-signal null
        y_tr = y_tr[np.random.default_rng(seed).permutation(len(y_tr))]
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        subsample_freq=1,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        n_jobs=cfg.n_jobs,
        random_state=seed,
        verbose=-1,
    )
    model.fit(X_tr, y_tr)
    return model


def _fit_predict(
    train_df,
    test_df,
    target_col: str,
    cfg: LGBMConfig,
    seed: int,
    shuffle: bool,
    feature_cols: list[str] | None = None,
) -> np.ndarray:
    cols = feature_cols if feature_cols is not None else FEATURE_COLS
    model = fit_lgbm_model(
        train_df, target_col, cfg, seed=seed, shuffle=shuffle, feature_cols=cols
    )
    return model.predict(test_df[cols])


def _target_col(horizon: str, target_mode: str) -> str:
    """Training-target column: raw return needs no precomputed column."""
    return f"r_{horizon}" if target_mode == "return" else f"y_{horizon}_{target_mode}"


def walk_forward_ic(
    panel,
    horizon: str = "1M",
    lgb_cfg: LGBMConfig | None = None,
    wf_cfg: WalkForwardConfig | None = None,
    seed: int = 1337,
    shuffle: bool = False,
    target_mode: str = "return",
    log=lambda *_: None,
    feature_cols: list[str] | None = None,
) -> dict:
    """Expanding-window walk-forward; return summary + per-fold rank-IC rows.

    Trains on `target_mode` (return/rank/quantile) but always SCORES rank-IC against
    the realized demeaned return `r_{horizon}`, so modes are directly comparable.
    """
    import pandas as pd

    lgb_cfg = lgb_cfg or LGBMConfig()
    wf_cfg = wf_cfg or WalkForwardConfig()
    r_col, m_col = f"r_{horizon}", f"mask_{horizon}"
    t_col = _target_col(horizon, target_mode)
    embargo_steps = max(1, math.ceil(HORIZON_TRADING_DAYS[horizon] / 21))

    grid_dates = sorted(panel["date"].unique())
    folds = walk_forward_folds(grid_dates, wf_cfg.min_train_months, embargo_steps)

    fold_rows: list[dict] = []
    for fi, (test_date, cutoff) in enumerate(folds):
        train = panel[(panel["date"] <= cutoff) & panel[m_col] & panel[t_col].notna()]
        if wf_cfg.max_train_months is not None:
            lower_idx = max(0, grid_dates.index(cutoff) - wf_cfg.max_train_months)
            train = train[train["date"] >= grid_dates[lower_idx]]
        test = panel[(panel["date"] == test_date) & panel[m_col]]
        if test.shape[0] < wf_cfg.min_names or train.empty:
            continue

        preds = _fit_predict(
            train, test, t_col, lgb_cfg, seed + fi, shuffle, feature_cols=feature_cols
        )
        ic = pd.Series(preds).corr(pd.Series(test[r_col].to_numpy(dtype=float)), method="spearman")
        if ic != ic:  # NaN (zero-variance cross-section)
            continue
        fold_rows.append({"date": test_date, "ic": float(ic), "n_test": int(test.shape[0]),
                          "n_train": int(train.shape[0])})
        log(
            f"  fold {test_date}  ic {ic:+.4f}  "
            f"n_test={test.shape[0]:>4d}  n_train={train.shape[0]}"
        )

    return {"summary": summarize([r["ic"] for r in fold_rows]), "folds": fold_rows}


def summarize(ics: list[float]) -> dict:
    """Across-fold IC statistics: mean, ICIR (mean/std), t-stat, hit rate."""
    a = np.asarray(ics, dtype=float)
    n = a.size
    if n == 0:
        return {"n_folds": 0, "mean_ic": float("nan"), "std_ic": float("nan"),
                "icir": float("nan"), "t_stat": float("nan"), "hit_rate": float("nan")}
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    icir = mean / std if std > 0 else float("nan")           # information ratio of IC
    t_stat = icir * math.sqrt(n) if std > 0 else float("nan")  # significance across folds
    return {"n_folds": n, "mean_ic": mean, "std_ic": std,
            "icir": icir, "t_stat": t_stat, "hit_rate": float((a > 0).mean())}


def block_bootstrap_summary(
    ics: list[float],
    block_size: int,
    reps: int = 2000,
    seed: int = 1337,
) -> dict:
    """Moving-block bootstrap for overlapping horizon IC folds.

    Monthly folds are autocorrelated when labels overlap (especially 6M/1Y). This
    resamples contiguous blocks to estimate a confidence interval for the mean IC
    and a centered-null two-sided p-value for mean_ic != 0.
    """
    a = np.asarray([v for v in ics if v == v], dtype=float)
    n = a.size
    if n == 0:
        return {
            "n_folds": 0, "block_size": block_size, "reps": reps,
            "mean_ic": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
            "p_value": float("nan"), "effective_blocks": 0.0, "t_block": float("nan"),
        }

    block_size = max(1, min(int(block_size), n))
    reps = max(0, int(reps))
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    effective_blocks = n / block_size
    t_block = mean / std * math.sqrt(effective_blocks) if std > 0 else float("nan")

    if reps == 0:
        return {
            "n_folds": n, "block_size": block_size, "reps": reps, "mean_ic": mean,
            "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan"),
            "effective_blocks": effective_blocks, "t_block": t_block,
        }

    rng = np.random.default_rng(seed)
    starts = np.arange(0, n - block_size + 1)
    centered = a - mean
    boot_means = np.empty(reps, dtype=float)
    null_means = np.empty(reps, dtype=float)
    n_blocks = math.ceil(n / block_size)
    for i in range(reps):
        idx = np.concatenate([
            np.arange(s, s + block_size)
            for s in rng.choice(starts, size=n_blocks, replace=True)
        ])[:n]
        boot_means[i] = float(a[idx].mean())
        null_means[i] = float(centered[idx].mean())

    p_value = (1.0 + float(np.sum(np.abs(null_means) >= abs(mean)))) / (reps + 1.0)
    return {
        "n_folds": n,
        "block_size": block_size,
        "reps": reps,
        "mean_ic": mean,
        "ci_low": float(np.percentile(boot_means, 2.5)),
        "ci_high": float(np.percentile(boot_means, 97.5)),
        "p_value": p_value,
        "effective_blocks": effective_blocks,
        "t_block": t_block,
    }


def single_split_ic(
    panel,
    horizon: str,
    fit_cutoff,
    holdout_start,
    lgb_cfg: LGBMConfig | None = None,
    target_mode: str = "return",
    seed: int = 1337,
    min_names: int = 30,
    feature_cols: list[str] | None = None,
) -> dict:
    """One fit on `date < fit_cutoff`, scored per holdout cross-section.

    For an apples-to-apples head-to-head vs a single-trained transformer: identical
    fitting boundary and holdout window, no monthly refit. Returns the same summary
    shape as `walk_forward_ic` (mean IC over holdout dates, ICIR, t, hit).
    """
    lgb_cfg = lgb_cfg or LGBMConfig()
    r_col, m_col = f"r_{horizon}", f"mask_{horizon}"
    t_col = _target_col(horizon, target_mode)
    train = panel[(panel["date"] < fit_cutoff) & panel[m_col] & panel[t_col].notna()]
    holdout = panel[(panel["date"] >= holdout_start) & panel[m_col]]
    if train.empty or holdout.empty:
        return {"summary": summarize([]), "folds": []}

    preds = _fit_predict(
        train, holdout, t_col, lgb_cfg, seed, shuffle=False, feature_cols=feature_cols
    )
    scored = holdout[["date", r_col]].copy()
    scored["pred"] = preds
    fold_rows = []
    for d, g in scored.groupby("date"):
        if len(g) < min_names:
            continue
        ic = g["pred"].corr(g[r_col], method="spearman")
        if ic == ic:
            fold_rows.append({"date": d, "ic": float(ic), "n_test": int(len(g))})
    return {"summary": summarize([r["ic"] for r in fold_rows]), "folds": fold_rows}


# =============================================================
# CLI shell (DB)
# =============================================================


def _print_summary(tag: str, s: dict) -> None:
    print(
        f"{tag:<5} n_folds={s['n_folds']:>3d}  mean_ic={s['mean_ic']:+.4f}  "
        f"icir={s['icir']:+.3f}  t={s['t_stat']:+.2f}  hit={s['hit_rate']:.3f}"
    )


def _print_bootstrap(tag: str, s: dict) -> None:
    print(
        f"{tag:<5} block={s['block_size']:>2d}  eff_blocks={s['effective_blocks']:.1f}  "
        f"ci95=[{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]  "
        f"t_block={s['t_block']:+.2f}  p={s['p_value']:.4f}"
    )


def _compose_feature_cols(args) -> list[str]:
    """Build the active feature list for this run from CLI pack flags.

    Order matters only for logging / model debugging; LightGBM is order-agnostic.
    Production default (no flags) returns FEATURE_COLS unchanged so smoke runs
    and the default reporting line still anchor the comparison.
    """
    cols: list[str] = list(FEATURE_COLS)
    if args.with_valuation:
        cols += list(VALUATION_FEATURES)
    if args.with_quality:
        cols += list(QUALITY_FEATURES)
    if args.with_residual_mom:
        cols += list(RESIDUAL_MOM_FEATURES)
    if args.with_earnings_reaction:
        cols += list(EARNINGS_REACTION_FEATURES)
    if args.with_analyst_revisions:
        cols += list(ANALYST_REVISION_FEATURES)
    if args.with_estimate_surprise:
        cols += list(ESTIMATE_SURPRISE_FEATURES)
    if args.with_forward_valuation:
        cols += list(FORWARD_VALUATION_FEATURES)
    # De-dupe while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


async def run(args) -> None:
    lgb_cfg = LGBMConfig()
    wf_cfg = WalkForwardConfig(
        min_train_months=args.min_train_months,
        max_train_months=args.max_train_months,
        min_names=args.min_names,
    )
    feature_cols = _compose_feature_cols(args)
    extras = [c for c in feature_cols if c not in FEATURE_COLS]

    async with pool_context() as pool:
        frames = await load_frames_cached(
            pool, symbols=args.symbols, refresh=args.refresh_cache
        )
        if not frames:
            raise SystemExit("no active tickers / frames loaded")
        grid = build_calendar_grid(frames)
        print(f"loaded {len(frames)} tickers; building monthly grid ({len(grid)} months) ...")
        if extras:
            print(f"feature packs active: +{', +'.join(extras)} (total={len(feature_cols)})")
        if args.industry_relative:
            print("industry-relative feature normalization: ON")

        panel = prepare_panel(
            frames,
            grid,
            n_buckets=args.n_buckets,
            rank_cols=feature_cols,
            industry_relative=args.industry_relative,
        )
        if panel.empty:
            raise SystemExit("empty panel (not enough history?)")
        dates = sorted(panel["date"].unique())
        print(
            f"panel: rows={len(panel)}  dates={len(dates)}  "
            f"range={dates[0]}..{dates[-1]}  tickers={panel['ticker_id'].nunique()}"
        )

        real = walk_forward_ic(panel, args.horizon, lgb_cfg, wf_cfg,
                               seed=args.seed, shuffle=False, target_mode=args.target,
                               log=print if args.verbose else (lambda *_: None),
                               feature_cols=feature_cols)
        print(f"\n--- {args.horizon} cross-sectional rank-IC "
              f"(expanding walk-forward, target={args.target}) ---")
        _print_summary("REAL", real["summary"])
        if args.block_bootstrap_reps > 0:
            block_size = args.block_size or max(
                1, math.ceil(HORIZON_TRADING_DAYS[args.horizon] / 21)
            )
            boot = block_bootstrap_summary(
                [r["ic"] for r in real["folds"]],
                block_size=block_size,
                reps=args.block_bootstrap_reps,
                seed=args.seed,
            )
            _print_bootstrap("BOOT", boot)

        if args.null_reps > 0:
            print(f"\nrunning {args.null_reps} shuffle-null reps ...")
            null_means = []
            for rep in range(args.null_reps):
                res = walk_forward_ic(panel, args.horizon, lgb_cfg, wf_cfg,
                                      seed=args.seed + 1000 * (rep + 1), shuffle=True,
                                      target_mode=args.target, feature_cols=feature_cols)
                m = res["summary"]["mean_ic"]
                null_means.append(m)
                print(f"  null rep {rep}: mean_ic={m:+.4f}")
            nm = np.asarray(null_means, dtype=float)
            null_mu, null_sd = float(nm.mean()), float(nm.std(ddof=1)) if nm.size > 1 else 0.0
            z = (real["summary"]["mean_ic"] - null_mu) / null_sd if null_sd > 0 else float("nan")
            print(f"\nNULL  reps={nm.size}  mean_ic={null_mu:+.4f} ± {null_sd:.4f}")
            print(f"VERDICT  real mean_ic {real['summary']['mean_ic']:+.4f} vs null "
                  f"{null_mu:+.4f}±{null_sd:.4f}  =>  z={z:+.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-sectional LightGBM walk-forward baseline")
    p.add_argument(
        "--horizon", default="1M", choices=list(HORIZONS),
        help="forward horizon to predict/score (1M is the only one with enough OOS dates)",
    )
    p.add_argument("--min-train-months", type=int, default=36)
    p.add_argument("--max-train-months", type=int, default=None,
                   help="rolling window length in months (default: expanding)")
    p.add_argument("--min-names", type=int, default=30, help="skip thinner test cross-sections")
    p.add_argument("--target", default="return",
                   choices=["return", "rank", "quantile", "sector_return", "beta_resid"],
                   help="training target transform (scoring is always vs realized "
                        "universe-demeaned return; sector_return / beta_resid are "
                        "test-4 phase-4 alpha-residual modes)")
    p.add_argument(
        "--n-buckets", type=int, default=5,
        help="equal-count buckets for --target quantile",
    )
    p.add_argument(
        "--null-reps", type=int, default=0,
        help="shuffle-null repetitions for the no-signal band",
    )
    p.add_argument("--block-bootstrap-reps", type=int, default=0,
                   help="moving-block bootstrap reps for overlap-aware IC significance")
    p.add_argument("--block-size", type=int, default=None,
                   help="block length in monthly folds (default = horizon months)")
    p.add_argument("--symbols", nargs="*", help="restrict to these symbols")
    p.add_argument("--refresh-cache", action="store_true",
                   help="re-pull frames from Supabase and overwrite the local "
                        "frame cache (do this after ingesting new data)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--verbose", action="store_true", help="print every fold")
    # --- Opt-in feature packs (test-3 + test-4). Production default unchanged. ---
    p.add_argument("--with-valuation", action="store_true",
                   help="add valuation pack (earnings_yield, book_to_market, ...)")
    p.add_argument("--with-quality", action="store_true",
                   help="add quality pack (roe_ttm, ttm margins, 4Q stability)")
    p.add_argument("--with-residual-mom", action="store_true",
                   help="add residual / structural momentum pack")
    p.add_argument("--with-earnings-reaction", action="store_true",
                   help="add filing-drift / surprise reaction pack")
    p.add_argument("--with-analyst-revisions", action="store_true",
                   help="add LSEG analyst-revision pack (rec level + 30/90d revisions, "
                        "price-target revision)")
    p.add_argument("--with-estimate-surprise", action="store_true",
                   help="add LSEG revenue-surprise pack")
    p.add_argument("--with-forward-valuation", action="store_true",
                   help="add LSEG forward-valuation pack (forward earnings/ebitda yield, "
                        "price-target upside)")
    p.add_argument("--industry-relative", action="store_true",
                   help="rank-normalize price/fundamental/valuation/quality "
                        "features within (date, industry) instead of universe-wide")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
