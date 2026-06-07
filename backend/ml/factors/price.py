"""Per-ticker price/volume factor features (computed from the ticker's own series)."""

from __future__ import annotations

import math

import numpy as np

from backend.ml.factors.util import _log_ratio


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

    # --- Lottery / idiosyncratic-vol pack (Experiment 1) ---
    # max_ret_21d: largest single-day SIMPLE return over the last ~month (Bali 2011
    # lottery proxy). daily is log returns; expm1(max log) = max simple return.
    last21 = daily[-21:]
    last21 = last21[~np.isnan(last21)]
    feats["max_ret_21d"] = float(np.expm1(last21.max())) if last21.size else 0.0
    # idio_vol: residual daily-return vol vs the universe (stock − beta·market),
    # reusing the aligned stock/market series + beta built above. Falls back to
    # total daily vol when the market series is too short (keeps it finite).
    if len(stock_daily) >= 60:
        resid = np.asarray(stock_daily, dtype=float) - feats["beta_252d"] * np.asarray(market_daily, dtype=float)
        feats["idio_vol"] = float(resid.std())
    else:
        feats["idio_vol"] = _std(daily[-120:])

    hi, lo = np.nanmax(window), np.nanmin(window)
    feats["dist_high_252"] = _log_ratio(P, hi)
    feats["dist_low_252"] = _log_ratio(P, lo)
    feats["ma_gap_50"] = _log_ratio(P, float(np.nanmean(window[-50:])))
    feats["ma_gap_200"] = _log_ratio(P, float(np.nanmean(window[-200:])))

    v = np.array(volume[pos - 119 : pos + 1], dtype=float)
    v_recent = float(v[-20:].mean()) if v[-20:].size else 0.0
    v_long = float(v.mean()) if v.size else 0.0
    feats["vol_trend"] = math.log(v_recent / v_long) if v_recent > 0 and v_long > 0 else 0.0

    # --- Microstructure / higher-moment pack (decorrelated 6M candidate) ---
    # Orthogonal axes to the 1st/2nd-moment book above: return skewness (3rd moment),
    # downside/total vol asymmetry, illiquidity (price impact), turnover, and a Kaufman
    # efficiency ratio flagging trend vs consolidation (the sideways-bias control the
    # diagnostic conditions on). All from `daily`/`window`/`v` already built above;
    # NaN/zero-safe so every name stays finite.
    d120 = daily[-120:]
    d120 = d120[~np.isnan(d120)]
    if d120.size >= 20:
        sd = float(d120.std())
        if sd > 1e-12:
            mu = float(d120.mean())
            feats["ret_skew_120d"] = float(np.mean(((d120 - mu) / sd) ** 3))
            downside = float(np.sqrt(np.mean(np.minimum(d120, 0.0) ** 2)))
            feats["downside_vol_ratio_120d"] = downside / sd
        else:
            feats["ret_skew_120d"] = 0.0
            feats["downside_vol_ratio_120d"] = 0.0
        path = float(np.sum(np.abs(d120)))
        feats["efficiency_ratio_120d"] = abs(float(d120.sum())) / path if path > 1e-12 else 0.0
    else:
        feats["ret_skew_120d"] = 0.0
        feats["downside_vol_ratio_120d"] = 0.0
        feats["efficiency_ratio_120d"] = 0.0

    # Amihud illiquidity: mean |daily ret| / dollar-volume over ~60d (×1e6 for
    # readability; rank-normalization makes the absolute scale irrelevant). Turnover:
    # mean share volume / shares outstanding over ~60d. Same day alignment (pos-59..pos).
    absret60 = np.abs(daily[-60:])
    px60 = window[-60:]
    vol60 = v[-60:] if v.size >= 60 else v
    m = min(absret60.size, px60.size, vol60.size)
    if m > 0:
        ar = absret60[-m:]
        dv = px60[-m:] * vol60[-m:]
        mask = np.isfinite(ar) & np.isfinite(dv) & (dv > 0)
        feats["amihud_illiq_60d"] = float(np.mean(ar[mask] / dv[mask]) * 1e6) if mask.any() else 0.0
    else:
        feats["amihud_illiq_60d"] = 0.0
    vv = vol60[np.isfinite(vol60)]
    if shares_outstanding and shares_outstanding > 0 and vv.size:
        feats["turnover_60d"] = float(vv.mean() / shares_outstanding)
    else:
        feats["turnover_60d"] = 0.0

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
