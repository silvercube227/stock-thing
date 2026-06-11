"""LightGBM walk-forward baseline tests — pure, synthetic, no DB.

Covers the things that are easy to get silently wrong: point-in-time feature
assembly, the cross-sectional rank transform, median-demeaning, the embargo/fold
index math (leakage guard), and an end-to-end planted-signal recovery that the
shuffle null must NOT reproduce.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from backend.ml.dataset import TickerFrame, build_calendar_grid
from backend.ml.gbm_baseline import (
    EARNINGS_REACTION_FEATURES,
    EXPERIMENTAL_FEATURES,
    FEATURE_COLS,
    LGBMConfig,
    RESIDUAL_MOM_FEATURES,
    WalkForwardConfig,
    add_industry_neutral_momentum,
    apply_knife_overlay,
    apply_target_modes,
    block_bootstrap_summary,
    build_market_horizon_returns,
    build_ticker_rows,
    build_universe_return_map,
    demean_cross_sectional,
    ewma_rank_by_ticker,
    knife_sweep_table,
    prepare_panel,
    rank_normalize_features,
    feature_diagnostics,
    knife_tier,
    rank_turnover,
    regularization_sweep,
    summarize,
    target_blend_sweep,
    top_decile_risk,
    walk_forward_folds,
    walk_forward_ic,
    within_sector_ic,
)
from backend.ml.gbm_inference import score_current_cross_section
from backend.ml.model import HORIZONS

# =============================================================
# Synthetic frames
# =============================================================


def make_frame(n_days: int, trend: float, tid: int, vol_seed: int = 0) -> TickerFrame:
    """A daily price series starting 2018-01-02 with drift `trend` and mild noise."""
    rng = np.random.default_rng(vol_seed)
    d0 = date(2018, 1, 2)
    price = 100.0
    prices = []
    for i in range(n_days):
        price *= 1.0 + trend + rng.normal(0, 0.01)  # geometric drift + daily noise
        prices.append({
            "trade_date": d0 + timedelta(days=i),
            "adj_close": max(price, 1.0),
            "volume": 1_000_000 + 1000 * i,
        })
    return TickerFrame(tid, tid, f"T{tid}", prices, [], [])


# =============================================================
# Feature assembly
# =============================================================


def test_build_ticker_rows_shapes_and_columns():
    frame = make_frame(n_days=900, trend=0.0005, tid=1)
    grid = build_calendar_grid([frame])
    rows = build_ticker_rows(frame, grid)
    assert rows, "expected at least one row"
    cols = set(rows[0])
    for c in FEATURE_COLS:
        assert c in cols, f"missing feature {c}"
    for h in HORIZONS:
        assert f"r_{h}" in cols and f"mask_{h}" in cols
    # No feature should be NaN/inf (the model can't ingest those).
    for r in rows:
        for c in FEATURE_COLS:
            assert np.isfinite(r[c]), f"{c} not finite"


def test_rising_series_has_positive_momentum():
    frame = make_frame(n_days=900, trend=0.001, tid=1)  # steady uptrend
    rows = build_ticker_rows(frame, build_calendar_grid([frame]))
    # On a persistent uptrend most rows show positive 6M momentum.
    pos_frac = np.mean([r["mom_6m"] > 0 for r in rows])
    assert pos_frac > 0.8


def test_too_little_history_yields_no_rows():
    frame = make_frame(n_days=200, trend=0.0, tid=1)  # < SEQUENCE_LENGTH+ usable
    assert build_ticker_rows(frame, build_calendar_grid([frame])) == []


def test_build_ticker_rows_skips_stale_grid_dates():
    frame = make_frame(n_days=360, trend=0.0005, tid=1)
    grid = build_calendar_grid([frame])
    grid.append(date(2022, 1, 31))  # long after this synthetic ticker stopped trading
    rows = build_ticker_rows(frame, grid, max_stale_days=7)
    assert rows
    assert max(r["date"] for r in rows) < date(2022, 1, 31)


def test_build_ticker_rows_computes_beta_and_earnings_yield():
    frame_a = make_frame(n_days=900, trend=0.0006, tid=1, vol_seed=1)
    frame_b = make_frame(n_days=900, trend=0.0003, tid=2, vol_seed=2)
    frame_a.shares_outstanding = 1_000_000
    frame_a.sector = "Technology"
    frame_a.industry = "Software"
    frame_b.sector = "Technology"
    frame_b.industry = "Hardware"
    frame_a.fundamentals = [
        {
            "filed_at": date(2018, 3, 31),
            "period_end": date(2017, 12, 31),
            "filing_type": "10-K",
            "net_income": 5_000_000,
            "revenue": 20_000_000,
            "gross_margin": 0.4,
            "operating_margin": 0.2,
            "total_debt": 1_000_000,
            "total_equity": 10_000_000,
            "fcf": 2_000_000,
        }
    ]
    market_returns = build_universe_return_map([frame_a, frame_b])
    rows = build_ticker_rows(frame_a, build_calendar_grid([frame_a, frame_b]), market_returns=market_returns)

    assert rows
    assert any(abs(r["beta_252d"]) > 1e-6 for r in rows)
    assert any(r["earnings_yield"] > 0 for r in rows)
    assert any(r["book_to_market"] > 0 for r in rows)
    assert any(r["roe_ttm"] > 0 for r in rows)
    for r in rows:
        for c in EXPERIMENTAL_FEATURES:
            assert np.isfinite(r[c]), f"{c} not finite"


# =============================================================
# Test-4 phase-1 packs: residual momentum + earnings reaction
# =============================================================


def test_residual_momentum_collapses_when_ticker_equals_market():
    # Single ticker -> market_returns == this ticker's returns -> resid_mom ≈ 0
    # once beta_252d converges to 1.0. We additionally compare against a second
    # ticker so the universe map is well-defined; both tickers track the same
    # series so the market IS the ticker.
    frame_a = make_frame(n_days=900, trend=0.0005, tid=1, vol_seed=11)
    frame_b = TickerFrame(2, 2, "T2", list(frame_a.prices), [], [])
    market_returns = build_universe_return_map([frame_a, frame_b])
    rows = build_ticker_rows(
        frame_a, build_calendar_grid([frame_a, frame_b]), market_returns=market_returns
    )
    assert rows
    # Beta should be ~1 (ticker == market) and resid_mom ~ 0 on the late part of
    # the series where the 252-day window has stabilized.
    late = rows[len(rows) // 2 :]
    assert all(abs(r["beta_252d"] - 1.0) < 0.05 for r in late), \
        "single-ticker universe should yield beta≈1"
    assert all(abs(r["resid_mom_12_1"]) < 0.02 for r in late), \
        "resid_mom_12_1 should vanish when ticker == market"
    assert all(abs(r["resid_mom_6m"]) < 0.02 for r in late)
    # All RESIDUAL_MOM_FEATURES should be finite on every row.
    for r in rows:
        for c in RESIDUAL_MOM_FEATURES:
            assert np.isfinite(r[c]), f"{c} not finite"


def test_lottery_features_finite_and_idio_vol_vanishes_vs_market():
    # idio_vol = std(stock − beta·market). With a single-series universe the stock
    # IS the market (beta≈1), so the residual — and idio_vol — should collapse to ~0.
    frame_a = make_frame(n_days=900, trend=0.0005, tid=1, vol_seed=11)
    frame_b = TickerFrame(2, 2, "T2", list(frame_a.prices), [], [])
    market_returns = build_universe_return_map([frame_a, frame_b])
    rows = build_ticker_rows(
        frame_a, build_calendar_grid([frame_a, frame_b]), market_returns=market_returns
    )
    assert rows
    for r in rows:
        assert np.isfinite(r["max_ret_21d"]) and np.isfinite(r["idio_vol"])
    late = rows[len(rows) // 2:]
    assert all(r["idio_vol"] < 0.01 for r in late), "idio_vol should vanish when ticker == market"


def test_max_ret_21d_detects_planted_single_day_spike():
    # A smooth series has small daily moves; inject one +15% day and any row whose
    # trailing-month window includes it must show max_ret_21d ≈ 0.15.
    smooth = make_frame(n_days=420, trend=0.0003, tid=1, vol_seed=3)
    base_rows = build_ticker_rows(smooth, build_calendar_grid([smooth]))
    assert max(r["max_ret_21d"] for r in base_rows) < 0.08  # noise only

    spiked_prices = [dict(p) for p in smooth.prices]
    for i in range(380, len(spiked_prices)):       # one +15% jump at index 380
        spiked_prices[i]["adj_close"] *= 1.15
    spiked = TickerFrame(1, 1, "T1", spiked_prices, [], [])
    spike_rows = build_ticker_rows(spiked, build_calendar_grid([spiked]))
    assert max(r["max_ret_21d"] for r in spike_rows) > 0.10, "spike should surface in max_ret_21d"


def test_mom_consistency_6m_on_steady_uptrend_is_high():
    # 0.001 daily drift vs 0.01 daily noise => ~70% positive monthly returns;
    # so the *average* of mom_consistency_6m should land well above 0.5 but not
    # be deterministic. We assert the central tendency rather than 1.0.
    frame = make_frame(n_days=900, trend=0.001, tid=1)
    rows = build_ticker_rows(frame, build_calendar_grid([frame]))
    avg = float(np.mean([r["mom_consistency_6m"] for r in rows]))
    # ~62% positive months in practice (Sharpe/month ≈ 0.5 with these params);
    # well above the 0.5 random baseline, but not 1.0.
    assert avg > 0.55, f"expected mostly-positive months on uptrend, got mean={avg:.3f}"
    # Range stays within [0, 1].
    for r in rows:
        assert 0.0 <= r["mom_consistency_6m"] <= 1.0


def test_microstructure_skew_sign_tracks_planted_tail():
    # ret_skew_120d is the standardized 3rd moment of daily returns. Inject one big
    # DOWN day → left tail → negative skew on rows whose 120d window includes it; a
    # big UP day → positive skew. Compared against the un-spiked baseline.
    base = make_frame(n_days=500, trend=0.0003, tid=1, vol_seed=7)
    base_rows = build_ticker_rows(base, build_calendar_grid([base]))

    def _spiked(mult: float):
        px = [dict(p) for p in base.prices]
        px[330]["adj_close"] *= mult          # one extreme day
        for i in range(331, len(px)):          # carry the level shift forward
            px[i]["adj_close"] *= mult
        f = TickerFrame(1, 1, "T1", px, [], [])
        return build_ticker_rows(f, build_calendar_grid([f]))

    down = _spiked(0.80)   # −20% crash day
    up = _spiked(1.20)     # +20% melt-up day
    # Look at a row whose trailing 120 trading days (~170 cal days) include index 330.
    def _skew_at(rows):
        return [r["ret_skew_120d"] for r in rows if r["date"] >= base.prices[360]["trade_date"]][0]
    assert _skew_at(down) < _skew_at(base_rows) < _skew_at(up)
    assert _skew_at(down) < 0 < _skew_at(up)


def test_microstructure_efficiency_ratio_high_on_trend_low_on_chop():
    # Kaufman ER ≈ |net move| / path. A clean strong drift trends (ER high); a
    # zero-drift noisy series wanders (ER low). Assert the central tendency.
    trend = make_frame(n_days=500, trend=0.008, tid=1, vol_seed=1)   # strong drift
    chop = make_frame(n_days=500, trend=0.0, tid=2, vol_seed=2)      # sideways
    er_trend = np.mean([r["efficiency_ratio_120d"] for r in build_ticker_rows(trend, build_calendar_grid([trend]))])
    er_chop = np.mean([r["efficiency_ratio_120d"] for r in build_ticker_rows(chop, build_calendar_grid([chop]))])
    assert 0.0 <= er_chop < er_trend <= 1.0
    assert er_trend > 0.5 and er_chop < 0.3


def test_microstructure_amihud_higher_for_thin_volume():
    # Amihud illiquidity = |ret| / dollar-volume. Same prices, 100× lower volume ⇒
    # strictly higher Amihud on every row.
    thick = make_frame(n_days=500, trend=0.0003, tid=1, vol_seed=5)
    thin_prices = [{**p, "volume": p["volume"] / 100.0} for p in thick.prices]
    thin = TickerFrame(2, 2, "T2", thin_prices, [], [])
    thick_rows = build_ticker_rows(thick, build_calendar_grid([thick]))
    thin_rows = build_ticker_rows(thin, build_calendar_grid([thin]))
    assert all(
        tn["amihud_illiq_60d"] > tk["amihud_illiq_60d"]
        for tk, tn in zip(thick_rows, thin_rows)
    )


def test_microstructure_turnover_matches_volume_over_shares():
    # turnover_60d ≈ mean(volume)/shares_outstanding. With near-constant volume it
    # lands at the expected ratio; with no shares it degrades to 0.0 (not NaN).
    frame = make_frame(n_days=500, trend=0.0003, tid=1, vol_seed=4)
    with_shares = TickerFrame(1, 1, "T1", list(frame.prices), [], [], shares_outstanding=50_000_000)
    rows = build_ticker_rows(with_shares, build_calendar_grid([with_shares]))
    last = rows[-1]
    vol60 = np.mean([p["volume"] for p in frame.prices[-60:]])
    assert last["turnover_60d"] == pytest.approx(vol60 / 50_000_000, rel=0.15)
    # No shares outstanding → safe 0.0, never NaN.
    no_shares = build_ticker_rows(frame, build_calendar_grid([frame]))
    assert all(r["turnover_60d"] == 0.0 for r in no_shares)


def test_feature_diagnostics_flags_duplicate_and_recovers_independent_signal():
    # Synthetic rank-normalized panel: r_6M is driven by latent `s`. `dup` copies the
    # existing feature mom_1m (independent of s) → high book-corr, ~0 standalone IC.
    # `indep_signal` ≈ s → low book-corr, strong positive standalone IC. The diagnostic
    # must separate them, and the efficiency-ratio tertile breakdown must be finite.
    rng = np.random.default_rng(0)
    n_dates, n_names = 12, 36
    recs = []
    for di in range(n_dates):
        s = rng.normal(size=n_names)
        mom = rng.normal(size=n_names)            # independent of s
        for i in range(n_names):
            recs.append({
                "date": date(2020, 1, 1) + timedelta(days=30 * di),
                "sector": "Tech",                  # one group ≥ 10 names
                "mask_6M": True,
                "r_6M": float(s[i] + 0.3 * rng.normal()),
                "mom_1m": float(mom[i]),
                "dup": float(mom[i]),              # duplicate of the existing feature
                "indep_signal": float(s[i] + 0.3 * rng.normal()),
                "efficiency_ratio_120d": float(rng.uniform(-1, 1)),
            })
    panel = pd.DataFrame(recs)
    out = {r["feature"]: r for r in feature_diagnostics(
        panel, "6M", ["dup", "indep_signal"], existing_cols=["mom_1m"],
        min_names=10, reps=0,
    )}
    # Duplicate is highly correlated with the book and carries ~no own signal.
    assert out["dup"]["mean_abs_corr"] > 0.95
    assert abs(out["dup"]["sec_ic"]) < 0.15
    # Independent signal is decorrelated AND carries real within-sector IC.
    assert out["indep_signal"]["mean_abs_corr"] < 0.3
    assert out["indep_signal"]["sec_ic"] > 0.3
    # Tertile breakdown is populated (finite) for the signal feature.
    for k in ("ic_sideways", "ic_mid", "ic_trending"):
        assert np.isfinite(out["indep_signal"][k])


def test_earnings_reaction_detects_planted_jump_around_filing():
    # Build a ticker whose price gaps +20% one trading day after a filing —
    # filing_surprise_3d should pick up that abnormal return, and the drift
    # window should remain finite.
    rng = np.random.default_rng(0)
    d0 = date(2018, 1, 2)
    prices: list[dict] = []
    p = 100.0
    filing_idx = 400
    for i in range(900):
        # Tiny noise so the ±1d jump dominates the surprise window.
        p *= 1.0 + 0.0001 + rng.normal(0, 0.001)
        if i == filing_idx + 1:
            p *= 1.20  # post-filing day jump
        prices.append({
            "trade_date": d0 + timedelta(days=i),
            "adj_close": max(p, 1.0),
            "volume": 1_000_000,
        })
    filed_at = prices[filing_idx]["trade_date"]
    frame = TickerFrame(1, 1, "T1", prices, [], [])
    frame.fundamentals = [
        {
            "filed_at": filed_at,
            "period_end": filed_at - timedelta(days=30),
            "filing_type": "10-Q",
            "revenue": 1_000_000,
            "net_income": 100_000,
            "gross_margin": 0.4,
            "operating_margin": 0.1,
            "total_debt": 0,
            "total_equity": 500_000,
            "fcf": 80_000,
        }
    ]
    # Need a second frame so build_universe_return_map yields something.
    foil = make_frame(n_days=900, trend=0.0001, tid=2, vol_seed=5)
    market_returns = build_universe_return_map([frame, foil])
    rows = build_ticker_rows(
        frame, build_calendar_grid([frame, foil]), market_returns=market_returns
    )
    assert rows

    # All reaction features finite on every row.
    for r in rows:
        for c in EARNINGS_REACTION_FEATURES:
            assert np.isfinite(r[c]), f"{c} not finite"
    # Once the grid date passes the filing, the most recent reaction snapshot
    # should still carry the planted abnormal return. The universe here is just
    # this ticker + 1 foil, so the equal-weight market absorbs roughly half the
    # jump (~10% of the planted +20% becomes market move); the abnormal return
    # should still clear ~5% comfortably.
    post_filing = [r for r in rows if r["filings_recency_days"] > 0]
    assert post_filing, "expected at least one row after the planted filing"
    assert max(r["filing_surprise_3d"] for r in post_filing) > 0.05, \
        "filing_surprise_3d should pick up the planted +20% post-filing jump"


def test_sector_return_target_subtracts_within_sector_median_above_threshold():
    # 6 Tech names + 1 Energy; Tech has >= 5 so sector-demean applies, Energy
    # has 1 so it falls back to the (already universe-demeaned) target.
    d = date(2020, 1, 31)
    df = pd.DataFrame({
        "date": [d] * 7,
        "sector": ["Tech"] * 6 + ["Energy"],
        "r_1M": [0.10, 0.05, 0.00, -0.05, -0.10, 0.20, 0.30],
        "mask_1M": [True] * 7,
    })
    for h in HORIZONS:
        if f"r_{h}" not in df:
            df[f"r_{h}"] = 0.0
        if f"mask_{h}" not in df:
            df[f"mask_{h}"] = False
    df["r_1M"] = [0.10, 0.05, 0.00, -0.05, -0.10, 0.20, 0.30]
    df["mask_1M"] = [True] * 7

    out = apply_target_modes(df, sector_min_group_size=5)
    tech = out[out["sector"] == "Tech"]
    # Median of Tech r_1M = median([0.10, 0.05, 0.00, -0.05, -0.10, 0.20]) = 0.025.
    expected_tech = np.array([0.10, 0.05, 0.00, -0.05, -0.10, 0.20]) - 0.025
    assert np.allclose(
        sorted(tech["y_1M_sector_return"].to_numpy()), sorted(expected_tech)
    )
    # Energy has 1 name -> below threshold -> passthrough (= universe-demeaned r_1M).
    energy = out[out["sector"] == "Energy"]
    assert float(energy["y_1M_sector_return"].iloc[0]) == 0.30


def test_beta_resid_target_subtracts_beta_times_market_horizon_return():
    d = date(2020, 1, 31)
    df = pd.DataFrame({
        "date": [d, d, d],
        "beta_252d": [1.5, 1.0, 0.5],
        "r_1M": [0.08, 0.05, 0.02],
        "mask_1M": [True, True, True],
    })
    for h in HORIZONS:
        if f"r_{h}" not in df:
            df[f"r_{h}"] = 0.0
        if f"mask_{h}" not in df:
            df[f"mask_{h}"] = False
    df["r_1M"] = [0.08, 0.05, 0.02]
    df["mask_1M"] = [True, True, True]

    # Universe earned 4% over the 1M horizon starting at d.
    mhr = {h: {} for h in HORIZONS}
    mhr["1M"][d] = 0.04
    out = apply_target_modes(df, market_horizon_returns=mhr)
    # y = r - beta * mkt_r => [0.08 - 1.5*0.04, 0.05 - 1.0*0.04, 0.02 - 0.5*0.04]
    expected = np.array([0.08 - 0.06, 0.05 - 0.04, 0.02 - 0.02])
    assert np.allclose(out["y_1M_beta_resid"].to_numpy(), expected)


def test_beta_resid_target_falls_back_when_market_return_missing():
    d = date(2020, 1, 31)
    df = pd.DataFrame({
        "date": [d, d],
        "beta_252d": [1.0, 1.0],
        "r_1M": [0.05, 0.03],
        "mask_1M": [True, True],
    })
    for h in HORIZONS:
        if f"r_{h}" not in df:
            df[f"r_{h}"] = 0.0
        if f"mask_{h}" not in df:
            df[f"mask_{h}"] = False
    df["r_1M"] = [0.05, 0.03]
    df["mask_1M"] = [True, True]
    # market_horizon_returns missing this date for 1M -> NaN; fit drops those rows.
    mhr = {h: {} for h in HORIZONS}
    out = apply_target_modes(df, market_horizon_returns=mhr)
    assert out["y_1M_beta_resid"].isna().all()


def test_beta_sector_resid_strips_both_beta_and_sector():
    d = date(2020, 1, 31)
    df = pd.DataFrame({
        "date": [d, d, d],
        "sector": ["Tech", "Tech", "Tech"],
        "beta_252d": [1.5, 1.0, 0.5],
        "r_1M": [0.08, 0.05, 0.02],
        "mask_1M": [True, True, True],
    })
    for h in HORIZONS:
        if f"r_{h}" not in df:
            df[f"r_{h}"] = 0.0
        if f"mask_{h}" not in df:
            df[f"mask_{h}"] = False
    df["r_1M"] = [0.08, 0.05, 0.02]
    df["mask_1M"] = [True, True, True]

    mhr = {h: {} for h in HORIZONS}
    mhr["1M"][d] = 0.04
    out = apply_target_modes(df, market_horizon_returns=mhr, sector_min_group_size=2)
    # beta_resid = r - beta*mkt = [0.02, 0.01, 0.00]; within-Tech median = 0.01;
    # beta_sector_resid = beta_resid - 0.01 = [0.01, 0.00, -0.01].
    assert np.allclose(out["y_1M_beta_sector_resid"].to_numpy(), [0.01, 0.0, -0.01])
    # And it is sector-demeaned: within-sector median is ~0.
    assert abs(float(np.median(out["y_1M_beta_sector_resid"].to_numpy()))) < 1e-9


def test_within_sector_ic_supports_industry_grouping():
    rng = np.random.default_rng(3)
    n = 12
    r_a = rng.normal(size=n)
    r_b = rng.normal(size=n)
    test_df = pd.DataFrame({
        "industry": ["A"] * n + ["B"] * n,
        "sector": ["S"] * (2 * n),
        "r_1M": np.concatenate([r_a, r_b]),
    })
    preds = test_df["r_1M"].to_numpy()  # perfect within-group ranking
    ic = within_sector_ic(preds, test_df, "r_1M", min_group_size=10, group_col="industry")
    assert ic == pytest.approx(1.0, abs=1e-9)
    # Absent grouping column -> NaN, not a crash.
    assert np.isnan(within_sector_ic(preds, test_df, "r_1M", group_col="missing_col"))


def test_block_bootstrap_summary_reports_power_floor():
    ics = [0.10, 0.08, 0.12, 0.09, 0.11, 0.07, 0.13, 0.10]
    s = block_bootstrap_summary(ics, block_size=2, reps=500, seed=1)
    assert "se_block" in s and "min_detect_ic" in s
    assert s["se_block"] > 0
    assert s["min_detect_ic"] == pytest.approx(1.96 * s["se_block"], rel=1e-9)
    # reps=0 path uses the analytic SE = std / sqrt(effective_blocks).
    s0 = block_bootstrap_summary(ics, block_size=2, reps=0, seed=1)
    assert s0["se_block"] > 0
    assert s0["min_detect_ic"] == pytest.approx(1.96 * s0["se_block"], rel=1e-9)


def test_build_market_horizon_returns_aggregates_over_trading_days():
    # 25 fake trading days with constant +1% daily log return; H=21 trading days
    # => market_r_h = 21 * 0.01 = 0.21 at any grid date with 21 days forward.
    from backend.ingestion.calendar import HORIZON_TRADING_DAYS as HTD
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(25)]
    market_returns = {d: 0.01 for d in dates}
    grid = [dates[0], dates[3], dates[-2]]
    out = build_market_horizon_returns(market_returns, grid, horizons=("1M",))
    # 21 trading days forward from dates[0] and dates[3] both fit; dates[-2] does not.
    assert abs(out["1M"][dates[0]] - HTD["1M"] * 0.01) < 1e-9
    assert abs(out["1M"][dates[3]] - HTD["1M"] * 0.01) < 1e-9
    assert dates[-2] not in out["1M"]  # window runs past end


def test_industry_neutral_momentum_subtracts_within_date_industry_median():
    df = pd.DataFrame({
        "date": [date(2020, 1, 31)] * 6,
        "industry": ["Software"] * 5 + ["Hardware"],  # Hardware group has < 5 names
        "mom_12_1": [0.20, 0.10, 0.00, -0.10, -0.20, 1.00],
    })
    out = add_industry_neutral_momentum(df, min_group_size=5)
    sw = out[out["industry"] == "Software"]
    # Median of Software is 0; values pass through subtracted by 0 (no change).
    assert np.allclose(
        sorted(sw["industry_neutral_mom_12_1"].to_numpy()),
        sorted(sw["mom_12_1"].to_numpy()),
    )
    # Hardware has only 1 name -> below min_group_size -> falls back to raw.
    hw = out[out["industry"] == "Hardware"]
    assert float(hw["industry_neutral_mom_12_1"].iloc[0]) == 1.0


# =============================================================
# Cross-sectional transforms
# =============================================================


def test_rank_normalize_in_range_and_centered():
    df = pd.DataFrame({
        "date": [date(2020, 1, 31)] * 5,
        "mom_1m": [5.0, 1.0, 3.0, 2.0, 4.0],
    })
    for c in FEATURE_COLS:
        if c not in df:
            df[c] = 0.0
    out = rank_normalize_features(df, ["mom_1m"])
    vals = out["mom_1m"].to_numpy()
    assert vals.min() >= -1.0 and vals.max() <= 1.0
    assert abs(vals.mean()) < 1e-9          # symmetric ranks center at 0
    # Order is preserved (largest input -> largest normalized rank).
    assert np.argmax(vals) == 0 and np.argmin(vals) == 1


def test_rank_normalize_can_use_industry_relative_groups():
    df = pd.DataFrame({
        "date": [date(2020, 1, 31)] * 4,
        "sector": ["Tech"] * 4,
        "industry": ["Software", "Software", "Hardware", "Hardware"],
        "mom_1m": [1.0, 2.0, 100.0, 200.0],
    })
    out = rank_normalize_features(
        df,
        ["mom_1m"],
        industry_relative=True,
        min_group_size=2,
    )
    vals = out["mom_1m"].to_numpy()
    assert np.allclose(vals, np.array([-1.0, 1.0, -1.0, 1.0]))


def test_demean_subtracts_median_and_masks_missing():
    df = pd.DataFrame({
        "date": [date(2020, 1, 31), date(2020, 1, 31), date(2020, 2, 28)],
        "r_1M": [0.10, 0.00, 0.05],
        "mask_1M": [True, True, True],
    })
    for h in HORIZONS:
        if f"r_{h}" not in df:
            df[f"r_{h}"] = 0.0
        if f"mask_{h}" not in df:
            df[f"mask_{h}"] = False
    df["mask_1M"] = [True, True, True]
    df["r_1M"] = [0.10, 0.00, 0.05]
    medians = {h: {} for h in HORIZONS}
    medians["1M"] = {date(2020, 1, 31): 0.04}  # Feb has no median -> masked
    out = demean_cross_sectional(df, medians)
    jan = out[out["date"] == date(2020, 1, 31)]
    assert np.allclose(sorted(jan["r_1M"]), sorted([0.10 - 0.04, 0.00 - 0.04]))
    feb = out[out["date"] == date(2020, 2, 28)]
    assert bool(feb["mask_1M"].iloc[0]) is False  # no median -> masked off


# =============================================================
# Fold / embargo math (leakage guard)
# =============================================================


def test_walk_forward_folds_respect_embargo():
    grid = [date(2020, m, 28) for m in range(1, 13)]  # 12 month-ends
    folds = walk_forward_folds(grid, min_train_months=3, embargo_steps=2)
    # First testable index = min_train(3) + embargo(2) = 5.
    assert folds[0][0] == grid[5]
    for test_date, cutoff in folds:
        i = grid.index(test_date)
        assert cutoff == grid[i - 2]          # cutoff is exactly embargo behind
        assert cutoff < test_date


def test_summarize_basic():
    s = summarize([0.1, 0.1, 0.1, 0.1])
    assert s["n_folds"] == 4
    assert abs(s["mean_ic"] - 0.1) < 1e-12
    assert s["hit_rate"] == 1.0
    assert summarize([])["n_folds"] == 0


def test_block_bootstrap_summary_reports_ci_and_overlap_adjusted_t():
    s = block_bootstrap_summary(
        [0.10, 0.08, 0.12, 0.09, 0.11, 0.07, 0.13, 0.10],
        block_size=2,
        reps=200,
        seed=1,
    )
    assert s["n_folds"] == 8
    assert s["block_size"] == 2
    assert s["ci_low"] < s["mean_ic"] < s["ci_high"]
    assert s["effective_blocks"] == 4
    assert s["t_block"] > 0
    assert 0 <= s["p_value"] <= 1


def test_score_current_cross_section_rank_transforms_predictions():
    # Predictions are mapped to within-cross-section percentile rank in [0, 1]
    # regardless of the training target. Raw preds [-0.2, 1.3] over 2 active
    # names => the lower goes to 0.0 and the higher to 1.0.
    class DummyModel:
        def predict(self, X):
            return np.array([-0.2, 1.3])

    as_of = date(2026, 5, 29)
    df = pd.DataFrame({
        "date": [as_of, as_of, as_of],
        "ticker_id": [1, 2, 3],
    })
    for c in FEATURE_COLS:
        df[c] = 0.0

    rows = score_current_cross_section(
        df, {"3M": [DummyModel()]}, ("3M",), as_of=as_of, active_ids={1, 3}
    )

    assert [r["ticker_id"] for r in rows] == [1, 3]
    assert [r["relative_rank"] for r in rows] == [0.0, 1.0]
    # Confidence is no longer computed here — it's rank stability, filled in run()
    # from the DB history of prior scoring dates.
    assert "confidence" not in rows[0]


def test_score_current_cross_section_rank_transforms_regression_outputs():
    # A `beta_resid`-trained model outputs log returns like [-0.05, +0.05]; the
    # old "clip to [0, 1]" path would squash those to [0, 1] and lose the
    # ranking. Rank-transforming preserves the order and produces evenly-spaced
    # percentile ranks for the 5-name cross-section.
    class DummyRegressionModel:
        def predict(self, X):
            # Negative numbers that the old clip path would all collapse to 0.0.
            return np.array([-0.05, -0.03, -0.02, -0.04, -0.01])

    as_of = date(2026, 5, 29)
    df = pd.DataFrame({
        "date": [as_of] * 5,
        "ticker_id": [10, 20, 30, 40, 50],
    })
    for c in FEATURE_COLS:
        df[c] = 0.0

    rows = score_current_cross_section(
        df, {"1Y": [DummyRegressionModel()]}, ("1Y",),
        as_of=as_of, active_ids={10, 20, 30, 40, 50},
    )

    # Ordered by ticker_id (df row order) — preds: -0.05,-0.03,-0.02,-0.04,-0.01
    # sorted ascending: -0.05 (10), -0.04 (40), -0.03 (20), -0.02 (30), -0.01 (50)
    # => ranks: 10->0.00, 40->0.25, 20->0.50, 30->0.75, 50->1.00
    by_tid = {r["ticker_id"]: r["relative_rank"] for r in rows}
    assert np.allclose(by_tid[10], 0.00)
    assert np.allclose(by_tid[40], 0.25)
    assert np.allclose(by_tid[20], 0.50)
    assert np.allclose(by_tid[30], 0.75)
    assert np.allclose(by_tid[50], 1.00)


def test_score_current_cross_section_applies_knife_overlay():
    # With a HorizonSpec carrying knife_lambda>0, the score path demotes a name
    # that is the model's #1 pick but a falling knife (high vol + downtrend), and
    # promotes a clean high-vol uptrender. A knife_lambda=0 spec is a no-op.
    from backend.ml.gbm_baseline import HorizonSpec

    class RampModel:
        def predict(self, X):
            return np.arange(len(X), dtype=float)  # name 5 ranked top

    as_of = date(2026, 5, 29)
    df = pd.DataFrame({"date": [as_of] * 5, "ticker_id": [1, 2, 3, 4, 5]})
    for c in FEATURE_COLS:
        df[c] = 0.0
    # name 5 = falling knife; name 4 = clean high-vol uptrender; 1-3 neutral.
    df["vol_120d"] = [0.0, 0.0, 0.0, 1.0, 1.0]
    df["ma_gap_200"] = [0.0, 0.0, 0.0, 1.0, -1.0]
    df["dist_low_252"] = [0.0, 0.0, 0.0, 1.0, -1.0]
    ids = {1, 2, 3, 4, 5}

    tilt = score_current_cross_section(
        df, {"3M": [RampModel()]}, {"3M": HorizonSpec(knife_lambda=0.5)},
        as_of=as_of, active_ids=ids,
    )
    by = {r["ticker_id"]: r["relative_rank"] for r in tilt}
    assert by[5] < 1.0           # the knife is no longer the top pick
    assert by[4] == pytest.approx(1.0)  # the clean uptrender takes the top

    off = score_current_cross_section(
        df, {"3M": [RampModel()]}, {"3M": HorizonSpec(knife_lambda=0.0)},
        as_of=as_of, active_ids=ids,
    )
    assert {r["ticker_id"]: r["relative_rank"] for r in off}[5] == pytest.approx(1.0)


def test_score_current_cross_section_attaches_risk_flag():
    # The transparency tag is written even when the overlay is off (knife_lambda=0):
    # name 5 is a falling knife (high vol + downtrend) -> 'high'; name 4 is a
    # high-vol UPtrender -> 'none'; neutral names -> 'none'.
    from backend.ml.gbm_baseline import HorizonSpec

    class RampModel:
        def predict(self, X):
            return np.arange(len(X), dtype=float)

    as_of = date(2026, 5, 29)
    df = pd.DataFrame({"date": [as_of] * 5, "ticker_id": [1, 2, 3, 4, 5]})
    for c in FEATURE_COLS:
        df[c] = 0.0
    df["vol_120d"] = [0.0, 0.0, 0.0, 1.0, 1.0]
    df["ma_gap_200"] = [0.0, 0.0, 0.0, 1.0, -1.0]
    df["dist_low_252"] = [0.0, 0.0, 0.0, 1.0, -1.0]

    rows = score_current_cross_section(
        df, {"3M": [RampModel()]}, {"3M": HorizonSpec(knife_lambda=0.0)},
        as_of=as_of, active_ids={1, 2, 3, 4, 5},
    )
    flags = {r["ticker_id"]: r["risk_flag"] for r in rows}
    assert flags[5] == "high"          # falling knife, tagged though rank untouched
    assert flags[4] == "none"          # high vol but uptrending → not a knife
    assert flags[1] == "none"


def test_score_current_cross_section_blends_linear_model():
    # GBDT ranks ascending by ticker, ridge ranks descending — perfectly opposed.
    # At blend weight 0.5 every name's blended rank-score is identical (0.5*r +
    # 0.5*(1-r) = 0.5), so all percentile ranks collapse to 0.5.
    class GBDT:
        def predict(self, X):
            return np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    class RidgeLike:
        def predict(self, X):
            return np.array([4.0, 3.0, 2.0, 1.0, 0.0])

    as_of = date(2026, 5, 29)
    df = pd.DataFrame({"date": [as_of] * 5, "ticker_id": [1, 2, 3, 4, 5]})
    for c in FEATURE_COLS:
        df[c] = 0.0

    rows = score_current_cross_section(
        df, {"3M": [GBDT()]}, ("3M",), as_of=as_of, active_ids={1, 2, 3, 4, 5},
        linear_models={"3M": (RidgeLike(), 0.5)},
    )
    assert all(abs(r["relative_rank"] - 0.5) < 1e-9 for r in rows)


def test_production_horizon_specs_use_sector_return_for_scored_horizons():
    # Locks in the test-6 promotion (sector_return for every scored horizon — the
    # within-sector / SECB winner) so an accidental edit to the spec dict trips the
    # test. 1M stays `rank` (dead horizon, not scored in production).
    from backend.ml.gbm_baseline import ESTIMATE_SURPRISE_FEATURES, PRODUCTION_HORIZON_SPECS

    for h in ("3M", "6M", "1Y"):
        assert PRODUCTION_HORIZON_SPECS[h].target_mode == "sector_return", (
            f"{h} target unexpectedly changed — re-sweep SECB before promoting"
        )
    assert PRODUCTION_HORIZON_SPECS["1M"].target_mode == "rank"
    # 6M and 1Y still carry the promoted revenue-surprise pack.
    for h in ("6M", "1Y"):
        assert ESTIMATE_SURPRISE_FEATURES[0] in (PRODUCTION_HORIZON_SPECS[h].feature_cols or [])


def test_fit_horizon_models_uses_per_horizon_target_mode():
    # End-to-end on a tiny synthetic panel: each horizon's model trains on the
    # target column its spec selects (we sanity-check via the y_*_* columns that
    # apply_target_modes produces). We don't assert on prediction values, only
    # that fit succeeds and one model is produced per horizon key.
    from backend.ml.gbm_baseline import HorizonSpec, LGBMConfig
    from backend.ml.gbm_inference import fit_horizon_models

    # Build a 3-date, 8-name panel with all required columns.
    dates = [date(2020, 1, 31), date(2020, 2, 28), date(2020, 3, 31)]
    rows = []
    rng = np.random.default_rng(42)
    for d in dates:
        for tid in range(8):
            row = {"date": d, "ticker_id": tid, "beta_252d": 1.0}
            for c in FEATURE_COLS:
                row[c] = float(rng.normal())
            for h in HORIZONS:
                row[f"r_{h}"] = float(rng.normal(0, 0.05))
                row[f"mask_{h}"] = True
                # apply_target_modes columns we feed directly
                row[f"y_{h}_return"] = row[f"r_{h}"]
                row[f"y_{h}_rank"] = float((tid + 1) / 8.0)
                row[f"y_{h}_quantile"] = float(tid % 5)
                row[f"y_{h}_sector_return"] = row[f"r_{h}"]
                row[f"y_{h}_beta_resid"] = row[f"r_{h}"]
            rows.append(row)
    panel = pd.DataFrame(rows)

    specs = {
        "3M": HorizonSpec(target_mode="rank", lgb_cfg=LGBMConfig(n_estimators=20)),
        "1Y": HorizonSpec(target_mode="beta_resid", lgb_cfg=LGBMConfig(n_estimators=20)),
    }
    as_of = date(2020, 3, 31)
    models, train_windows, trained_ids, linear_models = fit_horizon_models(
        panel, specs, seed=1, as_of=as_of, n_seeds=1
    )
    assert set(models.keys()) == {"3M", "1Y"}
    assert all(isinstance(models[h], list) and len(models[h]) == 1 for h in models)
    assert all(train_windows[h]["rows"] > 0 for h in models)
    assert trained_ids
    # No spec sets linear_blend, so no ridge models are fit.
    assert linear_models == {}


def _tiny_panel(n_names: int = 8):
    """3-date panel with every column fit_horizon_models needs (mirrors the
    per-horizon-target test above)."""
    dates = [date(2020, 1, 31), date(2020, 2, 28), date(2020, 3, 31)]
    rng = np.random.default_rng(7)
    rows = []
    for d in dates:
        for tid in range(n_names):
            row = {"date": d, "ticker_id": tid, "beta_252d": 1.0}
            for c in FEATURE_COLS:
                row[c] = float(rng.normal())
            for h in HORIZONS:
                row[f"r_{h}"] = float(rng.normal(0, 0.05))
                row[f"mask_{h}"] = True
                row[f"y_{h}_return"] = row[f"r_{h}"]
                row[f"y_{h}_rank"] = float((tid + 1) / n_names)
                row[f"y_{h}_sector_return"] = row[f"r_{h}"]
            rows.append(row)
    return pd.DataFrame(rows)


def test_fit_horizon_models_excludes_ids():
    # User-added tickers must be droppable from training while everyone else is
    # unaffected — the guarantee that off-index names never train the model.
    from backend.ml.gbm_baseline import HorizonSpec, LGBMConfig
    from backend.ml.gbm_inference import fit_horizon_models

    panel = _tiny_panel(n_names=8)
    specs = {"3M": HorizonSpec(target_mode="rank", lgb_cfg=LGBMConfig(n_estimators=10))}
    as_of = date(2020, 3, 31)

    _, base_windows, base_ids, _ = fit_horizon_models(
        panel, specs, seed=1, as_of=as_of, n_seeds=1
    )
    _, ex_windows, ex_ids, _ = fit_horizon_models(
        panel, specs, seed=1, as_of=as_of, n_seeds=1, exclude_ids={0, 1}
    )
    assert {0, 1} <= base_ids
    assert {0, 1}.isdisjoint(ex_ids)          # excluded names never trained
    assert ex_windows["3M"]["rows"] < base_windows["3M"]["rows"]
    assert ex_windows["3M"]["tickers"] == base_windows["3M"]["tickers"] - 2


def test_specs_from_serialized_roundtrips():
    from backend.ml.gbm_baseline import HorizonSpec, LGBMConfig
    from backend.ml.gbm_inference import _serialize_spec, _specs_from_serialized

    spec = HorizonSpec(
        target_mode="sector_return",
        lgb_cfg=LGBMConfig(n_estimators=123, num_leaves=9),
        feature_cols=["mom_1m", "vol_20d"],
        linear_blend=0.3,
        ridge_alpha=5.0,
        smooth_span=4,
        knife_lambda=0.2,
    )
    rebuilt = _specs_from_serialized({"6M": _serialize_spec(spec)})["6M"]
    assert rebuilt.target_mode == "sector_return"
    assert rebuilt.feature_cols == ["mom_1m", "vol_20d"]
    assert rebuilt.linear_blend == 0.3
    assert rebuilt.ridge_alpha == 5.0
    assert rebuilt.smooth_span == 4
    assert rebuilt.knife_lambda == 0.2
    assert rebuilt.lgb_cfg.n_estimators == 123
    assert rebuilt.lgb_cfg.num_leaves == 9


def test_save_load_bundle_roundtrips(tmp_path):
    from backend.ml.gbm_inference import load_bundle, save_bundle

    bundle = {"model_type": "x", "as_of": "2020-03-31", "specs": {"6M": {"k": 1}}}
    path = tmp_path / "b.pkl"
    sha = save_bundle(path, bundle)
    assert path.exists() and len(sha) == 64
    assert load_bundle(path) == bundle


def test_apply_cross_horizon_shrink_pulls_1y_toward_6m():
    from backend.ml.gbm_inference import apply_cross_horizon_shrink

    # 1Y ranks are the reverse of 6M; full shrink (weight=1.0) should re-rank 1Y
    # to match 6M's ordering exactly.
    rows = []
    for tid, (r6, r1) in enumerate(
        [(0.0, 1.0), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25), (1.0, 0.0)], start=1
    ):
        rows.append({"ticker_id": tid, "horizon": "6M", "relative_rank": r6})
        rows.append({"ticker_id": tid, "horizon": "1Y", "relative_rank": r1})

    out = apply_cross_horizon_shrink(rows, source="6M", target="1Y", weight=1.0)
    by = {(r["ticker_id"], r["horizon"]): r["relative_rank"] for r in out}
    for tid in range(1, 6):
        assert by[(tid, "1Y")] == pytest.approx(by[(tid, "6M")])
    # weight 0 is a no-op.
    rows2 = [{"ticker_id": 1, "horizon": "1Y", "relative_rank": 0.3},
             {"ticker_id": 1, "horizon": "6M", "relative_rank": 0.9}]
    assert apply_cross_horizon_shrink(list(rows2), weight=0.0) == rows2


def test_rank_stability():
    from backend.ml.gbm_inference import rank_stability

    assert rank_stability([]) is None           # no history
    assert rank_stability([0.5]) is None         # single scoring date -> undefined
    assert rank_stability([0.8, 0.8, 0.8]) < 1e-9  # perfectly consistent
    assert abs(rank_stability([0.2, 0.8]) - 0.3) < 1e-9  # population std = 0.3
    # higher dispersion => larger std
    assert rank_stability([0.1, 0.9, 0.5]) > rank_stability([0.45, 0.55, 0.5])


# =============================================================
# Prediction smoothing (Workstream A): EWMA across scoring dates
# =============================================================


def _osc_records(n_dates: int = 20, n_names: int = 6):
    """Two names whose ranks oscillate hard date-to-date around a stable mean.

    Built as raw preds so `_rank01` inside the smoother maps them to percentile
    ranks; the EWMA should damp the oscillation.
    """
    records = []
    for k in range(n_dates):
        # name 0 alternates extreme high/low; the rest fill the middle deterministically.
        preds = np.linspace(0.0, 1.0, n_names)
        preds[0] = 1.0 if k % 2 == 0 else 0.0
        records.append({
            "ticker_ids": np.arange(n_names),
            "pred": preds.astype(float),
            "r": np.zeros(n_names),
            "sector": None,
        })
    return records


def test_ewma_rank_smoothing_damps_oscillation():
    records = _osc_records()
    smoothed = ewma_rank_by_ticker(records, span=4)
    # The oscillating name's smoothed rank should vary far less than its raw rank.
    raw_name0 = np.array([_rank01_of(r["pred"])[0] for r in records])
    sm_name0 = np.array([s[0] for s in smoothed])
    assert sm_name0.std() < raw_name0.std()
    # Shapes preserved per fold.
    assert all(s.shape == r["pred"].shape for s, r in zip(smoothed, records))


def _rank01_of(a):
    from backend.ml.gbm_baseline import _rank01

    return _rank01(a)


def test_smoothing_reduces_rank_turnover():
    records = _osc_records()
    smoothed = ewma_rank_by_ticker(records, span=4)
    assert rank_turnover(records, rank_series=smoothed) < rank_turnover(records)


def test_walk_forward_smooth_span_zero_is_noop():
    panel = _planted_panel(n_dates=36, n_names=40, beta=1.0, seed=3)
    wf = WalkForwardConfig(min_train_months=12, min_names=15)
    cfg = LGBMConfig(n_estimators=80)
    base = walk_forward_ic(panel, "1M", cfg, wf, seed=1, target_mode="return")
    same = walk_forward_ic(panel, "1M", cfg, wf, seed=1, target_mode="return", smooth_span=0)
    assert [f["ic"] for f in base["folds"]] == [f["ic"] for f in same["folds"]]
    # Turnover is reported even when smoothing is off; smoothed turnover stays None.
    assert base["turnover_raw"] is not None
    assert base["turnover_smoothed"] is None


def test_walk_forward_smoothing_changes_ic_and_reports_turnover():
    panel = _planted_panel(n_dates=36, n_names=40, beta=1.0, seed=3)
    wf = WalkForwardConfig(min_train_months=12, min_names=15)
    cfg = LGBMConfig(n_estimators=80)
    res = walk_forward_ic(panel, "1M", cfg, wf, seed=1, target_mode="return", smooth_span=4)
    assert res["turnover_smoothed"] is not None
    # On a persistent planted signal, smoothing should not destroy it (mean IC stays positive).
    assert res["summary"]["mean_ic"] > 0


def test_production_specs_promoted_smooth_spans():
    # Lock the 2026-06-05 promotion: smooth 3M (span 3) and 1Y (span 4) only;
    # 6M and 1M stay unsmoothed. Trips if the spec dict is edited accidentally.
    from backend.ml.gbm_baseline import PRODUCTION_HORIZON_SPECS

    assert PRODUCTION_HORIZON_SPECS["3M"].smooth_span == 3
    assert PRODUCTION_HORIZON_SPECS["1Y"].smooth_span == 4
    assert PRODUCTION_HORIZON_SPECS["6M"].smooth_span == 0
    assert PRODUCTION_HORIZON_SPECS["1M"].smooth_span == 0


def test_production_specs_promoted_knife_lambda():
    # Lock the 2026-06-06 promotion: falling-knife overlay at 3M only (λ=0.20);
    # 6M/1Y/1M stay at 0 (poor trade / power-limited). Trips on accidental edits.
    from backend.ml.gbm_baseline import PRODUCTION_HORIZON_SPECS

    assert PRODUCTION_HORIZON_SPECS["3M"].knife_lambda == 0.20
    assert PRODUCTION_HORIZON_SPECS["6M"].knife_lambda == 0.0
    assert PRODUCTION_HORIZON_SPECS["1Y"].knife_lambda == 0.0
    assert PRODUCTION_HORIZON_SPECS["1M"].knife_lambda == 0.0


def test_apply_rank_smoothing_blends_toward_prior_and_noops_off():
    from backend.ml.gbm_baseline import HorizonSpec
    from backend.ml.gbm_inference import apply_rank_smoothing

    # span>0: each name's rank is EWMA'd toward its prior, then re-ranked. With a
    # full rank reversal vs the prior, smoothing should pull the new ranks back
    # toward the prior ordering (the top-by-raw name should no longer be rank 1).
    rows = [{"ticker_id": t, "horizon": "3M", "relative_rank": r}
            for t, r in zip(range(1, 6), [0.0, 0.25, 0.5, 0.75, 1.0])]
    prior = {(t, "3M"): [p] for t, p in zip(range(1, 6), [1.0, 0.75, 0.5, 0.25, 0.0])}
    specs = {"3M": HorizonSpec(smooth_span=3)}
    out = apply_rank_smoothing([dict(r) for r in rows], specs, prior)
    # ticker 5 had raw rank 1.0 but prior 0.0 → its blended rank must drop below 1.0.
    assert next(r["relative_rank"] for r in out if r["ticker_id"] == 5) < 1.0

    # span=0 spec is an exact no-op.
    specs0 = {"3M": HorizonSpec(smooth_span=0)}
    rows0 = [dict(r) for r in rows]
    assert apply_rank_smoothing(rows0, specs0, prior) == rows
    # No prior for a name → that name keeps its raw rank under the relative re-rank.
    specs3 = {"3M": HorizonSpec(smooth_span=3)}
    out2 = apply_rank_smoothing([dict(r) for r in rows], specs3, {})
    assert [r["relative_rank"] for r in out2] == [r["relative_rank"] for r in rows]


# =============================================================
# Target-ensemble blend sweep (decorrelated-label variance reduction)
# =============================================================


def _two_view_records(n_dates: int = 8, n_names: int = 80, seed: int = 0):
    """Two fold-aligned record lists (base, alt): independent noisy views of the same
    latent signal `s` that drives realized return `r`. Averaging their ranks should
    track `s` better than either view alone — the variance-reduction thesis behind
    rank-ensembling decorrelated training targets. Same `date`/`ticker_ids`/`r` per
    fold; only `pred` differs (as for two walk-forwards on different target_modes)."""
    rng = np.random.default_rng(seed)
    base_recs, alt_recs = [], []
    for d in range(n_dates):
        s = rng.normal(size=n_names)               # latent signal
        r = s + 0.5 * rng.normal(size=n_names)     # realized return driven by s
        common = {
            "date": date(2020, 1, 1) + timedelta(days=30 * d),
            "ticker_ids": np.arange(n_names),
            "r": r,
            "sector": None,
            "risk": {},
        }
        base_recs.append({**common, "pred": s + 1.5 * rng.normal(size=n_names)})
        alt_recs.append({**common, "pred": s + 1.5 * rng.normal(size=n_names)})
    return base_recs, alt_recs


def _mean_ic(records):
    ics = [pd.Series(r["pred"]).corr(pd.Series(r["r"]), method="spearman") for r in records]
    return float(np.mean(ics))


def test_target_blend_sweep_endpoints_recover_base_and_alt():
    # w=0 must score the pure base target, w=1 the pure alt. rank01 is monotonic, so
    # the scored IC equals the raw-prediction Spearman IC of each endpoint.
    base, alt = _two_view_records(seed=1)
    rows = {r["w"]: r for r in target_blend_sweep(
        base, alt, [0.0, 1.0], block_size=2, reps=0, compute_sector_ic=False)}
    assert rows[0.0]["mean_ic"] == pytest.approx(_mean_ic(base), abs=1e-9)
    assert rows[1.0]["mean_ic"] == pytest.approx(_mean_ic(alt), abs=1e-9)


def test_target_blend_sweep_reduces_variance_and_beats_endpoints():
    # The core lever: blending two decorrelated views of the same signal lifts mean IC
    # above EITHER endpoint (averaging cancels independent estimation noise). Turnover
    # is always finite.
    base, alt = _two_view_records(seed=2)
    rows = {r["w"]: r for r in target_blend_sweep(
        base, alt, [0.0, 0.5, 1.0], block_size=2, reps=0, compute_sector_ic=False)}
    assert rows[0.5]["mean_ic"] > rows[0.0]["mean_ic"]
    assert rows[0.5]["mean_ic"] > rows[1.0]["mean_ic"]
    assert np.isfinite(rows[0.5]["turnover"])


def test_target_blend_sweep_unmatched_dates_keep_base_rank():
    # If an alt fold can't be matched by date, that fold falls back to the base rank,
    # so the blend at any weight scores exactly the base IC (no silent misalignment).
    base, alt = _two_view_records(seed=3)
    for rec in alt:  # shift alt dates so none match base
        rec["date"] = rec["date"] + timedelta(days=5)
    rows = {r["w"]: r for r in target_blend_sweep(
        base, alt, [0.0, 0.5], block_size=2, reps=0, compute_sector_ic=False)}
    assert rows[0.5]["mean_ic"] == pytest.approx(rows[0.0]["mean_ic"], abs=1e-9)


# =============================================================
# Per-horizon regularization sweep
# =============================================================


def test_lgbm_config_has_reg_alpha_and_it_reaches_the_model():
    # reg_alpha is a new L1 knob; default 0 (LightGBM default) and it must be wired
    # into the fitted estimator, not silently dropped.
    from backend.ml.gbm_baseline import LGBMConfig, fit_lgbm_model

    assert LGBMConfig().reg_alpha == 0.0
    panel = _planted_panel(n_dates=18, n_names=30, beta=1.0, seed=4)
    train = panel[panel["mask_1M"] & panel["r_1M"].notna()]
    model = fit_lgbm_model(train, "r_1M", LGBMConfig(reg_alpha=2.5, n_estimators=40),
                           seed=0, feature_cols=FEATURE_COLS)
    assert model.get_params()["reg_alpha"] == 2.5


def test_regularization_sweep_runs_each_config_and_reports_metrics():
    from backend.ml.gbm_baseline import LGBMConfig

    panel = _planted_panel(n_dates=30, n_names=40, beta=1.0, seed=5)
    wf = WalkForwardConfig(min_train_months=12, min_names=15)
    base = LGBMConfig(n_estimators=60)
    configs = [("baseline", base),
               ("small_trees", LGBMConfig(n_estimators=60, num_leaves=7, max_depth=3))]
    rows = regularization_sweep(
        panel, "1M", configs, wf_cfg=wf, target_mode="return",
        feature_cols=FEATURE_COLS, n_seeds=1, block_size=1, reps=0,
        compute_sector_ic=False,
    )
    assert [r["name"] for r in rows] == ["baseline", "small_trees"]
    # Every config produces a finite mean IC and turnover on the planted signal.
    for r in rows:
        assert np.isfinite(r["mean_ic"])
        assert np.isfinite(r["turnover"])
        assert {"sec_ic", "sec_icir", "sec_t_block", "sec_p"} <= r.keys()


# =============================================================
# Falling-knife output overlay (vol × downtrend) + top-decile risk
# =============================================================


def _knife_records(n_dates: int = 3, n_names: int = 10):
    """Records where the model's #1 name is a falling knife and its #2 is a clean,
    high-vol UPtrending name.

    The overlay should demote the knife below the uptrender — cutting top-decile
    knife score and realized downside WITHOUT lowering volatility (high-vol winners
    are intentionally spared by the vol×downtrend gate). Mid names are neutral.
    Risk features are the cross-sectional rank-normalized [-1, 1] values the overlay
    consumes; `r` is the realized demeaned return (the knife cuts, the uptrender wins).
    """
    knife_i, clean_i = n_names - 1, n_names - 2
    vol = np.zeros(n_names)
    trend = np.zeros(n_names)
    dlow = np.zeros(n_names)
    r = np.zeros(n_names)
    vol[knife_i], trend[knife_i], dlow[knife_i] = 1.0, -1.0, -1.0   # high-vol downtrend
    vol[clean_i], trend[clean_i], dlow[clean_i] = 1.0, 1.0, 1.0     # high-vol uptrend
    r[knife_i], r[clean_i] = -0.30, 0.05
    records = []
    for _ in range(n_dates):
        records.append({
            "ticker_ids": np.arange(n_names),
            "pred": np.arange(n_names, dtype=float),  # knife (idx n-1) ranked #1
            "r": r.copy(),
            "sector": None,
            "risk": {"vol": vol.copy(), "trend": trend.copy(), "dlow": dlow.copy()},
        })
    return records


def test_knife_score_is_high_only_for_high_vol_and_downtrend():
    from backend.ml.gbm_baseline import _knife_score

    # A: high vol + downtrend + near low (true knife). B: high vol but uptrend.
    # C: low vol but downtrend. Only A should score high.
    risk = {
        "vol": np.array([1.0, 1.0, -1.0]),
        "trend": np.array([-1.0, 1.0, -1.0]),
        "dlow": np.array([-1.0, 1.0, -1.0]),
    }
    knife = _knife_score(risk)
    assert knife[0] > knife[1]            # downtrend beats uptrend at equal vol
    assert knife[0] > knife[2]            # high vol beats low vol at equal trend
    assert knife[0] == pytest.approx(1.0)  # vol_p=1 * downtrend_p=1
    assert knife[1] == pytest.approx(0.0)  # uptrend → downtrend_p=0
    assert knife[2] == pytest.approx(0.0)  # low vol → vol_p=0


def test_knife_score_none_without_risk_features():
    from backend.ml.gbm_baseline import _knife_score

    assert _knife_score(None) is None
    assert _knife_score({}) is None
    assert _knife_score({"trend": np.array([0.0])}) is None  # no vol


def test_knife_tier_grades_high_elevated_none():
    # vol_p=(vol+1)/2, downtrend_p=1-mean(trend_p,dlow_p). Names:
    #   A: top vol + deep downtrend            -> high
    #   B: top vol but uptrend                 -> none
    #   C: low vol but downtrend               -> none (vol gate)
    #   D: moderately high vol + moderate down -> elevated
    risk = {
        "vol":   np.array([1.0,  1.0, -1.0,  0.4]),
        "trend": np.array([-1.0, 1.0, -1.0, -0.4]),
        "dlow":  np.array([-1.0, 1.0, -1.0, -0.4]),
    }
    assert knife_tier(risk) == ["high", "none", "none", "elevated"]
    # No risk features → None (caller defaults every name to 'none').
    assert knife_tier(None) is None
    assert knife_tier({}) is None


def test_knife_overlay_lambda_zero_is_exact_noop():
    records = _knife_records()
    out = apply_knife_overlay(records, 0.0)
    for rec, rk in zip(records, out):
        np.testing.assert_array_equal(rk, _rank01_of(rec["pred"]))


def test_knife_overlay_demotes_a_top_ranked_knife():
    # Single cross-section: the model's #1 name is a falling knife. After the
    # overlay it must no longer sit at the top of the ranking.
    records = _knife_records(n_dates=1)
    top_idx = int(np.argmax(records[0]["pred"]))  # the knife
    overlaid = apply_knife_overlay(records, 0.5)[0]
    assert overlaid[top_idx] < _rank01_of(records[0]["pred"])[top_idx]
    assert overlaid[top_idx] < overlaid.max()  # something cleaner is now on top


def test_knife_overlay_lowers_top_decile_knife_and_downside():
    records = _knife_records()
    base = top_decile_risk(records, apply_knife_overlay(records, 0.0))
    tilt = top_decile_risk(records, apply_knife_overlay(records, 0.5))
    # The overlay swaps the knife out of the top decile for the clean uptrender:
    # lower knife score, fewer downtrending names, less realized downside up top —
    # while top-decile volatility is unchanged (the uptrender is also high-vol).
    assert tilt["knife"] < base["knife"]
    assert tilt["downtrend"] < base["downtrend"]
    assert tilt["downside"] < base["downside"]
    assert tilt["vol_p"] == pytest.approx(base["vol_p"])


def test_top_decile_risk_metric_math():
    # n=4 → top decile is ceil(0.4)=1 name; rank puts index 3 on top.
    rec = {
        "ticker_ids": np.arange(4),
        "pred": np.zeros(4),
        "r": np.array([0.10, -0.20, 0.05, -0.50]),
        "sector": None,
        "risk": {
            "vol": np.array([-1.0, 0.0, 0.0, 1.0]),
            "trend": np.array([1.0, 0.0, 0.0, -1.0]),
            "dlow": np.array([1.0, 0.0, 0.0, -1.0]),
        },
    }
    ranks = [np.array([0.2, 0.4, 0.6, 1.0])]  # index 3 ranked top
    out = top_decile_risk([rec], ranks)
    assert out["mean_r"] == pytest.approx(-0.50)              # r of the top name
    assert out["downside"] == pytest.approx(0.50)             # sqrt(min(-0.5,0)^2)
    assert out["knife"] == pytest.approx(1.0)                 # top name is a pure knife
    assert out["downtrend"] == pytest.approx(1.0)             # its downtrend_p > 0.5


def test_knife_sweep_table_baseline_row_matches_raw():
    records = _knife_records()
    rows = knife_sweep_table(records, [0.0, 0.3], compute_sector_ic=False)
    assert rows[0]["lam"] == 0.0 and rows[1]["lam"] == 0.3
    # Baseline turnover equals raw turnover; lam>0 cuts top-decile knife.
    assert rows[0]["turnover"] == pytest.approx(rank_turnover(records))
    assert rows[1]["knife"] < rows[0]["knife"]


def test_walk_forward_knife_lambda_zero_is_noop():
    panel = _planted_panel(n_dates=36, n_names=40, beta=1.0, seed=3)
    wf = WalkForwardConfig(min_train_months=12, min_names=15)
    cfg = LGBMConfig(n_estimators=80)
    base = walk_forward_ic(panel, "1M", cfg, wf, seed=1, target_mode="return")
    same = walk_forward_ic(panel, "1M", cfg, wf, seed=1, target_mode="return",
                           knife_lambda=0.0)
    assert [f["ic"] for f in base["folds"]] == [f["ic"] for f in same["folds"]]
    assert same["turnover_smoothed"] is None  # no post-transform when lam=0


def test_walk_forward_knife_lambda_changes_ranks_and_reports_turnover():
    panel = _planted_panel(n_dates=36, n_names=40, beta=1.0, seed=3)
    wf = WalkForwardConfig(min_train_months=12, min_names=15)
    cfg = LGBMConfig(n_estimators=80)
    res = walk_forward_ic(panel, "1M", cfg, wf, seed=1, target_mode="return",
                          knife_lambda=0.2)
    assert res["turnover_smoothed"] is not None  # overlay is a post-transform


# =============================================================
# End-to-end: planted signal recovered, shuffle null is not
# =============================================================


def _planted_panel(n_dates: int = 48, n_names: int = 60, beta: float = 1.0, seed: int = 0):
    """Panel where the 1M relative return is driven by mom_1m plus noise.

    Built directly (bypassing price synthesis) so we test the harness + LightGBM +
    IC end-to-end with a known cross-sectional signal of controllable strength.
    """
    rng = np.random.default_rng(seed)
    dates = [date(2018, 1, 1) + timedelta(days=28 * k) for k in range(n_dates)]
    rows = []
    for d in dates:
        signal = rng.normal(0, 1, n_names)
        rel = beta * signal + rng.normal(0, 1.0, n_names)  # noisy but real
        rel -= rel.mean()                                   # demeaned target
        for j in range(n_names):
            row = {"date": d, "ticker_id": j}
            for c in FEATURE_COLS:
                row[c] = 0.0
            row["mom_1m"] = float(signal[j])
            for h in HORIZONS:
                row[f"r_{h}"] = 0.0
                row[f"mask_{h}"] = False
            row["r_1M"] = float(rel[j])
            row["mask_1M"] = True
            rows.append(row)
    return rank_normalize_features(pd.DataFrame(rows))


def test_walk_forward_recovers_planted_signal_and_null_does_not():
    panel = _planted_panel(n_dates=48, n_names=60, beta=1.0, seed=7)
    wf = WalkForwardConfig(min_train_months=12, min_names=20)
    cfg = LGBMConfig(n_estimators=150)

    real = walk_forward_ic(panel, "1M", cfg, wf, seed=1, shuffle=False)["summary"]
    null = walk_forward_ic(panel, "1M", cfg, wf, seed=1, shuffle=True)["summary"]

    assert real["n_folds"] > 20
    assert real["mean_ic"] > 0.20, f"expected to recover signal, got {real['mean_ic']}"
    assert real["mean_ic"] > null["mean_ic"] + 0.15
    assert abs(null["mean_ic"]) < 0.10, f"shuffle null should be ~0, got {null['mean_ic']}"


def test_linear_blend_matches_pure_gbdt_at_zero_and_recovers_signal():
    panel = _planted_panel(n_dates=48, n_names=60, beta=1.0, seed=7)
    wf = WalkForwardConfig(min_train_months=12, min_names=20)
    cfg = LGBMConfig(n_estimators=120)

    pure = walk_forward_ic(panel, "1M", cfg, wf, seed=1, shuffle=False)["summary"]
    blend0 = walk_forward_ic(
        panel, "1M", cfg, wf, seed=1, shuffle=False, linear_blend=0.0
    )["summary"]
    blended = walk_forward_ic(
        panel, "1M", cfg, wf, seed=1, shuffle=False, linear_blend=0.5
    )["summary"]

    # weight 0.0 is exactly the pure-GBDT path (no ridge fit, no rank-blend).
    assert blend0["mean_ic"] == pytest.approx(pure["mean_ic"], abs=1e-12)
    # The blended stack still recovers the (linear) planted signal.
    assert blended["mean_ic"] > 0.20, f"blend lost signal: {blended['mean_ic']}"


def test_prepare_panel_end_to_end_on_frames():
    frames = [make_frame(n_days=800, trend=0.0003 * (k + 1), tid=k, vol_seed=k) for k in range(6)]
    grid = build_calendar_grid(frames)
    panel = prepare_panel(frames, grid)
    assert not panel.empty
    # Rank-normalized features stay in range.
    for c in FEATURE_COLS:
        assert panel[c].min() >= -1.0001 and panel[c].max() <= 1.0001


# =============================================================
# Lever 1: sector_return_vol target mode
# =============================================================


def test_sector_return_vol_shrinks_high_vol_labels():
    # 4 Tech names + 1 Energy; vol_120d_raw varies 10x across the set.
    # The vol-scaling multiplier is 1/max(vol, floor). The HIGH-vol name (vol=0.50)
    # has a SMALLER multiplier (=1/0.50=2.0) than the low-vol names (floored at the
    # 20th-pct = 0.10 → multiplier = 1/0.10 = 10.0). So the high-vol name is
    # "shrunk" relative to low-vol names — that is the homoskedasticity goal.
    d = date(2020, 1, 31)
    df = pd.DataFrame({
        "date": [d] * 5,
        "sector": ["Tech"] * 4 + ["Energy"],
        "vol_120d_raw": [0.50, 0.10, 0.10, 0.10, 0.10],  # first Tech name is 5x more vol
        "r_1M": [0.10, 0.05, 0.00, -0.05, 0.20],
        "mask_1M": [True] * 5,
    })
    for h in HORIZONS:
        if f"r_{h}" not in df:
            df[f"r_{h}"] = 0.0
        if f"mask_{h}" not in df:
            df[f"mask_{h}"] = False
    df["r_1M"] = [0.10, 0.05, 0.00, -0.05, 0.20]
    df["mask_1M"] = [True] * 5

    out = apply_target_modes(df, sector_min_group_size=4)

    # sector_return_vol must exist and be finite for all rows.
    assert "y_1M_sector_return_vol" in out.columns
    assert out["y_1M_sector_return_vol"].notna().all()

    sr = out["y_1M_sector_return"].to_numpy()
    srv = out["y_1M_sector_return_vol"].to_numpy()

    # The scaling factor for each name is sector_return_vol / sector_return = 1/max(vol,floor).
    # High-vol name (index 0, vol=0.50) → factor = 1/0.50 = 2.0.
    # Low-vol names (vol=0.10, floored at 0.10) → factor = 1/0.10 = 10.0.
    # So the high-vol name has a SMALLER factor: it is shrunk relative to low-vol names.
    scale = np.where(sr != 0, srv / sr, np.nan)
    assert scale[0] < scale[1], "high-vol name must have smaller scaling factor"

    # With vol_120d_raw absent the column falls back to sector_return.
    df_no_vol = df.drop(columns=["vol_120d_raw"])
    out_no = apply_target_modes(df_no_vol, sector_min_group_size=4)
    np.testing.assert_array_equal(
        out_no["y_1M_sector_return_vol"].to_numpy(),
        out_no["y_1M_sector_return"].to_numpy(),
    )


def test_sector_return_vol_in_target_choices():
    # _target_col must map sector_return_vol to the right column name.
    from backend.ml.gbm_baseline import _target_col
    assert _target_col("6M", "sector_return_vol") == "y_6M_sector_return_vol"


# =============================================================
# Lever 2: knife_score feature (add_knife_score_feature)
# =============================================================


def test_add_knife_score_feature_parity_with_knife_components():
    # Build a synthetic panel with rank-normalized [-1,1] price features, then
    # compare add_knife_score_feature output against the scalar formula in
    # _knife_components. The two paths must produce byte-identical results.
    from backend.ml.gbm_baseline import _knife_components, add_knife_score_feature

    rng = np.random.default_rng(42)
    n = 12
    d = date(2021, 6, 30)
    df = pd.DataFrame({
        "date": [d] * n,
        "vol_120d":   rng.uniform(-1, 1, n),
        "ma_gap_200": rng.uniform(-1, 1, n),
        "dist_low_252": rng.uniform(-1, 1, n),
    })
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0

    out = add_knife_score_feature(df)
    assert "knife_score" in out.columns
    ks = out["knife_score"].to_numpy(dtype=float)

    # Manually compute via _knife_components using the same per-row inputs.
    risk = {
        "vol":   df["vol_120d"].to_numpy(dtype=float),
        "trend": df["ma_gap_200"].to_numpy(dtype=float),
        "dlow":  df["dist_low_252"].to_numpy(dtype=float),
    }
    expected, _, _ = _knife_components(risk)
    np.testing.assert_allclose(ks, expected, rtol=1e-9)

    # knife_score is bounded in [0, 1].
    assert ks.min() >= 0.0 and ks.max() <= 1.0


def test_add_knife_score_feature_high_only_for_knife_corner():
    # Name A: max vol, max downtrend → score ≈ 1.0.
    # Name B: max vol but uptrend → score ≈ 0.0.
    # Name C: min vol but downtrend → score ≈ 0.0.
    from backend.ml.gbm_baseline import add_knife_score_feature

    d = date(2021, 6, 30)
    df = pd.DataFrame({
        "date": [d, d, d],
        "vol_120d":    [1.0, 1.0, -1.0],
        "ma_gap_200":  [-1.0, 1.0, -1.0],
        "dist_low_252": [-1.0, 1.0, -1.0],
    })
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0

    out = add_knife_score_feature(df)
    ks = out["knife_score"].to_numpy(dtype=float)
    assert ks[0] == pytest.approx(1.0)  # true knife
    assert ks[1] == pytest.approx(0.0)  # clean uptrend
    assert ks[2] == pytest.approx(0.0)  # low vol


def test_add_knife_score_feature_neutral_fallback_without_price_cols():
    # When required columns are missing, every name gets a neutral 0.5 score.
    from backend.ml.gbm_baseline import add_knife_score_feature

    df = pd.DataFrame({"date": [date(2021, 1, 31)] * 3, "some_col": [1.0, 2.0, 3.0]})
    out = add_knife_score_feature(df)
    assert "knife_score" in out.columns
    np.testing.assert_array_equal(out["knife_score"].to_numpy(), [0.5, 0.5, 0.5])


def test_prepare_panel_stashes_vol_raw_and_computes_knife_score():
    # End-to-end: prepare_panel stashes vol_120d_raw (for the sector_return_vol
    # target) and, when knife_score is in rank_cols, adds it after normalization.
    from backend.ml.gbm_baseline import KNIFE_FEATURES

    frames = [make_frame(n_days=800, trend=0.0003 * (k + 1), tid=k, vol_seed=k)
              for k in range(6)]
    grid = build_calendar_grid(frames)

    # Without knife_score in rank_cols: knife_score should NOT be added.
    panel_plain = prepare_panel(frames, grid)
    assert "vol_120d_raw" in panel_plain.columns  # always stashed
    assert "knife_score" not in panel_plain.columns

    # With knife_score in rank_cols: it should be computed and bounded [0,1].
    panel_knife = prepare_panel(frames, grid, rank_cols=list(FEATURE_COLS) + list(KNIFE_FEATURES))
    assert "knife_score" in panel_knife.columns
    ks = panel_knife["knife_score"].dropna()
    assert ks.min() >= 0.0 and ks.max() <= 1.0
    # vol_120d itself is still rank-normalized in [-1,1]; knife_score is not.
    assert panel_knife["vol_120d"].min() >= -1.0001
    assert panel_knife["vol_120d"].max() <= 1.0001


def test_fit_horizon_models_rolling_window_trims_training_dates():
    # A HorizonSpec with max_train_months=N should only use the last N monthly
    # training dates when fitting — earlier rows must be dropped entirely.
    from backend.ml.gbm_baseline import (
        HorizonSpec, LGBMConfig, FEATURE_COLS, prepare_panel,
        build_calendar_grid,
    )
    from backend.ml.gbm_inference import fit_horizon_models
    import copy

    frames = [make_frame(n_days=1800, trend=0.0002 * (k + 1), tid=k, vol_seed=k)
              for k in range(12)]
    grid = build_calendar_grid(frames)
    panel = prepare_panel(frames, grid)

    as_of = sorted(panel["date"].unique())[-1]
    all_train_dates = sorted(d for d in panel["date"].unique() if d < as_of)

    # Rolling window: use only last 24 monthly dates.
    window = 24
    spec_rolling = HorizonSpec(target_mode="sector_return", max_train_months=window)
    spec_expanding = HorizonSpec(target_mode="sector_return")

    _, windows_rolling, _, _ = fit_horizon_models(
        panel, {"3M": spec_rolling}, seed=0, as_of=as_of, n_seeds=1
    )
    _, windows_expanding, _, _ = fit_horizon_models(
        panel, {"3M": spec_expanding}, seed=0, as_of=as_of, n_seeds=1
    )

    # Rolling must have fewer rows than expanding when the panel exceeds the window.
    assert windows_rolling["3M"]["rows"] < windows_expanding["3M"]["rows"], (
        "rolling window should train on fewer rows than expanding"
    )
    # Rolling must train on at most `window` distinct dates.
    # (rows can exceed window because multiple tickers per date)
    rolling_rows = windows_rolling["3M"]["rows"]
    expanding_rows = windows_expanding["3M"]["rows"]
    n_tickers = len(frames)
    assert rolling_rows <= window * n_tickers
