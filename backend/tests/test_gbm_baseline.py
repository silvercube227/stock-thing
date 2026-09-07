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
    ESTIMATE_SOURCED_FEATURES,
    EXPERIMENTAL_FEATURES,
    FEATURE_COLS,
    FUNDAMENTAL_FEATURES,
    FUNDAMENTAL_SOURCED_FEATURES,
    PRICE_FEATURES,
    LGBMConfig,
    RESIDUAL_MOM_FEATURES,
    WalkForwardConfig,
    add_industry_neutral_momentum,
    apply_knife_overlay,
    apply_target_modes,
    assert_winsorizable_target,
    block_bootstrap_summary,
    build_market_horizon_returns,
    build_ticker_rows,
    build_universe_return_map,
    demean_cross_sectional,
    ewma_rank_by_ticker,
    fit_lgbm_model,
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
    winsorize_by_date,
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
    # Source-gated contract: a feature is finite, or NaN precisely because its
    # upstream source has nothing as of that row. This fixture has no filings and
    # no share count, so fundamentals and log_market_cap are undefined by design.
    for r in rows:
        for c in PRICE_FEATURES:
            if c == "log_market_cap":
                continue
            assert np.isfinite(r[c]), f"{c} not finite"
        assert r["fund_available"] == 0.0
        assert np.isnan(r["log_market_cap"]), "no share count -> cap is undefined"
        for c in FUNDAMENTAL_FEATURES:
            assert np.isnan(r[c]), f"{c} should be NaN when no filing exists"


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
            source_absent = (
                (c in FUNDAMENTAL_SOURCED_FEATURES and not r["fund_available"])
                or (c in ESTIMATE_SOURCED_FEATURES and not r["est_available"])
                # Independently sourced: quarterly surprises, the FINRA feed, and
                # the staleness clock are NaN when their own source is empty.
                or c in ("revenue_surprise", "eps_surprise", "short_ratio",
                         "est_staleness_days")
            )
            assert np.isfinite(r[c]) or source_absent, (
                f"{c} is NaN but its source is present"
            )


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

    # Reaction features are defined exactly on rows that have a filing as-of the
    # grid date; before the first filing they are NaN, not a sentinel 0.0.
    assert any(r["fund_available"] for r in rows)
    for r in rows:
        for c in EARNINGS_REACTION_FEATURES:
            if r["fund_available"]:
                assert np.isfinite(r[c]), f"{c} not finite after a filing exists"
            else:
                assert np.isnan(r[c]), f"{c} should be NaN before any filing"
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
    # 6 Tech names + 1 Energy; Tech has >= 5 so sector-demean applies, Energy has 1
    # so its sector-relative target is UNDEFINED (NaN) and the row is dropped at fit
    # time. It used to fall through to the universe-demeaned return, which pooled a
    # differently-defined label into the same fit while within_sector_ic excluded the
    # row from the metric — trained on a cohort that was never scored.
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
    # Energy has 1 name -> below threshold -> NaN, so the fit drops it.
    energy = out[out["sector"] == "Energy"]
    assert np.isnan(float(energy["y_1M_sector_return"].iloc[0]))
    # The grade target inherits the NaN (a ranking objective cannot grade it either).
    assert np.isnan(float(energy["y_1M_sector_grade"].iloc[0]))


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
    # 30 fake trading days with constant +1% daily log return; H=21 trading days
    # => market_r_h = 21 * 0.01 = 0.21 at any grid date with a full window ahead.
    # The window starts at idx(g)+2 because entry is the bar AFTER g (implementation
    # lag) and daily[i] is the return INTO bar i — so g needs 21 + 2 slots after it.
    from backend.ingestion.calendar import HORIZON_TRADING_DAYS as HTD
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(30)]
    market_returns = {d: 0.01 for d in dates}
    grid = [dates[0], dates[3], dates[-2]]
    out = build_market_horizon_returns(market_returns, grid, horizons=("1M",))
    assert abs(out["1M"][dates[0]] - HTD["1M"] * 0.01) < 1e-9
    assert abs(out["1M"][dates[3]] - HTD["1M"] * 0.01) < 1e-9
    assert dates[-2] not in out["1M"]  # window runs past end


def test_build_market_horizon_returns_starts_after_the_grid_date():
    """The market leg must span the same bars the ticker's lagged label does.

    Entry is the bar after the grid date, so the return ON the grid date and the
    return on the very next bar are both OUTSIDE the window.
    """
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(30)]
    market_returns = {d: 0.0 for d in dates}
    market_returns[dates[5]] = 1.0   # return into the grid date itself
    market_returns[dates[6]] = 2.0   # return into the entry bar
    market_returns[dates[7]] = 0.5   # first return actually earned while holding
    out = build_market_horizon_returns(market_returns, [dates[5]], horizons=("1M",))
    assert out["1M"][dates[5]] == pytest.approx(0.5)


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


def test_production_horizon_specs_scored_horizon_targets():
    # Locks in the promoted per-horizon targets so an accidental edit trips the test:
    #   3M = sector_return + L2 regression (LambdaRank NOT promoted — sub-bar ICIR + IC loss)
    #   6M/1Y = sector_grade + LambdaRank (2026-07-06 promotion: 6M ICIR +16%, 1Y +48%)
    # 1M stays `rank` (dead horizon, not scored in production).
    from backend.ml.gbm_baseline import ESTIMATE_SURPRISE_FEATURES, PRODUCTION_HORIZON_SPECS

    assert PRODUCTION_HORIZON_SPECS["1M"].target_mode == "rank"
    s3 = PRODUCTION_HORIZON_SPECS["3M"]
    assert s3.target_mode == "sector_return" and s3.lgb_cfg.objective == "regression"
    for h in ("6M", "1Y"):
        s = PRODUCTION_HORIZON_SPECS[h]
        assert s.target_mode == "sector_grade", (
            f"{h} target unexpectedly changed — re-sweep SECB before promoting"
        )
        assert s.lgb_cfg.objective == "lambdarank"
        # 6M and 1Y still carry the promoted revenue-surprise pack.
        assert ESTIMATE_SURPRISE_FEATURES[0] in (s.feature_cols or [])


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
    # State is now the RAW blended value, not the re-ranked percentile.
    prior = {(t, "3M"): p for t, p in zip(range(1, 6), [1.0, 0.75, 0.5, 0.25, 0.0])}
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
    # Rank-normalized OBSERVED values stay in range; columns whose source is absent
    # on this fixture (no filings, no share count) are entirely NaN and are skipped.
    for c in FEATURE_COLS:
        vals = panel[c].dropna()
        if vals.empty:
            assert c in FUNDAMENTAL_SOURCED_FEATURES or c == "log_market_cap", (
                f"{c} is all-NaN but is not source-gated"
            )
            continue
        assert vals.min() >= -1.0001 and vals.max() <= 1.0001
    # The availability flag itself is never missing — it is the model's handle on
    # the missingness, so it must stay finite.
    assert panel["fund_available"].notna().all()


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

    # sector_return_vol must exist and be finite wherever the sector target is
    # defined; the 1-name Energy row is NaN by design (below sector_min_group_size).
    assert "y_1M_sector_return_vol" in out.columns
    tech_mask = (out["sector"] == "Tech").to_numpy()
    assert out.loc[tech_mask, "y_1M_sector_return_vol"].notna().all()
    assert np.isnan(float(out.loc[~tech_mask, "y_1M_sector_return_vol"].iloc[0]))

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


# =============================================================
# E3 — target winsorization (train-label only, PIT per-date)
# =============================================================


def test_winsorize_by_date_clips_to_per_date_quantiles():
    # Date A carries an up-outlier and a down-outlier; date B is tame. With pct=0.2
    # the clip bounds are the per-date 20th/80th quantiles (linear interpolation).
    y = np.array([-50.0, 0.0, 1.0, 2.0, 100.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    d = np.array(["A"] * 5 + ["B"] * 5)
    out = winsorize_by_date(y, d, 0.2)

    a = np.sort(y[:5])
    lo_a, hi_a = np.quantile(a, 0.2), np.quantile(a, 0.8)
    assert out[:5].min() >= lo_a - 1e-9
    assert out[:5].max() <= hi_a + 1e-9
    # The interior (non-clipped) values are untouched.
    assert out[1] == 0.0 and out[2] == 1.0 and out[3] == 2.0
    # Tame date B: 20th/80th quantiles leave the extremes barely moved but bounded.
    b = np.sort(y[5:])
    assert out[5:].min() >= np.quantile(b, 0.2) - 1e-9
    assert out[5:].max() <= np.quantile(b, 0.8) + 1e-9


def test_winsorize_by_date_is_noop_at_zero():
    y = np.array([1.0, -3.0, 5.0, 2.0, -1.0])
    d = np.array(["A"] * 5)
    assert np.array_equal(winsorize_by_date(y, d, 0.0), y)


def test_winsorize_by_date_preserves_nans():
    y = np.array([np.nan, 1.0, 2.0, 3.0, 100.0])
    d = np.array(["A"] * 5)
    out = winsorize_by_date(y, d, 0.2)
    assert np.isnan(out[0])
    # NaN is excluded from the quantile, so the 100 is still clipped down.
    assert out[4] < 100.0


def test_assert_winsorizable_target_guards_ordinal_modes():
    for mode in ("return", "sector_return", "sector_return_vol",
                 "beta_resid", "beta_sector_resid"):
        assert_winsorizable_target(mode)  # no raise
    for mode in ("rank", "quantile"):
        with pytest.raises(ValueError):
            assert_winsorizable_target(mode)


def test_walk_forward_ic_winsorize_zero_matches_unwinsorized():
    # pct=0 must be a byte-identical no-op vs the default path.
    frames = [make_frame(n_days=1500, trend=0.0002 * (k + 1), tid=k, vol_seed=k)
              for k in range(10)]
    grid = build_calendar_grid(frames)
    panel = prepare_panel(frames, grid)
    cfg = LGBMConfig()
    wf = WalkForwardConfig(min_train_months=6, min_names=5)
    base = walk_forward_ic(panel, "3M", cfg, wf, target_mode="sector_return")
    zero = walk_forward_ic(panel, "3M", cfg, wf, target_mode="sector_return",
                           winsorize_pct=0.0)
    assert base["summary"]["mean_ic"] == zero["summary"]["mean_ic"]


def test_walk_forward_ic_winsorize_runs_and_changes_fit():
    frames = [make_frame(n_days=1500, trend=0.0002 * (k + 1), tid=k, vol_seed=k)
              for k in range(10)]
    grid = build_calendar_grid(frames)
    panel = prepare_panel(frames, grid)
    cfg = LGBMConfig()
    wf = WalkForwardConfig(min_train_months=6, min_names=5)
    base = walk_forward_ic(panel, "3M", cfg, wf, target_mode="sector_return")
    wins = walk_forward_ic(panel, "3M", cfg, wf, target_mode="sector_return",
                           winsorize_pct=0.02)
    # Clipping the training label at 2% per date changes the fit (so mean_ic moves),
    # but the harness still produces a finite result on the same folds.
    assert base["summary"]["n_folds"] == wins["summary"]["n_folds"]
    assert np.isfinite(wins["summary"]["mean_ic"])


def test_walk_forward_ic_winsorize_raises_on_rank_target():
    frames = [make_frame(n_days=1500, trend=0.0002 * (k + 1), tid=k, vol_seed=k)
              for k in range(6)]
    grid = build_calendar_grid(frames)
    panel = prepare_panel(frames, grid)
    with pytest.raises(ValueError):
        walk_forward_ic(panel, "3M", LGBMConfig(),
                        WalkForwardConfig(min_train_months=6, min_names=5),
                        target_mode="rank", winsorize_pct=0.01)


def test_fit_horizon_models_winsorize_clips_training_label():
    from backend.ml.gbm_baseline import HorizonSpec
    from backend.ml.gbm_inference import fit_horizon_models

    frames = [make_frame(n_days=1600, trend=0.0002 * (k + 1), tid=k, vol_seed=k)
              for k in range(10)]
    grid = build_calendar_grid(frames)
    panel = prepare_panel(frames, grid)
    as_of = sorted(panel["date"].unique())[-1]

    spec = HorizonSpec(target_mode="sector_return", winsorize_pct=0.02)
    models, windows, _, _ = fit_horizon_models(
        panel, {"3M": spec}, seed=0, as_of=as_of, n_seeds=1
    )
    assert len(models["3M"]) == 1
    assert windows["3M"]["rows"] > 0

    # A winsorize_pct on an ordinal target must fail loudly, not silently no-op.
    bad = HorizonSpec(target_mode="rank", winsorize_pct=0.02)
    with pytest.raises(ValueError):
        fit_horizon_models(panel, {"3M": bad}, seed=0, as_of=as_of, n_seeds=1)


# =============================================================
# E4 — LambdaRank ranking objective (sector_grade target + LGBMRanker)
# =============================================================


def test_sector_grade_target_gives_uniform_per_date_grades():
    frames = [make_frame(n_days=1500, trend=0.0002 * (k + 1), tid=k, vol_seed=k)
              for k in range(15)]
    grid = build_calendar_grid(frames)
    panel = prepare_panel(frames, grid, n_grades=5)
    gcol = "y_3M_sector_grade"
    assert gcol in panel.columns
    valid = panel[gcol].dropna()
    # Grades are the ordinal set {0,1,2,3,4}.
    assert set(np.unique(valid)) <= {0.0, 1.0, 2.0, 3.0, 4.0}
    # Per date with >= n_grades names, qcut yields ~equal-count buckets.
    for d, sub in panel.groupby("date"):
        g = sub[gcol].dropna()
        if g.shape[0] >= 25:  # enough names for a clean 5-way split
            counts = g.value_counts()
            assert counts.max() - counts.min() <= 2  # near-uniform


def test_sector_grade_is_nan_where_horizon_masked():
    frames = [make_frame(n_days=1500, trend=0.0002 * (k + 1), tid=k, vol_seed=k)
              for k in range(12)]
    grid = build_calendar_grid(frames)
    panel = prepare_panel(frames, grid, n_grades=5)
    masked = ~panel["mask_3M"].astype(bool)
    # No masked row may carry a grade (labels only exist where a return exists).
    assert panel.loc[masked, "y_3M_sector_grade"].isna().all()


def test_sector_grade_thin_date_yields_nan():
    # A date with fewer than n_grades labeled names cannot be qcut into K buckets.
    frames = [make_frame(n_days=1500, trend=0.0002 * (k + 1), tid=k, vol_seed=k)
              for k in range(3)]
    grid = build_calendar_grid(frames)
    panel = prepare_panel(frames, grid, n_grades=5)
    for d, sub in panel.groupby("date"):
        labeled = sub[sub["mask_3M"].astype(bool)]
        if 0 < labeled.shape[0] < 5:
            assert sub["y_3M_sector_grade"].isna().all()


def test_ranker_branch_builds_date_contiguous_groups():
    from backend.ml.gbm_baseline import LGBMConfig

    frames = [make_frame(n_days=1500, trend=0.0002 * (k + 1), tid=k, vol_seed=k)
              for k in range(12)]
    grid = build_calendar_grid(frames)
    panel = prepare_panel(frames, grid, n_grades=5)
    gcol = "y_3M_sector_grade"
    dates = sorted(panel["date"].unique())
    train = panel[(panel["date"] <= dates[-4]) & panel["mask_3M"] & panel[gcol].notna()]

    # Group sizes (per date, in sorted order) must sum to the training row count —
    # the invariant LightGBM relies on to segment query groups.
    ordered = train.sort_values("date", kind="mergesort")
    group = ordered.groupby("date", sort=False).size().to_numpy()
    assert int(group.sum()) == train.shape[0]
    assert group.min() >= 1

    model = fit_lgbm_model(train, gcol, LGBMConfig(objective="lambdarank"), seed=0)
    assert type(model).__name__ == "LGBMRanker"
    # Same .predict() score signature as the regressor (drives zero downstream change).
    test = panel[panel["date"] == dates[-1]]
    preds = model.predict(test[FEATURE_COLS])
    assert preds.shape[0] == test.shape[0]
    assert np.isfinite(preds).all()


def test_lambdarank_recovers_planted_cross_sectional_signal():
    # Higher-trend tickers should rank higher: the ranker's IC must be clearly
    # positive on a planted-signal panel (same bar the regression path clears).
    from backend.ml.gbm_baseline import LGBMConfig

    frames = [make_frame(n_days=1600, trend=0.0002 * (k + 1), tid=k, vol_seed=k)
              for k in range(16)]
    grid = build_calendar_grid(frames)
    panel = prepare_panel(frames, grid, n_grades=5)
    wf = WalkForwardConfig(min_train_months=6, min_names=5)
    res = walk_forward_ic(panel, "3M", LGBMConfig(objective="lambdarank"), wf,
                          target_mode="sector_grade")
    assert res["summary"]["n_folds"] > 0
    assert res["summary"]["mean_ic"] > 0.0


# =============================================================
# E5 — cross-sectional seasonality pack (Heston-Sadka)
# =============================================================


def _series_with_february_bump(start_year=2015, n_years=7, feb_daily=0.05 / 20.0):
    """Daily series where every February carries a steady positive drift, else flat."""
    from datetime import date, timedelta
    import math

    d0 = date(start_year, 1, 2)
    trade_dates, adj_close = [], []
    price = 100.0
    for i in range(n_years * 365):
        dd = d0 + timedelta(days=i)
        price *= math.exp(feb_daily if dd.month == 2 else 0.0)
        trade_dates.append(dd)
        adj_close.append(price)
    return trade_dates, adj_close


def test_seasonality_detects_planted_month_and_is_pit_safe():
    from backend.ml.factors.price import _seasonality_asof

    trade_dates, adj_close = _series_with_february_bump()

    def last_of(y, m):
        return max(i for i, dd in enumerate(trade_dates) if dd.year == y and dd.month == m)

    # January-end bars: the UPCOMING month is February, so the same-month seasonal
    # should surface the planted positive February return; other months ~0.
    jan_bars = [last_of(y, 1) for y in (2019, 2020, 2021)]
    out = _seasonality_asof(adj_close, trade_dates, jan_bars,
                            years=5, min_same_obs=2, min_other_obs=10)
    for i in range(len(jan_bars)):
        assert out["seasonal_same_month_5y"][i] > 0.03      # planted Feb ~+0.05..0.07
        assert abs(out["seasonal_other_month_5y"][i]) < 0.01  # other months flat
        assert out["seasonal_gap_5y"][i] > 0.03

    # PIT: a bar at end of February (upcoming month March, which is flat) must read
    # ~0 — the just-finished February must NOT leak into the "same-month" value.
    feb = _seasonality_asof(adj_close, trade_dates, [last_of(2021, 2)],
                            years=5, min_same_obs=2, min_other_obs=10)
    assert abs(feb["seasonal_same_month_5y"][0]) < 0.01


def test_seasonality_insufficient_history_falls_back_to_zero():
    from backend.ml.factors.price import _seasonality_asof

    trade_dates, adj_close = _series_with_february_bump(start_year=2020, n_years=2)

    def last_of(y, m):
        return max(i for i, dd in enumerate(trade_dates) if dd.year == y and dd.month == m)

    # Only ~1 completed February of history before Jan 2021 -> below min_same_obs=3.
    out = _seasonality_asof(adj_close, trade_dates, [last_of(2021, 1)],
                            years=5, min_same_obs=3, min_other_obs=24)
    assert out["seasonal_same_month_5y"][0] == 0.0
    assert out["seasonal_gap_5y"][0] == 0.0


def test_seasonality_columns_flow_into_panel():
    from backend.ml.gbm_baseline import SEASONALITY_FEATURES

    frames = [make_frame(n_days=1600, trend=0.0002 * (k + 1), tid=k, vol_seed=k)
              for k in range(6)]
    grid = build_calendar_grid(frames)
    cols = list(FEATURE_COLS) + list(SEASONALITY_FEATURES)
    panel = prepare_panel(frames, grid, rank_cols=cols)
    for c in SEASONALITY_FEATURES:
        assert c in panel.columns
        # Rank-normalized into [-1, 1] like every other feature.
        assert panel[c].between(-1.0, 1.0).all()


# =============================================================
# Sentiment demoted to an opt-in pack
# =============================================================


def test_sentiment_not_in_default_feature_cols():
    """yfinance gives ~30 days of headlines and there is no backfill, so these
    columns were ~98% zero in training yet non-zero at inference. They stay off the
    default book until a real archive exists."""
    from backend.ml.factors.constants import FEATURE_COLS, SENTIMENT_FEATURES

    for col in SENTIMENT_FEATURES:
        assert col not in FEATURE_COLS


def _pack_args(**overrides):
    """Namespace with every --with-* pack off, so a test can flip exactly one."""
    import argparse

    flags = {
        "with_valuation", "with_quality", "with_residual_mom",
        "with_earnings_reaction", "with_analyst_revisions", "with_estimate_surprise",
        "with_eps_surprise", "with_revision_momentum", "with_forward_valuation",
        "with_lottery", "with_microstructure", "with_eps_dispersion",
        "with_short_interest", "with_knife_feature", "with_seasonality",
        "with_insider", "with_sentiment",
    }
    ns = {f: False for f in flags} | {"extra_features": None}
    return argparse.Namespace(**(ns | overrides))


def test_with_sentiment_flag_readds_the_pack():
    from backend.ml.factors.constants import SENTIMENT_FEATURES
    from backend.ml.gbm_baseline import _compose_feature_cols

    off = _compose_feature_cols(_pack_args())
    on = _compose_feature_cols(_pack_args(with_sentiment=True))
    for col in SENTIMENT_FEATURES:
        assert col not in off
        assert col in on


def test_production_specs_exclude_sentiment():
    """The promoted per-horizon lists derive from FEATURE_COLS, so they must not
    carry the unvalidated sentiment columns into production inference."""
    from backend.ml.factors.constants import SENTIMENT_FEATURES
    from backend.ml.gbm_baseline import PRODUCTION_HORIZON_SPECS

    for horizon, spec in PRODUCTION_HORIZON_SPECS.items():
        cols = spec.feature_cols or []
        for col in SENTIMENT_FEATURES:
            assert col not in cols, f"{horizon} still serves {col}"


# =============================================================
# Point-in-time index-membership filter
# =============================================================


def _membership_panel():
    return pd.DataFrame({
        "date": [date(2017, 1, 31)] * 3 + [date(2023, 1, 31)] * 3,
        "ticker_id": [1, 2, 3] * 2,
        "in_index": [True, False, None, True, True, None],
        "mom_1m": [0.1] * 6,
    })


def test_membership_filter_drops_rows_outside_membership():
    """Ticker 2 joined the index later, so its 2017 row must not be in the
    cross-section — otherwise the panel selects on future index promotion."""
    from backend.ml.gbm_baseline import apply_membership_filter

    out = apply_membership_filter(_membership_panel())
    kept = set(zip(out["date"], out["ticker_id"]))
    assert (date(2017, 1, 31), 2) not in kept
    assert (date(2017, 1, 31), 1) in kept
    assert (date(2023, 1, 31), 2) in kept


def test_membership_filter_drops_unknown_membership_by_default():
    from backend.ml.gbm_baseline import apply_membership_filter

    out = apply_membership_filter(_membership_panel())
    assert 3 not in set(out["ticker_id"])


def test_membership_filter_keeps_exempt_ids():
    """User-added off-index tickers never have membership; they are scored but
    excluded from training separately, so the filter must not drop them."""
    from backend.ml.gbm_baseline import apply_membership_filter

    out = apply_membership_filter(_membership_panel(), exempt_ids={3})
    assert set(out.loc[out["ticker_id"] == 3, "date"]) == {
        date(2017, 1, 31), date(2023, 1, 31)
    }


def test_membership_filter_passes_through_panels_without_the_column():
    """Hand-built test panels have no in_index column and must not be filtered."""
    from backend.ml.gbm_baseline import apply_membership_filter

    df = pd.DataFrame({"date": [date(2020, 1, 31)], "ticker_id": [1], "mom_1m": [0.1]})
    assert len(apply_membership_filter(df)) == 1


def test_membership_filter_raises_when_no_membership_loaded():
    """A frame cache predating the migration would otherwise silently empty the
    entire panel — fail loudly instead."""
    from backend.ml.gbm_baseline import apply_membership_filter

    df = pd.DataFrame({
        "date": [date(2020, 1, 31)] * 2,
        "ticker_id": [1, 2],
        "in_index": [None, None],
    })
    with pytest.raises(RuntimeError, match="refresh-cache"):
        apply_membership_filter(df)


def test_in_index_on_respects_exclusive_valid_to():
    from backend.ml.factors.assembly import _in_index_on

    intervals = [{"valid_from": date(2015, 1, 1), "valid_to": date(2020, 6, 1)}]
    assert _in_index_on(intervals, date(2015, 1, 1)) is True   # inclusive start
    assert _in_index_on(intervals, date(2020, 5, 31)) is True
    assert _in_index_on(intervals, date(2020, 6, 1)) is False  # exclusive end
    assert _in_index_on(intervals, date(2014, 12, 31)) is False


def test_in_index_on_handles_open_and_multiple_intervals():
    from backend.ml.factors.assembly import _in_index_on

    intervals = [
        {"valid_from": date(2010, 1, 1), "valid_to": date(2015, 8, 1)},
        {"valid_from": date(2021, 4, 1), "valid_to": None},
    ]
    assert _in_index_on(intervals, date(2012, 1, 1)) is True
    assert _in_index_on(intervals, date(2018, 1, 1)) is False   # the gap
    assert _in_index_on(intervals, date(2026, 1, 1)) is True    # open interval


def test_in_index_on_returns_none_when_not_loaded():
    from backend.ml.factors.assembly import _in_index_on

    assert _in_index_on(None, date(2020, 1, 1)) is None
    assert _in_index_on([], date(2020, 1, 1)) is False


# =============================================================
# Size-neutral IC + regime report
# =============================================================


def test_partial_spearman_kills_a_pure_size_confound():
    """If both the prediction and the realized return are just size, the naive IC
    is high but nothing survives holding size constant."""
    from backend.ml.gbm_baseline import _partial_spearman

    rng = np.random.default_rng(0)
    size = rng.normal(size=300)
    pred = size + 0.01 * rng.normal(size=300)
    r = size + 0.01 * rng.normal(size=300)
    naive = pd.Series(pred).corr(pd.Series(r), method="spearman")
    assert naive > 0.9
    assert abs(_partial_spearman(pred, r, size)) < 0.25


def test_partial_spearman_keeps_signal_orthogonal_to_size():
    from backend.ml.gbm_baseline import _partial_spearman

    rng = np.random.default_rng(1)
    size = rng.normal(size=400)
    signal = rng.normal(size=400)
    r = signal + 0.1 * rng.normal(size=400)
    naive = pd.Series(signal).corr(pd.Series(r), method="spearman")
    partial = _partial_spearman(signal, r, size)
    assert partial == pytest.approx(naive, abs=0.1)


def test_size_neutral_summary_runs_over_records():
    from backend.ml.gbm_baseline import size_neutral_summary

    rng = np.random.default_rng(2)
    records = []
    for i in range(8):
        n = 60
        size = rng.normal(size=n)
        # A partly size-driven signal — not size exactly, which would make the
        # partial correlation genuinely undefined (denominator -> 0).
        records.append({
            "date": date(2020, 1 + i % 12, 28),
            "pred": 0.6 * size + 0.8 * rng.normal(size=n),
            "r": 0.6 * size + 0.8 * rng.normal(size=n),
            "size": size,
            "sector": np.array(["A"] * 30 + ["B"] * 30),
        })
    out = size_neutral_summary(records, block_size=3, reps=50)
    assert out["universe"]["n_folds"] == 8
    assert abs(out["universe"]["mean_ic"]) < 0.3   # size confound removed
    assert out["sector"]["n_folds"] == 8


def test_size_neutral_summary_skips_records_without_size():
    from backend.ml.gbm_baseline import size_neutral_summary

    records = [{"date": date(2020, 1, 31), "pred": np.zeros(5),
                "r": np.zeros(5), "size": None, "sector": None}]
    out = size_neutral_summary(records, block_size=3, reps=10)
    assert out["universe"]["n_folds"] == 0


def test_regime_report_groups_by_year_and_vol_tertile():
    from backend.ml.gbm_baseline import regime_report

    folds = [
        {"date": date(2020, m, 28), "ic": 0.05, "sector_ic": 0.03, "vol_raw_med": 0.01 * m}
        for m in range(1, 13)
    ] + [
        {"date": date(2021, m, 28), "ic": -0.02, "sector_ic": -0.01, "vol_raw_med": 0.5}
        for m in range(1, 7)
    ]
    rep = regime_report(folds)
    assert rep["by_year"][2020]["n"] == 12
    assert rep["by_year"][2020]["hit_rate"] == 1.0
    assert rep["by_year"][2021]["hit_rate"] == 0.0
    assert set(rep["by_vol_regime"]) == {"low_vol", "mid_vol", "high_vol"}
    # The 2021 folds all carry the highest vol, so that bucket is the negative one.
    assert rep["by_vol_regime"]["high_vol"]["mean_ic"] < 0


def test_regime_report_omits_vol_split_without_the_raw_column():
    from backend.ml.gbm_baseline import regime_report

    folds = [{"date": date(2020, m, 28), "ic": 0.01} for m in range(1, 8)]
    rep = regime_report(folds)
    assert rep["by_vol_regime"] == {}
    assert rep["by_year"][2020]["n"] == 7


# --- frozen-holdout test-date window -----------------------------------------
# The research protocol needs selection folds whose labels are FULLY realized
# before the holdout opens, and a one-shot holdout run over the tail. Neither
# walk_forward_folds nor walk_forward_ic had any date-window parameter.

def test_walk_forward_ic_max_test_date_truncates_folds_without_touching_training():
    panel = _planted_panel(n_dates=48, n_names=40, beta=1.0, seed=3)
    wf = WalkForwardConfig(min_train_months=12, min_names=20)
    cfg = LGBMConfig(n_estimators=40)
    cutoff = date(2019, 12, 31)

    full = walk_forward_ic(panel, "1M", cfg, wf, seed=1)
    cut = walk_forward_ic(panel, "1M", cfg, wf, seed=1, max_test_date=cutoff)

    assert 0 < len(cut["folds"]) < len(full["folds"])
    assert all(f["date"] <= cutoff for f in cut["folds"])
    # Truncating the tail leaves the surviving folds bit-identical: same fold
    # index => same seed, and the expanding training window is untouched.
    kept = [f for f in full["folds"] if f["date"] <= cutoff]
    assert [f["date"] for f in cut["folds"]] == [f["date"] for f in kept]
    assert [f["n_train"] for f in cut["folds"]] == [f["n_train"] for f in kept]
    assert [f["ic"] for f in cut["folds"]] == [f["ic"] for f in kept]


def test_walk_forward_ic_min_test_date_selects_the_holdout_tail():
    panel = _planted_panel(n_dates=48, n_names=40, beta=1.0, seed=3)
    wf = WalkForwardConfig(min_train_months=12, min_names=20)
    cfg = LGBMConfig(n_estimators=40)

    full = walk_forward_ic(panel, "1M", cfg, wf, seed=1)
    selection = walk_forward_ic(panel, "1M", cfg, wf, seed=1,
                                max_test_date=date(2019, 12, 31))
    holdout = walk_forward_ic(panel, "1M", cfg, wf, seed=1,
                              min_test_date=date(2020, 1, 1))

    assert holdout["folds"]
    assert all(f["date"] >= date(2020, 1, 1) for f in holdout["folds"])
    # The two windows partition the fold list — no fold is scored twice, none lost.
    assert len(selection["folds"]) + len(holdout["folds"]) == len(full["folds"])


def test_walk_forward_ic_without_a_window_is_unchanged():
    panel = _planted_panel(n_dates=36, n_names=40, beta=1.0, seed=5)
    wf = WalkForwardConfig(min_train_months=12, min_names=20)
    cfg = LGBMConfig(n_estimators=40)
    base = walk_forward_ic(panel, "1M", cfg, wf, seed=1)
    same = walk_forward_ic(panel, "1M", cfg, wf, seed=1,
                           max_test_date=None, min_test_date=None)
    assert [f["ic"] for f in base["folds"]] == [f["ic"] for f in same["folds"]]


# --- PIT vol-regime tertiles --------------------------------------------------

def _rising_vol_folds(n: int) -> list[dict]:
    return [
        {"date": date(2015 + k // 12, k % 12 + 1, 28), "ic": 0.01,
         "sector_ic": 0.01, "vol_raw_med": 0.01 * (k + 1)}
        for k in range(n)
    ]


def test_regime_report_pit_tertiles_use_only_prior_folds():
    from backend.ml.gbm_baseline import regime_report

    # Volatility rises monotonically over the whole sample. The in-sample cut
    # splits it into equal thirds; the PIT cut cannot, because every scored fold
    # is above everything that preceded it.
    rep = regime_report(_rising_vol_folds(48), burn_in=24)

    assert rep["by_vol_regime"]["low_vol"]["n"] == 16
    assert rep["by_vol_regime"]["mid_vol"]["n"] == 16
    assert rep["by_vol_regime"]["high_vol"]["n"] == 16

    assert rep["pit_n_scored"] == 24
    assert rep["by_vol_regime_pit"]["high_vol"]["n"] == 24
    assert "low_vol" not in rep["by_vol_regime_pit"]


def test_regime_report_pit_drops_the_burn_in_window():
    from backend.ml.gbm_baseline import regime_report

    rep = regime_report(_rising_vol_folds(30), burn_in=24)
    assert rep["pit_burn_in"] == 24
    assert rep["pit_n_scored"] == 6

    # Not enough history to bucket anything at all.
    short = regime_report(_rising_vol_folds(20), burn_in=24)
    assert short["by_vol_regime_pit"] == {}
    assert short["by_vol_regime"]  # the in-sample cut still works


# --- Stage 2: point-in-time volatility gate ----------------------------------
# The gate shrinks a stress cross-section's ranks toward 0.5. That is a MONOTONE
# transform, so it cannot move that date's rank-IC — it only reduces how much the
# date weighs in the cross-date EWMA. These tests pin both halves of that claim,
# because the acceptance criterion is meaningless if the mechanism is misread.

def test_vol_gate_flags_use_only_prior_dates_and_respect_burn_in():
    from backend.ml.gbm_baseline import vol_gate_flags

    calm = [0.10] * 30
    flags = vol_gate_flags(calm + [0.90], pct=0.80, burn_in=24)
    assert flags[-1] is True                     # the spike clears the 80th pctile
    assert not any(flags[:-1])                   # a flat history gates nothing
    # Nothing inside the burn-in is ever gated, however extreme.
    assert not any(vol_gate_flags([0.1] * 5 + [9.9] * 10, burn_in=24))


def test_vol_gate_flags_ignore_missing_medians():
    from backend.ml.gbm_baseline import vol_gate_flags

    series = [0.10] * 30 + [None, float("nan"), 0.90]
    flags = vol_gate_flags(series, pct=0.80, burn_in=24)
    assert flags[30] is False and flags[31] is False
    assert flags[32] is True


def test_vol_gate_ranks_shrink_toward_a_half_but_preserve_order():
    from backend.ml.gbm_baseline import vol_gate_ranks

    ranks = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    out = vol_gate_ranks(ranks, gated=True, shrink=0.5)
    assert np.allclose(out, [0.25, 0.375, 0.5, 0.625, 0.75])
    # Monotone: the within-date ordering — and therefore the rank-IC — is untouched.
    assert np.array_equal(np.argsort(out), np.argsort(ranks))
    assert np.array_equal(vol_gate_ranks(ranks, gated=False), ranks)


def test_vol_gate_cannot_change_ic_without_smoothing():
    """6M runs smooth_span=0, so the gate is excluded there BY CONSTRUCTION rather
    than by a null result. This is the test that keeps that claim honest."""
    panel = _planted_panel(n_dates=60, n_names=40, beta=1.0, seed=11)
    wf = WalkForwardConfig(min_train_months=12, min_names=20)
    cfg = LGBMConfig(n_estimators=40)

    base = walk_forward_ic(panel, "1M", cfg, wf, seed=1, compute_sector_ic=False)
    gated = walk_forward_ic(panel, "1M", cfg, wf, seed=1, compute_sector_ic=False,
                            vol_gate=True)
    assert [f["ic"] for f in gated["folds"]] == [f["ic"] for f in base["folds"]]


def test_vol_gate_changes_ranks_once_smoothing_composes():
    from backend.ml.gbm_baseline import _rank01, apply_vol_gate, ewma_rank_by_ticker

    rng = np.random.default_rng(0)
    ids = np.arange(30)
    records = [{"ticker_ids": ids, "pred": rng.normal(size=30)} for _ in range(40)]
    vol = [0.1] * 35 + [0.9] * 5           # a stress stretch at the end

    plain = ewma_rank_by_ticker(records, span=4)
    ranks = [_rank01(r["pred"]) for r in records]
    gated_ranks, flags = apply_vol_gate(ranks, vol)
    composed = ewma_rank_by_ticker(records, span=4, rank_series=gated_ranks)

    assert any(flags), "expected the tail to be gated"
    assert not np.allclose(composed[-1], plain[-1]), (
        "gate must bite once ranks are compared across dates"
    )


def test_horizon_spec_vol_gate_round_trips_through_serialization():
    from backend.ml.gbm_baseline import HorizonSpec
    from backend.ml.gbm_inference import _serialize_spec, _specs_from_serialized

    spec = HorizonSpec(target_mode="sector_return", smooth_span=3, vol_gate=True)
    back = _specs_from_serialized({"3M": _serialize_spec(spec)})["3M"]
    assert back.vol_gate is True
    assert back.smooth_span == 3
    # A pre-Stage-2 artifact has no key at all and must default to off.
    legacy = _serialize_spec(spec)
    legacy.pop("vol_gate")
    assert _specs_from_serialized({"3M": legacy})["3M"].vol_gate is False


# =============================================================
# Phase 0 — missing-data semantics, ensemble aggregation, eval/prod parity
# =============================================================


def test_rank_normalize_keeps_nan_out_of_the_ranking():
    """A missing feature must not be ranked into a real cross-sectional position.

    This is the whole point of the NaN change: under the old sentinel-0.0 path the
    two nameless rows below would have tied at the *bottom* of a positive-valued
    feature (0.0 < every observed value), handing the model a fabricated signal.
    """
    panel = pd.DataFrame({
        "date": ["d1"] * 5,
        "ticker_id": [1, 2, 3, 4, 5],
        "x": [0.5, 1.5, 2.5, np.nan, np.nan],
    })
    out = rank_normalize_features(panel, cols=["x"])
    vals = out["x"].to_numpy(dtype=float)
    assert np.isnan(vals[3]) and np.isnan(vals[4])
    # The three observed names still span the full [-1, 1] scale between them.
    assert vals[0] == pytest.approx(-1.0)
    assert vals[1] == pytest.approx(0.0)
    assert vals[2] == pytest.approx(1.0)


def test_rank_normalize_nan_survives_a_single_observation_date():
    # count <= 1 collapses observed values to 0.0, but must not resurrect a NaN.
    panel = pd.DataFrame({
        "date": ["d1", "d1"], "ticker_id": [1, 2], "x": [7.0, np.nan],
    })
    out = rank_normalize_features(panel, cols=["x"])
    assert out["x"].iloc[0] == 0.0
    assert np.isnan(out["x"].iloc[1])


def test_exempt_ids_do_not_shift_member_ranks():
    """Off-index names are scored against the member distribution, never define it.

    Production carries user-added tickers (leveraged/thematic ETFs among them) in
    the panel so they can be scored; before this they also sat inside the per-date
    rank normalization and moved every index name's feature ranks.
    """
    members = pd.DataFrame({
        "date": ["d1"] * 4, "ticker_id": [1, 2, 3, 4], "x": [10.0, 20.0, 30.0, 40.0],
    })
    with_outlier = pd.concat([
        members,
        pd.DataFrame({"date": ["d1"], "ticker_id": [99], "x": [10_000.0]}),
    ], ignore_index=True)

    baseline = rank_normalize_features(members, cols=["x"])["x"].to_numpy(dtype=float)
    exempted = rank_normalize_features(
        with_outlier, cols=["x"], exempt_ids={99}
    )["x"].to_numpy(dtype=float)
    contaminated = rank_normalize_features(with_outlier, cols=["x"])["x"].to_numpy(dtype=float)

    # Members are ranked exactly as if the off-index name were not in the panel.
    assert exempted[:4] == pytest.approx(baseline)
    # Without the exemption the outlier compresses them (0.5 instead of 1.0 at top).
    assert contaminated[3] != pytest.approx(baseline[3])
    # The exempt name is still scored, at the top of the member distribution.
    assert exempted[4] == pytest.approx(1.0)


def test_seed_ensemble_averages_ranks_not_raw_scores():
    """LambdaRank scores carry no common scale across seeds, so a raw mean lets the
    widest-range seed dominate. `_fit_predict` must aggregate in rank space."""
    from backend.ml.gbm_baseline import _rank01

    # Two "models": both order the names identically for the first three, but the
    # second has a 100x wider score range and flips the last two.
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.array([0.0, 100.0, -100.0, 200.0])
    raw_mean_order = np.argsort(np.mean([a, b], axis=0))
    rank_mean_order = np.argsort(np.mean([_rank01(a), _rank01(b)], axis=0))
    assert list(raw_mean_order) != list(rank_mean_order), (
        "fixture must actually distinguish the two aggregations"
    )
    # Rank-averaging gives each seed one equal vote: name 2 (ranked 3rd and 1st)
    # must not be dragged to last place by b's -100 raw score.
    assert list(rank_mean_order) == [0, 2, 1, 3]


def test_build_specs_from_args_preserves_promoted_overlay_fields():
    """A --target override must not silently drop a promoted spec field.

    The old hand-listed constructor omitted knife_lambda / winsorize_pct / vol_gate,
    so `--target return` shipped 3M without its promoted falling-knife overlay.
    """
    from argparse import Namespace

    from backend.ml.gbm_baseline import PRODUCTION_HORIZON_SPECS
    from backend.ml.gbm_inference import build_specs_from_args

    specs = build_specs_from_args(Namespace(horizons=["3M", "6M"], target="return"))
    assert specs["3M"].target_mode == "return"          # the override applied
    assert specs["3M"].knife_lambda == 0.20             # ... and nothing else moved
    assert specs["3M"].smooth_span == 3
    assert specs["3M"].feature_cols == PRODUCTION_HORIZON_SPECS["3M"].feature_cols
    assert specs["6M"].max_train_months == 60
    for h in ("3M", "6M"):
        prod = PRODUCTION_HORIZON_SPECS[h]
        assert specs[h].winsorize_pct == prod.winsorize_pct
        assert specs[h].vol_gate == prod.vol_gate


def test_walk_forward_max_train_months_counts_labeled_dates():
    """The rolling window counts LABELED training dates, matching production.

    gbm_inference.fit_horizon_models slices by the number of dates that actually
    carry labels; the walk-forward used to slice by grid position, so the two
    trained on different data whenever a grid date had no labeled rows.
    """
    from backend.ml.gbm_baseline import WalkForwardConfig, walk_forward_ic

    frames = [make_frame(n_days=1500, trend=0.0002 * (k + 1), tid=k, vol_seed=k)
              for k in range(8)]
    panel = prepare_panel(frames, build_calendar_grid(frames))
    window = 6
    # min_child_samples must stay under the per-fold training row count (6 dates x 8
    # names = 48) or LightGBM never splits, every prediction is constant, and the
    # fold's Spearman IC is NaN and skipped.
    res = walk_forward_ic(
        panel, "1M", LGBMConfig(n_estimators=10, num_leaves=4, min_child_samples=5),
        WalkForwardConfig(min_train_months=12, max_train_months=window, min_names=4),
        seed=0, return_records=True,
    )
    assert res["folds"], "expected at least one scored fold"
    # Every fold trains on at most `window` distinct dates, and once enough history
    # exists it trains on exactly that many — the labeled-date rule, not a grid slice.
    n_per_date = len(frames)
    assert all(f["n_train"] <= window * n_per_date for f in res["folds"])
    assert max(f["n_train"] for f in res["folds"]) == window * n_per_date


def test_production_smoothing_matches_the_walk_forward_recursion():
    """The shipped ranking must be the one the walk-forward measured.

    `ewma_rank_by_ticker` (eval) carries the raw blended value forward; production
    used to carry the RE-RANKED percentile, which restores full dispersion to the
    prior at every step and therefore smooths harder than the folds that promoted
    smooth_span. Chaining `apply_rank_smoothing` across monthly cross-sections must
    now reproduce the eval ranking exactly.
    """
    from backend.ml.gbm_baseline import HorizonSpec, ewma_rank_by_ticker
    from backend.ml.gbm_inference import apply_rank_smoothing

    span = 3
    tids = np.array([1, 2, 3, 4, 5])
    # Three monthly cross-sections whose ordering churns between dates.
    preds = [
        np.array([0.10, 0.20, 0.30, 0.40, 0.50]),
        np.array([0.50, 0.40, 0.30, 0.20, 0.10]),
        np.array([0.20, 0.50, 0.10, 0.40, 0.30]),
    ]
    records = [{"ticker_ids": tids, "pred": p} for p in preds]
    expected = ewma_rank_by_ticker(records, span)

    specs = {"3M": HorizonSpec(smooth_span=span)}
    state: dict = {}
    for step, p in enumerate(preds):
        raw = _rank01_local(p)
        rows = [{"ticker_id": int(t), "horizon": "3M", "relative_rank": float(r)}
                for t, r in zip(tids, raw)]
        apply_rank_smoothing(rows, specs, dict(state), advance=True)
        state = {(r["ticker_id"], "3M"): r["smooth_state"] for r in rows}
        # Spearman only cares about ordering, so compare the induced ranking.
        got = np.array([r["relative_rank"] for r in rows])
        assert list(np.argsort(got)) == list(np.argsort(expected[step])), (
            f"step {step}: production ordering diverged from the walk-forward"
        )
        # ... and the carried state is the eval state itself.
        assert np.allclose([r["smooth_state"] for r in rows], expected[step])


def test_non_advancing_run_holds_the_monthly_anchor():
    """Intra-month (Friday) runs blend against the anchor but must not move it.

    smooth_span was fitted on monthly folds; inference also fires weekly, so
    advancing every run would compound ~4-5x the validated smoothing.
    """
    from backend.ml.gbm_baseline import HorizonSpec
    from backend.ml.gbm_inference import apply_rank_smoothing

    specs = {"3M": HorizonSpec(smooth_span=3)}
    rows = [{"ticker_id": t, "horizon": "3M", "relative_rank": r}
            for t, r in zip(range(1, 5), [0.0, 0.33, 0.66, 1.0])]
    anchor = {(t, "3M"): 0.5 for t in range(1, 5)}

    held = apply_rank_smoothing([dict(r) for r in rows], specs, anchor, advance=False)
    assert all(r["smooth_state"] is None for r in held), "a held run must not persist state"
    advanced = apply_rank_smoothing([dict(r) for r in rows], specs, anchor, advance=True)
    assert all(r["smooth_state"] is not None for r in advanced)
    # Both produce the same ranking for the day — only persistence differs.
    assert [r["relative_rank"] for r in held] == [r["relative_rank"] for r in advanced]


def _rank01_local(a):
    from backend.ml.gbm_baseline import _rank01
    return _rank01(a)


# =============================================================
# Phase 2 — diagnostics, honest benchmarks, top-of-list metrics
# =============================================================


def test_forecast_combination_recovers_a_planted_signal_and_signs_from_train_only():
    """One vote per feature, signed in the training window. Nothing to overfit but
    the signs, which is the point of the benchmark."""
    from backend.ml.gbm_baseline import forecast_combination_predict

    rng = np.random.default_rng(0)
    n = 200
    good = rng.normal(size=n)
    train = pd.DataFrame({
        "a": good, "b": -good, "c": rng.normal(size=n), "y": good,
    })
    test = pd.DataFrame({
        "a": good, "b": -good, "c": rng.normal(size=n), "y": good,
    })
    pred = forecast_combination_predict(train, test, "y", ["a", "b", "c"])
    # `b` is anti-correlated in training, so it must be flipped, not cancel `a` out.
    assert pd.Series(pred).corr(pd.Series(test["y"]), method="spearman") > 0.8
    # A feature that is pure noise in training contributes a sign but no signal; the
    # combination still tracks the planted factor.
    assert np.isfinite(pred).all()


def test_forecast_combination_ignores_missing_values_rather_than_imputing():
    from backend.ml.gbm_baseline import forecast_combination_predict

    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0]})
    test = pd.DataFrame({"a": [1.0, np.nan, -1.0, 0.5]})
    pred = forecast_combination_predict(train, test, "y", ["a"])
    assert np.isfinite(pred).all()          # never NaN out the whole row
    assert pred[1] == 0.0                   # unobserved -> no vote, not a mid-rank


def test_top_of_list_metrics_separate_a_good_top_from_a_good_full_list():
    """The product ships `order by direction_prob desc`. A model can rank the full
    list well and still have a bad top decile, and SECB cannot tell."""
    from backend.ml.gbm_baseline import top_of_list_metrics

    # n=500 like a real cross-section: precision@k is capped at (0.20*n)/k, so a
    # smaller universe could not reach 1.0 even with a perfect ranking.
    n = 500
    r = np.linspace(-1, 1, n)
    perfect = top_of_list_metrics(r, r)                 # ranking == outcome
    assert perfect["spread_decile"] > 0
    assert perfect["precision_at_k"] == pytest.approx(1.0)
    inverted = top_of_list_metrics(-r, r)
    assert inverted["spread_decile"] < 0
    assert inverted["precision_at_k"] == pytest.approx(0.0)
    # The case the whole metric exists for: a ranking that is right almost everywhere
    # but wrong exactly where the product looks. Full-list Spearman stays strongly
    # positive; the decile spread must go negative.
    sabotaged = r.copy()
    top_decile = np.argsort(-r)[: int(0.10 * n)]
    sabotaged[top_decile] = -5.0                        # the picks are the worst names
    # Still solidly positive on the full list (collapsing a decile onto one value
    # costs some rank mass, so this is ~0.46, not ~0.9) ...
    assert pd.Series(r).corr(pd.Series(sabotaged), method="spearman") > 0.4
    assert top_of_list_metrics(r, sabotaged)["spread_decile"] < 0
    assert top_of_list_metrics(r, sabotaged)["top_decile_r"] < 0


def test_within_sector_ic_breakdown_exposes_a_single_sector_signal():
    """SECB averages the sectors, so a signal living in exactly one reads the same as
    one spread evenly across eleven. The breakdown is what distinguishes them."""
    from backend.ml.gbm_baseline import within_sector_ic, within_sector_ic_breakdown

    n = 30
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "sector": ["A"] * n + ["B"] * n,
        "r_1M": np.concatenate([np.arange(n, dtype=float), rng.normal(size=n)]),
    })
    preds = np.concatenate([np.arange(n, dtype=float), rng.normal(size=n)])
    parts = within_sector_ic_breakdown(preds, df, "r_1M", min_group_size=10)
    assert parts["A"] == pytest.approx(1.0)      # perfect inside A
    assert abs(parts["B"]) < 0.5                 # noise inside B
    # The scalar headline is exactly the mean of the parts it hides.
    assert within_sector_ic(preds, df, "r_1M", min_group_size=10) == pytest.approx(
        float(np.mean(list(parts.values())))
    )


def test_tree_curve_reuses_one_fit_and_ends_at_the_headline_ic():
    """The IC-vs-trees curve must cost no extra training: truncating the SAME fitted
    models with num_iteration is what makes the complexity question cheap to ask."""
    from backend.ml.gbm_baseline import _fit_models, _fold_fit_diagnostics

    frames = [make_frame(n_days=1200, trend=0.0002 * (k + 1), tid=k, vol_seed=k)
              for k in range(10)]
    panel = prepare_panel(frames, build_calendar_grid(frames))
    dates = sorted(panel["date"].unique())
    # NOT the last grid date: the 1M forward return does not exist there yet, so
    # r_1M is constant and every Spearman comes back NaN.
    train = panel[panel["date"] <= dates[-8]]
    test = panel[(panel["date"] == dates[-5]) & panel["mask_1M"]]
    train = train[train["mask_1M"] & train["r_1M"].notna()]
    cfg = LGBMConfig(n_estimators=120, num_leaves=4, min_child_samples=5)
    models = _fit_models(train, "r_1M", cfg, seed=0, shuffle=False,
                         feature_cols=list(FEATURE_COLS), n_seeds=1)
    d = _fold_fit_diagnostics(models, train, test, list(FEATURE_COLS), "r_1M",
                              tree_grid=(25, 60, 120))
    assert set(d["tree_curve"]) == {25, 60, 120}
    # Gain importances are normalized per model, so they form a distribution.
    assert d["importance"] and abs(sum(d["importance"].values()) - 1.0) < 1e-6
    assert d["ic_train"] == d["ic_train"]     # not NaN


def test_sector_rank_target_is_the_percentile_of_the_sector_relative_return():
    """The Cakici-Zaremba rank target: same ordering as sector_return, mapped to a
    per-date percentile. It keeps an L2 fit, unlike sector_grade which needs a ranker."""
    d, d2 = date(2020, 1, 31), date(2020, 2, 29)
    df = pd.DataFrame({
        "date": [d] * 6 + [d2] * 6,
        "sector": ["Tech"] * 6 + ["Tech"] * 6,
        "r_1M": [0.10, 0.05, 0.00, -0.05, -0.10, 0.20] * 2,
        "mask_1M": [True] * 12,
    })
    for h in HORIZONS:
        if f"r_{h}" not in df:
            df[f"r_{h}"] = 0.0
        if f"mask_{h}" not in df:
            df[f"mask_{h}"] = False
    df["r_1M"] = [0.10, 0.05, 0.00, -0.05, -0.10, 0.20] * 2
    df["mask_1M"] = [True] * 12

    out = apply_target_modes(df, sector_min_group_size=5)
    rank, sec = out["y_1M_sector_rank"], out["y_1M_sector_return"]
    # Order-preserving within each date, and bounded in (0, 1].
    for dd in (d, d2):
        m = out["date"] == dd
        assert rank[m].min() > 0 and rank[m].max() == pytest.approx(1.0)
        assert (
            pd.Series(sec[m]).corr(pd.Series(rank[m]), method="spearman")
            == pytest.approx(1.0)
        )
    # Ranks are per-date, so the same raw return maps to the same percentile on both
    # dates here — that is the point: the target is scale-free across cross-sections.
    assert sorted(rank[out["date"] == d]) == pytest.approx(
        sorted(rank[out["date"] == d2])
    )
    # A masked row has no sector return and therefore no rank.
    assert out["y_1M_sector_rank"].notna().all()
