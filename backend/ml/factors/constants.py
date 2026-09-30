"""Tabular factor column names: the production baseline + opt-in experimental packs.

Order within a list is informational only (LightGBM is order-agnostic). These were
extracted from gbm_baseline.py so the feature catalog lives next to the builders that
produce it; gbm_baseline re-exports them for backward compatibility.
"""

from __future__ import annotations

# Tabular factor columns the model trains on (order is informational only).
PRICE_FEATURES = [
    "mom_1m", "mom_3m", "mom_6m", "mom_12_1",   # momentum (12_1 skips the last month)
    "log_market_cap",                           # log(raw close × as-reported PIT shares)
    "vol_20d", "vol_60d", "vol_120d",           # realized vol
    "dist_high_252", "dist_low_252",            # distance to 52w extremes
    "ma_gap_50", "ma_gap_200",                  # gap vs moving averages
    "vol_trend",                                # 20d vs 120d dollar/volume trend
]
FUNDAMENTAL_FEATURES = [
    "revenue_growth", "gross_margin", "operating_margin", "debt_equity", "fcf_revenue",
]
# Binary indicator: 1 = ticker has at least one SEC filing as-of the row date, 0 = none.
# Lets the GBDT handle missing fundamentals explicitly rather than confounding "zero
# value" (real) with "zero value" (no filing available). Critical for removed-from-index
# names whose fundamental rows are absent.
FUNDAMENTAL_MISSING_FEATURES = ["fund_available"]
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
# Earnings-surprise / PEAD pack (Phase 2): the most-recent reported EPS vs its
# pre-report consensus. Computed downstream from eps_mean/eps_actual (this LSEG
# license has no direct EPSSurprise field). Earnings-surprise drift is a classic
# WITHIN-INDUSTRY stock-selection signal, strongest 3-9M.
EPS_SURPRISE_FEATURES = [
    "eps_surprise",
]
# Earnings-revision momentum (Phase 3, 3M-focused): analysts revising the forward
# EPS consensus up + rising coverage/PT-estimate breadth. This license has no
# recommendation-bucket counts, so conviction is proxied by rec_rev (in the analyst
# revision pack) + coverage/PT-estimate counts here. Strongest at short horizons.
REVISION_MOMENTUM_FEATURES = [
    "eps_est_rev_30d", "eps_est_rev_90d", "coverage_chg_90d", "pt_num_estimates",
]
# Forward valuation stored as yields (inverse multiples) so ranking is monotonic
# and negative/near-zero denominators don't blow up — mirrors earnings_yield.
FORWARD_VALUATION_FEATURES = [
    "forward_earnings_yield", "forward_ebitda_yield", "price_target_upside",
]
# Lottery / idiosyncratic-vol pack (Experiment 1): the volatility variants that
# carry the documented NEGATIVE cross-sectional signal, which total realized vol
# (vol_20/60/120) conflates with priced risk and loads on POSITIVELY.
#   max_ret_21d : max daily return over the last ~month — lottery-demand proxy
#                 (Bali-Cakici-Whitelaw 2011; subsumes the IVOL puzzle, robust in
#                 large caps). Expect a negative loading.
#   idio_vol    : residual daily-return vol vs the universe (stock − beta·market),
#                 the idiosyncratic-volatility puzzle factor (Ang et al 2006).
LOTTERY_FEATURES = ["max_ret_21d", "idio_vol"]
# Microstructure / higher-moment pack (decorrelated 6M candidate, price/volume only).
# These are orthogonal statistical axes to the 1st/2nd-moment book (momentum + symmetric
# vol): a 3rd-moment skew, a downside/total vol asymmetry, illiquidity (price impact), and
# turnover. `efficiency_ratio_120d` (Kaufman ER: |net move| / path) is the trend-vs-
# consolidation control the feature diagnostic conditions on (the sideways-bias check).
MICROSTRUCTURE_FEATURES = [
    "ret_skew_120d", "downside_vol_ratio_120d", "amihud_illiq_60d",
    "turnover_60d", "efficiency_ratio_120d",
]
# Falling-knife composite score feature (opt-in via --with-knife-feature).
# Computed from rank-normalized vol_120d / ma_gap_200 / dist_low_252 mapped to [0,1]:
# knife_score = vol_p * (1 - mean(trend_p, dlow_p)). Already in [0,1] within-date so
# it is NOT re-normalized in rank_normalize_features. Training signal: lets the GBDT
# learn conditional demotion of high-vol-AND-falling names rather than the overlay's
# unconditional rank penalty.
KNIFE_FEATURES = ["knife_score"]
# EPS estimate dispersion (Diether-Malloy-Scherbina 2002 short-selling proxy):
# eps_dispersion = eps_std_dev / max(|eps_mean|, 0.01). Higher disagreement among
# analysts → NEGATIVE expected return (short-selling constraint prevents full
# arbitrage of disagreed-on names). Opt-in via --with-eps-dispersion.
EPS_DISPERSION_FEATURES = ["eps_dispersion"]
# Net share issuance (Pontiff-Woodgate 2008; Fama-French 2008 find it pervasive ACROSS
# size groups, which is rare for an anomaly and the reason it is worth testing here).
# net_issuance = shares_now / shares_1y_ago - 1, both point-in-time from the filing
# cover page, so a buyback reads NEGATIVE and a raise reads positive. Needs no new
# ingestion: `fundamentals.shares_outstanding` has been PIT since migration 011, and
# no post-2010 large-cap test of this signal was located in the literature review —
# it is genuinely open rather than a known null.
ISSUANCE_FEATURES = ["net_issuance"]
# Payout: trailing 12-month cash dividends over price. `price_history.dividend` has
# been stored since the first ingest and never selected. Boudoukh et al. net payout
# yield; the buyback leg needs the EDGAR expansion, so this is the dividend half.
PAYOUT_FEATURES = ["dividend_yield_ttm"]
# Range-based realized volatility (Parkinson 1980) from the stored high/low, which
# were likewise never selected. Uses the intra-bar range rather than close-to-close,
# giving roughly 5x the efficiency per observation — a lower-variance estimate of the
# SAME quantity vol_20d/vol_60d already proxy, so this is a REPLACEMENT candidate for
# the collinear close-to-close vol block, not an addition to it.
RANGE_VOL_FEATURES = ["range_vol_20d", "range_vol_60d"]
# Analyst BREADTH, the thread the post-audit program flagged as most promising:
# coverage_chg_90d was the cleanest feature in the Stage-3 diagnostics (sec_ic +0.024,
# sec_p 0.009, only 0.057 correlated with the book). These extend that idea using LSEG
# columns already ingested but read by nothing — consensus REVENUE revisions (the
# analogue of the promoted eps_est_rev_*), the change in the number of included EPS
# estimates, coverage as a level, and coverage LOSS specifically.
ANALYST_BREADTH_FEATURES = [
    "revenue_est_rev_30d", "revenue_est_rev_90d",
    "eps_num_est_chg_90d", "coverage_level", "coverage_drop_90d",
]
# Short interest (FINRA Reg SHO): days-to-cover ratio (short_interest /
# avg_daily_volume). High DTC = crowded short = contrarian long candidate OR
# further squeeze risk. PIT-safe on publication_date (~14d after settlement).
# Requires short_interest table (migration 009) + backfill_short_interest.py.
SHORT_INTEREST_FEATURES = ["short_ratio"]
# Insider transactions (Cohen-Malloy-Pomorski 2012): open-market insider buying
# predicts positive abnormal returns. insider_net_buy_6m = signed P/S dollar flow
# (183d) scaled by market cap; insider_buyers_90d = distinct-insider cluster-buy
# breadth (90d); insider_net_ratio_12m = scale-free (buy-sell)/(buy+sell) value (365d).
# PIT-safe on filing_date. Requires insider_transactions (migration 010) + backfill.
INSIDER_FEATURES = [
    "insider_net_buy_6m", "insider_buyers_90d", "insider_net_ratio_12m",
]
# Cross-sectional seasonality (Heston-Sadka 2008): same-calendar-month return
# persistence. For the UPCOMING calendar month, `seasonal_same_month_5y` is the
# mean of the ticker's own historical returns in that month (<=5y), `..._other`
# the mean of the other months, and `..._gap` their difference (the promotable
# differential factor). PIT-safe: only completed months strictly before the bar's
# month are used (nearest same-month obs is ~11 months back, no forward overlap).
SEASONALITY_FEATURES = [
    "seasonal_same_month_5y", "seasonal_other_month_5y", "seasonal_gap_5y",
]
# FinBERT rolling news sentiment. NOT in FEATURE_COLS: yfinance only serves ~30 days
# of headlines and there is no backfill, so sentiment_daily covers a few months while
# the panel spans 2010+. The columns were ~98% zero in training yet non-zero at
# inference — a train/serve skew where the model never had the chance to learn the
# feature it was being served. Still computed on every row, so `--with-sentiment`
# turns them back on the moment a real headline archive exists.
SENTIMENT_FEATURES = ["sentiment_7d", "sentiment_14d"]
# Availability / staleness of the LSEG estimate snapshot, the analogue of
# `fund_available` for the analyst feed. `analyst_estimates` starts in 2012-13, so
# without this the promoted estimate packs are silently a coverage proxy on early
# rows. est_staleness_days = calendar days since the newest observed snapshot.
ESTIMATE_MISSING_FEATURES = ["est_available", "est_staleness_days"]
FEATURE_COLS = PRICE_FEATURES + FUNDAMENTAL_FEATURES + FUNDAMENTAL_MISSING_FEATURES
# EXPERIMENTAL_FEATURES: per-ticker features produced by build_ticker_rows (eligible for
# `--feature-diagnostics` and `--with-*` packs). knife_score is excluded because it is a
# PANEL-LEVEL feature computed by add_knife_score_feature AFTER cross-sectional normalization
# — it is not produced per-ticker and must not appear in build_ticker_rows row dicts.
EXPERIMENTAL_FEATURES = (
    VALUATION_FEATURES + QUALITY_FEATURES + RESIDUAL_MOM_FEATURES + EARNINGS_REACTION_FEATURES
    + ANALYST_REVISION_FEATURES + ESTIMATE_SURPRISE_FEATURES + EPS_SURPRISE_FEATURES
    + FORWARD_VALUATION_FEATURES + REVISION_MOMENTUM_FEATURES + LOTTERY_FEATURES
    + MICROSTRUCTURE_FEATURES + EPS_DISPERSION_FEATURES + SHORT_INTEREST_FEATURES
    + SEASONALITY_FEATURES + INSIDER_FEATURES + SENTIMENT_FEATURES
    + ESTIMATE_MISSING_FEATURES + ISSUANCE_FEATURES + PAYOUT_FEATURES
    + RANGE_VOL_FEATURES + ANALYST_BREADTH_FEATURES
    # KNIFE_FEATURES intentionally excluded — panel-level, not in build_ticker_rows
)
# The industry-relative *normalization* sweep (which hurt in test 3); residual /
# earnings-reaction features already adjust for market or filing context so they
# stay out of this list — double-grouping would re-shrink whatever signal they
# carry.
INDUSTRY_RELATIVE_FEATURES = (
    PRICE_FEATURES + FUNDAMENTAL_FEATURES + VALUATION_FEATURES + QUALITY_FEATURES
)

# ---------------------------------------------------------------------------
# Source-gated features: columns that are only DEFINED when their upstream source
# has an observation as of the row date. build_ticker_rows sets these to NaN when
# the source is absent, instead of the historical sentinel 0.0.
#
# Why this matters: a sentinel 0.0 is then RANKED by rank_normalize_features, so a
# name with no SEC filing lands at a real position in the cross-section — mid-pack
# for a signed feature like revenue_growth, in a tail for a positive-only one like
# earnings_yield. The imputation is silent, column-dependent and signal-bearing
# (Bryzgalova-Lerner-Lettau-Pelger 2025; Freyberger et al. 2025: characteristic
# missingness is systematic, not random). NaN lets LightGBM route missing values
# natively and keeps them out of the ranking entirely.
#
# The availability FLAGS (`fund_available`, `est_available`) stay finite 0/1 — they
# are the model's explicit handle on missingness. Partial-field gaps inside an
# otherwise-present source (e.g. total_equity null on a filing that has revenue)
# still fall back to 0.0; that is a documented follow-up, not this pass.
FUNDAMENTAL_SOURCED_FEATURES = (
    FUNDAMENTAL_FEATURES + VALUATION_FEATURES + QUALITY_FEATURES
    + EARNINGS_REACTION_FEATURES
)
ESTIMATE_SOURCED_FEATURES = (
    ANALYST_REVISION_FEATURES + FORWARD_VALUATION_FEATURES
    + REVISION_MOMENTUM_FEATURES + EPS_DISPERSION_FEATURES
    + ANALYST_BREADTH_FEATURES
)
