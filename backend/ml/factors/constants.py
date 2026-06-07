"""Tabular factor column names: the production baseline + opt-in experimental packs.

Order within a list is informational only (LightGBM is order-agnostic). These were
extracted from gbm_baseline.py so the feature catalog lives next to the builders that
produce it; gbm_baseline re-exports them for backward compatibility.
"""

from __future__ import annotations

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
SENTIMENT_FEATURES = ["sentiment_7d", "sentiment_14d"]
FEATURE_COLS = PRICE_FEATURES + FUNDAMENTAL_FEATURES + FUNDAMENTAL_MISSING_FEATURES + SENTIMENT_FEATURES
EXPERIMENTAL_FEATURES = (
    VALUATION_FEATURES + QUALITY_FEATURES + RESIDUAL_MOM_FEATURES + EARNINGS_REACTION_FEATURES
    + ANALYST_REVISION_FEATURES + ESTIMATE_SURPRISE_FEATURES + EPS_SURPRISE_FEATURES
    + FORWARD_VALUATION_FEATURES + REVISION_MOMENTUM_FEATURES + LOTTERY_FEATURES
    + MICROSTRUCTURE_FEATURES
)
# The industry-relative *normalization* sweep (which hurt in test 3); residual /
# earnings-reaction features already adjust for market or filing context so they
# stay out of this list — double-grouping would re-shrink whatever signal they
# carry.
INDUSTRY_RELATIVE_FEATURES = (
    PRICE_FEATURES + FUNDAMENTAL_FEATURES + VALUATION_FEATURES + QUALITY_FEATURES
)
