"""Factor layer: feature column catalogs + per-ticker point-in-time builders.

Extracted from the former 2300-line gbm_baseline.py so the feature definitions live
next to the code that produces them. `gbm_baseline` re-exports every name here, so
existing `from backend.ml.gbm_baseline import <feature symbol>` imports keep working.
"""

from __future__ import annotations

from backend.ml.factors.assembly import (
    build_market_horizon_returns,
    build_ticker_rows,
    build_universe_return_map,
)
from backend.ml.factors.constants import (
    ANALYST_REVISION_FEATURES,
    EARNINGS_REACTION_FEATURES,
    EPS_DISPERSION_FEATURES,
    EPS_SURPRISE_FEATURES,
    ESTIMATE_SURPRISE_FEATURES,
    EXPERIMENTAL_FEATURES,
    FEATURE_COLS,
    FORWARD_VALUATION_FEATURES,
    FUNDAMENTAL_FEATURES,
    FUNDAMENTAL_MISSING_FEATURES,
    INDUSTRY_RELATIVE_FEATURES,
    KNIFE_FEATURES,
    LOTTERY_FEATURES,
    MICROSTRUCTURE_FEATURES,
    PRICE_FEATURES,
    QUALITY_FEATURES,
    RESIDUAL_MOM_FEATURES,
    REVISION_MOMENTUM_FEATURES,
    SENTIMENT_FEATURES,
    SHORT_INTEREST_FEATURES,
    VALUATION_FEATURES,
)
from backend.ml.factors.estimates import _earnings_reaction_asof, _estimates_context_asof
from backend.ml.factors.fundamentals import _fundamental_context_asof, _ttm_net_income_asof
from backend.ml.factors.price import _price_features, _short_interest_asof
from backend.ml.factors.util import _log_ratio, _safe_ratio

__all__ = [
    # constants
    "ANALYST_REVISION_FEATURES", "EARNINGS_REACTION_FEATURES",
    "EPS_DISPERSION_FEATURES", "EPS_SURPRISE_FEATURES",
    "ESTIMATE_SURPRISE_FEATURES", "EXPERIMENTAL_FEATURES", "FEATURE_COLS",
    "FORWARD_VALUATION_FEATURES", "FUNDAMENTAL_FEATURES", "FUNDAMENTAL_MISSING_FEATURES",
    "INDUSTRY_RELATIVE_FEATURES", "KNIFE_FEATURES", "LOTTERY_FEATURES", "MICROSTRUCTURE_FEATURES",
    "PRICE_FEATURES", "QUALITY_FEATURES", "RESIDUAL_MOM_FEATURES",
    "REVISION_MOMENTUM_FEATURES", "SENTIMENT_FEATURES", "SHORT_INTEREST_FEATURES", "VALUATION_FEATURES",
    # builders
    "build_market_horizon_returns", "build_ticker_rows", "build_universe_return_map",
    "_price_features", "_short_interest_asof", "_fundamental_context_asof", "_ttm_net_income_asof",
    "_estimates_context_asof", "_earnings_reaction_asof",
    # helpers
    "_log_ratio", "_safe_ratio",
]
