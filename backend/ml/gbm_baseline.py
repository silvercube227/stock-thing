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
from datetime import date
from dataclasses import dataclass, field, replace

import numpy as np

from backend.ingestion.calendar import HORIZON_TRADING_DAYS
from backend.ingestion.db import pool_context
from backend.ml.dataset import (
    TickerFrame,
    build_calendar_grid,
    cross_sectional_medians,
    load_frames_cached,
)
# Feature catalogs + per-ticker point-in-time builders live in backend/ml/factors/;
# re-exported here so existing `from backend.ml.gbm_baseline import <feature symbol>`
# imports (gbm_inference, tests, scripts) keep working unchanged.
from backend.ml.factors import (  # noqa: F401
    ANALYST_REVISION_FEATURES,
    EARNINGS_REACTION_FEATURES,
    EPS_DISPERSION_FEATURES,
    EPS_SURPRISE_FEATURES,
    ESTIMATE_MISSING_FEATURES,
    ESTIMATE_SOURCED_FEATURES,
    ESTIMATE_SURPRISE_FEATURES,
    EXPERIMENTAL_FEATURES,
    FEATURE_COLS,
    FORWARD_VALUATION_FEATURES,
    FUNDAMENTAL_FEATURES,
    FUNDAMENTAL_MISSING_FEATURES,
    FUNDAMENTAL_SOURCED_FEATURES,
    INDUSTRY_RELATIVE_FEATURES,
    INSIDER_FEATURES,
    KNIFE_FEATURES,
    LOTTERY_FEATURES,
    MICROSTRUCTURE_FEATURES,
    PRICE_FEATURES,
    QUALITY_FEATURES,
    RESIDUAL_MOM_FEATURES,
    REVISION_MOMENTUM_FEATURES,
    SEASONALITY_FEATURES,
    SENTIMENT_FEATURES,
    SHORT_INTEREST_FEATURES,
    VALUATION_FEATURES,
    _earnings_reaction_asof,
    _estimates_context_asof,
    _fundamental_context_asof,
    _log_ratio,
    _price_features,
    _safe_ratio,
    _ttm_net_income_asof,
    build_market_horizon_returns,
    build_ticker_rows,
    build_universe_return_map,
)
from backend.ml.model import HORIZONS


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
    reg_alpha: float = 0.0          # L1 leaf penalty (LightGBM default 0)
    # Training objective. "regression" (default) = pointwise L2, the historical path.
    # "lambdarank" / "rank_xendcg" switch fit_lgbm_model to LGBMRanker (learning-to-
    # rank): the fit optimizes a listwise/pairwise ranking loss aligned with the
    # rank-IC scoring metric. Ranking objectives require an ordinal grade target
    # (--target sector_grade) and a per-date query group.
    objective: str = "regression"
    # LambdaRank list-truncation: pairs beyond this rank are ignored in the gradient.
    # Set >= the cross-section size (~500) for full-list ranking; lower values focus
    # the loss on the top of the list (the picks the product actually ships).
    lambdarank_truncation_level: int = 500
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
    # GBDT+ridge ensemble (Gu-Kelly-Xiu low-SNR robustness). 0.0 = pure GBDT
    # (unchanged behavior); >0 blends a ridge model at this rank weight.
    linear_blend: float = 0.0
    ridge_alpha: float = 10.0
    # Cross-date prediction smoothing: EWMA span for the name's percentile rank
    # over consecutive scoring dates (0 = off). Averages out per-date estimation
    # noise in a persistent signal → higher ICIR + much lower turnover. Helps most
    # on the noisiest horizons (3M, 1Y); 6M is already stable so it's left off.
    smooth_span: int = 0
    # Falling-knife output overlay: re-rank weight in [0, 1] (0 = off). Demotes
    # names that are BOTH high-vol AND downtrending (vol × downtrend) out of the
    # top of the ranking. Applied at score time before smoothing. Promoted at 3M
    # only — it's a downside/quality lever, NOT a churn lever (smoothing owns that).
    knife_lambda: float = 0.0
    # Rolling training window in monthly grid dates (None = expanding from 2010).
    # Mirrors WalkForwardConfig.max_train_months but lives on the spec so inference
    # can apply a per-horizon window without touching the CLI flag machinery.
    max_train_months: int | None = None
    # Per-date training-label winsorization: clip the training target to its per-date
    # [pct, 1−pct] quantiles (0 = off). Tames fat-tailed return labels so tail months
    # don't dominate the L2 split; scoring stays on unclipped realized returns. Only
    # valid for return-like target modes (asserted at fit time).
    winsorize_pct: float = 0.0
    # Shrink ranks toward 0.5 on point-in-time high-vol dates. Only meaningful when
    # smooth_span > 0 — see vol_gate_ranks for why it is inert on its own.
    vol_gate: bool = False


# Per-horizon production training defaults. Update this dict — and only this dict
# — when a sweep promotes a new target / hyperparameter / feature pack. The
# inference path reads it as its starting config; the walk-forward sweep tool
# (this file's CLI) tests *one* horizon at a time and is unaffected.
#
# Source of current values:
#   - 3M / 6M / 1Y: `sector_return` — train on the within-(date, sector)-relative
#     return (test-6 sweep, n_seeds=8, de-survivorshipped, block-bootstrap on the
#     WITHIN-SECTOR IC = SECB, the success bar). Targeting sector-relative return
#     directly optimizes within-sector stock selection rather than letting the tree
#     earn its universe IC from sector rotation. Before → after on SECB:
#         3M: rank          SECB t=0.91 p=0.246  →  sector_return t=1.51 p=0.052
#         6M: rank+surprise SECB t=1.86 p=0.009  →  sector_return  t=2.02 p=0.009
#         1Y: beta_resid+sp SECB t=0.87 p=0.266  →  sector_return  t=2.09 p=0.003
#     1Y is the headline: beta-residualization (the prior 1Y target) maximized
#     idiosyncratic-vs-MARKET alpha but left SECTOR tilt in, so it failed the
#     within-sector bar; sector_return clears it decisively (hit 0.81, ICIR 0.72).
#     3M is a real lift (SEC IC doubles, p 0.246→0.052) but sits right at the
#     detection floor — borderline-significant, not decisive like 6M/1Y. 1M stays
#     `rank` (dead horizon, not scored in production).
#   - 6M / 1Y feature_cols: + ESTIMATE_SURPRISE_FEATURES (LSEG revenue surprise).
#     De-survivorshipped walk-forward ablation (2026-05-29): 6M ICIR 0.409→0.442
#     (+0.033, bootstrap p=0.003, null z=6.5), 1Y 0.378→0.454 (+0.076, null z=9.7).
#     The other LSEG packs (analyst revisions, forward valuation) did not beat the
#     null net of surprise, and 3M saw no estimate signal — so only surprise is
#     promoted, and only at 6M/1Y. (Baselines here are below CLAUDE.md's older
#     survivor-only ICIRs because the universe now includes removed-from-index names.)
#   - 3M feature_cols: + REVISION_MOMENTUM_FEATURES (earnings-revision momentum).
#     Quarterly-rebuild ablation (2026-05-31, 8-seed, SECB): SEC IC +0.0222→+0.0269,
#     SECB p 0.052→0.011 — moves 3M off the detection floor into block-significant
#     within-sector selection (hit 0.615→0.661). eps_surprise stays OUT (negative
#     standalone at every horizon; only a within-noise combo lift). revenue_surprise
#     now sources from the quarterly earnings_surprises table (was annual): at 6M it
#     got stronger (+0.0375→+0.0421, p 0.008); at 1Y it became a wash (base +0.0514
#     ≥ +rev +0.0501) so 1Y is left unchanged pending a parsimony cleanup.
_BASELINE_PLUS_SURPRISE = FEATURE_COLS + ESTIMATE_SURPRISE_FEATURES
_BASELINE_PLUS_REVMOM = FEATURE_COLS + REVISION_MOMENTUM_FEATURES
#   - 3M / 1Y smooth_span: cross-date EWMA rank smoothing PROMOTED (2026-06-05,
#     8-seed walk-forward, SECB). The noisiest horizons gain the most from averaging
#     out per-date estimation noise:
#         3M (span 3): SECB IC +0.0496→+0.0575, ICIR +0.424→+0.494, t_block
#                      +2.93→+3.41, turnover 0.133→0.067 (−50%)
#         1Y (span 4): SECB IC +0.0388→+0.0405, ICIR +0.446→+0.525, t_block
#                      +1.44→+1.70, turnover 0.091→0.042 (−54%)
#     6M is left UNSMOOTHED: it's the strongest/most stable horizon, so smoothing
#     was ~flat on IC/ICIR (span 2: −0.0004 IC) — only a turnover trade, not promoted.
#   - 3M knife_lambda: falling-knife output overlay PROMOTED (2026-06-06, 8-seed
#     walk-forward, SECB, composed with smooth_span=3). Re-ranks vol×downtrend names
#     out of the top: top-decile knife score −47%, realized downside −7%, top-decile
#     mean return still positive, for only SECB IC +0.0575→+0.0563 (t_block +3.41→
#     +2.89, p 0.0005→0.0020 — still strongly significant). 6M/1Y NOT promoted: a
#     poor trade (6M erodes SECB ~6%/0.10λ for negligible downside; 1Y is power-
#     limited and drops below its detection floor). NOT a churn lever — turnover was
#     ~flat under the overlay (smoothing already owns turnover). See sweep CLI
#     `--knife-sweep` / `--knife-lambda` and `knife-overlay-falling-knife` memo.
#   - 6M max_train_months=60: rolling-60 window PROMOTED (2026-06-09, 8-seed walk-
#     forward, SECB). Focuses training on the most recent ~5 years, improving
#     ICIR by reducing influence of structurally stale 2010–2016 data:
#         6M: SECB IC +0.0474→+0.0556, ICIR 0.443→0.493 (+11.3%), t_block
#             +2.12→+2.36, turnover flat. Clearly clears the +10% bar.
#     3M/1Y left on expanding window: 3M +4.1% (below bar), 1Y −14.7% (kill —
#     long-horizon regime memory matters more than regime recency at 1Y).
#   - 6M / 1Y LambdaRank objective PROMOTED (2026-07-06, 8-seed walk-forward, SECB).
#     Train an LGBMRanker on the ordinal `sector_grade` target (per-date qcut of
#     sector_return into 5 grades) with a per-date query group — the fit optimizes a
#     listwise ranking loss aligned with the rank-IC scoring metric instead of L2
#     regress-then-rank (Poh 2020; LambdaRankIC 2026). trunc=100 (top-100 focus, also
#     what the product ships) — full-list trunc=500 was ~3 hr/horizon for negligible
#     extra signal. Before → after vs the same-seed regression baseline:
#         6M: SECB IC +0.0560→+0.0563, ICIR 0.494→0.575 (+16.4%), t_block 2.36→2.75,
#             p 0.0020→0.0005, turnover 0.116→0.105 (lower). Clears the +10% bar.
#         1Y: SECB IC +0.0579→+0.0690 (+19.2%), ICIR 0.747→1.109 (+48.5%), t_block
#             2.41→3.58, hit 0.82→0.89, turnover 0.042→0.038 (lower). Decisive.
#     3M NOT promoted: ICIR +8.9% (below bar) AND mean IC −9% (net loss). The overlays
#     (6M rolling-60, 1Y smooth_span=4) compose unchanged — they act on the ranker's
#     percentile ranks (monotone-invariant). Inference needs ZERO changes: fit_lgbm_model
#     branches on lgb_cfg.objective, LGBMRanker.predict has the same signature.
_LAMBDARANK_CFG = LGBMConfig(objective="lambdarank", lambdarank_truncation_level=100)
PRODUCTION_HORIZON_SPECS: dict[str, HorizonSpec] = {
    "1M": HorizonSpec(target_mode="rank"),
    "3M": HorizonSpec(target_mode="sector_return", feature_cols=_BASELINE_PLUS_REVMOM,
                      smooth_span=3, knife_lambda=0.20),
    "6M": HorizonSpec(target_mode="sector_grade", lgb_cfg=_LAMBDARANK_CFG,
                      feature_cols=_BASELINE_PLUS_SURPRISE, max_train_months=60),
    "1Y": HorizonSpec(target_mode="sector_grade", lgb_cfg=_LAMBDARANK_CFG,
                      feature_cols=_BASELINE_PLUS_SURPRISE, smooth_span=4),
}


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
    exempt_ids: set | None = None,
):
    """Map each feature to its within-date cross-sectional rank in [-1, 1].

    Point-in-time safe (only same-date rows) and robust to the heavy tails in raw
    factor values. Single-name (or empty) dates collapse to 0.

    MISSING VALUES STAY MISSING: a NaN feature is not ranked, so LightGBM routes it
    natively instead of the old sentinel 0.0 landing at a real cross-sectional
    position (see FUNDAMENTAL_SOURCED_FEATURES in factors/constants.py). The rank
    denominator is the count of OBSERVED values on that date.

    `exempt_ids` (production only) are names that must not DEFINE the distribution —
    user-added off-index tickers, which include leveraged and thematic ETFs whose
    momentum/vol would distort every index name's rank. They are still scored: their
    own value is placed against the member distribution. The walk-forward never
    passes this, so evaluation is unchanged.
    """
    out = panel.copy()
    g = out.groupby("date")
    exempt = (
        out["ticker_id"].isin(exempt_ids)
        if exempt_ids and "ticker_id" in out.columns
        else None
    )

    def _norm(rank_s, count_s):
        denom = (count_s - 1).clip(lower=1)
        scaled = (rank_s - 1) / denom * 2 - 1
        return np.where(
            np.isnan(np.asarray(rank_s, dtype=float)),
            np.nan,
            np.where(count_s > 1, scaled, 0.0),
        )

    for c in cols:
        if exempt is None or not bool(exempt.any()):
            r = g[c].rank(method="average")
            n = g[c].transform("count")
        else:
            # Members define the scale: rank them among themselves.
            gm = out.assign(_m=out[c].where(~exempt)).groupby("date")["_m"]
            r_mem, n = gm.rank(method="average"), gm.transform("count")
            # An exempt row's position among members = (its rank among ALL rows)
            # − (its rank among exempt rows): both count values at or below it, so
            # the difference is the member count below it. Exact up to average-tie
            # adjustment; clipped into [1, n_members] to share the member scale.
            ge = out.assign(_e=out[c].where(exempt)).groupby("date")["_e"]
            pos = (g[c].rank(method="average") - ge.rank(method="average")).clip(lower=1.0)
            r = r_mem.where(~exempt, np.minimum(pos, n.clip(lower=1)))
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


def add_knife_score_feature(panel):
    """Add `knife_score` column from rank-normalized price features.

    Must be called AFTER `rank_normalize_features` (inputs are in [-1, 1]).
    Byte-identical math to `_knife_components`: maps each input to [0, 1] via
    (x+1)/2, then knife_score = vol_p * (1 - mean(trend_p, dlow_p)).
    High only for names that are BOTH high-vol AND downtrending (below-200d-MA /
    near-52w-low). NaN rows are neutralized to the per-date cross-section median
    so no name is penalized for a missing price feature. The result is already in
    [0, 1] within each date so it is NOT re-run through rank_normalize_features.
    """
    needed = {"vol_120d", "ma_gap_200", "dist_low_252"}
    out = panel.copy()
    if not needed.issubset(out.columns):
        out["knife_score"] = 0.5
        return out
    to01 = lambda x: (x.astype(float) + 1.0) / 2.0
    vol_p = to01(out["vol_120d"])
    downtrend_p = 1.0 - (to01(out["ma_gap_200"]) + to01(out["dist_low_252"])) / 2.0
    knife = (vol_p * downtrend_p).clip(0.0, 1.0)
    if knife.isna().any():
        med = knife.groupby(out["date"]).transform("median")
        knife = knife.fillna(med).fillna(0.5)
    out["knife_score"] = knife.astype(float)
    return out


def apply_target_modes(
    panel,
    n_buckets: int = 5,
    market_horizon_returns: dict | None = None,
    sector_min_group_size: int = 5,
    n_grades: int = 5,
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
      y_{h}_beta_sector_resid = beta_resid minus its within-(date, sector)
                            median — strips market beta AND sector tilt, the
                            within-sector idiosyncratic-alpha target (graded on
                            SECB). Falls back to beta_resid for thin sectors.

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
        # eats the universe median.
        #
        # Rows whose sector is missing or too thin emit NaN and are DROPPED at fit
        # time. They used to fall back to the universe-demeaned return, which pooled
        # a differently-defined label into the same fit while `within_sector_ic`
        # (min_group_size=10) excluded those very rows from the metric — the model
        # was trained on a cohort it was never scored on. Making the target undefined
        # aligns the two.
        # A panel carrying NO sector labels at all has no sector dimension to be
        # relative to, so it degrades to the universe-demeaned target rather than an
        # all-NaN (untrainable) one. That is the `has_sector` guard's original job and
        # it still applies to direct callers and to inference before migration 015.
        if has_sector and out["sector"].notna().any():
            sec_grp = valid.groupby([out["date"], out["sector"]])
            sec_med = sec_grp.transform("median")
            sec_count = sec_grp.transform("count")
            use_sector = out["sector"].notna() & (sec_count >= sector_min_group_size)
            out[f"y_{h}_sector_return"] = np.where(use_sector, valid - sec_med, np.nan)
        else:
            out[f"y_{h}_sector_return"] = valid

        # --- Sector-relative GRADE target (E4 LambdaRank) ---
        # Per-date qcut of the sector-demeaned return into n_grades ordinal grades
        # (0..K-1) — the relevance labels a pairwise/NDCG ranking objective consumes.
        # Per-DATE (not within-sector) qcut so grades span the full ~500-name query
        # group; 20-70-name sector groups would degrade qcut (v1 decision). NaN where
        # the sector return is masked or the date has < n_grades names — those rows
        # are dropped at fit time exactly like the other precomputed targets.
        sec_ret = pd.Series(out[f"y_{h}_sector_return"], index=out.index)
        sec_ret = sec_ret.where(out[m].astype(bool))  # NaN outside the horizon mask

        def _grade_bucket(s):
            if s.notna().sum() < n_grades:
                return pd.Series(np.nan, index=s.index)
            return pd.qcut(s, n_grades, labels=False, duplicates="drop").astype(float)

        out[f"y_{h}_sector_grade"] = sec_ret.groupby(out["date"]).transform(_grade_bucket)

        # --- Vol-scaled sector-relative target (lever 1, 6M-focused) ---
        # Homoskedasticize label noise + shrink high-vol labels by dividing by
        # raw realized vol. Floor the denominator at the per-date 20th percentile
        # (PIT-safe: same-date cross-section only) so low-vol names are not
        # inflated; no cap on the top so high-vol labels are genuinely shrunk.
        # Requires `vol_120d_raw` stashed in prepare_panel before rank-normalization.
        # Falls back to `sector_return` when the column is absent (direct callers
        # of apply_target_modes in tests, or older inference paths).
        if "vol_120d_raw" in out.columns:
            vol = out["vol_120d_raw"].astype(float)
            floor = vol.groupby(out["date"]).transform(
                lambda s: np.nanpercentile(s, 20)
            )
            out[f"y_{h}_sector_return_vol"] = (
                out[f"y_{h}_sector_return"] / np.maximum(vol, floor).clip(lower=1e-4)
            )
        else:
            out[f"y_{h}_sector_return_vol"] = out[f"y_{h}_sector_return"]

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

        # --- Beta + sector double-residual target ---
        # Strip BOTH market beta and sector tilt, isolating within-sector
        # idiosyncratic alpha — the metric we now grade 1Y on (SECB). Sector-
        # demean the beta-residual within (date, sector); groups below the size
        # threshold fall back to the plain beta-residual. NaN where beta_resid is
        # NaN, so the same fit-time row filtering applies.
        if has_beta and has_sector:
            br = out[f"y_{h}_beta_resid"]
            bsec_grp = br.groupby([out["date"], out["sector"]])
            bsec_med = bsec_grp.transform("median")
            bsec_count = bsec_grp.transform("count")
            use_bsec = out["sector"].notna() & (bsec_count >= sector_min_group_size)
            out[f"y_{h}_beta_sector_resid"] = np.where(
                use_bsec & br.notna(), br - bsec_med, br
            )
        else:
            out[f"y_{h}_beta_sector_resid"] = out[f"y_{h}_beta_resid"]
    return out


def prepare_panel(
    frames: list[TickerFrame],
    grid: list,
    n_buckets: int = 5,
    max_stale_days: int = 7,
    rank_cols: list[str] | None = None,
    industry_relative: bool = False,
    min_group_size: int = 5,
    n_grades: int = 5,
    membership_filter: bool = False,
    membership_exempt_ids: set | None = None,
    log=lambda *_: None,
):
    """Full pipeline: assemble → demean target → rank-normalize features → targets.

    `membership_filter` drops (date, ticker) rows where the ticker was not an index
    member on that date. It runs BEFORE normalization on purpose: the per-date
    feature ranks must be computed over the universe that actually existed, not one
    padded with names the index had not yet promoted. `membership_exempt_ids` keeps
    rows that never have membership by design (user-added off-index tickers).
    """
    market_returns = build_universe_return_map(frames)
    panel = assemble_panel(
        frames,
        grid,
        max_stale_days=max_stale_days,
        market_returns=market_returns,
    )
    if panel.empty:
        return panel
    if membership_filter:
        panel = apply_membership_filter(panel, membership_exempt_ids, log=log)
        if panel.empty:
            return panel
    medians = cross_sectional_medians(frames)
    panel = demean_cross_sectional(panel, medians)
    panel = add_industry_neutral_momentum(panel, min_group_size=min_group_size)
    # Stash raw vol before normalization for the sector_return_vol target.
    if "vol_120d" in panel.columns:
        panel["vol_120d_raw"] = panel["vol_120d"].astype(float)
    # knife_score is computed post-normalization from the [-1,1] inputs and is
    # already [0,1] within-date — exclude it from the normalization step.
    base_cols = rank_cols or FEATURE_COLS
    norm_cols = [c for c in base_cols if c != "knife_score"]
    panel = rank_normalize_features(
        panel,
        cols=norm_cols,
        industry_relative=industry_relative,
        min_group_size=min_group_size,
        # Off-index names ride along in the panel (membership_exempt_ids) so they can
        # be scored, but they must not shift the index cross-section's ranks.
        exempt_ids=membership_exempt_ids,
    )
    # Add knife_score if requested (knife_score in base_cols means --with-knife-feature).
    if "knife_score" in base_cols:
        panel = add_knife_score_feature(panel)
    market_horizon_returns = build_market_horizon_returns(market_returns, grid)
    panel = apply_target_modes(
        panel, n_buckets, market_horizon_returns=market_horizon_returns,
        n_grades=n_grades,
    )
    return panel


# =============================================================
# Walk-forward
# =============================================================


def apply_membership_filter(panel, exempt_ids: set | None = None, log=lambda *_: None):
    """Keep only rows where the ticker was an index member on the row's date.

    Without this the cross-sections are padded with names the index promoted LATER,
    which selects on future success — the mirror image of survivorship bias. Rows
    for `exempt_ids` (user-added off-index tickers) are kept: they are scored, never
    trained on, and are excluded from training separately.

    Raises if membership was never loaded, rather than silently emptying the panel.
    Panels built by hand in tests have no `in_index` column and pass through.
    """
    if "in_index" not in panel.columns:
        return panel

    in_index = panel["in_index"]
    if in_index.isna().all():
        raise RuntimeError(
            "membership filter requested but no index_membership data is loaded — "
            "apply migration 012, run scripts.seed_index_membership, and rebuild the "
            "frame cache with --refresh-cache"
        )

    # `in_index` is object dtype (True/False/None), so compare explicitly rather
    # than fillna+astype, which pandas is deprecating for object columns.
    keep = in_index.eq(True)
    if exempt_ids:
        keep = keep | panel["ticker_id"].isin(exempt_ids)
    dropped = int((~keep).sum())
    log(f"membership filter: kept {int(keep.sum()):,} rows, dropped {dropped:,}")
    return panel[keep].reset_index(drop=True)


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
    if cfg.objective in _RANKING_OBJECTIVES:
        return _fit_lgbm_ranker(train_df, target_col, cfg, seed, shuffle, cols)
    # Pass DataFrames (not bare arrays) so feature names flow into LightGBM and
    # sklearn doesn't warn at predict time.
    X_tr = train_df[cols]
    y_tr = train_df[target_col].to_numpy(dtype=float)
    if shuffle:  # destroy feature->label link, preserve marginal => no-signal null
        y_tr = y_tr[np.random.default_rng(seed).permutation(len(y_tr))]
    model = lgb.LGBMRegressor(
        objective=cfg.objective,
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        subsample_freq=1,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        reg_alpha=cfg.reg_alpha,
        n_jobs=cfg.n_jobs,
        random_state=seed,
        verbose=-1,
    )
    model.fit(X_tr, y_tr)
    return model


# LightGBM learning-to-rank objectives (vs pointwise "regression"). Both consume an
# ordinal grade label + a per-date query group; both expose the same `.predict()`
# score signature as the regressor, so _fit_predict / walk_forward_ic / inference
# need no branch beyond fit_lgbm_model.
_RANKING_OBJECTIVES = frozenset({"lambdarank", "rank_xendcg"})


def _fit_lgbm_ranker(train_df, target_col, cfg, seed, shuffle, cols):
    """Fit an LGBMRanker on per-date query groups; returns a model with `.predict()`.

    The panel is ticker-major (assemble_panel extends per ticker), so the rows must
    be reordered to date-contiguous blocks and the query `group` sizes read off in
    that same order — LightGBM assumes each group occupies a consecutive row span.
    A stable (mergesort) sort keeps within-date order deterministic across seeds.
    label_gain is LINEAR (0,1,..,K−1): Spearman / SECB reward monotone ordering
    through the whole list, not the default 2^i−1 top-heavy gain.
    """
    import lightgbm as lgb

    ordered = train_df.sort_values("date", kind="mergesort")
    X_tr = ordered[cols]
    y_tr = ordered[target_col].to_numpy(dtype=float)
    if shuffle:  # no-signal null: permute grades, keep the group structure intact
        y_tr = y_tr[np.random.default_rng(seed).permutation(len(y_tr))]
    y_tr = np.rint(y_tr).astype(int)
    group = ordered.groupby("date", sort=False).size().to_numpy()
    n_labels = int(y_tr.max()) + 1 if y_tr.size else 2
    label_gain = list(range(max(n_labels, 2)))
    model = lgb.LGBMRanker(
        objective=cfg.objective,
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        subsample_freq=1,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        reg_alpha=cfg.reg_alpha,
        n_jobs=cfg.n_jobs,
        random_state=seed,
        verbose=-1,
        label_gain=label_gain,
        lambdarank_truncation_level=cfg.lambdarank_truncation_level,
    )
    model.fit(X_tr, y_tr, group=group)
    return model


def fit_linear_model(
    train_df,
    target_col: str,
    feature_cols: list[str] | None = None,
    alpha: float = 10.0,
    seed: int = 0,
    shuffle: bool = False,
):
    """Fit a ridge regressor on the same rank-normalized features as the GBDT.

    The features are already mapped to within-date ranks in [-1, 1], so no further
    scaling is needed and ridge is the natural low-variance complement to the tree:
    it captures the monotone linear part of the signal that a shallow GBDT
    overfits on a thin (~500-name) cross-section (Gu-Kelly-Xiu: linear models are
    competitive and more stable at low SNR; the ensemble dominates either alone).
    """
    from sklearn.linear_model import Ridge

    cols = feature_cols if feature_cols is not None else FEATURE_COLS
    # Ridge has no native missing-value handling (LightGBM does). Features are
    # within-date ranks in [-1, 1], so 0.0 is the neutral mid-rank — the least
    # informative fill available and the one that keeps the design matrix finite.
    X = np.nan_to_num(train_df[cols].to_numpy(dtype=float), nan=0.0)
    y = train_df[target_col].to_numpy(dtype=float)
    if shuffle:  # same no-signal null as the GBDT path
        y = y[np.random.default_rng(seed).permutation(len(y))]
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model


def _rank01(a: np.ndarray) -> np.ndarray:
    """Map a prediction vector to within-cross-section percentile rank in [0, 1].

    Used to put GBDT and ridge outputs (different scales/units) on a common
    footing before blending; a single-name cross-section collapses to 0.5.
    """
    import pandas as pd

    a = np.asarray(a, dtype=float)
    n = a.size
    if n < 2:
        return np.full(n, 0.5, dtype=float)
    return (pd.Series(a).rank(method="average").to_numpy() - 1.0) / (n - 1)


def blend_gbdt_linear(
    gbdt_pred: np.ndarray, linear_pred: np.ndarray, weight: float
) -> np.ndarray:
    """Weighted average of rank-transformed GBDT and ridge predictions.

    `weight` is the ridge share in [0, 1]; 0.0 is pure GBDT. Blending on RANKS
    (not raw outputs) keeps the two models commensurable regardless of target
    mode, matching how the score path rank-transforms before storage.
    """
    if weight <= 0:
        return np.asarray(gbdt_pred, dtype=float)
    return (1.0 - weight) * _rank01(gbdt_pred) + weight * _rank01(linear_pred)


def _fit_predict(
    train_df,
    test_df,
    target_col: str,
    cfg: LGBMConfig,
    seed: int,
    shuffle: bool,
    feature_cols: list[str] | None = None,
    n_seeds: int = 1,
) -> np.ndarray:
    cols = feature_cols if feature_cols is not None else FEATURE_COLS
    if n_seeds == 1:
        return fit_lgbm_model(
            train_df, target_col, cfg, seed=seed, shuffle=shuffle, feature_cols=cols
        ).predict(test_df[cols])
    # Average the seeds in RANK space, not raw-score space. LambdaRank scores have
    # no fixed scale or offset across independently-seeded models, so a raw mean is
    # a scale-weighted vote in which one wide-range seed dominates. `_rank01` is the
    # same normalization `blend_gbdt_linear` / the target blend already use.
    preds_all = np.stack([
        _rank01(fit_lgbm_model(
            train_df, target_col, cfg, seed=seed + s * 997, shuffle=shuffle, feature_cols=cols
        ).predict(test_df[cols]))
        for s in range(n_seeds)
    ])
    return preds_all.mean(axis=0)


def _target_col(horizon: str, target_mode: str) -> str:
    """Training-target column: raw return needs no precomputed column."""
    return f"r_{horizon}" if target_mode == "return" else f"y_{horizon}_{target_mode}"


# Target modes whose label is a raw return-like quantity (fat-tailed) and can be
# winsorized. rank / quantile targets are already bounded/ordinal, so clipping them
# is meaningless — the guard raises rather than silently no-op'ing.
_WINSORIZABLE_TARGET_MODES = frozenset(
    {"return", "sector_return", "sector_return_vol", "beta_resid", "beta_sector_resid"}
)


def assert_winsorizable_target(target_mode: str) -> None:
    """Guard: winsorization only makes sense for raw return-like training targets."""
    if target_mode not in _WINSORIZABLE_TARGET_MODES:
        raise ValueError(
            f"winsorization is only valid for return-like targets "
            f"{sorted(_WINSORIZABLE_TARGET_MODES)}, not {target_mode!r}"
        )


def winsorize_by_date(y, dates, pct: float):
    """Clip each date's cross-section of a return-like target to its per-date
    [pct, 1−pct] quantiles.

    PIT-safe: the clip bounds are computed from same-date rows only, so no future
    information leaks across the winsorization (same class as the median-demean).
    NaNs pass through untouched (excluded from both the quantile and the clip).
    `pct=0` (or falsy) is an exact no-op. Applied to the TRAINING label only; the
    scoring target `r_{h}` is never clipped.
    """
    import pandas as pd

    y = np.asarray(y, dtype=float)
    if not pct or pct <= 0:
        return y
    if pct >= 0.5:
        raise ValueError(f"winsorize pct must be in [0, 0.5), got {pct}")
    s = pd.Series(y)
    d = pd.Series(np.asarray(dates), index=s.index)
    lo = s.groupby(d).transform(lambda x: x.quantile(pct))
    hi = s.groupby(d).transform(lambda x: x.quantile(1.0 - pct))
    return s.clip(lower=lo, upper=hi).to_numpy()


def within_sector_ic(
    preds: np.ndarray,
    test_df,
    r_col: str,
    min_group_size: int = 10,
    group_col: str = "sector",
) -> float:
    """Mean Spearman IC averaged across GICS `group_col` groups in one cross-section.

    `group_col` is "sector" (default) or "industry" — industry is the finer cut
    and a stricter stock-selection test, but yields smaller groups, so fewer clear
    the min_group_size guard. Builds a fresh DataFrame from numpy arrays to avoid
    pandas index mis-alignment. Returns NaN when the column is absent or no group
    meets min_group_size.
    """
    import pandas as pd

    if group_col not in test_df.columns:
        return float("nan")
    tmp = pd.DataFrame({
        "pred": preds,
        "r": test_df[r_col].to_numpy(dtype=float),
        "grp": test_df[group_col].to_numpy(),
    })
    ics = []
    for _, grp in tmp.dropna(subset=["grp"]).groupby("grp"):
        if grp.shape[0] < min_group_size:
            continue
        ic = grp["pred"].corr(grp["r"], method="spearman")
        if ic == ic:
            ics.append(float(ic))
    return float(np.mean(ics)) if ics else float("nan")


def ewma_rank_by_ticker(
    records: list[dict], span: int, rank_series: list[np.ndarray] | None = None
) -> list[np.ndarray]:
    """Causal EWMA of each name's within-date percentile rank across scoring dates.

    `records` is the per-fold list (chronological) built by `walk_forward_ic`, each
    `{"ticker_ids": int array, "pred": float array, ...}`. We rank-transform each
    fold's raw predictions to [0, 1] (scale-free, since every fold refits a fresh
    model whose raw outputs aren't comparable across dates), then exponentially
    smooth each ticker's rank with `alpha = 2/(span+1)`, carrying state forward.
    A name's first appearance seeds its state with its raw rank. Returns one
    smoothed-rank array per fold, aligned to `records`.

    Pass `rank_series` (per-fold rank arrays already in [0, 1], e.g. the output of
    `apply_knife_overlay`) to smooth THOSE instead of the raw-prediction ranks — this
    is how the knife overlay composes ahead of cross-date smoothing.

    This mirrors the production blend (which smooths the stored percentile rank),
    so the walk-forward measures exactly what inference would ship.
    """
    alpha = 2.0 / (span + 1.0)
    state: dict[int, float] = {}
    out: list[np.ndarray] = []
    for idx, rec in enumerate(records):
        tids = rec["ticker_ids"]
        raw_rank = rank_series[idx] if rank_series is not None else _rank01(rec["pred"])
        sm = np.empty(len(tids), dtype=float)
        for i, t in enumerate(tids):
            t = int(t)
            state[t] = raw_rank[i] if t not in state else alpha * raw_rank[i] + (1 - alpha) * state[t]
            sm[i] = state[t]
        out.append(sm)
    return out


def rank_turnover(records: list[dict], rank_series: list[np.ndarray] | None = None) -> float:
    """Mean |Δ percentile-rank| of a name between consecutive scoring dates.

    A turnover proxy: lower = the model's relative view of names is steadier. Uses
    the raw within-date rank of each fold's predictions unless `rank_series`
    (e.g. the smoothed ranks from `ewma_rank_by_ticker`) is supplied. Averages the
    per-pair mean over names present in both consecutive cross-sections.
    """
    prev: dict[int, float] | None = None
    diffs: list[float] = []
    for idx, rec in enumerate(records):
        ranks = rank_series[idx] if rank_series is not None else _rank01(rec["pred"])
        cur = {int(t): float(ranks[i]) for i, t in enumerate(rec["ticker_ids"])}
        if prev is not None:
            shared = cur.keys() & prev.keys()
            if shared:
                diffs.append(float(np.mean([abs(cur[t] - prev[t]) for t in shared])))
        prev = cur
    return float(np.mean(diffs)) if diffs else float("nan")


def _knife_components(
    risk: dict | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Falling-knife components from a fold's carried risk features.

    `risk` holds the cross-sectionally rank-normalized [-1, 1] features
    (`vol`=vol_120d, `trend`=ma_gap_200, `dlow`=dist_low_252) attached to each
    record by `walk_forward_ic`. Map to [0, 1] percentiles: vol high = volatile;
    trend/dlow LOW = below the 200d MA / near the 52w low = falling. Returns
    `(knife, vol_p, downtrend_p)` where `downtrend_p = 1 - mean(trend_p, dlow_p)`
    and `knife = vol_p * downtrend_p` — high ONLY in the high-vol-AND-falling
    corner. Rare NaNs (a name missing a price feature) are neutralized to the
    cross-section median so they neither earn nor dodge the penalty. Returns
    `None` when the volatility feature or both trend features are absent.
    """
    if not risk or "vol" not in risk:
        return None
    to01 = lambda x: (np.asarray(x, dtype=float) + 1.0) / 2.0
    vol_p = to01(risk["vol"])
    trend_parts = [to01(risk[k]) for k in ("trend", "dlow") if k in risk]
    if not trend_parts:
        return None
    downtrend_p = 1.0 - np.mean(trend_parts, axis=0)
    knife = vol_p * downtrend_p
    if np.isnan(knife).any():
        med = np.nanmedian(knife)
        if not np.isfinite(med):
            return None
        knife = np.where(np.isnan(knife), med, knife)
        vol_p = np.nan_to_num(vol_p, nan=0.5)
        downtrend_p = np.nan_to_num(downtrend_p, nan=0.5)
    return knife, vol_p, downtrend_p


def _knife_score(risk: dict | None) -> np.ndarray | None:
    """The falling-knife penalty score in [0, 1] (see `_knife_components`)."""
    comp = _knife_components(risk)
    return None if comp is None else comp[0]


def knife_tier(risk: dict | None, hi: float = 0.8, mid: float = 0.6) -> list[str] | None:
    """Per-name falling-knife transparency tag from the same vol×downtrend components
    the overlay uses (`_knife_components`): vol_p and downtrend_p are cross-sectional
    percentiles in [0, 1] (high vol / below 200d-MA & near 52w-low).

        'high'     = vol_p >= hi  AND downtrend_p >= hi   (top-vol AND clearly falling)
        'elevated' = vol_p >= mid AND downtrend_p >= mid  (and not 'high')
        'none'     = otherwise

    Unlike the overlay this does NOT touch the rank — it just labels the stock so the
    dashboard can show *why* a high-vol downtrender sits where it does. Returns one tag
    per name aligned to the risk arrays, or `None` if the risk features are absent (the
    caller then defaults every name to 'none').
    """
    comp = _knife_components(risk)
    if comp is None:
        return None
    _knife, vol_p, downtrend_p = comp
    out: list[str] = []
    for v, d in zip(np.asarray(vol_p, dtype=float), np.asarray(downtrend_p, dtype=float)):
        if v >= hi and d >= hi:
            out.append("high")
        elif v >= mid and d >= mid:
            out.append("elevated")
        else:
            out.append("none")
    return out


def knife_overlay_ranks(
    model_ranks: np.ndarray, risk: dict | None, lam: float
) -> np.ndarray:
    """Falling-knife overlay for ONE cross-section: re-rank model percentile ranks.

        adj = (1-lam)*model_p - lam*knife_p ;  final = rank01(adj)
    pushes high-vol-AND-falling names down while sparing high-vol uptrenders and
    low-vol fallers. Returns `model_ranks` unchanged when `lam<=0` or `risk` lacks
    the features (exact no-op). Shared by the walk-forward (`apply_knife_overlay`)
    and production inference so both apply byte-identical math.
    """
    model_p = np.asarray(model_ranks, dtype=float)
    knife = _knife_score(risk) if lam > 0 else None
    if knife is None:
        return model_p
    return _rank01((1.0 - lam) * model_p - lam * _rank01(knife))


def vol_gate_flags(
    vol_med: list[float | None], pct: float = 0.80, burn_in: int = 24
) -> list[bool]:
    """Per-date "is this a stress cross-section?", knowable at the time.

    A date is gated when its cross-sectional median RAW realized vol exceeds the
    `pct` quantile of the medians on all STRICTLY EARLIER dates. Expanding window,
    so no future information; dates inside `burn_in` are never gated because a
    quantile over a handful of points is noise. Missing medians are never gated and
    do not enter the history.

    Near-parameter-free on purpose. Generic factor-vol-timing does not replicate
    (Cederburg et al. 2020); only own-vol scaling does (Barroso-Santa-Clara 2015),
    and with ~5 stress episodes in the panel there is nothing here to tune against.
    """
    flags: list[bool] = []
    history: list[float] = []
    for i, v in enumerate(vol_med):
        ok = v is not None and v == v
        if ok and i >= burn_in and len(history) >= burn_in:
            flags.append(bool(v > float(np.quantile(np.asarray(history), pct))))
        else:
            flags.append(False)
        if ok:
            history.append(float(v))
    return flags


def vol_gate_ranks(
    model_ranks: np.ndarray, gated: bool, shrink: float = 0.5
) -> np.ndarray:
    """Shrink a cross-section's percentile ranks toward 0.5 when the date is gated.

    ⚠️ This is a MONOTONE transform of the cross-section, so it leaves that date's
    Spearman rank-IC exactly unchanged — there is no shrink factor that moves IC
    toward zero (shrink=0 makes it undefined, not 0). It bites only where ranks are
    compared ACROSS dates, i.e. composed with `ewma_rank_by_ticker`: halving the
    spread of today's ranks halves their weight in the EWMA, so a gated date leans
    on the pre-stress rank instead. A horizon with `smooth_span == 0` is therefore
    provably unaffected, and 6M is excluded on that ground rather than by result.
    """
    ranks = np.asarray(model_ranks, dtype=float)
    if not gated or shrink >= 1.0:
        return ranks
    return 0.5 + (ranks - 0.5) * float(shrink)


def apply_vol_gate(
    rank_series: list[np.ndarray], vol_med: list[float | None],
    pct: float = 0.80, burn_in: int = 24, shrink: float = 0.5,
) -> tuple[list[np.ndarray], list[bool]]:
    """Walk-forward adapter: gate each fold's ranks, returning ranks + the flags."""
    flags = vol_gate_flags(vol_med, pct=pct, burn_in=burn_in)
    gated = [vol_gate_ranks(rk, g, shrink)
             for rk, g in zip(rank_series, flags, strict=True)]
    return gated, flags


def apply_knife_overlay(records: list[dict], lam: float) -> list[np.ndarray]:
    """Per-fold rank arrays after the falling-knife output overlay (see
    `knife_overlay_ranks`). `lam=0` returns each fold's model ranks unchanged.
    """
    return [knife_overlay_ranks(_rank01(rec["pred"]), rec.get("risk"), lam)
            for rec in records]


def top_decile_risk(
    records: list[dict], rank_series: list[np.ndarray], decile: float = 0.1
) -> dict:
    """Risk character of each fold's top-decile names, averaged across folds.

    `rank_series` is the per-fold FINAL rank arrays (e.g. from
    `apply_knife_overlay`), aligned to `records`. For each fold we take the top
    `ceil(decile*n)` names by rank and collect:
      knife      mean falling-knife score of the top names (ex-ante)
      vol_p      mean volatility percentile of the top names (ex-ante)
      downtrend  fraction of top names with downtrend_p > 0.5 (ex-ante)
      mean_r     mean realized demeaned return of the top names (ex-post alpha check)
      downside   downside semi-deviation sqrt(mean(min(r,0)^2)) of the top names
      tail_frac  fraction of top names whose realized r is below the fold's 10th pct
    Lower knife/vol_p/downtrend/downside/tail_frac at steady-or-higher mean_r is the
    win condition. Folds lacking risk features are skipped.
    """
    keys = ("knife", "vol_p", "downtrend", "mean_r", "downside", "tail_frac")
    acc: dict[str, list[float]] = {k: [] for k in keys}
    for rec, rk in zip(records, rank_series, strict=True):
        comp = _knife_components(rec.get("risk"))
        if comp is None:
            continue
        knife, vol_p, downtrend_p = comp
        r = np.asarray(rec["r"], dtype=float)
        rk = np.asarray(rk, dtype=float)
        n = rk.size
        k = max(1, math.ceil(decile * n))
        top = np.argsort(rk)[-k:]  # indices of the highest-ranked names
        rt = r[top]
        neg = np.minimum(rt, 0.0)
        tail = np.nanpercentile(r, 10)
        acc["knife"].append(float(np.mean(knife[top])))
        acc["vol_p"].append(float(np.mean(vol_p[top])))
        acc["downtrend"].append(float(np.mean(downtrend_p[top] > 0.5)))
        acc["mean_r"].append(float(np.nanmean(rt)))
        acc["downside"].append(float(np.sqrt(np.nanmean(neg ** 2))))
        acc["tail_frac"].append(float(np.nanmean(rt < tail)))
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}


def knife_sweep_table(
    records: list[dict],
    lambdas: list[float],
    smooth_span: int = 0,
    sector_group_col: str = "sector",
    compute_sector_ic: bool = True,
) -> list[dict]:
    """Evaluate the knife overlay across `lambdas` with NO refit (reuses records).

    For each lambda: overlay → optional cross-date smoothing → score mean rank-IC,
    within-sector IC, turnover, and the top-decile risk metric. `lambdas[0]` is the
    baseline (use 0.0). Mirrors the `--smooth-span` post-hoc sweep.
    """
    import pandas as pd

    rows: list[dict] = []
    for lam in lambdas:
        ranks = apply_knife_overlay(records, lam)
        if smooth_span > 0:
            ranks = ewma_rank_by_ticker(records, smooth_span, rank_series=ranks)
        ics: list[float] = []
        sec_ics: list[float] = []
        for rec, rk in zip(records, ranks, strict=True):
            ic = pd.Series(rk).corr(pd.Series(rec["r"]), method="spearman")
            if ic == ic:
                ics.append(float(ic))
            if compute_sector_ic and rec.get("sector") is not None:
                sdf = pd.DataFrame({"r": rec["r"], sector_group_col: rec["sector"]})
                s = within_sector_ic(rk, sdf, "r", group_col=sector_group_col)
                if s == s:
                    sec_ics.append(float(s))
        row = {
            "lam": lam,
            "mean_ic": float(np.mean(ics)) if ics else float("nan"),
            "sec_ic": float(np.mean(sec_ics)) if sec_ics else float("nan"),
            "turnover": rank_turnover(records, rank_series=ranks),
        }
        row.update(top_decile_risk(records, ranks))
        rows.append(row)
    return rows


def target_blend_sweep(
    records_base: list[dict],
    records_alt: list[dict],
    weights: list[float],
    *,
    block_size: int,
    reps: int = 2000,
    seed: int = 1337,
    sector_group_col: str = "sector",
    compute_sector_ic: bool = True,
) -> list[dict]:
    """Evaluate a two-target rank-ensemble across blend `weights` with NO refit.

    `records_base` / `records_alt` come from two `walk_forward_ic(..., return_records=
    True)` runs on the SAME horizon/panel but DIFFERENT `target_mode`. They are matched
    by each fold's `date` (so a skipped fold on one side never misaligns the other) and
    share that fold's realized `r` / sector. For weight `w` the per-fold final rank is
    `rank01((1-w)*rank01(pred_base) + w*rank01(pred_alt))` — w=0 is the base-target
    baseline, w=1 the pure alternate target. Decorrelated label errors average out, so
    the blend trades a little bias for lower cross-date IC variance (higher ICIR).
    Names absent from the matched alt fold keep their base rank. Reports mean rank-IC,
    within-sector IC + its moving-block t / p, and turnover per weight. Mirrors
    `knife_sweep_table`.
    """
    import pandas as pd

    alt_by_date = {rec.get("date"): rec for rec in records_alt}
    rows: list[dict] = []
    for w in weights:
        ranks_out: list[np.ndarray] = []
        ics: list[float] = []
        sec_ics: list[float] = []
        for rec in records_base:
            base_rank = _rank01(rec["pred"])
            arec = alt_by_date.get(rec.get("date"))
            if arec is None or w == 0.0:
                final = base_rank
            else:
                alt_rank = _rank01(arec["pred"])
                alt_map = {int(t): alt_rank[i] for i, t in enumerate(arec["ticker_ids"])}
                blended = np.array([
                    (1.0 - w) * base_rank[i] + w * alt_map.get(int(t), base_rank[i])
                    for i, t in enumerate(rec["ticker_ids"])
                ], dtype=float)
                final = _rank01(blended)
            ranks_out.append(final)
            ic = pd.Series(final).corr(pd.Series(rec["r"]), method="spearman")
            if ic == ic:
                ics.append(float(ic))
            if compute_sector_ic and rec.get("sector") is not None:
                sdf = pd.DataFrame({"r": rec["r"], sector_group_col: rec["sector"]})
                s = within_sector_ic(final, sdf, "r", group_col=sector_group_col)
                if s == s:
                    sec_ics.append(float(s))
        boot = block_bootstrap_summary(sec_ics, block_size=block_size, reps=reps, seed=seed)
        rows.append({
            "w": w,
            "mean_ic": float(np.mean(ics)) if ics else float("nan"),
            "sec_ic": float(np.mean(sec_ics)) if sec_ics else float("nan"),
            "sec_t_block": boot["t_block"],
            "sec_p": boot["p_value"],
            "turnover": rank_turnover(records_base, rank_series=ranks_out),
        })
    return rows


def regularization_sweep(
    panel,
    horizon: str,
    named_configs: list[tuple[str, "LGBMConfig"]],
    *,
    wf_cfg: "WalkForwardConfig",
    target_mode: str,
    feature_cols: list[str] | None,
    n_seeds: int,
    block_size: int,
    reps: int = 2000,
    seed: int = 1337,
    sector_group_col: str = "sector",
    compute_sector_ic: bool = True,
    max_test_date=None,
    min_test_date=None,
    log=lambda *_: None,
) -> list[dict]:
    """Compare LightGBM regularization configs at one horizon — each a FULL refit.

    Unlike the knife / target-blend sweeps this cannot reuse records (every config
    changes the fit), so it re-runs `walk_forward_ic` per config. Reports mean rank-IC,
    within-sector IC + its naive ICIR + moving-block t / p, and turnover so a config can
    be judged on variance reduction (higher ICIR / block-t and/or lower turnover at
    steady-or-higher mean IC). `named_configs[0]` should be the production baseline.
    """
    rows: list[dict] = []
    for name, cfg in named_configs:
        log(f"[reg-sweep] fitting '{name}' ...")
        res = walk_forward_ic(
            panel, horizon, cfg, wf_cfg, seed=seed, shuffle=False,
            target_mode=target_mode, feature_cols=feature_cols, n_seeds=n_seeds,
            compute_sector_ic=compute_sector_ic, sector_group_col=sector_group_col,
            max_test_date=max_test_date, min_test_date=min_test_date,
        )
        sec = res.get("sector_summary", {})
        boot = block_bootstrap_summary(
            res.get("sector_ic_values", []), block_size=block_size, reps=reps, seed=seed
        )
        rows.append({
            "name": name,
            "mean_ic": res["summary"]["mean_ic"],
            "sec_ic": sec.get("mean_ic", float("nan")),
            "sec_icir": sec.get("icir", float("nan")),
            "sec_t_block": boot["t_block"],
            "sec_p": boot["p_value"],
            "turnover": res.get("turnover_raw", float("nan")),
        })
    return rows


def feature_diagnostics(
    panel,
    horizon: str,
    candidate_cols: list[str],
    existing_cols: list[str] | None = None,
    *,
    consolidation_col: str = "efficiency_ratio_120d",
    sector_group_col: str = "sector",
    min_names: int = 30,
    block_size: int | None = None,
    reps: int = 2000,
    seed: int = 1337,
) -> list[dict]:
    """Vet candidate features for decorrelation + standalone signal — NO model fits.

    For each `candidate_col` (expects the rank-normalized prepared panel) this reports:
      * mean_abs_corr / top_corr — mean |Spearman corr| vs each `existing_cols` feature
        (the existing book) and the 3 highest individual correlations. Low = decorrelated.
      * er_corr — |corr| vs the consolidation proxy (`efficiency_ratio_120d`); high means
        the feature is largely a trend-vs-sideways proxy.
      * sec_ic / sec_t_block / sec_p — standalone within-sector IC at `horizon` vs the
        realized return, moving-block-bootstrapped (the gate's signal test; sign is
        informational — the GBDT can use either direction).
      * ic_sideways / ic_mid / ic_trending — the feature's plain cross-sectional IC within
        low / mid / high efficiency-ratio tertiles. Signal that lives ONLY in `ic_sideways`
        is a consolidation-regime bet, not broad alpha (the sideways-confound check).

    A candidate should advance only if it is low-correlation AND carries a real standalone
    SECB signal that is not confined to the sideways tertile. Pure panel statistics, so a
    full sweep over candidates is cheap.
    """
    import pandas as pd

    existing_cols = existing_cols or list(FEATURE_COLS)
    r_col, m_col = f"r_{horizon}", f"mask_{horizon}"
    block_size = block_size or max(1, math.ceil(HORIZON_TRADING_DAYS[horizon] / 21))
    pdf = panel[panel[m_col].astype(bool) & panel[r_col].notna()]

    rows: list[dict] = []
    for cand in candidate_cols:
        others = [c for c in existing_cols if c != cand]
        corr_acc: dict[str, list[float]] = {c: [] for c in others}
        er_corrs: list[float] = []
        ic_all: list[float] = []
        ic_tert: dict[int, list[float]] = {0: [], 1: [], 2: []}
        for _d, g in pdf.groupby("date"):
            if len(g) < min_names:
                continue
            cvals = g[cand]
            for c in others:
                cc = cvals.corr(g[c], method="spearman")
                if cc == cc:
                    corr_acc[c].append(abs(float(cc)))
            if consolidation_col in g.columns and consolidation_col != cand:
                ec = cvals.corr(g[consolidation_col], method="spearman")
                if ec == ec:
                    er_corrs.append(abs(float(ec)))
            sec_ic = within_sector_ic(cvals.to_numpy(), g, r_col, group_col=sector_group_col)
            if sec_ic == sec_ic:
                ic_all.append(float(sec_ic))
            if consolidation_col in g.columns:
                er = g[consolidation_col].to_numpy(dtype=float)
                q1, q2 = np.nanpercentile(er, [33.333, 66.667])
                for t, sel in ((0, er <= q1), (1, (er > q1) & (er <= q2)), (2, er > q2)):
                    sub = g[sel]
                    if len(sub) >= 10:
                        ic = sub[cand].corr(sub[r_col], method="spearman")
                        if ic == ic:
                            ic_tert[t].append(float(ic))
        mean_abs = {c: float(np.mean(v)) for c, v in corr_acc.items() if v}
        top3 = sorted(mean_abs.items(), key=lambda kv: kv[1], reverse=True)[:3]
        boot = block_bootstrap_summary(ic_all, block_size=block_size, reps=reps, seed=seed)
        rows.append({
            "feature": cand,
            "mean_abs_corr": float(np.mean(list(mean_abs.values()))) if mean_abs else float("nan"),
            "top_corr": top3,
            "er_corr": float(np.mean(er_corrs)) if er_corrs else float("nan"),
            "sec_ic": boot["mean_ic"],
            "sec_t_block": boot["t_block"],
            "sec_p": boot["p_value"],
            "ic_sideways": float(np.mean(ic_tert[0])) if ic_tert[0] else float("nan"),
            "ic_mid": float(np.mean(ic_tert[1])) if ic_tert[1] else float("nan"),
            "ic_trending": float(np.mean(ic_tert[2])) if ic_tert[2] else float("nan"),
        })
    return rows


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
    n_seeds: int = 1,
    compute_sector_ic: bool = False,
    sector_group_col: str = "sector",
    linear_blend: float = 0.0,
    ridge_alpha: float = 10.0,
    smooth_span: int = 0,
    knife_lambda: float = 0.0,
    winsorize_pct: float = 0.0,
    max_test_date=None,
    min_test_date=None,
    vol_gate: bool = False,
    return_records: bool = False,
) -> dict:
    """Expanding-window walk-forward; return summary + per-fold rank-IC rows.

    Trains on `target_mode` (return/rank/quantile) but always SCORES rank-IC against
    the realized demeaned return `r_{horizon}`, so modes are directly comparable.

    `knife_lambda > 0` applies the falling-knife output overlay (`apply_knife_overlay`)
    and `smooth_span > 0` the causal cross-date EWMA (`ewma_rank_by_ticker`) to the
    per-fold ranks BEFORE scoring — pure post-steps that leave the fit untouched but
    measure the de-risked / smoothed signal inference would ship. When both are on the
    overlay composes first, then smoothing. The result always carries `rank_turnover`
    (mean |Δ rank| between consecutive dates) for the raw signal, and the transformed
    turnover (`turnover_smoothed`) when either post-step is on.

    `max_test_date` / `min_test_date` restrict which fold TEST dates are scored;
    the training window per fold is unchanged. This is how the frozen holdout is
    enforced: selection runs pass `max_test_date = holdout_start - horizon` so
    every selection label is fully realized before the holdout opens, and the
    one-shot holdout run passes `min_test_date = holdout_start`. There is no
    default -- truncation is always explicit.
    """
    import pandas as pd

    lgb_cfg = lgb_cfg or LGBMConfig()
    wf_cfg = wf_cfg or WalkForwardConfig()
    r_col, m_col = f"r_{horizon}", f"mask_{horizon}"
    t_col = _target_col(horizon, target_mode)
    # Winsorize the TRAINING label only (per-date quantile clip); scoring stays on
    # the unclipped realized return `r_col`. Precompute a clipped column once over
    # the whole panel — the per-date clip uses same-date rows only, so it is
    # identical to clipping each fold's training slice (fold-invariant, PIT-safe).
    fit_col = t_col
    if winsorize_pct and winsorize_pct > 0:
        assert_winsorizable_target(target_mode)
        fit_col = f"{t_col}__wins"
        panel = panel.assign(**{
            fit_col: winsorize_by_date(
                panel[t_col].to_numpy(dtype=float),
                panel["date"].to_numpy(),
                winsorize_pct,
            )
        })
    embargo_steps = max(1, math.ceil(HORIZON_TRADING_DAYS[horizon] / 21))

    grid_dates = sorted(panel["date"].unique())
    folds = walk_forward_folds(grid_dates, wf_cfg.min_train_months, embargo_steps)
    if max_test_date is not None or min_test_date is not None:
        folds = [
            (td, cut) for td, cut in folds
            if (max_test_date is None or td <= max_test_date)
            and (min_test_date is None or td >= min_test_date)
        ]
        log(f"test-date window: {len(folds)} folds kept "
            f"[{min_test_date or '-'} .. {max_test_date or '-'}]")

    fold_rows: list[dict] = []
    records: list[dict] = []  # per-fold (ticker_ids, raw pred, realized r, sector) for smoothing/turnover
    for fi, (test_date, cutoff) in enumerate(folds):
        train = panel[(panel["date"] <= cutoff) & panel[m_col] & panel[t_col].notna()]
        if wf_cfg.max_train_months is not None:
            # Count LABELED training dates, matching gbm_inference.fit_horizon_models
            # exactly. Slicing by grid position instead would keep a different number
            # of real training dates whenever a grid date carries no labeled rows, so
            # the walk-forward would measure a different model than production ships.
            train_dates = sorted(train["date"].unique())
            if len(train_dates) > wf_cfg.max_train_months:
                train = train[train["date"] >= train_dates[-wf_cfg.max_train_months]]
        test = panel[(panel["date"] == test_date) & panel[m_col]]
        if test.shape[0] < wf_cfg.min_names or train.empty:
            continue

        preds = _fit_predict(
            train, test, fit_col, lgb_cfg, seed + fi, shuffle,
            feature_cols=feature_cols, n_seeds=n_seeds,
        )
        if linear_blend > 0:
            # `test` is a single month-end cross-section here, so rank-blending is
            # well-defined. The null path shuffles both models identically.
            lin = fit_linear_model(
                train, fit_col, feature_cols=feature_cols, alpha=ridge_alpha,
                seed=seed + fi, shuffle=shuffle,
            )
            cols = feature_cols if feature_cols is not None else FEATURE_COLS
            lin_pred = lin.predict(
                np.nan_to_num(test[cols].to_numpy(dtype=float), nan=0.0)
            )
            preds = blend_gbdt_linear(preds, lin_pred, linear_blend)
        ic = pd.Series(preds).corr(pd.Series(test[r_col].to_numpy(dtype=float)), method="spearman")
        if ic != ic:  # NaN (zero-variance cross-section)
            continue
        fold = {"date": test_date, "ic": float(ic), "n_test": int(test.shape[0]),
                "n_train": int(train.shape[0])}
        # Raw (pre-normalization) vol median: a regime proxy for --regime-report.
        # The normalized vol_120d column cannot serve — a per-date median of
        # within-date ranks is ~constant by construction.
        if "vol_120d_raw" in test.columns:
            fold["vol_raw_med"] = float(np.nanmedian(test["vol_120d_raw"].to_numpy(dtype=float)))
        if compute_sector_ic:
            fold["sector_ic"] = within_sector_ic(
                preds, test, r_col, group_col=sector_group_col
            )
        fold_rows.append(fold)
        risk = {
            key: test[col].to_numpy(dtype=float)
            for col, key in (("vol_120d", "vol"), ("ma_gap_200", "trend"),
                             ("dist_low_252", "dlow"))
            if col in test.columns
        }
        records.append({
            "date": test_date,
            "ticker_ids": test["ticker_id"].to_numpy(),
            "pred": np.asarray(preds, dtype=float),
            "r": test[r_col].to_numpy(dtype=float),
            "sector": test[sector_group_col].to_numpy() if sector_group_col in test.columns else None,
            # Rank-normalized size, for the partial-correlation size-neutral IC.
            "size": (test["log_market_cap"].to_numpy(dtype=float)
                     if "log_market_cap" in test.columns else None),
            "risk": risk,
        })
        log(
            f"  fold {test_date}  ic {ic:+.4f}  "
            f"n_test={test.shape[0]:>4d}  n_train={train.shape[0]}"
        )

    # Post-hoc rank transforms (no refit): falling-knife overlay, then cross-date
    # EWMA smoothing. Recompute each fold's IC / SECB on the transformed ranks
    # (records is 1:1 aligned with fold_rows). Spearman is invariant to the final
    # re-rank, so scoring on the transformed rank == scoring on its rank. lam/span = 0
    # are no-ops, so the default path keeps the raw per-fold IC scored in the loop.
    turnover_smoothed = None
    n_gated = 0
    if (knife_lambda > 0 or smooth_span > 0 or vol_gate) and records:
        ranks = apply_knife_overlay(records, knife_lambda)  # lam=0 → model ranks
        if vol_gate:
            # Must precede smoothing: the gate has no effect on a single date's
            # rank-IC (monotone), it only changes how much that date contributes
            # to the cross-date EWMA.
            ranks, gate_flags = apply_vol_gate(
                ranks, [f.get("vol_raw_med") for f in fold_rows]
            )
            n_gated = int(sum(gate_flags))
            for fold, g in zip(fold_rows, gate_flags, strict=True):
                fold["vol_gated"] = bool(g)
        if smooth_span > 0:
            ranks = ewma_rank_by_ticker(records, smooth_span, rank_series=ranks)
        for fold, rec, rk in zip(fold_rows, records, ranks, strict=True):
            fold["ic"] = float(pd.Series(rk).corr(pd.Series(rec["r"]), method="spearman"))
            if compute_sector_ic and rec["sector"] is not None:
                sdf = pd.DataFrame({r_col: rec["r"], sector_group_col: rec["sector"]})
                fold["sector_ic"] = within_sector_ic(rk, sdf, r_col, group_col=sector_group_col)
        turnover_smoothed = rank_turnover(records, rank_series=ranks)

    result = {"summary": summarize([r["ic"] for r in fold_rows]), "folds": fold_rows}
    result["turnover_raw"] = rank_turnover(records)
    result["turnover_smoothed"] = turnover_smoothed
    result["n_vol_gated"] = n_gated
    if return_records:
        # Raw per-fold (ticker_ids, pred, r, sector) so a caller can sweep smoothing
        # spans / turnover WITHOUT refitting (the fits dominate cost).
        result["records"] = records
    if compute_sector_ic:
        # sector_summary uses the same naive ICIR×√N formula as universe — caller
        # should apply block_bootstrap_summary on sector_ic_values for honest t.
        s_ics = [r["sector_ic"] for r in fold_rows if r["sector_ic"] == r["sector_ic"]]
        result["sector_summary"] = summarize(s_ics)
        result["sector_ic_values"] = s_ics
    return result


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


def _partial_spearman(pred: np.ndarray, r: np.ndarray, size: np.ndarray) -> float:
    """Spearman correlation between pred and r, holding size constant.

    Closed form: rho(p,r|s) = (rho_pr - rho_ps*rho_rs) / sqrt((1-rho_ps^2)(1-rho_rs^2)).
    Answers "does the signal rank names correctly among same-size peers?" — the
    size premium is a documented risk premium, so IC that survives this is the
    part not explained by a size tilt.
    """
    import pandas as pd

    ok = np.isfinite(pred) & np.isfinite(r) & np.isfinite(size)
    if ok.sum() < 10:
        return float("nan")
    p, y, s = pred[ok], r[ok], size[ok]
    if len(set(s.tolist())) < 2 or len(set(p.tolist())) < 2 or len(set(y.tolist())) < 2:
        return float("nan")

    def rho(a, b):
        return pd.Series(a).corr(pd.Series(b), method="spearman")

    r_pr, r_ps, r_rs = rho(p, y), rho(p, s), rho(y, s)
    if any(v != v for v in (r_pr, r_ps, r_rs)):
        return float("nan")
    denom = math.sqrt(max(0.0, (1 - r_ps ** 2) * (1 - r_rs ** 2)))
    if denom <= 1e-12:
        return float("nan")
    return float((r_pr - r_ps * r_rs) / denom)


def size_neutral_summary(
    records: list[dict],
    block_size: int,
    reps: int = 2000,
    min_group_size: int = 10,
) -> dict:
    """Universe and within-sector IC after partialling out size, per fold.

    Post-hoc on the walk-forward's own predictions — no refits. The audit found
    that dropping `log_market_cap` removed significance at every horizon, i.e. the
    result was load-bearing on a size tilt; this quantifies what is left once size
    is held constant.
    """
    import pandas as pd

    uni: list[float] = []
    sec: list[float] = []
    for rec in records:
        size = rec.get("size")
        if size is None:
            continue
        pred = np.asarray(rec["pred"], dtype=float)
        r = np.asarray(rec["r"], dtype=float)
        size = np.asarray(size, dtype=float)
        val = _partial_spearman(pred, r, size)
        if val == val:
            uni.append(val)

        sectors = rec.get("sector")
        if sectors is None:
            continue
        per_sector: list[float] = []
        for group in pd.unique(pd.Series(sectors).dropna()):
            m = np.asarray(pd.Series(sectors).to_numpy() == group)
            if m.sum() < min_group_size:
                continue
            val_s = _partial_spearman(pred[m], r[m], size[m])
            if val_s == val_s:
                per_sector.append(val_s)
        if per_sector:
            sec.append(float(np.mean(per_sector)))

    return {
        "universe": {**summarize(uni),
                     **block_bootstrap_summary(uni, block_size, reps=reps)},
        "sector": {**summarize(sec),
                   **block_bootstrap_summary(sec, block_size, reps=reps)},
    }


def regime_report(folds: list[dict], burn_in: int = 24) -> dict:
    """Per-fold IC broken out by calendar year and by volatility regime.

    The headline is a pooled mean over 2013-2026, which hides regime dependence —
    especially for a size-tilted signal, where a mega-cap-led stretch and a
    broadening stretch are different worlds. Pure function over existing fold rows.

    Two vol-regime tables are returned. `by_vol_regime` cuts tertiles over the WHOLE
    fold set — in-sample breakpoints, fine for describing the panel, useless as
    evidence for a tradeable rule. `by_vol_regime_pit` cuts each fold against
    quantiles of the folds strictly BEFORE it (expanding window, `burn_in` folds of
    history required), so every bucket assignment was knowable at the time. A regime
    claim that only survives the in-sample cut is not a claim.
    """
    from collections import defaultdict

    rows = [f for f in folds if f.get("ic") == f.get("ic")]
    by_year: dict[int, list[dict]] = defaultdict(list)
    for f in rows:
        by_year[f["date"].year].append(f)

    def agg(items: list[dict]) -> dict:
        ics = [f["ic"] for f in items]
        secs = [f["sector_ic"] for f in items
                if f.get("sector_ic") is not None and f["sector_ic"] == f["sector_ic"]]
        return {
            "n": len(items),
            "mean_ic": float(np.mean(ics)) if ics else float("nan"),
            "mean_sector_ic": float(np.mean(secs)) if secs else float("nan"),
            "hit_rate": float(np.mean([1.0 if v > 0 else 0.0 for v in ics])) if ics else float("nan"),
        }

    years = {year: agg(items) for year, items in sorted(by_year.items())}

    # Volatility regime: split folds into tertiles of the cross-sectional median
    # of RAW realized vol on the fold date.
    vol_rows = [f for f in rows if f.get("vol_raw_med") == f.get("vol_raw_med")
                and f.get("vol_raw_med") is not None]
    vol_rows.sort(key=lambda f: f["date"])
    regimes: dict[str, dict] = {}
    if len(vol_rows) >= 6:
        vols = np.asarray([f["vol_raw_med"] for f in vol_rows], dtype=float)
        lo, hi = np.quantile(vols, [1 / 3, 2 / 3])
        buckets = {
            "low_vol": [f for f in vol_rows if f["vol_raw_med"] <= lo],
            "mid_vol": [f for f in vol_rows if lo < f["vol_raw_med"] <= hi],
            "high_vol": [f for f in vol_rows if f["vol_raw_med"] > hi],
        }
        regimes = {name: agg(items) for name, items in buckets.items() if items}

    # Same cut, but each fold is bucketed against the quantiles of the folds that
    # PRECEDE it. Folds inside the burn-in have no usable history and are dropped
    # rather than bucketed on a handful of observations.
    pit_buckets: dict[str, list[dict]] = {"low_vol": [], "mid_vol": [], "high_vol": []}
    for i, f in enumerate(vol_rows):
        if i < burn_in:
            continue
        hist = np.asarray([g["vol_raw_med"] for g in vol_rows[:i]], dtype=float)
        p_lo, p_hi = np.quantile(hist, [1 / 3, 2 / 3])
        v = f["vol_raw_med"]
        name = "low_vol" if v <= p_lo else ("mid_vol" if v <= p_hi else "high_vol")
        pit_buckets[name].append(f)
    regimes_pit = {name: agg(items) for name, items in pit_buckets.items() if items}

    return {
        "by_year": years,
        "by_vol_regime": regimes,
        "by_vol_regime_pit": regimes_pit,
        "pit_burn_in": burn_in,
        "pit_n_scored": sum(len(v) for v in pit_buckets.values()),
    }


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
            "se_block": float("nan"), "min_detect_ic": float("nan"),
        }

    block_size = max(1, min(int(block_size), n))
    reps = max(0, int(reps))
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    effective_blocks = n / block_size
    t_block = mean / std * math.sqrt(effective_blocks) if std > 0 else float("nan")

    if reps == 0:
        # SE of the mean from the overlap-adjusted block count (no resampling).
        se_block = std / math.sqrt(effective_blocks) if effective_blocks > 0 else float("nan")
        return {
            "n_folds": n, "block_size": block_size, "reps": reps, "mean_ic": mean,
            "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan"),
            "effective_blocks": effective_blocks, "t_block": t_block,
            "se_block": se_block,
            "min_detect_ic": 1.96 * se_block if se_block == se_block else float("nan"),
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
    # Bootstrap SE of the mean IC; smallest |mean IC| a two-sided 95% test could
    # resolve at this block count. For power-limited horizons (1Y: annual labels
    # ⇒ few independent blocks) min_detect_ic can exceed any plausible true IC,
    # which is the honest read — not a model failure.
    se_block = float(boot_means.std(ddof=1)) if reps > 1 else float("nan")
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
        "se_block": se_block,
        "min_detect_ic": 1.96 * se_block if se_block == se_block else float("nan"),
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


def _print_power(tag: str, s: dict) -> None:
    """Statistical-power read: the smallest |mean IC| this block count can resolve.

    For overlapping long-horizon labels (1Y) eff_blocks is small, so min_detect|IC|
    is large — a non-significant result there may be a power ceiling, not a dead
    signal. Read mean_ic against min_detect, not just against zero.
    """
    print(
        f"{tag:<5} eff_blocks={s['effective_blocks']:.1f}  se={s['se_block']:.4f}  "
        f"min_detect|IC|@95%={s['min_detect_ic']:.4f}  "
        f"(observed mean_ic={s['mean_ic']:+.4f})"
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
    if args.with_eps_surprise:
        cols += list(EPS_SURPRISE_FEATURES)
    if args.with_revision_momentum:
        cols += list(REVISION_MOMENTUM_FEATURES)
    if args.with_forward_valuation:
        cols += list(FORWARD_VALUATION_FEATURES)
    if args.with_lottery:
        cols += list(LOTTERY_FEATURES)
    if args.with_microstructure:
        cols += list(MICROSTRUCTURE_FEATURES)
    if args.with_eps_dispersion:
        cols += list(EPS_DISPERSION_FEATURES)
    if args.with_short_interest:
        cols += list(SHORT_INTEREST_FEATURES)
    if args.with_knife_feature:
        cols += list(KNIFE_FEATURES)
    if args.with_seasonality:
        cols += list(SEASONALITY_FEATURES)
    if args.with_insider:
        cols += list(INSIDER_FEATURES)
    if getattr(args, "with_sentiment", False):
        cols += list(SENTIMENT_FEATURES)
    if getattr(args, "with_estimate_missing", False):
        cols += list(ESTIMATE_MISSING_FEATURES)
    # Ad-hoc single features (e.g. isolating one member of a pack for an ablation).
    if getattr(args, "extra_features", None):
        cols += [c.strip() for c in args.extra_features.split(",") if c.strip()]
    # De-dupe while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


async def run(args) -> None:
    lgb_cfg = LGBMConfig(
        objective=args.objective,
        lambdarank_truncation_level=args.lambdarank_truncation,
    )
    if args.subsample is not None:
        lgb_cfg = replace(lgb_cfg, subsample=args.subsample)
    if args.objective in _RANKING_OBJECTIVES:
        if args.target != "sector_grade":
            raise SystemExit(
                f"--objective {args.objective} requires --target sector_grade "
                f"(ordinal grade labels), got --target {args.target}"
            )
        if args.with_linear_blend > 0:
            raise SystemExit(
                "--with-linear-blend is unsupported with a ranking objective "
                "(untested ranker+ridge combination; no production horizon blends)"
            )
    # Default the rolling-window length to the horizon's PRODUCTION spec so a
    # re-validation measures the model that actually ships (6M is max_train_months=60).
    # Explicit --max-train-months wins; pass 0 to force an expanding window.
    spec_window = getattr(
        PRODUCTION_HORIZON_SPECS.get(args.horizon, HorizonSpec()), "max_train_months", None
    )
    if args.max_train_months is None:
        max_train_months = spec_window
        window_src = f"production spec for {args.horizon}"
    elif args.max_train_months == 0:
        max_train_months = None
        window_src = "forced expanding (--max-train-months 0)"
    else:
        max_train_months = args.max_train_months
        window_src = "--max-train-months"
    wf_cfg = WalkForwardConfig(
        min_train_months=args.min_train_months,
        max_train_months=max_train_months,
        min_names=args.min_names,
    )
    print(
        f"training window: "
        f"{'expanding' if max_train_months is None else f'rolling {max_train_months} months'}"
        f"  ({window_src})"
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
            n_grades=args.rank_grades,
            membership_filter=args.membership_filter,
            log=print,
        )
        if panel.empty:
            raise SystemExit("empty panel (not enough history?)")
        dates = sorted(panel["date"].unique())
        print(
            f"panel: rows={len(panel)}  dates={len(dates)}  "
            f"range={dates[0]}..{dates[-1]}  tickers={panel['ticker_id'].nunique()}"
        )

        if args.feature_diagnostics:
            # Default the candidates to the packs the user opted into (everything beyond
            # the production FEATURE_COLS). No model fits — just panel statistics.
            # --diagnose-features overrides that so an IN-BOOK feature can be vetted
            # too (e.g. re-checking fund_available after the survivorship backfill).
            if args.diagnose_features:
                candidates = [c.strip() for c in args.diagnose_features.split(",") if c.strip()]
                missing = [c for c in candidates if c not in panel.columns]
                if missing:
                    raise SystemExit(f"--diagnose-features: not in the panel: {missing}")
            else:
                candidates = [c for c in feature_cols if c not in FEATURE_COLS]
            if not candidates:
                raise SystemExit("--feature-diagnostics needs candidate features; "
                                 "add a pack (e.g. --with-microstructure) or name them "
                                 "with --diagnose-features")
            block_size = args.block_size or max(
                1, math.ceil(HORIZON_TRADING_DAYS[args.horizon] / 21)
            )
            # Vetting a candidate is a SELECTION decision, so it has to respect the
            # frozen holdout exactly like the fold list does. This branch returns
            # before walk_forward_ic ever runs, so the window is applied here.
            diag_panel = panel
            if args.max_test_date is not None or args.min_test_date is not None:
                keep = np.ones(len(diag_panel), dtype=bool)
                dates = diag_panel["date"].to_numpy()
                if args.max_test_date is not None:
                    keep &= np.asarray([d <= args.max_test_date for d in dates])
                if args.min_test_date is not None:
                    keep &= np.asarray([d >= args.min_test_date for d in dates])
                diag_panel = diag_panel[keep]
                print(f"diagnostics date window: {len(diag_panel):,} of {len(panel):,} rows")
            print(f"\n[feature diagnostics] {args.horizon}: decorrelation vs the book + "
                  f"standalone within-{args.neutralize_by} IC (block={block_size}); "
                  f"consolidation control = efficiency_ratio_120d:")
            print(f"  {'feature':>24} {'|corr|bk':>9} {'|corr|ER':>9} {'sec_ic':>8} "
                  f"{'sec_t':>7} {'sec_p':>8} {'ic_side':>8} {'ic_mid':>8} {'ic_trend':>8} "
                  f"  top correlates")
            for row in feature_diagnostics(
                diag_panel, args.horizon, candidates,
                sector_group_col=args.neutralize_by,
                min_names=args.min_names,
                block_size=block_size,
                reps=args.block_bootstrap_reps,
                seed=args.seed,
            ):
                top = ", ".join(f"{n}={v:.2f}" for n, v in row["top_corr"])
                print(f"  {row['feature']:>24} {row['mean_abs_corr']:>9.3f} "
                      f"{row['er_corr']:>9.3f} {row['sec_ic']:>+8.4f} "
                      f"{row['sec_t_block']:>+7.2f} {row['sec_p']:>8.4f} "
                      f"{row['ic_sideways']:>+8.4f} {row['ic_mid']:>+8.4f} "
                      f"{row['ic_trending']:>+8.4f}   {top}")
            return

        knife_grid = (
            [float(x) for x in args.knife_sweep.split(",")] if args.knife_sweep else None
        )
        blend_grid = (
            [float(x) for x in args.target_blend_sweep.split(",")]
            if args.target_blend_sweep else None
        )
        real = walk_forward_ic(panel, args.horizon, lgb_cfg, wf_cfg,
                               seed=args.seed, shuffle=False, target_mode=args.target,
                               log=print if args.verbose else (lambda *_: None),
                               feature_cols=feature_cols,
                               n_seeds=args.n_seeds,
                               compute_sector_ic=args.sector_neutral_ic,
                               sector_group_col=args.neutralize_by,
                               linear_blend=args.with_linear_blend,
                               ridge_alpha=args.ridge_alpha,
                               smooth_span=args.smooth_span,
                               knife_lambda=args.knife_lambda,
                               winsorize_pct=args.winsorize_target,
                               max_test_date=args.max_test_date,
                               min_test_date=args.min_test_date,
                               vol_gate=args.vol_gate,
                               return_records=(knife_grid is not None
                                               or blend_grid is not None
                                               or args.size_neutral_ic))
        block_size = args.block_size or max(
            1, math.ceil(HORIZON_TRADING_DAYS[args.horizon] / 21)
        )
        smooth_tag = f", smooth_span={args.smooth_span}" if args.smooth_span > 0 else ""
        knife_tag = f", knife_lambda={args.knife_lambda}" if args.knife_lambda > 0 else ""
        wins_tag = f", winsorize={args.winsorize_target}" if args.winsorize_target > 0 else ""
        obj_tag = f", objective={args.objective}" if args.objective != "regression" else ""
        print(f"\n--- {args.horizon} cross-sectional rank-IC "
              f"(expanding walk-forward, target={args.target}{obj_tag}{smooth_tag}{knife_tag}{wins_tag}) ---")
        if knife_grid is not None and real.get("records"):
            print(f"\n[knife sweep] vol×downtrend output overlay (no refit, "
                  f"smooth_span={args.smooth_span}); SECB={args.sector_neutral_ic}:")
            print(f"  {'lam':>5} {'mean_ic':>8} {'sec_ic':>8} {'turnover':>9} "
                  f"{'td_knife':>9} {'td_vol_p':>9} {'td_down':>8} {'td_meanr':>9} "
                  f"{'td_dnside':>10} {'td_tail':>8}")
            for row in knife_sweep_table(
                real["records"], knife_grid, smooth_span=args.smooth_span,
                sector_group_col=args.neutralize_by,
                compute_sector_ic=args.sector_neutral_ic,
            ):
                print(f"  {row['lam']:>5.2f} {row['mean_ic']:>+8.4f} "
                      f"{row['sec_ic']:>+8.4f} {row['turnover']:>9.4f} "
                      f"{row['knife']:>9.4f} {row['vol_p']:>9.4f} "
                      f"{row['downtrend']:>8.3f} {row['mean_r']:>+9.4f} "
                      f"{row['downside']:>10.4f} {row['tail_frac']:>8.3f}")
        if blend_grid is not None and args.target_blend and real.get("records"):
            print(f"\n[target-blend sweep] rank-ensemble {args.target} × "
                  f"{args.target_blend} (no refit, second {args.n_seeds}-seed fit on "
                  f"alt target); SECB={args.sector_neutral_ic}, block={block_size}:")
            alt = walk_forward_ic(panel, args.horizon, lgb_cfg, wf_cfg,
                                  seed=args.seed, shuffle=False,
                                  target_mode=args.target_blend,
                                  feature_cols=feature_cols, n_seeds=args.n_seeds,
                                  compute_sector_ic=args.sector_neutral_ic,
                                  sector_group_col=args.neutralize_by,
                                  max_test_date=args.max_test_date,
                                  min_test_date=args.min_test_date,
                                  return_records=True)
            print(f"  {'w':>5} {'mean_ic':>8} {'sec_ic':>8} {'sec_t':>7} "
                  f"{'sec_p':>8} {'turnover':>9}")
            for row in target_blend_sweep(
                real["records"], alt["records"], blend_grid,
                block_size=block_size, reps=args.block_bootstrap_reps, seed=args.seed,
                sector_group_col=args.neutralize_by,
                compute_sector_ic=args.sector_neutral_ic,
            ):
                print(f"  {row['w']:>5.2f} {row['mean_ic']:>+8.4f} "
                      f"{row['sec_ic']:>+8.4f} {row['sec_t_block']:>+7.2f} "
                      f"{row['sec_p']:>8.4f} {row['turnover']:>9.4f}")
        elif blend_grid is not None and not args.target_blend:
            print("\n[target-blend sweep] skipped: --target-blend-sweep requires "
                  "--target-blend <alt target>")
        if args.reg_sweep:
            base = LGBMConfig()
            reg_configs = [
                ("baseline", base),
                ("slow_shrink", replace(base, learning_rate=0.015, n_estimators=600)),
                ("small_trees", replace(base, num_leaves=7, max_depth=3,
                                        min_child_samples=100)),
                ("decorrelate", replace(base, colsample_bytree=0.5, subsample=0.7)),
                ("strong_l1l2", replace(base, reg_lambda=5.0, reg_alpha=1.0,
                                        min_child_samples=100)),
                ("conservative", replace(base, learning_rate=0.02, n_estimators=500,
                                         num_leaves=7, max_depth=3, min_child_samples=100,
                                         reg_lambda=5.0, reg_alpha=1.0,
                                         colsample_bytree=0.6, subsample=0.7)),
            ]
            print(f"\n[reg sweep] per-horizon LightGBM regularization at {args.horizon} "
                  f"(full refit each, {args.n_seeds}-seed, target={args.target}); "
                  f"block={block_size}:")
            print(f"  {'config':>13} {'mean_ic':>8} {'sec_ic':>8} {'sec_icir':>9} "
                  f"{'sec_t':>7} {'sec_p':>8} {'turnover':>9}")
            for row in regularization_sweep(
                panel, args.horizon, reg_configs, wf_cfg=wf_cfg,
                target_mode=args.target, feature_cols=feature_cols,
                n_seeds=args.n_seeds, block_size=block_size,
                reps=args.block_bootstrap_reps, seed=args.seed,
                sector_group_col=args.neutralize_by,
                compute_sector_ic=args.sector_neutral_ic,
                max_test_date=args.max_test_date,
                min_test_date=args.min_test_date,
                log=print if args.verbose else (lambda *_: None),
            ):
                print(f"  {row['name']:>13} {row['mean_ic']:>+8.4f} "
                      f"{row['sec_ic']:>+8.4f} {row['sec_icir']:>+9.3f} "
                      f"{row['sec_t_block']:>+7.2f} {row['sec_p']:>8.4f} "
                      f"{row['turnover']:>9.4f}")
        if real.get("turnover_raw") is not None:
            tr = real["turnover_raw"]
            line = f"[turnover] mean |Δ rank| consecutive dates: raw={tr:.4f}"
            if real.get("turnover_smoothed") is not None:
                ts = real["turnover_smoothed"]
                post_label = "knife" if args.knife_lambda > 0 and args.smooth_span == 0 else (
                    "knife+smooth" if args.knife_lambda > 0 else "smoothed")
                line += f"  {post_label}={ts:.4f}  ({(1 - ts / tr) * 100:+.0f}% vs raw)"
            print(line)

        # HEADLINE: within-sector stock selection — the bar a horizon must clear.
        sec_boot = None
        if args.sector_neutral_ic and "sector_summary" in real:
            print(f"[HEADLINE] within-{args.neutralize_by} stock selection (the success bar):")
            _print_summary("SEC ", real["sector_summary"])
            if args.block_bootstrap_reps > 0 and "sector_ic_values" in real:
                sec_boot = block_bootstrap_summary(
                    real["sector_ic_values"],
                    block_size=block_size,
                    reps=args.block_bootstrap_reps,
                    seed=args.seed,
                )
                _print_bootstrap("SECB", sec_boot)
                _print_power("SECB", sec_boot)

        # SIZE-NEUTRAL: the size premium is a known risk premium, not alpha. This is
        # what survives once size is held constant within each cross-section.
        if args.size_neutral_ic:
            if not real.get("records"):
                print("[size-neutral] no fold records available")
            else:
                sn = size_neutral_summary(
                    real["records"], block_size=block_size,
                    reps=args.block_bootstrap_reps or 2000,
                )
                print("\n[SIZE-NEUTRAL] partial Spearman IC holding log_market_cap constant:")
                _print_summary("SN-U", sn["universe"])
                _print_bootstrap("SN-U", sn["universe"])
                if sn["sector"]["n_folds"]:
                    _print_summary("SN-S", sn["sector"])
                    _print_bootstrap("SN-S", sn["sector"])
                    _print_power("SN-S", sn["sector"])
                print("  (SN-S is the honest read: within-sector selection net of the "
                      "size tilt.)")

        # REGIME: a pooled mean hides regime dependence, which a size-tilted signal
        # is especially prone to.
        if args.vol_gate:
            n_f = max(1, len(real["folds"]))
            print(f"[vol-gate] fired on {real['n_vol_gated']}/{len(real['folds'])} fold(s) "
                  f"({real['n_vol_gated'] / n_f:.0%})"
                  + ("" if args.smooth_span else
                     "  -- NOTE: smooth_span=0, so this is inert by construction"))
        if args.regime_report:
            rep = regime_report(real["folds"])
            print("\n[REGIME] per-calendar-year:")
            print(f"  {'year':<6} {'n':>4} {'mean_ic':>9} {'sector_ic':>10} {'hit':>6}")
            for year, agg in rep["by_year"].items():
                print(f"  {year:<6} {agg['n']:>4} {agg['mean_ic']:>9.4f} "
                      f"{agg['mean_sector_ic']:>10.4f} {agg['hit_rate']:>6.2f}")
            if rep["by_vol_regime"]:
                print("  by realized-vol regime (IN-SAMPLE tertiles of the cross-sectional "
                      "median — descriptive only):")
                for name, agg in rep["by_vol_regime"].items():
                    print(f"  {name:<9} {agg['n']:>4} {agg['mean_ic']:>9.4f} "
                          f"{agg['mean_sector_ic']:>10.4f} {agg['hit_rate']:>6.2f}")
            if rep["by_vol_regime_pit"]:
                print(f"  by realized-vol regime (PIT expanding tertiles, burn-in "
                      f"{rep['pit_burn_in']}, {rep['pit_n_scored']} folds scored — "
                      f"this is the one that counts):")
                for name, agg in rep["by_vol_regime_pit"].items():
                    print(f"  {name:<9} {agg['n']:>4} {agg['mean_ic']:>9.4f} "
                          f"{agg['mean_sector_ic']:>10.4f} {agg['hit_rate']:>6.2f}")

        # DIAGNOSTIC: universe IC includes sector rotation; high here + flat SECB
        # ⇒ the edge is sector timing, not stock selection (does NOT clear the bar).
        print("[diagnostic] universe IC (incl. sector rotation):")
        _print_summary("REAL", real["summary"])
        if args.block_bootstrap_reps > 0:
            boot = block_bootstrap_summary(
                [r["ic"] for r in real["folds"]],
                block_size=block_size,
                reps=args.block_bootstrap_reps,
                seed=args.seed,
            )
            _print_bootstrap("BOOT", boot)

        if args.null_reps > 0:
            print(f"\nrunning {args.null_reps} shuffle-null reps ...")
            null_means, null_sec_means = [], []
            for rep in range(args.null_reps):
                res = walk_forward_ic(panel, args.horizon, lgb_cfg, wf_cfg,
                                      seed=args.seed + 1000 * (rep + 1), shuffle=True,
                                      target_mode=args.target, feature_cols=feature_cols,
                                      # Must match the real run: a null built from a
                                      # 1-seed estimator has a wider sampling
                                      # distribution than an 8-seed real statistic,
                                      # so the z-band would compare unlike things.
                                      n_seeds=args.n_seeds,
                                      compute_sector_ic=args.sector_neutral_ic,
                                      sector_group_col=args.neutralize_by,
                                      linear_blend=args.with_linear_blend,
                                      ridge_alpha=args.ridge_alpha,
                                      smooth_span=args.smooth_span,
                                      knife_lambda=args.knife_lambda,
                                      winsorize_pct=args.winsorize_target,
                                      max_test_date=args.max_test_date,
                                      min_test_date=args.min_test_date,
                                      vol_gate=args.vol_gate)
                m = res["summary"]["mean_ic"]
                null_means.append(m)
                if args.sector_neutral_ic and "sector_summary" in res:
                    null_sec_means.append(res["sector_summary"]["mean_ic"])
                print(f"  null rep {rep}: mean_ic={m:+.4f}")

            def _verdict(tag: str, real_mean: float, nulls: list[float]) -> None:
                nm = np.asarray(nulls, dtype=float)
                mu = float(nm.mean())
                sd = float(nm.std(ddof=1)) if nm.size > 1 else 0.0
                z = (real_mean - mu) / sd if sd > 0 else float("nan")
                print(f"{tag}  reps={nm.size}  real {real_mean:+.4f} vs null "
                      f"{mu:+.4f}±{sd:.4f}  =>  z={z:+.2f}")

            print()
            # The success-bar verdict is on within-sector IC; universe is secondary.
            if null_sec_means:
                _verdict("VERDICT(SECB)", real["sector_summary"]["mean_ic"], null_sec_means)
            _verdict("verdict(univ)", real["summary"]["mean_ic"], null_means)


def _iso_date(raw: str) -> date:
    """argparse type for YYYY-MM-DD; panel grid dates are datetime.date."""
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {raw!r}") from exc


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-sectional LightGBM walk-forward baseline")
    p.add_argument(
        "--horizon", default="1M", choices=list(HORIZONS),
        help="forward horizon to predict/score (1M is the only one with enough OOS dates)",
    )
    p.add_argument("--min-train-months", type=int, default=36)
    p.add_argument("--max-train-months", type=int, default=None,
                   help="rolling training-window length in monthly grid dates. Default "
                        "(omitted) = the horizon's PRODUCTION_HORIZON_SPECS window, so a "
                        "re-validation measures the deployed model (6M ships rolling-60). "
                        "Pass 0 to force an expanding window.")
    p.add_argument("--min-names", type=int, default=30, help="skip thinner test cross-sections")
    p.add_argument("--max-test-date", type=_iso_date, default=None, metavar="YYYY-MM-DD",
                   help="score only folds whose TEST date is on or before this. Frozen-holdout "
                        "discipline: research/selection runs must stop far enough back that every "
                        "label is realized before the holdout opens (holdout 2024-01 => 3M "
                        "2023-09-30, 6M 2023-06-30, 1Y 2022-12-31). Training is unaffected.")
    p.add_argument("--min-test-date", type=_iso_date, default=None, metavar="YYYY-MM-DD",
                   help="score only folds whose TEST date is on or after this. Used ONCE per "
                        "finalized config for the frozen holdout run (--min-test-date 2024-01-01); "
                        "that run is a sign/sanity check, not a significance test.")
    p.add_argument("--target", default="return",
                   choices=["return", "rank", "quantile", "sector_return",
                            "sector_return_vol", "beta_resid", "beta_sector_resid",
                            "sector_grade"],
                   help="training target transform (scoring is always vs realized "
                        "universe-demeaned return; sector_return / beta_resid / "
                        "beta_sector_resid are alpha-residual modes; "
                        "sector_return_vol divides sector_return by raw vol_120d "
                        "floored at the per-date 20th pct — homoskedasticizes label "
                        "noise and shrinks high-vol labels, goal A + B lever; "
                        "sector_grade = per-date qcut of sector_return into ordinal "
                        "grades for a --objective lambdarank fit)")
    p.add_argument(
        "--n-buckets", type=int, default=5,
        help="equal-count buckets for --target quantile",
    )
    p.add_argument(
        "--null-reps", type=int, default=0,
        help="shuffle-null repetitions for the no-signal band",
    )
    p.add_argument("--block-bootstrap-reps", type=int, default=2000,
                   help="moving-block bootstrap reps for overlap-aware IC significance "
                        "(default 2000; set 0 to skip)")
    p.add_argument("--block-size", type=int, default=None,
                   help="block length in monthly folds (default = horizon months)")
    p.add_argument("--sector-neutral-ic", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="report within-sector IC averaged across GICS groups as the "
                        "HEADLINE success metric (stock selection, not sector "
                        "rotation). On by default; pass --no-sector-neutral-ic to skip")
    p.add_argument("--neutralize-by", default="sector", choices=["sector", "industry"],
                   help="grouping for the within-group IC headline (industry is the "
                        "finer, stricter stock-selection cut)")
    p.add_argument("--n-seeds", type=int, default=1,
                   help="seed-ensemble size for walk-forward (default 1; use 8 for "
                        "production-equivalent smoothed IC estimate)")
    p.add_argument("--smooth-span", type=int, default=0, metavar="N",
                   help="EWMA span for cross-date prediction smoothing (0 = off). "
                        "Causally smooths each name's percentile rank over scoring "
                        "dates before scoring, then reports turnover raw vs smoothed. "
                        "Reduces drift/turnover; longer spans suit longer horizons.")
    p.add_argument("--knife-lambda", type=float, default=0.0, metavar="L",
                   help="falling-knife output-overlay weight (0 = off). Re-ranks each "
                        "cross-section toward names that are NOT both high-vol and "
                        "downtrending; composes ahead of --smooth-span. Suppresses "
                        "'falling knife' picks at the top and lowers turnover.")
    p.add_argument("--objective", default="regression",
                   choices=["regression", "lambdarank", "rank_xendcg"],
                   help="LightGBM training objective. 'regression' = pointwise L2 "
                        "(default). 'lambdarank'/'rank_xendcg' switch to LGBMRanker "
                        "(learning-to-rank, aligned with rank-IC scoring); require "
                        "--target sector_grade (ordinal grades + per-date query group)")
    p.add_argument("--rank-grades", type=int, default=5, metavar="K",
                   help="number of ordinal grades for --target sector_grade (per-date "
                        "qcut buckets); default 5 matches the quantile-bucket convention")
    p.add_argument("--lambdarank-truncation", type=int, default=500, metavar="N",
                   help="LambdaRank list-truncation level (pairs beyond rank N are "
                        "ignored in the gradient); default 500 >= cross-section size "
                        "for full-list ranking. Lower focuses the loss on the top.")
    p.add_argument("--subsample", type=float, default=None, metavar="F",
                   help="row-bagging fraction override (default: LGBMConfig 0.8). "
                        "Used to smoke the bagging x pairwise-gradient interaction "
                        "under ranking objectives (0.8 vs 1.0).")
    p.add_argument("--winsorize-target", type=float, default=0.0, metavar="PCT",
                   help="per-date winsorization of the TRAINING label (0 = off). "
                        "Clips each date's cross-section of the return-like target to "
                        "its [PCT, 1−PCT] quantiles before fitting; scoring stays on "
                        "unclipped realized return. Only valid for return-like targets "
                        "(return/sector_return/beta_resid*). E.g. 0.01 clips the top/"
                        "bottom 1%% per date.")
    p.add_argument("--knife-sweep", default=None, metavar="L1,L2,...",
                   help="comma-separated knife-lambda grid for a no-refit sweep table "
                        "(e.g. '0,0.05,0.1,0.2'); reports mean/within-sector IC, "
                        "turnover, and top-decile risk per lambda. Composes with "
                        "--smooth-span.")
    p.add_argument("--target-blend", default=None,
                   choices=["return", "rank", "quantile", "sector_return",
                            "sector_return_vol", "beta_resid", "beta_sector_resid"],
                   help="alternate training target to rank-ensemble with --target: "
                        "fits a SECOND walk-forward on this target and averages the "
                        "per-date percentile ranks (decorrelated label errors → lower "
                        "cross-date IC variance). Pairs with --target-blend-sweep.")
    p.add_argument("--target-blend-sweep", default=None, metavar="W1,W2,...",
                   help="comma-separated alt-target blend-weight grid for a no-refit "
                        "sweep (e.g. '0,0.25,0.5,0.75,1.0'); 0 = pure --target, "
                        "1 = pure --target-blend. Requires --target-blend.")
    p.add_argument("--reg-sweep", action="store_true",
                   help="compare a curated grid of LightGBM regularization configs at "
                        "this horizon (FULL refit each): baseline vs slower-shrink, "
                        "smaller-trees, feature/row-decorrelation, stronger L1/L2, and "
                        "a conservative combo. Reports within-sector IC/ICIR + block t/p "
                        "and turnover to judge variance reduction.")
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
    p.add_argument("--with-eps-surprise", action="store_true",
                   help="add LSEG EPS-surprise (PEAD) pack — computed from quarterly "
                        "earnings_surprises; needs migration 005 + estimate backfill")
    p.add_argument("--with-revision-momentum", action="store_true",
                   help="add earnings-revision momentum pack (forward-EPS estimate "
                        "revisions + coverage/PT-estimate counts); 3M-focused")
    p.add_argument("--with-forward-valuation", action="store_true",
                   help="add LSEG forward-valuation pack (forward earnings/ebitda yield, "
                        "price-target upside)")
    p.add_argument("--with-lottery", action="store_true",
                   help="add lottery / idiosyncratic-vol pack (max_ret_21d, idio_vol); "
                        "the volatility variants with the documented NEGATIVE sign")
    p.add_argument("--with-microstructure", action="store_true",
                   help="add microstructure / higher-moment pack (ret_skew_120d, "
                        "downside_vol_ratio_120d, amihud_illiq_60d, turnover_60d, "
                        "efficiency_ratio_120d) — decorrelated 6M candidate (price/volume)")
    p.add_argument("--with-eps-dispersion", action="store_true",
                   help="add EPS estimate dispersion pack (eps_dispersion = "
                        "eps_std_dev / |eps_mean|) — Diether-Malloy-Scherbina 2002 "
                        "short-selling-constraint proxy; expected NEGATIVE loading; "
                        "requires analyst_estimates.eps_std_dev populated via "
                        "backfill_estimates.py after migration 008")
    p.add_argument("--with-short-interest", action="store_true",
                   help="add FINRA short interest pack (short_ratio = days-to-cover "
                        "from bimonthly Reg SHO files); requires migration 009 + "
                        "backfill_short_interest.py")
    p.add_argument("--with-knife-feature", action="store_true",
                   help="add knife_score as a training feature (vol_p × downtrend_p "
                        "in [0,1], computed from rank-normalized vol_120d/ma_gap_200/"
                        "dist_low_252); lets the GBDT learn conditional demotion of "
                        "high-vol-AND-falling names instead of the overlay's "
                        "unconditional rank penalty — lever 2 for 6M falling-knife "
                        "reduction")
    p.add_argument("--feature-diagnostics", action="store_true",
                   help="vet the opted-in pack features (--with-*) for decorrelation vs "
                        "the book + standalone within-sector IC + a consolidation "
                        "(efficiency-ratio tertile) breakdown, then exit — no model fits")
    p.add_argument("--with-insider", action="store_true",
                   help="add insider-transaction pack (Cohen-Malloy-Pomorski: "
                        "insider_net_buy_6m / insider_buyers_90d / insider_net_ratio_12m). "
                        "PIT-safe on filing_date; requires insider_transactions "
                        "(migration 010) + backfill_insiders.py.")
    p.add_argument("--with-seasonality", action="store_true",
                   help="add cross-sectional seasonality pack (Heston-Sadka same-"
                        "calendar-month return persistence: seasonal_same/other/gap_5y). "
                        "Expect signal at 3M; 1Y is a wash (forward window spans all "
                        "12 months). PIT-safe (completed months strictly before the bar).")
    p.add_argument("--diagnose-features", default=None, metavar="c1,c2,...",
                   help="run --feature-diagnostics on these exact columns instead of "
                        "the opted-in packs. Needed to vet a feature that is already "
                        "in the book, e.g. --diagnose-features fund_available,log_market_cap.")
    p.add_argument("--size-neutral-ic", action="store_true",
                   help="also report IC with log_market_cap partialled out (partial "
                        "Spearman, post-hoc on the same folds — no refits). The size "
                        "premium is a documented RISK premium; what survives this is "
                        "the part not explained by a size tilt.")
    p.add_argument("--vol-gate", action="store_true",
                   help="shrink ranks toward 0.5 on PIT high-vol dates (expanding 80th "
                        "pctile of the cross-sectional median raw vol, 24-fold burn-in). "
                        "Composes BEFORE smoothing; provably inert without it, so 6M "
                        "(smooth_span=0) is unaffected by construction.")
    p.add_argument("--regime-report", action="store_true",
                   help="break per-fold IC out by calendar year and by realized-vol "
                        "tertile. A pooled mean hides regime dependence.")
    p.add_argument("--membership-filter", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="restrict each cross-section to point-in-time index members "
                        "(index_membership, migration 012). ON by default: without it "
                        "a name promoted into the index in 2023 still appears in the "
                        "2017 cross-sections, which selects on future index promotion. "
                        "Use --no-membership-filter to measure that bias.")
    p.add_argument("--with-sentiment", action="store_true",
                   help="add the FinBERT rolling news sentiment pack (sentiment_7d/"
                        "sentiment_14d). OFF by default: yfinance serves only ~30 days "
                        "of headlines, so the columns are ~98%% zero across the panel "
                        "and cannot be validated. Turn on once a headline archive is "
                        "backfilled.")
    p.add_argument("--with-estimate-missing", action="store_true",
                   help="add the LSEG availability pack (est_available, "
                        "est_staleness_days) — the analyst-feed analogue of "
                        "fund_available. Opt-in: analyst_estimates starts in 2012-13, "
                        "so these flag the rows where the promoted estimate packs are "
                        "NaN rather than observed.")
    p.add_argument("--extra-features", default=None, metavar="c1,c2,...",
                   help="append these ad-hoc feature columns to the active list "
                        "(isolate one member of a pack for a targeted ablation, e.g. "
                        "--extra-features sales_to_price). Columns must be produced by "
                        "build_ticker_rows.")
    p.add_argument("--industry-relative", action="store_true",
                   help="rank-normalize price/fundamental/valuation/quality "
                        "features within (date, industry) instead of universe-wide")
    p.add_argument("--with-linear-blend", type=float, default=0.0, metavar="W",
                   help="blend a ridge cross-sectional model with the GBDT at this "
                        "weight (0..1, ridge share); 0 = pure GBDT. Blends on ranks "
                        "(Gu-Kelly-Xiu: ensembles dominate at low SNR)")
    p.add_argument("--ridge-alpha", type=float, default=10.0,
                   help="L2 strength for the --with-linear-blend ridge model")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
