# Cardinal / Stock Trend Predictor — Quant Interview Deep-Dive

**Audience:** you, prepping to defend this project in quant research / quant trading interviews.
**Method:** written from the code (`backend/ml/`, `backend/ingestion/`, `backend/db/`), not the README, plus
live measurements run against the cached data panel (`.frame_cache/frames_all.pkl`, pulled 2026-07-31).
Every number below that is labeled **[measured]** was computed during this audit and is reproducible.

> **Read section 0 first.** The audit turned up two things that change what you should claim.
> Neither is fatal, but both are exactly what a good interviewer probes for, and you want to be
> the one who raises them.

---

## 0. Bottom line up front — read before you write a resume bullet

### 0.1 The universe IC is materially inflated by a data-availability artifact

`fund_available` is a binary feature: "does this ticker have at least one SEC filing on or before this date?"
It exists because 121 of the 638 price-history tickers — the *removed-from-index* cohort seeded by
`scripts/seed_sp500_historical.py` — have **zero** EDGAR fundamentals, **zero** shares outstanding, and
**no GICS sector**. That cohort is defined by "was dropped from the S&P 500 at some point after 2016."

So `fund_available == 0` is, in practice, a flag for *this name will be removed from the index*, known
from day one of the panel. That is future information encoded as a missingness pattern.

**[measured]** Standalone cross-sectional Spearman IC of `fund_available` alone against the realized
demeaned forward return (per-date IC, moving-block bootstrap, 1000 reps):

| Horizon | universe IC | t_block | p | within-sector IC | t_block | p |
|---|---|---|---|---|---|---|
| 3M | **+0.0525** | +3.97 | 0.001 | −0.0112 | −0.66 | 0.46 |
| 6M | **+0.0722** | +3.57 | 0.001 | −0.0133 | −0.54 | 0.51 |
| 1Y | **+0.0899** | +2.75 | 0.002 | −0.0143 | −0.46 | 0.47 |

The single availability flag has almost exactly the universe IC that the *whole 6M model* reports
(+0.078). **Do not put "universe IC +0.078" on the resume without a caveat.**

**The saving grace, and it's a genuinely good one:** the within-sector IC (SECB) is *immune* to this,
because `within_sector_ic()` drops rows with a null sector — which is precisely the leaky cohort.
That's why the headline metric being SECB is the right call, and you can say so.

**[measured] Ablation**, 6M, single seed, L2 regression on `sector_return`, rolling-60 window,
production feature pack:

| Variant | universe IC (t) | within-sector IC (t, p) |
|---|---|---|
| production-equivalent | +0.0743 (3.14) | +0.0483 (2.29, p=0.003) |
| − `fund_available` | +0.0674 (2.73) | +0.0454 (2.05, p=0.006) |
| − `fund_available` − all 5 EDGAR fundamentals | +0.0487 (2.26) | +0.0451 (2.27, p=0.001) |

The entire fundamentals block is worth **34% of universe IC and ~7% of within-sector IC**. That is the
signature of a block that is mostly sorting survivors from non-survivors, not picking stocks.

### 0.2 The within-sector signal is load-bearing on the size factor — and that factor has a construction bug

**[measured]** Standalone within-sector IC of `log_market_cap` alone: −0.037 (3M, t=−2.61), −0.049
(6M, t=−2.36), −0.073 (1Y, t=−2.57). That is the *strongest single standalone within-sector signal in
the entire book* — larger in magnitude at 1Y than the full model's SECB.

**[measured] Drop-one ablation** (single seed, L2 regression on `sector_return`, production overlays on):

| Horizon | config | within-sector IC | t_block | p |
|---|---|---|---|---|
| 3M | production-equivalent | +0.0300 | +2.01 | 0.016 |
| 3M | − `log_market_cap` | +0.0205 | +1.42 | 0.070 |
| 3M | − `log_market_cap` − `fund_available` | +0.0179 | +1.15 | 0.173 |
| 6M | production-equivalent | +0.0483 | +2.29 | 0.003 |
| 6M | − `log_market_cap` | +0.0203 | +0.88 | 0.243 |
| 6M | − `log_market_cap` − `fund_available` | +0.0143 | +0.59 | 0.417 |
| 1Y | production-equivalent | +0.0367 | +1.91 | 0.011 |
| 1Y | − `log_market_cap` | **−0.0131** | −0.45 | 0.550 |
| 1Y | − `log_market_cap` − `fund_available` | −0.0060 | −0.19 | 0.811 |

**Remove size and nothing is statistically significant at any horizon.** At 1Y the sign flips.

Two consequences:

1. **What you have is a small-cap-within-sector tilt plus a modest technical overlay.** The size
   effect is a documented *risk premium*, not alpha. Saying "my model finds alpha" is overselling;
   saying "my model recovers a known within-sector size effect out-of-sample, and I can show the
   technical book adds a smaller independent piece" is honest and still impressive.
2. **The size feature is constructed wrong** — see §4.3. `log_market_cap = log(adj_close_t × shares_now)`.
   `adj_close_t` is yfinance's *back-adjusted* close, i.e. `close_t × Π(dividend/split factors AFTER t)`.
   That product is future information. And `shares_now` is today's share count applied to 2012.
   The raw `close` column is already stored in `price_history` (`auto_adjust=False`), so the fix is a
   one-line change to the SQL in `dataset.py` plus a shares-outstanding time series.

**Caveat on the ablations:** these were run single-seed with the L2 regression objective, not the
promoted 8-seed LambdaRank config (which is ~14× slower per fit). Absolute levels are therefore lower
than the documented headline numbers; the *ratios* are what the ablation establishes, and a ranking
objective is not going to manufacture within-sector signal that a regressor could not find at all.

### 0.3 What is genuinely strong here

Do not let §0.1–0.2 make you underclaim. The following are real, verified in code, and above the bar
for most candidate projects:

- A correct **purged expanding-window walk-forward** where the purge length equals the label horizon
  (`walk_forward_folds`) — no training label window can overlap a test prediction window.
- **Overlapping-label-aware significance**: a moving-block bootstrap with block length = horizon in
  months, a centered-null two-sided p-value, and an explicit **minimum-detectable-IC** power number.
  Most candidates report a naive t-stat inflated 2–3× by label overlap. You explicitly do not.
- **Genuinely point-in-time fundamental and estimate joins**, keyed on SEC receipt date (`filed_at`)
  and LSEG observation date (`as_of_date`), each field looked up independently, enforced by unit tests.
- **Objective/metric alignment**: recognizing that "fit L2, score Spearman" is a misalignment and
  switching to `LGBMRanker` with per-date query groups and linear `label_gain`.
- **A pre-registered feature gate** (`feature_diagnostics`) that rejects candidates on decorrelation +
  standalone block-IC + a trend-vs-consolidation confound check *before* any promotion fit. Eight
  feature packs were killed by it. Reporting your kills is more credible than reporting your wins.
- **This audit itself.** "I found a survivorship artifact worth +0.07 universe IC in my own feature
  set and quantified it" is one of the best answers you can give to "how do you know this isn't overfit."

---

## 1. The 30-second summary (say this out loud)

> It's a cross-sectional equity ranking model on ~600 large-cap US names, monthly rebalance, three
> horizons — 3, 6 and 12 months. Features are ~21 point-in-time factors: momentum, realized vol,
> trend/52-week position, size, five EDGAR fundamentals joined on filing receipt date, and LSEG
> analyst estimate revisions and revenue surprise. The model is a shallow LightGBM cross-sectional
> ranker, one per horizon, eight-seed ensembled.
>
> I grade it on **within-sector rank information coefficient** — mean Spearman correlation between
> predicted rank and realized forward return, computed inside each GICS sector and then averaged —
> because plain universe IC can be earned by sector rotation rather than stock selection. Out-of-sample
> that's about **+0.05 at 6M and +0.04 at 1Y**, over 126–145 monthly walk-forward folds from 2013 to 2026.
>
> The validation is the part I'd want to talk about. It's an expanding-window walk-forward with the
> training set purged by a full label horizon, so no training label window overlaps a test window.
> And because monthly folds with six- and twelve-month labels are heavily autocorrelated, I don't use
> the naive t-stat — I moving-block bootstrap the fold ICs with block length equal to the horizon,
> which cuts 1Y from ~126 folds to about 10 effective independent blocks. That's what makes the number
> honest, and it's also why 1Y is power-limited: my minimum detectable IC there is ~0.03.
>
> I'll also tell you the two things I found when I audited it. First, a fundamentals-availability
> indicator was carrying a +0.07 universe IC by itself, because the de-survivorshipped names never got
> their fundamentals backfilled — so missingness was a proxy for future index removal. Within-sector IC
> is immune to that because those names have no sector label, which is part of why I grade on it.
> Second, if I drop the size feature, the within-sector signal loses significance at every horizon. So
> what I actually have is a size tilt plus a smaller technical component — a risk premium, not alpha.
> The next thing I'd build is a size neutralization and re-measure what's left.

**Framing note:** lead with the validation methodology and the self-audit, not the IC. An interviewer
hears "+0.078 IC" from every candidate and assumes it's contaminated. Hearing "here is the contamination
I found in my own number" flips the dynamic entirely.

---

## 2. Pipeline overview

```
yfinance (OHLCV)          SEC EDGAR companyfacts       LSEG Workspace           yfinance news
        │                          │                        │                        │
        ▼                          ▼                        ▼                        ▼
 ingestion/prices.py     ingestion/fundamentals.py   ingestion/estimates.py   ingestion/headlines.py
 incremental + drift     10-K/10-Q XBRL, TTM roll-up  monthly PIT snapshots    FinBERT on MPS
 re-pull; floor 2010     keyed on filed_at            + quarterly FQ0 grid     → rolling 7d/14d
        │                          │                        │                        │
        └──────────────────────────┴────────────────────────┴────────────────────────┘
                                          │  asyncpg upserts
                                          ▼
                    Supabase Postgres — price_history, fundamentals,
                    analyst_estimates, earnings_surprises, sentiment_daily
                                          │
                                          ▼   dataset.load_frames → TickerFrame per ticker
                    ml/factors/*  per-ticker point-in-time feature builders
                                          │
                                          ▼   gbm_baseline.prepare_panel
                    ┌──────────────────────────────────────────────────┐
                    │ 1. assemble_panel     one row per (ticker, month-end) │
                    │ 2. demean_cross_sectional   r_h −= per-date median   │
                    │ 3. rank_normalize_features  per-date rank → [−1, 1]  │
                    │ 4. apply_target_modes       sector_return, sector_grade… │
                    └──────────────────────────────────────────────────┘
                                          │
                  ┌───────────────────────┴────────────────────────┐
                  ▼                                                ▼
   RESEARCH: gbm_baseline.walk_forward_ic            PRODUCTION: gbm_inference.py
   expanding window, purge = H months                fit on all rows with date < as_of
   per-fold Spearman IC + within-sector IC           score today's cross-section
   moving-block bootstrap                            → percentile rank → predictions table
                  │                                                │
                  ▼                                                ▼
     promotion decision (manual, edit                 FastAPI → Next.js dashboard
     PRODUCTION_HORIZON_SPECS)                        (rank gauges, no order generation)
```

### 2.1 Universe and period — the actual numbers

**[measured] from the cached frame pull (2026-07-31):**

| Fact | Value |
|---|---|
| Rows in `tickers` | 729 |
| With any price history | 638 |
| With a GICS sector *and* EDGAR fundamentals | 517 / 511 |
| With prices but **no** sector, **no** fundamentals, **no** shares outstanding | **121** |
| Total price rows | 1,819,660 |
| Price date range | 2010-01-04 → 2026-07-31 |
| Monthly grid dates | **199** (2010-01-29 → 2026-07-31) |
| Panel rows after assembly | **79,457** × 114 columns |
| Labeled rows (3M / 6M / 1Y) | 77,020 / 75,072 / 71,320 |

**Cross-section width is not constant, and this matters.** 411 tickers have their first bar on exactly
**2016-01-04** (a 10-year yfinance pull boundary); 146 start on 2010-01-04.

| Year | names with bars | with sector | with fundamentals |
|---|---|---|---|
| 2010 | 152 | 150 | 146 |
| 2013 | 159 | 156 | 152 |
| 2016 | 581 | 480 | 475 |
| 2020 | 598 | 501 | 496 |
| 2026 | 625 | 517 | 511 |

**[measured]** Every one of the 146 deep-history (2010) names has a GICS sector — i.e. **all of them are
current S&P 500 constituents.** The 2010–2016 portion of the panel is a ~150-name, 100%-survivor
cross-section. If asked "over what period is this really evaluated," the honest answer is:
**folds run 2013→2026, but only the 2017+ folds see a realistic ~580-name universe.**

Good news: **[measured]** restricting scoring to folds from 2017-01 onward barely moves anything
(6M universe IC +0.0743 → +0.0795; SECB +0.0483 → +0.0485, t 2.29 → 2.21 on 108 folds). The thin early
panel is not propping up the result. Have that number ready.

### 2.2 The panel construction, precisely

`build_calendar_grid` (`dataset.py`) takes the **last trading day of each calendar month** across the
union of all tickers' price dates → 199 grid dates.

For each (ticker, grid date), `build_ticker_rows` (`factors/assembly.py`):
- finds `pos = bisect_right(trade_dates, g) - 1` — the ticker's last bar on or before the grid date;
- **skips the row if that bar is more than `max_stale_days=7` old** (this is what stops a delisted name
  from being carried forward forever — it's the survivorship mechanism, see §4.2);
- requires `pos >= 252` so momentum and the 52-week window have full lookback;
- computes all features from data at or before `pos`;
- computes the label with `compute_targets(adj_close, pos)`.

The row's `date` is set to the **grid date**, not the bar date, so every ticker lands in the same
monthly cross-section.

### 2.3 The label

```python
# dataset.compute_targets
j = end_idx + HORIZON_TRADING_DAYS[h]        # 21 / 63 / 126 / 252
r = log(adj_close[j] / adj_close[end_idx])   # masked if bar j doesn't exist
```

Then `demean_cross_sectional` subtracts the per-date cross-sectional **median** forward return, giving
`r_h` = universe-relative log return. This is the **scoring** target at every horizon.

Note: subtracting a per-date constant is a monotone within-date transform, so the demean does **not**
change Spearman IC at all. It matters for the *regression training* target, not for the metric.

### 2.4 Feature normalization

`rank_normalize_features` maps every feature to its **within-date** cross-sectional rank in [−1, 1]:

```python
r = panel.groupby("date")[c].rank(method="average")
n = panel.groupby("date")[c].transform("count")
out[c] = (r - 1) / max(n - 1, 1) * 2 - 1
```

This is point-in-time safe (same-date rows only), kills outliers, and makes the model scale-free.
It's the right choice and easy to defend.

### 2.5 Training targets (`apply_target_modes`)

All are per-date relabelings of the same future information — no additional leak beyond the label itself:

| mode | definition | used by |
|---|---|---|
| `return` | `r_h` raw | — |
| `rank` | within-date percentile of `r_h` | 1M (not scored) |
| `quantile` | within-date equal-count bucket | — |
| **`sector_return`** | `r_h` − within-(date, sector) median; falls back to universe-demean when the sector has < 5 names | **3M production** |
| **`sector_grade`** | per-date `qcut(sector_return, 5)` → ordinal 0–4 | **6M, 1Y production** (LambdaRank) |
| `sector_return_vol` | `sector_return` / vol, floored at per-date 20th pct | rejected |
| `beta_resid` | `r_h` − β₂₅₂ · market_r_h | superseded |
| `beta_sector_resid` | `beta_resid` sector-demeaned | superseded |

**Point worth making in an interview:** the target moved from universe-relative to *sector*-relative
deliberately, because universe IC can be earned by sector rotation. That reframing is what took 1Y from
failing (`beta_resid`, SECB p = 0.266) to passing (`sector_return`, p = 0.003). It's a good "I changed
what I was optimizing once I understood what I was measuring" story.

### 2.6 Model configuration

`LGBMConfig` — deliberately small: `n_estimators=300, learning_rate=0.03, num_leaves=15, max_depth=4,
min_child_samples=50, subsample=0.8 (freq 1), colsample_bytree=0.8, reg_lambda=1.0, n_jobs=1`.

`PRODUCTION_HORIZON_SPECS` (`gbm_baseline.py`) is the single source of truth:

| H | objective | target | extra features | train window | smoothing | knife λ |
|---|---|---|---|---|---|---|
| 3M | L2 regression | `sector_return` | + revision momentum (4) | expanding | EWMA span 3 | 0.20 |
| 6M | **LambdaRank** | `sector_grade` | + `revenue_surprise` | **rolling 60 months** | off | 0 |
| 1Y | **LambdaRank** | `sector_grade` | + `revenue_surprise` | expanding | EWMA span 4 | 0 |
| 1M | — | — | — | — | — | — (dead horizon, not scored) |

Production inference runs an **8-seed ensemble** per horizon; predictions are averaged, *then*
rank-transformed, then the knife overlay and cross-date smoothing are applied.

`n_jobs=1` is not a perf choice — the process also imports torch (via `dataset` → `model`), whose bundled
`libomp.dylib` is a second LLVM OpenMP runtime; LightGBM spawning its own thread team segfaults on macOS.
That's a fine detail to mention if someone asks about engineering constraints.

---

## 3. Feature engineering deep-dive

`FEATURE_COLS` = 13 price + 5 fundamental + 1 availability + 2 sentiment = **21 base features**
(`factors/constants.py`). Promoted per-horizon additions: +4 revision-momentum at 3M, +1 revenue
surprise at 6M/1Y.

### 3.1 Momentum (4)

`mom_1m`, `mom_3m`, `mom_6m` = `log(P_t / P_{t−21,63,126})`; `mom_12_1` = `log(P_{t−21} / P_{t−252})`.

*Intuition:* Jegadeesh–Titman cross-sectional momentum. `mom_12_1` skips the most recent month to avoid
the well-documented short-term reversal that contaminates raw 12-month momentum. `mom_1m` is
deliberately kept as a separate reversal proxy — the tree can use it with the opposite sign.

*Leakage:* clean. All indices `≤ pos`.

*Honest note:* **[measured]** `mom_12_1` standalone has **no significant signal** in this panel —
universe IC +0.027/+0.030/+0.028 (t = 1.17/0.89/0.62), within-sector ≈ 0 at every horizon. If asked
"is momentum working in your data," the answer is "not standalone, over 2013–2026, in large-cap US."
That's consistent with the well-documented post-2009 momentum drawdown and is a better answer than
pretending it works.

### 3.2 Volatility and trend (7)

`vol_20d/60d/120d` (std of daily log returns), `dist_high_252`, `dist_low_252` (log distance to
52-week extremes), `ma_gap_50`, `ma_gap_200` (log gap to moving averages), plus `vol_trend`
(log of 20-day mean volume over 120-day mean volume).

*Intuition:* the low-volatility anomaly (Ang et al.); 52-week-high proximity as an anchoring/underreaction
proxy (George–Hwang); MA gaps as trend-persistence. `vol_trend` is a crude attention/participation proxy.

*Leakage:* clean — `window = adj_close[pos-251 : pos+1]`, strictly backward.

*Note:* these three vol windows are highly collinear with each other. In a 15-leaf, depth-4 tree that's
tolerable but it does mean effective feature count is lower than 21.

### 3.3 Size — `log_market_cap` (1)

`log(adj_close[pos] × shares_outstanding)`.

*Intuition:* the size premium; also the single most important *control* in a cross-sectional model,
since almost every other factor loads on size.

*Leakage:* **⚠️ this one is contaminated on two counts.** See §4.3 and §0.2. It is also, empirically,
the load-bearing feature for the entire within-sector result.

### 3.4 Fundamentals (5) + availability flag (1)

`revenue_growth`, `gross_margin`, `operating_margin`, `debt_equity`, `fcf_revenue`, from SEC EDGAR
companyfacts (10-K + 10-Q), joined by `bisect_right(filed_ats, d) - 1` — the most recent filing whose
**`filed_at`** (SEC receipt timestamp) is ≤ the sample date. `fund_available` is 1 iff any such filing exists.

*Intuition:* profitability and quality (Novy-Marx: gross profitability is the cleanest quality signal);
leverage as distress/risk; revenue growth as fundamental momentum.

*Leakage:* the **`filed_at` join is genuinely correct and tested** — `test_features_no_lookahead.py`
asserts a filing dated after `sample_end` contributes nothing, a filing dated exactly on `sample_end`
is included, forward-fill holds the earlier filing's values between filings, and all-future filings
yield zeros. This is the strongest PIT guarantee in the codebase and you should say so.

**⚠️ But `fund_available` is a leak** — see §0.1. It flags future index removal via missingness.

*Residual concern:* EDGAR `companyfacts` serves currently-known values. A **restatement** replaces or
supplements the original figure. If the ingestion parser keys on the accession number of the original
filing but stores the restated value, the panel would show restated fundamentals at the original filing
date. `fundamentals` has PK `(ticker_id, accession_number)` and stores the original `filed_at`, so a
restatement filed later gets its *own* accession and its own (later) `filed_at` — which is correct.
Worth one sentence of acknowledgement: "I rely on EDGAR emitting restatements as separate accessions;
I haven't independently verified that for every issuer."

### 3.5 Sentiment (2) — **effectively dead in-sample**

`sentiment_7d`, `sentiment_14d`: rolling FinBERT (`ProsusAI/finbert`) scores over yfinance headlines,
run locally on MPS.

**[measured] This is a train/serve skew, not a signal.** yfinance news returns only ~30 days of
history, so there is no backfill. The cached panel has **11,797 sentiment rows, all between
2026-05-14 and 2026-08-01**. Only **2.1% of panel rows** have a non-zero sentiment feature. Standalone
IC is undefined (constant input) at every horizon.

So: 2 of the 21 base features are identically zero across ~98% of the training panel, and the model has
never had a chance to learn how to use them — yet they go live in production inference. Not harmful
(a constant column gets zero split gain), but it means the honest feature count is **19, not 21**, and
you should say "19 live factors plus two sentiment channels I built but that have no usable history"
rather than "21 features including NLP sentiment."

If an interviewer likes the NLP angle, the defensible framing is: "I built a FinBERT pipeline, it runs
daily on MPS and writes rolling scores; I *cannot* validate it because yfinance gives ~30 days of
headline history, so I've never claimed it as a validated factor. Backfilling it needs a paid news
archive."

### 3.6 Promoted estimate features (LSEG)

**3M — revision momentum (4):** `eps_est_rev_30d`, `eps_est_rev_90d` (% change in the forward EPS
consensus over 30/90 calendar days), `coverage_chg_90d` (Δ analyst count), `pt_num_estimates`
(price-target estimate count).

*Intuition:* analyst revision momentum (Chan–Jegadeesh–Lakonishok) — consensus revisions are sticky and
drift, so recent upward revisions predict continued upward revisions and price drift. Coverage change is
an attention/breadth proxy.

**6M / 1Y — `revenue_surprise` (1):** `(rev_actual − rev_consensus) / |rev_consensus|` from the quarterly
`earnings_surprises` table, carried forward from `report_date`.

*Intuition:* post-earnings-announcement drift. Revenue surprise is harder to manage than EPS surprise
(no buyback/accrual lever), which is the stated reason it survived where `eps_surprise` didn't.

*Leakage:* the PIT construction here is careful and worth describing. `_estimates_context_asof` looks
up **each field independently** with `bisect_right(dates, d) - 1` because LSEG fields land on different
dates. Surprises are anchored on **`report_date`** (announcement date), not `period_end`. The consensus
stored is the **pre-report `Period=FQ0`** value, which `scripts/_probe_lseg.py` verified matches the
last pre-report monthly snapshot rather than the post-announcement revision. That last check is the
single most impressive PIT detail in the project — it's exactly the mistake most people make with
surprise data, and you caught it with a targeted probe.

### 3.7 Rejected packs — lead with these

Eight-plus packs were built, gated, and killed. Naming your kills is more credible than naming your wins:

| Pack | Why killed |
|---|---|
| `eps_surprise` (PEAD) | Negative standalone at every horizon, even after the quarterly rebuild |
| `microstructure` (skew, downside ratio, Amihud, turnover, efficiency ratio) | 4 of 5 carry no within-sector signal; Amihud does, but at ρ≈0.87 with `log_market_cap` — it *is* the size factor |
| `seasonality` (Heston–Sadka same-calendar-month) | Zero within-sector signal at every horizon, sec_p ≫ 0.10 |
| `insider` (Cohen–Malloy–Pomorski Form 4) | 994k rows backfilled 2010–2026; only `net_ratio_12m` passed the p-gate, and it's 0.55-correlated with `fund_available` (i.e. a filer-availability proxy); 8-seed net-of-book ablation *hurt* mean IC 6–8% |
| `eps_dispersion` (Diether–Malloy–Scherbina) | It's a short-constraint signal; long-only can't extract it (sec_p 0.53) |
| `short_interest` (FINRA Reg SHO) | ρ 0.34 with `log_market_cap` → size proxy; and the FINRA feed ends 2022-11 |
| `--winsorize-target` | 3M degrades monotonically; 6M/1Y sub-threshold |
| `--industry-relative` normalization | Hurts every horizon |
| `linear_blend` (GBDT + ridge stack) | Built, never beat pure GBDT enough to promote |

### 3.8 The gate itself — `feature_diagnostics()`

Before *any* promotion fit, a candidate is scored on:
1. **mean |Spearman corr| vs every existing book feature** + the top-3 individual correlations (decorrelation);
2. **|corr| vs `efficiency_ratio_120d`** (Kaufman efficiency ratio — a trend-vs-consolidation proxy);
3. **standalone within-sector IC**, moving-block bootstrapped (`sec_ic`, `sec_t_block`, `sec_p`);
4. **IC within low / mid / high efficiency-ratio tertiles** — signal that lives *only* in the sideways
   tertile is a consolidation-regime bet, not broad alpha.

This is a real pre-registration discipline and it costs nothing (pure panel statistics, no fits). It is
one of the most defensible things in the project. **Say the name of the confound you were controlling
for** — "I was worried a new feature would just be picking off sideways/consolidating names, so the gate
splits IC by a trend-efficiency tertile" is a sophisticated thing to have thought of.

---

## 4. Validation methodology — the section that matters

### 4.1 Split design: purged expanding-window walk-forward

```python
# gbm_baseline.walk_forward_folds
def walk_forward_folds(grid_dates, min_train_months, embargo_steps):
    return [(grid_dates[i], grid_dates[i - embargo_steps])
            for i in range(min_train_months + embargo_steps, len(grid_dates))]
```

and in `walk_forward_ic`:

```python
embargo_steps = max(1, ceil(HORIZON_TRADING_DAYS[horizon] / 21))   # 3M→3, 6M→6, 1Y→12
train = panel[(panel.date <= cutoff) & panel[mask_h] & panel[target].notna()]
test  = panel[(panel.date == test_date) & panel[mask_h]]
```

**What this means precisely:**
- **Expanding window** by default; 6M overrides to a **rolling 60-month** window in production.
- `min_train_months = 36` — no test fold until 36 grid dates of history exist.
- The train cutoff sits `embargo_steps` month-ends *before* the test date, where `embargo_steps` is
  exactly the label horizon in months. A training row dated at the cutoff has a label window
  `[cutoff, cutoff + H]` that ends **on** the test date — so it cannot overlap `[test, test + H]`.
- **Refit every single fold.** No model is reused across folds.
- Test cross-sections thinner than `min_names = 30` are skipped.

**[measured] Fold counts and effective sample size:**

| Horizon | purge (months) | folds generated | folds scored | block size | **effective blocks** |
|---|---|---|---|---|---|
| 3M | 3 | 148 | **145** | 3 | 48.3 |
| 6M | 6 | 145 | **138** | 6 | 23.0 |
| 1Y | 12 | 139 | **126** | 12 | **10.5** |

**Vocabulary precision — get this right.** In López de Prado's taxonomy, what the code does is a
**purge**: dropping training observations whose label windows overlap the test window. His **embargo**
is an *additional* buffer *after* the test set to handle serial correlation in features. The code
calls its variable `embargo_steps` but its length is exactly the horizon, so it is a purge with no
extra embargo.

- ✅ Say: "purged walk-forward, purge length = label horizon."
- ❌ Don't say: "purged and embargoed combinatorial cross-validation" — CPCV is not implemented.

**Why not k-fold** (have this ready, it's a near-certain question): shuffled k-fold on a time series
puts future observations in the training set for past test observations. With 6-month overlapping
labels, a random fold split would let the model see the realized outcome of a nearly identical
(ticker, date+1day) sample. IC would go up dramatically and mean nothing.

**⚠️ There is no final untouched holdout for the promoted config.** `single_split_ic()` exists in
`gbm_baseline.py` with `fit_cutoff` / `holdout_start` parameters, but it is **not wired into the
walk-forward sweep CLI** — its only caller is `backend/ml/compare_transformer_gbm.py`, a separate
transformer-vs-GBM comparison. Every promotion number comes from the same walk-forward over the same
panel that was also used for every selection decision. See §4.5.

### 4.2 Survivorship — the actual mechanism, and its actual limits

**What's implemented.** Two seed scripts:
- `scripts/seed_sp500.py` — scrapes the **current** S&P 500 constituent table from Wikipedia,
  `INSERT ... ON CONFLICT DO NOTHING`, never deletes. Docstring: *"removing a name later means
  active=false, never a delete — survivorship bias."*
- `scripts/seed_sp500_historical.py` — parses Wikipedia's **"Selected changes"** table for names
  **removed since `--since` (default 2016-01-01)**, inserts them with `active = false, removed_at = <date>`,
  and backfills their price history.

The panel then keeps a name alive only while it actually trades, via `max_stale_days = 7` in
`build_ticker_rows` — a ticker whose last bar is more than 7 days before the grid date is dropped from
that cross-section. That's the mechanism, and it's a clean one.

`load_frames` selects `from tickers order by ticker_id` with **no `active` filter** — removed names are
deliberately included in the training panel. The de-survivorship is real and intentional.

**Now the five limits. Know all of them.**

**(a) Index *inclusion* is not point-in-time — this is the mirror-image bias and it is not addressed.**
`load_frames` has no membership-as-of-date logic. A stock added to the S&P 500 in 2023 has price history
back to 2016 in the panel and appears in the 2017 cross-section, even though it wasn't in the index then.
That is **look-ahead universe bias**: the historical cross-sections are pre-loaded with names that later
earned promotion into the index — a selection on future success. De-survivorshipping deletions without
also handling additions fixes half the problem.

- ✅ "The universe includes removed-from-index names, so it isn't survivor-only."
- ❌ "Survivorship-bias-free universe." It isn't, and the inclusion side is arguably the bigger effect.

**(b) The de-survivorshipped cohort has no metadata — which creates the §0.1 leak.**
**[measured]** 121 tickers have prices but no sector (`0/121` have fundamentals; `2/121` have shares
outstanding). `seed_sp500_historical.py` inserts only `(symbol, name, active, removed_at)` — no sector,
no CIK, hence no EDGAR backfill.

Consequences:
1. `within_sector_ic()` does `tmp.dropna(subset=["grp"])` → **the headline SECB metric is computed only
   on the ~517 current constituents.** The de-survivorshipped names contribute training rows but never
   appear in the metric.
2. `apply_target_modes` gates on `sector.notna() & count >= 5`, so those names train against the
   universe-demeaned target while everyone else trains against the sector-demeaned target.
3. `fund_available`, `log_market_cap` and all 5 EDGAR features are identically zero for exactly that
   cohort → the model can trivially identify "will be removed from the index" (§0.1).

**(c) Coverage starts in 2016 for most names, and pre-2016 is 100% survivors.**
**[measured]** 411 of 638 tickers have their first bar on 2016-01-04; all 146 names with 2010 history
have a sector, i.e. all are current constituents. Combined with `--since 2016` on the historical seed,
the pre-2016 panel is entirely survivor names. (Mitigating: restricting scoring to 2017+ folds barely
changes SECB — see §2.1.)

**(d) Delisting returns are not captured.** `compute_targets` masks a horizon when
`end_idx + H` doesn't exist in the ticker's own series. A name that goes bankrupt simply has no label
for windows spanning the delisting — the −100% is never observed. **[measured]** in practice only **13**
tickers' price series end before 2026 (2017:1, 2018:7, 2019:2, 2020:2, 2022:1), because most index
removals were size-based demotions or acquisitions where the ticker kept trading or yfinance stops
serving it. So the panel contains almost **no realized failures**. That's a real, if small, upward bias
and a good "what would you fix" answer (CRSP delisting returns).

**(e) The label uses an *index* shift on the ticker's own bar series, not a calendar shift.**
`adj_close[end_idx + 126]` is "126 available bars later." For a name with a long trading halt, that can
span far more than 126 calendar trading days, making its label non-comparable to peers in the same
cross-section. Rare in large-cap US post-2010, but it's the kind of thing worth naming before someone
else does.

### 4.3 Look-ahead audit — findings, ranked

#### ✅ Clean

- **Price/technical features.** Every index in `_price_features` is `≤ pos`. Windows are
  `adj_close[pos-251 : pos+1]`. No forward slices anywhere.
- **Returns and momentum from `adj_close`.** yfinance's back-adjusted close is
  `close_t × Π_{s>t} f_s`. In a ratio `adj_close[t]/adj_close[t−k]`, all factors after `t` cancel and
  what remains is the correct total return over the window. **Ratios are clean.** (Levels are not — see below.)
- **Fundamentals join on `filed_at`**, unit-tested four ways.
- **Estimates join on `as_of_date`, per field independently; surprises on `report_date`**, with the
  pre-report FQ0 consensus verified by a targeted LSEG probe.
- **Cross-sectional transforms are all per-date `groupby("date")`** — rank normalization, median demean,
  qcut bucketing, grade assignment, winsorization. None of them see another date. This is important:
  it's why computing the panel once up front and *then* walking forward is not a leak.
- **Purge length ≥ label horizon** in both research and production paths.
- **Production training filter** is `panel.date < as_of` **AND** `mask_h`, and `mask_h` is false for the
  last H bars, so production automatically excludes unrealized labels. Research and production use the
  same rule.
- **Seasonality** (rejected, but built correctly) uses only completed calendar months strictly before
  the bar's month.

#### ⚠️ Real problems

**1. `fund_available` encodes future index removal.** §0.1. The most serious one. Universe IC +0.052 to
+0.090 standalone.

**2. `log_market_cap` is contaminated twice over.**
```python
log_mcap = math.log(P * shares_outstanding)     # P = adj_close[pos]
```
- `shares_outstanding` is a **static scalar on the `tickers` table** — today's share count. The schema
  comment even says so: `-- current shares; with adj_close gives historical market cap`. Applying 2026
  shares to a 2012 price misstates market cap by the entire intervening buyback/issuance history —
  which is itself correlated with returns.
- `adj_close` is back-adjusted, so its **level** at date `t` embeds every dividend and split **after** `t`.
  A consistent dividend payer has a systematically depressed historical `adj_close`, so it looks smaller
  than it was. That is literally future information in a level.

  Magnitude: at a 2% dividend yield over 10 years the cumulative factor is ~0.82, i.e. ~0.20 in log
  terms, against a cross-sectional log-market-cap spread of ~4–6. Modest per name, but **systematic and
  monotone in future dividend yield**, and largest for the oldest data.

  This matters more than it otherwise would because **[measured]** `log_market_cap` is the load-bearing
  feature (§0.2).

  **The fix is cheap:** `price_history` already stores raw `close` (`auto_adjust=False`); `dataset._PRICE_SQL`
  just doesn't select it. Use `close × shares` for the level, keep `adj_close` for returns, and add a
  point-in-time shares series (EDGAR `dei:EntityCommonStockSharesOutstanding` is already being fetched
  from companyfacts for other fields).

**3. `sector` and `industry` are today's GICS classification applied retroactively.** The `tickers` table
holds one current sector per name. The 2018 GICS restructuring alone moved Alphabet, Meta and Netflix
into Communication Services. Since the **headline SECB metric and the production training target are
both defined by sector membership**, the metric is computed with hindsight sector labels. Small, but it
is a hindsight input to the primary metric — name it before the interviewer does.

**4. No implementation lag.** Features are computed from the close of bar `pos`, and the label is
`log(adj_close[pos+H] / adj_close[pos])` — the return **starts at the same close** the signal is
computed from. That assumes you can observe the close and trade at that close. At a monthly rebalance
this is a small effect, but it is a free half-day of alpha and the standard fix (signal at close `t`,
return from close `t+1`) is one index. Say: "no implementation lag is modeled — signal and entry are
both at the same close."

**5. Same-day filing inclusion.** `bisect_right(filed_ats, d) - 1` includes a filing whose `filed_at`
is *on* day `d`. A 10-Q hitting EDGAR at 4:05pm ET is used with that day's close. Deliberate (there's a
test asserting it), and defensible for a monthly rebalance, but strictly it should be next-open.

**6. `price_target_upside` mixes adjusted and nominal prices** — `(pt − adj_close) / adj_close`, where
`pt` is a nominal price target as of that date. Same back-adjustment contamination as `log_market_cap`.
Not in production (it's in the rejected `forward_valuation` pack) but worth knowing.

**7. yfinance data is itself point-in-time-unsafe.** yfinance serves *current* adjusted history, and
`prices.py` re-pulls the full history from 2010 whenever drift is detected. So historical `adj_close`
in the DB is periodically **rewritten** to reflect newer corporate actions. This is fine for returns
(ratios cancel) but it means the database is not an as-of archive.

### 4.4 Transaction costs and slippage — **not modeled at all**

**[searched the whole repo]** There is no backtest, no portfolio construction, no P&L series, no
position sizing, no commission or spread model, no market-impact model, no capacity analysis. The only
"Sharpe" in the codebase is a **trailing realized 1-year Sharpe of an individual stock**, computed in
`api/routers/rankings.py` for display/sorting in the screener UI, with `rf = 0`. It is not a strategy
Sharpe and must never be described as one.

What *does* exist as a cost proxy:

```python
def rank_turnover(records, rank_series=None):
    """Mean |Δ percentile-rank| of a name between consecutive scoring dates."""
```

Measured turnover: **[from project records]** ~0.105 at 6M, ~0.038 at 1Y, ~0.067 at 3M post-smoothing
(down from 0.133 raw). This is a genuine, honestly-defined stability metric and cross-date EWMA smoothing
was promoted partly on it (3M turnover −50%, 1Y −54%).

**But mean |Δ rank| is not turnover in the portfolio sense** — it doesn't map to shares traded or dollars
crossed without a portfolio construction step. Be precise:

- ✅ "I track a rank-turnover proxy — mean absolute change in percentile rank between rebalances — and I
  promoted a smoothing overlay partly on that, cutting it in half at 3M."
- ❌ "Turnover-aware," "cost-adjusted," "net of costs," "Sharpe."

**Rough sizing you should be able to do live.** A monthly-rebalanced long-only top-decile book of ~60
large-cap names, with rank turnover ~0.10, plausibly trades on the order of 20–40% of the book a month
→ ~2.4–4.8× annual turnover. At 5–10 bps round-trip for US large caps, that's roughly **25–100 bps/year**
of cost. Then: an IC of 0.05 with breadth ~500 gives, by the fundamental law
(`IR ≈ IC × √breadth`, before any cross-sectional correlation haircut), a gross IR around 1.1 —
so the cost drag is real but not obviously fatal at these horizons. **Say this as an order-of-magnitude
estimate you can do in your head, not as a result.**

### 4.5 The overfitting risk that the walk-forward does *not* protect against

This is the most important honest caveat and you should volunteer it.

The walk-forward is out-of-sample **per fold** — no fold's model ever saw its own test data. But the
**production configuration was selected by looking at the walk-forward results**, and there is no held-out
set that was never consulted. Counting the CLI surface in `gbm_baseline.py`:

- 8 `--target` modes
- 14 `--with-*` feature packs
- `--smooth-span`, `--knife-lambda` (+ `--knife-sweep`), `--max-train-months`, `--objective`,
  `--rank-grades`, `--lambdarank-truncation`, `--subsample`, `--winsorize-target`, `--industry-relative`,
  `--with-linear-blend`, `--ridge-alpha`, `--reg-sweep`, `--target-blend-sweep`
- × 3 horizons

That is easily **100+ distinct configurations evaluated against the same 126–145 folds.** The promotion
bar was a **+10% ICIR improvement**, applied per-experiment, with **no multiple-testing correction** —
no Bonferroni, no White's Reality Check, no Hansen SPA, no Deflated Sharpe Ratio.

With ~10 effective independent blocks at 1Y and 100+ trials, the maximum observed t-statistic under a
pure null is not small. The promoted 1Y config (LambdaRank, ICIR 1.109, t_block 3.58) is the *winner of
a large search*, and its reported statistics are conditioned on having won.

**What genuinely mitigates this** (say all of these):
1. The **direction** of the model space was theory-driven, not grid-searched — LambdaRank because the
   metric is Spearman; sector-relative target because universe IC can be sector rotation; rolling window
   because of regime staleness. Each was a hypothesis with a prior, not a knob turned at random.
2. **Most experiments were killed, and the kills are documented.** A search that accepts 5 of 100 with
   pre-registered reasons behaves very differently from one that accepts the argmax.
3. The `feature_diagnostics` gate is a genuine **pre-registration** step — candidates had to clear
   decorrelation + standalone significance + a confound check *before* any promotion fit.
4. The **shuffle null** (`--null-reps`, permuting training labels while preserving marginals) gives a
   no-signal band the real IC must clear, and there's a synthetic planted-signal test
   (`test_walk_forward_recovers_planted_signal_and_null_does_not`) validating the harness itself.

**What would actually fix it, and you should offer this unprompted:** hold out 2024–2026 entirely, run
the whole selection process on 2010–2023 only, then evaluate the frozen config once on the held-out
period. The plumbing already exists — `single_split_ic()` takes exactly a `fit_cutoff` /
`holdout_start` pair; it just isn't wired into the sweep CLI.

---

## 5. Metrics — exactly how each number is computed

### 5.1 Rank information coefficient (universe IC)

```python
# gbm_baseline.walk_forward_ic, inside the fold loop
ic = pd.Series(preds).corr(pd.Series(test[r_col].to_numpy(float)), method="spearman")
```

- **Spearman**, not Pearson — a rank correlation, robust to the fat tails in returns and matched to how
  the output is consumed (a percentile rank).
- Computed **within a single monthly cross-section** (`test` is one grid date), across all names with a
  valid label — typically ~580 post-2016, ~150 pre-2016.
- `preds` = the 1-seed or 8-seed-averaged model output; `r_col` = `r_{h}`, the universe-median-demeaned
  realized forward log return. (The demean is a within-date constant shift, so it does not affect the
  Spearman value.)
- Then averaged across folds: `summarize()` returns `mean_ic`, `std_ic`, `icir = mean/std`,
  `t_stat = icir × √n`, `hit_rate = fraction of folds with IC > 0`.

**Out-of-sample?** Yes at the fold level — the model for fold *i* was fit only on data up to
`grid_dates[i − embargo_steps]`. **But the configuration was chosen using these same folds** (§4.5).
The precise honest phrasing: *"out-of-sample per fold; the configuration was selected on the same folds,
so it isn't a clean held-out estimate of the promoted config."*

### 5.2 Within-sector IC (SECB) — the headline

```python
def within_sector_ic(preds, test_df, r_col, min_group_size=10, group_col="sector"):
    for _, grp in tmp.dropna(subset=["grp"]).groupby("grp"):
        if grp.shape[0] < min_group_size: continue
        ics.append(grp["pred"].corr(grp["r"], method="spearman"))
    return np.mean(ics)
```

Per date: Spearman IC computed **inside each GICS sector** with ≥ 10 names, then **equally averaged
across sectors**; then averaged across folds and block-bootstrapped.

**Why it's the right headline:** universe IC rewards a model that ranks Energy above Tech in 2022. That's
sector rotation — cheap to get from a sector ETF, and not what a stock-selection model should be paid
for. SECB asks the harder question: *within* Energy, did you rank the right names? Every promotion in
the project was graded on it.

**Two caveats to state yourself:**
- **[measured]** SECB is computed only on the ~517 names that have a sector label — i.e. the current-
  constituent (survivor) subset. The de-survivorshipped cohort is invisible to the metric.
- Sector labels are today's GICS, applied retroactively.

### 5.3 Significance — moving-block bootstrap (the sophisticated part)

```python
# gbm_baseline.block_bootstrap_summary
block_size      = ceil(HORIZON_TRADING_DAYS[h] / 21)      # 3, 6, 12 monthly folds
effective_blocks = n_folds / block_size
t_block         = mean / std * sqrt(effective_blocks)
# + 2000-rep moving-block resample of contiguous blocks:
#   boot_means  -> 95% CI
#   null_means  -> centered null (a - mean); p = (1 + #{|null| >= |mean|}) / (reps + 1)
#   se_block    -> min_detect_ic = 1.96 * se_block
```

**Why this exists — this is your best "I understand the statistics" moment.** Fold ICs are computed
monthly but the labels are 6- or 12-month forward returns, so consecutive folds share ~5/6 or ~11/12 of
their label window. They are massively autocorrelated. The naive `t = ICIR × √n` treats 126 folds as 126
independent observations. The block bootstrap resamples **contiguous blocks of length = the horizon in
months**, preserving that autocorrelation.

**[measured] The correction is large:** at 1Y, 126 folds collapse to **10.5 effective blocks** — the
naive t-stat is inflated by roughly **√12 ≈ 3.5×**. The reported `t_block` for 1Y is 3.58 where the
naive `t_naive` would be far higher. CLAUDE.md is explicit that `t_naive` is not reported for this reason.

And `min_detect_ic = 1.96 × se_block` is an honest **power** statement: the smallest |IC| a two-sided
95% test could resolve at that block count. At 1Y it's ≈ 0.032 — comparable to the effect being measured.
Volunteering "my 1Y horizon is power-limited, my minimum detectable IC is about the size of my measured
IC" is exactly the kind of self-awareness that lands well.

### 5.4 The reported table — read it correctly

From `CLAUDE.md` (8-seed ensemble, production per-horizon specs, block-bootstrapped):

| Horizon | mean IC | t_block | p_block | **SEC IC** | **SECB t** | **SECB p** |
|---|---|---|---|---|---|---|
| 3M (+revmom, smooth 3, knife 0.20) | +0.047 | 2.89 | 0.002 | **+0.027** | 1.96 | 0.011 |
| 6M (+rev, rolling-60, LambdaRank) | +0.078 | 3.44 | 0.0005 | **+0.056** | 2.75 | 0.0005 |
| 1Y (+rev, smooth 4, LambdaRank) | +0.059 | 1.86 | 0.021 | **+0.069** | 3.58 | 0.0005 |

**How to quote these:**
- **Quote the SEC IC column, not mean IC.** The universe column is the one contaminated by §0.1.
- Period: **145 / 138 / 126 monthly walk-forward folds**, roughly **2013–2026**, with a realistic
  ~580-name cross-section only from 2017 on.
- Universe: **~600 US large caps** — current S&P 500 plus names removed since 2016. Do *not* say
  "the S&P 500," because point-in-time membership isn't reconstructed.
- Always attach "out-of-sample per fold; configuration selected on the same folds."

**Benchmark context — memorize this.** A cross-sectional rank IC of 0.03–0.05 is a *normal, respectable*
equity factor result. Qlib's published LightGBM/Alpha158 benchmark on CSI300 sits around **0.045**.
Grinold–Kahn's fundamental law, `IR ≈ IC × √breadth`: at IC 0.05 with 500 names, gross IR ≈ 1.1 —
which is why small ICs matter at scale. Anyone quoting IC of 0.15+ on a liquid large-cap universe is
either using a very short horizon, or leaking. **+0.05 is the right order of magnitude for real signal,
and saying so shows you know the literature.**

### 5.5 Other reported metrics

| Metric | Definition in code | Notes |
|---|---|---|
| **ICIR** | `mean_ic / std_ic` across folds | Naive across-fold; not block-corrected. 6M 0.575, 1Y 1.109 after LambdaRank. |
| **hit rate** | fraction of folds with IC > 0 | 1Y 0.89. Beware: with 12-month overlapping labels this is ~10 independent observations, not 126. |
| **rank turnover** | mean \|Δ percentile rank\| between consecutive dates | Stability proxy, not portfolio turnover. |
| **top_decile_risk** | ex-ante knife/vol/downtrend + ex-post downside semi-deviation and tail fraction of the top decile | Real risk metric; drove the knife overlay promotion (top-decile knife −47%, realized downside −7%). |
| **`predictions.confidence`** | std of predicted rank over the last ≤ 3 scoring dates | Rank *stability*, not a calibrated confidence. |
| **`predictions.direction_prob`** | the clipped predicted percentile rank in [0,1] | **Not a probability.** The column name is misleading; CLAUDE.md flags this and the UI copy says "relative rank." Volunteer this — it shows you audit your own naming. |

---

## 6. Model choice — why LightGBM

### The argument

**Data characteristics:** ~79k rows × ~22 features, tabular, heterogeneous (momentum, ratios, binary
availability flags, counts), heavily non-Gaussian, non-stationary, and — critically — with a
signal-to-noise ratio where the best achievable R² on individual returns is on the order of 0.1–1%.

**Why gradient-boosted trees:**
1. **Tabular + low SNR is where GBDTs dominate.** This is the single most replicated result in applied
   ML on tabular data (Grinsztajn et al. 2022, "Why do tree-based models still outperform deep learning
   on tabular data?"). Gu–Kelly–Xiu (2020) find trees and shallow nets both beat linear models on the
   US equity panel, with trees far more robust.
2. **Monotone-transform invariance.** Trees split on order, so the per-date rank normalization is a free
   robustness win rather than a modeling constraint.
3. **Native interaction discovery** without specifying the interaction — e.g. "high vol AND below the
   200-day MA," which the falling-knife overlay later made explicit.
4. **Fast enough to refit per fold.** 145 refits × 8 seeds is minutes for a regressor. **[measured]**
   ~25s for a full single-seed 138-fold 6M walk-forward. That's what makes the walk-forward affordable
   at all, and affordability is what made the ablations in §0 possible. This is a genuinely good
   argument: *the model was chosen partly so the validation could be rigorous.*
5. **Interpretability adequate to the task** — split gain, drop-one ablation, and the standalone-IC
   feature gate all work naturally.
6. **LightGBM specifically over XGBoost/CatBoost:** leaf-wise growth with explicit `num_leaves` control
   suits shallow, heavily-regularized fits; and crucially it ships `LGBMRanker` with LambdaRank and
   query groups, which is what made the objective/metric alignment possible without leaving the library.

**Why the model is deliberately tiny** (`num_leaves=15, max_depth=4, n_estimators=300, lr=0.03,
subsample=0.8, colsample=0.8, reg_lambda=1.0, min_child_samples=50`): with ~580 names per cross-section
and true R² well under 1%, an unconstrained GBDT memorizes cross-sections instantly. The comment in
the code says exactly this. Pair that with **8-seed ensembling** to average out the variance a
high-variance learner has at this SNR.

### The LambdaRank decision — your best modeling story

The fit was pointwise L2 (predict the return, then rank), but the *metric* is pure Spearman rank IC.
That's a known misalignment (Poh et al. 2020 on learning-to-rank for portfolio selection).

The fix, in `_fit_lgbm_ranker`:
- Train an `LGBMRanker` on an ordinal `sector_grade` target — per-date `qcut(sector_return, 5)`.
- **Per-date query group.** The panel is ticker-major, so rows are re-sorted to date-contiguous blocks
  with a **stable mergesort** (deterministic across seeds) and group sizes read off in that order —
  LightGBM requires each query group to occupy a consecutive row span. That's a real implementation
  detail most people get wrong.
- **Linear `label_gain` `(0,1,…,K−1)` instead of LightGBM's default `2^i − 1`.** The default is
  top-heavy (built for NDCG); Spearman and SECB reward monotone ordering through the *whole* list. This
  is the detail that shows you actually understood the loss rather than flipping a flag.
- **`lambdarank_truncation_level=100`** rather than full-list 500: a ranker fit is ~14× a regressor fit
  at truncation 100 and ~34× at 500 (~3 hr/horizon in walk-forward) for negligible extra signal — and
  100 is what the product actually surfaces. An explicit, quantified compute/signal trade-off.
- **Zero inference changes required:** `fit_lgbm_model` branches on `lgb_cfg.objective`, and
  `LGBMRanker.predict` has the same signature. The overlays compose unchanged because they operate on
  **percentile ranks**, and a ranker's score change is monotone-invariant at that layer.

Result: 6M SEC ICIR 0.494 → 0.575 (+16.4%), 1Y 0.747 → 1.109 (+48.5%). **3M was not promoted** — ICIR
+8.9% (below the +10% bar) *and* mean IC −9%, a net loss. Saying "and I didn't promote it at 3M because
it failed my pre-set bar" is worth more than the promotion itself.

### Alternatives — what to say if pushed

| Alternative | Honest take |
|---|---|
| **Linear / ridge on rank-normalized features** | Should be the baseline and arguably should have been the *first* thing tried. Gu–Kelly–Xiu show linear models are competitive at low SNR. `fit_linear_model` + `blend_gbdt_linear` are implemented (`linear_blend`), tested, and **never beat pure GBDT enough to promote**. Given §0.2 — a signal dominated by one monotone feature — a ridge might well match the GBDT. **Concede this readily**; it's a strength to say "a linear model would probably do nearly as well, and that's informative about how much structure is really there." |
| **XGBoost / CatBoost** | Would perform comparably. LightGBM was chosen for leaf-wise control and `LGBMRanker`. CatBoost's ordered boosting would be a reasonable robustness check. |
| **Random forest** | Lower variance, but no natural ranking objective and worse at the shallow-regularized regime this needs. |
| **Neural nets (MLP / PatchTST)** | Tried. `backend/ml/model.py` is a 4-layer PatchTST encoder with a FeatureGate variable-selection layer and multi-horizon heads, ~1M params. **It failed its holdout — but it was evaluated on the 1M horizon, which the walk-forward later showed has no cross-sectional signal at all (t = 0.59).** So the honest read is "the evaluation was mis-specified, not necessarily the model." It was shelved rather than re-run because ~79k rows is far too little for a 1M-param sequence model, and GBDT was clearly good enough. **This is a strong story: a negative result whose cause you later diagnosed.** |
| **A proper factor model (Fama–MacBeth / Barra)** | This is what a quant interviewer will *actually* suggest, and given §0.2 they'd be right: cross-sectional regression with explicit factor exposures and residual (idiosyncratic) return as the target would immediately reveal that the signal is mostly size, and would give you neutralization for free. **Have "that's the next thing I'd build" ready.** |

---

## 7. Hardest technical problem — two stories

### Story A (primary): "I audited my own model and found the signal was mostly a data artifact plus size"

*Use this one.* It's specific, quantitative, honest, and demonstrates exactly the skepticism the job requires.

**Setup.** The model reported a universe IC of +0.078 at 6M and a within-sector IC of +0.056, both
block-bootstrap significant. Good numbers. But the universe number felt too good relative to the
within-sector number, and I'd deliberately de-survivorshipped the universe by adding back names removed
from the index — so I went looking for what those added names were actually contributing.

**Investigation.** The removed-name cohort was seeded from Wikipedia's index-change table with only
symbol, name and removal date — no sector, no CIK. No CIK means no EDGAR backfill. So 121 of 638 tickers
had *zero* fundamentals, and my `fund_available` indicator — which I'd added specifically so the tree
could distinguish "margin is genuinely zero" from "no filing yet" — was in fact a flag for
**"this name will be dropped from the index."** Known from the first day of the panel. That's future
information laundered through a missingness pattern.

**Quantification.** Standalone, `fund_available` alone has a universe IC of **+0.072 at 6M (t = 3.57)** —
essentially the entire model's universe IC. Its within-sector IC is ≈ 0, because `within_sector_ic` drops
null-sector rows, which is exactly that cohort. Drop-one ablation confirmed it: removing the flag costs
9% of universe IC but only 6% of within-sector IC; removing the whole fundamentals block costs 34% of
universe IC and 7% of within-sector IC.

**Second finding.** While I had the ablation harness up, I ran drop-one across the book. Removing
`log_market_cap` takes within-sector IC from +0.048 (t = 2.29) to +0.020 (t = 0.88) at 6M, and from
+0.037 to **−0.013** at 1Y. So the within-sector signal is load-bearing on size — a documented risk
premium, not alpha. And the size feature is itself built wrong: `log(adj_close × current_shares)`, where
the back-adjusted close embeds all *future* dividends and the share count is today's.

**Resolution and what I'd do next.** The immediate conclusions: report within-sector IC rather than
universe IC (it's structurally immune to the availability artifact — which retroactively validates the
choice of headline metric); either backfill CIK/sector/fundamentals for the removed cohort or drop
`fund_available` entirely; rebuild `log_market_cap` from the raw `close` column already in the database
plus a point-in-time share count; and then re-run with an explicit size neutralization to find out how
much signal survives. That last number is the one I'd actually want before putting money behind it.

**Why this story works:** you found it, you measured it, you know the fix, and you're not defensive.
Every interviewer has met the candidate who defends a contaminated backtest.

### Story B (backup): "My t-statistics were lying to me because of overlapping labels"

**Problem.** Rebalance monthly, but predict 6- and 12-month forward returns. Two consecutive folds share
5/6 (or 11/12) of their label window — so 126 monthly fold ICs are nowhere near 126 independent
observations. The naive `t = ICIR × √n` was giving t-stats of 5–6 and I didn't believe them.

**Approach.** Implemented a **moving-block bootstrap** (`block_bootstrap_summary`) with block length =
the horizon in months. It resamples contiguous blocks of fold ICs with replacement, preserving the
autocorrelation that the label overlap induces. From 2000 replicates: a percentile CI for the mean IC,
and — separately — a **centered null** (resample `a − mean`) giving a two-sided p-value for `mean_IC ≠ 0`.

**The number that made it worth it.** At 1Y, 126 folds → **10.5 effective blocks**. The correction is
~√12 ≈ 3.5× on the t-statistic. Several results that looked significant under the naive t did not
survive — `beta_resid` at 1Y went to SECB p = 0.266 and got killed, which is what prompted the target
redefinition to `sector_return`.

**The part I'm most pleased with.** I also report `min_detect_ic = 1.96 × se_block` — the smallest |IC|
a two-sided 95% test could resolve given the block count. At 1Y that's ≈ 0.032, comparable to the effect
size being measured. So when 1Y looks weak, the harness tells you whether that's the model or the power,
and I stopped over-interpreting 1Y results as a result.

**Trade-off I accepted.** The block bootstrap is conservative — it treats the horizon as the full
decorrelation length when the true one is shorter. I'd rather under-claim.

### Story C (if they want a data-engineering angle): the LSEG point-in-time consensus probe

Quarterly surprise features need the **pre-report** consensus. Most vendor endpoints hand you the
current consensus for a fiscal period — which, after the announcement, has already been revised to
reflect the actual. Using that silently makes surprise ≈ 0 for everyone and destroys the feature, or
worse, leaks. I wrote a one-off probe (`scripts/_probe_lseg.py`) to pull the `Period=FQ0, Frq=FQ` grid
and check it against the last monthly snapshot **before** each report date, confirming it was the
pre-announcement value and not the post-announcement revision. Rebuilding `revenue_surprise` on the
verified quarterly grid (it had been on annual data) strengthened it at 6M and made it a wash at 1Y.
`eps_surprise` stayed rejected — negative standalone at every horizon, even on clean quarterly data,
which is itself a reasonable finding about PEAD in post-2010 large-cap US.

---

## 8. Likely interview questions, with answers

**"How do you know this isn't overfit?"**
- Every fold refits from scratch on data ending a full label horizon before the test date. No fold's
  model saw its own test data.
- The model is tiny by design: 15 leaves, depth 4, 300 trees at lr 0.03, 80% row and column bagging,
  L2 penalty, min 50 samples per leaf — on 79k rows.
- A shuffle null (permuted training labels) gives the no-signal band, and there's a synthetic
  planted-signal test that verifies the harness recovers a known signal and the null doesn't.
- Significance uses a moving-block bootstrap, not the naive t.
- **And here's what it does *not* protect against:** the configuration was selected on those same folds,
  across 100+ variants, with no multiple-testing correction. So the promoted config's reported stats are
  conditioned on having won a search. The fix is to freeze 2024–2026, redo selection on 2010–2023, and
  evaluate once.

**"Is that in-sample or out-of-sample?"**
> Out-of-sample per fold — strictly. But I want to be precise: it is not a clean held-out estimate of
> the promoted configuration, because the configuration was chosen by looking at these folds. Those are
> different claims and I'd fail my own review if I conflated them.

**"Why walk-forward instead of k-fold?"**
- Shuffled k-fold puts the future in the training set. With 6-month overlapping labels, a random split
  lets the model train on the realized outcome of a nearly identical sample — a (ticker, t+1day) row
  shares ~99.6% of its label window with the (ticker, t) row in the test fold.
- Walk-forward also matches deployment: production refits monthly on everything available and scores
  today's cross-section, which is exactly what each fold simulates.
- And it lets you *see* regime dependence across folds — a single number can't.

**"What's your IC benchmark? Is +0.078 good?"**
- Cross-sectional rank IC of **0.03–0.05** is the normal band for a real equity factor model. Qlib's
  published LightGBM/Alpha158 benchmark is ≈ 0.045. So my within-sector +0.05 at 6M is in the right
  neighborhood — which is the point: anything much higher on a liquid large-cap universe should make you
  suspect leakage.
- By the fundamental law, IC 0.05 with breadth 500 implies a gross IR around 1.1 before costs and before
  any correlation haircut — that's why an IC that sounds tiny is economically meaningful at scale.
- **And I'd flag the +0.078 specifically.** That's the *universe* IC and I've measured that it's
  inflated by a data-availability artifact. The number I'd stand behind is the within-sector one.

**"Walk me through your survivorship handling."**
- Two seeds: current constituents, plus every name removed from the index since 2016 with its price
  history, inserted `active = false` and never deleted. `load_frames` applies no active filter, so
  removed names are in the training panel. A name stays in a cross-section only while it actually trades
  — `max_stale_days = 7` drops it once its last bar is stale.
- **Then the limits, which I'd rather state myself:** (i) *inclusion* is not point-in-time — a name added
  to the index in 2023 is in my 2017 cross-section, which is look-ahead universe bias and the mirror
  image of survivorship; (ii) the removed cohort was seeded without sector or CIK, so they have no
  fundamentals — which created the availability leak I mentioned; (iii) only 13 names' price series
  actually terminate, so realized delistings are essentially absent and I have no delisting returns —
  CRSP would fix that; (iv) pre-2016 the panel is ~150 names and all of them are survivors, though
  restricting scoring to 2017+ folds barely moves the result.

**"What would break this in live trading?"**
1. **Costs and capacity are not modeled at all** — there is no backtest, no P&L, no spread or impact
   model. I track a rank-turnover proxy, not portfolio turnover.
2. **The signal is dominated by a size tilt.** In a large-cap-led regime (2020–2021, 2023–2024) a
   small-cap-within-sector tilt bleeds, and my walk-forward would have shown that as bad folds rather
   than as a broken model.
3. **Two features are dead in-sample but live in production** — the FinBERT sentiment channels have no
   backfill, so the model has never learned to use them but they're non-zero at inference. Harmless
   today (a constant column earns no split gain) but it's a train/serve skew I'd close.
4. **Crowding.** Momentum, size, low-vol and revision momentum are all public factors. Realized IC on
   published factors decays post-publication (McLean–Pontiff); nothing here is proprietary.
5. **The data stack is fragile:** yfinance is unofficial and rate-limited; LSEG requires a running
   desktop session (the pipeline logs a `skipped` run when it's down); FinBERT runs locally on MPS.
6. **Point-in-time integrity of the price store.** yfinance serves current adjusted history and the
   drift detector re-pulls from 2010, so historical `adj_close` gets rewritten. Fine for returns, but
   the database is not an as-of archive.

**"How would you turn this into a tradeable strategy?"**
> Right now it isn't one, and I'd say that plainly — it produces a ranking, not positions.
> The path: (1) fix the size contamination and re-measure to find out how much idiosyncratic signal
> actually survives neutralization; (2) build portfolio construction — long-only top-decile or a
> sector-neutral long/short, with position limits and a size/beta/sector risk constraint; (3) then a
> real backtest with a next-day-open implementation lag, a spread + Amihud-based impact model, and
> realistic financing; (4) report net IR and drawdown, not IC; (5) capacity-test by scaling notional
> against ADV until impact eats the edge. At a monthly rebalance on ~600 large caps I'd guess a few
> hundred million in capacity, but that's a guess until step 5.

**"Why sector-neutral IC instead of plain IC?"**
> Because plain universe IC can be earned by sector rotation, which I can get from an ETF and which
> isn't stock selection. When I switched the *target* to within-sector relative return and re-graded on
> within-sector IC, 1Y went from failing (p = 0.27 under a beta-residual target) to clearly passing —
> the old target was maximizing idiosyncratic-vs-market alpha while leaving sector tilt in, which was
> exactly the thing I didn't want to be paid for. It also turned out to protect me from the availability
> artifact, since the contaminated names have no sector label.

**"Why LambdaRank? Isn't that overkill?"**
> The fit was L2 regression and the metric was Spearman — I was optimizing squared error on a target I
> only ever consumed as an ordering. LambdaRank optimizes a listwise ranking loss directly, with each
> date as a query group. Two details mattered: I used **linear** `label_gain` rather than LightGBM's
> default `2^i − 1`, because Spearman rewards monotone ordering through the whole list rather than just
> the top; and I set truncation to 100 rather than the full 500, because a ranker fit is 14× a regressor
> at 100 and 34× at 500 for negligible extra signal. It gave +16% ICIR at 6M and +48% at 1Y — and I
> *didn't* promote it at 3M, where it cleared neither my ICIR bar nor held mean IC flat.

**"What's your Sharpe?"**
> I don't have one, and I'd be suspicious of anyone who quotes one from an IC study without portfolio
> construction. I have information coefficients and a rank-turnover proxy. The only Sharpe in the
> codebase is a trailing realized Sharpe of individual stocks for a screener UI — it's not a strategy
> statistic and I wouldn't present it as one.

**"Your 1Y numbers look best. Should I believe them?"**
> Less than the 6M ones. 1Y has 126 folds but only ~10.5 effective independent blocks after the overlap
> correction, and my minimum detectable IC there is ~0.032, which is the same order as the measured
> effect. So 1Y is the horizon where I'd expect the promoted config to have benefited most from the
> selection search. 6M is the one I'd actually trust: most effective blocks among the passing horizons,
> the most stable ICIR, and the least sensitive to overlays.

**"Why is 1M dead?"**
> No detectable cross-sectional signal — t = 0.59 in the walk-forward, so it's not scored in production.
> That's consistent with short-horizon cross-sectional returns in liquid large caps being dominated by
> microstructure and reversal effects my daily-bar feature set can't see. It's also the reason my
> earlier transformer looked like a failure: I evaluated it on 1M, the one horizon with nothing to find.

**"How do you handle missing data?"**
> Zeros plus an explicit availability indicator, so the tree can separate "genuinely zero" from "not
> reported yet" — and after per-date rank normalization a zero lands at a consistent position in the
> cross-section rather than at an arbitrary scale. **That design is also what bit me:** the indicator
> became a proxy for future index removal because one cohort had no fundamentals at all. The lesson is
> that a missingness indicator is only safe if missingness is independent of the outcome, and I hadn't
> checked that.

**"What's the most surprising thing you learned?"**
> That 12-1 momentum — the single most-cited cross-sectional anomaly — has no significant standalone
> signal in my panel: universe IC +0.03 with t under 1.2, within-sector essentially zero, at every
> horizon. Post-2009 US large cap. It made me a lot less willing to assume a published factor is live
> in my data just because it's published.

---

## 9. Honest weaknesses — the "what would you improve" list

Ordered by how likely they are to come up.

1. **No transaction costs, no market impact, no capacity analysis, no backtest, no P&L.** The project
   produces rankings, not positions. Rank turnover is a stability proxy, not portfolio turnover.
   *Fix:* portfolio construction → next-day-open lag → spread + Amihud impact → report net IR.

2. **The within-sector signal is dominated by size, which is a risk premium, not alpha.** **[measured]**
   Removing `log_market_cap` kills significance at all three horizons (1Y flips sign).
   *Fix:* neutralize size (and beta, and sector) explicitly — Fama–MacBeth style — and re-measure the
   residual. That number is the one that matters.

3. **The size feature is constructed with future information** (back-adjusted close × today's share count).
   *Fix:* one-line change to select the raw `close` already stored in `price_history`, plus a PIT shares
   series from EDGAR.

4. **`fund_available` is a survivorship-availability leak.** **[measured]** +0.072 universe IC standalone
   at 6M. *Fix:* backfill CIK/sector/fundamentals for the removed cohort, or drop the feature.

5. **Universe membership is not point-in-time on the inclusion side.** Names added to the index later are
   present in earlier cross-sections. *Fix:* a `(date, ticker)` membership table and a per-date universe filter.

6. **No held-out final test set; no multiple-testing correction.** 100+ configurations, ~10 effective
   blocks at 1Y, a +10% ICIR promotion bar with no Bonferroni / White Reality Check / Deflated Sharpe.
   *Fix:* freeze 2024–2026, redo selection on 2010–2023, evaluate the frozen config exactly once.
   `single_split_ic()` already implements the split; it just isn't wired into the sweep CLI.

7. **Delisting returns are absent.** Only 13 tickers' series terminate; bankruptcies and their −100%
   are not observed. *Fix:* CRSP delisting returns.

8. **Two of 21 features are dead in-sample and live in production** (FinBERT sentiment: **[measured]**
   2.1% non-zero coverage, all of it after 2026-05-14). *Fix:* a paid news archive, or drop them from
   the served feature set until backfilled.

9. **Sector labels are current GICS applied retroactively** — and they define both the training target
   and the headline metric.

10. **No implementation lag.** Signal and entry are both at the same close.

11. **No regime analysis.** IC is reported as a pooled mean; there is no conditional breakdown by
    volatility regime, rate regime, or factor-crowding state. Given a size-tilted signal, regime
    dependence is very likely and currently invisible.
    *Fix:* fold-level IC time series plots and a VIX-tertile split. Cheap — the per-fold records are
    already returned by `walk_forward_ic(return_records=True)`.

12. **Would it still work if everyone did it?** Largely, no — and say so cleanly:
    > Every input is public: prices, EDGAR filings, sell-side consensus. Nothing here is a proprietary
    > dataset or a novel factor. McLean and Pontiff find published anomaly returns decay ~58% after
    > publication, and my strongest component is the size effect, which is the most crowded factor in
    > existence. What the project demonstrates isn't an edge — it's that I can build a
    > point-in-time-correct research pipeline, hold myself to an honest significance bar, and find the
    > flaws in my own results. That's the transferable part.

13. **Single-market, single-asset-class, single-frequency.** US large-cap equities, monthly, long-only.
    No shorts, no international, no cross-asset, no intraday.

14. **Manual promotion.** No automated candidate → production gate; you edit `PRODUCTION_HORIZON_SPECS`
    by hand. Defensible for a personal tool, but say "manual by design" rather than letting it look like
    an oversight.

---

## 10. Precise vocabulary

### ✅ Terms this project earns

| Term | Why it's justified |
|---|---|
| **Cross-sectional rank information coefficient (rank IC)** | Per-date Spearman between predicted rank and realized forward return. Exactly what's computed. |
| **Within-sector / sector-neutral IC** | Per-date IC computed inside each GICS sector (≥10 names) and averaged across sectors. |
| **ICIR** | mean IC / std IC across folds. |
| **Walk-forward out-of-sample evaluation** | Expanding (or rolling-60) window, refit every fold. |
| **Purged walk-forward; purge length = label horizon** | `embargo_steps = ceil(H/21)` months between train cutoff and test date. Say **purge**, not "embargo." |
| **Moving-block bootstrap for overlapping labels** | Contiguous-block resampling with block = horizon in months, centered-null p-value, 2000 reps. |
| **Effective sample size / effective independent blocks** | `n_folds / block_size` — reported. |
| **Minimum detectable IC (statistical power)** | `1.96 × bootstrap SE`. Explicitly computed and printed. |
| **Shuffle null / permutation null** | `--null-reps` permutes training labels preserving marginals. |
| **Cross-sectionally rank-normalized features** | Per-date rank → [−1, 1]. |
| **Point-in-time joins on filing receipt date** | `filed_at` for EDGAR, `as_of_date` for LSEG, `report_date` for surprises; per-field independent lookup; unit-tested. |
| **Learning-to-rank / LambdaRank with per-date query groups** | `LGBMRanker`, `group` = per-date sizes, linear `label_gain`, truncation 100. |
| **Ordinal relevance grades** | Per-date `qcut(sector_return, 5)`. |
| **Seed ensembling** | 8 seeds, averaged before rank transform. |
| **Cross-date EWMA rank smoothing** | Causal `alpha = 2/(span+1)` on stored percentile ranks. |
| **Rank turnover (stability proxy)** | Mean \|Δ percentile rank\| between consecutive dates. |
| **Feature decorrelation and standalone-signal gate** | `feature_diagnostics()`: mean \|corr\| vs the book + standalone block-bootstrapped SECB + efficiency-ratio tertile confound check. |
| **Regime/consolidation confound control** | Kaufman efficiency-ratio tertile split. |
| **Partially de-survivorshipped universe** | Accurate: removed-from-index names since 2016 are included, with their price history. |
| **Ablation study** | Drop-one and drop-block feature ablations under the same walk-forward. |

### ❌ Terms that would be overstating

| Don't say | Why | Say instead |
|---|---|---|
| "Survivorship-bias-free universe" | Inclusion side unhandled; removed cohort lacks metadata; delisting returns absent; pre-2016 is all survivors. | "Partially de-survivorshipped — I include names removed since 2016, and here's what's still missing." |
| "Purged and embargoed CPCV" | No combinatorial purged CV; the purge has no extra embargo. | "Purged expanding-window walk-forward." |
| "Backtest" / "backtested strategy" | No portfolio construction, no P&L, no positions. | "Walk-forward IC study." |
| "Sharpe ratio of X" | No strategy return series exists. | "IC of X and ICIR of Y." |
| "Net of transaction costs" / "cost-aware" | No cost model of any kind. | "I track a rank-turnover proxy but haven't modeled costs." |
| "Alpha" (unqualified) | The signal is size-dominated; it's a risk premium. | "Cross-sectional predictive signal, largely a size tilt at present." |
| "Calibrated probability" | `direction_prob` is a clipped percentile rank. The column name is misleading. | "Predicted relative percentile rank." |
| "Confidence interval on the prediction" | `predictions.confidence` is std of rank across ≤3 dates. | "Rank stability across recent scoring dates." |
| "Production trading system" | It's a dashboard; no orders, no broker, no execution. | "Production inference pipeline feeding a research dashboard." |
| "Statistically significant out-of-sample" (unqualified) | Config was selected on the same folds; no multiple-testing correction. | "Block-bootstrap significant per fold; not a clean held-out estimate of the promoted config." |
| "Deep learning model in production" | PatchTST is shelved and failed its (mis-specified) holdout. | "I built a PatchTST variant; it failed a holdout that I later realized was on a dead horizon, and I shelved it." |
| "Market-neutral" / "sector-neutral portfolio" | Only the *target* and *metric* are sector-relative; there is no portfolio. | "Sector-relative training target and within-sector evaluation metric." |
| "Real-time" | Monthly cross-section; daily batch pipeline at 17:30 local. | "Daily batch pipeline, monthly rebalance cadence." |
| "Factor model" | No factor exposures, no covariance matrix, no risk model. | "Cross-sectional ranking model on factor-style features." |
| "Point-in-time database" | yfinance history is rewritten on drift re-pulls; sector/shares are current values. | "Point-in-time feature joins on top of a current-value price store." |

### Suggested resume bullets

> **Cross-sectional equity ranking model (Python, LightGBM, Postgres).** Built an end-to-end research
> pipeline over ~600 US large caps and 1.8M daily bars (2010–2026): point-in-time factor construction
> from prices, SEC EDGAR filings (joined on filing receipt date) and LSEG analyst estimates; per-horizon
> LightGBM rankers with a LambdaRank objective aligned to the rank-IC scoring metric.

> **Validation.** Purged expanding-window walk-forward (purge = full label horizon) over 126–145 monthly
> folds. Because monthly folds with 6–12-month labels are heavily autocorrelated, significance uses a
> moving-block bootstrap with block length = horizon (reducing 126 folds to ~10 effective independent
> blocks at 1Y) plus an explicit minimum-detectable-IC power statement. Out-of-sample **within-sector
> rank IC +0.056 at 6M** (block-bootstrap t = 2.75) — graded within GICS sector so the result reflects
> stock selection rather than sector rotation.

> **Self-audit.** Ablation testing revealed that a fundamentals-availability indicator carried a +0.072
> standalone universe IC by acting as a proxy for future index removal, and that the within-sector
> signal is load-bearing on the size factor (dropping it removes significance at every horizon).
> Documented both, and redirected reporting to the within-sector metric, which is structurally immune
> to the first.

That third bullet is unusual and it will get you asked about. That's the point.

---

## Appendix A — reproducing the measurements in this document

```bash
source stockproject/bin/activate

# Build the panel once from the cached frames (~40s)
python - <<'PY'
import pickle
from backend.ml.dataset import build_calendar_grid
from backend.ml.gbm_baseline import prepare_panel
frames = pickle.load(open(".frame_cache/frames_all.pkl","rb"))
panel = prepare_panel(frames, build_calendar_grid(frames))
panel.to_pickle("/tmp/panel.pkl")
print(panel.shape)   # (79457, 114)
PY

# Standalone feature IC + drop-one ablations: see the scripts used for this audit under
#   <scratchpad>/audit2.py, leak_test.py, leak_test2.py, leak_test3.py
# Each full single-seed 6M walk-forward (138 folds) runs in ~25-30s.

# The project's own sweep CLI:
python -m backend.ml.gbm_baseline --horizon 6M --target sector_grade \
    --objective lambdarank --lambdarank-truncation 100 \
    --max-train-months 60 --n-seeds 8 --sector-neutral-ic --block-bootstrap-reps 2000
```

## Appendix B — file map for "show me the code"

| Claim | File / symbol |
|---|---|
| Walk-forward + purge | `backend/ml/gbm_baseline.py::walk_forward_folds`, `::walk_forward_ic` |
| Block bootstrap / power | `gbm_baseline.py::block_bootstrap_summary` |
| Within-sector IC | `gbm_baseline.py::within_sector_ic` |
| Production specs | `gbm_baseline.py::PRODUCTION_HORIZON_SPECS` |
| LambdaRank + query groups | `gbm_baseline.py::_fit_lgbm_ranker` |
| Feature gate | `gbm_baseline.py::feature_diagnostics` |
| Target definitions | `gbm_baseline.py::apply_target_modes` |
| Rank normalization | `gbm_baseline.py::rank_normalize_features` |
| Label construction | `backend/ml/dataset.py::compute_targets` |
| Monthly grid | `dataset.py::build_calendar_grid` |
| Universe query (no PIT membership filter) | `dataset.py::load_frames` |
| Row assembly + staleness drop | `backend/ml/factors/assembly.py::build_ticker_rows` |
| Price features | `factors/price.py::_price_features` |
| EDGAR PIT join | `factors/fundamentals.py::_fundamental_context_asof`, `backend/ml/features.py::_build_fundamental_series` |
| LSEG PIT join | `factors/estimates.py::_estimates_context_asof` |
| Look-ahead tests | `backend/tests/test_features_no_lookahead.py` (203 tests total in the suite) |
| Purge test | `backend/tests/test_gbm_baseline.py::test_walk_forward_folds_respect_embargo` |
| Planted-signal + null test | `test_gbm_baseline.py::test_walk_forward_recovers_planted_signal_and_null_does_not` |
| Production inference | `backend/ml/gbm_inference.py::fit_horizon_models`, `::score_current_cross_section` |
| Survivorship seeds | `scripts/seed_sp500.py`, `scripts/seed_sp500_historical.py` |
| Single-split holdout helper (not wired into the sweep CLI) | `gbm_baseline.py::single_split_ic`, used only by `compare_transformer_gbm.py` |
