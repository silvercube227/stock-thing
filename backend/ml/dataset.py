"""Training-sample assembly: DB rows → feature windows + horizon labels → splits.

`features.build_sample` produces one (252, 12) window; this module orchestrates
it across the whole universe, attaches labels, and carves a leak-safe
time-based split (plan §4 training regime).

Label definition: for a sample ending at trading day `t`, the label for horizon
H (21/63/126/252 trading days) is `sign(adj_close[t+H] - adj_close[t])`, i.e.
1 if the price is higher H trading days later, else 0. We shift by INDEX within
the ticker's own sorted price series — not the NYSE calendar — so the horizon is
exactly H *available* bars forward and stays self-consistent with the data we
hold. Horizons whose label date runs past the last available bar are MASKED
(weight 0 in the loss), which is also how recent samples in the holdout window
legitimately lack their 6M/1Y labels.

Split (by sample-end date, T = latest bar in the data):
    train:   sample_end <  T - 18 months
    val:     T - 18 months <= sample_end < T - 6 months
    holdout: sample_end >= T - 6 months
The 12-month val band sits between train and the 6-month holdout so a train
sample's 1Y label can't reach into the holdout inputs.

DB loading lives here (async); the assembly/label/split core is pure and tested
in tests/test_dataset.py with synthetic frames.
"""

from __future__ import annotations

import bisect
import hashlib
import math
import pickle
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np

from backend.config import get_settings
from backend.ingestion.calendar import HORIZON_TRADING_DAYS
from backend.ml.features import SEQUENCE_LENGTH, build_sample
from backend.ml.model import HORIZONS
from collections import defaultdict

Split = str  # "train" | "val" | "holdout"


# =============================================================
# Config + containers
# =============================================================


@dataclass
class SplitConfig:
    """Time-based split boundaries, in calendar months before the latest bar."""

    holdout_months: int = 6
    val_months: int = 18  # val spans (T-18mo, T-6mo]; train is everything earlier


@dataclass
class TickerFrame:
    """All rows for one ticker, as plain dicts (asyncpg Records coerced)."""

    ticker_id: int
    embedding_idx: int
    symbol: str
    prices: list[dict]        # trade_date, adj_close, volume
    fundamentals: list[dict]  # filed_at, period_end, filing_type, revenue, ...
    sentiment: list[dict]     # score_date, rolling_7d, rolling_14d
    shares_outstanding: int | None = None  # from tickers table; used for log_market_cap
    sector: str | None = None
    industry: str | None = None
    # analyst_estimates rows (as_of_date PIT); None tolerates pre-estimates pickled caches.
    estimates: list[dict] | None = None


@dataclass
class Sample:
    ticker_id: int
    embedding_idx: int
    sample_end: date
    features: np.ndarray              # (SEQUENCE_LENGTH, FEATURE_DIM) float32
    labels: dict[str, int]            # horizon -> 0/1 (meaningless where masked)
    returns: dict[str, float]         # horizon -> log-return target (0.0 where masked)
    mask: dict[str, bool]             # horizon -> target available?


# =============================================================
# Calendar-free month arithmetic (avoid a dateutil dependency)
# =============================================================


def months_before(d: date, months: int) -> date:
    """Date `months` calendar months before `d`, clamping the day if needed."""
    total = (d.year * 12 + (d.month - 1)) - months
    year, month = divmod(total, 12)
    month += 1
    # Clamp day for short months (e.g., Mar 31 - 1mo -> Feb 28/29).
    last_day = [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
                31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
    return date(year, month, min(d.day, last_day))


# =============================================================
# Label computation (index-shift on the ticker's own series)
# =============================================================


def compute_targets(
    adj_close: list[float], end_idx: int
) -> tuple[dict[str, int], dict[str, float], dict[str, bool]]:
    """Direction labels and log-return targets for each horizon at `end_idx`.

    `adj_close` is ascending by trade_date. For horizon H the target is
    `log(adj_close[end+H] / adj_close[end])` and the label is its sign. A horizon
    is available iff the bar `end_idx + H` exists with positive prices; otherwise
    it is masked (label 0, return 0.0).
    """
    base = adj_close[end_idx]
    labels: dict[str, int] = {}
    returns: dict[str, float] = {}
    mask: dict[str, bool] = {}
    n = len(adj_close)
    for h in HORIZONS:
        j = end_idx + HORIZON_TRADING_DAYS[h]
        future = adj_close[j] if j < n else None
        if future is not None and base is not None and base > 0 and future > 0:
            r = math.log(future / base)
            labels[h] = 1 if r > 0 else 0
            returns[h] = r
            mask[h] = True
        else:
            labels[h] = 0
            returns[h] = 0.0
            mask[h] = False
    return labels, returns, mask


# =============================================================
# Sample assembly (pure)
# =============================================================

def build_calendar_grid(frames: TickerFrame):
    #make set of all dates
    all_dates: set[date] = set()
    #iterate through frames, add each date with a non none close 
    for f in frames:
        for r in f.prices:
            if r["adj_close"] is not None:
                all_dates.add(_as_date(r["trade_date"]))
    #dictionary of every trading month and year as tuple, plus the list for that month 
    by_month: dict[tuple[int, int], list[date]] = defaultdict(list)
    for d in all_dates:
        by_month[(d.year, d.month)].append(d)
    #find largest date per month as the end point
    return sorted(max(v) for v in by_month.values())


def assemble_ticker_samples(frame: TickerFrame, stride: int = 1) -> list[Sample]:
    """Build every valid (window, labels) sample for one ticker.

    A sample is emitted for each trade date with a full 252-day history behind
    it AND at least one available horizon label. `stride` subsamples end dates
    (stride>1 thins label autocorrelation and speeds up dataset builds).
    """
    prices = sorted(frame.prices, key=lambda r: _as_date(r["trade_date"]))
    if len(prices) <= SEQUENCE_LENGTH:
        return []
    adj_close = [
        float(r["adj_close"]) if r["adj_close"] is not None else None for r in prices
    ]

    samples: list[Sample] = []
    for end_idx in range(SEQUENCE_LENGTH, len(prices), stride):
        sample_end = _as_date(prices[end_idx]["trade_date"])
        feats = build_sample(
            frame.ticker_id, sample_end, prices, frame.fundamentals, frame.sentiment
        )
        if feats is None:
            continue
        labels, returns, mask = compute_targets(adj_close, end_idx)
        if not any(mask.values()):
            continue
        samples.append(
            Sample(
                ticker_id=frame.ticker_id,
                embedding_idx=frame.embedding_idx,
                sample_end=sample_end,
                features=feats,
                labels=labels,
                returns=returns,
                mask=mask,
            )
        )
    return samples


def assemble_samples(frames: list[TickerFrame], stride: int = 1) -> list[Sample]:
    out: list[Sample] = []
    for frame in frames:
        out.extend(assemble_ticker_samples(frame, stride=stride))
    return out


def assemble_ticker_samples_aligned(frame: TickerFrame, grid: list[date]) -> list[Sample]:
    """Build one sample per grid date for one ticker.

    Uses the ticker's last bar at or before each grid date. sample_end is set to
    the grid date (not the actual bar date) so all tickers on the same month-end
    land in the same cross-section for rank-IC grouping.
    """
    prices = sorted(frame.prices, key=lambda r: _as_date(r["trade_date"]))
    if len(prices) <= SEQUENCE_LENGTH:
        return []
    trade_dates = [_as_date(r["trade_date"]) for r in prices]
    adj_close = [
        float(r["adj_close"]) if r["adj_close"] is not None else None for r in prices
    ]
    samples: list[Sample] = []
    for g in grid:
        pos = bisect.bisect_right(trade_dates, g) - 1
        if pos < SEQUENCE_LENGTH:
            continue
        feats = build_sample(
            frame.ticker_id, trade_dates[pos], prices, frame.fundamentals, frame.sentiment
        )
        if feats is None:
            continue
        labels, returns, mask = compute_targets(adj_close, pos)
        if not any(mask.values()):
            continue
        samples.append(
            Sample(
                ticker_id=frame.ticker_id,
                embedding_idx=frame.embedding_idx,
                sample_end=g,
                features=feats,
                labels=labels,
                returns=returns,
                mask=mask,
            )
        )
    return samples


def assemble_calendar_aligned(frames: list[TickerFrame], grid: list[date]) -> list[Sample]:
    out: list[Sample] = []
    for frame in frames:
        out.extend(assemble_ticker_samples_aligned(frame, grid))
    return out


# =============================================================
# Cross-sectional relabeling (relative-to-universe target)
# =============================================================


def cross_sectional_medians(
    frames: list[TickerFrame],
) -> dict[str, dict[date, float]]:
    """Per-horizon cross-sectional median log-return, keyed by end date.

    Built densely (every trading day, every ticker) from a wide adj_close panel so
    a strided sample can look up the universe median at its own end date — the
    demeaning reference for cross-sectional labels. The H-day forward return uses a
    panel row shift, which equals an H-bar shift on each ticker's own series since
    all tickers share the NYSE calendar. Returns {horizon: {end_date: median}}.
    """
    import pandas as pd

    series = {
        f.ticker_id: pd.Series(
            {_as_date(r["trade_date"]): float(r["adj_close"])
             for r in f.prices if r["adj_close"] is not None}
        )
        for f in frames if f.prices
    }
    panel = pd.DataFrame(series).sort_index()        # index = date, cols = ticker
    medians: dict[str, dict[date, float]] = {}
    for h in HORIZONS:
        H = HORIZON_TRADING_DAYS[h]
        logret = np.log(panel.shift(-H) / panel)     # H trading days forward
        med = logret.median(axis=1, skipna=True)     # cross-sectional median per date
        medians[h] = {d: float(v) for d, v in med.items() if pd.notna(v)}
    return medians


def relabel_cross_sectional(
    samples: list[Sample], medians: dict[str, dict[date, float]]
) -> list[Sample]:
    """Rewrite labels/returns in place to be relative to the universe.

    For each unmasked (sample, horizon): the return target becomes the demeaned
    (relative) log-return `r - median`, and the direction label becomes "did this
    ticker beat the universe median?" (1/0). Samples whose end date has no median
    are masked for that horizon. Labels are ~50/50 by construction, so class
    weighting becomes a near no-op.
    """
    for s in samples:
        for h in HORIZONS:
            if not s.mask[h]:
                continue
            med = medians[h].get(s.sample_end)
            if med is None:
                s.mask[h] = False
                s.labels[h], s.returns[h] = 0, 0.0
                continue
            rel = s.returns[h] - med
            s.returns[h] = rel
            s.labels[h] = 1 if rel > 0 else 0
    return samples


# =============================================================
# Time-based split
# =============================================================


def latest_sample_end(samples: list[Sample]) -> date:
    return max(s.sample_end for s in samples)


def split_samples(
    samples: list[Sample], cfg: SplitConfig | None = None, T: date | None = None
) -> dict[Split, list[Sample]]:
    """Partition samples into train/val/holdout by sample_end (see module docs)."""
    if not samples:
        return {"train": [], "val": [], "holdout": []}
    cfg = cfg or SplitConfig()
    T = T or latest_sample_end(samples)
    holdout_start = months_before(T, cfg.holdout_months)
    val_start = months_before(T, cfg.val_months)

    out: dict[Split, list[Sample]] = {"train": [], "val": [], "holdout": []}
    for s in samples:
        if s.sample_end >= holdout_start:
            out["holdout"].append(s)
        elif s.sample_end >= val_start:
            out["val"].append(s)
        else:
            out["train"].append(s)
    return out


# =============================================================
# Tensor conversion
# =============================================================


def to_arrays(samples: list[Sample]) -> dict[str, np.ndarray]:
    """Stack samples into numpy arrays for the DataLoader.

    Returns:
        x:          (N, 252, 12) float32
        ticker_idx: (N,)         int64
        y:          (N, 4)       int64   (direction labels; horizon order = HORIZONS)
        r:          (N, 4)       float32 (log-return targets; 0.0 where masked)
        mask:       (N, 4)       float32 (1.0 valid, 0.0 masked)
    """
    n = len(samples)
    x = np.empty((n, SEQUENCE_LENGTH, samples[0].features.shape[1]), dtype=np.float32) if n else \
        np.empty((0, SEQUENCE_LENGTH, 0), dtype=np.float32)
    ticker_idx = np.empty(n, dtype=np.int64)
    y = np.empty((n, len(HORIZONS)), dtype=np.int64)
    r = np.empty((n, len(HORIZONS)), dtype=np.float32)
    mask = np.empty((n, len(HORIZONS)), dtype=np.float32)
    for i, s in enumerate(samples):
        x[i] = s.features
        ticker_idx[i] = s.embedding_idx
        for j, h in enumerate(HORIZONS):
            y[i, j] = s.labels[h]
            r[i, j] = s.returns[h]
            mask[i, j] = 1.0 if s.mask[h] else 0.0
    return {"x": x, "ticker_idx": ticker_idx, "y": y, "r": r, "mask": mask}


# =============================================================
# DB loading (async)
# =============================================================


_PRICE_SQL = """
select ticker_id, trade_date, adj_close, volume
  from price_history
 where ticker_id = any($1::bigint[])
 order by ticker_id, trade_date
"""

_FUND_SQL = """
select ticker_id, filed_at, period_end, filing_type, revenue, net_income, gross_margin,
       operating_margin, total_debt, total_equity, fcf
  from fundamentals
 where ticker_id = any($1::bigint[])
 order by ticker_id, filed_at
"""

_SENT_SQL = """
select ticker_id, score_date, rolling_7d, rolling_14d
  from sentiment_daily
 where ticker_id = any($1::bigint[])
 order by ticker_id, score_date
"""

_EST_SQL = """
select ticker_id, as_of_date, rec_mean, price_target_mean,
       revenue_mean, revenue_actual, fwd_pe, fwd_ev_ebitda
  from analyst_estimates
 where ticker_id = any($1::bigint[])
 order by ticker_id, as_of_date
"""


async def load_frames(pool, symbols: list[str] | None = None) -> list[TickerFrame]:
    """Pull all training data from Supabase into per-ticker frames.

    If `symbols` is given, restrict to those (used for the single-ticker overfit
    sanity check); otherwise load every active ticker.
    """
    if symbols:
        ticker_rows = await pool.fetch(
            "select ticker_id, symbol, embedding_idx, shares_outstanding, sector, industry "
            "from tickers "
            "where symbol = any($1::text[]) order by ticker_id",
            symbols,
        )
    else:
        ticker_rows = await pool.fetch(
            "select ticker_id, symbol, embedding_idx, shares_outstanding, sector, industry "
            "from tickers order by ticker_id"
        )
    ids = [r["ticker_id"] for r in ticker_rows]
    if not ids:
        return []

    price_rows = await pool.fetch(_PRICE_SQL, ids)
    fund_rows = await pool.fetch(_FUND_SQL, ids)
    sent_rows = await pool.fetch(_SENT_SQL, ids)
    est_rows = await pool.fetch(_EST_SQL, ids)

    by_ticker_prices = _group(price_rows)
    by_ticker_fund = _group(fund_rows)
    by_ticker_sent = _group(sent_rows)
    by_ticker_est = _group(est_rows)

    frames: list[TickerFrame] = []
    for r in ticker_rows:
        tid = r["ticker_id"]
        frames.append(
            TickerFrame(
                ticker_id=tid,
                embedding_idx=r["embedding_idx"],
                symbol=r["symbol"],
                prices=by_ticker_prices.get(tid, []),
                fundamentals=by_ticker_fund.get(tid, []),
                sentiment=by_ticker_sent.get(tid, []),
                shares_outstanding=r["shares_outstanding"],
                sector=r["sector"],
                industry=r["industry"],
                estimates=by_ticker_est.get(tid, []),
            )
        )
    return frames


def _frame_cache_key(symbols: list[str] | None) -> str:
    """Stable cache filename stem for a symbol set ("all" when unrestricted)."""
    if not symbols:
        return "all"
    digest = hashlib.sha1(",".join(sorted(symbols)).encode()).hexdigest()[:16]
    return f"sym-{digest}"


async def load_frames_cached(
    pool, symbols: list[str] | None = None, refresh: bool = False
) -> list[TickerFrame]:
    """load_frames() with an on-disk pickle cache, for local experiments only.

    Every walk-forward / sweep / backtest invocation otherwise re-pulls the full
    price+fundamentals+sentiment history through the Supabase pooler (~tens of MB
    each) — the dominant Shared-Pooler egress source. Historical frames are
    static, so we cache them per symbol set under `settings.frame_cache_dir` and
    reuse on subsequent runs. Pass `refresh=True` after ingesting new data to
    re-pull and overwrite the cache.

    Production inference (`gbm_inference.py`) deliberately does NOT use this — it
    calls `load_frames` so it always scores on fresh data.
    """
    cache_dir = get_settings().frame_cache_dir
    path = cache_dir / f"frames_{_frame_cache_key(symbols)}.pkl"
    if path.exists() and not refresh:
        with path.open("rb") as f:
            frames = pickle.load(f)
        print(f"[frame-cache] hit: {path} ({len(frames)} tickers) — no Supabase egress")
        return frames

    why = "refresh" if refresh else "miss"
    print(f"[frame-cache] {why}: pulling full history from Supabase ...")
    frames = await load_frames(pool, symbols=symbols)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump(frames, f)
    tmp.replace(path)  # atomic: never leave a half-written cache file
    print(f"[frame-cache] wrote {path} ({len(frames)} tickers)")
    return frames


def _group(rows) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for r in rows:
        out.setdefault(r["ticker_id"], []).append(dict(r))
    return out


def _as_date(val) -> date:
    # datetime subclasses date, so check datetime first (cf. features._to_date).
    return val.date() if isinstance(val, datetime) else val
