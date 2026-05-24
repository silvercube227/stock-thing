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

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np

from backend.ingestion.calendar import HORIZON_TRADING_DAYS
from backend.ml.features import SEQUENCE_LENGTH, build_sample
from backend.ml.model import HORIZONS

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


@dataclass
class Sample:
    ticker_id: int
    embedding_idx: int
    sample_end: date
    features: np.ndarray              # (SEQUENCE_LENGTH, FEATURE_DIM) float32
    labels: dict[str, int]            # horizon -> 0/1 (meaningless where masked)
    mask: dict[str, bool]             # horizon -> label available?


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


def compute_labels(adj_close: list[float], end_idx: int) -> tuple[dict[str, int], dict[str, bool]]:
    """Direction labels for each horizon at series position `end_idx`.

    `adj_close` is ascending by trade_date. A horizon is available iff the bar
    `end_idx + H` exists; otherwise it is masked.
    """
    base = adj_close[end_idx]
    labels: dict[str, int] = {}
    mask: dict[str, bool] = {}
    n = len(adj_close)
    for h in HORIZONS:
        j = end_idx + HORIZON_TRADING_DAYS[h]
        if j < n and adj_close[j] is not None and base is not None:
            labels[h] = 1 if adj_close[j] > base else 0
            mask[h] = True
        else:
            labels[h] = 0
            mask[h] = False
    return labels, mask


# =============================================================
# Sample assembly (pure)
# =============================================================


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
        labels, mask = compute_labels(adj_close, end_idx)
        if not any(mask.values()):
            continue
        samples.append(
            Sample(
                ticker_id=frame.ticker_id,
                embedding_idx=frame.embedding_idx,
                sample_end=sample_end,
                features=feats,
                labels=labels,
                mask=mask,
            )
        )
    return samples


def assemble_samples(frames: list[TickerFrame], stride: int = 1) -> list[Sample]:
    out: list[Sample] = []
    for frame in frames:
        out.extend(assemble_ticker_samples(frame, stride=stride))
    return out


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
        y:          (N, 4)       int64   (horizon order = HORIZONS)
        mask:       (N, 4)       float32 (1.0 valid, 0.0 masked)
    """
    n = len(samples)
    x = np.empty((n, SEQUENCE_LENGTH, samples[0].features.shape[1]), dtype=np.float32) if n else \
        np.empty((0, SEQUENCE_LENGTH, 0), dtype=np.float32)
    ticker_idx = np.empty(n, dtype=np.int64)
    y = np.empty((n, len(HORIZONS)), dtype=np.int64)
    mask = np.empty((n, len(HORIZONS)), dtype=np.float32)
    for i, s in enumerate(samples):
        x[i] = s.features
        ticker_idx[i] = s.embedding_idx
        for j, h in enumerate(HORIZONS):
            y[i, j] = s.labels[h]
            mask[i, j] = 1.0 if s.mask[h] else 0.0
    return {"x": x, "ticker_idx": ticker_idx, "y": y, "mask": mask}


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
select ticker_id, filed_at, period_end, filing_type, revenue, gross_margin,
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


async def load_frames(pool, symbols: list[str] | None = None) -> list[TickerFrame]:
    """Pull all training data from Supabase into per-ticker frames.

    If `symbols` is given, restrict to those (used for the single-ticker overfit
    sanity check); otherwise load every active ticker.
    """
    if symbols:
        ticker_rows = await pool.fetch(
            "select ticker_id, symbol, embedding_idx from tickers "
            "where active and symbol = any($1::text[]) order by ticker_id",
            symbols,
        )
    else:
        ticker_rows = await pool.fetch(
            "select ticker_id, symbol, embedding_idx from tickers "
            "where active order by ticker_id"
        )
    ids = [r["ticker_id"] for r in ticker_rows]
    if not ids:
        return []

    price_rows = await pool.fetch(_PRICE_SQL, ids)
    fund_rows = await pool.fetch(_FUND_SQL, ids)
    sent_rows = await pool.fetch(_SENT_SQL, ids)

    by_ticker_prices = _group(price_rows)
    by_ticker_fund = _group(fund_rows)
    by_ticker_sent = _group(sent_rows)

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
            )
        )
    return frames


def _group(rows) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for r in rows:
        out.setdefault(r["ticker_id"], []).append(dict(r))
    return out


def _as_date(val) -> date:
    # datetime subclasses date, so check datetime first (cf. features._to_date).
    return val.date() if isinstance(val, datetime) else val
