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
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from backend.config import get_settings
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
    shares_outstanding: int | None = None  # from tickers table; used for log_market_cap
    sector: str | None = None
    industry: str | None = None
    # analyst_estimates rows (as_of_date PIT); None tolerates pre-estimates pickled caches.
    estimates: list[dict] | None = None
    # earnings_surprises rows (quarterly, report_date PIT); None tolerates older caches.
    surprises: list[dict] | None = None
    # short_interest rows (publication_date PIT); None when table not yet populated.
    short_interest: list[dict] | None = None
    # insider_transactions rows (filing_date PIT); None tolerates older pickled caches.
    insiders: list[dict] | None = None
    # index_membership intervals [{valid_from, valid_to}] (valid_to exclusive, null =
    # current). None means "not loaded" (pre-migration cache) and is distinct from []
    # ("loaded; never a member") — the membership filter has to tell those apart.
    membership: list[dict] | None = None
    # sector_history intervals [{valid_from, valid_to, sector, industry}] (valid_to
    # exclusive, null = current). None = table not loaded; build_ticker_rows then
    # falls back to the static `sector`/`industry` above.
    sector_history: list[dict] | None = None
    # tickers.removed_at — the date this name left the index (None = still a member).
    # Used to (a) detect symbol REUSE at load time and (b) tell a delisted series
    # apart from one that is merely right-censored by the panel's end date.
    removed_at: date | None = None
    security_events: list[dict] | None = None
    accounting_facts: list[dict] | None = None
    fixed_estimates: list[dict] | None = None
    news: list[dict] | None = None
    news_coverage: list[dict] | None = None
    price_source_exclusions: dict | None = None
    price_history_start: date | None = None
    price_history_start_source: dict | None = None


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
    adj_close: list[float], end_idx: int, terminal: bool = False
) -> tuple[dict[str, int], dict[str, float], dict[str, bool]]:
    """Direction labels and log-return targets for each horizon at `end_idx`.

    `adj_close` is ascending by trade_date. Features are computed from the close of
    bar `end_idx`, so the position cannot be entered until the NEXT bar: the return
    runs `log(adj_close[end+1+H] / adj_close[end+1])`. Without that one-bar
    implementation lag the label would start at the very close the signal is
    computed from, which assumes observing a close and trading at it simultaneously.

    A horizon is available iff both bar `end_idx + 1` (the entry) and bar
    `end_idx + 1 + H` (the exit) exist with positive prices; otherwise it is masked
    (label 0, return 0.0).

    `terminal` is a deprecated compatibility argument and cannot establish proceeds.
    Missing exits remain masked. GBM assembly uses documented_targets for explicit
    corporate-action outcomes and actual execution/realization dates.
    """
    labels: dict[str, int] = {}
    returns: dict[str, float] = {}
    mask: dict[str, bool] = {}
    n = len(adj_close)
    entry_idx = end_idx + 1
    base = adj_close[entry_idx] if entry_idx < n else None
    for h in HORIZONS:
        j = entry_idx + HORIZON_TRADING_DAYS[h]
        if j < n:
            future = adj_close[j]
        else:
            future = None
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

    The shifts mirror `compute_targets`' one-bar implementation lag: entry is the
    bar AFTER the sample date, so the reference return runs from shift(-1) to
    shift(-(H+1)). Keeping the two in step matters because `r_h - median` would
    otherwise mix a next-bar base against a same-bar reference.
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
        # Entry at the next bar, exit H bars later (one-bar implementation lag).
        logret = np.log(panel.shift(-(H + 1)) / panel.shift(-1))
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


# Yahoo `close` is split-adjusted. Assembly reconstructs the as-traded price
# using explicit split/source metadata before multiplying by dated shares.
_PRICE_SQL = """
select ticker_id, trade_date, close, adj_close, volume, high, low, dividend,
       split_factor, source, to_jsonb(price_history)->>'price_basis' as price_basis,
       (to_jsonb(price_history)->>'share_split_factor')::numeric as share_split_factor,
       to_jsonb(price_history)->>'share_action_source' as share_action_source
  from price_history
 where ticker_id = any($1::bigint[])
 order by ticker_id, trade_date
"""

_FUND_SQL = """
select ticker_id, filed_at, period_end, filing_type, revenue, net_income, gross_margin,
       operating_margin, total_debt, total_equity, fcf, shares_outstanding,
       (to_jsonb(fundamentals)->>'shares_measured_at')::date as shares_measured_at,
       to_jsonb(fundamentals)->>'shares_concept' as shares_concept,
       to_jsonb(fundamentals)->>'shares_kind' as shares_kind,
       to_jsonb(fundamentals)->>'shares_basis' as shares_basis
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
       revenue_mean, revenue_actual, eps_mean, eps_actual, fwd_pe, fwd_ev_ebitda,
       num_analysts, pt_num_estimates, eps_std_dev, eps_num_inc_estimates
  from analyst_estimates
 where ticker_id = any($1::bigint[])
 order by ticker_id, as_of_date
"""

_SURPRISE_SQL = """
select ticker_id, period_end, report_date,
       eps_consensus, eps_actual, rev_consensus, rev_actual
  from earnings_surprises
 where ticker_id = any($1::bigint[])
 order by ticker_id, report_date
"""

_SI_SQL = """
select ticker_id, settlement_date, publication_date,
       short_interest, avg_daily_volume, days_to_cover
  from short_interest
 where ticker_id = any($1::bigint[])
 order by ticker_id, publication_date
"""

_INSIDER_SQL = """
select ticker_id, accession_number, transaction_idx, insider_cik,
       is_officer, is_director, is_ten_pct_owner, filing_date,
       transaction_code, shares, price_per_share, value
  from insider_transactions
 where ticker_id = any($1::bigint[])
 order by ticker_id, filing_date
"""

_MEMBERSHIP_SQL = """
select ticker_id, valid_from, valid_to,
       coalesce(to_jsonb(index_membership)->>'index_id', 'SPX') as index_id
  from index_membership
 where ticker_id = any($1::bigint[])
 order by ticker_id, valid_from
"""

_SECTOR_HISTORY_SQL = """
select ticker_id, valid_from, valid_to, sector, industry
  from sector_history
 where ticker_id = any($1::bigint[])
 order by ticker_id, valid_from
"""


# A delisted ticker's symbol can be REASSIGNED to an unrelated company years later
# (SE -> Sea Limited, EMC, APC, INFO, STI, CA, SPLS, ...). yfinance then returns the
# NEW company's bars under the old ticker_id. Those rows sit outside every index
# cross-section (the membership interval closed at removal) but still feed
# `build_universe_return_map` (the market-return series behind beta / residual
# momentum / the beta_resid target) and `cross_sectional_medians` (the demean
# baseline for every label) — so a handful of wrong-company series quietly moves the
# benchmark for the whole panel.
#
# The tell is a GAP, not the removal date itself: names that keep trading after
# leaving the index (FOSL, GME, AA, RIG ...) are exactly the de-survivorship data the
# panel wants and trade continuously through their removal. Reuse always shows up as
# a series that either starts long after removal or resumes after a multi-quarter
# hole. Truncate at that discontinuity; leave continuous series untouched.
_REUSE_MIN_GAP_DAYS = 180
_REUSE_MIN_LAG_DAYS = 90


def _drop_reused_symbol_bars(
    price_rows: list[dict],
    removed_at: date | None,
    security_retired_at: date | None = None,
) -> list[dict]:
    """Drop bars that belong to a different company trading under a reused symbol.

    Two independent signals. `security_retired_at` is the vendor's date for the
    security ceasing to exist, and it is authoritative: Starwood, Rockwell Collins
    and Harman have bars running six months, two years and five years past their
    retirement with no gap at all, so the shape-based rule below cannot see them.
    `removed_at` only records leaving the index, which is not the end of a listing
    (TechnipFMC left in 2021 and still trades), so it remains a heuristic applied
    to the series shape.
    """
    if not price_rows:
        return price_rows
    if security_retired_at is not None:
        limit = security_retired_at
        kept = [r for r in price_rows if _as_date(r["trade_date"]) <= limit]
        if len(kept) != len(price_rows):
            price_rows = kept
            if not price_rows:
                return price_rows
    if removed_at is None:
        return price_rows
    cutoff = removed_at + timedelta(days=_REUSE_MIN_LAG_DAYS)
    prev: date | None = None
    for i, row in enumerate(price_rows):
        d = _as_date(row["trade_date"])
        gap_days = (d - prev).days if prev is not None else None
        # Either the series begins well after removal (i == 0), or it resumes after a
        # hole no live listing would leave. Everything from here on is another issuer.
        if d > cutoff and (gap_days is None or gap_days > _REUSE_MIN_GAP_DAYS):
            return price_rows[:i]
        prev = d
    return price_rows


async def _fetch_chunked(pool, sql: str, ids: list[int], chunk: int = 100) -> list:
    """Run an `ids = any($1)` query in ticker-id batches and concatenate results.

    A single full-history fetch can exceed the Supabase session pooler's transfer
    limit and get dropped (ConnectionDoesNotExistError); batching keeps each
    transfer small. Whole tickers stay within one batch so per-ticker row order
    (and thus _group) is preserved.
    """
    out: list = []
    for i in range(0, len(ids), chunk):
        out.extend(await pool.fetch(sql, ids[i:i + chunk]))
    return out


def _price_source_exclusions(value):
    """Parse reviewed exclusions strictly; malformed evidence must stop loading."""
    import json

    if value is None:
        return {}
    exclusions = json.loads(value) if isinstance(value, str) else value
    if not isinstance(exclusions, dict):
        raise ValueError('Price source exclusions must be an evidence mapping')
    for source, evidence in exclusions.items():
        if (not isinstance(source, str) or not source or not isinstance(evidence, dict)
                or not isinstance(evidence.get('reason'), str) or not evidence['reason']
                or not isinstance(evidence.get('artifact'), str) or not evidence['artifact']
                or not isinstance(evidence.get('sha256'), str) or len(evidence['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in evidence['sha256'])):
            raise ValueError('Price source exclusion lacks audit provenance')
    return exclusions


def _exclude_price_sources(rows, exclusions):
    return [r for r in rows if r.get('source') not in exclusions] if exclusions else rows


def _price_history_start(value, source):
    if value is None and source is None:
        return None, None
    if value is None or source is None:
        raise ValueError('Price history start requires paired evidence')
    import json
    evidence = json.loads(source) if isinstance(source, str) else source
    # Same strict artifact/hash/reason requirements as source quarantines.
    _price_source_exclusions({'history_start':evidence})
    start = date.fromisoformat(value) if isinstance(value,str) else _as_date(value)
    if not isinstance(start,date):
        raise ValueError('Invalid price history start date')
    return start, evidence


def _trim_predecessor_prices(rows, start):
    return [r for r in rows if _as_date(r['trade_date']) >= start] if start else rows


async def load_frames(pool, symbols: list[str] | None = None,
                      index_ids: tuple[str, ...] = ("SPX",),
                      include_research: bool = False) -> list[TickerFrame]:
    """Pull all training data from Supabase into per-ticker frames.

    If `symbols` is given, restrict to those (used for the single-ticker overfit
    sanity check); otherwise load every active ticker.
    """
    if symbols:
        ticker_rows = await pool.fetch(
            "select ticker_id, symbol, embedding_idx, shares_outstanding, sector, "
            "industry, removed_at, "
            "(to_jsonb(tickers)->>'security_retired_at') security_retired_at, "
            "(to_jsonb(tickers)->>'price_history_start') price_history_start, "
            "(to_jsonb(tickers)->>'price_history_start_source') price_history_start_source, "
            "(to_jsonb(tickers)->>'price_source_exclusions') price_source_exclusions "
            "from tickers "
            "where symbol = any($1::text[]) "
            "and (coalesce((to_jsonb(tickers)->>'research_only')::boolean,false)=false "
            "or $2::boolean) order by ticker_id",
            symbols,
            include_research,
        )
    else:
        ticker_rows = await pool.fetch(
            "select ticker_id, symbol, embedding_idx, shares_outstanding, sector, "
            "industry, removed_at, "
            "(to_jsonb(tickers)->>'security_retired_at') security_retired_at, "
            "(to_jsonb(tickers)->>'price_history_start') price_history_start, "
            "(to_jsonb(tickers)->>'price_history_start_source') price_history_start_source, "
            "(to_jsonb(tickers)->>'price_source_exclusions') price_source_exclusions from tickers "
            "where coalesce((to_jsonb(tickers)->>'research_only')::boolean,false)=false "
            "or $1::boolean order by ticker_id", include_research,
        )
    ids = [r["ticker_id"] for r in ticker_rows]
    if not ids:
        return []
    exclusions_by_id = {
        r['ticker_id']: _price_source_exclusions(r.get('price_source_exclusions'))
        for r in ticker_rows
    }
    starts_by_id = {r['ticker_id']:_price_history_start(r.get('price_history_start'),
                    r.get('price_history_start_source')) for r in ticker_rows}
    _removed_at_by_id = {
        r["ticker_id"]: (_as_date(r["removed_at"]) if r["removed_at"] else None)
        for r in ticker_rows
    }
    # Read through to_jsonb so a database without migration 020 still loads; the
    # value therefore arrives as text.
    _retired_at_by_id = {
        r["ticker_id"]: (
            date.fromisoformat(r["security_retired_at"])
            if r["security_retired_at"]
            else None
        )
        for r in ticker_rows
    }

    # Fetch in ticker-id batches: the full price pull (1.6M rows) is large enough
    # that the Supabase session pooler drops the single transfer mid-stream. Each
    # ticker's rows stay contiguous within a batch, so _group ordering is intact.
    price_rows = await _fetch_chunked(pool, _PRICE_SQL, ids)
    fund_rows = await _fetch_chunked(pool, _FUND_SQL, ids)
    sent_rows = await _fetch_chunked(pool, _SENT_SQL, ids)
    est_rows = await _fetch_chunked(pool, _EST_SQL, ids)
    surprise_rows = await _fetch_chunked(pool, _SURPRISE_SQL, ids)
    # short_interest table is optional (migration 009). Tolerate its absence so
    # existing load_frames callers work before the migration is applied.
    try:
        si_rows = await _fetch_chunked(pool, _SI_SQL, ids)
    except Exception:  # noqa: BLE001 — table may not exist yet
        si_rows = []
    # insider_transactions is optional (migration 010). Tolerate its absence so
    # existing load_frames callers work before the migration is applied.
    try:
        ins_rows = await _fetch_chunked(pool, _INSIDER_SQL, ids)
    except Exception:  # noqa: BLE001 — table may not exist yet
        ins_rows = []
    # index_membership is optional (migration 012). `membership_loaded` distinguishes
    # "no intervals for this ticker" from "the table isn't there", which the
    # membership filter must not confuse — the first excludes a name, the second
    # would silently empty the whole panel.
    membership_loaded = True
    try:
        membership_rows = await _fetch_chunked(pool, _MEMBERSHIP_SQL, ids)
    except Exception:  # noqa: BLE001 — table may not exist yet
        membership_rows = []
        membership_loaded = False
    # Missing table retains legacy compatibility; a successfully loaded table
    # with no intervals means UNKNOWN, never today's sector backfilled in time.
    sector_history_loaded = True
    import asyncpg
    try:
        if hasattr(pool, 'transaction'):
            async with pool.transaction():
                sector_history_rows = await _fetch_chunked(pool, _SECTOR_HISTORY_SQL, ids)
        else:
            sector_history_rows = await _fetch_chunked(pool, _SECTOR_HISTORY_SQL, ids)
    except asyncpg.UndefinedTableError:
        sector_history_rows = []
        sector_history_loaded = False

    by_ticker_prices = {
        tid: _drop_reused_symbol_bars(
            _trim_predecessor_prices(_exclude_price_sources(rows, exclusions_by_id[tid]),
                                     starts_by_id[tid][0]),
            _removed_at_by_id.get(tid), _retired_at_by_id.get(tid)
        )
        for tid, rows in _group(price_rows).items()
    }
    by_ticker_fund = _group(fund_rows)
    by_ticker_sent = _group(sent_rows)
    by_ticker_est = _group(est_rows)
    by_ticker_surprise = _group(surprise_rows)
    by_ticker_si = _group(si_rows)
    by_ticker_ins = _group(ins_rows)
    by_ticker_membership = _group(membership_rows)
    by_ticker_membership = {
        tid: [r for r in rows if r.get("index_id", "SPX") in index_ids]
        for tid, rows in by_ticker_membership.items()
    }
    by_ticker_sector_history = _group(sector_history_rows)

    optional = {}
    for attribute, table in (("security_events", "security_events"),
                             ("accounting_facts", "accounting_facts"),
                             ("fixed_estimates", "fixed_estimates"),
                             ("news", "research_news_daily"),
                             ("news_coverage", "research_news_coverage")):
        import asyncpg
        try:
            sql = f"select * from {table} where ticker_id = any($1::bigint[])"
            if hasattr(pool, "transaction"):
                async with pool.transaction():
                    optional[attribute] = _group(await _fetch_chunked(pool, sql, ids))
            else:
                optional[attribute] = _group(await _fetch_chunked(pool, sql, ids))
        except asyncpg.UndefinedTableError:
            optional[attribute] = {}

    frames: list[TickerFrame] = []
    for r in ticker_rows:
        tid = r["ticker_id"]
        frames.append(
            TickerFrame(
                ticker_id=tid,
                **{key: grouped.get(tid, []) for key, grouped in optional.items()},
                embedding_idx=r["embedding_idx"],
                symbol=r["symbol"],
                prices=by_ticker_prices.get(tid, []),
                price_source_exclusions=exclusions_by_id[tid],
                price_history_start=starts_by_id[tid][0],
                price_history_start_source=starts_by_id[tid][1],
                fundamentals=by_ticker_fund.get(tid, []),
                sentiment=by_ticker_sent.get(tid, []),
                shares_outstanding=r["shares_outstanding"],
                sector=r["sector"],
                industry=r["industry"],
                estimates=by_ticker_est.get(tid, []),
                surprises=by_ticker_surprise.get(tid, []),
                short_interest=by_ticker_si.get(tid, []),
                insiders=by_ticker_ins.get(tid, []),
                membership=(by_ticker_membership.get(tid, [])
                            if membership_loaded else None),
                sector_history=(by_ticker_sector_history.get(tid, [])
                                if sector_history_loaded else None),
                removed_at=_as_date(r["removed_at"]) if r["removed_at"] else None,
            )
        )
    return frames


FRAME_CACHE_VERSION = 8


class IncompatibleFrameCache(ValueError):
    """A caller must reload data instead of supplying missing correctness fields."""


def _frame_cache_key(symbols: list[str] | None, universe_tag: str = "sp500") -> str:
    """Stable cache filename stem for a symbol set ("all" when unrestricted)."""
    if not symbols:
        return "all" if universe_tag == "sp500" else f"all_{universe_tag}"
    digest = hashlib.sha1(",".join(sorted(symbols)).encode()).hexdigest()[:16]
    return f"sym-{digest}" + (f"_{universe_tag}" if universe_tag != "sp500" else "")


def write_frames_cache(frames: list[TickerFrame], symbols: list[str] | None = None,
                       universe_tag: str = "sp500") -> "Path":
    """Atomically write `frames` to the on-disk cache for a symbol set.

    Lets a caller that already has frames in memory (e.g. production inference,
    which pulls fresh) warm the cache with no extra Supabase egress, so the
    single-ticker scoring path can read the universe from disk instead of
    re-pulling the full history.
    """
    cache_dir = get_settings().frame_cache_dir
    path = cache_dir / f"frames_{_frame_cache_key(symbols, universe_tag)}.pkl"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pkl.tmp")
    import gzip
    with gzip.open(tmp, "wb", compresslevel=6) as f:
        pickle.dump({"version": FRAME_CACHE_VERSION, "universe": universe_tag,
                     "frames": frames}, f)
    tmp.replace(path)  # atomic: never leave a half-written cache file
    return path


async def load_frames_cached(
    pool, symbols: list[str] | None = None, refresh: bool = False,
    universe_tag: str = "sp500",
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
    path = cache_dir / f"frames_{_frame_cache_key(symbols, universe_tag)}.pkl"
    if path.exists() and not refresh:
        from backend.ml.compressed_io import open_artifact
        with open_artifact(path) as f:
            cached = pickle.load(f)
        if not isinstance(cached, dict) or cached.get("version") != FRAME_CACHE_VERSION:
            raise IncompatibleFrameCache("incompatible frame cache: rerun with --refresh-cache")
        if cached.get("universe") != universe_tag:
            raise ValueError("frame cache universe mismatch")
        frames = cached["frames"]
        print(f"[frame-cache] hit: {path} ({len(frames)} tickers) — no Supabase egress")
        return frames

    why = "refresh" if refresh else "miss"
    print(f"[frame-cache] {why}: pulling full history from Supabase ...")
    if universe_tag == "sp500":
        frames = await load_frames(pool, symbols=symbols)
    elif universe_tag == "sp1500":
        frames = await load_frames(pool, symbols=symbols, index_ids=("SPX", "SP400", "SP600"),
                                   include_research=True)
    else:
        raise ValueError("unknown universe")
    write_frames_cache(frames, symbols=symbols, universe_tag=universe_tag)
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
