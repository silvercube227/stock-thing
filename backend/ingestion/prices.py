"""Price/volume ingestion from yfinance.

Two entry points:
  - ingest_full_history(...)   one-shot backfill from 2010-01-01 (initial or after a drift)
  - ingest_recent(...)         daily incremental, with drift detection

Both write to price_history (idempotent upsert on (ticker_id, trade_date))
and log a row in ingestion_runs.

yfinance is sync; we wrap blocking calls in asyncio.to_thread so the async
orchestration (semaphore-limited concurrency, pooled DB writes) still works.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

import asyncpg
import pandas as pd
import yfinance as yf

from backend.ingestion.calendar import expected_bar_count

log = logging.getLogger(__name__)

# Relative-difference threshold above which we conclude that a split or
# dividend has shifted the adjusted-close series for a ticker and its history
# must be re-pulled (from HISTORY_START) to re-derive adj_close.
DRIFT_REL_THRESHOLD: float = 1e-3

# Earliest date we keep price history from. Both the initial backfill and the
# on-drift re-pull use this floor, so the panel stays rectangular — a drifted
# ticker doesn't get a deeper history than the rest of the universe.
HISTORY_START: date = date(2010, 1, 1)

# Cap on concurrent yfinance fetches. yfinance unofficially rate-limits at
# the cookie level; 4 is a reasonable balance for ~35 tickers without 429s.
DEFAULT_CONCURRENCY: int = 4


# =============================================================
# Public API: results
# =============================================================


@dataclass
class TickerResult:
    ticker_id: int
    symbol: str
    rows_written: int
    drifted: bool = False
    error: str | None = None
    missing_bars: int = 0


@dataclass
class IngestionResult:
    job_name: str
    started_at: datetime
    finished_at: datetime
    rows_inserted: int = 0
    drifted_tickers: list[str] = field(default_factory=list)
    failed_tickers: list[str] = field(default_factory=list)
    per_ticker: list[TickerResult] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.failed_tickers and self.rows_inserted == 0:
            return "failed"
        if self.failed_tickers:
            return "partial"
        return "success"


# =============================================================
# yfinance fetch (sync, wrapped in to_thread)
# =============================================================


def _yf_history(symbol: str, period: str | None = None, start: date | None = None) -> pd.DataFrame:
    """Synchronous yfinance fetch with our preferred options.

    auto_adjust=False -> raw OHLC + separate Adj Close column (we store both).
    actions=True      -> Dividends + Stock Splits columns (we record per-row).
    """
    t = yf.Ticker(symbol)
    if start is not None:
        df = t.history(start=start.isoformat(), auto_adjust=False, actions=True)
    else:
        df = t.history(period=period or "5y", auto_adjust=False, actions=True)
    return df


def _df_to_rows(ticker_id: int, df: pd.DataFrame) -> list[tuple]:
    """Convert a yfinance DataFrame to a list of upsert tuples."""
    rows: list[tuple] = []
    for ts, row in df.iterrows():
        # yfinance returns a tz-aware DatetimeIndex; we only want the date.
        trade_date = ts.date() if hasattr(ts, "date") else ts

        def _f(col: str) -> float | None:
            v = row.get(col)
            return None if v is None or pd.isna(v) else float(v)

        def _i(col: str) -> int | None:
            v = row.get(col)
            return None if v is None or pd.isna(v) else int(v)

        split = row.get("Stock Splits")
        split_factor = 1.0 if split in (None, 0) or pd.isna(split) else float(split)
        dividend = row.get("Dividends")
        dividend = 0.0 if dividend is None or pd.isna(dividend) else float(dividend)

        rows.append(
            (
                ticker_id,
                trade_date,
                _f("Open"),
                _f("High"),
                _f("Low"),
                _f("Close"),
                _f("Adj Close"),
                _i("Volume"),
                split_factor,
                dividend,
                "yfinance",
            )
        )
    return rows


# =============================================================
# Drift detection (pure; unit-testable)
# =============================================================


def detect_drift(stored_adj_close: float | None, new_adj_close: float | None) -> bool:
    """Return True if the new adj_close differs from the stored value enough
    to imply a corporate action happened since the stored row was written.

    Pure function — separated from DB I/O for unit testing.
    """
    if stored_adj_close is None or new_adj_close is None:
        return False
    if stored_adj_close == 0:
        return False
    rel = abs(new_adj_close - stored_adj_close) / abs(stored_adj_close)
    return rel > DRIFT_REL_THRESHOLD


# =============================================================
# DB I/O
# =============================================================


_UPSERT_SQL = """
insert into price_history (
    ticker_id, trade_date, open, high, low, close, adj_close,
    volume, split_factor, dividend, source, ingested_at
) values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, now())
on conflict (ticker_id, trade_date) do update set
    open         = excluded.open,
    high         = excluded.high,
    low          = excluded.low,
    close        = excluded.close,
    adj_close    = excluded.adj_close,
    volume       = excluded.volume,
    split_factor = excluded.split_factor,
    dividend     = excluded.dividend,
    source       = excluded.source,
    ingested_at  = now()
"""


async def _fetch_active_tickers(pool: asyncpg.Pool) -> list[tuple[int, str]]:
    rows = await pool.fetch(
        "select ticker_id, symbol from tickers where active = true order by ticker_id"
    )
    return [(r["ticker_id"], r["symbol"]) for r in rows]


async def _check_drift_against_db(
    conn: asyncpg.Connection, ticker_id: int, df: pd.DataFrame
) -> bool:
    """Compare the earliest overlapping bar's adj_close in the new pull to
    what's stored. Caller uses this to decide whether to re-pull full history.
    """
    if df.empty:
        return False
    earliest = df.index.min()
    earliest_date = earliest.date() if hasattr(earliest, "date") else earliest
    stored = await conn.fetchrow(
        "select adj_close from price_history where ticker_id = $1 and trade_date = $2",
        ticker_id,
        earliest_date,
    )
    if stored is None:
        return False
    new_adj = df["Adj Close"].iloc[0]
    return detect_drift(
        float(stored["adj_close"]) if stored["adj_close"] is not None else None,
        None if pd.isna(new_adj) else float(new_adj),
    )


async def _log_run_start(pool: asyncpg.Pool, job_name: str) -> int:
    row = await pool.fetchrow(
        "insert into ingestion_runs (job_name) values ($1) returning run_id",
        job_name,
    )
    return int(row["run_id"])


async def _log_run_finish(pool: asyncpg.Pool, run_id: int, result: IngestionResult) -> None:
    await pool.execute(
        """
        update ingestion_runs set
            finished_at   = now(),
            status        = $2,
            rows_inserted = $3,
            error_message = $4,
            metadata      = $5
         where run_id = $1
        """,
        run_id,
        result.status,
        result.rows_inserted,
        "; ".join(
            f"{tr.symbol}: {tr.error}" for tr in result.per_ticker if tr.error
        )
        or None,
        # asyncpg encodes JSON via stringification; use json.dumps for portability.
        __import__("json").dumps(
            {
                "drifted_tickers": result.drifted_tickers,
                "failed_tickers": result.failed_tickers,
                "ticker_count": len(result.per_ticker),
            }
        ),
    )


# =============================================================
# Per-ticker worker
# =============================================================


async def _ingest_one(
    pool: asyncpg.Pool,
    ticker_id: int,
    symbol: str,
    period: str | None = None,
    start: date | None = None,
) -> TickerResult:
    """Pull, optionally re-pull on drift, upsert."""
    try:
        df = await asyncio.to_thread(_yf_history, symbol, period=period, start=start)
    except Exception as exc:  # noqa: BLE001 — surface anything yfinance throws
        return TickerResult(ticker_id, symbol, 0, error=f"fetch failed: {exc}")

    if df is None or df.empty:
        return TickerResult(ticker_id, symbol, 0, error="empty dataframe")

    drifted = False
    try:
        async with pool.acquire() as conn:
            # Only run drift detection on incremental pulls (when start is given).
            # On full-period pulls we're already overwriting everything.
            if start is not None:
                drifted = await _check_drift_against_db(conn, ticker_id, df)
                if drifted:
                    log.warning(
                        "drift detected for %s — re-pulling from %s", symbol, HISTORY_START
                    )
                    try:
                        df = await asyncio.to_thread(
                            _yf_history, symbol, start=HISTORY_START
                        )
                    except Exception as exc:  # noqa: BLE001
                        return TickerResult(
                            ticker_id, symbol, 0, drifted=True,
                            error=f"redrift fetch failed: {exc}",
                        )

            rows = _df_to_rows(ticker_id, df)
            async with conn.transaction():
                await conn.executemany(_UPSERT_SQL, rows)
    except Exception as exc:  # noqa: BLE001 — a Supabase pooler connection drop / DB error
        # becomes a per-ticker failure instead of aborting the whole gather and
        # orphaning the prices_daily run row as "running".
        return TickerResult(
            ticker_id, symbol, 0, drifted=drifted, error=f"db write failed: {exc}"
        )

    # NYSE-calendar gap check (informational only — logged to metadata).
    missing = 0
    if not df.empty:
        first = df.index.min().date()
        last = df.index.max().date()
        expected = expected_bar_count(first, last)
        missing = max(0, expected - len(df))
        if missing > 0:
            log.warning(
                "%s: %d missing bars in [%s, %s]", symbol, missing, first, last
            )

    return TickerResult(ticker_id, symbol, len(rows), drifted=drifted, missing_bars=missing)


# =============================================================
# Public entry points
# =============================================================


async def ingest_full_history(
    pool: asyncpg.Pool,
    tickers: Iterable[tuple[int, str]] | None = None,
    start_date: date = HISTORY_START,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> IngestionResult:
    """Pull history from `start_date` for each ticker and overwrite price_history.

    Use this for the initial bootstrap and as the recovery path after drift.
    """
    if tickers is None:
        tickers = await _fetch_active_tickers(pool)
    tickers = list(tickers)

    started = datetime.now(timezone.utc)
    run_id = await _log_run_start(pool, "prices_full")

    sem = asyncio.Semaphore(concurrency)

    async def _wrapped(tid: int, sym: str) -> TickerResult:
        async with sem:
            return await _ingest_one(pool, tid, sym, start=start_date)

    per_ticker = await asyncio.gather(*(_wrapped(t, s) for t, s in tickers))
    finished = datetime.now(timezone.utc)

    result = IngestionResult(
        job_name="prices_full",
        started_at=started,
        finished_at=finished,
        rows_inserted=sum(tr.rows_written for tr in per_ticker),
        failed_tickers=[tr.symbol for tr in per_ticker if tr.error],
        per_ticker=list(per_ticker),
    )
    await _log_run_finish(pool, run_id, result)
    return result


async def ingest_recent(
    pool: asyncpg.Pool,
    tickers: Iterable[tuple[int, str]] | None = None,
    days: int = 7,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> IngestionResult:
    """Daily incremental ingest: pull the last `days` of bars and upsert.

    If the earliest pulled bar's adj_close diverges from the stored value, the
    ticker is re-pulled from HISTORY_START to absorb the split/dividend.
    """
    if tickers is None:
        tickers = await _fetch_active_tickers(pool)
    tickers = list(tickers)

    started = datetime.now(timezone.utc)
    run_id = await _log_run_start(pool, "prices_daily")

    start_date = date.today() - timedelta(days=days)
    sem = asyncio.Semaphore(concurrency)

    async def _wrapped(tid: int, sym: str) -> TickerResult:
        async with sem:
            return await _ingest_one(pool, tid, sym, start=start_date)

    per_ticker_raw = await asyncio.gather(
        *(_wrapped(t, s) for t, s in tickers), return_exceptions=True
    )
    # Map any exception that still escaped _ingest_one to a failed TickerResult, so
    # one bad ticker can't abort the batch (which would skip _log_run_finish and
    # leave the run row stuck at "running").
    per_ticker = [
        res if isinstance(res, TickerResult)
        else TickerResult(tid, sym, 0, error=f"unhandled: {res}")
        for (tid, sym), res in zip(tickers, per_ticker_raw)
    ]
    finished = datetime.now(timezone.utc)

    result = IngestionResult(
        job_name="prices_daily",
        started_at=started,
        finished_at=finished,
        rows_inserted=sum(tr.rows_written for tr in per_ticker),
        drifted_tickers=[tr.symbol for tr in per_ticker if tr.drifted],
        failed_tickers=[tr.symbol for tr in per_ticker if tr.error],
        per_ticker=list(per_ticker),
    )
    await _log_run_finish(pool, run_id, result)
    return result
