"""LSEG/I-B-E-S analyst-estimate ingestion via the lseg.data desktop session.

For each active equity we resolve its RIC, pull monthly point-in-time history of
the consensus fields, and upsert RAW values into `analyst_estimates`. Derived
features (recommendation revisions, revenue surprise, forward yields, price-target
upside) are computed downstream in backend/ml/gbm_baseline.py — same split as
fundamentals -> features.

CRITICAL: `as_of_date` is the LSEG observation date (when the consensus was
published). All point-in-time joins use it, exactly like fundamentals.filed_at.

Requires Workspace running locally and LSEG_APP_KEY set. lseg.data is synchronous
and the desktop session is not safe for concurrent calls, so tickers are processed
sequentially; only the DB writes are async.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Iterable

import asyncpg

from backend.config import get_settings

log = logging.getLogger(__name__)

# Monthly point-in-time history from here forward (probe confirmed coverage from
# ~2013-2014; earlier dates simply return nothing).
START_DATE = "2013-01-01"

# TR field codes requested per ticker (confirmed available by scripts/_probe_lseg.py).
ESTIMATE_FIELDS = [
    "TR.RecMean",
    "TR.PriceTargetMean",
    "TR.RevenueMean",
    "TR.RevenueActValue",
    "TR.FwdPE",
    "TR.FwdEVToEBITDA",
]

# Map a returned column's display label (substring, case-insensitive) -> db column.
# Labels are the exact strings the probe printed; substring match tolerates the
# trailing qualifiers ("(1-5)", "(Daily Time Series Ratio)").
COLUMN_MATCHERS: list[tuple[str, str]] = [
    ("recommendation - mean", "rec_mean"),
    ("price target - mean", "price_target_mean"),
    ("revenue - mean", "revenue_mean"),
    ("revenue - actual", "revenue_actual"),
    ("forward enterprise value to ebitda", "fwd_ev_ebitda"),
    ("forward p/e", "fwd_pe"),
]
DB_FIELDS = ["rec_mean", "price_target_mean", "revenue_mean",
             "revenue_actual", "fwd_pe", "fwd_ev_ebitda"]


# =============================================================
# Result dataclasses (mirror fundamentals.py)
# =============================================================


@dataclass
class TickerResult:
    ticker_id: int
    symbol: str
    ric: str | None
    rows_written: int = 0
    error: str | None = None


@dataclass
class IngestionResult:
    started_at: datetime
    finished_at: datetime
    rows_inserted: int = 0
    failed_tickers: list[str] = field(default_factory=list)
    skipped_no_ric: list[str] = field(default_factory=list)
    per_ticker: list[TickerResult] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.failed_tickers and self.rows_inserted == 0:
            return "failed"
        if self.failed_tickers:
            return "partial"
        return "success"


# =============================================================
# Pure parsing (no I/O) — exercised by tests with a synthetic frame
# =============================================================


def _num(v) -> float | None:
    """Coerce a cell to float, mapping NaN/None to None."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _column_dbfield(label: str) -> str | None:
    low = str(label).lower()
    for sub, dbf in COLUMN_MATCHERS:
        if sub in low:
            return dbf
    return None


def parse_history(df, ticker_id: int) -> list[dict]:
    """Convert a single-ticker get_history frame into raw upsert rows.

    `df` is Date-indexed with columns named by LSEG display labels. We map each
    column to a db field, then emit one row per date that has at least one value.
    """
    if df is None or getattr(df, "empty", True):
        return []
    colmap = {col: _column_dbfield(col[-1] if isinstance(col, tuple) else col)
              for col in df.columns}
    colmap = {c: f for c, f in colmap.items() if f is not None}

    rows: list[dict] = []
    for ts, r in df.iterrows():
        vals = {dbf: _num(r[col]) for col, dbf in colmap.items()}
        if all(v is None for v in vals.values()):
            continue
        as_of = ts.date() if hasattr(ts, "date") else ts
        rows.append({"ticker_id": ticker_id, "as_of_date": as_of,
                     **{f: vals.get(f) for f in DB_FIELDS}})
    return rows


# =============================================================
# LSEG session + symbology (lazy import so the module loads without Workspace)
# =============================================================


def _open_session(app_key: str):
    import lseg.data as ld  # noqa: PLC0415
    ld.open_session(app_key=app_key)
    return ld


def resolve_rics(ld, symbols: list[str]) -> dict[str, str]:
    """Map ticker symbols -> primary RIC via LSEG symbology (one batch call)."""
    from lseg.data.content import symbol_conversion  # noqa: PLC0415
    resp = symbol_conversion.Definition(
        symbols=symbols,
        from_symbol_type=symbol_conversion.SymbolTypes.TICKER_SYMBOL,
        to_symbol_types=[symbol_conversion.SymbolTypes.RIC],
    ).get_data()
    df = resp.data.df
    if df is None or df.empty:
        return {}
    ric_col = next((c for c in df.columns if "RIC" in str(c).upper()), None)
    if ric_col is None:
        return {}
    # The input symbol is echoed either as the index or as an instrument-like column.
    sym_col = next(
        (c for c in df.columns
         if any(k in str(c).lower() for k in ("instrument", "symbol", "ticker"))),
        None,
    )
    syms = df[sym_col] if sym_col is not None else df.index
    out: dict[str, str] = {}
    for sym, ric in zip(syms, df[ric_col], strict=False):
        if ric is not None and str(ric).lower() != "nan":
            out[str(sym)] = str(ric)
    return out


# =============================================================
# DB I/O
# =============================================================


_UPSERT_SQL = """
insert into analyst_estimates (
    ticker_id, as_of_date, rec_mean, price_target_mean,
    revenue_mean, revenue_actual, fwd_pe, fwd_ev_ebitda, ingested_at
) values ($1, $2, $3, $4, $5, $6, $7, $8, now())
on conflict (ticker_id, as_of_date) do update set
    rec_mean          = excluded.rec_mean,
    price_target_mean = excluded.price_target_mean,
    revenue_mean      = excluded.revenue_mean,
    revenue_actual    = excluded.revenue_actual,
    fwd_pe            = excluded.fwd_pe,
    fwd_ev_ebitda     = excluded.fwd_ev_ebitda,
    ingested_at       = now()
"""


async def _upsert_estimates(conn: asyncpg.Connection, rows: list[dict]) -> int:
    if not rows:
        return 0
    payload = [
        (r["ticker_id"], r["as_of_date"], r["rec_mean"], r["price_target_mean"],
         r["revenue_mean"], r["revenue_actual"], r["fwd_pe"], r["fwd_ev_ebitda"])
        for r in rows
    ]
    async with conn.transaction():
        await conn.executemany(_UPSERT_SQL, payload)
    return len(payload)


async def _fetch_estimate_universe(pool: asyncpg.Pool) -> list[tuple[int, str]]:
    """Equities that participate in the panel (have price history), active OR
    removed-from-index. Including removed names avoids survivorship bias in the
    estimate features; the price_history join skips names that never enter the
    panel (no point pulling estimates we'd never use)."""
    rows = await pool.fetch(
        "select distinct t.ticker_id, t.symbol from tickers t "
        "join price_history p using (ticker_id) "
        "where t.asset_type = 'equity' order by t.symbol"
    )
    return [(r["ticker_id"], r["symbol"]) for r in rows]


async def _log_run_start(pool: asyncpg.Pool) -> int:
    row = await pool.fetchrow(
        "insert into ingestion_runs (job_name) values ($1) returning run_id", "estimates"
    )
    return int(row["run_id"])


async def _log_run_finish(pool: asyncpg.Pool, run_id: int, result: IngestionResult) -> None:
    await pool.execute(
        """
        update ingestion_runs set
            finished_at = now(), status = $2, rows_inserted = $3,
            error_message = $4, metadata = $5
         where run_id = $1
        """,
        run_id,
        result.status,
        result.rows_inserted,
        "; ".join(f"{tr.symbol}: {tr.error}" for tr in result.per_ticker if tr.error) or None,
        json.dumps({
            "failed_tickers": result.failed_tickers,
            "skipped_no_ric": result.skipped_no_ric,
            "ticker_count": len(result.per_ticker),
        }),
    )


# =============================================================
# Public entry point
# =============================================================


async def ingest_estimates(
    pool: asyncpg.Pool,
    tickers: Iterable[tuple[int, str]] | None = None,
) -> IngestionResult:
    """Pull LSEG estimate history for active equities and upsert raw snapshots.

    `tickers` is an iterable of (ticker_id, symbol); if None, all active equities.
    Opens its own desktop session (Workspace must be running, LSEG_APP_KEY set).
    """
    settings = get_settings()
    if not settings.lseg_app_key:
        raise RuntimeError("LSEG_APP_KEY is not set in .env")

    candidates = list(tickers) if tickers is not None else await _fetch_estimate_universe(pool)
    started = datetime.now(timezone.utc)
    run_id = await _log_run_start(pool)
    end = date.today().isoformat()

    ld = _open_session(settings.lseg_app_key)
    per_ticker: list[TickerResult] = []
    try:
        rics = await asyncio.to_thread(resolve_rics, ld, [s for _, s in candidates])
        for ticker_id, symbol in candidates:
            ric = rics.get(symbol)
            if not ric:
                per_ticker.append(TickerResult(ticker_id, symbol, None, error="no RIC"))
                continue
            try:
                df = await asyncio.to_thread(
                    ld.get_history, universe=ric, fields=ESTIMATE_FIELDS,
                    interval="monthly", start=START_DATE, end=end,
                )
            except Exception as exc:  # noqa: BLE001
                per_ticker.append(TickerResult(ticker_id, symbol, ric, error=f"fetch: {exc}"))
                continue
            try:
                rows = parse_history(df, ticker_id)
            except Exception as exc:  # noqa: BLE001
                per_ticker.append(TickerResult(ticker_id, symbol, ric, error=f"parse: {exc}"))
                continue
            n = 0
            if rows:
                async with pool.acquire() as conn:
                    n = await _upsert_estimates(conn, rows)
            per_ticker.append(TickerResult(ticker_id, symbol, ric, rows_written=n))
    finally:
        ld.close_session()

    finished = datetime.now(timezone.utc)
    result = IngestionResult(
        started_at=started,
        finished_at=finished,
        rows_inserted=sum(tr.rows_written for tr in per_ticker),
        failed_tickers=[tr.symbol for tr in per_ticker
                        if tr.error and tr.error != "no RIC"],
        skipped_no_ric=[tr.symbol for tr in per_ticker if tr.error == "no RIC"],
        per_ticker=per_ticker,
    )
    await _log_run_finish(pool, run_id, result)
    return result
