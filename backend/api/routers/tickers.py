"""Ticker catalog search, per-ticker detail bundle, and price history.

The detail bundle resolves the *active model* (production if one exists, else the
most recently created candidate — the GBDT inference writer registers candidates)
and returns its latest predictions per horizon as percentile ranks.

This module deliberately does not import anything from backend.ml.* — that would
pull torch/lightgbm into the API process (libomp segfault risk). Predictions are
read straight from the `predictions` table.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query
from starlette.concurrency import run_in_threadpool

from backend.api.auth import get_current_user
from backend.api.deps import get_pool
from backend.api.schemas import (
    AddTickerRequest,
    AddTickerResponse,
    FundamentalsSnapshot,
    HorizonPrediction,
    PricePoint,
    SentimentSnapshot,
    TickerDetail,
    TickerStatus,
    TickerSummary,
)
from backend.config import REPO_ROOT

router = APIRouter(prefix="/tickers", tags=["tickers"])

# Display order; kept local to avoid importing backend.ml.model (torch).
HORIZON_ORDER = ("1M", "3M", "6M", "1Y")

# A user-added job still polling after this long is treated as failed, so the
# frontend never polls forever if the worker was hard-killed before its finally.
_ADD_STALE_SECONDS = 15 * 60

# lookback token -> trading-window in calendar days ("max" => no lower bound).
_LOOKBACK_DAYS = {"1m": 31, "3m": 93, "6m": 186, "1y": 366, "2y": 731, "5y": 1827}


async def _resolve_active_model(conn: asyncpg.Connection) -> tuple[str, str] | None:
    """Return (model_version_id, status) for the active model, or None."""
    row = await conn.fetchrow(
        "select model_version_id, status from model_versions "
        "where status = 'production' order by promoted_at desc nulls last limit 1"
    )
    if row is None:
        row = await conn.fetchrow(
            "select model_version_id, status from model_versions "
            "order by created_at desc limit 1"
        )
    if row is None:
        return None
    return str(row["model_version_id"]), str(row["status"])


async def _ticker_by_symbol(conn: asyncpg.Connection, symbol: str) -> asyncpg.Record:
    row = await conn.fetchrow(
        """
        select ticker_id, symbol, name, sector, industry, asset_type, user_added
          from tickers where upper(symbol) = upper($1)
        """,
        symbol,
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"Unknown ticker: {symbol}")
    return row


@router.get("", response_model=list[TickerSummary])
async def search_tickers(
    q: str = Query(default="", description="Symbol or name substring"),
    limit: int = Query(default=20, le=100),
    _user: str = Depends(get_current_user),
    pool: asyncpg.Pool = Depends(get_pool),
) -> list[TickerSummary]:
    pattern = f"%{q.strip()}%"
    rows = await pool.fetch(
        """
        select ticker_id, symbol, name, sector, industry, asset_type, user_added
          from tickers
         where active = true
           and (symbol ilike $1 or name ilike $1)
         order by (upper(symbol) = upper($2)) desc, symbol
         limit $3
        """,
        pattern,
        q.strip(),
        limit,
    )
    return [TickerSummary(**dict(r)) for r in rows]


def _asset_type_from_info(info: dict) -> str | None:
    """Map yfinance quoteType to our asset_type, or None for unsupported/unknown."""
    qt = (info.get("quoteType") or "").upper()
    if qt == "EQUITY":
        return "equity"
    if qt == "ETF":
        return "etf"
    return None


@router.post("", response_model=AddTickerResponse, status_code=202)
async def add_ticker(
    body: AddTickerRequest,
    _user: str = Depends(get_current_user),
    pool: asyncpg.Pool = Depends(get_pool),
) -> AddTickerResponse:
    """Queue ingestion + scoring of an off-index ticker, returning immediately.

    Inserts the `tickers` row synchronously (so the detail page resolves right
    away while scoring is in flight), then a detached worker
    (`backend.jobs.add_ticker`) pulls prices/fundamentals/sentiment + enriches
    metadata and scores the symbol against the current S&P cross-section. The
    client polls GET /tickers/{symbol}/status.
    """
    symbol = body.symbol.strip().upper()
    if not symbol or len(symbol) > 12:
        raise HTTPException(status_code=400, detail="Invalid symbol")

    existing = await pool.fetchrow(
        "select ticker_id, active from tickers where upper(symbol) = upper($1)",
        symbol,
    )
    if existing is not None and existing["active"]:
        return AddTickerResponse(symbol=symbol, status="exists", ticker_id=int(existing["ticker_id"]))

    if existing is not None:
        # Reactivate a removed-from-index name as user-added.
        ticker_id = int(existing["ticker_id"])
        await pool.execute(
            "update tickers set active = true, user_added = true, removed_at = null where ticker_id = $1",
            ticker_id,
        )
    else:
        # New symbol: validate via yfinance and insert a minimal row. yfinance is
        # already used by the quotes router, so this stays out of backend.ml.
        import yfinance as yf

        info = await run_in_threadpool(lambda: yf.Ticker(symbol).info or {})
        asset_type = _asset_type_from_info(info)
        if asset_type is None:
            raise HTTPException(status_code=404, detail=f"Unknown or unsupported symbol: {symbol}")
        ticker_id = int(
            await pool.fetchval(
                """
                insert into tickers (symbol, name, asset_type, active, user_added)
                values ($1, $2, $3, true, true)
                returning ticker_id
                """,
                symbol,
                info.get("longName") or info.get("shortName"),
                asset_type,
            )
        )

    run_id = int(
        await pool.fetchval(
            "insert into ingestion_runs (job_name) values ($1) returning run_id",
            f"add_ticker:{symbol}",
        )
    )
    # Detached so it outlives the request; it owns its own ingestion_runs status.
    subprocess.Popen(
        [sys.executable, "-m", "backend.jobs.add_ticker", "--symbol", symbol, "--run-id", str(run_id)],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return AddTickerResponse(symbol=symbol, status="queued", run_id=run_id, ticker_id=ticker_id)


@router.get("/{symbol}/status", response_model=TickerStatus)
async def ticker_status(
    symbol: str,
    _user: str = Depends(get_current_user),
    pool: asyncpg.Pool = Depends(get_pool),
) -> TickerStatus:
    symbol = symbol.strip().upper()
    row = await pool.fetchrow(
        """
        select status, started_at, error_message, metadata
          from ingestion_runs
         where job_name = $1
         order by started_at desc
         limit 1
        """,
        f"add_ticker:{symbol}",
    )
    if row is None:
        # No add job for this symbol (e.g. a covered S&P name) — ready if scored.
        has_pred = await pool.fetchval(
            "select exists(select 1 from predictions p "
            "join tickers t on t.ticker_id = p.ticker_id where upper(t.symbol) = upper($1))",
            symbol,
        )
        return TickerStatus(symbol=symbol, status="ready" if has_pred else "unknown")

    raw_meta = row["metadata"]
    meta = json.loads(raw_meta) if isinstance(raw_meta, str) else (raw_meta or {})
    outcome = meta.get("outcome")
    status = row["status"]

    if status == "running":
        age = (datetime.now(timezone.utc) - row["started_at"]).total_seconds()
        if age > _ADD_STALE_SECONDS:
            return TickerStatus(symbol=symbol, status="failed", message="Timed out — worker did not finish")
        return TickerStatus(symbol=symbol, status="running", message="Pulling data and scoring…")
    if status == "success":
        if outcome == "insufficient_history":
            return TickerStatus(
                symbol=symbol, status="insufficient_history",
                message="Not enough price history (need ~1 year) to score this ticker.",
            )
        return TickerStatus(symbol=symbol, status="ready")
    return TickerStatus(symbol=symbol, status="failed", message=row["error_message"] or "Job failed")


@router.get("/{symbol}", response_model=TickerDetail)
async def ticker_detail(
    symbol: str,
    _user: str = Depends(get_current_user),
    pool: asyncpg.Pool = Depends(get_pool),
) -> TickerDetail:
    async with pool.acquire() as conn:
        t = await _ticker_by_symbol(conn, symbol)
        ticker_id = int(t["ticker_id"])

        predictions: list[HorizonPrediction] = []
        as_of_date = model_version_id = model_status = None
        active = await _resolve_active_model(conn)
        if active is not None:
            model_version_id, model_status = active
            pred_rows = await conn.fetch(
                """
                select horizon, direction_prob, confidence, as_of_date
                  from predictions
                 where ticker_id = $1 and model_version_id = $2
                   and as_of_date = (
                       select max(as_of_date) from predictions
                        where ticker_id = $1 and model_version_id = $2
                   )
                """,
                ticker_id,
                model_version_id,
            )
            by_h = {r["horizon"]: r for r in pred_rows}
            for h in HORIZON_ORDER:
                r = by_h.get(h)
                if r is None:
                    continue
                as_of_date = r["as_of_date"]
                predictions.append(
                    HorizonPrediction(
                        horizon=h,
                        percentile_rank=float(r["direction_prob"]),
                        rank_std=float(r["confidence"]) if r["confidence"] is not None else None,
                    )
                )

        f = await conn.fetchrow(
            """
            select period_end, filed_at::date as filed_at, filing_type,
                   revenue, net_income, gross_margin, operating_margin,
                   total_debt, total_equity, fcf
              from fundamentals where ticker_id = $1 order by filed_at desc limit 1
            """,
            ticker_id,
        )
        s = await conn.fetchrow(
            """
            select score_date, mean_score, headline_count, rolling_7d, rolling_14d
              from sentiment_daily where ticker_id = $1 order by score_date desc limit 1
            """,
            ticker_id,
        )
        close_row = await conn.fetchrow(
            "select close from price_history where ticker_id = $1 "
            "order by trade_date desc limit 1",
            ticker_id,
        )

    return TickerDetail(
        ticker=TickerSummary(**dict(t)),
        as_of_date=as_of_date,
        model_version_id=model_version_id,
        model_status=model_status,
        predictions=predictions,
        fundamentals=FundamentalsSnapshot(**dict(f)) if f else None,
        sentiment=SentimentSnapshot(**dict(s)) if s else None,
        last_close=(
            float(close_row["close"])
            if close_row and close_row["close"] is not None
            else None
        ),
    )


@router.get("/{symbol}/prices", response_model=list[PricePoint])
async def ticker_prices(
    symbol: str,
    lookback: str = Query(default="1y"),
    _user: str = Depends(get_current_user),
    pool: asyncpg.Pool = Depends(get_pool),
) -> list[PricePoint]:
    async with pool.acquire() as conn:
        t = await _ticker_by_symbol(conn, symbol)
        ticker_id = int(t["ticker_id"])
        days = _LOOKBACK_DAYS.get(lookback.lower())
        if days is None and lookback.lower() != "max":
            days = 366
        if days is None:  # "max"
            rows = await conn.fetch(
                "select trade_date, adj_close from price_history "
                "where ticker_id = $1 and adj_close is not null order by trade_date",
                ticker_id,
            )
        else:
            rows = await conn.fetch(
                "select trade_date, adj_close from price_history "
                "where ticker_id = $1 and adj_close is not null "
                "and trade_date >= (current_date - make_interval(days => $2)) "
                "order by trade_date",
                ticker_id,
                days,
            )
    return [PricePoint(date=r["trade_date"], close=float(r["adj_close"])) for r in rows]
