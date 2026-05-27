"""Portfolio CRUD, scoped to the authenticated Supabase user.

RLS is bypassed by the direct DB connection (see backend/api/auth.py), so every
statement here filters/sets `user_id` explicitly. There is no unique index on
(user_id, ticker_id), so add/update is a manual upsert inside a transaction to
keep at most one row per (user, ticker).
"""

from __future__ import annotations

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, status

from backend.api.auth import get_current_user
from backend.api.deps import get_pool
from backend.api.schemas import HoldingCreate, HoldingUpdate, PortfolioRow

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


async def _resolve_ticker_id(pool: asyncpg.Pool, symbol: str) -> int:
    row = await pool.fetchrow(
        "select ticker_id from tickers where upper(symbol) = upper($1) and active = true",
        symbol,
    )
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown or inactive ticker: {symbol}",
        )
    return int(row["ticker_id"])


@router.get("", response_model=list[PortfolioRow])
async def list_portfolio(
    user_id: str = Depends(get_current_user),
    pool: asyncpg.Pool = Depends(get_pool),
) -> list[PortfolioRow]:
    rows = await pool.fetch(
        """
        select h.ticker_id, t.symbol, t.name, t.sector,
               h.shares, h.cost_basis, h.acquired_at,
               ph.close as last_close, ph.trade_date as last_close_date
          from portfolio_holdings h
          join tickers t on t.ticker_id = h.ticker_id
          left join lateral (
              select close, trade_date from price_history
               where ticker_id = h.ticker_id
               order by trade_date desc limit 1
          ) ph on true
         where h.user_id = $1
         order by t.symbol
        """,
        user_id,
    )
    return [PortfolioRow(**dict(r)) for r in rows]


@router.post("", response_model=PortfolioRow, status_code=status.HTTP_201_CREATED)
async def add_holding(
    body: HoldingCreate,
    user_id: str = Depends(get_current_user),
    pool: asyncpg.Pool = Depends(get_pool),
) -> PortfolioRow:
    ticker_id = await _resolve_ticker_id(pool, body.symbol)
    async with pool.acquire() as conn, conn.transaction():
        existing = await conn.fetchrow(
            "select id from portfolio_holdings where user_id = $1 and ticker_id = $2",
            user_id,
            ticker_id,
        )
        if existing is None:
            await conn.execute(
                """
                    insert into portfolio_holdings
                        (user_id, ticker_id, shares, cost_basis, acquired_at, notes)
                    values ($1, $2, $3, $4, $5, $6)
                    """,
                user_id,
                ticker_id,
                body.shares,
                body.cost_basis,
                body.acquired_at,
                body.notes,
            )
        else:
            await conn.execute(
                """
                    update portfolio_holdings
                       set shares = $3, cost_basis = $4, acquired_at = $5, notes = $6
                     where user_id = $1 and ticker_id = $2
                    """,
                user_id,
                ticker_id,
                body.shares,
                body.cost_basis,
                body.acquired_at,
                body.notes,
            )
    return await _row_for(pool, user_id, ticker_id)


@router.patch("/{ticker_id}", response_model=PortfolioRow)
async def update_holding(
    ticker_id: int,
    body: HoldingUpdate,
    user_id: str = Depends(get_current_user),
    pool: asyncpg.Pool = Depends(get_pool),
) -> PortfolioRow:
    if body.shares is None and body.cost_basis is None:
        raise HTTPException(status_code=400, detail="No fields to update")
    result = await pool.execute(
        """
        update portfolio_holdings
           set shares     = coalesce($3, shares),
               cost_basis = coalesce($4, cost_basis)
         where user_id = $1 and ticker_id = $2
        """,
        user_id,
        ticker_id,
        body.shares,
        body.cost_basis,
    )
    if result.endswith("0"):
        raise HTTPException(status_code=404, detail="Holding not found")
    return await _row_for(pool, user_id, ticker_id)


@router.delete("/{ticker_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_holding(
    ticker_id: int,
    user_id: str = Depends(get_current_user),
    pool: asyncpg.Pool = Depends(get_pool),
) -> None:
    result = await pool.execute(
        "delete from portfolio_holdings where user_id = $1 and ticker_id = $2",
        user_id,
        ticker_id,
    )
    if result.endswith("0"):
        raise HTTPException(status_code=404, detail="Holding not found")


async def _row_for(pool: asyncpg.Pool, user_id: str, ticker_id: int) -> PortfolioRow:
    row = await pool.fetchrow(
        """
        select h.ticker_id, t.symbol, t.name, t.sector,
               h.shares, h.cost_basis, h.acquired_at,
               ph.close as last_close, ph.trade_date as last_close_date
          from portfolio_holdings h
          join tickers t on t.ticker_id = h.ticker_id
          left join lateral (
              select close, trade_date from price_history
               where ticker_id = h.ticker_id
               order by trade_date desc limit 1
          ) ph on true
         where h.user_id = $1 and h.ticker_id = $2
        """,
        user_id,
        ticker_id,
    )
    return PortfolioRow(**dict(row))
