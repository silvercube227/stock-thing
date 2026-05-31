"""Cross-sectional rankings for the screener / compare view.

Returns the active model's latest cross-section for a horizon, ordered by
percentile rank — i.e. how every covered ticker stacks up against the universe.
"""

from __future__ import annotations

import asyncpg
from fastapi import APIRouter, Depends, Query

from backend.api.auth import get_current_user
from backend.api.deps import get_pool
from backend.api.routers.tickers import HORIZON_ORDER, _resolve_active_model
from backend.api.schemas import RankingResponse, RankingRow

router = APIRouter(prefix="/rankings", tags=["rankings"])


@router.get("", response_model=RankingResponse)
async def rankings(
    horizon: str = Query(default="6M"),
    limit: int = Query(default=500, le=1000),
    _user: str = Depends(get_current_user),
    pool: asyncpg.Pool = Depends(get_pool),
) -> RankingResponse:
    h = horizon.upper()
    if h not in HORIZON_ORDER:
        h = "6M"

    async with pool.acquire() as conn:
        active = await _resolve_active_model(conn)
        if active is None:
            return RankingResponse(horizon=h)
        model_version_id, model_status = active

        rows = await conn.fetch(
            """
            select t.ticker_id, t.symbol, t.name, t.sector,
                   p.direction_prob, p.confidence, p.as_of_date
              from predictions p
              join tickers t on t.ticker_id = p.ticker_id
             where p.model_version_id = $1 and p.horizon = $2
               and t.user_added = false
               and p.as_of_date = (
                   select max(as_of_date) from predictions
                    where model_version_id = $1 and horizon = $2
               )
             order by p.direction_prob desc
             limit $3
            """,
            model_version_id,
            h,
            limit,
        )

    as_of = rows[0]["as_of_date"] if rows else None
    return RankingResponse(
        horizon=h,
        as_of_date=as_of,
        model_version_id=model_version_id,
        model_status=model_status,
        rows=[
            RankingRow(
                ticker_id=r["ticker_id"],
                symbol=r["symbol"],
                name=r["name"],
                sector=r["sector"],
                percentile_rank=float(r["direction_prob"]),
                rank_std=float(r["confidence"]) if r["confidence"] is not None else None,
            )
            for r in rows
        ],
    )
