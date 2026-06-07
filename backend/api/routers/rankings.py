"""Cross-sectional rankings for the screener / compare view.

Returns the active model's latest cross-section for a horizon, ordered by
percentile rank — i.e. how every covered ticker stacks up against the universe.
Also surfaces a within-sector percentile (the selection the model is trained on)
and each name's trailing realized Sharpe for sorting/filtering.
"""

from __future__ import annotations

from collections import defaultdict

import asyncpg
from fastapi import APIRouter, Depends, Query

from backend.api.auth import get_current_user
from backend.api.deps import get_pool
from backend.api.routers.tickers import HORIZON_ORDER, _resolve_active_model
from backend.api.schemas import RankingResponse, RankingRow

router = APIRouter(prefix="/rankings", tags=["rankings"])

# Trailing window for the realized Sharpe (≈1 trading year); need a reasonable
# minimum of observations before the ratio is meaningful.
_SHARPE_WINDOW = 252
_SHARPE_MIN_OBS = 30

# Annualized Sharpe of daily log returns over the last _SHARPE_WINDOW trading rows
# per ticker, for the given set of ticker_ids. Population over the window: mean/std
# of daily log returns, annualized by sqrt(252). Null when too few returns.
# Risk-free rate is 0 — this is an excess-return-over-zero Sharpe (no rf is
# subtracted before dividing by volatility). Fine for relative ranking; runs a bit
# high versus a textbook rf-adjusted Sharpe.
_SHARPE_SQL = """
with ranked as (
    select ticker_id, trade_date, adj_close,
           row_number() over (partition by ticker_id order by trade_date desc) as rn
      from price_history
     where ticker_id = any($1::bigint[]) and adj_close is not null and adj_close > 0
       -- Bound the scan to ~recent history so the PK (ticker_id, trade_date)
       -- range-scans instead of reading each ticker's full history. 420 calendar
       -- days comfortably contains the last 252 trading rows kept by rn below.
       and trade_date >= current_date - interval '420 days'
),
windowed as (
    select ticker_id, trade_date, adj_close from ranked where rn <= $2
),
rets as (
    select ticker_id,
           ln(adj_close / lag(adj_close) over (
               partition by ticker_id order by trade_date)) as ret
      from windowed
)
select ticker_id,
       case when count(ret) >= $3 and stddev_samp(ret) > 0
            then (avg(ret) / stddev_samp(ret)) * sqrt(252.0)
            else null end as sharpe
  from rets
 where ret is not null
 group by ticker_id
"""


async def _trailing_sharpe(conn: asyncpg.Connection, ticker_ids: list[int]) -> dict[int, float]:
    if not ticker_ids:
        return {}
    rows = await conn.fetch(_SHARPE_SQL, ticker_ids, _SHARPE_WINDOW, _SHARPE_MIN_OBS)
    return {
        int(r["ticker_id"]): float(r["sharpe"])
        for r in rows
        if r["sharpe"] is not None
    }


def _within_sector_ranks(rows) -> dict[int, tuple[float, str]]:
    """Map ticker_id -> (within-sector percentile in [0, 1], "pos/n" label).

    Rows must already be ordered by descending model score. Groups with fewer than
    2 named members or a null sector are omitted (no meaningful within-sector rank).
    """
    by_sector: dict[str, list] = defaultdict(list)
    for r in rows:
        if r["sector"]:
            by_sector[r["sector"]].append(r)
    out: dict[int, tuple[float, str]] = {}
    for members in by_sector.values():
        n = len(members)
        if n < 2:
            continue
        # members are in descending-score order; position 1 = best in sector.
        ordered = sorted(members, key=lambda r: r["direction_prob"], reverse=True)
        for i, r in enumerate(ordered):
            pct = (n - 1 - i) / (n - 1)  # 1.0 = best, 0.0 = worst
            out[int(r["ticker_id"])] = (pct, f"{i + 1}/{n}")
    return out


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
                   p.direction_prob, p.confidence, p.risk_flag, p.as_of_date
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

        sector_ranks = _within_sector_ranks(rows)
        sharpe = await _trailing_sharpe(conn, [int(r["ticker_id"]) for r in rows])

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
                risk_flag=r["risk_flag"] or "none",
                sector_rank=sector_ranks.get(int(r["ticker_id"]), (None, None))[0],
                sector_rank_label=sector_ranks.get(int(r["ticker_id"]), (None, None))[1],
                sharpe=sharpe.get(int(r["ticker_id"])),
            )
            for r in rows
        ],
    )
