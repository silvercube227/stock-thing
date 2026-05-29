"""Attribute Supabase egress to the read paths that produce it.

Run: python -m scripts.measure_egress

Supabase's metered "egress" is bytes leaving Supabase. For this app that's
almost entirely Postgres result rows flowing out over two paths:

  1. The daily pipeline's ml.dataset.load_frames() — pulls the FULL price /
     fundamentals / sentiment history for every ticker on each inference run
     (Fridays + first trading day of month), Supabase -> local M4. Deterministic:
     we can size it exactly and multiply by run count from ingestion_runs.

  2. FastAPI on Railway answering read endpoints (/rankings, /tickers/{sym},
     /prices) — Supabase -> Railway, once per uncached request. We can size one
     call of each; request volume comes from Railway/Supabase logs, not the DB,
     so those are reported as bytes-per-call for you to multiply.

Sizes use pg_column_size summed over the exact columns each query selects. That
is the on-the-wire *data* size; it excludes wire-protocol framing (~a few bytes
per field) and TLS overhead, so treat results as a lower bound / relative
attribution, not a billing-exact figure. The authoritative monthly total is in
the Supabase dashboard (Reports -> Usage -> Egress); this script tells you which
path it's coming from.

Read-only. Touches nothing.
"""
from __future__ import annotations

import asyncio
from datetime import date

from backend.ingestion.db import pool_context

MB = 1024 * 1024
GB = 1024 * MB


def _fmt(nbytes: float) -> str:
    if nbytes >= GB:
        return f"{nbytes / GB:.3f} GB"
    if nbytes >= MB:
        return f"{nbytes / MB:.2f} MB"
    return f"{nbytes / 1024:.1f} KB"


# Columns each load_frames() query selects, mirrored from ml/dataset.py so the
# size estimate matches what the pipeline actually pulls.
_FULL_PULL = {
    "price_history": "ticker_id, trade_date, adj_close, volume",
    "fundamentals": (
        "ticker_id, filed_at, period_end, filing_type, revenue, net_income, "
        "gross_margin, operating_margin, total_debt, total_equity, fcf"
    ),
    "sentiment_daily": "ticker_id, score_date, rolling_7d, rolling_14d",
}


def _size_expr(cols: str) -> str:
    """sum(pg_column_size(c1) + ...) over selected cols.

    Each term is coalesced to 0: pg_column_size(NULL) is NULL, and NULL in any
    addend would poison the whole row's sum and silently drop it from the total.
    """
    parts = [f"coalesce(pg_column_size({c.strip()}),0)" for c in cols.split(",")]
    return "coalesce(sum(" + " + ".join(parts) + "), 0)"


async def _table_pull_bytes(pool, table: str, cols: str) -> tuple[int, int]:
    row = await pool.fetchrow(
        f"select count(*) as rows, {_size_expr(cols)} as bytes from {table}"
    )
    return int(row["rows"]), int(row["bytes"])


async def measure_full_pull(pool) -> tuple[int, dict[str, tuple[int, int]]]:
    """Bytes one full load_frames() pulls, per table and total."""
    per_table: dict[str, tuple[int, int]] = {}
    total = 0
    for table, cols in _FULL_PULL.items():
        rows, nbytes = await _table_pull_bytes(pool, table, cols)
        per_table[table] = (rows, nbytes)
        total += nbytes
    return total, per_table


async def count_inference_runs(pool, since: date) -> int:
    return int(
        await pool.fetchval(
            "select count(*) from ingestion_runs "
            "where job_name = 'gbm_inference' and started_at >= $1",
            since,
        )
    )


async def measure_api_calls(pool) -> dict[str, tuple[int, int]]:
    """Bytes for one representative call of each public read endpoint.

    Keyed by endpoint -> (rows, bytes). Uses the heaviest ticker for /prices so
    the figure is a realistic upper bound for that endpoint.
    """
    out: dict[str, tuple[int, int]] = {}

    # Heaviest ticker by bar count — worst case for /prices.
    heavy = await pool.fetchrow(
        "select p.ticker_id, t.symbol, count(*) as bars "
        "from price_history p join tickers t using (ticker_id) "
        "group by p.ticker_id, t.symbol order by bars desc limit 1"
    )

    # /tickers/{sym}/prices?lookback=1y  (trade_date, adj_close; ~252 rows)
    if heavy is not None:
        r = await pool.fetchrow(
            "select count(*) as rows, "
            "coalesce(sum(coalesce(pg_column_size(trade_date),0) + coalesce(pg_column_size(adj_close),0)),0) as bytes "
            "from price_history where ticker_id = $1 and adj_close is not null "
            "and trade_date >= (current_date - make_interval(days => 366))",
            heavy["ticker_id"],
        )
        out[f"/tickers/{heavy['symbol']}/prices?lookback=1y"] = (
            int(r["rows"]), int(r["bytes"])
        )
        # ?lookback=max — full history, the worst case
        r = await pool.fetchrow(
            "select count(*) as rows, "
            "coalesce(sum(coalesce(pg_column_size(trade_date),0) + coalesce(pg_column_size(adj_close),0)),0) as bytes "
            "from price_history where ticker_id = $1 and adj_close is not null",
            heavy["ticker_id"],
        )
        out[f"/tickers/{heavy['symbol']}/prices?lookback=max"] = (
            int(r["rows"]), int(r["bytes"])
        )

    # /rankings — latest cross-section, one row per active covered ticker.
    active_model = await pool.fetchrow(
        "select model_version_id from model_versions where status = 'production' "
        "order by promoted_at desc nulls last limit 1"
    )
    if active_model is None:
        active_model = await pool.fetchrow(
            "select model_version_id from model_versions order by created_at desc limit 1"
        )
    if active_model is not None:
        mv = active_model["model_version_id"]
        r = await pool.fetchrow(
            """
            select count(*) as rows,
                   coalesce(sum(
                       coalesce(pg_column_size(t.ticker_id),0) + coalesce(pg_column_size(t.symbol),0)
                     + coalesce(pg_column_size(t.name),0) + coalesce(pg_column_size(t.sector),0)
                     + coalesce(pg_column_size(p.direction_prob),0) + coalesce(pg_column_size(p.confidence),0)
                     + coalesce(pg_column_size(p.as_of_date),0)
                   ),0) as bytes
              from predictions p join tickers t on t.ticker_id = p.ticker_id
             where p.model_version_id = $1 and p.horizon = '6M'
               and p.as_of_date = (select max(as_of_date) from predictions
                                    where model_version_id = $1 and horizon = '6M')
            """,
            mv,
        )
        out["/rankings?horizon=6M"] = (int(r["rows"]), int(r["bytes"]))

    return out


async def amain() -> None:
    async with pool_context() as pool:
        n_active = await pool.fetchval("select count(*) from tickers where active = true")

        full_total, per_table = await measure_full_pull(pool)

        today = date.today()
        month_start = today.replace(day=1)
        runs_this_month = await count_inference_runs(pool, month_start)
        # Typical cadence: ~4-5 Fridays + 1 month-start = ~5 inference runs/month.
        est_runs = max(runs_this_month, 5)

        api = await measure_api_calls(pool)

    print(f"\nActive tickers: {n_active}\n")

    print("=" * 64)
    print("PATH 1  pipeline load_frames() — full-history pull, per run")
    print("=" * 64)
    for table, (rows, nbytes) in per_table.items():
        print(f"  {table:<16} {rows:>9,} rows   {_fmt(nbytes):>12}")
    print(f"  {'TOTAL / run':<16} {'':>9}        {_fmt(full_total):>12}")
    print()
    print(f"  inference runs so far this month (ingestion_runs): {runs_this_month}")
    print(f"  estimated runs/month (>= typical 5): {est_runs}")
    print(f"  => estimated pipeline egress/month:  {_fmt(full_total * est_runs)}")
    print()

    print("=" * 64)
    print("PATH 2  FastAPI read endpoints — bytes per single uncached call")
    print("=" * 64)
    for ep, (rows, nbytes) in api.items():
        print(f"  {ep:<42} {rows:>6,} rows  {_fmt(nbytes):>10}")
    print()
    print("  Multiply each by its monthly request count (Railway logs / Supabase")
    print("  log Reports) to attribute Path-2 egress. A CDN in front of Railway")
    print("  cuts ONLY this path, and only on cache hits.")
    print()

    print("=" * 64)
    print("Authoritative total: Supabase dashboard -> Reports -> Usage -> Egress.")
    print("If that figure ~= the Path-1 estimate above, the fix is the pipeline")
    print("(incremental load_frames), not a CDN.")
    print("=" * 64)


if __name__ == "__main__":
    asyncio.run(amain())
