"""Background orchestrator for a user-added (off-index) ticker.

Spawned (detached) by the API's POST /tickers. Steps:
  1. Resolve metadata from yfinance (.info): name, asset_type, GICS sector/industry
     (Yahoo taxonomy mapped to GICS, validated against sectors already in the DB),
     shares_outstanding; CIK from the SEC ticker map for equities.
  2. Upsert the `tickers` row (active = true, user_added = true).
  3. Ingest prices (full history) + fundamentals (if CIK) + sentiment for just this ticker.
  4. Score it against the current S&P cross-section with the production model
     (subprocess `gbm_inference --score-ticker`), which writes only this ticker's rows.

The whole run is tracked in a single `ingestion_runs` row (job_name
`add_ticker:<SYMBOL>`). A try/finally GUARANTEES a terminal status is written on
every exit path — including a non-zero scoring subprocess — so the frontend poll
never hangs on `running`. (A hard kill of this process is caught by the status
endpoint's stale-run guard.)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import traceback
from datetime import date

import asyncpg

from backend.config import get_settings
from backend.ingestion.db import pool_context
from backend.ingestion.fundamentals import ingest_fundamentals
from backend.ingestion.headlines import ingest_sentiment
from backend.ingestion.prices import ingest_full_history
from scripts.backfill_ciks import fetch_sec_mapping

# Yahoo Finance uses its own sector taxonomy; the S&P universe in `tickers` is
# labeled with GICS sectors (from the Wikipedia seed). Map Yahoo -> GICS so an
# added ticker groups with its real peers. Anything not here (or not present in
# the DB's sector set) is stored as null rather than guessing a peer group.
_YAHOO_TO_GICS = {
    "Technology": "Information Technology",
    "Financial Services": "Financials",
    "Healthcare": "Health Care",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Energy": "Energy",
    "Industrials": "Industrials",
    "Basic Materials": "Materials",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
    "Communication Services": "Communication Services",
}


def _fetch_yf_info(symbol: str) -> dict:
    """Blocking yfinance .info pull (call via asyncio.to_thread)."""
    import yfinance as yf

    return yf.Ticker(symbol).info or {}


def _asset_type_from_quote(quote_type: str | None) -> str | None:
    qt = (quote_type or "").upper()
    if qt == "EQUITY":
        return "equity"
    if qt == "ETF":
        return "etf"
    return None  # INDEX / MUTUALFUND / CRYPTO / unknown — unsupported


async def _resolve_metadata(pool: asyncpg.Pool, symbol: str) -> dict:
    """Build the ticker row fields from yfinance + the SEC CIK map."""
    info = await asyncio.to_thread(_fetch_yf_info, symbol)
    asset_type = _asset_type_from_quote(info.get("quoteType"))
    if asset_type is None:
        raise ValueError(
            f"{symbol}: unsupported instrument type {info.get('quoteType')!r} "
            "(only equity/ETF are supported)"
        )

    name = info.get("longName") or info.get("shortName")
    gics = _YAHOO_TO_GICS.get(info.get("sector"))
    known = {
        r["sector"]
        for r in await pool.fetch("select distinct sector from tickers where sector is not null")
    }
    sector = gics if gics in known else None  # store null rather than a wrong peer group
    industry = info.get("industry") or None
    shares = info.get("sharesOutstanding")
    shares_outstanding = int(shares) if isinstance(shares, (int, float)) and shares > 0 else None

    cik = None
    if asset_type == "equity":
        settings = get_settings()
        mapping = await fetch_sec_mapping(settings.sec_edgar_user_agent)
        cik = mapping.get(symbol.upper())

    return {
        "name": name,
        "asset_type": asset_type,
        "sector": sector,
        "industry": industry,
        "shares_outstanding": shares_outstanding,
        "cik": cik,
    }


async def _upsert_ticker(pool: asyncpg.Pool, symbol: str, meta: dict) -> int:
    """Insert the user-added ticker (or reuse/reactivate an existing row). Returns ticker_id."""
    existing = await pool.fetchrow(
        "select ticker_id, active from tickers where upper(symbol) = upper($1)", symbol
    )
    if existing is not None:
        ticker_id = int(existing["ticker_id"])
        # Reactivate a removed-from-index name as user-added; never re-flag an
        # already-active index member. Fill metadata gaps without clobbering.
        await pool.execute(
            """
            update tickers set
                active             = true,
                user_added         = case when active then user_added else true end,
                removed_at         = null,
                name               = coalesce(name, $2),
                sector             = coalesce(sector, $3),
                industry           = coalesce(industry, $4),
                cik                = coalesce(cik, $5),
                shares_outstanding = coalesce(shares_outstanding, $6)
             where ticker_id = $1
            """,
            ticker_id, meta["name"], meta["sector"], meta["industry"],
            meta["cik"], meta["shares_outstanding"],
        )
        return ticker_id

    row = await pool.fetchrow(
        """
        insert into tickers (symbol, name, asset_type, sector, industry, cik,
                             shares_outstanding, active, user_added)
        values ($1, $2, $3, $4, $5, $6, $7, true, true)
        returning ticker_id
        """,
        symbol.upper(), meta["name"], meta["asset_type"], meta["sector"],
        meta["industry"], meta["cik"], meta["shares_outstanding"],
    )
    return int(row["ticker_id"])


def _parse_score_outcome(output: str) -> dict | None:
    """Extract the single JSON outcome line printed by `--score-ticker`."""
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "status" in obj:
                return obj
    return None


async def _score_subprocess(symbol: str) -> tuple[int, str, dict | None]:
    """Run gbm_inference --score-ticker as a subprocess; return (rc, output, outcome)."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "backend.ml.gbm_inference", "--score-ticker", symbol,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    stdout_bytes, _ = await proc.communicate()
    output = stdout_bytes.decode(errors="replace").strip() if stdout_bytes else ""
    return proc.returncode, output, _parse_score_outcome(output)


async def _finish(
    pool: asyncpg.Pool, run_id: int, status: str, *,
    error: str | None = None, rows: int | None = None, metadata: dict | None = None,
) -> None:
    await pool.execute(
        """
        update ingestion_runs set
            finished_at = now(), status = $2, error_message = $3,
            rows_inserted = $4, metadata = $5
         where run_id = $1
        """,
        run_id, status, (error[:2000] if error else None), rows,
        json.dumps(metadata or {}),
    )


async def _ingest_ticker(pool: asyncpg.Pool, ticker_id: int, symbol: str, metadata: dict) -> None:
    """Idempotent ingest of one ticker's prices + fundamentals + sentiment."""
    await ingest_full_history(pool, tickers=[(ticker_id, symbol)], start_date=date(2010, 1, 1))
    if metadata["cik"]:
        await ingest_fundamentals(pool, tickers=[(ticker_id, symbol, metadata["cik"])])
    await ingest_sentiment(pool, tickers=[(ticker_id, symbol)])


async def _record_terminal(
    run_id: int, status: str, *,
    error: str | None = None, rows: int | None = None, metadata: dict | None = None,
) -> None:
    """Write the run's terminal status on a FRESH short-lived pool.

    The failure we're recording is often a dropped Supabase connection, which also
    poisons the ingest pool — writing the status on that same pool would raise too,
    stranding the run at `running`. The UI misreads a stuck `running` as a scoring
    hang ("remove and re-add"), so guaranteeing a terminal write is what makes the
    error honest (and re-adds effective). Best-effort: the status endpoint's
    stale-run guard is the last-resort backstop if even this write can't land.
    """
    try:
        async with pool_context(command_timeout=60, max_size=2) as p:
            await _finish(p, run_id, status, error=error, rows=rows, metadata=metadata)
    except Exception:  # noqa: BLE001
        pass


async def run_add(symbol: str, run_id: int | None) -> int:
    symbol = symbol.upper().strip()
    meta: dict = {"symbol": symbol}
    # Small pool: the API's uvicorn and the score subprocess each hold their own
    # connections, so a fat worker pool adds free-tier connection pressure — the very
    # thing that drops the ingest mid-pull.
    async with pool_context(command_timeout=300, max_size=2) as pool:
        if run_id is None:
            run_id = int(
                await pool.fetchval(
                    "insert into ingestion_runs (job_name) values ($1) returning run_id",
                    f"add_ticker:{symbol}",
                )
            )
        try:
            metadata = await _resolve_metadata(pool, symbol)
            ticker_id = await _upsert_ticker(pool, symbol, metadata)
            meta["ticker_id"] = ticker_id
            meta["sector"] = metadata["sector"]

            # Bounded retries: a transient pooler drop self-heals — asyncpg replaces
            # the dead connection and every ingest step is an idempotent upsert.
            last_exc: Exception | None = None
            for attempt in range(3):
                try:
                    await _ingest_ticker(pool, ticker_id, symbol, metadata)
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    await asyncio.sleep(2 * (attempt + 1))
            if last_exc is not None:
                raise last_exc

            rc, output, outcome = await _score_subprocess(symbol)
            if rc != 0 or outcome is None:
                meta["outcome"] = "failed"
                await _record_terminal(run_id, "failed",
                                       error=f"scoring exit {rc}\n{output[-1500:]}", metadata=meta)
                return 1
            if outcome["status"] == "insufficient_history":
                meta["outcome"] = "insufficient_history"
                await _record_terminal(run_id, "success", rows=0, metadata=meta)
                return 0
            meta["outcome"] = "scored"
            meta["ranks"] = outcome.get("ranks")
            await _record_terminal(run_id, "success",
                                   rows=len(outcome.get("ranks", {})), metadata=meta)
            return 0
        except Exception:
            meta["outcome"] = "failed"
            await _record_terminal(run_id, "failed", error=traceback.format_exc(), metadata=meta)
            return 1


async def claim_queued(pool: asyncpg.Pool, limit: int) -> list[tuple[int, str]]:
    """Atomically claim up to `limit` queued add-ticker jobs, oldest first.

    `for update skip locked` inside the same statement that flips the status makes a
    double-drain (two launchd firings overlapping, or a manual run beside the agent)
    safe: a row can only ever be claimed once.
    """
    rows = await pool.fetch(
        """
        with claimed as (
            select run_id from ingestion_runs
             where status = 'queued' and job_name like 'add_ticker:%'
             order by started_at
             limit $1
             for update skip locked
        )
        update ingestion_runs r
           set status = 'running', started_at = now()
          from claimed
         where r.run_id = claimed.run_id
        returning r.run_id, r.job_name
        """,
        limit,
    )
    return [(int(r["run_id"]), r["job_name"].split(":", 1)[1]) for r in rows]


async def drain(limit: int) -> int:
    """Run every queued add-ticker job. Entry point for the local drain agent.

    The hosted read-API cannot execute the worker (no pandas/lightgbm/torch, no model
    artifact), so it records the request as 'queued' instead of spawning a process that
    would die silently. This is the consumer that makes those requests actually happen.
    """
    async with pool_context(command_timeout=300, max_size=2) as pool:
        jobs = await claim_queued(pool, limit)
    if not jobs:
        return 0
    print(f"draining {len(jobs)} queued add-ticker job(s): "
          f"{', '.join(s for _, s in jobs)}")
    worst = 0
    for run_id, symbol in jobs:
        # Sequential on purpose: each job ingests prices and then shells out to a
        # LightGBM scoring subprocess, and this shares a laptop with the daily pipeline.
        rc = await run_add(symbol, run_id)
        print(f"  {symbol}: rc={rc}")
        worst = max(worst, rc)
    return worst


def main() -> int:
    p = argparse.ArgumentParser(description="Ingest + score a single user-added ticker")
    p.add_argument("--symbol", help="symbol to add (omit with --drain)")
    p.add_argument("--run-id", type=int, default=None,
                   help="existing ingestion_runs row to update (the API creates it); "
                        "a new row is created when omitted")
    p.add_argument("--drain", action="store_true",
                   help="run every add-ticker job the hosted API left 'queued', "
                        "instead of adding one symbol. This is what the local drain "
                        "agent runs; safe to run concurrently with itself.")
    p.add_argument("--limit", type=int, default=5,
                   help="max queued jobs to claim in one --drain pass (default 5)")
    args = p.parse_args()
    if args.drain:
        if args.symbol:
            p.error("--drain takes no --symbol")
        return asyncio.run(drain(args.limit))
    if not args.symbol:
        p.error("--symbol is required unless --drain is given")
    return asyncio.run(run_add(args.symbol, args.run_id))


if __name__ == "__main__":
    raise SystemExit(main())
