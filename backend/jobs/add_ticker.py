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


async def run_add(symbol: str, run_id: int | None) -> int:
    symbol = symbol.upper().strip()
    async with pool_context(command_timeout=300) as pool:
        if run_id is None:
            run_id = int(
                await pool.fetchval(
                    "insert into ingestion_runs (job_name) values ($1) returning run_id",
                    f"add_ticker:{symbol}",
                )
            )
        meta: dict = {"symbol": symbol}
        try:
            metadata = await _resolve_metadata(pool, symbol)
            ticker_id = await _upsert_ticker(pool, symbol, metadata)
            meta["ticker_id"] = ticker_id
            meta["sector"] = metadata["sector"]

            await ingest_full_history(pool, tickers=[(ticker_id, symbol)], start_date=date(2010, 1, 1))
            if metadata["cik"]:
                await ingest_fundamentals(pool, tickers=[(ticker_id, symbol, metadata["cik"])])
            await ingest_sentiment(pool, tickers=[(ticker_id, symbol)])

            rc, output, outcome = await _score_subprocess(symbol)
            if rc != 0 or outcome is None:
                meta["outcome"] = "failed"
                await _finish(pool, run_id, "failed",
                              error=f"scoring exit {rc}\n{output[-1500:]}", metadata=meta)
                return 1
            if outcome["status"] == "insufficient_history":
                meta["outcome"] = "insufficient_history"
                await _finish(pool, run_id, "success", rows=0, metadata=meta)
                return 0
            meta["outcome"] = "scored"
            meta["ranks"] = outcome.get("ranks")
            await _finish(pool, run_id, "success",
                          rows=len(outcome.get("ranks", {})), metadata=meta)
            return 0
        except Exception:
            meta["outcome"] = "failed"
            await _finish(pool, run_id, "failed", error=traceback.format_exc(), metadata=meta)
            return 1


def main() -> int:
    p = argparse.ArgumentParser(description="Ingest + score a single user-added ticker")
    p.add_argument("--symbol", required=True)
    p.add_argument("--run-id", type=int, default=None,
                   help="existing ingestion_runs row to update (the API creates it); "
                        "a new row is created when omitted")
    args = p.parse_args()
    return asyncio.run(run_add(args.symbol, args.run_id))


if __name__ == "__main__":
    raise SystemExit(main())
