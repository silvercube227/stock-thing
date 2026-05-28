"""Daily ingestion + inference orchestrator.

Run by launchd after NYSE close on weekdays. Exits 0 on success, including
the "not a trading day" early-exit so launchd does not treat holidays as errors.

Stages (in order):
  1. prices_daily  — incremental price ingest               every trading day
  2. sentiment     — FinBERT headline scoring + aggregation  every trading day
  3. fundamentals  — EDGAR companyfacts refresh              Fridays only
  4. gbm_inference — GBDT retrain + write predictions        Fridays + month-starts
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from datetime import date
from typing import Any

from backend.ingestion.calendar import is_trading_day, trading_days_between
from backend.ingestion.db import pool_context
from backend.ingestion.fundamentals import ingest_fundamentals
from backend.ingestion.headlines import ingest_sentiment
from backend.ingestion.prices import ingest_recent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schedule predicates
# ---------------------------------------------------------------------------


def _is_first_trading_day_of_month(today: date) -> bool:
    first = date(today.year, today.month, 1)
    days = trading_days_between(first, today)
    return bool(days) and days[0] == today


def _run_fundamentals_today(today: date) -> bool:
    """EDGAR filings change quarterly; pulling once a week (Fridays) is enough."""
    return today.weekday() == 4


def _run_inference_today(today: date) -> bool:
    """Retrain + score on Fridays (weekly cadence for rank stability) and on
    the first trading day of each month (keeps predictions current at month-turn)."""
    return today.weekday() == 4 or _is_first_trading_day_of_month(today)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def _run(today: date) -> int:
    async with pool_context() as pool:
        run_id = int(
            await pool.fetchval(
                "insert into ingestion_runs (job_name) values ('daily_pipeline') returning run_id"
            )
        )
        errors: list[str] = []
        meta: dict[str, Any] = {"date": today.isoformat(), "stages": {}}

        # Stage 1: prices
        log.info("[1/4] prices — incremental ingest")
        try:
            r = await ingest_recent(pool)
            log.info(
                "  prices: %s  rows=%d  drifted=%s  failed=%s",
                r.status, r.rows_inserted,
                r.drifted_tickers or "none",
                r.failed_tickers or "none",
            )
            meta["stages"]["prices"] = {"status": r.status, "rows": r.rows_inserted}
            if r.status == "failed":
                errors.append("prices")
        except Exception as exc:
            log.exception("prices stage raised")
            errors.append("prices")
            meta["stages"]["prices"] = {"status": "failed", "error": str(exc)}

        # Stage 2: sentiment
        log.info("[2/4] sentiment — FinBERT scoring + rolling aggregation")
        try:
            r = await ingest_sentiment(pool)
            log.info(
                "  sentiment: %s  headlines=%d  days_updated=%d  failed=%s",
                r.status, r.headlines_inserted, r.sentiment_days_upserted,
                r.failed_tickers or "none",
            )
            meta["stages"]["sentiment"] = {
                "status": r.status,
                "headlines": r.headlines_inserted,
                "days": r.sentiment_days_upserted,
            }
            if r.status == "failed":
                errors.append("sentiment")
        except Exception as exc:
            log.exception("sentiment stage raised")
            errors.append("sentiment")
            meta["stages"]["sentiment"] = {"status": "failed", "error": str(exc)}

        # Stage 3: fundamentals (Fridays only)
        if _run_fundamentals_today(today):
            log.info("[3/4] fundamentals — EDGAR companyfacts refresh")
            try:
                r = await ingest_fundamentals(pool)
                log.info(
                    "  fundamentals: %s  rows=%d  failed=%s  no_cik=%d",
                    r.status, r.rows_inserted,
                    r.failed_tickers or "none",
                    len(r.skipped_no_cik),
                )
                meta["stages"]["fundamentals"] = {"status": r.status, "rows": r.rows_inserted}
                if r.status == "failed":
                    errors.append("fundamentals")
            except Exception as exc:
                log.exception("fundamentals stage raised")
                errors.append("fundamentals")
                meta["stages"]["fundamentals"] = {"status": "failed", "error": str(exc)}
        else:
            log.info("[3/4] fundamentals — skipped (not Friday)")

        # Stage 4: GBM inference (Fridays + month-starts)
        if _run_inference_today(today):
            log.info("[4/4] gbm_inference — retrain rankers + score cross-section")
            gbm_run_id = int(
                await pool.fetchval(
                    "insert into ingestion_runs (job_name) values ('gbm_inference') returning run_id"
                )
            )
            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "backend.ml.gbm_inference",
                    "--target", "rank",
                    "--horizons", "3M", "6M", "1Y",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                stdout_bytes, _ = await proc.communicate()
                output = stdout_bytes.decode(errors="replace").strip() if stdout_bytes else ""
                if proc.returncode == 0:
                    log.info("  gbm_inference: success\n%s", output)
                    await pool.execute(
                        "update ingestion_runs set finished_at=now(), status='success' where run_id=$1",
                        gbm_run_id,
                    )
                    meta["stages"]["gbm_inference"] = {"status": "success"}
                else:
                    log.error("  gbm_inference: FAILED (rc=%d)\n%s", proc.returncode, output)
                    errors.append("gbm_inference")
                    await pool.execute(
                        """update ingestion_runs
                           set finished_at=now(), status='failed', error_message=$2
                         where run_id=$1""",
                        gbm_run_id,
                        f"exit {proc.returncode}",
                    )
                    meta["stages"]["gbm_inference"] = {
                        "status": "failed", "rc": proc.returncode
                    }
            except Exception as exc:
                log.exception("gbm_inference stage raised")
                errors.append("gbm_inference")
                await pool.execute(
                    """update ingestion_runs
                       set finished_at=now(), status='failed', error_message=$2
                     where run_id=$1""",
                    gbm_run_id,
                    str(exc),
                )
                meta["stages"]["gbm_inference"] = {"status": "failed", "error": str(exc)}
        else:
            log.info("[4/4] gbm_inference — skipped (not Friday or month-start)")

        overall = "success" if not errors else "partial"
        await pool.execute(
            """update ingestion_runs
                  set finished_at=now(), status=$2, metadata=$3
                where run_id=$1""",
            run_id,
            overall,
            json.dumps(meta),
        )
        if errors:
            log.warning("=== daily pipeline %s — failed stages: %s ===", overall, errors)
        else:
            log.info("=== daily pipeline complete: %s ===", overall)
        return 1 if errors else 0


def main() -> None:
    today = date.today()
    if not is_trading_day(today):
        log.info("Not a NYSE trading day (%s) — exiting cleanly.", today)
        sys.exit(0)
    log.info("=== daily pipeline %s ===", today)
    sys.exit(asyncio.run(_run(today)))


if __name__ == "__main__":
    main()
