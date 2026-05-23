"""SEC EDGAR fundamentals ingestion via the companyfacts XBRL API.

For each ticker with a CIK, we:
  1. Pull https://data.sec.gov/api/xbrl/companyfacts/CIK<padded>.json
  2. Walk the us-gaap concept tree, applying a fallback list per field
     (e.g. revenue lives under several XBRL tag names across vintages
     and accounting standards)
  3. Group entries by accession number (one accn = one filing) and pick
     each concept's value whose `start`/`end` span matches the filing's
     natural period (annual for 10-K, quarterly for 10-Q)
  4. Compute derived ratios (gross_margin, operating_margin, debt, fcf)
  5. Upsert into `fundamentals`

CRITICAL: every row's `filed_at` is the SEC-received timestamp, NOT the
fiscal `period_end`. All point-in-time joins must use `filed_at` or we'll
leak future earnings data into past samples.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Iterable

import asyncpg
import httpx

from backend.config import get_settings

log = logging.getLogger(__name__)

EDGAR_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# Concept-name fallback lists. First match in `parse_companyfacts` wins.
# Order matters: prefer newer ASC 606 names, fall back to older labels.
CONCEPT_FALLBACKS: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "net_income": ["NetIncomeLoss"],
    "gross_profit": ["GrossProfit"],
    "cost_of_revenue": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
    ],
    "operating_income": ["OperatingIncomeLoss"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "lt_debt_noncurrent": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "lt_debt_current": ["LongTermDebtCurrent"],
    "short_term_debt": ["ShortTermBorrowings"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
}

# Period-length tolerances (days) for matching the filing's natural span.
ANNUAL_RANGE = (340, 380)
QUARTERLY_RANGE = (80, 100)


# =============================================================
# Result dataclasses
# =============================================================


@dataclass
class TickerResult:
    ticker_id: int
    symbol: str
    cik: str | None
    rows_written: int = 0
    error: str | None = None


@dataclass
class IngestionResult:
    started_at: datetime
    finished_at: datetime
    rows_inserted: int = 0
    failed_tickers: list[str] = field(default_factory=list)
    skipped_no_cik: list[str] = field(default_factory=list)
    per_ticker: list[TickerResult] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.failed_tickers and self.rows_inserted == 0:
            return "failed"
        if self.failed_tickers:
            return "partial"
        return "success"


# =============================================================
# Pure parsing logic (no I/O)
# =============================================================


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _is_natural_period(entry: dict, form: str) -> bool:
    """Filter out cumulative / year-to-date rows; keep only the filing's natural
    standalone period: 1 year for 10-K, 1 quarter for 10-Q.

    Balance-sheet items (no `start`, just `end`) are always point-in-time and
    pass through.
    """
    if "start" not in entry:
        return True
    try:
        days = (_parse_date(entry["end"]) - _parse_date(entry["start"])).days
    except (KeyError, ValueError):
        return False
    if form == "10-K":
        return ANNUAL_RANGE[0] <= days <= ANNUAL_RANGE[1]
    if form == "10-Q":
        return QUARTERLY_RANGE[0] <= days <= QUARTERLY_RANGE[1]
    return False


def parse_companyfacts(facts_json: dict, ticker_id: int) -> list[dict]:
    """Convert EDGAR companyfacts JSON into a list of upsert-ready row dicts.

    A single filing (one accn) can contain restated prior-year figures alongside
    current-year figures. We must pick the value whose period `end` matches the
    filing's natural reporting period — otherwise the parser silently associates
    e.g. FY2020 revenue with a 10-K filed for FY2022.

    Two-pass approach:
      1. Group every (form, period) entry by accn, recording max(end) bounded
         by the filing date (entries with end > filed are forward-looking and
         excluded from the period_end determination).
      2. For each accn × field, choose the entry whose end matches the
         determined period_end (±5 days for fiscal calendar drift), preferring
         the leftmost concept-name in the fallback list.

    Pure function — exercised by unit tests without network or DB.
    """
    us_gaap = facts_json.get("facts", {}).get("us-gaap", {})

    # accn -> {filing_type, filed_at (str), period_end (str),
    #          candidates: {field: [(end, val, concept_priority), ...]}}
    accn_data: dict[str, dict] = {}

    for field_name, concept_names in CONCEPT_FALLBACKS.items():
        for priority, concept in enumerate(concept_names):
            entries = us_gaap.get(concept, {}).get("units", {}).get("USD", [])
            for entry in entries:
                form = entry.get("form")
                if form not in ("10-K", "10-Q"):
                    continue
                if not _is_natural_period(entry, form):
                    continue
                accn = entry.get("accn")
                filed = entry.get("filed")
                end = entry.get("end")
                if not accn or not filed or not end:
                    continue
                # A filing cannot describe a period that ends after the filing
                # was received. Drop forward-looking entries entirely so they
                # can't pollute period_end either.
                if end > filed:
                    continue

                f = accn_data.setdefault(
                    accn,
                    {
                        "filing_type": form,
                        "filed_at": filed,
                        "period_end": end,
                        "candidates": {},
                    },
                )
                if end > f["period_end"]:
                    f["period_end"] = end
                f["candidates"].setdefault(field_name, []).append(
                    (end, entry["val"], priority)
                )

    rows: list[dict] = []
    for accn, f in accn_data.items():
        target_end = f["period_end"]
        target_end_d = _parse_date(target_end)
        values: dict[str, float] = {}
        for field_name, cands in f["candidates"].items():
            # Best = matches target period (±5 days), lowest priority idx.
            best: tuple[int, float] | None = None  # (priority, val)
            for end, val, priority in cands:
                if abs((_parse_date(end) - target_end_d).days) > 5:
                    continue
                if best is None or priority < best[0]:
                    best = (priority, val)
            if best is not None:
                values[field_name] = best[1]

        v = values
        revenue = v.get("revenue")
        gross_profit = v.get("gross_profit")
        cost = v.get("cost_of_revenue")
        if gross_profit is None and revenue is not None and cost is not None:
            gross_profit = revenue - cost
        gross_margin = (
            gross_profit / revenue if revenue and gross_profit is not None else None
        )

        op_income = v.get("operating_income")
        operating_margin = (
            op_income / revenue if revenue and op_income is not None else None
        )

        equity = v.get("equity")

        lt_nc = v.get("lt_debt_noncurrent") or 0
        lt_c = v.get("lt_debt_current") or 0
        st = v.get("short_term_debt") or 0
        total_debt_raw = lt_nc + lt_c + st
        total_debt = total_debt_raw if total_debt_raw > 0 else None

        ocf = v.get("operating_cash_flow")
        capex = v.get("capex")
        fcf = (ocf - (capex or 0)) if ocf is not None else None

        rows.append(
            {
                "ticker_id": ticker_id,
                "accession_number": accn,
                "filing_type": f["filing_type"],
                "period_end": _parse_date(f["period_end"]),
                "filed_at": _parse_date(f["filed_at"]),
                "revenue": revenue,
                "net_income": v.get("net_income"),
                "gross_margin": gross_margin,
                "operating_margin": operating_margin,
                "total_debt": total_debt,
                "total_equity": equity,
                "fcf": fcf,
            }
        )
    return rows


# =============================================================
# Network
# =============================================================


async def fetch_companyfacts(client: httpx.AsyncClient, cik: str) -> dict | None:
    """Fetch and parse the EDGAR companyfacts JSON for a CIK.

    Returns None on 404 (no XBRL data for this filer).
    """
    url = EDGAR_COMPANYFACTS_URL.format(cik=cik)
    resp = await client.get(url)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


# =============================================================
# DB I/O
# =============================================================


_UPSERT_SQL = """
insert into fundamentals (
    ticker_id, accession_number, filing_type, period_end, filed_at,
    revenue, net_income, gross_margin, operating_margin,
    total_debt, total_equity, fcf, ingested_at
) values (
    $1, $2, $3, $4, $5,
    $6, $7, $8, $9,
    $10, $11, $12, now()
)
on conflict (ticker_id, accession_number) do update set
    filing_type      = excluded.filing_type,
    period_end       = excluded.period_end,
    filed_at         = excluded.filed_at,
    revenue          = excluded.revenue,
    net_income       = excluded.net_income,
    gross_margin     = excluded.gross_margin,
    operating_margin = excluded.operating_margin,
    total_debt       = excluded.total_debt,
    total_equity     = excluded.total_equity,
    fcf              = excluded.fcf,
    ingested_at      = now()
"""


async def _upsert_filings(conn: asyncpg.Connection, rows: list[dict]) -> int:
    if not rows:
        return 0
    payload = [
        (
            r["ticker_id"],
            r["accession_number"],
            r["filing_type"],
            r["period_end"],
            r["filed_at"],
            r["revenue"],
            r["net_income"],
            r["gross_margin"],
            r["operating_margin"],
            r["total_debt"],
            r["total_equity"],
            r["fcf"],
        )
        for r in rows
    ]
    async with conn.transaction():
        await conn.executemany(_UPSERT_SQL, payload)
    return len(payload)


async def _fetch_equities_with_cik(pool: asyncpg.Pool) -> list[tuple[int, str, str]]:
    rows = await pool.fetch(
        """
        select ticker_id, symbol, cik
          from tickers
         where active = true
           and asset_type = 'equity'
         order by ticker_id
        """
    )
    return [(r["ticker_id"], r["symbol"], r["cik"]) for r in rows]


async def _log_run_start(pool: asyncpg.Pool) -> int:
    row = await pool.fetchrow(
        "insert into ingestion_runs (job_name) values ($1) returning run_id",
        "fundamentals",
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
        json.dumps(
            {
                "failed_tickers": result.failed_tickers,
                "skipped_no_cik": result.skipped_no_cik,
                "ticker_count": len(result.per_ticker),
            }
        ),
    )


# =============================================================
# Public entry point
# =============================================================


async def ingest_fundamentals(
    pool: asyncpg.Pool,
    tickers: Iterable[tuple[int, str, str | None]] | None = None,
    concurrency: int = 5,
) -> IngestionResult:
    """Pull fundamentals for each equity with a CIK. ETFs are skipped.

    `tickers` is an iterable of (ticker_id, symbol, cik); if None, all
    active equities with a non-null cik are processed.
    """
    settings = get_settings()
    if tickers is None:
        candidates = await _fetch_equities_with_cik(pool)
    else:
        candidates = [(tid, sym, cik) for tid, sym, cik in tickers]

    started = datetime.now(timezone.utc)
    run_id = await _log_run_start(pool)

    # SEC requires a User-Agent identifying the requester per their fair-access
    # policy. They allow ~10 req/sec; concurrency=5 keeps us well under.
    headers = {
        "User-Agent": settings.sec_edgar_user_agent,
        "Accept": "application/json",
    }
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=60.0, headers=headers) as client:
        async def one(ticker_id: int, symbol: str, cik: str | None) -> TickerResult:
            if not cik:
                return TickerResult(ticker_id, symbol, cik, error="no CIK")
            async with sem:
                try:
                    facts = await fetch_companyfacts(client, cik)
                except Exception as exc:  # noqa: BLE001
                    return TickerResult(ticker_id, symbol, cik, error=f"fetch: {exc}")
                if facts is None:
                    return TickerResult(ticker_id, symbol, cik, error="no companyfacts (404)")
                try:
                    rows = parse_companyfacts(facts, ticker_id)
                except Exception as exc:  # noqa: BLE001
                    return TickerResult(ticker_id, symbol, cik, error=f"parse: {exc}")
                if not rows:
                    return TickerResult(ticker_id, symbol, cik, rows_written=0)
                async with pool.acquire() as conn:
                    n = await _upsert_filings(conn, rows)
                return TickerResult(ticker_id, symbol, cik, rows_written=n)

        per_ticker = await asyncio.gather(*(one(t, s, c) for t, s, c in candidates))

    finished = datetime.now(timezone.utc)
    result = IngestionResult(
        started_at=started,
        finished_at=finished,
        rows_inserted=sum(tr.rows_written for tr in per_ticker),
        failed_tickers=[tr.symbol for tr in per_ticker if tr.error and "no CIK" not in (tr.error or "")],
        skipped_no_cik=[tr.symbol for tr in per_ticker if tr.error == "no CIK"],
        per_ticker=list(per_ticker),
    )
    await _log_run_finish(pool, run_id, result)
    return result
