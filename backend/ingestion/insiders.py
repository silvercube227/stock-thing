"""SEC Form 3/4/5 insider-transaction ingestion (Cohen-Malloy-Pomorski 2012).

Two ingest paths, both landing rows in `insider_transactions`:

  1. BULK BACKFILL from the SEC DERA "Insider Transactions Data Sets" — one ZIP per
     calendar quarter (2006+), each holding tab-delimited SUBMISSION / REPORTINGOWNER
     / NONDERIV_TRANS tables. ~82 requests cover the whole history vs ~1M per-filing
     XML fetches. `parse_insider_dataset` is a PURE function over the ZIP bytes so it
     unit-tests on synthetic fixtures.

  2. DAILY INCREMENTAL from the per-CIK submissions JSON (`data.sec.gov/submissions/
     CIK{cik}.json`, `form == "4"`) + the Form 4 primary XML (`parse_form4_xml`).

PIT invariant: the join key is FILING_DATE (SEC receipt of the Form 4), never the
transaction date — the trade is unknowable to the market until the filing is public.

Design mirrors fundamentals.py / short_interest.py: pure parsers + a thin async DB
shell, `sec_edgar_user_agent` on every request, bounded concurrency.
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Iterator

import asyncpg
import httpx

from backend.config import get_settings
from backend.ingestion.db import pool_context

log = logging.getLogger(__name__)

# DERA insider datasets: one ZIP per quarter. 2006q1 is the earliest published.
DERA_URL = (
    "https://www.sec.gov/files/structureddata/data/"
    "insider-transactions-data-sets/{year}q{q}_form345.zip"
)
# Per-CIK submissions index (recent filings) for the daily incremental.
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
# Form 4 primary XML lives in the filing's directory.
FILING_XML_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"

# Form types we ingest. Form 4 (and its /A amendment) are the open-market trade
# reports; 3 (initial) and 5 (annual) are stored too so the feature layer sees the
# full picture, but the promotable signal is the Form 4 P/S flow.
INSIDER_FORMS = ("3", "4", "5", "3/A", "4/A", "5/A")

_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}


def _parse_dera_date(s: str | None) -> date | None:
    """Parse a DERA date: 'DD-MON-YYYY' (e.g. 31-JAN-2024) or ISO 'YYYY-MM-DD'."""
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        pass
    parts = s.split("-")
    if len(parts) == 3 and parts[1].upper() in _MONTHS:
        try:
            return date(int(parts[2]), _MONTHS[parts[1].upper()], int(parts[0]))
        except (ValueError, KeyError):
            return None
    return None


def _to_int(s: str | None) -> int | None:
    try:
        return int(str(s).strip())
    except (TypeError, ValueError):
        return None


def _to_float(s: str | None) -> float | None:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _iter_tsv(z: zipfile.ZipFile, name: str) -> Iterator[dict]:
    """Yield each row of a DERA TSV as a header->value dict (tab-delimited, unquoted)."""
    if name not in z.namelist():
        return
    with z.open(name) as fh:
        text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace", newline="")
        reader = csv.reader(text, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader, None)
        if not header:
            return
        for row in reader:
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            yield dict(zip(header, row))


def parse_insider_dataset(
    zip_bytes: bytes,
    cik_to_ticker: dict[int, int],
    forms: Iterable[str] = INSIDER_FORMS,
) -> list[dict]:
    """Parse one DERA quarterly ZIP into upsert-ready `insider_transactions` rows.

    Only filings whose issuer CIK is in `cik_to_ticker` (our universe) and whose
    DOCUMENT_TYPE is in `forms` are kept, so a full-market quarter reduces to our
    ~700 names before the join. `value` is signed by the acquired/disposed code
    (+buy / -sell), so the feature layer can sum a net dollar flow directly.
    """
    forms = set(forms)
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))

    # Pass 1: relevant submissions only (issuer in universe + right form).
    subs: dict[str, dict] = {}
    for r in _iter_tsv(z, "SUBMISSION.tsv"):
        if (r.get("DOCUMENT_TYPE") or "").strip() not in forms:
            continue
        cik = _to_int(r.get("ISSUERCIK"))
        tid = cik_to_ticker.get(cik) if cik is not None else None
        if tid is None:
            continue
        acc = (r.get("ACCESSION_NUMBER") or "").strip()
        fdate = _parse_dera_date(r.get("FILING_DATE"))
        if not acc or fdate is None:
            continue
        subs[acc] = {"ticker_id": tid, "filing_date": fdate,
                     "document_type": (r.get("DOCUMENT_TYPE") or "").strip()}
    if not subs:
        return []

    # Pass 2: first reporting owner per relevant accession.
    owners: dict[str, dict] = {}
    for r in _iter_tsv(z, "REPORTINGOWNER.tsv"):
        acc = (r.get("ACCESSION_NUMBER") or "").strip()
        if acc not in subs or acc in owners:
            continue
        rel = (r.get("RPTOWNER_RELATIONSHIP") or "").lower()
        owners[acc] = {
            "insider_cik": (r.get("RPTOWNERCIK") or "").strip() or None,
            "insider_name": (r.get("RPTOWNERNAME") or "").strip() or None,
            "is_officer": "officer" in rel,
            "is_director": "director" in rel,
            "is_ten_pct_owner": ("10%" in rel or "ten percent" in rel
                                 or "10 percent" in rel),
        }

    # Pass 3: non-derivative transactions grouped per relevant accession.
    trans: dict[str, list[dict]] = {}
    for r in _iter_tsv(z, "NONDERIV_TRANS.tsv"):
        acc = (r.get("ACCESSION_NUMBER") or "").strip()
        if acc not in subs:
            continue
        trans.setdefault(acc, []).append(r)

    rows: list[dict] = []
    for acc, sub in subs.items():
        owner = owners.get(acc, {})
        for idx, t in enumerate(trans.get(acc, [])):
            shares = _to_float(t.get("TRANS_SHARES"))
            price = _to_float(t.get("TRANS_PRICEPERSHARE"))
            disp = (t.get("TRANS_ACQUIRED_DISP_CD") or "").strip().upper()
            sign = 1.0 if disp == "A" else (-1.0 if disp == "D" else 1.0)
            value = sign * shares * price if shares is not None and price is not None else None
            rows.append({
                "ticker_id": sub["ticker_id"],
                "accession_number": acc,
                "transaction_idx": idx,
                "insider_cik": owner.get("insider_cik"),
                "insider_name": owner.get("insider_name"),
                "is_officer": bool(owner.get("is_officer", False)),
                "is_director": bool(owner.get("is_director", False)),
                "is_ten_pct_owner": bool(owner.get("is_ten_pct_owner", False)),
                "filing_date": sub["filing_date"],
                "acceptance_datetime": None,
                "transaction_date": _parse_dera_date(t.get("TRANS_DATE")),
                "transaction_code": (t.get("TRANS_CODE") or "").strip() or None,
                "shares": shares,
                "price_per_share": price,
                "value": value,
            })
    return rows


# =============================================================
# Form 4 XML parse (daily incremental)
# =============================================================


def _xml_text(node, path: str) -> str | None:
    if node is None:
        return None
    el = node.find(path)
    if el is None:
        return None
    # Form 4 wraps scalars in a <value> child; fall back to the element text.
    v = el.find("value")
    txt = (v.text if v is not None else el.text)
    return txt.strip() if txt else None


def parse_form4_xml(
    xml_bytes: bytes,
    ticker_id: int,
    accession_number: str,
    filing_date: date,
    acceptance_datetime: datetime | None = None,
) -> list[dict]:
    """Parse a single Form 4 primary XML document into `insider_transactions` rows.

    `filing_date` / `acceptance_datetime` come from the submissions index (the SEC
    receipt time), NOT from the XML (which only carries periodOfReport). Mirrors the
    bulk parser's row shape so both paths upsert identically.
    """
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []

    rel = root.find("reportingOwner/reportingOwnerRelationship")

    def _flag(tag: str) -> bool:
        t = _xml_text(rel, tag) if rel is not None else None
        return str(t).strip() in ("1", "true", "True")

    insider_cik = _xml_text(root, "reportingOwner/reportingOwnerId/rptOwnerCik")
    insider_name = _xml_text(root, "reportingOwner/reportingOwnerId/rptOwnerName")
    is_officer = _flag("isOfficer")
    is_director = _flag("isDirector")
    is_ten = _flag("isTenPercentOwner")

    rows: list[dict] = []
    txns = root.findall("nonDerivativeTable/nonDerivativeTransaction")
    for idx, tx in enumerate(txns):
        code = _xml_text(tx, "transactionCoding/transactionCode")
        shares = _to_float(_xml_text(tx, "transactionAmounts/transactionShares"))
        price = _to_float(_xml_text(tx, "transactionAmounts/transactionPricePerShare"))
        disp = (_xml_text(tx, "transactionAmounts/transactionAcquiredDisposedCode") or "").upper()
        tdate = None
        tdate_raw = _xml_text(tx, "transactionDate")
        if tdate_raw:
            try:
                tdate = date.fromisoformat(tdate_raw[:10])
            except ValueError:
                tdate = None
        sign = 1.0 if disp == "A" else (-1.0 if disp == "D" else 1.0)
        value = sign * shares * price if shares is not None and price is not None else None
        rows.append({
            "ticker_id": ticker_id,
            "accession_number": accession_number,
            "transaction_idx": idx,
            "insider_cik": insider_cik,
            "insider_name": insider_name,
            "is_officer": is_officer,
            "is_director": is_director,
            "is_ten_pct_owner": is_ten,
            "filing_date": filing_date,
            "acceptance_datetime": acceptance_datetime,
            "transaction_date": tdate,
            "transaction_code": code,
            "shares": shares,
            "price_per_share": price,
            "value": value,
        })
    return rows


# =============================================================
# DB shell
# =============================================================

_UPSERT_SQL = """
insert into insider_transactions (
    ticker_id, accession_number, transaction_idx, insider_cik, insider_name,
    is_officer, is_director, is_ten_pct_owner, filing_date, acceptance_datetime,
    transaction_date, transaction_code, shares, price_per_share, value, ingested_at
) values ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15, now())
on conflict (accession_number, transaction_idx) do update set
    ticker_id           = excluded.ticker_id,
    insider_cik         = excluded.insider_cik,
    insider_name        = excluded.insider_name,
    is_officer          = excluded.is_officer,
    is_director         = excluded.is_director,
    is_ten_pct_owner    = excluded.is_ten_pct_owner,
    filing_date         = excluded.filing_date,
    acceptance_datetime = excluded.acceptance_datetime,
    transaction_date    = excluded.transaction_date,
    transaction_code    = excluded.transaction_code,
    shares              = excluded.shares,
    price_per_share     = excluded.price_per_share,
    value               = excluded.value,
    ingested_at         = now()
"""


async def _upsert_rows(conn: asyncpg.Connection, rows: list[dict]) -> int:
    if not rows:
        return 0
    payload = [
        (r["ticker_id"], r["accession_number"], r["transaction_idx"], r["insider_cik"],
         r["insider_name"], r["is_officer"], r["is_director"], r["is_ten_pct_owner"],
         r["filing_date"], r["acceptance_datetime"], r["transaction_date"],
         r["transaction_code"], r["shares"], r["price_per_share"], r["value"])
        for r in rows
    ]
    async with conn.transaction():
        await conn.executemany(_UPSERT_SQL, payload)
    return len(payload)


async def _cik_to_ticker(conn: asyncpg.Connection) -> dict[int, int]:
    """Map issuer CIK (int, padding-agnostic) -> ticker_id for names with a CIK."""
    recs = await conn.fetch(
        "select ticker_id, cik from tickers where cik is not null"
    )
    out: dict[int, int] = {}
    for r in recs:
        cik = _to_int(r["cik"])
        if cik is not None:
            out[cik] = r["ticker_id"]
    return out


def _quarter_list(start_year: int, start_q: int, end_year: int, end_q: int
                  ) -> list[tuple[int, int]]:
    out = []
    y, q = start_year, start_q
    while (y, q) <= (end_year, end_q):
        out.append((y, q))
        q += 1
        if q > 4:
            q, y = 1, y + 1
    return out


async def _fetch_quarter(client: httpx.AsyncClient, year: int, q: int) -> bytes | None:
    url = DERA_URL.format(year=year, q=q)
    resp = await client.get(url)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.content


async def ingest_insider_backfill(
    pool: asyncpg.Pool,
    quarters: list[tuple[int, int]],
    concurrency: int = 4,
) -> dict:
    """Download + parse + upsert the given DERA quarters. Returns per-run totals."""
    settings = get_settings()
    headers = {"User-Agent": settings.sec_edgar_user_agent,
               "Accept-Encoding": "gzip, deflate"}
    async with pool.acquire() as conn:
        cik_map = await _cik_to_ticker(conn)
        log.info("insider backfill: %d issuer CIKs in universe, %d quarters",
                 len(cik_map), len(quarters))
        total_rows = 0
        sem = asyncio.Semaphore(concurrency)
        async with httpx.AsyncClient(timeout=120.0, headers=headers,
                                     follow_redirects=True) as client:
            async def one(year: int, q: int) -> tuple[str, int]:
                async with sem:
                    try:
                        blob = await _fetch_quarter(client, year, q)
                    except Exception as exc:  # noqa: BLE001
                        return (f"{year}q{q}", -1)
                if blob is None:
                    return (f"{year}q{q}", 0)  # 404: quarter not published
                return (f"{year}q{q}", parse_insider_dataset(blob, cik_map))

            results = await asyncio.gather(*(one(y, q) for y, q in quarters))
        # Upsert sequentially (single connection) to keep transactions small.
        for label, payload in results:
            if isinstance(payload, int):  # download failed (-1) or 404 (0)
                if payload < 0:
                    log.warning("quarter %s failed to download", label)
                continue
            n = await _upsert_rows(conn, payload)
            total_rows += n
            log.info("quarter %s: upserted %d insider rows", label, n)
    return {"quarters": len(quarters), "rows": total_rows}


async def _latest_filing_date(conn: asyncpg.Connection) -> date | None:
    return await conn.fetchval("select max(filing_date) from insider_transactions")


async def ingest_insider_transactions(pool: asyncpg.Pool, lookback_days: int = 10) -> dict:
    """Daily incremental: pull recent Form 4s per CIK and upsert new transactions.

    Reads each universe CIK's submissions index, keeps `form == "4"` filings whose
    filing_date is within `lookback_days` of today's newest stored row, fetches the
    primary XML, and upserts. Idempotent via the (accession, idx) PK.
    """
    settings = get_settings()
    headers = {"User-Agent": settings.sec_edgar_user_agent}
    async with pool.acquire() as conn:
        cik_map = await _cik_to_ticker(conn)
        cutoff = await _latest_filing_date(conn)
    total = 0
    sem = asyncio.Semaphore(5)
    async with httpx.AsyncClient(timeout=60.0, headers=headers,
                                 follow_redirects=True) as client:
        async def one(cik: int, tid: int) -> int:
            cik10 = f"{cik:010d}"
            async with sem:
                try:
                    resp = await client.get(SUBMISSIONS_URL.format(cik10=cik10))
                    if resp.status_code != 200:
                        return 0
                    js = resp.json()
                except Exception:  # noqa: BLE001
                    return 0
                recent = js.get("filings", {}).get("recent", {})
                forms = recent.get("form", [])
                accs = recent.get("accessionNumber", [])
                fdates = recent.get("filingDate", [])
                docs = recent.get("primaryDocument", [])
                accepts = recent.get("acceptanceDateTime", [])
                n_local = 0
                for i, form in enumerate(forms):
                    if form != "4":
                        continue
                    try:
                        fdate = date.fromisoformat(fdates[i][:10])
                    except (ValueError, IndexError):
                        continue
                    if cutoff is not None and fdate <= cutoff:
                        continue
                    acc = accs[i]
                    acc_nodash = acc.replace("-", "")
                    doc = docs[i] if i < len(docs) else None
                    if not doc:
                        continue
                    try:
                        xresp = await client.get(FILING_XML_URL.format(
                            cik=cik, acc_nodash=acc_nodash, doc=doc))
                        if xresp.status_code != 200:
                            continue
                    except Exception:  # noqa: BLE001
                        continue
                    accept_dt = None
                    if i < len(accepts) and accepts[i]:
                        try:
                            accept_dt = datetime.fromisoformat(accepts[i])
                        except ValueError:
                            accept_dt = None
                    rows = parse_form4_xml(xresp.content, tid, acc, fdate, accept_dt)
                    n_local += rows and len(rows) or 0
                    if rows:
                        async with pool.acquire() as c2:
                            await _upsert_rows(c2, rows)
                return n_local
        counts = await asyncio.gather(*(one(cik, tid) for cik, tid in cik_map.items()))
        total = sum(counts)
    return {"rows": total}


async def _main() -> None:
    logging.basicConfig(level=logging.INFO)
    async with pool_context() as pool:
        res = await ingest_insider_transactions(pool)
        print(res)


if __name__ == "__main__":
    asyncio.run(_main())
