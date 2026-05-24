"""FinBERT sentiment pipeline.

For each ticker:
  1. Pull recent news from yfinance (Ticker.news).
  2. Score all headlines in one FinBERT batch across the whole universe.
  3. Upsert new headlines into `headlines` (dedupe by URL).
  4. Recompute `sentiment_daily` for the last 14 days.

Time-bucketing rule (NYSE close = 16:00 ET):
  - published_at <= 16:00 ET → score_date = ET calendar date
  - published_at >  16:00 ET → score_date = ET calendar date + 1 day
score_date is stored as a column in `headlines` for efficient aggregation.
The dashboard reads `sentiment_daily`, not `headlines` directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

import asyncpg
import yfinance as yf
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
NYSE_CLOSE_HOUR = 16  # 16:00 ET

FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_BATCH_SIZE = 32


# =============================================================
# Pure helpers (no I/O — unit-tested)
# =============================================================


def bucket_score_date(published_at: datetime) -> date:
    """Map a tz-aware publication timestamp to an NYSE score date.

    Headlines published before 16:00 ET belong to that ET calendar date.
    Headlines published at or after 16:00 ET belong to the *next* calendar
    date (they arrive after the market close, so they inform the next session).
    Non-trading days are valid score_dates; the feature builder handles alignment.
    """
    et = published_at.astimezone(ET)
    if et.hour >= NYSE_CLOSE_HOUR:
        return et.date() + timedelta(days=1)
    return et.date()


# =============================================================
# Result dataclasses
# =============================================================


@dataclass
class TickerResult:
    ticker_id: int
    symbol: str
    headlines_fetched: int = 0
    headlines_inserted: int = 0
    sentiment_days_updated: int = 0
    error: str | None = None


@dataclass
class SentimentResult:
    started_at: datetime
    finished_at: datetime
    headlines_inserted: int = 0
    sentiment_days_upserted: int = 0
    failed_tickers: list[str] = field(default_factory=list)
    per_ticker: list[TickerResult] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.failed_tickers and self.headlines_inserted == 0:
            return "failed"
        if self.failed_tickers:
            return "partial"
        return "success"


# =============================================================
# FinBERT (lazy-loaded; runs on MPS → CPU fallback)
# =============================================================

_finbert_pipeline = None


def _load_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is not None:
        return _finbert_pipeline

    import torch
    from transformers import pipeline

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    log.info("Loading FinBERT (%s) on device=%s", FINBERT_MODEL, device)
    _finbert_pipeline = pipeline(
        "text-classification",
        model=FINBERT_MODEL,
        tokenizer=FINBERT_MODEL,
        device=device,
        top_k=None,      # return all 3 class probabilities
        truncation=True,
        max_length=512,
    )
    log.info("FinBERT loaded")
    return _finbert_pipeline


def score_texts(texts: list[str], batch_size: int = FINBERT_BATCH_SIZE) -> list[tuple[float, str]]:
    """Score a list of texts with FinBERT.

    Returns (signed_score, label) per input, where:
      signed_score = prob_positive - prob_negative  ∈ [-1, 1]
      label        = 'pos' | 'neg' | 'neu'
    """
    if not texts:
        return []
    pipe = _load_finbert()
    results = pipe(texts, batch_size=batch_size)
    out: list[tuple[float, str]] = []
    for result in results:
        by_label = {r["label"].lower(): r["score"] for r in result}
        signed = float(by_label.get("positive", 0.0) - by_label.get("negative", 0.0))
        best = max(by_label, key=lambda k: by_label[k])
        abbr = {"positive": "pos", "negative": "neg", "neutral": "neu"}.get(best, "neu")
        out.append((signed, abbr))
    return out


# =============================================================
# yfinance news fetch
# =============================================================


def _parse_published_at(item: dict, content: dict) -> datetime | None:
    """Extract publication timestamp from a yfinance news item.

    yfinance has shipped two formats:
      - Old: top-level `providerPublishTime` (UNIX int)
      - New: `content.pubDate` as ISO 8601 string (e.g. "2026-05-23T21:05:00Z")
    """
    pub_date = content.get("pubDate") or content.get("displayTime")
    if pub_date:
        try:
            return datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
        except ValueError:
            pass
    ts = item.get("providerPublishTime")
    if ts is not None:
        try:
            return datetime.fromtimestamp(int(ts), tz=timezone.utc)
        except (ValueError, OSError):
            pass
    return None


def fetch_news_yf(symbol: str) -> list[dict]:
    """Pull recent news from yfinance for one symbol.

    Returns normalized dicts with keys: url, title, summary, published_at, source.
    yfinance typically returns 10-50 items from the last ~30 days.
    """
    raw = yf.Ticker(symbol).news or []
    out: list[dict] = []
    for item in raw:
        try:
            content = item.get("content") or {}
            published_at = _parse_published_at(item, content)
            if published_at is None:
                continue

            url = (
                (content.get("canonicalUrl") or {}).get("url")
                or item.get("link", "")
            )
            title = content.get("title") or item.get("title", "")
            summary = (
                content.get("summary")
                or content.get("description")
                or item.get("summary", "")
            )
            source = (
                (content.get("provider") or {}).get("displayName")
                or item.get("publisher", "")
            )
            if not url or not title:
                continue
            out.append(
                {
                    "url": url,
                    "title": title,
                    "summary": summary or "",
                    "published_at": published_at,
                    "source": source,
                }
            )
        except Exception as exc:
            log.debug("Skipping malformed news item for %s: %s", symbol, exc)
    return out


# =============================================================
# DB I/O
# =============================================================

_UPSERT_HEADLINE_SQL = """
insert into headlines (
    ticker_id, published_at, score_date, source, url, title, summary,
    sentiment_score, sentiment_label, finbert_version, scored_at, ingested_at
) values (
    $1, $2, $3, $4, $5, $6, $7,
    $8, $9, $10, now(), now()
)
on conflict (url) do nothing
"""

_RECOMPUTE_SQL = """
with src as (
    -- Include 13 extra days of context before the lookback window so rolling
    -- averages at the window's start have full trailing history.
    select score_date, sentiment_score
      from headlines
     where ticker_id = $1
       and score_date >= $2::date - interval '13 days'
       and sentiment_score is not null
),
daily as (
    select
        score_date,
        avg(sentiment_score)::numeric(6, 4) as mean_score,
        count(*)::integer                   as headline_count
      from src
     group by score_date
),
rolling as (
    select
        score_date,
        mean_score,
        headline_count,
        avg(mean_score) over (
            order by score_date
            range between '6 days' preceding and current row
        )::numeric(6, 4)  as rolling_7d,
        avg(mean_score) over (
            order by score_date
            range between '13 days' preceding and current row
        )::numeric(6, 4)  as rolling_14d
      from daily
)
insert into sentiment_daily (
    ticker_id, score_date, mean_score, headline_count,
    rolling_7d, rolling_14d, computed_at
)
select $1, score_date, mean_score, headline_count, rolling_7d, rolling_14d, now()
  from rolling
 where score_date >= $2::date
on conflict (ticker_id, score_date) do update set
    mean_score      = excluded.mean_score,
    headline_count  = excluded.headline_count,
    rolling_7d      = excluded.rolling_7d,
    rolling_14d     = excluded.rolling_14d,
    computed_at     = now()
"""


async def _upsert_headlines(conn: asyncpg.Connection, ticker_id: int, rows: list[dict]) -> int:
    if not rows:
        return 0
    urls = [r["url"] for r in rows]
    existing = {
        row["url"]
        for row in await conn.fetch(
            "select url from headlines where url = any($1::text[])", urls
        )
    }
    new_rows = [r for r in rows if r["url"] not in existing]
    if not new_rows:
        return 0
    payload = [
        (
            ticker_id,
            r["published_at"],
            r["score_date"],
            r.get("source"),
            r["url"],
            r.get("title"),
            r.get("summary"),
            r.get("sentiment_score"),
            r.get("sentiment_label"),
            FINBERT_MODEL,
        )
        for r in new_rows
    ]
    async with conn.transaction():
        await conn.executemany(_UPSERT_HEADLINE_SQL, payload)
    return len(new_rows)


async def _recompute_sentiment_daily(
    conn: asyncpg.Connection, ticker_id: int, lookback_start: date
) -> int:
    """Upsert sentiment_daily rows for score_date >= lookback_start.

    Uses a 13-day look-behind window so rolling_7d / rolling_14d are correct
    even at the start of the lookback period.
    """
    result = await conn.execute(_RECOMPUTE_SQL, ticker_id, lookback_start)
    try:
        return int(result.split()[-1])
    except (IndexError, ValueError):
        return 0


async def _fetch_active_tickers(pool: asyncpg.Pool) -> list[tuple[int, str]]:
    rows = await pool.fetch(
        "select ticker_id, symbol from tickers where active = true order by ticker_id"
    )
    return [(r["ticker_id"], r["symbol"]) for r in rows]


# =============================================================
# Public entry point
# =============================================================


async def ingest_sentiment(
    pool: asyncpg.Pool,
    tickers: Iterable[tuple[int, str]] | None = None,
    fetch_concurrency: int = 10,
    upsert_concurrency: int = 5,
) -> SentimentResult:
    """Run the sentiment pipeline for all (or a given subset of) active tickers.

    Two-phase design:
      1. Fetch news for all tickers concurrently (I/O-bound).
      2. Score all headlines in one FinBERT pass (compute-bound, MPS).
      3. Upsert + recompute sentiment_daily per ticker (async I/O).
    """
    if tickers is None:
        candidates = await _fetch_active_tickers(pool)
    else:
        candidates = list(tickers)

    started = datetime.now(timezone.utc)
    run_id = int(
        await pool.fetchval(
            "insert into ingestion_runs (job_name) values ($1) returning run_id",
            "sentiment",
        )
    )

    # --- Phase 1: fetch news ---
    fetch_sem = asyncio.Semaphore(fetch_concurrency)
    ticker_news: dict[int, list[dict]] = {}
    fetch_errors: dict[int, str] = {}

    async def _fetch_one(tid: int, sym: str) -> None:
        async with fetch_sem:
            try:
                items = await asyncio.to_thread(fetch_news_yf, sym)
                ticker_news[tid] = items
                log.debug("%s: fetched %d headlines", sym, len(items))
            except Exception as exc:
                log.warning("Failed to fetch news for %s: %s", sym, exc)
                fetch_errors[tid] = str(exc)
                ticker_news[tid] = []

    await asyncio.gather(*(_fetch_one(tid, sym) for tid, sym in candidates))

    # --- Phase 2: score all headlines in one FinBERT pass ---
    all_pairs: list[tuple[int, dict]] = []  # (ticker_id, item)
    for tid, items in ticker_news.items():
        for item in items:
            all_pairs.append((tid, item))

    if all_pairs:
        log.info("Scoring %d headlines with FinBERT ...", len(all_pairs))
        texts = [
            f"{it['title']} {it['summary']}".strip() if it.get("summary") else it["title"]
            for _, it in all_pairs
        ]
        scored = await asyncio.to_thread(score_texts, texts)
        for (tid, item), (signed, label) in zip(all_pairs, scored):
            item["sentiment_score"] = signed
            item["sentiment_label"] = label
            item["score_date"] = bucket_score_date(item["published_at"])
        log.info("FinBERT scoring done")

    # --- Phase 3: upsert per ticker ---
    today_et = datetime.now(ET).date()
    lookback_start = today_et - timedelta(days=14)
    upsert_sem = asyncio.Semaphore(upsert_concurrency)

    async def _upsert_one(tid: int, sym: str) -> TickerResult:
        tr = TickerResult(
            ticker_id=tid,
            symbol=sym,
            headlines_fetched=len(ticker_news.get(tid, [])),
            error=fetch_errors.get(tid),
        )
        if tr.error:
            return tr
        async with upsert_sem:
            try:
                async with pool.acquire() as conn:
                    tr.headlines_inserted = await _upsert_headlines(
                        conn, tid, ticker_news.get(tid, [])
                    )
                    tr.sentiment_days_updated = await _recompute_sentiment_daily(
                        conn, tid, lookback_start
                    )
            except Exception as exc:
                log.exception("DB error for %s", sym)
                tr.error = str(exc)
        return tr

    per_ticker = list(
        await asyncio.gather(*(_upsert_one(tid, sym) for tid, sym in candidates))
    )

    finished = datetime.now(timezone.utc)
    result = SentimentResult(
        started_at=started,
        finished_at=finished,
        headlines_inserted=sum(tr.headlines_inserted for tr in per_ticker),
        sentiment_days_upserted=sum(tr.sentiment_days_updated for tr in per_ticker),
        failed_tickers=[tr.symbol for tr in per_ticker if tr.error],
        per_ticker=per_ticker,
    )

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
        result.headlines_inserted,
        "; ".join(f"{tr.symbol}: {tr.error}" for tr in per_ticker if tr.error) or None,
        json.dumps(
            {
                "failed_tickers": result.failed_tickers,
                "ticker_count": len(per_ticker),
                "sentiment_days_upserted": result.sentiment_days_upserted,
            }
        ),
    )
    return result
