"""Version-preserving Reuters English archive; local raw text only."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from backend.ingestion.calendar import NYSE

NEWS_QUERY = "R:{ric} AND NS:RTRS AND L:EN"
NOVELTY_THRESHOLD = 0.90


def open_store(path):
    conn = sqlite3.connect(path)
    conn.executescript("""
    pragma foreign_keys=on;
    create table if not exists versions (
        version_key text primary key, story_id text not null, version_created text not null,
        first_created text not null, title text not null, text_hash text not null,
        source text not null, language text not null, fetched_at text not null);
    create table if not exists relevance (
        version_key text not null references versions(version_key), ticker_id integer not null,
        ric text not null, primary key(version_key,ticker_id));
    create table if not exists scores (
        version_key text not null references versions(version_key), ticker_id integer not null,
        scorer_revision text not null, score real not null, label text not null,
        relevant integer not null, event_type text not null, guidance integer not null,
        primary key(version_key,ticker_id,scorer_revision));
    create table if not exists coverage (
        ticker_id integer not null, start_at text not null, end_at text not null,
        status text not null, n_rows integer not null, error text,
        primary key(ticker_id,start_at,end_at));
    """)
    return conn


def utc(value):
    dt = (
        value
        if isinstance(value, datetime)
        else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    )
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def parse_headline(row, fetched_at):
    if not row.get("story_id") or not row.get("title"):
        raise ValueError("missing story identity/text")
    first, version = utc(row["first_created"]), utc(row["version_created"])
    if version < first:
        raise ValueError("version predates first publication")
    title = row["title"]
    digest = hashlib.sha256(title.encode()).hexdigest()
    key = hashlib.sha256(f"{row['story_id']}|{version.isoformat()}|{digest}".encode()).hexdigest()
    return (
        key,
        row["story_id"],
        version.isoformat(),
        first.isoformat(),
        title,
        digest,
        json.dumps(row.get("info_source")),
        json.dumps(row.get("language")),
        utc(fetched_at).isoformat(),
    )


def store_rows(conn, rows, ticker_id, ric):
    now = datetime.now(timezone.utc)
    with conn:
        for row in rows:
            parsed = parse_headline(row, now)
            # Only exact immutable versions deduplicate. Changed text gets a new key.
            conn.execute(
                "insert into versions values (?,?,?,?,?,?,?,?,?) on conflict do nothing", parsed
            )
            conn.execute(
                "insert into relevance values (?,?,?) on conflict do nothing",
                (parsed[0], ticker_id, ric),
            )


def fetch_window(ld, ric, start, end, cap=10000):
    """All-or-error recursive fetch. Overflow at one second remains incomplete."""
    response = ld.content.news.headlines.Definition(
        query=NEWS_QUERY.format(ric=ric),
        date_from=start,
        date_to=end,
        count=cap,
        sort_order="newToOld",
    ).get_data()
    if getattr(response, "errors", None):
        raise RuntimeError("LSEG headline response contains errors")
    raw = response.data.headlines
    rows = raw.to_dict("records") if hasattr(raw, "to_dict") else list(raw or [])
    fields = (
        "story_id",
        "first_created",
        "version_created",
        "title",
        "info_source",
        "language",
        "urgency",
    )
    rows = [r if isinstance(r, dict) else {k: getattr(r, k, None) for k in fields} for r in rows]
    if len(rows) >= cap:
        if end - start <= timedelta(seconds=1):
            raise RuntimeError("unresolved headline overflow")
        midpoint = start + (end - start) / 2
        # Inclusive overlap avoids dropping records on the split boundary.
        return fetch_window(ld, ric, start, midpoint, cap) + fetch_window(
            ld, ric, midpoint, end, cap
        )
    return rows


def ingest_window(ld, conn, ticker_id, ric, start, end, raw_archive=None):
    start, end = utc(start), utc(end)
    previous = conn.execute(
        "select status from coverage where ticker_id=? and start_at=? and end_at=?",
        (ticker_id, start.isoformat(), end.isoformat()),
    ).fetchone()
    if previous and previous[0] in ("complete", "empty"):
        return previous[0]
    n, status, error = 0, "error", None
    try:
        rows = fetch_window(ld, ric, start, end)
        if raw_archive is not None:
            import gzip

            archive = Path(raw_archive)
            archive.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                dict(ric=ric, start=start.isoformat(), end=end.isoformat(), rows=rows),
                sort_keys=True,
                default=str,
            ).encode()
            artifact = archive / (hashlib.sha256(payload).hexdigest() + ".json.gz")
            if not artifact.exists():
                with gzip.open(artifact, "xb") as stream:
                    stream.write(payload)
        store_rows(conn, rows, ticker_id, ric)
        n, status = len(rows), "complete" if rows else "empty"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    with conn:
        conn.execute(
            "insert or replace into coverage values (?,?,?,?,?,?)",
            (ticker_id, start.isoformat(), end.isoformat(), status, n, error),
        )
    return status


@lru_cache(maxsize=100000)
def score_session(timestamp):
    """First exchange session whose close is strictly after publication."""
    ts = utc(timestamp)
    # The archive probe returned midnight-only version stamps. Without proof of
    # intraday precision, wait through that UTC date rather than inventing an
    # opening-time publication. This also conservatively delays real midnight news.
    if ts.hour == ts.minute == ts.second == ts.microsecond == 0:
        ts += timedelta(days=1)
    schedule = NYSE.schedule(
        start_date=(ts - timedelta(days=1)).date(), end_date=(ts + timedelta(days=10)).date()
    )
    for session, row in schedule.iterrows():
        if ts < row.market_close.to_pydatetime():
            return session.date()
    raise ValueError("no exchange close available")


def normalized_title(title):
    title = re.sub(r"^(UPDATE\s*\d+|BRIEF|RPT)\s*[-–:]\s*", "", title, flags=re.I)
    return " ".join(re.findall(r"[\w.%+-]+", title.lower()))


def is_duplicate(title, previous):
    clean = normalized_title(title)
    for other in previous:
        # Changed numbers and direction are substantive even in near-identical text.
        a, b = set(re.findall(r"\d+(?:\.\d+)?", clean)), set(re.findall(r"\d+(?:\.\d+)?", other))
        directions = {
            "raise",
            "raises",
            "raised",
            "cut",
            "cuts",
            "lower",
            "lowers",
            "increase",
            "decrease",
        }
        if a != b or set(clean.split()) & directions != set(other.split()) & directions:
            continue
        if SequenceMatcher(None, clean, other).ratio() >= NOVELTY_THRESHOLD:
            return True
    return False


def validate_annotations(rows):
    """Locked validation scores only; annotation labels must come from reviewers."""
    from sklearn.metrics import f1_score, precision_score

    if len(rows) != 200 or any(r.get("split") != "validation" for r in rows):
        raise ValueError("exactly 200 locked validation annotations required")
    for r in rows:
        if (
            r.get("gold_relevant") not in (True, False)
            or r.get("gold_sentiment") not in ("positive", "negative", "neutral")
            or r.get("gold_event") not in ("other", "guidance", "operating")
        ):
            raise ValueError("independent gold relevance/event/sentiment annotations required")
    relevance = precision_score(
        [r["gold_relevant"] for r in rows], [r["pred_relevant"] for r in rows], zero_division=0
    )
    relevant = [r for r in rows if r["gold_relevant"]]
    sentiment = f1_score(
        [r["gold_sentiment"] for r in relevant],
        [r["pred_sentiment"] for r in relevant],
        average="macro",
        labels=["positive", "negative", "neutral"],
        zero_division=0,
    )
    precision = {}
    for category in ("guidance", "operating"):
        precision[category] = precision_score(
            [r["gold_event"] == category for r in relevant],
            [r["pred_event"] == category for r in relevant],
            zero_division=0,
        )
    return {
        "relevance_precision": float(relevance),
        "sentiment_macro_f1": float(sentiment),
        "event_precision": precision,
        "passed": bool(relevance >= 0.9 and sentiment >= 0.7 and min(precision.values()) >= 0.8),
    }


def daily_aggregate(conn, scorer_revision):
    rows = conn.execute(
        """select r.ticker_id,v.version_created,v.title,s.score,s.label,
        s.event_type,s.guidance from versions v join relevance r using(version_key)
        join scores s on s.version_key=v.version_key and s.ticker_id=r.ticker_id
        where s.scorer_revision=? and s.relevant=1 order by v.version_created,v.version_key
        """,
        (scorer_revision,),
    ).fetchall()
    out, history = {}, {}
    for tid, stamp, title, score, label, event_type, guidance in rows:
        session = score_session(stamp)
        prior = history.setdefault(tid, [])
        prior[:] = [(d, t) for d, t in prior if (session - d).days <= 7]
        if is_duplicate(title, [t for _, t in prior]):
            continue
        prior.append((session, normalized_title(title)))
        row = out.setdefault(
            (tid, session),
            dict(
                ticker_id=tid,
                score_date=session,
                scorer_revision=scorer_revision,
                n_events=0,
                sentiment_sum=0.0,
                n_negative=0,
                guidance_up=0,
                guidance_down=0,
                operating_n=0,
                operating_sentiment_sum=0.0,
            ),
        )
        row["n_events"] += 1
        row["sentiment_sum"] += score
        row["n_negative"] += label == "negative"
        row["guidance_up"] += event_type == "guidance" and guidance == 1
        row["guidance_down"] += event_type == "guidance" and guidance == -1
        if event_type == "operating":
            row["operating_n"] += 1
            row["operating_sentiment_sum"] += score
    return list(out.values())


@lru_cache(maxsize=4096)
def _alias_pattern(alias):
    return re.compile(r"\b" + re.escape(alias) + r"\b", flags=re.I)


def entity_event(title, ticker_id, aliases, published_at):
    """Conservative clause attribution using dated, reviewed issuer aliases.

    Multi-issuer clauses are ambiguous: do not give every tagged issuer the same
    sentiment. Such cases remain irrelevant until an annotation-validated model
    can attribute them. This rule is included in the scorer fingerprint.
    """
    when = utc(published_at).date().isoformat()
    target = []
    for clause in re.split(r";|\bwhile\b|\bwhereas\b", title, flags=re.I):
        hits = set()
        for tid, names in aliases.items():
            for item in names:
                if item["valid_from"] <= when and (
                    not item.get("valid_to") or when < item["valid_to"]
                ) and _alias_pattern(item["alias"]).search(clause):
                    hits.add(int(tid))
        if hits == {int(ticker_id)}:
            target.append(clause.strip())
    text = "; ".join(target)
    category, direction = "other", 0
    if text and re.search(r"\b(guidance|outlook|forecast)\b", text, re.I):
        up = bool(re.search(r"\b(raise[sd]?|increases?|lifts?)\b", text, re.I))
        down = bool(re.search(r"\b(cuts?|lowers?|reduces?)\b", text, re.I))
        if up != down and not re.search(r"\b(not|no|denies)\b", text, re.I):
            category, direction = "guidance", 1 if up else -1
    elif text and re.search(
        r"\b(contract|orders?|production|recall|plant|factory|capacity|launch)\b", text, re.I
    ):
        category = "operating"
    return {"text": text, "relevant": bool(text), "event_type": category, "guidance": direction}


def scorer_fingerprint(model_revision, aliases):
    if not re.fullmatch(r"[0-9a-f]{40}", model_revision):
        raise ValueError("pin FinBERT to its exact 40-character repository revision")
    from pathlib import Path

    digest = hashlib.sha256(
        Path(__file__).read_bytes() + json.dumps(aliases, sort_keys=True).encode()
    ).hexdigest()
    return f"finbert:{model_revision}:rules:{digest}"


def score_archive(conn, aliases, model_revision, validation=None, batch_size=32, *,
                  exploratory=False, local_files_only=False, device=None):
    """Validated scoring by default; exploratory scores have a distinct revision."""
    revision = scorer_fingerprint(model_revision, aliases)
    if not exploratory and (
        not validation or validation.get("scorer_revision") != revision
        or not validate_annotations(validation["rows"])["passed"]
    ):
        raise ValueError("news scoring blocked: locked annotation validation failed/mismatched")
    if exploratory:
        revision = "exploratory:" + revision
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    model_name = "ProsusAI/finbert"
    classifier = pipeline(
        "text-classification",
        model=AutoModelForSequenceClassification.from_pretrained(
            model_name, revision=model_revision, local_files_only=local_files_only,
        ),
        tokenizer=AutoTokenizer.from_pretrained(
            model_name, revision=model_revision, local_files_only=local_files_only,
        ),
        top_k=None,
        device=device,
    )
    cursor = conn.execute(
        """select v.version_key,r.ticker_id,v.title,v.version_created
        from versions v join relevance r using(version_key)
        where not exists(select 1 from scores s where s.version_key=v.version_key
            and s.ticker_id=r.ticker_id and s.scorer_revision=?)
        order by v.version_created,v.version_key,r.ticker_id""",
        (revision,),
    )
    n = 0
    while batch := cursor.fetchmany(batch_size):
        events = [entity_event(title, tid, aliases, stamp) for _, tid, title, stamp in batch]
        positions = [i for i, e in enumerate(events) if e["relevant"]]
        predictions = (
            classifier(
                [events[i]["text"] for i in positions],
                truncation=True,
                max_length=512,
                batch_size=batch_size,
            )
            if positions
            else []
        )
        scored = dict(zip(positions, predictions, strict=True))
        with conn:
            for i, (key, tid, _, _) in enumerate(batch):
                probs = {p["label"].lower(): p["score"] for p in scored.get(i, [])}
                signed = probs.get("positive", 0.0) - probs.get("negative", 0.0)
                label = max(probs, key=probs.get) if probs else "neutral"
                e = events[i]
                conn.execute(
                    "insert into scores values (?,?,?,?,?,?,?,?)",
                    (
                        key,
                        tid,
                        revision,
                        signed,
                        label,
                        int(e["relevant"]),
                        e["event_type"],
                        e["guidance"],
                    ),
                )
                n += 1
    return n
