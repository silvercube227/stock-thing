import gzip
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime

import pytest

from backend.ingestion import news_lseg
from scripts.prepare_news_review import select_sample


def test_review_split_is_distinct_balanced_and_order_independent():
    rows = [
        dict(
            story_id=str(i),
            version_key=str(i),
            ticker_id=i % 61,
            version_created="2018-06-01",
            stratum=[str(i % 61)],
        )
        for i in range(997)
    ]
    # A later version must not put the same story in the other split.
    rows += [
        {**r, "version_key": "later-" + r["version_key"], "version_created": "2018-06-02"}
        for r in rows[:100]
    ]
    sample = select_sample(rows)
    assert sample == select_sample(list(reversed(rows)))
    assert len({r["story_id"] for r in sample}) == 400
    assert Counter(r["split"] for r in sample) == {"development": 200, "validation": 200}
    buckets = defaultdict(Counter)
    for row in sample:
        buckets[tuple(row["stratum"])][row["split"]] += 1
        assert row["gold_relevant"] is row["gold_event"] is row["gold_sentiment"] is None
        assert not row["version_key"].startswith("later-")
    assert all(abs(c["development"] - c["validation"]) <= 1 for c in buckets.values())


def test_versions_cannot_pad_annotation_sample():
    rows = [
        dict(
            story_id="same-story",
            version_key=str(i),
            ticker_id=1,
            version_created="2018-06-01",
            stratum=["one"],
        )
        for i in range(450)
    ]
    with pytest.raises(ValueError, match="distinct stories"):
        select_sample(rows)


def test_rejected_news_window_is_archived_without_partial_ingestion(tmp_path, monkeypatch):
    row = dict(
        story_id="story",
        title="Issuer raises guidance",
        first_created="2018-06-01T15:20:00Z",
        version_created="2018-06-01T13:40:00Z",
    )
    monkeypatch.setattr(news_lseg, "fetch_window", lambda *args: [row])
    conn = news_lseg.open_store(":memory:")
    status = news_lseg.ingest_window(
        None,
        conn,
        1,
        "A.N",
        datetime(2018, 6, 1, tzinfo=UTC),
        datetime(2018, 7, 1, tzinfo=UTC),
        tmp_path,
    )
    assert status == "error"
    assert conn.execute("select count(*) from versions").fetchone()[0] == 0
    (artifact,) = tmp_path.glob("*.json.gz")
    assert json.loads(gzip.decompress(artifact.read_bytes()))["rows"] == [row]
    assert (
        conn.execute("select error from coverage")
        .fetchone()[0]
        .endswith("version predates first publication")
    )
    conn.close()
