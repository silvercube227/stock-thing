from datetime import date, timedelta

import numpy as np
import pytest

from backend.ingestion.news_lseg import open_store, score_archive, utc
from backend.ml.factors.long_horizon import news_features
from scripts.exploratory_news import archive_window, retained_rows


def headline(first="2018-06-01T12:00:00Z", version="2018-06-01T13:00:00Z"):
    return dict(story_id="story", title="Issuer raises guidance",
                first_created=first, version_created=version)


def test_bad_timestamps_are_excluded_without_retiming_valid_records():
    good = headline()
    retained, rejected = retained_rows(
        [good, headline(version="2018-06-01T11:00:00Z"),
         headline(version="2018-09-01T00:00:00Z")],
        utc("2018-06-01"), utc("2018-09-01"),
    )
    assert retained == [good]
    assert rejected == {"version predates first publication": 1, "outside_requested_window": 1}


def test_partial_archive_keeps_valid_rows_and_reuses_raw_source(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.exploratory_news.fetch_window", lambda *a: [
        headline(), headline(version="2018-06-01T11:00:00Z")])
    conn = open_store(":memory:")
    request = dict(ticker_id=1, ric="A.N", start="2018-06-01", end="2018-09-01")
    first = archive_window(None, conn, request, tmp_path)
    assert first["status"] == "partial" and first["retained"] == 1
    monkeypatch.setattr("scripts.exploratory_news.fetch_window", lambda *a: pytest.fail("refetch"))
    assert archive_window(None, conn, request, tmp_path) == first
    assert conn.execute("select count(*) from versions").fetchone()[0] == 1
    assert conn.execute("select status from coverage").fetchone()[0] == "partial"
    conn.close()


def test_partial_and_unvalidated_news_require_explicit_exploration():
    d = date(2020, 6, 30)
    row = dict(score_date=d, scorer_revision="exploratory:pinned", n_events=1,
               sentiment_sum=.5, n_negative=0, guidance_up=1, guidance_down=0,
               operating_n=1, operating_sentiment_sum=.5)
    coverage = [dict(start_date=d-timedelta(days=89), end_date=d, status="partial")]
    assert np.isnan(news_features([row], coverage, [d])[0]["news_sent_90d"])
    assert news_features([row], coverage, [d], exploratory=True)[0]["news_sent_90d"] == .5
    coverage[0]["status"] = "complete"
    assert np.isnan(news_features([row], coverage, [d])[0]["news_sent_90d"])
    coverage[0]["status"] = "error"
    assert np.isnan(news_features([row], coverage, [d], exploratory=True)[0]["news_sent_90d"])


def test_default_scoring_still_requires_independent_validation():
    with pytest.raises(ValueError, match="locked annotation"):
        score_archive(None, {}, "0" * 40)
