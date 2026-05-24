"""Unit tests for the NYSE close time-bucketing rule.

bucket_score_date() is pure — no network, no DB.
"""

from __future__ import annotations

from datetime import date, datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import pytest

from backend.ingestion.headlines import bucket_score_date

ET = ZoneInfo("America/New_York")
UTC = timezone.utc


def _et(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    """Construct a timezone-aware datetime in ET, returned as UTC-aware."""
    return datetime(year, month, day, hour, minute, tzinfo=ET)


def _utc(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


# --- Core cases -------------------------------------------------------

def test_before_close_same_day() -> None:
    # 15:00 ET → score_date = same ET calendar date
    published = _et(2025, 1, 10, 15, 0)
    assert bucket_score_date(published) == date(2025, 1, 10)


def test_exactly_at_close_next_day() -> None:
    # 16:00 ET exactly → score_date = next calendar date (>= 16 triggers roll)
    published = _et(2025, 1, 10, 16, 0)
    assert bucket_score_date(published) == date(2025, 1, 11)


def test_after_close_next_day() -> None:
    # 17:30 ET → score_date = next calendar date
    published = _et(2025, 1, 10, 17, 30)
    assert bucket_score_date(published) == date(2025, 1, 11)


def test_midnight_et_is_same_day() -> None:
    # 00:00 ET (just after midnight) → still same day (hour < 16)
    published = _et(2025, 1, 10, 0, 0)
    assert bucket_score_date(published) == date(2025, 1, 10)


def test_just_before_close_same_day() -> None:
    published = _et(2025, 1, 10, 15, 59)
    assert bucket_score_date(published) == date(2025, 1, 10)


def test_just_after_close_next_day() -> None:
    published = _et(2025, 1, 10, 16, 1)
    assert bucket_score_date(published) == date(2025, 1, 11)


# --- UTC input -----------------------------------------------------------

def test_utc_timestamp_converted_correctly() -> None:
    # 21:00 UTC = 16:00 ET (standard time, UTC-5).
    # In January, ET is EST (UTC-5), so 21:00 UTC = 16:00 ET → rolls to next day.
    published = _utc(2025, 1, 10, 21, 0)
    assert bucket_score_date(published) == date(2025, 1, 11)


def test_utc_before_close_same_day() -> None:
    # 20:59 UTC = 15:59 ET (UTC-5 in Jan) → same day
    published = _utc(2025, 1, 10, 20, 59)
    assert bucket_score_date(published) == date(2025, 1, 10)


# --- DST edge cases ------------------------------------------------------

def test_dst_summer_utc_offset() -> None:
    # In summer ET is EDT (UTC-4). 20:00 UTC = 16:00 EDT → rolls to next day.
    published = _utc(2025, 7, 10, 20, 0)
    assert bucket_score_date(published) == date(2025, 7, 11)


def test_dst_summer_before_close() -> None:
    # 19:59 UTC = 15:59 EDT → same day
    published = _utc(2025, 7, 10, 19, 59)
    assert bucket_score_date(published) == date(2025, 7, 10)


# --- Weekend / end-of-month roll -----------------------------------------

def test_friday_after_close_rolls_to_saturday() -> None:
    # Friday after close → score_date = Saturday (not next Monday).
    # The feature builder is responsible for aligning non-trading days.
    friday = _et(2025, 1, 10, 17, 0)  # 2025-01-10 is a Friday
    assert bucket_score_date(friday) == date(2025, 1, 11)  # Saturday


def test_month_boundary_roll() -> None:
    # Jan 31 after close → Feb 1
    published = _et(2025, 1, 31, 17, 0)
    assert bucket_score_date(published) == date(2025, 2, 1)


def test_year_boundary_roll() -> None:
    # Dec 31 after close → Jan 1 next year
    published = _et(2024, 12, 31, 16, 30)
    assert bucket_score_date(published) == date(2025, 1, 1)
