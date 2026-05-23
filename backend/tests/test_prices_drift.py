"""Drift-detection logic, exercised without a DB."""

from __future__ import annotations

from backend.ingestion.prices import DRIFT_REL_THRESHOLD, detect_drift


def test_no_drift_on_identical_values() -> None:
    assert detect_drift(150.00, 150.00) is False


def test_no_drift_within_floating_tolerance() -> None:
    # 0.01% drift — well under the threshold
    assert detect_drift(150.00, 150.015) is False


def test_drift_on_2_for_1_split() -> None:
    # Yesterday's adj_close was 200; after a 2:1 split today's pull adjusts the
    # historical bar to 100. That's a 50% relative shift — must trigger.
    assert detect_drift(200.00, 100.00) is True


def test_drift_on_small_dividend() -> None:
    # A typical dividend payout might be ~0.5% of the share price — that's
    # large enough to push the adj_close past our threshold.
    assert detect_drift(100.00, 99.40) is True


def test_threshold_boundary() -> None:
    # Right at the threshold should NOT trigger (strictly greater).
    boundary = 100.0 * (1 + DRIFT_REL_THRESHOLD)
    assert detect_drift(100.00, boundary) is False
    just_over = 100.0 * (1 + DRIFT_REL_THRESHOLD * 1.01)
    assert detect_drift(100.00, just_over) is True


def test_handles_missing_stored_value() -> None:
    assert detect_drift(None, 100.00) is False


def test_handles_missing_new_value() -> None:
    assert detect_drift(100.00, None) is False


def test_handles_stored_zero() -> None:
    # Avoid divide-by-zero; treat as "can't determine".
    assert detect_drift(0.0, 50.00) is False
