"""Shared scalar helpers for the factor builders."""

from __future__ import annotations

import math


def _log_ratio(a: float | None, b: float | None) -> float:
    """log(a/b), or 0.0 if either price is missing/non-positive."""
    if a is None or b is None or a <= 0 or b <= 0:
        return 0.0
    return math.log(a / b)


def _safe_ratio(num: float | None, den: float | None) -> float:
    if num is None or den is None or abs(den) <= 1e-9:
        return 0.0
    return float(num / den)
