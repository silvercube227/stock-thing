"""NYSE trading calendar helpers.

Single source of truth for "what days does the market trade?" so that:
  - Daily ingestion can verify it pulled the right number of bars
  - Label horizons (21, 63, 126, 252 trading days) are computed honestly
  - Feature builders can align series to actual trading dates
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache

import pandas_market_calendars as mcal

NYSE = mcal.get_calendar("NYSE")

# Horizon labels -> number of trading days. Centralized so train.py, infer.py,
# and the dashboard all agree on what "1M" means.
HORIZON_TRADING_DAYS: dict[str, int] = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
}


@lru_cache(maxsize=64)
def trading_days_between(start: date, end: date) -> tuple[date, ...]:
    """Inclusive list of NYSE trading dates between start and end.

    Cached because schedule() is non-trivial and we hit it a lot in label math.
    """
    sched = NYSE.schedule(start_date=start, end_date=end)
    return tuple(ts.date() for ts in sched.index)


def is_trading_day(d: date) -> bool:
    return len(trading_days_between(d, d)) > 0


def expected_bar_count(start: date, end: date) -> int:
    """How many daily bars *should* exist between start and end (inclusive)."""
    return len(trading_days_between(start, end))


def shift_trading_days(d: date, n: int) -> date | None:
    """Return the date that is `n` trading days after `d` (n>=0).

    Used for label generation: shift_trading_days(sample_end, 21) -> 1M label date.
    Returns None if we'd need to look beyond the calendar's currently-loaded range.
    """
    if n < 0:
        raise ValueError("shift_trading_days expects n >= 0; use a negative window instead")
    # NYSE calendar is generated lazily — request a window we know will contain n+slack days.
    # Roughly 252 trading days per year => allocate ceil(n/252)+1 calendar years of slack.
    end_guess = date(d.year + (n // 200) + 1, 12, 31)
    days = trading_days_between(d, end_guess)
    if d in days:
        idx = days.index(d) + n
    else:
        # d is not a trading day; align forward to the next one.
        # NYSE.schedule excludes weekends/holidays so "days" never contains them.
        idx = n - 1  # treat next trading day as offset 0
        if idx >= len(days):
            return None
    if idx >= len(days):
        return None
    return days[idx]
