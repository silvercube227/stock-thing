"""Explicit price/share bases and documented terminal outcomes."""

from __future__ import annotations

import bisect
import json
import math
from datetime import date, datetime


from backend.ingestion.calendar import HORIZON_TRADING_DAYS, shift_trading_days


def day(value):
    if isinstance(value, datetime):
        return value.date()
    return value if isinstance(value, date) else date.fromisoformat(value)


def as_traded_closes(prices):
    """Yahoo close is split-adjusted. Missing action metadata is unknown, not 1."""
    out = [float("nan")] * len(prices)
    suffix = 1.0
    for i in range(len(prices) - 1, -1, -1):
        p = prices[i]
        close = p.get("close")
        convention = p.get("price_basis")
        if convention is None and p.get("source") == "yfinance":
            convention = "split_adjusted"
        if close is not None and close > 0:
            if convention == "as_traded":
                out[i] = float(close)
            elif convention == "split_adjusted":
                out[i] = float(close) * suffix
        factor = p.get("split_factor")
        suffix *= float(factor) if factor is not None and factor > 0 else float("nan")
    return out


def aligned_shares(filings, prices, dates):
    """Only measured, nominal, point-in-time shares; weighted averages stay absent."""
    usable = sorted(
        (
            r
            for r in filings
            if r.get("shares_outstanding") is not None
            and r.get("shares_measured_at")
            and r.get("shares_basis") == "as_reported"
            and r.get("shares_kind") == "point_in_time"
        ),
        key=lambda r: day(r["filed_at"]),
    )
    pub = [day(r["filed_at"]) for r in usable]
    pdates = [day(r["trade_date"]) for r in prices]
    out = []
    for d in dates:
        d = day(d)
        i = bisect.bisect_right(pub, d) - 1
        value = float("nan")
        if i >= 0:
            r = usable[i]
            measured = day(r["shares_measured_at"])
            if measured <= pub[i] <= d and pdates and pdates[0] <= measured:
                value = float(r["shares_outstanding"])
                for p in prices[
                    bisect.bisect_right(pdates, measured) : bisect.bisect_right(pdates, d)
                ]:
                    # Yahoo also reports pricing-only spin-off adjustments in
                    # Stock Splits. Those do not increase shares of this security.
                    factor = p.get('share_split_factor')
                    if factor is None and p.get('split_factor') == 1:
                        factor = 1.0
                    value *= float(factor) if factor is not None and factor > 0 else float("nan")
        out.append(value if value > 0 else float("nan"))
    return out


def documented_targets(prices, pos, events, observation_end, dates=None):
    """Calendar-mature labels; unresolved exits are masked, never last-price guesses.

    Documented zero recovery is represented by log(1e-12) for finite model input;
    the outcome is also tagged explicitly for sensitivity/coverage reports.
    Terminal values must be verified on the same adjusted basis as entry prices.
    """
    dates = dates if dates is not None else [day(p["trade_date"]) for p in prices]
    entry = pos + 1
    out = {}
    if entry >= len(prices):
        return {h: (0.0, False, None, None, "no_entry") for h in HORIZON_TRADING_DAYS}
    entry_date = dates[entry]
    def reviewed_last_trade(event):
        source = event.get('source')
        if isinstance(source, str):
            try:
                source = json.loads(source)
            except (ValueError, TypeError):
                return None
        if isinstance(source, dict) and source.get('last_trade_date'):
            return day(source['last_trade_date'])
        return None

    def after_terminal(event, d):
        last = reviewed_last_trade(event) if event.get('verified') else None
        return d > last if last is not None else day(event['effective_date']) <= d

    if any(e.get('verified') and e.get('source') and
           e.get('event_type') in ('acquisition', 'bankruptcy', 'liquidation',
                                    'security_replacement', 'trading_termination') and
           after_terminal(e, entry_date) for e in events or []):
        return {h: (0.0, False, entry_date, None, 'entry_after_terminal') for h in HORIZON_TRADING_DAYS}
    base = prices[entry].get("adj_close")
    if entry_date != shift_trading_days(dates[pos], 1):
        return {h: (0.0, False, entry_date, None, "missing_entry") for h in HORIZON_TRADING_DAYS}
    for h, steps in HORIZON_TRADING_DAYS.items():
        end = shift_trading_days(entry_date, steps)
        value, status = None, "right_censored"
        if end <= day(observation_end):
            j = bisect.bisect_left(dates, end)
            status = "unresolved_gap"
            if j < len(dates) and dates[j] == end:
                value, status = prices[j].get("adj_close"), "observed"
            # An identified termination overrides spurious successor/reused bars.
            matches = [e for e in events or []
                       if not after_terminal(e, entry_date) and after_terminal(e, end)]
            if len(matches) > 1:
                raise ValueError("multiple terminal events for one security horizon")
            if matches:
                event = matches[0]
                value, status = None, event["event_type"]
                if (
                    event.get("verified")
                    and event.get("source")
                    and event.get("proceeds_basis") == "adj_close"
                ):
                    if event.get("consideration_type") == "cash":
                        value = event.get("cash_value")
                    elif event.get("consideration_type") == "successor":
                        values = event.get("horizon_values") or {}
                        if isinstance(values, str):
                            values = json.loads(values)
                        value = values.get(end.isoformat())
        valid = (
            base is not None
            and math.isfinite(float(base))
            and base > 0
            and value is not None
            and math.isfinite(float(value))
            and value >= 0
            and end <= day(observation_end)
        )
        ret = math.log(max(float(value) / float(base), 1e-12)) if valid else 0.0
        if valid and value == 0:
            status = "documented_zero_recovery"
        out[h] = (ret, bool(valid), entry_date, end, status)
    return out
