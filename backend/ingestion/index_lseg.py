"""Strict interval reconstruction using security IDs, never issuer/symbol merges."""

from __future__ import annotations

from datetime import date, datetime


def day(value):
    if isinstance(value, datetime):
        return value.date()
    return value if isinstance(value, date) else date.fromisoformat(value)


def build_jl_intervals(current, events, window_start, observed_on, *, reviewed_zero_duration=(),
                       reviewed_continuity=()):
    """Reconstruct backwards from the observed set; inconsistent events fail."""
    start, observed = day(window_start), day(observed_on)
    active = {key: None for key in current}
    if len(active) != len(current):
        raise ValueError("duplicate security in current constituents")
    intervals = []
    seen = set()
    grouped = {}
    for e in events:
        d, key, change = day(e["date"]), e["security_id"], e["change"].lower()
        if change not in ("joiner", "leaver"):
            raise ValueError("unknown membership event")
        if start <= d <= observed:
            seen.add((d, key, change))
    for d, key, change in seen:
        grouped.setdefault((d, key), set()).add(change)
    # Source-documented intraday replacements can have no session membership.
    # Never generalize this to cancelling arbitrary contradictory event pairs.
    for review in reviewed_zero_duration:
        key = (day(review['date']), review['security_id'])
        if not review.get('evidence') or not review.get('reason'):
            raise ValueError('zero-duration review requires source evidence and a reason')
        if grouped.get(key) != {'joiner', 'leaver'}:
            raise ValueError('review does not match an exact same-day event pair')
        del grouped[key]
    continuity = set()
    for review in reviewed_continuity:
        key = (day(review['date']), review['security_id'])
        if not review.get('evidence') or not review.get('reason'):
            raise ValueError('continuity review requires source evidence and a reason')
        if grouped.get(key) != {'joiner', 'leaver'}:
            raise ValueError('continuity review does not match an exact same-day event pair')
        continuity.add(key)
    for (d, key), changes in sorted(grouped.items(), key=lambda item: item[0][0], reverse=True):
        if (d, key) in continuity:
            # A reviewed no-change pair preserves an already-active security.
            # It cannot introduce a member absent from the reconstructed set.
            if key not in active:
                raise ValueError('reviewed continuity security is not active at event date')
            continue
        if len(changes) > 1:
            raise ValueError("same-day join/leave requires verified identity resolution")
        change = next(iter(changes))
        if change == "joiner":
            if key not in active:
                raise ValueError(f"joiner {key} absent from reconstructed membership at {d}")
            intervals.append(
                dict(security_id=key, valid_from=d, valid_to=active.pop(key), source="lseg_jl")
            )
        else:
            if key in active:
                raise ValueError(f"leaver {key} already present before reversing {d}")
            active[key] = d
    for key, end in active.items():
        if end is None or end > start:
            intervals.append(
                dict(security_id=key, valid_from=start, valid_to=end, source="lseg_jl_pre_window")
            )
    return sorted(intervals, key=lambda r: (str(r["security_id"]), r["valid_from"]))


def index_on(intervals, when):
    hits = {
        r.get("index_id", "SPX")
        for r in intervals or []
        if day(r["valid_from"]) <= day(when)
        and (r.get("valid_to") is None or day(when) < day(r["valid_to"]))
    }
    if len(hits) > 1:
        raise ValueError("security simultaneously belongs to multiple S&P size indices")
    return next(iter(hits), None)


def fetch_jl_events(ld, index_ric, start, end, security_map):
    df = ld.get_data(
        index_ric,
        [
            "TR.IndexJLConstituentRIC",
            "TR.IndexJLConstituentRIC.date",
            "TR.IndexJLConstituentRIC.change",
        ],
        {"SDate": start, "EDate": end, "IC": "B"},
    )
    rows = []
    for r in df.to_dict("records"):
        ric = r["Constituent RIC"]
        if ric not in security_map:
            raise ValueError(f"unresolved dated security identity: {ric}")
        rows.append(
            dict(
                date=day(str(r["Date"])[:10]),
                security_id=security_map[ric],
                ric=ric,
                change=r["Change"],
            )
        )
    return rows


def fetch_current_constituents(ld, chain, security_map):
    df = ld.get_data(chain, ["TR.RIC"])
    rics = df["RIC"].dropna().tolist()
    missing = set(rics) - set(security_map)
    if missing:
        raise ValueError(f"unresolved current security identities: {sorted(missing)}")
    return [security_map[ric] for ric in rics]
