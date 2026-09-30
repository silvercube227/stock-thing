"""Actual ALFRED value versions for DGS10 and the working credit series BAA10Y.

Every value comes from an explicit real-time interval, with availability delayed
through its reported vintage date. Older observations first seen in an archived
vintage are available only from that vintage, never backdated to the observation.
"""

from __future__ import annotations

from bisect import bisect_right
from datetime import date, timedelta
import gzip
import hashlib
import json
import math
from pathlib import Path

import httpx

FRED = "https://api.stlouisfed.org"
SERIES = ("DGS10", "BAA10Y")
VERIFIED_SOURCE = 'alfred_realtime_v2'
# ALFRED rejects any real-time range spanning more than this many vintage dates.
MAX_VINTAGES_PER_REQUEST = 2000


def parse_alfred(payload, series_id):
    """Rows from an explicit real-time query, where each row carries its vintage."""
    rows = []
    for r in payload.get("observations", []):
        observation = date.fromisoformat(r["date"])
        vintage = date.fromisoformat(r["realtime_start"])
        # ALFRED is date-granular. Next calendar day avoids assuming release was
        # available by the trading close on the vintage date.
        available = vintage + timedelta(days=1)
        if vintage < observation:
            raise ValueError("vintage precedes observation")
        value = None if r['value'] == '.' else float(r['value'])
        if value is not None and not math.isfinite(value):
            raise ValueError('nonfinite ALFRED value')
        rows.append(
            dict(
                series_id=series_id,
                obs_date=observation,
                vintage_date=vintage,
                available_from=available,
                value=value,
                source=VERIFIED_SOURCE,
                verified=True,
            )
        )
    return rows


def availability(obs_date, vintages):
    """First published vintage strictly after `obs_date`, or None if pre-archive."""
    index = bisect_right(vintages, obs_date)
    return vintages[index] if index < len(vintages) else None


async def _get(client, path, api_key, *, archive_dir=None, **params):
    try:
        response = await client.get(
            f"{FRED}/fred/{path}", params={"api_key": api_key, "file_type": "json", **params}
        )
    except httpx.RequestError:
        raise ValueError(f'FRED {path} transport failed') from None
    if response.is_error:
        # HTTPStatusError includes the request URL, which contains the API key.
        raise ValueError(f'FRED {path} returned HTTP {response.status_code}')
    payload = response.json()
    if archive_dir is not None:
        raw = json.dumps(dict(endpoint=path, parameters=params, response=payload),
                         sort_keys=True, separators=(',', ':')).encode()
        root = Path(archive_dir)
        root.mkdir(parents=True, exist_ok=True)
        dest = root / (hashlib.sha256(raw).hexdigest()+'.json.gz')
        if not dest.exists():
            with dest.open('xb') as out, gzip.GzipFile(fileobj=out, mode='wb', mtime=0) as stream:
                stream.write(raw)
    return payload


async def fetch_vintage_dates(client, series_id, api_key, *, archive_dir=None, as_of=None):
    dates, offset, count = [], 0, None
    as_of = as_of or date.today()
    while True:
        payload = await _get(
            client, "series/vintagedates", api_key,
            series_id=series_id, limit=10000, offset=offset,
            realtime_start='1776-07-04', realtime_end=str(as_of), archive_dir=archive_dir,
        )
        if count is not None and int(payload['count']) != count:
            raise ValueError('Vintage-calendar pagination count changed')
        count = int(payload['count'])
        batch = [date.fromisoformat(d) for d in payload.get("vintage_dates", [])]
        dates.extend(batch)
        offset += len(batch)
        if not batch and offset < count:
            raise ValueError('Incomplete vintage-calendar pagination')
        if offset > count or len(set(dates)) != len(dates):
            raise ValueError('Invalid vintage-calendar pagination')
        if any(d > as_of for d in batch):
            raise ValueError('Vintage calendar exceeds requested cutoff')
        if offset == count:
            return sorted(dates)


async def fetch_vintages(client, series_id, api_key, *, archive_dir=None, as_of=None,
                         observation_start='2000-01-01'):
    """Fetch real value histories in bounded annual real-time windows.

Annual boundaries can repeat unchanged values with a clipped realtime_start.
Suppress those repetitions, retaining revisions and explicit missing-value
revisions. Pagination must complete before any rows are returned to the caller.
"""
    if series_id not in SERIES:
        raise ValueError("unregistered macro series")
    if not api_key:
        raise ValueError("FRED_API_KEY required for ALFRED vintage retrieval")
    as_of = as_of or date.today()
    vintages = await fetch_vintage_dates(client, series_id, api_key,
                                        archive_dir=archive_dir, as_of=as_of)
    if not vintages:
        raise ValueError(f"{series_id}: no vintage calendar; availability is unknown")
    rows, previous = [], {}
    start = vintages[0]
    while start <= as_of:
        end = min(date(start.year, 12, 31), as_of)
        offset, count, batch = 0, None, []
        while count is None or offset < count:
            payload = await _get(client, 'series/observations', api_key,
                series_id=series_id, observation_start=observation_start,
                observation_end=str(end), realtime_start=str(start), realtime_end=str(end),
                output_type=1, limit=100000, offset=offset, archive_dir=archive_dir)
            if count is not None and int(payload['count']) != count:
                raise ValueError('ALFRED pagination count changed')
            count = int(payload['count'])
            page = payload['observations']
            if not page and offset < count:
                raise ValueError('Incomplete ALFRED observation pagination')
            batch.extend(parse_alfred(payload, series_id))
            offset += len(page)
            if offset > count:
                raise ValueError('ALFRED pagination exceeded reported count')
        seen = {}
        for row in sorted(batch, key=lambda r:(r['vintage_date'], r['obs_date'])):
            if not start <= row['vintage_date'] <= end:
                raise ValueError('ALFRED returned a vintage outside the requested interval')
            obs, value = row['obs_date'], row['value']
            if obs < date.fromisoformat(observation_start) or obs > end:
                raise ValueError('Observation outside requested interval')
            key = (obs, row['vintage_date'])
            if key in seen and seen[key] != value:
                raise ValueError('Conflicting values at the same ALFRED vintage')
            seen[key] = value
            prior = previous.get(obs)
            if prior and prior['vintage_date'] == row['vintage_date'] and prior['value'] != value:
                raise ValueError('Conflicting values at the same ALFRED vintage')
            if prior is None or prior['value'] != value:
                rows.append(row)
                previous[obs] = row
        start = end + timedelta(days=1)
    return rows


async def audit_revisions(client, series_id, api_key, start, end):
    """Do stored values change across vintages? Bounded window, under the cap."""
    vintages = await fetch_vintage_dates(client, series_id, api_key)
    window = [v for v in vintages if start <= v <= end]
    if len(window) > MAX_VINTAGES_PER_REQUEST:
        window = window[:MAX_VINTAGES_PER_REQUEST]
    if not window:
        return {"series_id": series_id, "observations": 0, "revised": 0, "vintages": 0}
    payload = await _get(
        client, "series/observations", api_key, series_id=series_id,
        observation_start=str(start), observation_end=str(end),
        realtime_start=str(window[0]), realtime_end=str(window[-1]),
        output_type=1, limit=100000,
    )
    seen = {}
    for r in payload.get("observations", []):
        seen.setdefault(r["date"], set()).add(r["value"])
    revised = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
    return {
        "series_id": series_id, "observations": len(seen), "revised": len(revised),
        "vintages": len(window), "realtime_start": str(window[0]),
        "realtime_end": str(window[-1]), "examples": dict(list(revised.items())[:5]),
    }
