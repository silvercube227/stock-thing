"""Independent as-of FRED checks and diagnostic coverage; never fit a model."""

import argparse
import asyncio
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd

from backend.config import get_settings
from backend.ingestion.macro import SERIES, _get
from backend.ml.dataset import build_calendar_grid
from backend.ml.factors.assembly import _in_index_on, build_universe_return_map
from backend.ml.factors.long_horizon import MACRO_FEATURES, add_macro_features
from backend.ml.research import SELECTION_CUTOFFS, load_snapshot, write_json_new
from scripts.rebuild_macro_vintages import replay
from scripts.signal_research import select_frames


def values_as_of(rows, series, vintage):
    available = vintage + timedelta(days=1)
    return {str(r['obs_date']):r['value'] for r in sorted(rows, key=lambda r:r['vintage_date'])
            if r['series_id'] == series and r['available_from'] <= available}


async def spotcheck(directory, output, vintages=None):
    if Path(output).exists():
        raise ValueError('Audit already exists')
    rows, digest = await replay(directory)
    checks = []
    async with httpx.AsyncClient(timeout=60) as client:
        for series in SERIES:
            first = min(r['vintage_date'] for r in rows if r['series_id'] == series)
            for vintage in vintages or [first, date(2020, 7, 20), date(2020, 7, 21), date(2024, 12, 3)]:
                payload = await _get(client, 'series/observations', get_settings().fred_api_key,
                    series_id=series, observation_start='2000-01-01', observation_end=str(vintage),
                    realtime_start=str(vintage), realtime_end=str(vintage), output_type=1,
                    limit=100000, offset=0, archive_dir=Path(output).with_suffix('.raw'))
                if int(payload['count']) != len(payload['observations']):
                    raise ValueError('Incomplete independent snapshot')
                expected = {r['date']:None if r['value'] == '.' else float(r['value'])
                            for r in payload['observations']}
                if len(expected) != len(payload['observations']):
                    raise ValueError('Duplicate independent snapshot observation')
                actual = values_as_of(rows, series, vintage)
                if expected != actual:
                    raise ValueError(f'Historical snapshot mismatch: {series} {vintage}')
                checks.append(dict(series_id=series, vintage_date=vintage, observations=len(actual),
                                   exact_match=True))
                print(series, vintage, len(actual), 'exact as-of match', flush=True)
    write_json_new(output, dict(candidate_sha256=digest, checks=checks))


async def coverage(directory, snapshot, output):
    if Path(output).exists():
        raise ValueError('Audit already exists')
    rows, digest = await replay(directory)
    inputs, manifest = load_snapshot(snapshot)
    frames = select_frames(inputs['frames'])
    grid = [d for d in build_calendar_grid(frames) if d <= SELECTION_CUTOFFS['6M']]
    panel = pd.DataFrame([dict(date=d, ticker_id=f.ticker_id) for d in grid for f in frames
                          if _in_index_on(f.membership, d)])
    market = build_universe_return_map(frames, membership_filter=True)
    result = add_macro_features(panel, frames, rows, market)
    monthly = []
    for d, group in result.groupby('date'):
        monthly.append(dict(date=d, member_rows=len(group),
            observed={c:int(group[c].notna().sum()) for c in MACRO_FEATURES},
            both_observed=int(group[MACRO_FEATURES].notna().all(axis=1).sum())))
    first = {c:min((r['date'] for r in monthly if r['observed'][c]), default=None)
             for c in MACRO_FEATURES}
    report = dict(candidate_sha256=digest, reference_input_sha256=manifest['input_sha256'],
        status='diagnostic_only', first_feature_date=first,
        months_with_both=sum(r['both_observed'] > 0 for r in monthly),
        limitations='Uncertified reference; member feature availability only, not paired model/label coverage. '
                    'No models fit, no performance evaluated, no gate certified.', monthly=monthly)
    write_json_new(output, report)
    print(dict(first_feature_date=first, months_with_both=report['months_with_both']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['spotcheck', 'coverage'])
    p.add_argument('--directory', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--snapshot')
    p.add_argument('--vintages', type=date.fromisoformat, nargs='+')
    a = p.parse_args()
    if a.mode == 'spotcheck':
        asyncio.run(spotcheck(a.directory, a.output, a.vintages))
    else:
        if not a.snapshot:
            p.error('coverage requires --snapshot')
        asyncio.run(coverage(a.directory, a.snapshot, a.output))
