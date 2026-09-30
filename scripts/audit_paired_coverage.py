"""Audit actual pre-fit paired samples without fitting or certifying source gates."""

import argparse
from collections import Counter
from datetime import date
import gzip
import hashlib
import math
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from backend.ingestion.calendar import HORIZON_TRADING_DAYS
from backend.ml.dataset import build_calendar_grid
from backend.ml.factors.long_horizon import PACKS
from backend.ml.gbm_baseline import (PRODUCTION_HORIZON_SPECS, WalkForwardConfig,
    build_universe_return_map, prepare_panel, select_walk_forward_samples, walk_forward_folds)
from backend.ml.research import (PRIMARY_HORIZONS, SELECTION_CUTOFFS, code_identity,
    load_snapshot, write_json_new)
from scripts.signal_research import select_frames


def assert_paired_panels(baseline, candidate, columns):
    keys = ['date', 'ticker_id']
    for panel in (baseline, candidate):
        if panel.duplicated(keys).any():
            raise ValueError('Duplicate security/date in paired panel')
    try:
        pd.testing.assert_frame_equal(baseline.set_index(keys)[columns].sort_index(),
                                      candidate.set_index(keys)[columns].sort_index())
    except AssertionError as exc:
        raise ValueError('Unmatched baseline controls, labels or sample identities') from exc


def calendar_support(dates, block):
    months = sorted({pd.Period(d, freq='M').ordinal for d in dates})
    runs = []
    for i, month in enumerate(months):
        if i == 0 or month != months[i-1]+1:
            runs.append(0)
        runs[-1] += 1
    return dict(months=len(months), contiguous_segments=runs, block_months=block,
                effective_blocks=len(months)/block,
                calendar_sufficient=bool(runs) and min(runs) >= block and len(months)/block >= 5,
                inference_computed=False)


def summarize(panel, family):
    horizon, cols = PRIMARY_HORIZONS[family], PACKS[family]
    spec = PRODUCTION_HORIZON_SPECS[horizon]
    cfg = WalkForwardConfig(max_train_months=spec.max_train_months)
    embargo = max(1, math.ceil(HORIZON_TRADING_DAYS[horizon]/21))
    folds, usable, observed = [], [], []
    for test_date, cutoff in walk_forward_folds(sorted(panel.date.unique()), cfg.min_train_months, embargo):
        if test_date > SELECTION_CUTOFFS[horizon]:
            continue
        train, test = select_walk_forward_samples(panel, horizon, spec.target_mode,
            test_date, cutoff, cfg, label_realized_before=date(2024, 1, 1))
        eligible = len(test) >= cfg.min_names and not train.empty
        both = test[cols].notna().all(axis=1)
        groups = test.dropna(subset=['sector']).groupby('sector')
        sectors = {str(k):len(g) for k,g in groups if len(g) >= 10}
        sized = {str(k):int(np.isfinite(g.log_market_cap).sum()) for k,g in groups
                 if np.isfinite(g.log_market_cap).sum() >= 10}
        row = dict(date=test_date, train_rows=len(train), train_dates=train.date.nunique(),
            test_rows=len(test), eligible_for_fitting=eligible,
            features_observed_train={c:int(train[c].notna().sum()) for c in cols},
            features_observed_test={c:int(test[c].notna().sum()) for c in cols},
            all_pack_features_observed_test=int(both.sum()),
            sector_groups_with_10_names=sectors, size_groups_with_10_names=sized,
            missing_sector_test=int(test.sector.isna().sum()),
            missing_size_test=int(test.log_market_cap.isna().sum()))
        folds.append(row)
        if eligible:
            usable.append(test_date)
            if both.any():
                observed.append(test_date)
    block = math.ceil(HORIZON_TRADING_DAYS[horizon]/21)
    return dict(family=family, horizon=horizon, folds=folds,
        runner_calendar=calendar_support(usable, block),
        pack_observed_calendar=calendar_support(observed, block),
        pack_observed_doubled_blocks=calendar_support(observed, 2*block))


def run(snapshot, family, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    inputs, manifest = load_snapshot(snapshot)
    frames = select_frames(inputs['frames'])
    spec = PRODUCTION_HORIZON_SPECS[PRIMARY_HORIZONS[family]]
    # Preserve the complete assembly grid: some label/observation endpoints use
    # its last date. The runner's actual cutoff is applied during fold selection.
    grid = build_calendar_grid(frames)
    market = build_universe_return_map(frames, membership_filter=True)
    print('Assembling baseline panel', flush=True)
    baseline = prepare_panel(frames, grid, rank_cols=list(spec.feature_cols),
                             membership_filter=True, market_returns_override=market)
    print('Assembling candidate panel', flush=True)
    candidate = prepare_panel(frames, grid, rank_cols=list(spec.feature_cols)+PACKS[family],
        membership_filter=True, market_returns_override=market, macro=inputs['macro'])
    if baseline.empty or candidate.empty:
        raise ValueError('Empty research panels')
    controls = [c for c in baseline.columns if c not in ('date', 'ticker_id')]
    assert_paired_panels(baseline, candidate, controls)
    panel_bytes = pickle.dumps(candidate, protocol=5)
    with gzip.open(output/'panel.pkl.gz', 'wb') as out:
        out.write(panel_bytes)
    report = summarize(candidate, family)
    report.update(status='diagnostic_only', input_sha256=manifest['input_sha256'],
        code=code_identity(), panel_sha256=hashlib.sha256(panel_bytes).hexdigest(),
        baseline_candidate_controls_match=True, model_fits=0,
        source_statuses={k:v.get('status') for k,v in manifest['metadata']['sources'].items()},
        frame_census=dict(total=len(inputs['frames']), selected=len(frames),
            without_prices=sum(not f.prices for f in inputs['frames']),
            without_membership=sum(not f.membership for f in inputs['frames']),
            without_sector_intervals=sum(f.sector_history == [] for f in inputs['frames'])),
        unknown_sector_full_panel_rows=int(candidate.sector.isna().sum()),
        unknown_sector_full_panel_by_security={str(tid):len(rows) for tid,rows in
            candidate[candidate.sector.isna()].groupby('ticker_id')},
        label_status_counts={c:dict(Counter(candidate[c])) for c in candidate
                             if c.startswith('outcome_')},
        limitations=['Source gates remain authoritative; this audit cannot certify them.',
            'Feature-missing rows remain in the registered runner; coverage is not a new sample filter.',
            'Calendar support is necessary, not proof of finite ICs or statistical power.',
            'Predictions and return-performance statistics were not computed.'])
    write_json_new(output/'report.json', report)
    print({k:report[k] for k in ('frame_census','runner_calendar','pack_observed_calendar')})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--family', choices=['stress','macro'], required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    run(a.snapshot, a.family, a.output)
