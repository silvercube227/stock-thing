"""Offline correctness and accounting coverage checks on an immutable audit snapshot."""
import argparse
from collections import defaultdict
from datetime import date

import numpy as np
import pandas as pd
import json
from pathlib import Path

from backend.ml.factors.long_horizon import accounting_features, ACCOUNTING_FEATURES
from backend.ml.factors.pit import as_traded_closes
from backend.ml.research import load_snapshot, write_json_new


def run(snapshot, price_probe, output):
    payload, manifest = load_snapshot(snapshot)
    probe = json.loads(Path(price_probe).read_text())['sources']
    report = {'snapshot_sha256': manifest['input_sha256'], 'price_comparisons': [],
              'accounting_coverage': [], 'limitations': [
                  'Existing membership and historical sectors are not certified.',
                  'Price probes are selected cases, not proof of full action coverage.',
                  'Coverage is diagnostic and does not select features or evaluate returns.']}
    by_symbol = {f.symbol: f for f in payload['frames']}
    for symbol in ('AAPL', 'NVDA', 'GE'):
        frame = by_symbol[symbol]
        computed = dict(zip([r['trade_date'] for r in frame.prices], as_traded_closes(frame.prices)))
        rows = []
        for r in probe['price_basis_' + symbol.lower()]['records']:
            d = date.fromisoformat(r['Date'][:10])
            source = float(r['close_adjusted_0'])
            value = computed.get(d, float('nan'))
            rows.append({'date': d, 'lseg_as_traded': source, 'reconstructed': value,
                         'relative_error': abs(value/source - 1)})
        report['price_comparisons'].append({'symbol': symbol, 'observations': rows,
            'max_relative_error': max(r['relative_error'] for r in rows),
            'recorded_actions': [dict(date=r['trade_date'], factor=float(r['split_factor']))
                                 for r in frame.prices if r['split_factor'] != 1]})
    dates = [d.date() for d in pd.date_range('2010-01-01', '2022-12-31', freq='ME')]
    counts = defaultdict(lambda: defaultdict(int))
    for n, frame in enumerate(payload['frames']):
        membership = frame.membership or []
        eligible = [d for d in dates if any(r['valid_from'] <= d and
                     (r.get('valid_to') is None or d < r['valid_to']) for r in membership)]
        if not eligible:
            continue
        sectors = []
        for d in eligible:
            matching = [r['sector'] for r in frame.sector_history or [] if r['valid_from'] <= d
                        and (r.get('valid_to') is None or d < r['valid_to'])]
            sectors.append(matching[0] if len(matching) == 1 else None)
        values = accounting_features(frame.accounting_facts or [], eligible, sectors)
        for d, sector, value in zip(eligible, sectors, values):
            group = counts[(d.year, 'unknown_sector' if sector is None else
                            'financials' if sector == 'Financials' else 'nonfinancial')]
            group['observations'] += 1
            for feature in ACCOUNTING_FEATURES:
                group[feature] += int(np.isfinite(value[feature]))
        if n % 100 == 0:
            print(f'coverage: {n} securities processed', flush=True)
    for (year, sector_group), group in sorted(counts.items()):
        report['accounting_coverage'].append({'year': year, 'sector_group': sector_group,
            'observations': group['observations'],
            'available_fraction': {c: group[c]/group['observations'] for c in ACCOUNTING_FEATURES}})
    write_json_new(output, report)
    print({'price_comparisons': [{k:v for k,v in r.items() if k != 'observations'}
                                 for r in report['price_comparisons']],
           'coverage_groups': len(report['accounting_coverage'])})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--price-probe', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.snapshot, args.price_probe, args.output)
