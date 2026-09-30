"""Replay source-to-candidate price parity without invoking the normalizer."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle

from backend.ml.research import write_json_new
from scripts.audit_vendor_adjustments import load_rows


def run(candidate, raw_manifest, adjusted_manifest, output):
    root = Path(candidate)
    report = json.loads((root/'report.json').read_text())
    with gzip.open(root/'prices.pkl.gz', 'rb') as stream:
        payload = stream.read()
    if hashlib.sha256(payload).hexdigest() != report['input_sha256']:
        raise ValueError('Candidate payload changed')
    for path in (raw_manifest, adjusted_manifest):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() not in report['source_hashes'].values():
            raise ValueError('Candidate source manifest changed')
    raw = {r['ric']:r for r in json.loads(Path(raw_manifest).read_text())['securities']}
    adjusted = {r['ric']:r for r in json.loads(Path(adjusted_manifest).read_text())['securities']}
    reviews = {r['ric']:r for r in report['securities']}
    rows_by_ric = pickle.loads(payload)
    results = []
    for ric, rows in rows_by_ric.items():
        native = {r['Date'][:10]:r for r in load_rows(raw[ric])['records']}
        capital = {r['Date'][:10]:r for r in load_rows(adjusted[ric])['records']}
        embedded = reviews[ric].get('embedded_cash_review', {})
        max_error = 0
        for i,row in enumerate(rows):
            d = str(row['trade_date'])
            if row['close'] != float(native[d]['TRDPRC_1']) or row['price_basis'] != 'as_traded':
                raise ValueError('Nominal source parity failed')
            if row['ticker_id'] != reviews[ric]['ticker_id']:
                raise ValueError('Candidate security identity changed')
            if not math.isfinite(row['adj_close']) or row['adj_close'] <= 0:
                raise ValueError('Invalid candidate adjusted price')
            if i == 0:
                continue
            prior = rows[i-1]
            pd = str(prior['trade_date'])
            ci, cp = float(capital[d]['TRDPRC_1']), float(capital[pd]['TRDPRC_1'])
            remaining = row['dividend']-float(embedded.get(d, {}).get('amount', 0))
            # Forward ratio of the source series after removing only cash not
            # already embedded. The normalizer works backward on level factors.
            expected = ci/(cp-remaining*ci/row['close'])
            actual = row['adj_close']/prior['adj_close']
            error = abs(actual/expected-1)
            max_error = max(max_error,error)
            if not math.isclose(actual, expected, rel_tol=1e-12):
                raise ValueError(f'Adjusted return parity failed: {ric} {d}')
        if rows[-1]['adj_close'] != float(capital[str(rows[-1]['trade_date'])]['TRDPRC_1']):
            raise ValueError('Final source scale changed')
        results.append(dict(ric=ric, rows=len(rows), max_relative_return_error=max_error))
    receipt = dict(candidate_sha256=report['input_sha256'],
                   candidate_report_sha256=hashlib.sha256((root/'report.json').read_bytes()).hexdigest(),
                   securities=results, rows=sum(r['rows'] for r in results),
                   status='transformation_parity_passed_source_certification_still_required')
    write_json_new(output, receipt)
    print({'securities':len(results),'rows':receipt['rows'],
           'max_relative_return_error':max(r['max_relative_return_error'] for r in results)})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', required=True)
    p.add_argument('--raw', default='.research/verification/missing-prices-backfill-live.json')
    p.add_argument('--adjusted', default='.research/verification/missing-prices-adjusted.json')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    run(a.candidate,a.raw,a.adjusted,a.output)
