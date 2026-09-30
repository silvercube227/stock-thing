"""Independent preservation and price/dividend parity checks on the PX candidate."""

import argparse
from dataclasses import asdict
from datetime import date
import hashlib
import json
import math
from pathlib import Path

from backend.ml.factors.pit import as_traded_closes
from backend.ml.research import json_value, load_snapshot, write_json_new


def fingerprint(frame):
    return hashlib.sha256(json.dumps(json_value(asdict(frame)), sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def run(parent, candidate, output):
    before, original = load_snapshot(parent)
    after, staged = load_snapshot(candidate)
    if staged['metadata']['parent_input_sha256'] != original['input_sha256']:
        raise ValueError('Wrong parent snapshot')
    for path, digest in staged['metadata']['source_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('Stage source changed')
    a, b = ({f.ticker_id:f for f in data['frames']} for data in (before, after))
    if a.keys() != b.keys() or before['macro'] != after['macro']:
        raise ValueError('Unrelated snapshot inputs changed')
    if original['metadata']['sources'] != staged['metadata']['sources']:
        raise ValueError('Staging changed gate statuses')
    if any(v['status'] == 'verified' for k,v in staged['metadata']['sources'].items()
           if k in ('price_share_basis','membership','terminal_outcomes')):
        raise ValueError('Candidate must retain incomplete gates')
    changed = [tid for tid in a if fingerprint(a[tid]) != fingerprint(b[tid])]
    if set(changed) != {360,947}:
        raise ValueError('Unexpected changed security')
    boundary = date(2018,10,31)
    expected = [p for p in a[360].prices if p['trade_date'] >= boundary]
    if expected != b[360].prices:
        raise ValueError('Successor prices changed beyond the reviewed date boundary')
    allowed = {'prices','membership','sector_history','security_events','removed_at'}
    for tid in changed:
        left, right = asdict(a[tid]), asdict(b[tid])
        if json_value({k:v for k,v in left.items() if k not in allowed}) != json_value(
                {k:v for k,v in right.items() if k not in allowed}):
            raise ValueError('Issuer filings or unrelated security attributes changed')
    target = b[947]
    old = {p['trade_date']:(p, nominal) for p,nominal in
           zip(a[360].prices, as_traded_closes(a[360].prices), strict=True)}
    nominal_errors, cash_conflicts, return_errors = [], [], []
    for i, p in enumerate(target.prices):
        d = p['trade_date']
        if d not in old:
            raise ValueError('Independent historical price date missing')
        q, nominal = old[d]
        nominal_errors.append(abs(float(p['close'])/nominal-1))
        if not math.isclose(float(p['dividend']), float(q['dividend']), abs_tol=1e-6):
            cash_conflicts.append(dict(date=d, native=float(p['dividend']), stored=float(q['dividend'])))
        if i:
            prev = target.prices[i-1]
            old_prev = old[prev['trade_date']][0]
            expected = float(q['adj_close'])/float(old_prev['adj_close'])
            actual = float(p['adj_close'])/float(prev['adj_close'])
            error = abs(actual/expected-1)
            if error > 1e-6:
                return_errors.append(dict(date=d, relative_error=error))
    report = dict(status='candidate_verification_not_global_certification', database_writes=0,
        parent_input_sha256=original['input_sha256'], candidate_input_sha256=staged['input_sha256'],
        unchanged_frames=len(a)-len(changed), changed_ids=changed,
        macro_unchanged=True, gate_statuses_unchanged=True, issuer_attributes_unchanged=True,
        native_nominal_comparisons=len(nominal_errors), max_nominal_relative_error=max(nominal_errors),
        nominal_parity_pass=max(nominal_errors) <= 1e-6, cash_conflicts=cash_conflicts,
        adjusted_daily_return_conflicts=return_errors,
        native_cash_dates=sum(float(p['dividend']) != 0 for p in target.prices),
        note='The old LIN history is used only to corroborate PX prices before the merger, '
             'not to assert that PX and LIN are the same security. Cross-feed agreement is '
             'evidence, not a guarantee that both vendors captured every distribution.')
    write_json_new(output, report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('cash_conflicts','adjusted_daily_return_conflicts')}, indent=2))
    print(dict(cash_conflicts=len(cash_conflicts), daily_return_conflicts=len(return_errors)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    run(a.parent, a.candidate, a.output)
