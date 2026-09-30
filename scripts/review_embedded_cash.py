"""Explain coincident price adjustments using declared special cash, without guessing share ratios."""

import argparse
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

from backend.ingestion.dividend_review import review_dividends
from backend.ml.research import write_json_new
from scripts.audit_vendor_adjustments import changes, load_rows


def review(raw, adjusted, dividends, actions, source):
    cash = review_dividends(dividends, [r['Date'][:10] for r in raw])
    if cash['issues']:
        return {}, [{'reason':'dividend_source_issues', 'issues':cash['issues']}]
    nominal = {r['Date'][:10]:r for r in raw}
    explained, pending = {}, []
    for event in changes(raw, adjusted)['events']:
        d = event['date']
        if d not in cash['cash_by_ex_date']:
            continue
        # Group declared types before summing: identical source rows must not
        # multiply the amount. Conflicting same-type rows fail above.
        amounts = {(r['Dividend Type'], Decimal(r['Gross Dividend Amount']))
                   for r in dividends if r.get('Dividend Ex Date','')[:10] == d
                   and r.get('Dividend Type') in ('Special','Extra') and r.get('Dividend Currency') == 'USD'}
        amount = sum((value for _,value in amounts), Decimal(0))
        same_day_actions = [r for r in actions if r.get('Corporate Change Event Type') not in ('','Buyback')
                            and d in (r.get('Capital Change Ex Date','')[:10], r.get('Capital Change Effective Date','')[:10])]
        prior = float(nominal[event['previous_date']]['TRDPRC_1'])
        expected = prior/(prior-float(amount)) if 0 < amount < prior else None
        if same_day_actions or expected is None or not math.isclose(event['price_adjustment_ratio'], expected, rel_tol=1e-5):
            pending.append(dict(date=d, reason='declared_special_cash_does_not_uniquely_explain_step',
                                declared_special_cash=str(amount), observed_step=event['price_adjustment_ratio']))
            continue
        explained[d] = dict(amount=str(amount), total_cash=cash['cash_by_ex_date'][d],
                            source=source, rule='declared_special_cash_exact_factor_1e-5',
                            prior_nominal=prior, expected_step=expected,
                            observed_step=event['price_adjustment_ratio'], share_ratio=None)
    return explained, pending


def load_review(path, raw_path, adjusted_path):
    if path is None:
        return {}
    data = json.loads(Path(path).read_text())
    for name, digest in data['source_hashes'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError('Embedded-cash review input changed')
    if any(str(p) not in data['source_hashes'] for p in (raw_path, adjusted_path)):
        raise ValueError('Embedded-cash review belongs to different price inputs')
    return data['securities']


def run(raw_path, adjusted_path, dividends_path, actions_path, output):
    paths = list(dict.fromkeys([raw_path, adjusted_path, dividends_path, actions_path]))
    blobs = {p:Path(p).read_bytes() for p in paths}
    data = {p:json.loads(b) for p,b in blobs.items()}
    adjusted = {r['ric']:r for r in data[adjusted_path]['securities']}
    dividend_data = data[dividends_path]
    dividends = (dividend_data['sources']['dividends'] if 'sources' in dividend_data else dividend_data)['records']
    actions = data[actions_path]['sources']['corporate_actions']['records']
    results = {}
    for entry in data[raw_path]['securities']:
        ric = entry['ric']
        left, right = load_rows(entry), load_rows(adjusted[ric])
        if left['request']['adjustments'] != 'unadjusted' or right['request']['adjustments'] != ['CCH','CRE','RPO','RTS']:
            raise ValueError('Unexpected price adjustment convention')
        source = f'{dividends_path}#{ric};sha256={hashlib.sha256(blobs[dividends_path]).hexdigest()}'
        explained, pending = review(left['records'], right['records'],
            [r for r in dividends if r['Instrument'] == ric], [r for r in actions if r['Instrument'] == ric], source)
        if explained or pending:
            results[ric] = dict(embedded_cash=explained, pending=pending)
    report = dict(securities=results, source_hashes={p:hashlib.sha256(b).hexdigest() for p,b in blobs.items()},
                  status='price_transformation_review_only', database_writes=0,
                  rule='Only declared USD special/extra cash whose exact-date factor matches within 1e-5; reject reported coincident capital actions.',
                  source_semantics='https://community.developers.refinitiv.com/discussion/39737/dividend-adjusted-historical-prices-for-us-stocks/p1',
                  limitation='Does not certify source completeness, security identity, or share-count adjustments.')
    write_json_new(output, report)
    print({'explained_events':sum(len(r['embedded_cash']) for r in results.values()),
           'pending_events':sum(len(r['pending']) for r in results.values())})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', default='.research/verification/missing-prices-backfill-live.json')
    p.add_argument('--adjusted', default='.research/verification/missing-prices-adjusted.json')
    p.add_argument('--dividends', default='.research/verification/registered-historical-dividends.json')
    p.add_argument('--actions', default='.research/verification/registered-historical-actions.json')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    run(a.raw, a.adjusted, a.dividends, a.actions, a.output)
