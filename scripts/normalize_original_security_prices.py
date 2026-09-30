"""Build original-security recovery candidates and disclose distribution conflicts."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle

from backend.ingestion.dividend_review import review_dividends
from backend.ingestion.prices_lseg import normalize_vendor_pair
from backend.ml.research import write_json_new
from scripts.audit_vendor_adjustments import changes, load_rows
from scripts.normalize_historical_prices import trim_unpriced_boundaries


def run(output, embedded_cash_review=None):
    paths = dict(review=Path('docs/original_security_price_review.json'),
                 raw=Path('.research/verification/original-security-native-prices.json'),
                 adjusted=Path('.research/verification/original-security-adjusted-prices.json'),
                 distributions=Path('.research/verification/original-security-distributions.json'),
                 parity=Path('.research/verification/original-security-price-parity.json'))
    blobs = {k:p.read_bytes() for k,p in paths.items()}
    data = {k:json.loads(v) for k,v in blobs.items()}
    from scripts.review_embedded_cash import load_review
    embedded = load_review(embedded_cash_review, paths['raw'], paths['adjusted'])
    if embedded_cash_review:
        paths['embedded_cash_review'] = Path(embedded_cash_review)
        blobs['embedded_cash_review'] = paths['embedded_cash_review'].read_bytes()
    for kind in ('raw', 'adjusted'):
        if data[kind]['inventory_sha256'] != hashlib.sha256(blobs['review']).hexdigest():
            raise ValueError('Price inventory changed')
    raw = {r['ric']:r for r in data['raw']['securities']}
    adjusted = {r['ric']:r for r in data['adjusted']['securities']}
    with gzip.open(data['parity']['stored_capture'], 'rb') as stream:
        captured = stream.read()
    if hashlib.sha256(captured).hexdigest() != data['parity']['stored_capture_sha256']:
        raise ValueError('Stored evidence changed')
    stored = json.loads(captured)
    normalized, results = {}, []
    for security in data['review']['securities']:
        ric, tid = security['ric'], security['ticker_id']
        left, right = load_rows(raw[ric]), load_rows(adjusted[ric])
        if left['request']['adjustments'] != 'unadjusted' or right['request']['adjustments'] != ['CCH','CRE','RPO','RTS']:
            raise ValueError('Unexpected price adjustment convention')
        review = review_dividends(
            [r for r in data['distributions']['sources']['dividends']['records'] if r['Instrument'] == ric],
            [r['Date'][:10] for r in left['records']])
        capital = changes(left['records'], right['records'])
        result = dict(ric=ric, ticker_id=tid, symbol=security['symbol'], dividend_review=review,
                      capital_adjustments=capital, status='candidate_requires_source_certification')
        try:
            if review['issues']:
                raise ValueError('Unresolved dividend parsing issues')
            a,b,missing = trim_unpriced_boundaries(left['records'],right['records'],review['cash_by_ex_date'])
            included = embedded.get(ric, {}).get('embedded_cash', {})
            if any(review['cash_by_ex_date'].get(d) != r['total_cash'] for d,r in included.items()):
                raise ValueError('Reviewed cash totals changed')
            normalized[ric] = normalize_vendor_pair(tid,a,b,review['cash_by_ex_date'], embedded_cash=included)
            result['embedded_cash_review'] = included
            result.update(rows=len(normalized[ric]), missing_boundaries=missing)
        except ValueError as exc:
            result.update(status='blocked_normalization', reason=str(exc))
        # This comparison is meaningful only for HOT, whose stored nominal
        # prices match the native instrument. COL/HAR are disproved identities.
        if security['symbol'] == 'HOT':
            cash = review['cash_by_ex_date']
            step_dates = {r['date'] for r in capital['events']}
            prices = [p for p in stored[str(tid)]['prices']
                      if left['records'][0]['Date'][:10] <= p['trade_date'] <= max(r['Date'][:10] for r in left['records'])]
            result['cash_conflicts_excluding_capital_steps'] = [
                dict(date=p['trade_date'], stored_dividend=p['dividend'],
                     native_dividend=cash.get(p['trade_date']), reason='cash_amount_or_ex_date_disagreement')
                for p in prices if p['trade_date'] not in step_dates
                and not math.isclose(float(p['dividend']),float(cash.get(p['trade_date'],0)),abs_tol=1e-6)]
        results.append(result)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    payload = pickle.dumps(normalized, protocol=5)
    with (output/'prices.pkl.gz').open('xb') as dest, gzip.GzipFile(fileobj=dest, mode='wb', mtime=0) as stream:
        stream.write(payload)
    report = dict(securities=results, price_rows=sum(len(v) for v in normalized.values()),
                  input_sha256=hashlib.sha256(payload).hexdigest(), database_writes=0,
                  source_hashes={str(paths[k]):hashlib.sha256(v).hexdigest() for k,v in blobs.items()},
                  status='recovery_candidates_not_certified', pending=[
                      'Cash and noncash distribution completeness, including disclosed HOT conflicts',
                      'Original-security history application with archived preimages',
                      'Terminal consideration on certified price bases',
                      'Reconstructed membership and historical sectors'])
    write_json_new(output/'report.json', report)
    print({'candidate_rows':report['price_rows'], 'securities':len(normalized), 'database_writes':0})
    for r in results:
        print({k:r[k] for k in ('symbol','status','cash_conflicts_excluding_capital_steps') if k in r})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--embedded-cash-review')
    a = p.parse_args()
    run(a.output, a.embedded_cash_review)
