"""Build a compact research price candidate; source readiness remains explicit."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import pickle
from pathlib import Path

from backend.ingestion.prices_lseg import normalize_vendor_pair
from backend.ml.research import write_json_new
from scripts.audit_vendor_adjustments import load_rows


def trim_unpriced_boundaries(raw_rows, adjusted_rows, cash_dividends):
    """Retain available bars without inventing prices for boundary placeholders.

    Interior gaps and disagreement between sources still require review. Omitted
    dates are reported explicitly and must remain missing in coverage and labels.
    """
    def index(rows):
        result = {}
        for row in rows:
            day = row['Date'][:10]
            if day in result:
                raise ValueError('duplicate vendor date')
            result[day] = row
        return result

    def priced(row):
        try:
            value = float(row['TRDPRC_1'])
            return math.isfinite(value) and value > 0
        except (KeyError, ValueError, TypeError):
            return False

    raw, adjusted = index(raw_rows), index(adjusted_rows)
    if raw.keys() != adjusted.keys():
        raise ValueError('raw/adjusted date mismatch')
    valid, missing = [], []
    for day in sorted(raw):
        left, right = priced(raw[day]), priced(adjusted[day])
        if left != right:
            raise ValueError(f'raw/adjusted price availability mismatch: {day}')
        (valid if left else missing).append(day)
    if not valid:
        raise ValueError('no valid source prices')
    for day in missing:
        if valid[0] < day < valid[-1]:
            raise ValueError(f'interior price gap requires review: {day}')
        if day in cash_dividends:
            raise ValueError(f'distribution on unpriced boundary: {day}')
    return ([raw[d] for d in valid], [adjusted[d] for d in valid],
            [dict(date=d, reason='both_sources_missing_close', price_available=False)
             for d in missing])


def run(output, embedded_cash_review=None):
    paths={
        'raw':Path('.research/verification/missing-prices-backfill-live.json'),
        'adjusted':Path('.research/verification/missing-prices-adjusted.json'),
        'dividends':Path('.research/verification/historical-dividend-review.json'),
        'registration':Path('.research/verification/historical-registration-applied.json')}
    data={k:json.loads(p.read_text()) for k,p in paths.items()}
    from scripts.review_embedded_cash import load_review
    embedded = load_review(embedded_cash_review, paths['raw'], paths['adjusted'])
    if embedded_cash_review:
        paths['embedded_cash_review'] = Path(embedded_cash_review)
    raw={r['ric']:r for r in data['raw']['securities']}
    adjusted={r['ric']:r for r in data['adjusted']['securities']}
    dividends={r['ric']:r for r in data['dividends']['securities']}
    result=[]
    normalized={}
    for security in data['registration']['securities']:
        ric=security['ric']
        review=dividends[ric]
        if review['issues']:
            result.append(dict(ric=ric,status='blocked_dividend_source',issues=review['issues']))
            continue
        try:
            left,right=load_rows(raw[ric]),load_rows(adjusted[ric])
            if left['request']['adjustments']!='unadjusted' or right['request']['adjustments']!=['CCH','CRE','RPO','RTS']:
                raise ValueError('unexpected price adjustment convention')
            raw_rows, adjusted_rows, missing = trim_unpriced_boundaries(
                left['records'], right['records'], review['cash_by_ex_date'])
            included = embedded.get(ric, {}).get('embedded_cash', {})
            if any(review['cash_by_ex_date'].get(d) != r['total_cash'] for d,r in included.items()):
                raise ValueError('Reviewed cash totals changed')
            rows=normalize_vendor_pair(security['ticker_id'],raw_rows,adjusted_rows,review['cash_by_ex_date'],
                                       embedded_cash=included)
        except (ValueError,KeyError) as exc:
            result.append(dict(ric=ric,status='blocked_normalization',reason=str(exc)))
            continue
        normalized[ric]=rows
        result.append(dict(ric=ric,ticker_id=security['ticker_id'],status='candidate_requires_source_certification',
                           embedded_cash_review=included,
                           rows=len(rows),empty_dividend_rows=review['empty_rows'],
                           missing_price_boundaries=missing))
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    payload=pickle.dumps(normalized,protocol=5)
    with gzip.open(output/'prices.pkl.gz','wb',compresslevel=6) as stream:stream.write(payload)
    report=dict(securities=result,counts=dict(Counter(r['status'] for r in result)),
                price_rows=sum(len(r) for r in normalized.values()),input_sha256=hashlib.sha256(payload).hexdigest(),
                source_hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
                status='research_candidate_not_certified',database_writes=0,
                pending=['noncash distribution completeness','historical security continuity',
                         'cash-dividend source completeness','membership and historical sectors'])
    write_json_new(output/'report.json',report)
    print(report['counts'],report['price_rows'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--embedded-cash-review')
    a=p.parse_args()
    run(a.output,a.embedded_cash_review)
