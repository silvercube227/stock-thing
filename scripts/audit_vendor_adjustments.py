"""Compare vendor raw/adjusted histories; never infer share ratios from prices."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from backend.ml.research import write_json_new


def load_rows(entry):
    raw=Path(entry['artifact']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Price archive changed')
    return json.loads(raw)


def changes(raw_rows, adjusted_rows):
    def index(rows):
        result={}
        for row in rows:
            d=row['Date'][:10]
            if d in result:raise ValueError('Duplicate vendor date')
            try:v=float(row['TRDPRC_1'])
            except (ValueError,TypeError,KeyError):continue
            if math.isfinite(v) and v>0:result[d]=v
        return result
    raw,adjusted=index(raw_rows),index(adjusted_rows)
    dates=sorted(raw.keys() & adjusted.keys())
    events=[]
    previous=None
    for d in dates:
        ratio=adjusted[d]/raw[d]
        if previous and not math.isclose(ratio,previous[1],rel_tol=1e-5):
            events.append(dict(date=d,previous_date=previous[0],
                               price_adjustment_ratio=ratio/previous[1],
                               share_ratio=None))
        previous=(d,ratio)
    return dict(matched_dates=len(dates),raw_only_dates=sorted(raw.keys()-adjusted.keys()),
                adjusted_only_dates=sorted(adjusted.keys()-raw.keys()),events=events,
                final_raw_to_adjusted_scale=previous[1] if previous else None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',default='.research/verification/missing-prices-backfill-live.json')
    p.add_argument('--adjusted',default='.research/verification/missing-prices-adjusted.json')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    raw_manifest=json.loads(Path(a.raw).read_text())
    adjusted={r['ric']:r for r in json.loads(Path(a.adjusted).read_text())['securities']}
    results=[]
    for entry in raw_manifest['securities']:
        ric=entry['ric']
        if ric not in adjusted:raise ValueError('Missing adjusted history')
        left,right=load_rows(entry),load_rows(adjusted[ric])
        if left['request']['adjustments']!='unadjusted' or right['request']['adjustments']!=['CCH','CRE','RPO','RTS']:
            raise ValueError('Unexpected adjustment conventions')
        if 'records' not in right:
            results.append(dict(ric=ric,status='adjusted_source_failed'))
            continue
        result=changes(left['records'],right['records'])
        status='date_mismatch' if result['raw_only_dates'] or result['adjusted_only_dates'] else 'aligned_requires_event_review'
        results.append(dict(ric=ric,status=status,**result))
    report=dict(securities=results,counts=dict(Counter(r['status'] for r in results)),
                note='Price basis reconciliation only. Ratios are not share-count adjustments or proof of complete dividends.')
    write_json_new(a.output,report)
    print(report['counts'],'price adjustment steps',sum(len(r.get('events',[])) for r in results))
