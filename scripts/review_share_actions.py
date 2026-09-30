"""Generate reproducible exact-date/action-term matches; leave ambiguous events unresolved."""
import argparse
import hashlib
import json
from pathlib import Path

from backend.ingestion.share_actions import classify_share_action
from backend.ml.research import write_json_new


def run(mapping_path, source_path, output):
    mapping_raw, source_raw = Path(mapping_path).read_bytes(), Path(source_path).read_bytes()
    mapping, source = json.loads(mapping_raw), json.loads(source_raw)['sources']['corporate_actions']['records']
    by_id = {r['candidate_ticker_ids'][0]:r for r in mapping['mappings']
             if r['status']=='unique_candidate_requires_history_review' and len(r['symbols'])==1}
    events, unresolved = [], []
    for action in mapping['unclassified_actions']:
        match = by_id.get(action['ticker_id'])
        if match is None:
            unresolved.append({**action,'reason':'security mapping not unique'})
            continue
        scoped = [(i,r) for i,r in enumerate(source) if r['Instrument']==match['ric']]
        result = classify_share_action(action['trade_date'],action['split_factor'],[r for _,r in scoped])
        if result is None:
            unresolved.append({**action,'ric':match['ric'],'reason':'no unique exact-date/action-term match'})
            continue
        index, ratio = result
        events.append(dict(ticker_id=action['ticker_id'],symbol=match['symbols'][0],ric=match['ric'],
                           trade_date=action['trade_date'],expected_price_factor=float(action['split_factor']),
                           share_split_factor=ratio,event_type=scoped[index][1]['Corporate Change Event Type'],
                           source_record_index=scoped[index][0]))
    report = {'source_artifact':source_path,'source_sha256':hashlib.sha256(source_raw).hexdigest(),
              'mapping_artifact':mapping_path,'mapping_sha256':hashlib.sha256(mapping_raw).hexdigest(),
              'rule':'Exact event date, non-rescinded, source action type and ratio corroborate observed adjustment; no historical membership approval.',
              'events':events,'unresolved':unresolved}
    write_json_new(output,report)
    print({'corroborated':len(events),'unresolved':len(unresolved)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mapping',required=True)
    p.add_argument('--source',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    run(a.mapping,a.source,a.output)
