"""Audit archived raw histories against exact-RIC membership and exchange sessions."""
import argparse
from collections import defaultdict, Counter
from datetime import date
import hashlib
import json
import math
from pathlib import Path

from backend.ingestion.calendar import trading_days_between
from backend.ml.research import write_json_new


def summarize(records, intervals, sessions):
    valid,invalid,duplicates=set(),[],[]
    seen=set()
    for row in records:
        d=row['Date'][:10]
        if d in seen: duplicates.append(d)
        seen.add(d)
        try:
            close=float(row['TRDPRC_1'])
            ok=math.isfinite(close) and close>0
        except (KeyError,TypeError,ValueError):ok=False
        if ok:valid.add(d)
        else:invalid.append(d)
    expected={str(d) for d in sessions if any(
        r['valid_from']<=str(d) and (r['valid_to'] is None or str(d)<r['valid_to'])
        for r in intervals)}
    missing=sorted(expected-valid)
    return dict(valid_price_dates=len(valid),membership_sessions=len(expected),
                missing_membership_sessions=len(missing),missing_dates=missing,
                invalid_price_dates=invalid,duplicate_dates=duplicates,
                price_dates_outside_membership=len(valid-expected),
                status='missing_membership_prices' if missing else 'membership_prices_present',
                note='Coverage only; neither action completeness nor security continuity is certified.')


def run(output):
    archive_path=Path('.research/verification/missing-prices-backfill-live.json')
    membership_path=Path('.research/verification/membership-reconciled.json')
    actions_path=Path('.research/verification/registered-historical-actions.json')
    archive=json.loads(archive_path.read_text())
    membership=json.loads(membership_path.read_text())
    grouped=defaultdict(list)
    for interval in membership['intervals']:grouped[interval['security_id']].append(interval)
    actions=defaultdict(list)
    for row in json.loads(actions_path.read_text())['sources']['corporate_actions']['records']:
        if row['Corporate Change Event Type']:actions[row['Instrument']].append(row)
    sessions=trading_days_between(date(2010,1,1),date(2026,9,4))
    results=[]
    for security in archive['securities']:
        raw=Path(security['artifact']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=security['sha256']:raise ValueError('Price archive changed')
        payload=json.loads(raw)
        ric=security['ric']
        summary=summarize(payload['records'],grouped[ric],sessions)
        types=dict(Counter(r['Corporate Change Event Type'] for r in actions[ric]))
        results.append(dict(ric=ric,**summary,action_types=types,
                            action_status='records_require_review' if types else 'no_action_records_not_proof_of_completeness'))
    report=dict(securities=results,counts=dict(Counter(r['status'] for r in results)),
                missing_sessions=sum(r['missing_membership_sessions'] for r in results),
                input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (archive_path,membership_path,actions_path)})
    write_json_new(output,report)
    print(report['counts'],'missing sessions',report['missing_sessions'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    run(p.parse_args().output)
