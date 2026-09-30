"""Sequential, resumable raw LSEG histories keyed by exact historical security RIC.

This archive precedes adjustment/sector certification and never writes model-ready
prices or infers missing corporate actions. Every response is hash-addressed.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from backend.config import get_settings
from backend.ingestion.estimates import _open_session
from backend.ml.research import write_json_new

FIELDS=['OPEN_PRC','HIGH_1','LOW_1','TRDPRC_1','ACVOL_UNS']


def run(inventory_path, root, output, end, adjusted=False, start='2010-01-01'):
    import lseg.data as ld
    raw=Path(inventory_path).read_bytes()
    inventory=json.loads(raw)
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    completed=[]
    adjustments=['CCH','CRE','RPO','RTS'] if adjusted else 'unadjusted'
    _open_session(get_settings().lseg_app_key)
    try:
        securities=[s for s in inventory['securities'] if s['status'] in (
            'missing_security_requires_backfill', 'unverified_existing_history_requires_backfill')]
        for index,security in enumerate(securities):
            request=dict(ric=security['ric'], fields=FIELDS, start=start,end=end,
                         interval='1d',adjustments=adjustments,count=10000)
            key=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
            marker=root/f'{key}.json'
            if marker.exists():
                record=json.loads(marker.read_text())
                blob=Path(record['artifact']).read_bytes()
                if hashlib.sha256(blob).hexdigest()!=record['sha256']:
                    raise ValueError('Cached historical response changed')
            else:
                try:
                    df=ld.get_history(universe=request['ric'],fields=FIELDS,
                                      start=request['start'],end=end,interval='1d',
                                      adjustments=adjustments,count=10000)
                    if len(df)>=10000:
                        raise ValueError('Capped response requires subdivision')
                    rows=df.reset_index().astype(str).to_dict('records')
                    count=int(df['TRDPRC_1'].notna().sum()) if 'TRDPRC_1' in df else 0
                    payload=dict(request=request, records=rows,
                                 non_null={str(c):int(df[c].notna().sum()) for c in df.columns},
                                 status='observed_requires_action_and_coverage_review' if count else 'empty_prices')
                except Exception as exc:
                    if 'Session is not opened' in str(exc):
                        raise RuntimeError('Desktop session unavailable; stop before caching source failures') from exc
                    payload=dict(request=request,status='error',error=type(exc).__name__,detail=str(exc)[:400])
                    count=0
                payload['retrieved_at']=datetime.now(timezone.utc).isoformat()
                blob=json.dumps(payload,sort_keys=True).encode()
                digest=hashlib.sha256(blob).hexdigest()
                artifact=root/f'{digest}.raw.json'
                if not artifact.exists():
                    with artifact.open('xb') as stream:stream.write(blob)
                record=dict(ric=security['ric'],status=payload['status'],price_rows=count,
                            artifact=str(artifact),sha256=digest,request=request)
                write_json_new(marker,record)
            completed.append(record)
            print(f"{index+1}/{len(securities)} {security['ric']}: {record['status']} ({record['price_rows']} prices)",flush=True)
    finally:
        ld.close_session()
    report=dict(inventory_sha256=hashlib.sha256(raw).hexdigest(),securities=completed,
                counts=dict(Counter(r['status'] for r in completed)),
                total_price_rows=sum(r['price_rows'] for r in completed),database_writes=0,
                note='Raw venue-level histories; action completeness, dividends and historical sectors are not certified.')
    write_json_new(output,report)
    print(report['counts'],report['total_price_rows'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory',default='.research/verification/missing-security-inventory.json')
    parser.add_argument('--root',default='.research/missing-price-archive')
    parser.add_argument('--end',default='2026-09-04')
    parser.add_argument('--start',default='2010-01-01')
    parser.add_argument('--output',required=True)
    parser.add_argument('--adjusted',action='store_true',help='Archive vendor capital-action adjustments; ordinary dividends remain separate')
    a=parser.parse_args()
    if datetime.fromisoformat(a.start) > datetime.fromisoformat(a.end):
        parser.error('--start must not follow --end')
    run(a.inventory,a.root,a.output,a.end,a.adjusted,a.start)
