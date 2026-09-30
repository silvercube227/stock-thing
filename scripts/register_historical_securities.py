"""Register separately identified research securities with archived source histories.

The internal symbol is namespaced by exact RIC. No current ticker is reused,
no historical sector is inferred, and production loader defaults exclude these
research-only rows until the reconstructed reference is explicitly enabled.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from backend.ingestion.db import pool_context
from backend.ml.research import write_json_new


def prepare(inventory, archive):
    histories={r['ric']:r for r in archive['securities']}
    rows=[]
    for security in inventory['securities']:
        if security['status']!='missing_security_requires_backfill':
            continue
        history=histories.get(security['ric'])
        if not history or history['price_rows']<=0 or history['status']!='observed_requires_action_and_coverage_review':
            continue
        if len(security['isins'])!=1 or security['existing_candidate_ids']:
            raise ValueError('Ambiguous security must not be registered')
        blob=Path(history['artifact']).read_bytes()
        if hashlib.sha256(blob).hexdigest()!=history['sha256']:
            raise ValueError('Raw historical price archive changed')
        rows.append(dict(ric=security['ric'],symbol='LSEG:'+security['ric'],
                         cik=security['issuer_ciks'][0].zfill(10) if len(security['issuer_ciks'])==1 else None,
                         source_isin=security['isins'][0], price_artifact=history['artifact'],
                         price_sha256=history['sha256']))
    if len({r['ric'] for r in rows})!=len(rows):
        raise ValueError('Duplicate native RIC')
    return rows


async def run(inventory_path,archive_path,output,apply):
    inventory_raw,archive_raw=Path(inventory_path).read_bytes(),Path(archive_path).read_bytes()
    inventory,archive=json.loads(inventory_raw),json.loads(archive_raw)
    if archive['inventory_sha256']!=hashlib.sha256(inventory_raw).hexdigest():
        raise ValueError('Price archive belongs to a different security inventory')
    rows=prepare(inventory,archive)
    report=dict(applied=apply,inventory_sha256=hashlib.sha256(inventory_raw).hexdigest(),
                archive_sha256=hashlib.sha256(archive_raw).hexdigest(),securities=[])
    async with pool_context(max_size=1) as pool:
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute("select pg_advisory_xact_lock(hashtext('register_historical_securities'))")
            for row in rows:
                existing=await conn.fetch('select ticker_id,ric,symbol,research_only,active from tickers where ric=$1 or symbol=$2',row['ric'],row['symbol'])
                if existing and (len(existing)!=1 or existing[0]['ric']!=row['ric'] or
                                 existing[0]['symbol']!=row['symbol'] or not existing[0]['research_only'] or existing[0]['active']):
                    raise ValueError('Existing security conflicts with research registration')
                tid=existing[0]['ticker_id'] if existing else None
                if apply and not existing:
                    tid=await conn.fetchval('''insert into tickers(symbol,ric,cik,name,asset_type,active,research_only)
                        values($1,$2,$3,$2,'equity',false,true) returning ticker_id''',row['symbol'],row['ric'],row['cik'])
                report['securities'].append(dict(**row,ticker_id=tid,status='existing' if existing else 'created' if apply else 'proposed'))
    write_json_new(output,report)
    print({'applied':apply,'securities':len(report['securities'])})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',default='.research/verification/missing-security-inventory.json')
    p.add_argument('--archive',default='.research/verification/missing-prices-backfill-live.json')
    p.add_argument('--output',required=True)
    p.add_argument('--apply',action='store_true')
    a=p.parse_args()
    asyncio.run(run(a.inventory,a.archive,a.output,a.apply))
