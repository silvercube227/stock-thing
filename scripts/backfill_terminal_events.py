"""Apply reviewed cash terminal outcomes only after issuer/security and price-basis checks."""
import argparse
import asyncio
from datetime import date
import hashlib
import json
from pathlib import Path

import httpx

from backend.config import get_settings
from backend.ingestion.db import pool_context
from backend.ml.dataset import load_frames
from backend.ml.factors.pit import documented_targets
from backend.ml.research import write_json_new


async def run(reviews, output):
    raw = Path(reviews).read_bytes()
    review = json.loads(raw)
    identity_raw = Path(review['identity_artifact']).read_bytes()
    if hashlib.sha256(identity_raw).hexdigest() != review['identity_sha256']:
        raise ValueError('Security identity evidence changed')
    identities = json.loads(identity_raw)['sources']['security_identity']['records']
    evidence=[]
    async with httpx.AsyncClient(timeout=60,follow_redirects=True,
            headers={'User-Agent':get_settings().sec_edgar_user_agent}) as client:
        for event in review['events']:
            response = await client.get(event['source'])
            response.raise_for_status()
            source_raw = response.content
            digest = hashlib.sha256(source_raw).hexdigest()
            path = Path('.research/terminal_sources') / f'{digest}.html'
            path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():
                path.write_bytes(source_raw)
            evidence.append({**event,'source_sha256':digest,'archived_source':str(path)})
    result={'events':[]}
    async with pool_context(max_size=1) as pool:
        async with pool.acquire() as conn, conn.transaction():
            for event in evidence:
                d=date.fromisoformat(event['effective_date'])
                matches=[r for r in identities if r['RIC']==event['ric'] and r['ISIN']==event['isin']
                         and int(r['CIK Number'])==int(event['cik'])]
                if len(matches)!=1:
                    raise ValueError('Terminal issuer/security identity is not established')
                tickers=await conn.fetch('select ticker_id,cik from tickers where symbol=$1',event['symbol'])
                if len(tickers)!=1 or int(tickers[0]['cik'])!=int(event['cik']):
                    raise ValueError('Database issuer differs from reviewed security')
                tid=tickers[0]['ticker_id']
                tail=await conn.fetchrow('''select trade_date,close,adj_close,source from price_history
                    where ticker_id=$1 order by trade_date desc limit 1''',tid)
                if (not tail or str(tail['trade_date'])!=event['last_trade_date'] or
                    tail['close'] is None or tail['close']<=0 or tail['adj_close']!=tail['close']):
                    raise ValueError('Terminal unit price basis cannot be established; no automatic rescaling')
                provenance=json.dumps({'review_sha256':hashlib.sha256(raw).hexdigest(),**event},sort_keys=True)
                previous=await conn.fetchrow('select * from security_events where ticker_id=$1 and effective_date=$2',tid,d)
                if previous:
                    if previous['source']!=provenance:
                        raise ValueError('Conflicting terminal event already exists')
                else:
                    await conn.execute('''insert into security_events
                        (ticker_id,effective_date,event_type,consideration_type,cash_value,proceeds_basis,source,verified)
                        values($1,$2,'acquisition','cash',$3,'adj_close',$4,true)''',tid,d,event['cash_value'],provenance)
                frame=(await load_frames(conn,symbols=[event['symbol']]))[0]
                # Last valid pre-close entry must now realize documented cash at every mature horizon.
                labels=documented_targets(frame.prices,len(frame.prices)-2,frame.security_events,date(2020,1,1))
                if not all(v[1] and v[4]=='acquisition' for v in labels.values()):
                    raise ValueError('Cash terminal labels failed live verification')
                result['events'].append({'ticker_id':tid,**event,'verified_labels':labels})
    write_json_new(output,result)
    print({'applied':len(result['events']),'cash_label_checks':'passed'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reviews',default='docs/terminal_event_reviews.json')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    asyncio.run(run(args.reviews,args.output))
