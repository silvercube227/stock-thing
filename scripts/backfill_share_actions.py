"""Apply reviewed share-action ratios, preserving all stored prices and price factors."""
import argparse
import asyncio
from datetime import date
import hashlib
import json
import math
from pathlib import Path

from backend.ingestion.db import pool_context
from backend.ml.research import write_json_new


async def run(reviews, output, apply):
    raw = Path(reviews).read_bytes()
    review = json.loads(raw)
    source_raw = Path(review['source_artifact']).read_bytes()
    if hashlib.sha256(source_raw).hexdigest() != review['source_sha256']:
        raise ValueError('Corporate action source hash mismatch')
    source = json.loads(source_raw)['sources']['corporate_actions']['records']
    if review.get('mapping_artifact'):
        mapping_raw = Path(review['mapping_artifact']).read_bytes()
        if hashlib.sha256(mapping_raw).hexdigest() != review['mapping_sha256']:
            raise ValueError('Security mapping source hash mismatch')
    provenance = 'reviewed_share_action:' + hashlib.sha256(raw).hexdigest()
    report = {'applied': apply, 'provenance': provenance, 'events': []}
    async with pool_context(max_size=1) as pool:
        async with pool.acquire() as conn:
            tx = conn.transaction()
            await tx.start()
            try:
                for event in review['events']:
                    symbol, d = event['symbol'], date.fromisoformat(event['trade_date'])
                    ric = event.get('ric') or {'AAPL':'AAPL.O', 'NVDA':'NVDA.O', 'GE':'GE.N'}[symbol]
                    candidates = [r for index,r in enumerate(source) if r['Instrument'] == ric
                        and ('source_record_index' not in event or index == event['source_record_index'])
                        and r['Corporate Change Event Type'] == event['event_type']
                        and abs((date.fromisoformat(r['Capital Change Effective Date'])-d).days) < 30
                        and r['Capital Change Is Rescinded'] == 'False']
                    if len(candidates) != 1:
                        raise ValueError('Reviewed action does not match exactly one source record')
                    r = candidates[0]
                    if 'source_record_index' in event:
                        from backend.ingestion.share_actions import classify_share_action
                        if classify_share_action(d,event['expected_price_factor'],[r]) is None:
                            raise ValueError('Automatic match no longer satisfies exact-date/action-term rule')
                    ratio = 1.0 if r['Adjustment Type'] == 'LSEG Pricing Only' else (
                        float(r['Terms New Shares']) / float(r['Terms Old Shares']))
                    if not math.isfinite(ratio) or ratio <= 0 or ratio != event['share_split_factor']:
                        raise ValueError('Share ratio disagrees with source terms')
                    stored = await conn.fetch('''select p.ticker_id,p.split_factor,p.source
                        from price_history p join tickers t using(ticker_id)
                        where t.symbol=$1 and p.trade_date=$2 for update of p''', symbol, d)
                    if len(stored) != 1 or stored[0]['source'] != 'yfinance' or not math.isclose(
                            float(stored[0]['split_factor']), event['expected_price_factor'], rel_tol=1e-9):
                        raise ValueError(f'Price history differs from reviewed event: {symbol} {d}')
                    if 'ticker_id' in event and stored[0]['ticker_id'] != event['ticker_id']:
                        raise ValueError('Security mapping changed since review')
                    await conn.execute('''update price_history set share_split_factor=$3, share_action_source=$4
                        where ticker_id=$1 and trade_date=$2''', stored[0]['ticker_id'], d, ratio, provenance)
                    report['events'].append({**event, 'source_record': r})
                if apply:
                    await tx.commit()
                else:
                    await tx.rollback()
            except BaseException:
                await tx.rollback()
                raise
    write_json_new(output, report)
    print({'applied': apply, 'events': len(report['events']), 'provenance': provenance})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reviews', default='docs/share_action_reviews.json')
    p.add_argument('--output', required=True)
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    asyncio.run(run(args.reviews, args.output, args.apply))
