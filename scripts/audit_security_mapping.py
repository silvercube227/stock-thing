"""Build explicit review candidates from security symbols AND issuer IDs, never CIK alone."""
import argparse
import asyncio
from collections import defaultdict, Counter
import hashlib
import json
import re
from pathlib import Path

from backend.ingestion.db import pool_context
from backend.ml.research import write_json_new


def normalize_symbol(value):
    """Normalize a share-class separator without dropping the class suffix."""
    return re.sub(r'^([A-Z]+)[ .-]([A-Z])$', r'\1-\2', value)


def build_candidates(data, tickers):
    grouped = defaultdict(list)
    for r in data['records']:
        grouped[r['Instrument']].append(r)
    proposed = []
    def nonempty(v):
        return v not in ('', '<NA>', 'None', 'nan', 'NaT')
    for ric in data['requested_rics']:
        rows = grouped[ric]
        symbols = {r[field] for r in rows for field in ('Ticker Symbol', 'Exchange Ticker')
                   if nonempty(r.get(field, ''))}
        ciks = {str(int(r['CIK Number'])) for r in rows if r.get('CIK Number','').isdigit()}
        isins = {r['ISIN'] for r in rows if nonempty(r.get('ISIN',''))}
        candidates = [r for r in tickers if normalize_symbol(r['symbol']) in
                      {normalize_symbol(s) for s in symbols}
                      and r['cik'] and str(int(r['cik'])) in ciks]
        issuer_ids = [r['ticker_id'] for r in tickers if r['cik'] and str(int(r['cik'])) in ciks]
        proposed.append({'ric':ric, 'symbols':sorted(symbols), 'ciks':sorted(ciks),
                         'isins':sorted(isins), 'candidate_ticker_ids':[r['ticker_id'] for r in candidates],
                         'issuer_only_review_ids': issuer_ids,
                         'retire_dates': sorted({r['RetireDate'][:10] for r in rows
                                                if nonempty(r.get('RetireDate', ''))}),
                         'first_trade_dates': sorted({r['First Trade Date'][:10] for r in rows
                                                     if nonempty(r.get('First Trade Date', ''))})})
    reverse = defaultdict(list)
    for r in proposed:
        if len(r['candidate_ticker_ids']) == 1:
            reverse[r['candidate_ticker_ids'][0]].append(r['ric'])
    for r in proposed:
        ids = r['candidate_ticker_ids']
        r['status'] = ('missing_or_mismatched' if not ids else
                       'ambiguous' if len(ids)!=1 or len(reverse[ids[0]])!=1 or len(r['isins'])!=1 else
                       'unique_candidate_requires_history_review')
        r['review_reason'] = (
            'conflicting_security_candidates' if r['status'] == 'ambiguous' else
            'historical_identity_not_certified' if ids else
            'source_issuer_missing' if not r['ciks'] else
            'issuer_absent_from_database' if not r['issuer_only_review_ids'] else
            'source_symbol_missing' if not r['symbols'] else
            'symbol_or_issuer_mismatch')
    return proposed


def price_history_conflicts(mappings, tickers):
    """Flag possible symbol reuse; retirement metadata alone cannot repair prices."""
    inventory = {r['ticker_id']: r for r in tickers}
    conflicts = []
    for mapping in mappings:
        if not mapping['retire_dates']:
            continue
        for ticker_id in mapping['candidate_ticker_ids']:
            ticker = inventory[ticker_id]
            last = ticker.get('last_price_date')
            if last is not None and str(last)[:10] > mapping['retire_dates'][-1]:
                conflicts.append(dict(ric=mapping['ric'], ticker_id=ticker_id,
                                      symbol=ticker['symbol'], last_price_date=str(last)[:10],
                                      retire_dates=mapping['retire_dates'],
                                      mapping_status=mapping['status'],
                                      status='requires_source_history_review'))
    return conflicts


async def run(archive, output):
    raw = Path(archive).read_bytes()
    data = json.loads(raw)['sources']['identity_archive']
    async with pool_context(max_size=1) as pool:
        async with pool.acquire() as conn, conn.transaction(isolation='repeatable_read', readonly=True):
            tickers = [dict(r) for r in await conn.fetch('''
                select t.ticker_id,t.symbol,t.cik,t.active,min(p.trade_date) first_price_date,
                       max(p.trade_date) last_price_date
                from tickers t left join price_history p using(ticker_id)
                where t.asset_type='equity' group by t.ticker_id''')]
            actions = [dict(r) for r in await conn.fetch('''select ticker_id,trade_date,split_factor
                from price_history where split_factor<>1 and share_split_factor is null
                order by ticker_id,trade_date''')]
    proposed = build_candidates(data, tickers)
    report = {'source_sha256':hashlib.sha256(raw).hexdigest(), 'counts':dict(Counter(r['status'] for r in proposed)),
              'review_counts':dict(Counter(r['review_reason'] for r in proposed)),
              'mappings':proposed, 'ticker_inventory': tickers, 'unclassified_actions':actions,
              'source_failures':data.get('failures', []),
              'price_history_conflicts':price_history_conflicts(proposed, tickers),
              'note':'Candidates are not approved dated mappings and have not been written to the database.'}
    write_json_new(output,report)
    print(report['counts'])


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    asyncio.run(run(args.archive,args.output))
