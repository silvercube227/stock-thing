"""Validate source-corroborated successor deals live, then optionally apply atomically."""
import argparse
import asyncio
from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path

from backend.ingestion.calendar import HORIZON_TRADING_DAYS, shift_trading_days
from backend.ingestion.db import pool_context
from backend.ml.dataset import load_frames
from backend.ml.factors.pit import as_traded_closes, day, documented_targets
from backend.ml.research import write_json_new
from backend.ml.terminal import successor_horizon_values


# Explicit source-reviewed security pairs; no issuer-only resolution.
PAIRS = {'SCG': ('SCG.N^A19', 'D.N', '754737', '715957'),
         'ESRX': ('ESRX.OQ^L18', 'CI.N', '1532063', '1739940'),
         'AET': ('AET.N^K18', 'CVS.N', '1122304', '64803'),
         'TWX': ('TWX.N^F18', 'T.N', '1105705', '732717'),
         'BMS': ('BMS.N^F19', 'AMCR.N', '11199', '1748790')}


def corroborate(prices, ric, session, probes):
    matches = [r for r in probes[ric].get('records', []) if r['Date'] == str(session)]
    if len(matches) != 1:
        raise ValueError(f'No independent exact-session price for {ric} {session}')
    indices = [i for i,p in enumerate(prices) if day(p['trade_date']) == session]
    if len(indices) != 1:
        raise ValueError('Missing exact stored price session')
    actual = as_traded_closes(prices)[indices[0]]
    expected = float(matches[0]['close_adjusted_0'] if 'close_adjusted_0' in matches[0]
                     else matches[0]['TRDPRC_1'])
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-6):
        raise ValueError(f'Nominal price disagreement: {ric} {session}')
    return dict(ric=ric, date=str(session), stored_as_traded=actual, lseg_as_traded=expected)


async def run(symbols, reviews, probe_path, output, apply, pricing_service_probe=None):
    raw, probe_raw = Path(reviews).read_bytes(), Path(probe_path).read_bytes()
    events = {e['symbol']:e for e in json.loads(raw)['events']}
    probes = {p['ric']:p for p in json.loads(probe_raw)['sources']['terminal_prices']}
    result = dict(applied=apply, events=[], review_sha256=hashlib.sha256(raw).hexdigest(),
                  probe_sha256=hashlib.sha256(probe_raw).hexdigest())
    if pricing_service_probe:
        pricing_raw=Path(pricing_service_probe).read_bytes()
        result['pricing_service_probe_sha256']=hashlib.sha256(pricing_raw).hexdigest()
        for sample in json.loads(pricing_raw)['sources']['historical_pricing_service']:
            if sample.get('records') and not probes.get(sample['ric'],{}).get('records'):
                probes[sample['ric']]=sample
    async with pool_context(max_size=1) as pool:
        async with pool.acquire() as conn, conn.transaction(isolation='repeatable_read'):
            observation_end = await conn.fetchval('select max(trade_date) from price_history')
            for symbol in symbols:
                review = events[symbol]
                if hashlib.sha256(Path(review['archived_source']).read_bytes()).hexdigest() != review['source_sha256']:
                    raise ValueError('Archived completion source changed')
                target_ric, successor_ric, target_cik, successor_cik = PAIRS[symbol]
                successor_symbol = review['terms']['successor_symbol']
                tickers = await conn.fetch('select ticker_id,symbol,cik from tickers where symbol=any($1) for update',
                                           [symbol, successor_symbol])
                by_symbol = {r['symbol']:r for r in tickers}
                for sym,cik in ((symbol,target_cik),(successor_symbol,successor_cik)):
                    if sym not in by_symbol or int(by_symbol[sym]['cik']) != int(cik):
                        raise ValueError(f'Database issuer mismatch: {sym}')
                frames = {f.symbol:f for f in await load_frames(conn,symbols=[symbol,successor_symbol])}
                final, anchor = day(review['last_trade_date']), day(review['ownership_close'])
                target = [p for p in frames[symbol].prices if day(p['trade_date']) <= final]
                successor = frames[successor_symbol].prices
                checks = [corroborate(target,target_ric,final,probes),
                          corroborate(successor,successor_ric,anchor,probes)]
                ends = set()
                for p in target:
                    entry = day(p['trade_date'])
                    if entry < final - timedelta(days=400):
                        continue
                    for steps in HORIZON_TRADING_DAYS.values():
                        end = shift_trading_days(entry,steps)
                        if final < end <= observation_end:
                            ends.add(end)
                valuation = successor_horizon_values(target_prices=target, successor_prices=successor,
                    target_last_trade=final, ownership_close=anchor, horizon_dates=ends,
                    cash_per_share=review['terms']['cash_per_share'], exchange_ratio=review['terms']['exchange_ratio'])
                if valuation['missing_horizons']:
                    raise ValueError(f'Incomplete successor horizon prices: {symbol}')
                provenance = {**review, 'review_sha256':result['review_sha256'],
                              'probe_sha256':result['probe_sha256'], 'price_checks':checks,
                              'pricing_service_probe_sha256':result.get('pricing_service_probe_sha256'),
                              'target_prices_sha256':hashlib.sha256(json.dumps(target,sort_keys=True,default=str).encode()).hexdigest(),
                              'successor_prices_sha256':hashlib.sha256(json.dumps(successor,sort_keys=True,default=str).encode()).hexdigest(),
                              'valuation_policy':valuation['stock_distribution_policy']}
                source = json.dumps(provenance,sort_keys=True)
                event = dict(effective_date=review['effective_date'], event_type='acquisition',
                             consideration_type='successor', proceeds_basis='adj_close', source=source,
                             verified=True, horizon_values=valuation['horizon_values'])
                labels = documented_targets(target,len(target)-2,[event],observation_end)
                if not all(v[1] and v[4]=='acquisition' for v in labels.values()):
                    raise ValueError('Mature terminal label integration failed')
                tid = by_symbol[symbol]['ticker_id']
                previous = await conn.fetchrow('select * from security_events where ticker_id=$1 and effective_date=$2',
                                               tid,day(review['effective_date']))
                if previous:
                    raise ValueError('Existing event requires explicit comparison; never overwrite')
                if apply:
                    await conn.execute('''insert into security_events
                        (ticker_id,effective_date,event_type,consideration_type,proceeds_basis,horizon_values,source,verified)
                        values($1,$2,'acquisition','successor','adj_close',$3::jsonb,$4,true)''',
                        tid,day(review['effective_date']),json.dumps(valuation['horizon_values']),source)
                    stored = dict(await conn.fetchrow('select * from security_events where ticker_id=$1 and effective_date=$2',
                                                      tid,day(review['effective_date'])))
                    replay = documented_targets(target,len(target)-2,[stored],observation_end)
                    if replay != labels:
                        raise ValueError('Stored event failed exact label replay; transaction rolled back')
                result['events'].append(dict(symbol=symbol,ticker_id=tid, price_checks=checks,
                    horizon_count=len(valuation['horizon_values']), labels=labels,
                    valuation=valuation, source=provenance))
    write_json_new(output,result)
    print({'applied':apply,'events':[{k:e[k] for k in ('symbol','horizon_count')} for e in result['events']]})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbols',nargs='+',choices=tuple(PAIRS),required=True)
    parser.add_argument('--reviews',default='docs/terminal_event_research.json')
    parser.add_argument('--price-probe',default='.research/verification/terminal-price-bases.json')
    parser.add_argument('--pricing-service-probe')
    parser.add_argument('--output',required=True)
    parser.add_argument('--apply',action='store_true')
    a=parser.parse_args()
    asyncio.run(run(a.symbols,a.reviews,a.price_probe,a.output,a.apply,a.pricing_service_probe))
