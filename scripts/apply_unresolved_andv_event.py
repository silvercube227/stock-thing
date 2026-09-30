"""Record Andeavor's documented acquisition while leaving its payoff unknown."""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path

from backend.ingestion.db import pool_context
from backend.ingestion.calendar import HORIZON_TRADING_DAYS, trading_days_between
from backend.ml.dataset import load_frames
from backend.ml.factors.pit import day, documented_targets
from backend.ml.research import write_json_new
from scripts.apply_successor_events import corroborate


async def run(output, apply=False):
    paths = dict(review=Path('docs/terminal_event_research.json'),
                 identity=Path('.research/verification/identity-lifecycle-archive.json'),
                 prices=Path('.research/verification/terminal-price-bases.json'))
    blobs = {k: p.read_bytes() for k, p in paths.items()}
    data = {k: json.loads(b) for k, b in blobs.items()}
    event = next(e for e in data['review']['events'] if e['symbol'] == 'ANDV')
    if event['cik'] != '50104' or event['effective_date'] != '2018-10-01':
        raise ValueError('Reviewed acquisition identity or date changed')
    if hashlib.sha256(Path(event['archived_source']).read_bytes()).hexdigest() != event['source_sha256']:
        raise ValueError('Completion filing changed')
    # Exact reviewed security, distinct from successor MPC and ticker-stem guesses.
    def records(value):
        if isinstance(value, dict):
            if value.get('RIC') == 'ANDV.N^J18':
                yield value
            else:
                for child in value.values():
                    yield from records(child)
        elif isinstance(value, list):
            for child in value:
                yield from records(child)
    identity = list(records(data['identity']))
    if len(identity) != 1 or identity[0]['ISIN'] != 'US03349M1053' or identity[0]['RetireDate'] != '2018-10-01':
        raise ValueError('Reviewed Andeavor security identity differs')
    provenance = dict(event_classification='documented_acquisition', payoff_status='unresolved_election_proration',
                      last_trade_date='2018-09-28', source=event['source'],
                      source_sha256=event['source_sha256'], ric='ANDV.N^J18', isin='US03349M1053',
                      input_hashes={k: hashlib.sha256(b).hexdigest() for k, b in blobs.items()})
    result = dict(applied=apply, provenance=provenance)
    async with pool_context(max_size=1) as pool:
        async with pool.acquire() as conn:
            tx = conn.transaction(isolation='repeatable_read')
            await tx.start()
            try:
                ticker = await conn.fetchrow('select ticker_id,cik from tickers where symbol=$1 for update', 'ANDV')
                if not ticker or int(ticker['cik']) != 50104:
                    raise ValueError('Database issuer does not match completion filing')
                tid = ticker['ticker_id']
                if await conn.fetchval('select exists(select 1 from security_events where ticker_id=$1)', tid):
                    raise ValueError('Existing security event requires explicit review')
                frame = (await load_frames(conn, symbols=['ANDV']))[0]
                probes = {r['ric']: r for r in data['prices']['sources']['terminal_prices']}
                result['price_corroboration'] = corroborate(frame.prices, 'ANDV.N^J18', day('2018-09-28'), probes)
                await conn.execute('''insert into security_events
                    (ticker_id,effective_date,event_type,consideration_type,source,verified)
                    values($1,$2,'acquisition','unresolved_election',$3,true)''',
                    tid, day('2018-10-01'), json.dumps(provenance, sort_keys=True))
                frame = (await load_frames(conn, symbols=['ANDV']))[0]
                observation_end = await conn.fetchval('select max(trade_date) from price_history')
                counts = Counter()
                dates = [day(r['trade_date']) for r in frame.prices]
                sessions = trading_days_between(dates[0], day(provenance['last_trade_date']))
                earliest_relevant = sessions[max(0, len(sessions)-max(HORIZON_TRADING_DAYS.values())-1)]
                for pos in range(len(frame.prices)-1):
                    # Earlier entries mature before the acquisition at every
                    # configured horizon, so cannot contribute exit exclusions.
                    if dates[pos+1] < earliest_relevant:
                        continue
                    if pos % 25 == 0:
                        # Calendar construction is CPU-heavy; keep the transaction
                        # responsive while checking the full exclusion census.
                        await conn.execute('select 1')
                    if pos % 250 == 0:
                        print(f'Andeavor label census {pos}/{len(frame.prices)-1}', flush=True)
                    labels = documented_targets(frame.prices, pos, frame.security_events, observation_end, dates)
                    for horizon, label in labels.items():
                        if label[4] in ('acquisition', 'entry_after_terminal'):
                            if label[1]:
                                raise ValueError('Unresolved consideration produced a valid label')
                            counts[(horizon, label[2].year, label[4])] += 1
                if not all(any(k[0] == h and k[2] == 'acquisition' for k in counts) for h in ['3M', '6M', '1Y']):
                    raise ValueError('Missing mature acquisition exclusions')
                result['excluded_labels'] = [dict(horizon=h, entry_year=y, exit_type=t, count=n)
                                             for (h,y,t),n in sorted(counts.items())]
                result['ticker_id'] = tid
                if apply:
                    rehearsal = json.loads(Path('.research/verification/andv-unresolved-rehearsal.json').read_text())
                    if (rehearsal['provenance'] != provenance or
                        rehearsal['excluded_labels'] != result['excluded_labels']):
                        raise ValueError('Application differs from the verified full-history rehearsal')
                    await tx.commit()
                else:
                    await tx.rollback()
            except BaseException:
                if not conn.is_closed():
                    await tx.rollback()
                raise
    write_json_new(output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    asyncio.run(run(args.output, args.apply))
