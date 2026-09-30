"""Read-only live source coverage audit; never certifies missing provenance."""
import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from backend.ingestion.db import pool_context
from backend.ml.research import write_json_new

QUERIES = {
    'tickers': '''select asset_type, active, count(*) n,
        count(cik) with_cik, count(ric) with_ric from tickers group by 1,2''',
    'shares': '''select shares_kind, shares_basis, count(*) n,
        count(shares_measured_at) dated, count(distinct ticker_id) securities
        from fundamentals group by 1,2''',
    'prices': '''select source, price_basis, count(*) n,
        count(*) filter (where split_factor is null or split_factor <= 0) invalid_splits,
        count(*) filter (where split_factor <> 1) split_events,
        min(trade_date) first_date, max(trade_date) last_date
        from price_history group by 1,2''',
    'share_actions': '''select count(*) filter (where split_factor <> 1) price_adjustments,
        count(*) filter (where share_split_factor is not null) classified,
        count(*) filter (where split_factor <> 1 and share_split_factor=1) pricing_only,
        count(*) filter (where split_factor <> 1 and share_split_factor is null) unclassified
        from price_history''',
    'evhc_terminal_basis': '''select t.ticker_id,t.cik,p.trade_date,p.close,p.adj_close,p.source,
        p.split_factor from tickers t join price_history p using(ticker_id)
        where t.symbol='EVHC' order by p.trade_date desc limit 3''',
    'membership': '''select index_id, source, count(*) n,
        count(*) filter (where valid_to is null) open_intervals
        from index_membership group by 1,2''',
    'membership_overlaps': '''select count(*) n from index_membership a
        join index_membership b on a.ticker_id=b.ticker_id and a.index_id=b.index_id
        and a.valid_from < b.valid_from and (a.valid_to is null or a.valid_to > b.valid_from)''',
    'accounting': '''select metric, count(*) n, count(distinct ticker_id) securities,
        min(filed_at) first_filing, max(filed_at) last_filing
        from accounting_facts group by 1''',
    'terminal_events': '''select event_type, verified, count(*) n from security_events group by 1,2''',
    'missing_terminal_records': '''select t.symbol, max(p.trade_date) last_trade
        from tickers t join price_history p using(ticker_id)
        where not t.active and t.asset_type='equity'
        and not exists (select 1 from security_events e where e.ticker_id=t.ticker_id)
        group by t.ticker_id having max(p.trade_date) < current_date - 365''',
}


async def replay(conn, checkpoint):
    from backend.ingestion.fundamentals import parse_companyfacts
    from backend.ingestion.research_fundamentals import parse_accounting_facts

    results = []
    # Fixed named cases: major splits, multiple classes, financials and exited names.
    tickers = await conn.fetch("""select ticker_id,symbol from tickers
        where symbol = any($1::text[]) order by ticker_id""",
        ['AAPL', 'NVDA', 'GE', 'GOOG', 'GOOGL', 'JPM', 'EVHC', 'COL'])
    for ticker in tickers:
        tid = ticker['ticker_id']
        marker = Path(checkpoint) / f'accounting-{tid}.json'
        if not marker.exists():
            results.append({'symbol': ticker['symbol'], 'status': 'not_backfilled'})
            continue
        meta = json.loads(marker.read_text())
        digest = meta['source_sha256']
        raw = (Path(checkpoint) / 'raw' / f'{digest}.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('Raw SEC source hash mismatch')
        payload = json.loads(raw)
        facts = parse_accounting_facts(payload, tid)
        stored = [dict(r) for r in await conn.fetch(
            'select * from accounting_facts where ticker_id=$1', tid)]
        key_fields = ('accession_number', 'metric', 'period_start', 'period_end', 'source_concept')
        indexed = {tuple(r[k] for k in key_fields): r for r in stored}
        for expected in facts:
            actual = indexed.get(tuple(expected[k] for k in key_fields))
            if actual is None or any(actual[k] != v for k, v in expected.items()):
                raise ValueError(f'Accounting source replay mismatch for {tid}')
        filings = parse_companyfacts(payload, tid)
        stored_filings = {r['accession_number']: dict(r) for r in await conn.fetch(
            'select * from fundamentals where ticker_id=$1', tid)}
        share_fields = ('shares_outstanding', 'shares_measured_at', 'shares_kind',
                        'shares_basis', 'shares_concept')
        for expected in filings:
            actual = stored_filings.get(expected['accession_number'])
            if actual is None or any(actual[k] != expected.get(k) for k in share_fields):
                raise ValueError(f'Share metadata replay mismatch for {tid}')
        results.append({'symbol': ticker['symbol'], 'status': 'passed',
                        'facts': len(facts), 'filings': len(filings), 'sha256': digest})
    return results


async def run(output, checkpoint=None):
    report = {'as_of': datetime.now(timezone.utc).isoformat(), 'checks': {}}
    async with pool_context(max_size=1, command_timeout=300) as pool:
        async with pool.acquire() as conn, conn.transaction(isolation='repeatable_read', readonly=True):
            for name, query in QUERIES.items():
                report['checks'][name] = [dict(r) for r in await conn.fetch(query)]
            if checkpoint:
                report['checks']['source_replay'] = await replay(conn, checkpoint)
            from backend.ml.dataset import load_frames
            from backend.ml.factors.pit import aligned_shares
            from datetime import date

            frames = await load_frames(conn, symbols=['GE'])
            dates = [date(2019,2,26), date(2021,8,2), date(2023,1,4), date(2024,4,2)]
            if len(frames) == 1:
                frame = frames[0]
                corrected = aligned_shares(frame.fundamentals, frame.prices, dates)
                legacy_prices = [{**p, 'share_split_factor': p['split_factor']} for p in frame.prices]
                legacy = aligned_shares(frame.fundamentals, legacy_prices, dates)
                report['checks']['ge_share_correction'] = [dict(date=d, before=a, after=b)
                                                          for d,a,b in zip(dates,legacy,corrected)]
    write_json_new(output, report)
    for name, rows in report['checks'].items():
        print(name, rows if name != 'missing_terminal_records' else {'securities': len(rows)}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    asyncio.run(run(args.output, args.checkpoint))
