"""Quarantine source-disproved histories, with raw data retained and a rollback rehearsal."""

import argparse
import asyncio
import gzip
import hashlib
import json
from decimal import Decimal
from pathlib import Path

from backend.ingestion.db import pool_context
from backend.ml.dataset import load_frames
from backend.ml.research import write_json_new
from scripts.audit_original_security_prices import audit


def serialized(value):
    return json.loads(json.dumps(value, sort_keys=True,
                      default=lambda v: float(v) if isinstance(v, Decimal) else str(v)))


async def run(audit_path, output, apply=False, rehearsal=None):
    if Path(output).exists():
        raise ValueError('Receipt already exists')
    audit_raw = Path(audit_path).read_bytes()
    report = json.loads(audit_raw)
    digest = hashlib.sha256(audit_raw).hexdigest()
    for path, expected in report['input_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('Audit input changed')
    with gzip.open(report['stored_capture'], 'rb') as stream:
        capture_raw = stream.read()
    if hashlib.sha256(capture_raw).hexdigest() != report['stored_capture_sha256']:
        raise ValueError('Stored evidence capture changed')
    captured = json.loads(capture_raw)
    review_path = 'docs/original_security_price_review.json'
    archive_path = '.research/verification/original-security-native-prices.json'
    if review_path not in report['input_hashes'] or archive_path not in report['input_hashes']:
        raise ValueError('Missing reviewed native-price inputs')
    replay = audit(json.loads(Path(review_path).read_text()), json.loads(Path(archive_path).read_text()), captured)
    if replay != report['securities']:
        raise ValueError('Audit does not reproduce')
    selected = [r for r in replay if r['mismatched_dates'] and r['parity_dates'] == 0]
    if {r['symbol'] for r in selected} != {'COL', 'HAR'}:
        raise ValueError('Expected the two individually reviewed source-disproved histories')
    evidence = dict(reason='All overlapping native-security dates disagree; original-security price basis unverified',
                    artifact=audit_path, sha256=digest)
    migration = Path('backend/db/migrations/021_price_source_exclusions.sql').read_bytes()
    result = dict(applied=apply, audit_sha256=digest, migration_sha256=hashlib.sha256(migration).hexdigest(),
                  updates=[], raw_price_rows_deleted=0)
    async with pool_context(max_size=1) as pool:
        async with pool.acquire() as conn:
            tx = conn.transaction(isolation='repeatable_read')
            await tx.start()
            try:
                await conn.execute("set local lock_timeout = '10s'")
                await conn.execute(migration.decode())
                for r in selected:
                    tid = r['ticker_id']
                    ticker = await conn.fetchrow('select ticker_id,symbol,cik,security_retired_at,price_source_exclusions from tickers where ticker_id=$1 for update', tid)
                    prior = json.loads(ticker['price_source_exclusions'])
                    if prior:
                        raise ValueError('Existing exclusion requires explicit comparison')
                    expected = captured[str(tid)]
                    if serialized({k:ticker[k] for k in expected['ticker']}) != expected['ticker']:
                        raise ValueError('Ticker changed since source audit')
                    before = [dict(p) for p in await conn.fetch('select * from price_history where ticker_id=$1 order by trade_date', tid)]
                    if serialized(before) != expected['prices']:
                        raise ValueError('Price history changed since source audit')
                    if any(p['source'] != 'yfinance' for p in before):
                        raise ValueError('Mixed price history needs separate review')
                    await conn.execute('update tickers set price_source_exclusions=$2::jsonb where ticker_id=$1',
                                       tid, json.dumps({'yfinance':evidence}, sort_keys=True))
                    frames = await load_frames(conn, symbols=[r['symbol']])
                    if len(frames) != 1 or frames[0].prices or frames[0].price_source_exclusions != {'yfinance':evidence}:
                        raise ValueError('Excluded prices still reach model frames')
                    after = [dict(p) for p in await conn.fetch('select * from price_history where ticker_id=$1 order by trade_date', tid)]
                    if before != after:
                        raise ValueError('Quarantine modified raw price evidence')
                    result['updates'].append(dict(ticker_id=tid, symbol=r['symbol'], excluded_source='yfinance',
                        raw_rows_retained=len(before), previously_loaded_rows=r['retirement_filtered_rows'],
                        new_loaded_rows=0, prior_exclusions=prior, evidence=evidence))
                if apply:
                    if not rehearsal:
                        raise ValueError('Application requires a successful rollback rehearsal')
                    rehearsed = json.loads(Path(rehearsal).read_text())
                    if rehearsed != dict(result, applied=False):
                        raise ValueError('Application differs from rehearsal')
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', default='.research/verification/original-security-price-parity.json')
    p.add_argument('--output', required=True)
    p.add_argument('--rehearsal')
    p.add_argument('--apply', action='store_true')
    a = p.parse_args()
    asyncio.run(run(a.audit, a.output, a.apply, a.rehearsal))
