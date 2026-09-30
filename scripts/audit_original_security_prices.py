"""Compare original-security native prices with stored history, including pre-retirement bars.

The compressed database capture permits exact offline replay. Price disagreements
are not automatically treated as currency conversions or corporate actions.
"""

import argparse
import asyncio
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path

from backend.ingestion.db import pool_context
from backend.ml.dataset import _drop_reused_symbol_bars
from backend.ml.factors.pit import as_traded_closes, day
from backend.ml.research import write_json_new
from scripts.audit_vendor_adjustments import load_rows


def compare(stored, native, retired_at):
    prices = _drop_reused_symbol_bars(
        [dict(p, trade_date=day(p['trade_date'])) for p in stored], None, retired_at)
    nominal = dict(zip((str(p['trade_date']) for p in prices), as_traded_closes(prices)))
    original = {}
    for row in native:
        d = row['Date'][:10]
        if d in original:
            raise ValueError('Duplicate native date')
        try:
            value = float(row['TRDPRC_1'])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0:
            original[d] = value
    if not original:
        raise ValueError('No valid native prices')
    shared = sorted(nominal.keys() & original.keys())
    mismatches = []
    for d in shared:
        if not math.isfinite(nominal[d]) or not math.isclose(nominal[d], original[d], rel_tol=1e-5):
            mismatches.append(dict(date=d, stored_as_traded=nominal[d], native_as_traded=original[d],
                                   ratio=nominal[d] / original[d]))
    return dict(stored_rows=len(stored), retirement_filtered_rows=len(prices),
                native_rows=len(original), native_first=min(original), native_last=max(original),
                matched_dates=len(shared), parity_dates=len(shared)-len(mismatches),
                mismatched_dates=len(mismatches),
                mismatches_by_year=dict(Counter(r['date'][:4] for r in mismatches)),
                mismatch_examples=mismatches[:3]+mismatches[-3:] if len(mismatches)>6 else mismatches,
                missing_native_dates=sorted(original.keys()-nominal.keys()),
                stored_only_dates=sorted(nominal.keys()-original.keys()),
                status='price_disagreement_requires_quarantine_or_repair' if mismatches
                else 'no_original_history' if not shared else 'sampled_price_parity_not_certification')


def audit(review, archive, capture):
    identities = json.loads(Path(review['identity_evidence']).read_text())['sources']['identity_archive']['records']
    histories = {r['ric']: r for r in archive['securities']}
    results = []
    for security in review['securities']:
        matches = [r for r in identities if r.get('RIC') == security['ric']]
        if not matches or any(r['ISIN'] != security['isin'] or int(r['CIK Number']) != int(security['cik'])
                              for r in matches):
            raise ValueError('Original-security identity disagreement')
        stored = capture[str(security['ticker_id'])]
        ticker = stored['ticker']
        if ticker['symbol'] != security['symbol'] or int(ticker['cik']) != int(security['cik']):
            raise ValueError('Database identity changed')
        entry = histories[security['ric']]
        if entry['request']['adjustments'] != 'unadjusted':
            raise ValueError('Native history must be unadjusted')
        native = load_rows(entry)['records']
        retirement = day(ticker['security_retired_at']) if ticker['security_retired_at'] else None
        results.append(dict(symbol=security['symbol'], ticker_id=security['ticker_id'], ric=security['ric'],
                            **compare(stored['prices'], native, retirement)))
    return results


async def run(review_path, archive_path, output, replay=None):
    output = Path(output)
    if output.exists():
        raise ValueError('Audit output already exists')
    review_raw, archive_raw = Path(review_path).read_bytes(), Path(archive_path).read_bytes()
    review, archive = json.loads(review_raw), json.loads(archive_raw)
    if archive['inventory_sha256'] != hashlib.sha256(review_raw).hexdigest():
        raise ValueError('Native history belongs to a different review')
    if replay:
        with gzip.open(replay, 'rb') as stream:
            capture_raw = stream.read()
    else:
        capture = {}
        async with pool_context(max_size=1) as pool:
            async with pool.acquire() as conn, conn.transaction(isolation='repeatable_read', readonly=True):
                for security in review['securities']:
                    tid = security['ticker_id']
                    ticker = await conn.fetchrow('select ticker_id,symbol,cik,security_retired_at from tickers where ticker_id=$1', tid)
                    prices = await conn.fetch('select * from price_history where ticker_id=$1 order by trade_date', tid)
                    capture[str(tid)] = dict(ticker=dict(ticker), prices=[dict(p) for p in prices])
        # Numeric values remain numeric in the replay; dates retain ISO format.
        from decimal import Decimal
        capture_raw = json.dumps(capture, sort_keys=True, default=lambda v: float(v) if isinstance(v, Decimal) else str(v)).encode()
    digest = hashlib.sha256(capture_raw).hexdigest()
    capture_path = Path('.research/original-security-price-archive') / f'{digest}.stored.json.gz'
    if not capture_path.exists():
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        with capture_path.open('xb') as dest, gzip.GzipFile(fileobj=dest, mode='wb', mtime=0) as stream:
            stream.write(capture_raw)
    results = audit(review, archive, json.loads(capture_raw))
    report = dict(securities=results, database_writes=0, stored_capture=str(capture_path),
                  stored_capture_sha256=digest,
                  input_hashes={review_path:hashlib.sha256(review_raw).hexdigest(),
                                archive_path:hashlib.sha256(archive_raw).hexdigest(),
                                review['identity_evidence']:hashlib.sha256(Path(review['identity_evidence']).read_bytes()).hexdigest()},
                  status='not_certified', limitations=[
                      'Retirement truncation alone cannot repair wrong-security prices before retirement.',
                      'Price parity does not certify distribution completeness or terminal consideration.'])
    write_json_new(output, report)
    for r in results:
        print({k:r[k] for k in ('symbol','native_rows','matched_dates','parity_dates','mismatched_dates','status')})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review', default='docs/original_security_price_review.json')
    p.add_argument('--archive', default='.research/verification/original-security-native-prices.json')
    p.add_argument('--output', required=True)
    p.add_argument('--replay', help='Replay a compressed stored-price capture without database access')
    a = p.parse_args()
    asyncio.run(run(a.review, a.archive, a.output, a.replay))
