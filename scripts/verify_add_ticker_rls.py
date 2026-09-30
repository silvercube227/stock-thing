"""Exercise actual add-ticker DB helpers with RLS; roll back all test rows and DDL.

This tests database integration, not external feeds or the trained model.
PostgreSQL sequence allocations are not rolled back.
"""
import argparse
import asyncio
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import asyncpg

from backend.ingestion.db import pool_context
from backend.ingestion.fundamentals import _upsert_filings
from backend.jobs.add_ticker import _upsert_ticker, _finish
from backend.ml.dataset import load_frames
from backend.ml.research import write_json_new


async def run(output):
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'rolled_back': True}
    async with pool_context(max_size=1, command_timeout=120) as pool:
        async with pool.acquire() as conn:
            tx = conn.transaction()
            await tx.start()
            try:
                await conn.execute("set local lock_timeout='10s'")
                await conn.execute(Path('backend/db/migrations/018_signal_research_rls.sql').read_text())
                report['backend_role'] = await conn.fetchval('select current_user')
                symbol = 'RLS' + uuid4().hex[:8].upper()
                metadata = dict(name='Rollback RLS verification', asset_type='equity',
                                sector=None, industry=None, cik=None, shares_outstanding=None)
                tid = await _upsert_ticker(conn, symbol, metadata)
                assert await _upsert_ticker(conn, symbol, metadata) == tid
                run_id = await conn.fetchval("insert into ingestion_runs(job_name) values($1) returning run_id",
                                            f'add_ticker:{symbol}')
                filing = dict(ticker_id=tid, accession_number='rls-verification', filing_type='10-Q',
                              period_end=date(2020,3,31), filed_at=datetime(2020,5,1,tzinfo=timezone.utc),
                              revenue=100, net_income=10, gross_margin=.3, operating_margin=.1,
                              total_debt=0, total_equity=50, fcf=10, shares_outstanding=10,
                              shares_measured_at=date(2020,4,20), shares_kind='point_in_time',
                              shares_basis='as_reported', shares_concept='EntityCommonStockSharesOutstanding')
                await _upsert_filings(conn, [filing])
                from backend.ingestion.prices import _UPSERT_SQL as PRICE_UPSERT
                bar = (tid,date(2020,5,1),10,10,10,10,10,100,2,0,'yfinance')
                await conn.execute(PRICE_UPSERT,*bar)
                await conn.execute('''update price_history set share_split_factor=2,
                    share_action_source='rollback-test' where ticker_id=$1''',tid)
                await conn.execute(PRICE_UPSERT,*bar)
                assert await conn.fetchval('select share_split_factor from price_history where ticker_id=$1',tid)==2
                changed = (*bar[:8],3,*bar[9:])
                await conn.execute(PRICE_UPSERT,*changed)
                assert await conn.fetchval('select share_split_factor from price_history where ticker_id=$1',tid) is None
                report['price_upsert_and_action_invalidation']='passed'
                await conn.execute("""insert into accounting_facts
                    (ticker_id,accession_number,filed_at,period_end,metric,value,source_concept)
                    values($1,'rls-verification','2020-05-01','2020-03-31','assets',100,'Assets')""", tid)
                frames = await load_frames(conn, symbols=[symbol])
                assert len(frames) == 1 and len(frames[0].accounting_facts) == 1
                await _finish(conn, run_id, 'success', rows=1)
                assert await conn.fetchval('select status from ingestion_runs where run_id=$1', run_id) == 'success'
                report['backend_insert_update_load_status'] = 'passed'
                report['api_roles'] = {}
                for role in ('anon', 'authenticated'):
                    for operation in ('select count(*) from accounting_facts',
                                      'delete from accounting_facts where false'):
                        try:
                            async with conn.transaction():
                                await conn.execute(f'set local role {role}')
                                await conn.execute(operation)
                        except asyncpg.InsufficientPrivilegeError:
                            pass
                        else:
                            raise AssertionError(f'{role} unexpectedly allowed: {operation}')
                    report['api_roles'][role] = 'read_and_write_denied'
            finally:
                await tx.rollback()
            assert not await conn.fetchval('select exists(select 1 from tickers where symbol=$1)', symbol)
    write_json_new(output, report)
    print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    asyncio.run(run(parser.parse_args().output))
