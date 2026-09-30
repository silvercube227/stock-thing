"""Rehearse migration 017 in a rolled-back transaction, or apply it atomically."""
import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from backend.ingestion.db import pool_context


async def run(apply: bool, output: str, migration='017_signal_research.sql'):
    path = Path('backend/db/migrations') / migration
    sql = path.read_text()
    report = {'migration': path.name, 'sha256': hashlib.sha256(sql.encode()).hexdigest(),
              'started_at': datetime.now(timezone.utc).isoformat(), 'applied': False}
    async with pool_context(max_size=1, command_timeout=120) as pool:
        async with pool.acquire() as conn:
            tx = conn.transaction()
            await tx.start()
            try:
                await conn.execute("set local lock_timeout = '10s'")
                await conn.execute("select pg_advisory_xact_lock(170917)")
                # No CASCADE: dependent foreign keys fail safely instead of being removed.
                before = await conn.fetchval('select count(*) from index_membership')
                await conn.execute(sql)
                after = await conn.fetchval('select count(*) from index_membership')
                if before != after:
                    raise RuntimeError('Migration changed membership row count')
                report['membership_rows'] = after
                report['primary_key'] = await conn.fetchval("""
                    select pg_get_constraintdef(oid) from pg_constraint
                    where conrelid = 'index_membership'::regclass and contype = 'p'
                """)
                if report['primary_key'] != 'PRIMARY KEY (ticker_id, index_id, valid_from)':
                    raise RuntimeError('Unexpected membership primary key')
                report['tables'] = {}
                if migration == '019_share_action_basis.sql':
                    report['share_action_columns'] = await conn.fetchval("""
                        select count(*) from information_schema.columns
                        where table_schema='public' and table_name='price_history'
                        and column_name in ('share_split_factor','share_action_source')
                    """)
                    if report['share_action_columns'] != 2:
                        raise RuntimeError('Share adjustment metadata columns missing')
                for table in ('security_identifiers', 'security_events', 'accounting_facts',
                              'fixed_estimates', 'macro_vintages', 'research_news_daily',
                              'research_news_coverage'):
                    report['tables'][table] = await conn.fetchval(f'select count(*) from {table}')
                if migration == '018_signal_research_rls.sql':
                    report['rls'] = []
                    for table in report['tables']:
                        enabled = await conn.fetchval(
                            'select relrowsecurity from pg_class where oid=$1::regclass', table)
                        privileges = {}
                        for role in ('anon', 'authenticated'):
                            privileges[role] = await conn.fetchval(
                                "select has_table_privilege($1, $2, 'SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER')",
                                role, table)
                        if not enabled or any(privileges.values()):
                            raise RuntimeError(f'RLS/access verification failed for {table}')
                        report['rls'].append({'table': table, 'enabled': enabled,
                                              'api_role_privileges': privileges})
                if apply:
                    await tx.commit()
                    report['applied'] = True
                else:
                    await tx.rollback()
            except BaseException:
                await tx.rollback()
                raise
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--output', required=True)
    parser.add_argument('--migration', choices=('017_signal_research.sql', '018_signal_research_rls.sql', '019_share_action_basis.sql'),
                        default='017_signal_research.sql')
    args = parser.parse_args()
    asyncio.run(run(args.apply, args.output, args.migration))
