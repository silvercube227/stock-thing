"""Archive/replay actual macro versions; rehearse before replacing legacy rows.

download --directory DIR creates an immutable candidate without database writes.
apply --directory DIR --output RECEIPT rolls back unless --commit is supplied
with the exact successful --rehearsal receipt. Legacy rows are archived first;
noncolliding legacy rows remain stored but unverified.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import date
import gzip
import hashlib
import json
from pathlib import Path

import httpx

from backend.config import get_settings
from backend.ingestion.db import pool_context
from backend.ingestion.macro import SERIES, VERIFIED_SOURCE, fetch_vintages
from backend.ml.research import json_value, write_json_new

COLUMNS = ('series_id', 'obs_date', 'available_from', 'vintage_date', 'value', 'source', 'verified')
KEY = ('series_id', 'obs_date', 'vintage_date')
LEGACY_SOURCES = {'alfred', 'alfred_vintage', 'fred_pre_archive'}


def canonical(value):
    return json.dumps(json_value(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


class ArchiveClient:
    """Replay only the exact archived request parameters, with hash verification."""

    def __init__(self, root, names):
        self.responses = {}
        for name in names:
            if Path(name).name != name:
                raise ValueError('Archive entry must be a basename')
            raw = gzip.decompress((root / name).read_bytes())
            if name != hashlib.sha256(raw).hexdigest()+'.json.gz':
                raise ValueError('Macro source archive hash mismatch')
            record = json.loads(raw)
            key = canonical([record['endpoint'], record['parameters']])
            if key in self.responses:
                raise ValueError('Ambiguous archived request')
            self.responses[key] = record['response']

    async def get(self, url, params):
        params = {k:v for k,v in params.items() if k not in ('api_key', 'file_type')}
        key = canonical([url.split('/fred/')[1], params])
        if key not in self.responses:
            raise ValueError('Missing archived request')
        return httpx.Response(200, json=self.responses[key])


async def replay(directory):
    directory = Path(directory)
    raw = (directory / 'candidate.json').read_bytes()
    candidate = json.loads(raw)
    client = ArchiveClient(directory / 'raw', candidate['archives'])
    rows = []
    for series in SERIES:
        rows.extend(await fetch_vintages(client, series, 'archive-replay',
            as_of=date.fromisoformat(candidate['as_of']),
            observation_start=candidate['observation_start']))
    if json_value(rows) != candidate['rows']:
        raise ValueError('Macro candidate does not reproduce from source archives')
    return rows, hashlib.sha256(raw).hexdigest()


async def download(directory, as_of):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    rows = []
    async with httpx.AsyncClient(timeout=60) as client:
        for series in SERIES:
            batch = await fetch_vintages(client, series, get_settings().fred_api_key,
                as_of=as_of, archive_dir=directory / 'raw')
            rows.extend(batch)
            print(series, len(batch), 'actual value versions archived', flush=True)
    write_json_new(directory / 'candidate.json', dict(as_of=str(as_of),
        observation_start='2000-01-01', archives=sorted(p.name for p in (directory / 'raw').iterdir()),
        rows=rows))
    replayed, digest = await replay(directory)
    summary = dict(candidate_sha256=digest, source_replay=True, applied=False, series={})
    for series in SERIES:
        batch = [r for r in replayed if r['series_id'] == series]
        grouped = {}
        for row in batch:
            grouped.setdefault(str(row['obs_date']), []).append(row)
        revised = {k:v for k,v in grouped.items() if len(v) > 1}
        summary['series'][series] = dict(rows=len(batch), observations=len(grouped),
            first_available_from=min(r['available_from'] for r in batch),
            last_vintage=max(r['vintage_date'] for r in batch), revised_observations=len(revised),
            revision_examples=dict(list(revised.items())[:5]))
    write_json_new(directory / 'summary.json', summary)
    print(json.dumps(json_value(summary), indent=2))


def validate_existing(existing, rows):
    incoming = {tuple(r[k] for k in KEY):r for r in rows}
    if len(incoming) != len(rows):
        raise ValueError('Duplicate candidate keys')
    for row in existing:
        if row['source'] not in LEGACY_SOURCES | {VERIFIED_SOURCE}:
            raise ValueError('Unreviewed existing macro source')
        if row['source'] == VERIFIED_SOURCE:
            if incoming.get(tuple(row[k] for k in KEY)) != row:
                raise ValueError('Existing actual vintage differs from candidate; refusing overwrite')


async def apply(directory, output, commit=False, rehearsal=None):
    if Path(output).exists():
        raise ValueError('Receipt already exists')
    rows, digest = await replay(directory)
    result = dict(applied=commit, candidate_sha256=digest,
        database_sha256=hashlib.sha256(get_settings().database_url.encode()).hexdigest(),
        candidate_rows=len(rows))
    async with pool_context(max_size=1, command_timeout=300) as pool, pool.acquire() as conn:
        tx = conn.transaction(isolation='repeatable_read')
        await tx.start()
        try:
            await conn.execute("set local lock_timeout = '10s'")
            await conn.execute('lock table macro_vintages in share row exclusive mode')
            query = f"select {','.join(COLUMNS)} from macro_vintages where series_id=any($1::text[]) order by series_id,obs_date,vintage_date"
            existing = [dict(r) for r in await conn.fetch(query, list(SERIES))]
            validate_existing(existing, rows)
            preimage = canonical(existing)
            before_hash = hashlib.sha256(preimage).hexdigest()
            archive = Path(directory) / (before_hash+'.preimage.json.gz')
            if not archive.exists():
                with archive.open('xb') as out, gzip.GzipFile(fileobj=out, mode='wb', mtime=0) as stream:
                    stream.write(preimage)
            if gzip.decompress(archive.read_bytes()) != preimage:
                raise ValueError('Preimage archive verification failed')
            result.update(preimage_sha256=before_hash, preimage_archive=str(archive),
                          legacy_rows=sum(r['source'] != VERIFIED_SOURCE for r in existing))
            await conn.execute('update macro_vintages set verified=false where series_id=any($1::text[]) and source<>$2', list(SERIES), VERIFIED_SOURCE)
            await conn.executemany(
                f"insert into macro_vintages ({','.join(COLUMNS)}) values ($1,$2,$3,$4,$5,$6,$7) "
                "on conflict (series_id,obs_date,vintage_date) do update set "
                "available_from=excluded.available_from,value=excluded.value,source=excluded.source,verified=excluded.verified",
                [tuple(r[k] for k in COLUMNS) for r in rows])
            after = [dict(r) for r in await conn.fetch(query, list(SERIES))]
            actual = [r for r in after if r['source'] == VERIFIED_SOURCE]
            sort_key = lambda r: tuple(r[k] for k in KEY)
            if sorted(actual, key=sort_key) != sorted(rows, key=sort_key):
                raise ValueError('Macro database readback mismatch')
            if any(r['verified'] for r in after if r['source'] != VERIFIED_SOURCE):
                raise ValueError('Legacy macro remains verified')
            actual_keys = {sort_key(r) for r in rows}
            retained = [r for r in after if r['source'] != VERIFIED_SOURCE]
            expected = [dict(r, verified=False) for r in existing if sort_key(r) not in actual_keys]
            if retained != expected:
                raise ValueError('Noncolliding legacy evidence changed')
            result.update(stored_rows=len(after), retained_unverified_rows=len(retained),
                          readback_verified=True)
            if commit:
                if not rehearsal or json.loads(Path(rehearsal).read_text()) != dict(result, applied=False):
                    raise ValueError('Application requires an exact successful rollback rehearsal')
                await tx.commit()
            else:
                await tx.rollback()
        except BaseException:
            if not conn.is_closed():
                await tx.rollback()
            raise
    write_json_new(output, result)
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    d = sub.add_parser('download')
    d.add_argument('--directory', required=True)
    d.add_argument('--as-of', type=date.fromisoformat, default=date.today())
    a = sub.add_parser('apply')
    a.add_argument('--directory', required=True)
    a.add_argument('--output', required=True)
    a.add_argument('--commit', action='store_true')
    a.add_argument('--rehearsal')
    args = p.parse_args()
    if args.command == 'download':
        asyncio.run(download(args.directory, args.as_of))
    else:
        asyncio.run(apply(args.directory, args.output, args.commit, args.rehearsal))


if __name__ == '__main__':
    main()
