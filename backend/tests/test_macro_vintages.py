import asyncio
from datetime import date
import gzip
import hashlib
import json

import httpx
import pytest

from backend.ingestion import macro


def obs(day, vintage, value):
    return dict(date=day, realtime_start=vintage, value=value)


def test_actual_versions_pagination_year_boundary_and_withdrawal(monkeypatch):
    calls = []

    async def fake_get(client, path, key, **params):
        calls.append((path, params))
        if path.endswith('vintagedates'):
            return dict(count=1, vintage_dates=['2019-12-30'])
        assert params['output_type'] == 1
        if params['realtime_start'] == '2019-12-30':
            assert params['realtime_end'] == '2019-12-31'
            rows = [obs('2010-01-04', '2019-12-30', '3.12'),
                    obs('2010-01-04', '2019-12-31', '3.14')]
            return dict(count=2, observations=rows[params['offset']:params['offset']+1])
        assert params['realtime_start'] == '2020-01-01'
        assert params['realtime_end'] == '2020-01-03'
        return dict(count=3, observations=[obs('2010-01-04', '2020-01-01', '3.14'),
                    obs('2010-01-04', '2020-01-02', '.'),
                    obs('2010-01-04', '2020-01-03', '3.12')])

    monkeypatch.setattr(macro, '_get', fake_get)
    rows = asyncio.run(macro.fetch_vintages(None, 'BAA10Y', 'secret', as_of=date(2020, 1, 3)))
    assert [r['value'] for r in rows] == [3.12, 3.14, None, 3.12]
    assert rows[0]['available_from'] == date(2019, 12, 31)
    assert rows[-1]['available_from'] == date(2020, 1, 4)
    assert all(r['source'] == macro.VERIFIED_SOURCE and r['verified'] for r in rows)
    assert len(calls) == 4


@pytest.mark.parametrize('failure', ['empty', 'count_change', 'outside', 'conflict'])
def test_bad_observation_pages_fail_closed(monkeypatch, failure):
    async def fake_get(client, path, key, **params):
        if path.endswith('vintagedates'):
            return dict(count=1, vintage_dates=['2020-01-01'])
        if failure == 'empty':
            return dict(count=1, observations=[])
        if failure == 'outside':
            return dict(count=1, observations=[obs('2010-01-04', '2019-12-31', '1')])
        if failure == 'conflict':
            return dict(count=2, observations=[obs('2010-01-04', '2020-01-01', '1'),
                                               obs('2010-01-04', '2020-01-01', '2')])
        return dict(count=2 if params['offset'] == 0 else 3,
                    observations=[obs('2010-01-04', '2020-01-01', '1')])
    monkeypatch.setattr(macro, '_get', fake_get)
    with pytest.raises(ValueError):
        asyncio.run(macro.fetch_vintages(None, 'BAA10Y', 'secret', as_of=date(2020, 1, 3)))


@pytest.mark.parametrize('value', ['nan', 'inf', '-inf'])
def test_nonfinite_values_refused(value):
    with pytest.raises(ValueError, match='nonfinite'):
        macro.parse_alfred({'observations': [obs('2020-01-01', '2020-01-02', value)]}, 'DGS10')


def test_archive_is_content_addressed_and_has_no_credentials(tmp_path):
    async def run():
        transport = httpx.MockTransport(lambda r: httpx.Response(200, json={'count': 0}))
        async with httpx.AsyncClient(transport=transport) as client:
            for _ in range(2):
                await macro._get(client, 'series/observations', 'secret',
                                 archive_dir=tmp_path, series_id='DGS10')
    asyncio.run(run())
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    raw = gzip.decompress(files[0].read_bytes())
    assert b'secret' not in raw and b'api_key' not in raw
    assert files[0].name == hashlib.sha256(raw).hexdigest()+'.json.gz'
    assert json.loads(raw)['parameters'] == {'series_id': 'DGS10'}


def test_http_errors_do_not_expose_key():
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda r: httpx.Response(400, json={'error': 'bad'}))) as client:
            await macro._get(client, 'series/observations', 'secret')
    with pytest.raises(ValueError) as exc:
        asyncio.run(run())
    assert 'secret' not in str(exc.value)


@pytest.mark.parametrize('dates,count', [([], 1), (['2020-01-01'] * 2, 2), (['2021-01-01'], 1)])
def test_invalid_vintage_calendar_rejected(monkeypatch, dates, count):
    async def fake_get(*args, **kwargs):
        return dict(count=count, vintage_dates=dates)
    monkeypatch.setattr(macro, '_get', fake_get)
    with pytest.raises(ValueError):
        asyncio.run(macro.fetch_vintage_dates(None, 'DGS10', 'secret', as_of=date(2020, 1, 3)))


def test_rebuild_refuses_changed_corrected_rows_and_unknown_sources():
    from scripts.rebuild_macro_vintages import validate_existing
    row = macro.parse_alfred({'observations': [obs('2020-01-01', '2020-01-02', '2')]}, 'DGS10')[0]
    validate_existing([row], [row])
    validate_existing([dict(row, source='alfred_vintage', value=99)], [row])
    with pytest.raises(ValueError, match='refusing overwrite'):
        validate_existing([dict(row, value=99)], [row])
    with pytest.raises(ValueError, match='Unreviewed'):
        validate_existing([dict(row, source='some_new_source')], [row])
    with pytest.raises(ValueError, match='Duplicate'):
        validate_existing([], [row, row])


def test_archive_replay_requires_hash_and_exact_parameters(tmp_path):
    from scripts.rebuild_macro_vintages import ArchiveClient, canonical
    raw = canonical(dict(endpoint='series/observations', parameters={'series_id': 'DGS10'},
                         response={'observations': []}))
    name = hashlib.sha256(raw).hexdigest()+'.json.gz'
    path = tmp_path / name
    path.write_bytes(gzip.compress(raw))
    client = ArchiveClient(tmp_path, [name])
    response = asyncio.run(client.get(macro.FRED+'/fred/series/observations',
                                      dict(series_id='DGS10', api_key='ignored', file_type='json')))
    assert response.json() == {'observations': []}
    with pytest.raises(ValueError, match='Missing archived request'):
        asyncio.run(client.get(macro.FRED+'/fred/series/observations', dict(series_id='BAA10Y')))
    path.write_bytes(gzip.compress(b'{}'))
    with pytest.raises(ValueError, match='hash mismatch'):
        ArchiveClient(tmp_path, [name])


def test_legacy_backfill_route_refuses_before_database_access():
    from types import SimpleNamespace
    from scripts.backfill_signal_data import backfill
    with pytest.raises(ValueError, match='rollback rehearsal'):
        asyncio.run(backfill(SimpleNamespace(source='macro')))
