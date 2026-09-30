"""Resumable sequential archive of S&P 500 sector-index history (research only)."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from backend.config import get_settings
from backend.ingestion.estimates import _open_session
from backend.ml.research import write_json_new

# Candidate identifiers: returned index display names must be reviewed before use.
INDICES = ['.SPNY', '.SPLRCM', '.SPLRCI', '.SPLRCD', '.SPLRCS', '.SPXHC',
           '.SPSY', '.SPLRCT', '.SPLRCL', '.SPLRCU', '.SPLRCR']


def run(archive, output, requests=None):
    archive = Path(archive)
    archive.mkdir(parents=True, exist_ok=True)
    if requests is None:
        requests = default_requests()
    ld = _open_session(get_settings().lseg_app_key)
    entries = []
    try:
        for i, request in enumerate(requests):
            entry = archive_request(ld, archive, request)
            entries.append(entry)
            print(f'{i+1}/{len(requests)} {request["universe"]} {entry["rows"]}', flush=True)
    finally:
        ld.close_session()
    write_json_new(output, dict(status='archive_not_certified', entries=entries,
                                database_writes=0))


def default_requests():
    requests = [dict(universe=INDICES, fields=['TR.InstrumentDescription', 'TR.IndexName'], parameters={})]
    for index in INDICES:
        requests.append(dict(universe=index,
                             fields=['TR.IndexJLConstituentRIC', 'TR.IndexJLConstituentRIC.date',
                                     'TR.IndexJLConstituentRIC.change'],
                             parameters={'SDate': '2010-01-01', 'EDate': '2026-09-04', 'IC': 'B'}))
        for date in ['2010-01-04', '2018-08-31', '2018-10-01', '2026-09-04']:
            requests.append(dict(universe=index, fields=['TR.IndexConstituentRIC'],
                                 parameters={'SDate': date}))
    return requests


def archive_request(ld, archive, request):
    key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    marker = archive / (key + '.json')
    if marker.exists():
        entry = json.loads(marker.read_text())
        blob = gzip.decompress(Path(entry['artifact']).read_bytes())
        if hashlib.sha256(blob).hexdigest() != entry['sha256']:
            raise ValueError('sector archive changed')
        if json.loads(blob)['request'] != request:
            raise ValueError('sector archive request mismatch')
    else:
        # Failed requests do not become reusable cache entries.
        frame = ld.get_data(request['universe'], request['fields'], request['parameters'])
        payload = dict(request=request, columns=list(frame.columns),
                       records=frame.astype(str).to_dict('records'))
        blob = json.dumps(payload, sort_keys=True).encode()
        digest = hashlib.sha256(blob).hexdigest()
        artifact = archive / (digest + '.json.gz')
        if not artifact.exists():
            with gzip.open(artifact, 'xb') as stream:
                stream.write(blob)
        entry = dict(request=request, artifact=str(artifact), sha256=digest,
                     rows=len(frame), status='observed_requires_semantics_review')
        write_json_new(marker, entry)
    return entry

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', default='.research/sector-history-archive')
    parser.add_argument('--output', required=True)
    parser.add_argument('--requests', help='JSON array of explicit read-only LSEG requests')
    args = parser.parse_args()
    requests = json.loads(Path(args.requests).read_text()) if args.requests else None
    run(args.archive, args.output, requests)
