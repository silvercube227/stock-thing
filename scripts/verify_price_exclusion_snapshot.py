"""Verify a post-exclusion audit snapshot and its production frame cache offline."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle

from backend.config import get_settings
from backend.ml.compressed_io import open_artifact
from backend.ml.dataset import FRAME_CACHE_VERSION
from backend.ml.research import load_snapshot, write_json_new
from scripts.signal_research import source_ready


def run(snapshot, output):
    inputs, manifest = load_snapshot(snapshot)
    frames = inputs['frames']
    cache_path = get_settings().frame_cache_dir/'frames_all.pkl'
    with open_artifact(cache_path) as stream:
        cache = pickle.load(stream)
    if cache['version'] != FRAME_CACHE_VERSION or cache['frames'] != frames:
        raise ValueError('Production cache does not match the post-exclusion snapshot')
    applied_path = Path('.research/verification/price-source-exclusions-applied.json')
    applied = json.loads(applied_path.read_text())
    by_symbol = {f.symbol:f for f in frames}
    for change in applied['updates']:
        frame = by_symbol[change['symbol']]
        if frame.ticker_id != change['ticker_id'] or frame.prices:
            raise ValueError('Wrong-security prices survived snapshot/cache refresh')
        if frame.price_source_exclusions != {change['excluded_source']:change['evidence']}:
            raise ValueError('Snapshot lost exclusion provenance')
    blocked = []
    for key in ('price_share_basis','membership','terminal_outcomes'):
        try:
            source_ready(manifest['metadata']['sources'], key)
        except ValueError:
            blocked.append(key)
    if len(blocked) != 3:
        raise ValueError('Audit snapshot unexpectedly claims foundation certification')
    result = dict(status='audit_snapshot_and_cache_verified_not_reference_certification',
                  snapshot=str(snapshot), snapshot_sha256=manifest['input_sha256'],
                  source_archive_sha256=manifest['source_archive_sha256'],
                  application_receipt_sha256=hashlib.sha256(applied_path.read_bytes()).hexdigest(),
                  frame_cache_version=FRAME_CACHE_VERSION, cache_matches_snapshot=True,
                  n_securities=len(frames), price_rows=sum(len(f.prices) for f in frames),
                  excluded_symbols=[r['symbol'] for r in applied['updates']],
                  blocked_shared_sources=blocked, database_writes=0, experiments_run=0)
    write_json_new(output,result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    run(a.snapshot,a.output)
