"""Losslessly compress snapshots/caches in place after streaming hash verification."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from backend.ml.research import write_json_new


def compress(path, expected=None):
    before=path.stat().st_size
    with path.open('rb') as stream:
        if stream.read(2)==b'\x1f\x8b':
            return dict(path=str(path),status='already_compressed',bytes=before)
        stream.seek(0)
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if expected and digest!=expected:
        raise ValueError(f'Snapshot hash mismatch before compression: {path}')
    temp=path.with_name(path.name+'.compressing')
    with path.open('rb') as source, temp.open('xb') as dest:
        with gzip.GzipFile(fileobj=dest,mode='wb',compresslevel=6,mtime=0) as packed:
            shutil.copyfileobj(source,packed,length=1024*1024)
    with gzip.open(temp,'rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=digest:
            raise ValueError('Compression replay hash mismatch; original preserved')
    # Refuse to replace a cache refreshed by another writer while compressing.
    with path.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=digest:
            raise ValueError('Input changed during compression; original preserved')
    temp.replace(path)
    return dict(path=str(path),status='compressed',before_bytes=before,
                after_bytes=path.stat().st_size,uncompressed_sha256=digest)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    results=[]
    for manifest in sorted(Path('.research').glob('*/manifest.json')):
        data=json.loads(manifest.read_text())
        target=manifest.parent/'inputs.pkl'
        if target.exists() and data.get('input_sha256'):
            result=compress(target,data['input_sha256'])
            results.append(result)
            print(result,flush=True)
    for target in sorted(Path('.frame_cache').glob('*.pkl')):
        result=compress(target)
        results.append(result)
        print(result,flush=True)
    write_json_new(a.output,dict(artifacts=results,
        note='Gzip transport at existing paths. Snapshot hashes remain hashes of decompressed content.'))
