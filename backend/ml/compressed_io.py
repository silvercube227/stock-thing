"""Transparent gzip transport for local artifacts; legacy plain files still read."""
import gzip
from pathlib import Path


def open_artifact(path):
    with Path(path).open('rb') as stream:
        compressed = stream.read(2) == b'\x1f\x8b'
    return gzip.open(path, 'rb') if compressed else Path(path).open('rb')
