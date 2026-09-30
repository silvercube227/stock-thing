import hashlib
import pytest

from backend.ml.compressed_io import open_artifact
from scripts.compress_research_storage import compress


def test_lossless_compression_and_legacy_reading(tmp_path):
    path=tmp_path/'inputs.pkl'
    content=b'reproducible snapshot\n'*1000
    path.write_bytes(content)
    with open_artifact(path) as stream:
        assert stream.read()==content
    result=compress(path,hashlib.sha256(content).hexdigest())
    assert result['after_bytes']<result['before_bytes']
    with open_artifact(path) as stream:
        assert stream.read()==content
    assert compress(path)['status']=='already_compressed'


def test_hash_mismatch_preserves_original(tmp_path):
    path=tmp_path/'inputs.pkl'
    path.write_bytes(b'original')
    with pytest.raises(ValueError,match='before compression'):
        compress(path,'incorrect')
    assert path.read_bytes()==b'original'
    assert not path.with_name('inputs.pkl.compressing').exists()
