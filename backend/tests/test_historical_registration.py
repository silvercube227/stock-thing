import hashlib
import pytest

from scripts.register_historical_securities import prepare


def test_shared_issuer_does_not_merge_distinct_historical_securities(tmp_path):
    raw=tmp_path/'raw.json'
    raw.write_bytes(b'{}')
    securities=[dict(ric=r, isins=[isin], issuer_ciks=['123'], existing_candidate_ids=[],
                     status='missing_security_requires_backfill')
                for r,isin in [('OLD.N^F11','OLD'),('NEW.N','NEW')]]
    histories=[dict(ric=s['ric'], price_rows=1,
                    status='observed_requires_action_and_coverage_review',artifact=str(raw),
                    sha256=hashlib.sha256(b'{}').hexdigest()) for s in securities]
    result=prepare({'securities':securities},{'securities':histories})
    assert len(result)==2
    assert result[0]['symbol']!=result[1]['symbol']
    assert all(r['cik']=='0000000123' for r in result)
    raw.write_bytes(b'changed')
    with pytest.raises(ValueError,match='archive changed'):
        prepare({'securities':securities},{'securities':histories})


def test_ambiguous_or_failed_histories_are_not_registered():
    s=dict(ric='X.N', isins=['X'],issuer_ciks=[],existing_candidate_ids=[],
           status='conflicting_source_rows')
    assert prepare({'securities':[s]},{'securities':[]})==[]
    s['status']='missing_security_requires_backfill'
    assert prepare({'securities':[s]},{'securities':[dict(ric='X.N',price_rows=0,status='error')]})==[]
