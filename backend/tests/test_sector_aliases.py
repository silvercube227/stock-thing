import json
import pytest

from scripts.audit_sector_membership import reviewed_aliases, canonical_membership_rows


def review_file(tmp_path):
    path = tmp_path / 'review.json'
    path.write_text(json.dumps(dict(scope='sector_membership_identity_only', aliases=[
        dict(rics=['ABC.N','ABC.OQ'], security_id='ISIN:US1', isin='US1', cusip='C1',
             cik='001', transfer_source='exchange-notice', listing_transfer_date='2021-12-07',
             evidence={})])))
    return path


def test_same_security_can_reconcile_venue_aliases(tmp_path, monkeypatch):
    monkeypatch.setattr('scripts.audit_sector_membership.load_entry', lambda _: dict(records=[
        {'RIC':ric, 'ISIN':'US1', 'CUSIP':'C1', 'CIK Number':'001'} for ric in ['ABC.N','ABC.OQ']]))
    assert reviewed_aliases(review_file(tmp_path)) == {'ABC.N':'ISIN:US1', 'ABC.OQ':'ISIN:US1'}


def test_same_issuer_cannot_merge_distinct_securities(tmp_path, monkeypatch):
    monkeypatch.setattr('scripts.audit_sector_membership.load_entry', lambda _: dict(records=[
        {'RIC':'ABC.N', 'ISIN':'US1', 'CUSIP':'C1', 'CIK Number':'001'},
        {'RIC':'ABC.OQ', 'ISIN':'US2', 'CUSIP':'C2', 'CIK Number':'001'}]))
    with pytest.raises(ValueError, match='not unambiguous'):
        reviewed_aliases(review_file(tmp_path))


def test_reviewed_duplicate_listing_is_counted_once_and_disclosed():
    rows = [{'Constituent RIC':r} for r in ['HST.N','HST.OQ','GOOG.OQ','GOOGL.OQ']]
    normalized, duplicates = canonical_membership_rows(rows, {'HST.N':'ISIN:HST','HST.OQ':'ISIN:HST'})
    assert [r['Constituent RIC'] for r in normalized] == ['ISIN:HST','GOOG.OQ','GOOGL.OQ']
    assert duplicates == [dict(security_id='ISIN:HST', rics=['HST.N','HST.OQ'])]
    assert rows[0]['Constituent RIC'] == 'HST.N'


def test_repeated_exact_ric_is_not_excused_by_alias_review():
    with pytest.raises(ValueError, match='duplicate historical constituent'):
        canonical_membership_rows([{'Constituent RIC':'HST.N'}]*2, {'HST.N':'ISIN:HST'})
