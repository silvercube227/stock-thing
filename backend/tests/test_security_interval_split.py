from datetime import date
import pytest
from backend.ingestion.security_intervals import split_predecessor_intervals


def split(rows):
    return split_predecessor_intervals(rows, predecessor='PX', successor='LIN',
                                      effective_date='2018-10-31', evidence='completion-filing-hash')


def test_merger_splits_identity_on_exact_boundary():
    source = [dict(security_id='LIN', valid_from='2010-01-01', valid_to=None)]
    result = split(source)
    assert [(r['security_id'], r['valid_from'], r['valid_to']) for r in result] == [
        ('PX', '2010-01-01', date(2018,10,31)), ('LIN', date(2018,10,31), None)]
    assert source[0]['security_id'] == 'LIN'


def test_membership_gaps_are_not_filled_by_security_transition():
    result = split([dict(security_id='LIN',valid_from='2010-01-01',valid_to='2012-01-01'),
                    dict(security_id='LIN',valid_from='2020-01-01',valid_to=None)])
    assert len(result)==2
    assert result[0]['security_id']=='PX' and result[0]['valid_to']==date(2012,1,1)
    assert result[1]['valid_from']=='2020-01-01'


def test_existing_predecessor_overlap_requires_review():
    with pytest.raises(ValueError,match='overlapping'):
        split([dict(security_id='PX',valid_from='2010-01-01',valid_to='2018-10-31'),
               dict(security_id='LIN',valid_from='2010-01-01',valid_to=None)])
