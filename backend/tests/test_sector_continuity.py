import pytest
from datetime import date

from backend.ingestion.index_lseg import build_jl_intervals


EVENTS = [dict(date='2016-09-01', security_id='HON', change=c)
          for c in ('joiner', 'leaver')]
REVIEW = dict(date='2016-09-01', security_id='HON',
              reason='Identical independently queried before/after membership sets',
              evidence=['dated source snapshots'])


def test_reviewed_continuity_preserves_membership_without_splitting_interval():
    result = build_jl_intervals(['HON'], EVENTS, '2010-01-01', '2018-10-01',
                                reviewed_continuity=[REVIEW])
    assert result == [dict(security_id='HON', valid_from=date(2010, 1, 1),
                           valid_to=None, source='lseg_jl_pre_window')]


def test_continuity_review_cannot_create_an_absent_member():
    with pytest.raises(ValueError, match='not active'):
        build_jl_intervals([], EVENTS, '2010-01-01', '2018-10-01',
                           reviewed_continuity=[REVIEW])


def test_unreviewed_pair_and_review_without_exact_pair_still_fail():
    with pytest.raises(ValueError, match='same-day'):
        build_jl_intervals(['HON'], EVENTS, '2010-01-01', '2018-10-01')
    with pytest.raises(ValueError, match='exact same-day'):
        build_jl_intervals(['HON'], EVENTS[:1], '2010-01-01', '2018-10-01',
                           reviewed_continuity=[REVIEW])
