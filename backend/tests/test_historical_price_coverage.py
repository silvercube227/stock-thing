from datetime import date
from scripts.audit_historical_price_coverage import summarize


def test_reentries_do_not_demand_off_index_prices():
    days=[date(2020,1,d) for d in (2,3,6,7)]
    intervals=[dict(valid_from='2020-01-02',valid_to='2020-01-03'),
               dict(valid_from='2020-01-07',valid_to=None)]
    rows=[dict(Date='2020-01-02',TRDPRC_1='10'),dict(Date='2020-01-07',TRDPRC_1='11')]
    result=summarize(rows,intervals,days)
    assert result['membership_sessions']==2
    assert result['missing_dates']==[]


def test_nonfinite_price_is_missing_and_duplicate_is_disclosed():
    result=summarize([dict(Date='2020-01-02',TRDPRC_1='nan')]*2,
                     [dict(valid_from='2020-01-02',valid_to=None)],[date(2020,1,2)])
    assert result['missing_dates']==['2020-01-02']
    assert result['duplicate_dates']==['2020-01-02']
