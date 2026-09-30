from datetime import date
import pytest
from backend.ingestion.dividend_review import review_dividends
from backend.ingestion.prices_lseg import to_price_rows


def row(amount='1.0',kind='Interim',currency='USD'):
    return {'Dividend Ex Date':'2020-01-02','Dividend Pay Date':'2020-01-10',
            'Gross Dividend Amount':amount,'Dividend Type':kind,'Dividend Currency':currency}


def test_numeric_duplicate_does_not_double_count_and_special_is_additive():
    result=review_dividends([row('1'),row('1.0'),row('2','Special')],['2020-01-02'])
    assert result['cash_by_ex_date']=={'2020-01-02':'3'}


def test_conflicting_amount_or_unknown_currency_masks_whole_date():
    for records in ([row('1'),row('2')],[row(),row('1',currency='<NA>')]):
        result=review_dividends(records,['2020-01-02'])
        assert result['issues'] and not result['cash_by_ex_date']


@pytest.mark.parametrize('amount',[float('nan'),float('inf'),-1])
def test_normalizer_rejects_invalid_dividends(amount):
    d=date(2020,1,2)
    with pytest.raises(ValueError,match='cash dividend'):
        to_price_rows(1,[dict(trade_date=d,close=10)],{d:amount},{},verified=True)
