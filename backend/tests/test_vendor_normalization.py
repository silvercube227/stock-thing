import pytest
from backend.ingestion.prices_lseg import normalize_vendor_pair
from scripts.normalize_historical_prices import trim_unpriced_boundaries
from scripts.review_embedded_cash import review as review_embedded_cash


def rows(values):
    return [dict(Date=f'2020-01-0{i+2}',TRDPRC_1=str(v)) for i,v in enumerate(values)]


def test_vendor_split_and_cash_dividend_preserve_nominal_prices():
    result=normalize_vendor_pair(1,rows([100,50,49]),rows([50,50,49]),{'2020-01-04':'1'})
    assert [r['close'] for r in result]==[100,50,49]
    assert [r['adj_close'] for r in result]==pytest.approx([49,49,49])
    assert result[1]['split_factor']==2
    assert result[1]['share_split_factor'] is None
    assert all(r['price_basis']=='as_traded' for r in result)


def test_coincident_cash_and_capital_adjustment_requires_review():
    with pytest.raises(ValueError,match='coincides'):
        normalize_vendor_pair(1,rows([100,50]),rows([50,50]),{'2020-01-03':'1'})


def test_reviewed_special_and_ordinary_cash_are_counted_once():
    result = normalize_vendor_pair(1, rows([100,98]), rows([99,98]), {'2020-01-03':'2'},
        embedded_cash={'2020-01-03': {'amount':'1', 'source':'declared special cash source'}})
    assert [r['adj_close'] for r in result] == pytest.approx([98,98])
    assert result[1]['dividend'] == 2
    assert result[1]['share_split_factor'] is None


@pytest.mark.parametrize('review', [
    {'amount':'1', 'source':'special cash'},
    {'amount':'2', 'source':'special cash'},
    {'amount':'nan', 'source':'special cash'},
    {'amount':'1', 'source':''},
])
def test_review_cannot_explain_away_a_split_or_invalid_cash(review):
    with pytest.raises(ValueError, match='embedded cash'):
        normalize_vendor_pair(1, rows([100,50]), rows([50,50]), {'2020-01-03':'1'},
                             embedded_cash={'2020-01-03':review})


def cash_row(amount, kind='Special'):
    return {'Dividend Ex Date':'2020-01-03', 'Dividend Pay Date':'2020-01-10',
            'Dividend Currency':'USD', 'Dividend Type':kind, 'Gross Dividend Amount':amount}


def test_embedded_review_uses_declared_amounts_and_deduplicates_numeric_formatting():
    included, pending = review_embedded_cash(rows([100,98]), rows([99,98]),
        [cash_row('1'), cash_row('1.0'), cash_row('1','Interim')], [], 'archive')
    assert pending == []
    assert float(included['2020-01-03']['amount']) == 1
    assert float(included['2020-01-03']['total_cash']) == 2
    assert included['2020-01-03']['share_ratio'] is None


def test_ordinary_dividend_or_coincident_demerger_cannot_authorize_embedded_cash():
    included, pending = review_embedded_cash(rows([100,99]), rows([99,99]),
                                            [cash_row('1','Interim')], [], 'archive')
    assert not included and pending
    included, pending = review_embedded_cash(rows([100,99]), rows([99,99]), [cash_row('1')],
        [{'Corporate Change Event Type':'Demerger', 'Capital Change Ex Date':'2020-01-03'}], 'archive')
    assert not included and pending


def test_missing_or_invalid_source_prices_fail():
    with pytest.raises(ValueError,match='date mismatch'):
        normalize_vendor_pair(1,rows([100,50]),rows([50]),{})
    with pytest.raises(ValueError,match='invalid vendor close'):
        normalize_vendor_pair(1,rows([float('nan')]),rows([50]),{})


def test_boundary_placeholders_remain_disclosed_missing_dates():
    raw, adjusted, missing = trim_unpriced_boundaries(
        rows(['<NA>', 100, 101, '<NA>']), rows(['<NA>', 100, 101, '<NA>']), {})
    normalized = normalize_vendor_pair(1, raw, adjusted, {})
    assert [r['close'] for r in normalized] == [100, 101]
    assert [r['date'] for r in missing] == ['2020-01-02', '2020-01-05']
    assert all(r['price_available'] is False for r in missing)


@pytest.mark.parametrize('raw,adjusted,cash,reason', [
    ([100, '<NA>', 101], [100, '<NA>', 101], {}, 'interior price gap'),
    (['<NA>', 100], [99, 100], {}, 'availability mismatch'),
    (['<NA>', 100], ['<NA>', 100], {'2020-01-02': '1'}, 'distribution'),
    (['<NA>'], ['<NA>'], {}, 'no valid source prices'),
])
def test_unpriced_boundaries_cannot_hide_other_source_failures(raw, adjusted, cash, reason):
    with pytest.raises(ValueError, match=reason):
        trim_unpriced_boundaries(rows(raw), rows(adjusted), cash)
