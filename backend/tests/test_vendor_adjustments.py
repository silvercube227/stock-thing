from scripts.audit_vendor_adjustments import changes


def test_price_adjustment_is_not_claimed_as_share_split():
    raw=[{'Date':'2020-01-02','TRDPRC_1':'100'}, {'Date':'2020-01-03','TRDPRC_1':'50'}]
    adjusted=[{'Date':'2020-01-02','TRDPRC_1':'50'}, {'Date':'2020-01-03','TRDPRC_1':'50'}]
    r=changes(raw,adjusted)
    assert r['events'][0]['price_adjustment_ratio']==2
    assert r['events'][0]['share_ratio'] is None


def test_missing_adjusted_date_is_disclosed():
    r=changes([{'Date':'2020-01-02','TRDPRC_1':'100'}],[])
    assert r['raw_only_dates']==['2020-01-02']
    assert r['final_raw_to_adjusted_scale'] is None
