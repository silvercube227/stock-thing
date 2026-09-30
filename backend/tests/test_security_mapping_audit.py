from scripts.audit_security_mapping import build_candidates, price_history_conflicts


def test_share_classes_and_reused_symbols_remain_separate():
    tickers = [dict(ticker_id=1, symbol='BRK-B', cik='123'),
               dict(ticker_id=2, symbol='BRK-A', cik='123'),
               dict(ticker_id=3, symbol='PLD', cik='456')]
    rows = [dict(Instrument=ric, ISIN=isin, **{'Ticker Symbol': symbol, 'CIK Number': cik})
            for ric, isin, symbol, cik in (
                ('BRKb.N', 'B', 'BRK B', '000123'),
                ('BRKa.N', 'A', 'BRK A', '000123'),
                ('PLD.N', 'new', 'PLD', '456'),
                ('PLD.N^F11', 'old', 'PLD', '456'))]
    result = build_candidates(dict(records=rows, requested_rics=[r['Instrument'] for r in rows]), tickers)
    assert result[0]['candidate_ticker_ids'] == [1]
    assert result[1]['candidate_ticker_ids'] == [2]
    assert all(r['status'] == 'ambiguous' for r in result[2:])


def test_missing_symbol_is_not_resolved_from_issuer_or_ric_stem():
    rows = [dict(Instrument='OLD.N^J19', ISIN='old', **{'CIK Number': '123'})]
    result = build_candidates(dict(records=rows, requested_rics=['OLD.N^J19']),
                              [dict(ticker_id=1, symbol='OLD', cik='123')])[0]
    assert result['candidate_ticker_ids'] == []
    assert result['issuer_only_review_ids'] == [1]
    assert result['review_reason'] == 'source_symbol_missing'


def test_exchange_symbol_and_missing_source_are_explicit():
    rows = [dict(Instrument='X.N', ISIN='x', **{'CIK Number': '123', 'Exchange Ticker': 'X'})]
    result = build_candidates(dict(records=rows, requested_rics=['X.N', 'Y.N']),
                              [dict(ticker_id=1, symbol='X', cik='123')])
    assert result[0]['candidate_ticker_ids'] == [1]
    assert result[1]['review_reason'] == 'source_issuer_missing'


def test_price_tail_conflict_does_not_infer_outcomes_or_flag_missing_prices():
    mappings = [dict(ric='OLD.N', candidate_ticker_ids=[1, 2],
                     retire_dates=['2016-06-01'], status='ambiguous')]
    inventory = [dict(ticker_id=1, symbol='OLD', last_price_date='2026-09-04'),
                 dict(ticker_id=2, symbol='OTHER', last_price_date=None)]
    conflicts = price_history_conflicts(mappings, inventory)
    assert len(conflicts) == 1
    assert conflicts[0]['ticker_id'] == 1
    assert conflicts[0]['status'] == 'requires_source_history_review'
    assert 'terminal_proceeds' not in conflicts[0]
