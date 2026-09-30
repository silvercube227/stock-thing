import pytest
import json
from datetime import date

from backend.ml.terminal import successor_horizon_values
from backend.ml.factors.pit import documented_targets
from backend.ingestion.calendar import shift_trading_days


def price(d, close, adj, split=1):
    return dict(trade_date=d, close=close, adj_close=adj, split_factor=split,
                source='yfinance')


def value(successor, **kwargs):
    args = dict(target_prices=[price('2018-06-14', 100, 80)],
                successor_prices=successor, target_last_trade='2018-06-14',
                ownership_close='2018-06-14', horizon_dates=['2018-09-14'],
                cash_per_share=50, exchange_ratio=2)
    args.update(kwargs)
    return successor_horizon_values(**args)


def test_cash_is_not_reinvested_and_target_basis_is_preserved():
    result = value([price('2018-06-14', 25, 20), price('2018-09-14', 30, 30)])
    # Stock grows from 2*25 to 75 including distributions; cash stays 50.
    assert result['horizon_values']['2018-09-14'] == pytest.approx((50+75)*0.8)


def test_successor_split_does_not_halve_the_consideration():
    result = value([price('2018-06-14', 25, 25), price('2018-09-14', 25, 25, 2)])
    # Initial split-adjusted 25 is nominal 50; two initial shares become four.
    assert result['horizon_values']['2018-09-14'] == pytest.approx((50+100)*0.8)


def test_missing_horizon_is_unknown_not_last_observation_carried_forward():
    result = value([price('2018-06-14', 25, 25), price('2018-09-13', 30, 30)])
    assert result['horizon_values'] == {}
    assert result['missing_horizons'] == ['2018-09-14']


def test_reused_target_tail_or_missing_anchor_fails():
    with pytest.raises(ValueError, match='final trading'):
        value([price('2018-06-14', 25, 25)],
              target_prices=[price('2018-06-14', 100, 80), price('2018-06-15', 50, 50)])
    with pytest.raises(ValueError, match='ownership close'):
        value([price('2018-06-15', 25, 25)])


def test_unknown_adjustments_and_invalid_terms_fail():
    with pytest.raises(ValueError, match='adjustment basis'):
        value([price('2018-06-14', 25, 25), price('2018-09-14', 25, 25, None)])
    with pytest.raises(ValueError, match='consideration'):
        value([price('2018-06-14', 25, 25)], exchange_ratio=float('nan'))


def test_last_trading_session_entry_survives_but_placeholder_entry_does_not():
    final = date(2018, 6, 14)
    end = shift_trading_days(final, 63)
    event = dict(effective_date=final, event_type='acquisition', verified=True,
                 source=json.dumps({'last_trade_date':str(final)}),
                 consideration_type='successor', proceeds_basis='adj_close',
                 horizon_values={str(end):110})
    rows = [price('2018-06-13',100,100), price('2018-06-14',100,100),
            price('2018-06-15',100,100)]
    labels = documented_targets(rows,0,[event],date(2020,1,1))
    assert labels['3M'][1]
    assert labels['3M'][4]=='acquisition'
    assert all(v[4]=='entry_after_terminal' for v in
               documented_targets(rows,1,[event],date(2020,1,1)).values())


def test_documented_acquisition_with_unresolved_election_masks_payoff_and_reused_bars():
    final = date(2018, 9, 28)
    event = dict(effective_date=date(2018, 10, 1), event_type='acquisition',
                 verified=True, source=json.dumps({'last_trade_date': str(final)}),
                 consideration_type='unresolved_election', cash_value=None,
                 proceeds_basis=None, horizon_values=None)
    rows = [price('2018-09-27', 150, 150), price(final, 150, 150),
            price('2018-10-01', 80, 80)]
    # A successor's apparent horizon bar cannot substitute for unknown proceeds.
    rows.append(price(shift_trading_days(final, 63), 200, 200))
    result = documented_targets(rows, 0, [event], date(2020, 1, 1))
    assert all(not r[1] and r[4] == 'acquisition' for r in result.values())
    result = documented_targets(rows, 1, [event], date(2020, 1, 1))
    assert all(not r[1] and r[4] == 'entry_after_terminal' for r in result.values())


def test_unresolved_gap_is_not_proof_of_terminal_entry_cutoff():
    final = date(2018, 9, 28)
    rows = [price(final, 100, 100), price('2018-10-01', 100, 100)]
    event = dict(effective_date=final, event_type='unresolved_gap', verified=True,
                 source='reviewed missing data, not a confirmed termination')
    result = documented_targets(rows, 0, [event], date(2020, 1, 1))
    assert all(r[4] != 'entry_after_terminal' for r in result.values())
