from datetime import date
import json

import pytest

from backend.ml.dataset import _exclude_price_sources, _price_source_exclusions, load_frames
from backend.ml.factors.assembly import build_universe_return_map
from scripts.audit_original_security_prices import compare


EVIDENCE = {'yfinance': {'reason': 'native price disagreement', 'artifact': 'audit.json', 'sha256': 'a'*64}}


def test_pre_retirement_wrong_prices_do_not_pass_tail_guard_audit():
    stored = [dict(trade_date='2017-03-09', close=3200, split_factor=1, source='yfinance'),
              dict(trade_date='2017-03-10', close=3210, split_factor=1, source='yfinance'),
              dict(trade_date='2017-03-14', close=3220, split_factor=1, source='yfinance')]
    native = [dict(Date='2017-03-09', TRDPRC_1='111.5'), dict(Date='2017-03-10', TRDPRC_1='112')]
    result = compare(stored, native, date(2017, 3, 13))
    assert result['retirement_filtered_rows'] == 2
    assert result['mismatched_dates'] == 2
    assert result['parity_dates'] == 0
    assert result['status'] == 'price_disagreement_requires_quarantine_or_repair'


def test_source_exclusion_preserves_raw_rows_and_other_sources():
    original = [dict(source='yfinance', close=3200), dict(source='lseg', close=112)]
    assert _exclude_price_sources(original, EVIDENCE) == [original[1]]
    assert len(original) == 2
    assert _price_source_exclusions(json.dumps(EVIDENCE)) == EVIDENCE


@pytest.mark.parametrize('value', [[], '{invalid', {'yfinance': {}}, {'yfinance': 'unverified'}])
def test_malformed_exclusion_stops_loading(value):
    with pytest.raises(ValueError):
        _price_source_exclusions(value)


@pytest.mark.asyncio
async def test_loader_excludes_prices_from_frames_and_market_controls():
    class Pool:
        async def fetch(self, sql, *args):
            if 'from tickers' in sql:
                return [dict(ticker_id=1, symbol='BAD', embedding_idx=1, shares_outstanding=None,
                             sector='Industrials', industry=None, removed_at=None,
                             security_retired_at='2017-03-13', price_source_exclusions=json.dumps(EVIDENCE))]
            if 'from price_history' in sql:
                return [dict(ticker_id=1, source='yfinance', trade_date=date(2017,3,9), adj_close=3200),
                        dict(ticker_id=1, source='yfinance', trade_date=date(2017,3,10), adj_close=3300)]
            return []

    frames = await load_frames(Pool())
    assert frames[0].prices == []
    assert frames[0].price_source_exclusions == EVIDENCE
    assert build_universe_return_map(frames) == {}
