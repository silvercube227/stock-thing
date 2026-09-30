from datetime import date
from decimal import Decimal

import pytest

from scripts.apply_praxair_reference import verify_stored_prices


def test_price_readback_accepts_only_schema_quantization():
    expected = [
        dict(trade_date=date(2018, 1, 1), close=100.12345649, dividend=0.7875, source="lseg")
    ]
    stored = [
        dict(
            trade_date=date(2018, 1, 1),
            close=Decimal("100.123456"),
            dividend=Decimal("0.7875"),
            source="lseg",
        )
    ]
    verify_stored_prices(expected, stored)
    stored[0]["close"] = Decimal("100.123455")
    with pytest.raises(ValueError, match="quantization"):
        verify_stored_prices(expected, stored)


@pytest.mark.parametrize(
    "changed",
    [dict(dividend=Decimal("0.788")), dict(source="yfinance"), dict(trade_date=date(2018, 1, 2))],
)
def test_price_readback_rejects_changed_components(changed):
    expected = [dict(trade_date=date(2018, 1, 1), close=100, dividend=0.7875, source="lseg")]
    stored = [dict(expected[0], **changed)]
    with pytest.raises(ValueError):
        verify_stored_prices(expected, stored)
