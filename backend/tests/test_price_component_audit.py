from datetime import date

import pytest

from backend.ingestion.prices_lseg import normalize_vendor_pair
from scripts.audit_praxair_price_basis import component_parity


def history(cash):
    rows = [dict(Date="2020-01-02", TRDPRC_1="100"), dict(Date="2020-01-03", TRDPRC_1="101")]
    return normalize_vendor_pair(1, rows, rows, {"2020-01-03": str(cash)})


def test_cash_rounding_does_not_excuse_a_vendor_derived_residual():
    native, vendor = history(0.7875), history(0.788)
    result = component_parity(native, vendor)
    assert len(result["cash_differences"]) == 1
    assert len(result["vendor_raw_ratio_differences"]) == 1
    assert not result["native_formula_errors"]
    assert not result["vendor_derived_residuals"]
    vendor[1]["adj_close"] *= 1.00001
    assert len(component_parity(native, vendor)["vendor_derived_residuals"]) == 1


def test_native_formula_error_is_not_hidden_by_matching_vendor_prices():
    native, vendor = history(0.5), history(0.5)
    native[1]["adj_close"] *= 1.00001
    assert component_parity(native, vendor)["native_formula_errors"]


@pytest.mark.parametrize("change", ["split", "missing_date", "duplicate", "nonfinite"])
def test_unreviewed_price_components_fail_closed(change):
    native, vendor = history(0.5), history(0.5)
    if change == "split":
        native[1]["split_factor"] = 2
    elif change == "missing_date":
        vendor[1]["trade_date"] = date(2020, 1, 6)
    elif change == "duplicate":
        native[1]["trade_date"] = native[0]["trade_date"]
    else:
        native[1]["close"] = float("nan")
    with pytest.raises(ValueError):
        component_parity(native, vendor)
