"""Pure-logic tests for RIC -> ticker resolution (no LSEG, no database)."""

from datetime import date

from scripts.map_security_identity import build_mappings, history_check, ric_stem


def _t(first, last):
    return {"first_px": first, "last_px": last}


def test_ric_stem_strips_retirement_and_exchange():
    assert ric_stem("AABA.OQ^J19") == "AABA"
    assert ric_stem("BNI.N^B10") == "BNI"
    assert ric_stem("AAPL.O") == "AAPL"
    assert ric_stem("A.N") == "A"


def test_ric_stem_preserves_share_class():
    # A class letter is identity, not noise: BRKb is not BRK.
    assert ric_stem("BRKb.N") == "BRK-B"
    assert ric_stem("BFb.N") == "BF-B"


def test_history_inside_lifecycle_is_consistent():
    state, note = history_check(
        _t(date(2012, 1, 3), date(2018, 6, 15)), date(2010, 1, 1), date(2018, 6, 20)
    )
    assert state == "consistent" and note is None


def test_bars_long_after_retirement_are_reuse_not_a_mapping():
    # Signature Bank failed in 2023; bars in 2026 belong to another company.
    state, note = history_check(_t(date(2011, 1, 3), date(2026, 9, 4)), None, date(2023, 3, 28))
    assert state == "reuse_conflict"
    assert "2026-09-04" in note


def test_bars_just_after_retirement_stay_within_grace():
    # A delisting tail inside the guard window is the same security, not reuse.
    state, _ = history_check(_t(date(2011, 1, 3), date(2018, 7, 2)), None, date(2018, 6, 15))
    assert state == "consistent"


def test_bars_before_first_trade_flag_a_spliced_predecessor():
    # Alcoa Corp first traded in 2016; 2010 bars are a different issuer.
    state, note = history_check(_t(date(2010, 1, 4), date(2026, 9, 4)), date(2016, 10, 18), None)
    assert state == "predates_security"
    assert "2010-01-04" in note


def test_ticker_without_price_history_is_not_claimed_consistent():
    state, _ = history_check(_t(None, None), date(2010, 1, 1), date(2015, 1, 1))
    assert state == "no_price_history"


def _security(ric, isin="class-a", cik="123", symbol=""):
    return dict(Instrument=ric, RIC=ric, ISIN=isin, **{"CIK Number": cik, "Ticker Symbol": symbol})


def _ticker(tid, symbol, cik="123", ric=None):
    return dict(ticker_id=tid, symbol=symbol, cik=cik, ric=ric, first_px=None, last_px=None)


def _map(rows, tickers):
    data = dict(records=rows, requested_rics=list(dict.fromkeys(r["Instrument"] for r in rows)))
    return build_mappings(data, tickers)


def test_unique_issuer_cannot_select_a_different_share_class():
    result = _map([_security("FOXA.OQ")], [_ticker(1, "FOX")])[0]
    assert result["candidate_ticker_ids"] == []
    assert result["status"] == "issuer_only_requires_security_review"


def test_share_classes_resolve_separately_by_symbol_and_issuer():
    result = _map(
        [_security("FOXA.OQ"), _security("FOX.OQ", isin="class-b")],
        [_ticker(1, "FOX"), _ticker(2, "FOXA")],
    )
    assert [r["candidate_ticker_ids"] for r in result] == [[2], [1]]
    assert all(r["status"] == "identity_candidate_no_history" for r in result)


def test_blank_ric_context_cannot_turn_valaris_into_valspar():
    security = _security("VAL.N^H20", isin="valaris", cik="1476009")
    context = dict(security, RIC="", ISIN="", **{"CIK Number": "314808"})
    result = _map([security, context], [_ticker(1, "VAL", cik="314808")])
    assert len(result) == 1
    assert result[0]["candidate_ticker_ids"] == []
    assert result[0]["issuer_context_rows"] == 1
    assert result[0]["cik"] == "1476009"


def test_registered_exact_ric_wins_over_another_security_of_the_issuer():
    result = _map(
        [_security("OLD.N^J18")], [_ticker(1, "NEW"), _ticker(2, "LSEG:OLD.N^J18", ric="OLD.N^J18")]
    )[0]
    assert result["candidate_ticker_ids"] == [2]
    assert result["rule"] == "exact_stored_ric"


def test_stored_ric_with_conflicting_issuer_remains_blocked():
    result = _map([_security("X.N")], [_ticker(1, "X", cik="456", ric="X.N")])[0]
    assert result["status"] == "blocked_stored_issuer_conflict"


def test_missing_isin_and_duplicate_exact_rows_are_not_approved():
    row = _security("X.N", isin="")
    assert _map([row], [_ticker(1, "X")])[0]["status"] == "blocked_missing_security_identifier"
    assert _map([row, row], [_ticker(1, "X")])[0]["status"] == "blocked_source_security_rows"


def test_same_symbol_cannot_map_two_successive_securities_without_review():
    result = _map([_security("X.N"), _security("X.N^J18", isin="old")], [_ticker(1, "X")])
    assert all(r["status"] == "blocked_multiple_ric_candidates" for r in result)
