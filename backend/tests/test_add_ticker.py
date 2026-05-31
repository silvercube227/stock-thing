"""Pure-function tests for the user-added-ticker orchestrator.

The full ingest+score path needs a DB + a trained model and is exercised by the
verification steps in the plan; here we cover the parsing/mapping helpers.
"""

from __future__ import annotations

from backend.jobs.add_ticker import (
    _YAHOO_TO_GICS,
    _asset_type_from_quote,
    _parse_score_outcome,
)


def test_asset_type_from_quote():
    assert _asset_type_from_quote("EQUITY") == "equity"
    assert _asset_type_from_quote("equity") == "equity"
    assert _asset_type_from_quote("ETF") == "etf"
    assert _asset_type_from_quote("INDEX") is None      # unsupported -> caller rejects
    assert _asset_type_from_quote(None) is None


def test_yahoo_to_gics_covers_all_sectors():
    # 11 GICS sectors; the values must be the canonical strings used in the seed.
    assert _YAHOO_TO_GICS["Technology"] == "Information Technology"
    assert _YAHOO_TO_GICS["Financial Services"] == "Financials"
    assert _YAHOO_TO_GICS["Healthcare"] == "Health Care"
    assert len(set(_YAHOO_TO_GICS.values())) == 11


def test_parse_score_outcome_finds_json_line():
    out = (
        "[frame-cache] hit: ...\n"
        "some lightgbm noise\n"
        '{"status": "scored", "symbol": "RIVN", "ranks": {"3M": 0.42}}\n'
        "scored RIVN as_of=2020-03-31 ranks={'3M': 0.42}"
    )
    parsed = _parse_score_outcome(out)
    assert parsed is not None
    assert parsed["status"] == "scored"
    assert parsed["ranks"] == {"3M": 0.42}


def test_parse_score_outcome_insufficient_history():
    parsed = _parse_score_outcome('{"status": "insufficient_history", "symbol": "NEW"}')
    assert parsed == {"status": "insufficient_history", "symbol": "NEW"}


def test_parse_score_outcome_none_when_absent():
    assert _parse_score_outcome("no json here\njust logs") is None
    assert _parse_score_outcome("{not valid json}") is None
