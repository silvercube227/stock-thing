"""Insider-transaction ingestion + feature tests — pure, synthetic, no network/DB.

Covers the correctness-critical pieces: the DERA ZIP parser (column-name join across
three TSVs, universe filter, signed value), the Form 4 XML parser, and the point-in-
time feature join (a filing after the as-of date must not leak).
"""

from __future__ import annotations

import io
import zipfile
from datetime import date

from backend.ingestion.insiders import (
    parse_insider_dataset,
    parse_form4_xml,
    _parse_dera_date,
)
from backend.ml.factors.insiders import _insider_context_asof


def _make_dera_zip(submission_rows, owner_rows, trans_rows) -> bytes:
    """Build an in-memory DERA-style ZIP from lists of (header, rows) TSV tables."""
    def _tsv(header, rows):
        lines = ["\t".join(header)]
        for r in rows:
            lines.append("\t".join(str(x) for x in r))
        return ("\n".join(lines) + "\n").encode("utf-8")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("SUBMISSION.tsv", _tsv(*submission_rows))
        z.writestr("REPORTINGOWNER.tsv", _tsv(*owner_rows))
        z.writestr("NONDERIV_TRANS.tsv", _tsv(*trans_rows))
    return buf.getvalue()


def test_parse_dera_date_handles_both_formats():
    assert _parse_dera_date("31-JAN-2024") == date(2024, 1, 31)
    assert _parse_dera_date("2024-03-15") == date(2024, 3, 15)
    assert _parse_dera_date("") is None
    assert _parse_dera_date(None) is None
    assert _parse_dera_date("garbage") is None


def test_parse_insider_dataset_joins_and_signs_value():
    sub = (
        ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERTRADINGSYMBOL"],
        [
            ["0000-1", "15-MAR-2024", "4", "0000320193", "AAPL"],   # in universe
            ["0000-2", "16-MAR-2024", "4", "0000999999", "ZZZZ"],   # NOT in universe -> dropped
        ],
    )
    owner = (
        ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP"],
        [
            ["0000-1", "0001111111", "Cook Timothy", "Officer, Director"],
            ["0000-2", "0002222222", "Nobody", "Director"],
        ],
    )
    trans = (
        ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
         "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"],
        [
            ["0000-1", "14-MAR-2024", "P", "1000", "10.0", "A"],   # buy: +10000
            ["0000-1", "14-MAR-2024", "S", "500", "20.0", "D"],    # sell: -10000
            ["0000-2", "15-MAR-2024", "P", "1", "1.0", "A"],       # dropped issuer
        ],
    )
    blob = _make_dera_zip(sub, owner, trans)
    rows = parse_insider_dataset(blob, {320193: 42})

    assert len(rows) == 2  # only the in-universe accession, both its transactions
    assert {r["ticker_id"] for r in rows} == {42}
    assert all(r["filing_date"] == date(2024, 3, 15) for r in rows)
    assert all(r["is_officer"] and r["is_director"] for r in rows)
    assert [r["transaction_idx"] for r in rows] == [0, 1]
    by_code = {r["transaction_code"]: r for r in rows}
    assert by_code["P"]["value"] == 10000.0     # +buy
    assert by_code["S"]["value"] == -10000.0    # -sell


def test_parse_insider_dataset_form_filter_and_missing_owner():
    sub = (
        ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK"],
        [
            ["A1", "01-FEB-2024", "8-K", "0000320193"],   # not an insider form -> dropped
            ["A2", "02-FEB-2024", "4", "0000320193"],     # kept even with no owner row
        ],
    )
    owner = (["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP"], [])
    trans = (
        ["ACCESSION_NUMBER", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE",
         "TRANS_ACQUIRED_DISP_CD"],
        [["A2", "P", "100", "5.0", "A"]],
    )
    rows = parse_insider_dataset(_make_dera_zip(sub, owner, trans), {320193: 7})
    assert len(rows) == 1
    r = rows[0]
    assert r["accession_number"] == "A2" and r["transaction_code"] == "P"
    assert r["is_officer"] is False and r["insider_cik"] is None  # tolerated missing owner


FORM4_XML = b"""<?xml version="1.0"?>
<ownershipDocument>
  <issuer><issuerCik>0000320193</issuerCik><issuerTradingSymbol>AAPL</issuerTradingSymbol></issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerCik>0001111111</rptOwnerCik><rptOwnerName>Cook Timothy</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isOfficer>1</isOfficer><isDirector>0</isDirector><isTenPercentOwner>0</isTenPercentOwner></reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2024-03-14</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>100</value></transactionShares>
        <transactionPricePerShare><value>5.0</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""


def test_parse_form4_xml_extracts_transaction():
    rows = parse_form4_xml(FORM4_XML, ticker_id=42, accession_number="X1",
                           filing_date=date(2024, 3, 15))
    assert len(rows) == 1
    r = rows[0]
    assert r["ticker_id"] == 42 and r["accession_number"] == "X1"
    assert r["transaction_code"] == "P"
    assert r["is_officer"] is True and r["is_director"] is False
    assert r["value"] == 500.0  # +100 * 5.0
    assert r["transaction_date"] == date(2024, 3, 14)
    assert r["filing_date"] == date(2024, 3, 15)


def test_insider_context_is_point_in_time():
    # Two buys and one sale, spread over time. The as-of date sits between filings.
    rows = [
        {"filing_date": date(2024, 1, 10), "transaction_code": "P", "value": 10000.0,
         "insider_cik": "A"},
        {"filing_date": date(2024, 2, 10), "transaction_code": "P", "value": 4000.0,
         "insider_cik": "B"},
        {"filing_date": date(2024, 6, 10), "transaction_code": "S", "value": -6000.0,
         "insider_cik": "A"},  # AFTER the as-of date -> must not leak
    ]
    ctx = _insider_context_asof(rows, [date(2024, 3, 1)])
    # 6M net buy sees the two buys (14000), not the June sale.
    assert ctx["net_buy_value_6m"][0] == 14000.0
    # 90d buyers as-of Mar 1: both Jan-10 (A, ~51d) and Feb-10 (B, ~20d) are inside
    # the window -> two distinct cluster buyers.
    assert ctx["insider_buyers_90d"][0] == 2.0
    # 12m net ratio: pure buying so far (June sale is in the future) -> +1.0.
    assert ctx["insider_net_ratio_12m"][0] == 1.0

    # A tighter as-of date drops the January buy from the 90d breadth window.
    ctx2 = _insider_context_asof(rows, [date(2024, 4, 20)])
    assert ctx2["insider_buyers_90d"][0] == 1.0  # only Feb-10 within 90d of Apr-20


def test_insider_context_empty_is_zero():
    ctx = _insider_context_asof([], [date(2024, 3, 1)])
    assert ctx["net_buy_value_6m"] == [0.0]
    assert ctx["insider_buyers_90d"] == [0.0]
    assert ctx["insider_net_ratio_12m"] == [0.0]


def test_insider_net_ratio_mixes_buy_and_sell():
    rows = [
        {"filing_date": date(2024, 1, 5), "transaction_code": "P", "value": 9000.0,
         "insider_cik": "A"},
        {"filing_date": date(2024, 1, 6), "transaction_code": "S", "value": -3000.0,
         "insider_cik": "B"},
    ]
    ctx = _insider_context_asof(rows, [date(2024, 2, 1)])
    # (B - S)/(B + S) = (9000 - 3000)/(9000 + 3000) = 0.5
    assert abs(ctx["insider_net_ratio_12m"][0] - 0.5) < 1e-9
