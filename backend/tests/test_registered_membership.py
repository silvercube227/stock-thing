from datetime import date

import pytest

from scripts.apply_registered_membership import proposed_memberships


def inputs():
    registration = dict(ric="OLD.N^J18", source_isin="old-class", ticker_id=800, cik="0000000123")
    identity = dict(
        Instrument="OLD.N^J18",
        RIC="OLD.N^J18",
        ISIN="old-class",
        **{"CIK Number": "123", "First Trade Date": "NaT", "RetireDate": "2018-10-12"},
    )
    interval = dict(security_id="OLD.N^J18", valid_from="2018-10-01", valid_to="2018-10-15")
    return registration, identity, interval


def propose(registration, identities, intervals):
    return proposed_memberships([registration], identities, intervals, date(2018, 10, 31))


def test_exclusive_weekend_boundary_does_not_invent_late_sessions():
    r, i, m = inputs()
    accepted, blocked = propose(r, [i], [m])
    assert len(accepted) == 1 and not blocked
    assert accepted[0]["membership"][0]["valid_to"] == date(2018, 10, 15)


def test_late_membership_session_blocks_whole_security_without_clipping():
    r, i, m = inputs()
    m["valid_to"] = "2018-10-16"
    accepted, blocked = propose(r, [i], [m])
    assert not accepted
    assert blocked[0]["dates"] == [date(2018, 10, 15)]
    assert blocked[0]["intervals"] == [m]


def test_membership_before_first_trade_is_not_approved():
    r, i, m = inputs()
    i["First Trade Date"] = "2018-10-02"
    accepted, blocked = propose(r, [i], [m])
    assert not accepted
    assert blocked[0]["dates"] == [date(2018, 10, 1)]


def test_exact_isin_cannot_be_borrowed_from_issuer_context():
    r, i, m = inputs()
    context = dict(i, RIC="")
    i["ISIN"] = "different-class"
    with pytest.raises(ValueError, match="ISIN"):
        propose(r, [i, context], [m])


def test_missing_cik_stays_missing_and_no_context_issuer_is_assigned():
    r, i, m = inputs()
    r["cik"] = None
    context = dict(i, RIC="", **{"CIK Number": "456"})
    accepted, blocked = propose(r, [i, context], [m])
    assert not blocked and accepted[0]["cik"] is None


def test_reentries_preserve_gap_and_distinct_episodes():
    r, i, m = inputs()
    m["valid_to"] = "2018-10-05"
    second = dict(m, valid_from="2018-10-10", valid_to="2018-10-15")
    accepted, blocked = propose(r, [i], [second, m])
    assert not blocked
    assert [x["valid_from"] for x in accepted[0]["membership"]] == [
        date(2018, 10, 1),
        date(2018, 10, 10),
    ]


def test_overlapping_episodes_are_rejected():
    r, i, m = inputs()
    with pytest.raises(ValueError, match="overlapping"):
        propose(r, [i], [m, dict(m, valid_from="2018-10-10")])


def test_wrong_registered_issuer_is_rejected():
    r, i, m = inputs()
    r["cik"] = "456"
    with pytest.raises(ValueError, match="issuer"):
        propose(r, [i], [m])
