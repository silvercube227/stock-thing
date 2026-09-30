from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from backend.ingestion.index_lseg import build_jl_intervals, index_on
from backend.ingestion.macro import parse_alfred
from backend.ingestion.news_lseg import (
    daily_aggregate,
    fetch_window,
    is_duplicate,
    normalized_title,
    open_store,
    parse_headline,
    score_session,
    store_rows,
)
from backend.ml.factors.pit import aligned_shares, as_traded_closes, documented_targets


def test_split_basis_and_dated_shares():
    dates = [date(2020, 8, 28), date(2020, 8, 31), date(2020, 9, 1)]
    prices = [
        dict(trade_date=d, close=25.0, adj_close=24.0, split_factor=s, share_split_factor=s, source="yfinance")
        for d, s in zip(dates, [1.0, 4.0, 1.0])
    ]
    assert as_traded_closes(prices) == [100.0, 25.0, 25.0]
    filing = dict(
        shares_outstanding=100.0,
        shares_measured_at=dates[0],
        filed_at=dates[0],
        shares_kind="point_in_time",
        shares_basis="as_reported",
    )
    assert aligned_shares([filing], prices, dates) == [100.0, 400.0, 400.0]
    filing["shares_kind"] = "weighted_average"
    assert np.isnan(aligned_shares([filing], prices, dates)).all()
    prices[1]["split_factor"] = None
    assert np.isnan(as_traded_closes(prices)[0])


def test_reverse_split_and_measurement_after_split():
    dates = [date(2020, 8, 28), date(2020, 8, 31)]
    p = [
        dict(trade_date=d, close=100.0, source="yfinance", split_factor=s, share_split_factor=s)
        for d, s in zip(dates, [1.0, 0.1])
    ]
    assert as_traded_closes(p) == [10.0, 100.0]
    f = dict(
        shares_outstanding=10,
        shares_measured_at=dates[1],
        filed_at=dates[1],
        shares_kind="point_in_time",
        shares_basis="as_reported",
    )
    assert np.isnan(aligned_shares([f], p, dates)[0])
    assert aligned_shares([f], p, dates)[1] == 10


def test_price_only_distribution_does_not_increase_parent_shares():
    dates = [date(2024, 4, 1), date(2024, 4, 2)]
    prices = [dict(trade_date=dates[0], split_factor=1),
              dict(trade_date=dates[1], split_factor=1.253, share_split_factor=1)]
    filing = dict(shares_outstanding=100, shares_measured_at=dates[0],
                  filed_at=dates[0], shares_kind='point_in_time', shares_basis='as_reported')
    assert aligned_shares([filing], prices, dates) == [100, 100]
    prices[1]['share_split_factor'] = None
    assert np.isnan(aligned_shares([filing], prices, dates)[1])
    prices[1].update(split_factor=.125, share_split_factor=.125)
    assert aligned_shares([filing], prices, dates)[1] == 12.5


def test_share_action_corroboration_rejects_dates_ratios_and_ambiguous_events():
    from backend.ingestion.share_actions import classify_share_action

    row = {'Capital Change Is Rescinded':'False', 'Capital Change Ex Date':'2024-06-10',
           'Capital Change Effective Date':'2024-06-07',
           'Corporate Change Event Type':'Share Split', 'Adjustment Type':'Capital Change Type',
           'Terms Old Shares':'1', 'Terms New Shares':'10', 'Adjustment Factor':'0.1'}
    assert classify_share_action('2024-06-10',10,[row]) == (0,10)
    assert classify_share_action('2024-06-07',10,[row]) is None
    assert classify_share_action('2024-06-10',2,[row]) is None
    assert classify_share_action('2024-06-10',10,[row,row]) is None
    assert classify_share_action('2024-06-10',10,[{**row,'Capital Change Is Rescinded':'True'}]) is None
    distribution = {**row,'Corporate Change Event Type':'Demerger',
                    'Adjustment Type':'LSEG Pricing Only','Adjustment Factor':'0.8'}
    assert classify_share_action('2024-06-10',1.25,[distribution]) == (0,1)


def test_terminal_requires_documented_proceeds_and_maturity():
    p = [
        dict(trade_date=date(2020, 1, 2), adj_close=100.0),
        dict(trade_date=date(2020, 1, 3), adj_close=100.0),
    ]
    end = date(2021, 2, 1)
    assert not documented_targets(p, 0, [], end)["3M"][1]
    e = dict(
        effective_date=date(2020, 1, 6),
        event_type="acquisition",
        verified=True,
        source="deal_filing",
        proceeds_basis="adj_close",
        consideration_type="cash",
        cash_value=120.0,
    )
    observed = documented_targets(p, 0, [e], end)
    assert observed["3M"][0] == pytest.approx(np.log(1.2))
    assert observed["3M"][1]
    assert not documented_targets(p, 0, [e], date(2020, 2, 1))["3M"][1]
    e.update(event_type="bankruptcy", cash_value=0.0)
    assert documented_targets(p, 0, [e], end)["3M"][4] == "documented_zero_recovery"


def test_successor_needs_horizon_value():
    p = [
        dict(trade_date=date(2020, 1, 2), adj_close=100.0),
        dict(trade_date=date(2020, 1, 3), adj_close=100.0),
    ]
    e = dict(
        effective_date=date(2020, 1, 6),
        event_type="security_replacement",
        verified=True,
        source="filing",
        proceeds_basis="adj_close",
        consideration_type="successor",
    )
    target = documented_targets(p, 0, [e], date(2021, 2, 1))["3M"]
    assert not target[1]
    e["horizon_values"] = {target[3].isoformat(): 130.0}
    assert documented_targets(p, 0, [e], date(2021, 2, 1))["3M"][1]


def test_confirmed_exit_rejects_entries_into_reused_price_tail():
    p = [dict(trade_date=date(2020,1,2),adj_close=100),
         dict(trade_date=date(2020,1,3),adj_close=110)]
    event = dict(effective_date=date(2020,1,2),event_type='acquisition',
                 consideration_type='cash',cash_value=120,verified=True,source='merger_filing')
    result = documented_targets(p,0,[event],date(2021,2,1))
    assert all(not v[1] and v[4]=='entry_after_terminal' for v in result.values())


def test_membership_reentries_and_inconsistency():
    events = [
        dict(date="2010-02-01", security_id="A", change="Leaver"),
        dict(date="2011-02-01", security_id="A", change="Joiner"),
    ]
    intervals = build_jl_intervals(["A", "B"], events, "2010-01-01", "2020-01-01")
    assert len(intervals) == 3
    a = [r for r in intervals if r["security_id"] == "A"]
    assert index_on(a, date(2010, 5, 1)) is None
    assert index_on(a, date(2011, 5, 1)) == "SPX"
    with pytest.raises(ValueError):
        build_jl_intervals([], [events[1]], "2010-01-01", "2020-01-01")


def test_only_evidenced_zero_duration_membership_is_excluded():
    events = [dict(date='2016-12-02', security_id='old', change=c)
              for c in ('Joiner', 'Leaver')]
    events.append(dict(date='2016-12-02', security_id='new', change='Joiner'))
    review = dict(date='2016-12-02', security_id='old',
                  reason='documented successor replaces security before session',
                  evidence=['https://example.org/index-notice'])
    with pytest.raises(ValueError, match='same-day'):
        build_jl_intervals(['new'], events, '2010-01-01', '2020-01-01')
    rows = build_jl_intervals(['new'], events, '2010-01-01', '2020-01-01',
                              reviewed_zero_duration=[review])
    assert len(rows) == 1 and rows[0]['security_id'] == 'new'
    assert rows[0]['valid_from'] == date(2016, 12, 2)
    with pytest.raises(ValueError, match='evidence'):
        build_jl_intervals(['new'], events, '2010-01-01', '2020-01-01',
                           reviewed_zero_duration=[{**review, 'evidence': []}])
    with pytest.raises(ValueError, match='exact same-day'):
        build_jl_intervals(['new'], events[1:], '2010-01-01', '2020-01-01',
                           reviewed_zero_duration=[review])


def test_news_versions_and_opposing_company_scores():
    conn = open_store(":memory:")
    row = dict(
        story_id="story",
        first_created="2020-01-02T15:00:00Z",
        version_created="2020-01-02T15:00:00Z",
        title="A buys B",
        language="en",
        info_source="RTRS",
    )
    store_rows(conn, [row], 1, "A")
    store_rows(conn, [row], 2, "B")
    store_rows(conn, [row], 1, "A")
    updated = dict(row, version_created="2020-01-03T15:00:00Z", title="A cuts guidance")
    store_rows(conn, [updated], 1, "A")
    assert conn.execute("select count(*) from versions").fetchone()[0] == 2
    key = parse_headline(row, datetime.now(timezone.utc))[0]
    conn.execute(
        "insert into scores values (?,?,?,?,?,?,?,?)",
        (key, 1, "v1", -0.8, "negative", 1, "operating", 0),
    )
    conn.execute(
        "insert into scores values (?,?,?,?,?,?,?,?)",
        (key, 2, "v1", 0.9, "positive", 1, "operating", 0),
    )
    aggregates = daily_aggregate(conn, "v1")
    assert {r["ticker_id"]: r["sentiment_sum"] for r in aggregates} == {1: -0.8, 2: 0.9}


def test_news_early_close_and_material_update():
    # Black Friday closes at 13:00 ET, 18:00 UTC.
    assert score_session("2023-11-24T17:59:00Z") == date(2023, 11, 24)
    assert score_session("2023-11-24T18:00:00Z") == date(2023, 11, 27)
    old = normalized_title("UPDATE 1-A raises outlook to 100 million")
    assert is_duplicate("UPDATE 2-A raises outlook to 100 million", [old])
    assert not is_duplicate("A cuts outlook to 100 million", [old])
    assert not is_duplicate("A raises outlook to 110 million", [old])


def test_macro_vintage_not_reference_period_lag():
    rows = parse_alfred(
        {"observations": [dict(date="2020-01-01", realtime_start="2020-02-10", value="2")]}, "DGS10"
    )
    assert rows[0]["available_from"] == date(2020, 2, 11)


def test_capped_news_recursively_refetches():
    from types import SimpleNamespace

    calls = []

    class Definition:
        def __init__(self, **kw):
            self.kw = kw

        def get_data(self):
            calls.append(self.kw)
            wide = self.kw["date_to"] - self.kw["date_from"] > timedelta(days=1)
            return SimpleNamespace(data=SimpleNamespace(headlines=[{}] * (2 if wide else 1)))

    ld = SimpleNamespace(
        content=SimpleNamespace(
            news=SimpleNamespace(headlines=SimpleNamespace(Definition=Definition))
        )
    )
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    assert len(fetch_window(ld, "A", start, start + timedelta(days=2), cap=2)) == 2
    assert len(calls) == 3


def test_lseg_adjustments_and_unverified_refusal():
    from backend.ingestion.prices_lseg import to_price_rows

    dates = [date(2020, 1, 2) + timedelta(days=i) for i in range(8)]
    bars = [
        dict(
            trade_date=d,
            close=100.0 if i < 3 else 50.0 if i < 5 else 49.0,
            open=100.0 if i < 3 else 50.0,
            high=101.0 if i < 3 else 51.0,
            low=99.0 if i < 3 else 48.0,
            volume=10 if i < 3 else 20,
        )
        for i, d in enumerate(dates)
    ]
    with pytest.raises(ValueError, match="verified"):
        to_price_rows(1, bars, {}, {})
    result = to_price_rows(1, bars, {dates[5]: 1.0}, {dates[3]: 2.0}, verified=True)
    assert result[0]["close"] == 50.0
    assert result[0]["adj_close"] == 49.0
    assert result[0]["volume"] == 20
    assert result[5]["adj_close"] == 49.0
    assert as_traded_closes(result)[0] == 100.0
    result = to_price_rows(1, bars, {}, {}, verified=True)
    assert all(r["close"] == r["adj_close"] for r in result)


@pytest.mark.asyncio
async def test_lseg_refuses_mixed_source_security():
    from backend.ingestion.prices_lseg import ingest_verified_prices

    class Connection:
        async def fetchval(self, *_):
            return True

        async def executemany(self, *_):
            pytest.fail("mixed-source data must not be written")

    with pytest.raises(ValueError, match="mixed-source"):
        await ingest_verified_prices(Connection(), [{"ticker_id": 1}])


def test_accounting_parser_preserves_both_quarter_and_ytd():
    from backend.ingestion.research_fundamentals import parse_accounting_facts

    common = dict(form="10-Q", accn="a", filed="2020-08-01", end="2020-06-30")
    payload = {
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            dict(common, start="2020-01-01", val=10),
                            dict(common, start="2020-04-01", val=6),
                        ]
                    }
                }
            }
        }
    }
    rows = parse_accounting_facts(payload, 1)
    assert len(rows) == 2
    assert {r["value"] for r in rows} == {10, 6}


def test_entity_attribution_does_not_assign_whole_story_to_every_company():
    from backend.ingestion.news_lseg import entity_event

    aliases = {
        "1": [{"alias": "Apple", "valid_from": "2000-01-01", "valid_to": None}],
        "2": [{"alias": "Microsoft", "valid_from": "2000-01-01", "valid_to": None}],
    }
    stamp = "2020-01-02T15:00:00Z"
    a = entity_event("Apple raises guidance while Microsoft cuts outlook", 1, aliases, stamp)
    b = entity_event("Apple raises guidance while Microsoft cuts outlook", 2, aliases, stamp)
    assert a["guidance"] == 1 and b["guidance"] == -1
    assert not entity_event("Apple buys Microsoft", 1, aliases, stamp)["relevant"]


def test_day_only_news_is_delayed_conservatively():
    assert score_session("2023-11-24T00:00:00Z") == date(2023, 11, 27)


def test_documented_targets_accept_database_decimal_prices():
    from decimal import Decimal

    prices = [
        dict(trade_date=date(2020, 1, 2), adj_close=Decimal("100")),
        dict(trade_date=date(2020, 1, 3), adj_close=Decimal("100")),
    ]
    event = dict(
        effective_date=date(2020, 1, 6),
        event_type="acquisition",
        verified=True,
        source="filing",
        proceeds_basis="adj_close",
        consideration_type="cash",
        cash_value=Decimal("120"),
    )
    result = documented_targets(prices, 0, [event], date(2021, 2, 1))
    assert result["3M"][0] == pytest.approx(np.log(1.2))
    assert result["3M"][1]


def test_availability_is_the_next_published_vintage_not_a_lag():
    from backend.ingestion.macro import availability

    vintages = [date(2015, 6, 2), date(2015, 6, 3), date(2015, 6, 8)]
    # Friday observation: the next vintage FRED actually published is the Monday.
    assert availability(date(2015, 6, 5), vintages) == date(2015, 6, 8)
    # Same-day publication is never assumed.
    assert availability(date(2015, 6, 2), vintages) == date(2015, 6, 3)


def test_observation_newer_than_every_vintage_is_unavailable():
    from backend.ingestion.macro import availability

    assert availability(date(2026, 9, 5), [date(2026, 9, 3), date(2026, 9, 4)]) is None


def test_observation_before_the_archive_is_reported_unverified():
    from backend.ingestion.macro import availability

    vintages = [date(2014, 1, 27), date(2014, 1, 28)]
    # There is no published vintage covering a 2010 observation, so a caller that
    # requires observed publication dates must exclude it; availability() still
    # returns a real vintage date and never invents one before the archive.
    assert availability(date(2010, 6, 1), vintages) == date(2014, 1, 27)


def test_unregistered_macro_series_is_refused():
    import asyncio

    from backend.ingestion.macro import fetch_vintages

    with pytest.raises(ValueError, match="unregistered"):
        asyncio.run(fetch_vintages(None, "GDP", "key"))


def _action(**over):
    base = {
        "Capital Change Is Rescinded": "False",
        "Capital Change Ex Date": "2013-05-02",
        "Capital Change Effective Date": "2013-05-01",
        "Corporate Change Event Type": "Demerger",
        "Adjustment Type": "LSEG Pricing Only",
        "Adjustment Factor": "0.913357",
        "Terms New Shares": "1.0",
        "Terms Old Shares": "9.0",
    }
    return {**base, **over}


def test_pricing_only_distribution_leaves_the_share_count_alone():
    from backend.ingestion.share_actions import classify_pricing_only

    hit = classify_pricing_only(date(2013, 5, 2), 1.0940, [_action()])
    assert hit is not None
    _, ratio, disagreement = hit
    assert ratio == 1.0 and disagreement < 0.01


def test_pricing_only_rejects_a_price_factor_it_cannot_explain():
    from backend.ingestion.share_actions import classify_pricing_only

    assert classify_pricing_only(date(2013, 5, 2), 2.0, [_action()]) is None


def test_pricing_only_refuses_when_a_share_changing_action_shares_the_date():
    from backend.ingestion.share_actions import classify_pricing_only

    split = _action(**{"Corporate Change Event Type": "Share Split",
                       "Adjustment Type": "Capital Change Type"})
    assert classify_pricing_only(date(2013, 5, 2), 1.0940, [_action(), split]) is None


def test_near_duplicate_actions_do_not_square_the_adjustment():
    from backend.ingestion.share_actions import classify_pricing_only, dedupe_actions

    # The same event from two pulls, terms written '9.0' and '9'.
    pair = [_action(), _action(**{"Terms Old Shares": "9"})]
    assert len(dedupe_actions(pair)) == 1
    assert classify_pricing_only(date(2013, 5, 2), 1.0940, dedupe_actions(pair)) is not None
    # Left undeduplicated the factor squares and the rule must refuse.
    assert classify_pricing_only(date(2013, 5, 2), 1.0940, pair) is None


def test_terms_offset_needs_the_declared_ratio_to_match_exactly():
    from backend.ingestion.share_actions import classify_terms_offset

    split = _action(**{"Corporate Change Event Type": "Share Split",
                       "Adjustment Type": "Capital Change Type",
                       "Capital Change Ex Date": "2018-03-16",
                       "Capital Change Effective Date": "2018-03-16",
                       "Terms New Shares": "2", "Terms Old Shares": "1"})
    assert classify_terms_offset(date(2018, 3, 19), 2.0, [split])[1] == 2.0
    assert classify_terms_offset(date(2018, 3, 19), 2.05, [split]) is None
    # Beyond the window the offset rule must not reach for it.
    assert classify_terms_offset(date(2018, 4, 19), 2.0, [split]) is None
    assert classify_terms_offset(date(2018, 3, 15), 2.0, [split]) is None
    assert classify_terms_offset(date(2018, 3, 16), 2.0, [split]) is None


def test_mixed_event_takes_the_ratio_only_from_the_share_changing_record():
    from backend.ingestion.share_actions import classify_mixed_capital_change

    consolidation = _action(**{"Corporate Change Event Type": "Share Consolidation",
                               "Adjustment Type": "Capital Change Type",
                               "Capital Change Ex Date": "2011-01-04",
                               "Capital Change Effective Date": "2011-01-04",
                               "Adjustment Factor": "7.0",
                               "Terms New Shares": "1.0", "Terms Old Shares": "7"})
    spinoff = _action(**{"Capital Change Ex Date": "2011-01-04",
                         "Capital Change Effective Date": "2011-01-04",
                         "Adjustment Factor": "0.585071",
                         "Terms New Shares": "1.0", "Terms Old Shares": "8"})
    hit = classify_mixed_capital_change(date(2011, 1, 4), 0.2474, [consolidation, spinoff])
    assert hit is not None
    _, ratio, _ = hit
    assert ratio == pytest.approx(1 / 7)
    # Two share-changing records on one date are not resolvable by this rule.
    assert classify_mixed_capital_change(
        date(2011, 1, 4), 0.2474, [consolidation, consolidation, spinoff]
    ) is None
