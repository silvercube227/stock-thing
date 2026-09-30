from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from backend.ml.dataset import TickerFrame
from backend.ml.factors.long_horizon import (
    MACRO_FEATURES,
    STRESS_FEATURES,
    accounting_features,
    add_macro_features,
    fixed_estimate_features,
    news_features,
)
from backend.ml.gbm_baseline import (
    FEATURE_COLS,
    PRODUCTION_HORIZON_SPECS,
    prepare_panel,
    vol_gate_flags,
)


def test_fixed_period_rollover_and_basis_changes():
    rows = [
        dict(
            as_of_date=d,
            fiscal_period_end=date(2021, 12, 31),
            contributor_id="consensus",
            eps=v,
            adjustment_basis="basis1",
            verified=True,
        )
        for d, v in zip(
            (date(2020, 1, 31), date(2020, 2, 29), date(2020, 3, 31), date(2020, 4, 30)),
            (1.0, 2.0, 3.0, 4.0),
        )
    ]
    result = fixed_estimate_features(rows, [date(2020, 4, 30)])[0]
    assert result["fixed_eps_rev_30d"] == pytest.approx(1 / 3)
    assert result["fixed_eps_persistence"] == 1
    rows[-1]["fiscal_period_end"] = date(2022, 12, 31)
    assert np.isnan(fixed_estimate_features(rows, [date(2020, 4, 30)])[0]["fixed_eps_rev_30d"])
    rows[-1]["fiscal_period_end"] = date(2021, 12, 31)
    rows[-1]["adjustment_basis"] = "basis2"
    assert np.isnan(fixed_estimate_features(rows, [date(2020, 4, 30)])[0]["fixed_eps_rev_30d"])


def test_breadth_uses_only_continuing_analysts():
    rows = []
    for contributor in ["consensus", "a", "b", "c", "d", "e"]:
        for d, eps in [(date(2020, 1, 31), 1.0), (date(2020, 4, 30), 2.0)]:
            rows.append(
                dict(
                    as_of_date=d,
                    fiscal_period_end=date(2021, 12, 31),
                    contributor_id=contributor,
                    eps=eps,
                    adjustment_basis="same",
                    verified=True,
                )
            )
    rows.append(dict(rows[-1], contributor_id="new", eps=-100))
    assert fixed_estimate_features(rows, [date(2020, 4, 30)])[0]["fixed_eps_revision_breadth"] == 1


def accounting_fixture():
    rows = []
    for end, value in [(date(2019, 12, 31), 100.0), (date(2020, 12, 31), 120.0)]:
        rows.append(
            dict(
                accession_number=str(end),
                filed_at=end + timedelta(days=40),
                period_start=None,
                period_end=end,
                metric="assets",
                value=value,
            )
        )
    for metric, values in (
        ("net_income", [5.0, 11.0, 18.0, 26.0]),
        ("operating_cash_flow", [4.0, 9.0, 15.0, 22.0]),
        ("gross_profit", [10.0, 22.0, 35.0, 50.0]),
    ):
        for end, value in zip(
            (date(2020, 3, 31), date(2020, 6, 30), date(2020, 9, 30), date(2020, 12, 31)), values
        ):
            rows.append(
                dict(
                    accession_number=str(end),
                    filed_at=end + timedelta(days=40),
                    period_start=date(2020, 1, 1),
                    period_end=end,
                    metric=metric,
                    value=value,
                )
            )
    return rows


def test_accounting_ytd_original_filings_and_financials():
    facts = accounting_fixture()
    result = accounting_features(facts, [date(2021, 3, 1)], ["Industrials"])[0]
    assert result["accruals_assets"] == pytest.approx(4 / 110)
    assert result["gross_profit_assets"] == pytest.approx(50 / 110)
    assert result["asset_growth_yoy"] == pytest.approx(0.2)
    facts.append(dict(facts[-1], accession_number="later", filed_at=date(2022, 1, 1), value=1000.0))
    assert accounting_features(facts, [date(2021, 3, 1)], ["Industrials"])[0] == result
    assert np.isnan(
        list(accounting_features(facts, [date(2021, 3, 1)], ["Financials"])[0].values())
    ).all()
    assert np.isnan(
        accounting_features(facts, [date(2021, 1, 1)], ["Industrials"])[0]["asset_growth_yoy"]
    )


@pytest.mark.parametrize('sector', [None, '', 'Unknown', float('nan'), 'Financial Services'])
def test_accounting_requires_known_nonfinancial_sector(sector):
    result = accounting_features(accounting_fixture(), [date(2021, 3, 1)], [sector])[0]
    assert np.isnan(list(result.values())).all()


def test_accounting_sector_gap_masks_only_the_unclassified_date():
    results = accounting_features(accounting_fixture(), [date(2021, 3, 1)] * 3,
                                  ['Industrials', None, 'Industrials'])
    assert results[0] == results[2]
    assert np.isfinite(list(results[0].values())).all()
    assert np.isnan(list(results[1].values())).all()


def test_accounting_missing_flow_is_not_zero():
    facts = [r for r in accounting_fixture() if r["metric"] != "operating_cash_flow"]
    result = accounting_features(facts, [date(2021, 3, 1)], ["Industrials"])[0]
    assert np.isnan(result["accruals_assets"])
    assert result["gross_profit_assets"] > 0


def test_news_zero_events_differs_from_missing_coverage():
    d = date(2020, 4, 1)
    coverage = [dict(start_date=date(2020, 1, 1), end_date=d, status="complete")]
    row = dict(
        score_date=d,
        scorer_revision="fixed",
        n_events=2,
        sentiment_sum=0.8,
        n_negative=1,
        guidance_up=1,
        guidance_down=0,
        operating_n=0,
        operating_sentiment_sum=0.0,
    )
    features = news_features([row], coverage, [d])[0]
    assert features["news_sent_90d"] == 0.4
    assert features["news_guidance_balance_90d"] == 1
    assert np.isnan(features["news_operating_sent_90d"])
    assert np.isnan(list(news_features([row], [], [d])[0].values())).all()
    assert np.isnan(list(news_features([], coverage, [d])[0].values())).all()


def test_stress_products_are_post_normalization_and_causal():
    from backend.ml.dataset import build_calendar_grid
    from backend.tests.test_gbm_baseline import make_frame

    frames = [make_frame(1000, 0.0001 * i, i, i) for i in range(4)]
    grid = build_calendar_grid(frames)
    panel = prepare_panel(frames, grid, rank_cols=FEATURE_COLS + STRESS_FEATURES)
    medians = panel.groupby("date").vol_120d_raw.median().sort_index()
    flags = dict(zip(medians.index, vol_gate_flags(medians.tolist()), strict=True))
    for c, mom in zip(STRESS_FEATURES, ("mom_3m", "mom_12_1")):
        np.testing.assert_allclose(panel[c], panel.date.map(flags) * panel[mom])
    assert not set(STRESS_FEATURES) & set(PRODUCTION_HORIZON_SPECS["6M"].feature_cols)


def test_macro_future_vintage_and_future_price_do_not_change_features():
    rng = np.random.default_rng(3)
    dates = pd.date_range("2005-01-31", periods=100, freq="ME")
    m = rng.normal(0, 0.02, len(dates))
    factor = rng.normal(0, 0.1, len(dates))
    market = {d.date(): r for d, r in zip(dates, m)}
    macro = [
        dict(
            series_id="DGS10",
            source="alfred_realtime_v2",
            obs_date=d.date(),
            available_from=d.date(),
            vintage_date=d.date(),
            value=5 + v,
            verified=True,
        )
        for d, v in zip(dates, factor.cumsum())
    ]
    frames = []
    for tid, beta in enumerate((-0.1, 0.1, 0.2)):
        prices = np.exp(np.cumsum(0.6 * m + beta * factor)) * 100
        frames.append(
            TickerFrame(
                tid,
                tid,
                str(tid),
                [dict(trade_date=d.date(), adj_close=p) for d, p in zip(dates, prices)],
                [],
                [],
            )
        )
    asof = dates[90].date()
    panel = pd.DataFrame(dict(date=[asof] * 3, ticker_id=[0, 1, 2]))
    result = add_macro_features(panel, frames, macro, market)
    assert result[MACRO_FEATURES[0]].notna().all()
    assert len(set(result[MACRO_FEATURES[0]])) == 3
    macro.append(
        dict(macro[0], vintage_date=date(2026, 1, 1), available_from=date(2026, 1, 1), value=10000)
    )
    frames[0].prices[-1]["adj_close"] *= 1000
    pd.testing.assert_frame_equal(result, add_macro_features(panel, frames, macro, market))
    assert result[MACRO_FEATURES[1]].isna().all()
    legacy = [dict(r, source='alfred_vintage') for r in macro]
    assert add_macro_features(panel, frames, legacy, market)[MACRO_FEATURES[0]].isna().all()
    # A missing-value revision to the latest monthly point must not resurrect
    # the old value; the remaining last point is too stale for a current shock.
    withdrawn = macro + [dict(macro[90], vintage_date=asof, available_from=asof, value=None)]
    assert add_macro_features(panel, frames, withdrawn, market)[MACRO_FEATURES[0]].isna().all()
