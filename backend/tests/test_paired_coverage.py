from datetime import date

import numpy as np
import pandas as pd
import pytest

from backend.ml.gbm_baseline import WalkForwardConfig, select_walk_forward_samples
from scripts.audit_paired_coverage import assert_paired_panels, calendar_support


def sample_panel():
    dates = [date(2020, m, 28) for m in range(1, 7)]
    return pd.DataFrame(dict(date=dates, ticker_id=[1]*6, mask_6M=[True]*6,
        y_6M_sector_grade=[1, np.nan, 2, 3, 2, 1],
        label_end_6M=[date(2020, 5, 1)]*4+[date(2020, 6, 28), date(2023, 12, 29)],
        entry_6M=[d.replace(day=29) for d in dates], macro=[np.nan]*6))


def test_shared_selection_uses_labeled_dates_and_strict_realization():
    p = sample_panel()
    train, test = select_walk_forward_samples(p, '6M', 'sector_grade', date(2020, 6, 28),
        date(2020, 5, 28), WalkForwardConfig(max_train_months=2), date(2024, 1, 1))
    assert list(train.date) == [date(2020, 3, 28), date(2020, 4, 28)]
    assert list(test.date) == [date(2020, 6, 28)]
    assert test.macro.isna().all()  # Source gaps are disclosed, not silently dropped.
    p.loc[5, 'label_end_6M'] = date(2024, 1, 1)
    _, test = select_walk_forward_samples(p, '6M', 'sector_grade', date(2020, 6, 28),
        date(2020, 5, 28), WalkForwardConfig(), date(2024, 1, 1))
    assert test.empty


def test_shared_selection_rejects_entry_at_cutoff():
    p = sample_panel()
    p.loc[5, 'entry_6M'] = p.loc[5, 'date']
    with pytest.raises(ValueError, match='entry must follow'):
        select_walk_forward_samples(p, '6M', 'sector_grade', date(2020, 6, 28),
                                    date(2020, 5, 28), WalkForwardConfig())


def test_paired_audit_rejects_missing_names_and_changed_controls():
    baseline = sample_panel()
    columns = [c for c in baseline if c not in ('date', 'ticker_id')]
    assert_paired_panels(baseline, baseline.iloc[::-1], columns)
    with pytest.raises(ValueError, match='Unmatched'):
        assert_paired_panels(baseline, baseline.iloc[:-1], columns)
    changed = baseline.copy()
    changed.loc[0, 'label_end_6M'] = date(2021, 1, 1)
    with pytest.raises(ValueError, match='Unmatched'):
        assert_paired_panels(baseline, changed, columns)
    with pytest.raises(ValueError, match='Duplicate'):
        assert_paired_panels(baseline, pd.concat([baseline, baseline.iloc[:1]]), columns)


def test_calendar_support_never_glues_isolated_segments():
    months = pd.date_range('2010-01-31', periods=72, freq='ME')
    assert calendar_support(months, 12)['calendar_sufficient']
    gap = months.delete(2)
    report = calendar_support(gap, 12)
    assert report['contiguous_segments'] == [2, 69]
    assert not report['calendar_sufficient']
    assert not report['inference_computed']
