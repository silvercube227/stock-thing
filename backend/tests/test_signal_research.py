from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from backend.ml.research import (
    append_prospective,
    calendar_block_summary,
    create_snapshot,
    holm_adjust,
    load_snapshot,
    paired_comparison,
    verdict,
)


def test_pairing_checks_controls_but_allows_new_feature_normalization():
    from scripts.audit_paired_coverage import assert_paired_panels
    from scripts.signal_research import paired_control_columns

    baseline = pd.DataFrame(
        dict(date=["2020-01-31"], ticker_id=[1], r_1Y=[0.1], base=[0.2], accruals_assets=[0.07])
    )
    candidate = baseline.copy()
    candidate["accruals_assets"] = -0.5
    controls = paired_control_columns(baseline, ["base"], ["accruals_assets"])
    assert_paired_panels(baseline, candidate, controls)
    assert "base" in controls and "r_1Y" in controls
    candidate["r_1Y"] = 0.2
    with pytest.raises(ValueError, match="Unmatched"):
        assert_paired_panels(baseline, candidate, controls)


def test_audit_json_preserves_decimal_and_serializes_before_creation(tmp_path):
    import json
    from decimal import Decimal
    from backend.ml.research import write_json_new

    target = tmp_path / "audit.json"
    with pytest.raises(TypeError):
        write_json_new(target, {"unsupported": object()})
    assert not target.exists()
    write_json_new(target, {"ratio": Decimal("1.281000000000000001")})
    assert json.loads(target.read_text())["ratio"] == "1.281000000000000001"
    with pytest.raises(FileExistsError):
        write_json_new(target, {"ratio": 2})


def records(n=72):
    rng = np.random.default_rng(1)
    return [
        dict(
            date=d.date(),
            ticker_ids=np.arange(30),
            pred=rng.normal(size=30),
            r=rng.normal(size=30),
            sector=np.array(["s"] * 30),
            size=rng.normal(size=30),
        )
        for d in pd.date_range("2000-01-31", periods=n, freq="ME")
    ]


def test_paired_identity_and_order():
    a = records()
    b = [{k: (v[::-1] if isinstance(v, np.ndarray) else v) for k, v in r.items()} for r in a]
    result = paired_comparison(a, b, "3M", reps=200)
    assert result["paired"]["mean_ic"] == 0
    assert result["paired"]["se_block"] == 0
    assert result["paired"]["p_value"] == 1
    assert result["top_decile_delta"] == 0


@pytest.mark.parametrize("key,value", [("r", 200.0), ("size", 100.0), ("ticker_ids", 1000)])
def test_unmatched_inputs_refused(key, value):
    a = records(1)
    b = [{k: v.copy() if isinstance(v, np.ndarray) else v for k, v in a[0].items()}]
    b[0][key][0] = value
    with pytest.raises(ValueError, match="unmatched"):
        paired_comparison(a, b, "3M", reps=20)


def test_block_gaps_and_too_few_blocks():
    dates = pd.date_range("2000-01-31", periods=36, freq="ME")
    values = np.arange(36, dtype=float)
    assert calendar_block_summary(dates, values, 12, 100)["status"] == "insufficient"
    # A short isolated segment remains a gap; it is never glued to distant dates.
    values[2] = np.nan
    result = calendar_block_summary(dates, values, 3, 100)
    assert result["status"] == "insufficient"
    assert result["n_dates"] == 35
    assert result["p_value"] is None


def test_paired_noise_cancellation():
    rng = np.random.default_rng(2)
    dates = pd.date_range("2000-01-31", periods=120, freq="ME")
    common = rng.normal(0, 0.1, 120)
    delta = 0.005 + rng.normal(0, 0.001, 120)
    absolute = calendar_block_summary(dates, common, 6, 1000)
    paired = calendar_block_summary(dates, (common + delta) - common, 6, 1000)
    assert paired["ci_low"] > 0
    assert paired["se_block"] < absolute["se_block"] / 20


def test_holm_includes_unexecuted_comparisons():
    adjusted = holm_adjust({"news": 0.005, "stress": 0.01})
    assert len(adjusted) == 7
    assert adjusted["news"] == pytest.approx(0.035)
    assert adjusted["stress"] == pytest.approx(0.06)
    assert adjusted["accounting"] == 1


def test_snapshot_archives_code_and_rejects_source_tampering(tmp_path):
    import zipfile

    root = tmp_path / "source-snapshot"
    manifest = create_snapshot(root, [], [], {})
    assert manifest["source_archive_sha256"]
    with zipfile.ZipFile(root / "source.zip") as archive:
        assert "backend/ml/research.py" in archive.namelist()
        assert not any(name.endswith(".env") for name in archive.namelist())
    (root / "source.zip").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="source archive hash"):
        load_snapshot(root)


def test_snapshot_verifies_inputs_and_rejects_overwrite(tmp_path):
    root = tmp_path / "snapshot"
    create_snapshot(root, [], [{"series_id": "x", "value": 1}], {"test": True})
    inputs, _ = load_snapshot(root)
    assert inputs["macro"][0]["value"] == 1
    with pytest.raises(FileExistsError):
        create_snapshot(root, [], [], {})
    (root / "inputs.pkl").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_snapshot(root)


def test_prospective_log_requires_future_entry(tmp_path):
    now = datetime(2026, 9, 7, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="future"):
        append_prospective(tmp_path, {"entry_at": now.isoformat()}, now)
    append_prospective(tmp_path, {"entry_at": (now + timedelta(days=1)).isoformat()}, now)
    with pytest.raises(FileExistsError):
        append_prospective(tmp_path, {"entry_at": (now + timedelta(days=1)).isoformat()}, now)


def test_confirmation_needs_all_practical_checks():
    result = {
        "paired": {"status": "estimated", "mean_ic": 0.004, "ci_low": 0.001},
        "sensitivity": {"status": "estimated", "ci_low": 0.0001},
        "size_neutral_delta": 0.001,
        "top_decile_delta": 0.001,
    }
    assert verdict(result, 0.01, True) == "strong_historical_incremental_evidence"
    result["top_decile_delta"] = -0.001
    assert verdict(result, 0.01, True) == "inconclusive"


def test_new_cache_rejects_legacy_pickle(monkeypatch, tmp_path):
    import asyncio
    import pickle
    from types import SimpleNamespace

    from backend.ml import dataset

    monkeypatch.setattr(dataset, "get_settings", lambda: SimpleNamespace(frame_cache_dir=tmp_path))
    (tmp_path / "frames_all.pkl").write_bytes(pickle.dumps([]))
    with pytest.raises(ValueError, match="incompatible"):
        asyncio.run(dataset.load_frames_cached(None))
    assert dataset._frame_cache_key(None, "sp1500") != dataset._frame_cache_key(None)


@pytest.mark.parametrize("exploratory,low_coverage", [(False, False), (True, False), (True, True)])
def test_registered_runner_writes_replayable_paired_predictions(
    monkeypatch, tmp_path, exploratory, low_coverage
):
    import json
    from types import SimpleNamespace

    from backend.ml.gbm_baseline import FEATURE_COLS, HorizonSpec, LGBMConfig
    from backend.tests.test_gbm_baseline import make_frame
    from scripts import signal_research

    frames = [make_frame(1400, 0.0001 * (i % 5), i, i) for i in range(40)]
    for f in frames:
        f.sector = "Industrials"
        f.membership = [dict(valid_from=f.prices[0]["trade_date"], valid_to=None, index_id="SPX")]
        f.fundamentals = [
            dict(
                filed_at=f.prices[0]["trade_date"],
                period_end=f.prices[0]["trade_date"],
                filing_type="10-K",
                shares_outstanding=1000000,
                shares_measured_at=f.prices[0]["trade_date"],
                shares_kind="point_in_time",
                shares_basis="as_reported",
            )
        ]
        for p in f.prices:
            p.update(close=p["adj_close"], source="yfinance", split_factor=1.0)
    monkeypatch.setitem(
        signal_research.PRODUCTION_HORIZON_SPECS,
        "6M",
        HorizonSpec(
            target_mode="sector_return",
            feature_cols=list(FEATURE_COLS),
            lgb_cfg=LGBMConfig(n_estimators=5, num_leaves=4, min_child_samples=5),
        ),
    )
    source = {
        k: {"status": "verified", "evidence": ["synthetic fixture only"]}
        for k in ("price_share_basis", "membership", "terminal_outcomes")
    }
    snapshot = tmp_path / "snapshot"
    metadata = {"sources": source}
    if exploratory:
        for value in source.values():
            value["status"] = "incomplete"
        metadata["exploratory"] = dict(
            intended_cohort=[dict(ticker_id=f.ticker_id, membership=f.membership) for f in frames],
            exclusions=[],
            limitations=["Synthetic exploratory test fixture"],
        )
        if low_coverage:
            metadata["exploratory"]["intended_cohort"] += [
                dict(ticker_id=-i - 1, membership=frames[0].membership) for i in range(40)
            ]
    create_snapshot(snapshot, frames, [], metadata)
    args = SimpleNamespace(
        snapshot=str(snapshot),
        output=str(tmp_path / "runs"),
        family="stress",
        phase="screening",
        exploratory=exploratory,
    )
    signal_research.run(args)
    directory = tmp_path / "runs"
    if exploratory:
        directory /= "exploratory"
    directory = directory / "stress" / "screening"
    result = json.loads((directory / "result.json").read_text())
    assert result["folds"]
    if exploratory:
        assert result["exploratory"] is True
        assert result["source_certified"] is False
        coverage = json.loads((directory / "exploratory_coverage.json").read_text())
        if low_coverage:
            assert coverage["usable_fraction"] <= 0.5
            assert coverage["status"] == "below_exploratory_target"
        else:
            assert coverage["usable_fraction"] >= 0.9
    assert (directory / "diagnostics.json").exists()
    a = json.loads((directory / "baseline_predictions.json").read_text())
    b = json.loads((directory / "candidate_predictions.json").read_text())
    assert [r["ticker_ids"] for r in a] == [r["ticker_ids"] for r in b]
    assert all(end < "2024-01-01" for rec in a for end in rec["label_end"])
    with pytest.raises(ValueError, match="already exists"):
        signal_research.run(args)
