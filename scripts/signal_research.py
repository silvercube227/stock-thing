"""Snapshot, evaluate, and report the seven registered comparisons.

Examples:
  python -m scripts.signal_research snapshot --output .research/ref --source-report report.json
  python -m scripts.signal_research run --snapshot .research/ref --family stress --output runs
  python -m scripts.signal_research report --output .research/runs
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

from backend.ingestion.db import pool_context
from backend.ml.dataset import build_calendar_grid, load_frames
from backend.ml.factors.long_horizon import BREADTH_FEATURES, PACKS
from backend.ml.gbm_baseline import (
    PRODUCTION_HORIZON_SPECS,
    WalkForwardConfig,
    build_universe_return_map,
    feature_diagnostics,
    prepare_panel,
    walk_forward_ic,
)
from backend.ml.research import (
    PRIMARY_HORIZONS,
    SELECTION_CUTOFFS,
    code_identity,
    coverage_report,
    create_snapshot,
    holm_adjust,
    load_snapshot,
    paired_comparison,
    verdict,
    write_json_new,
)


def source_ready(report, key):
    evidence = report.get(key, {})
    if evidence.get("status") != "verified" or not evidence.get("evidence"):
        raise ValueError(f"blocked: {key} requires verified source evidence")


def paired_control_columns(panel, baseline_features, candidate_features):
    """Unused raw pack columns may become normalized candidate features."""
    transformed_only_in_candidate = set(candidate_features) - set(baseline_features)
    return [
        c for c in panel.columns if c not in {"date", "ticker_id"} | transformed_only_in_candidate
    ]


async def snapshot(args):
    if args.refresh_frame_cache and args.include_research:
        raise ValueError("Production frame-cache refresh requires the production universe")
    source_report = json.loads(Path(args.source_report).read_text())
    # A partial snapshot is useful for audits; run() separately enforces readiness.
    async with pool_context(command_timeout=300) as pool, pool.acquire() as conn:  # noqa: SIM117
        async with conn.transaction(isolation="repeatable_read", readonly=True):
            frames = await load_frames(
                conn,
                index_ids=("SPX", "SP400", "SP600") if args.include_research else ("SPX",),
                include_research=args.include_research,
            )
            import asyncpg

            # A savepoint prevents a missing optional table aborting the snapshot.
            try:
                async with conn.transaction():
                    macro = [dict(r) for r in await conn.fetch("select * from macro_vintages")]
            except asyncpg.UndefinedTableError:
                macro = []
    create_snapshot(
        args.output,
        frames,
        macro,
        {
            "sources": source_report,
            "universe": "sp1500" if args.include_research else "sp500",
            "selection_cutoffs": SELECTION_CUTOFFS,
        },
    )
    if args.refresh_frame_cache:
        from backend.ml.dataset import write_frames_cache

        write_frames_cache(frames)


def select_frames(frames, expanded=False):
    selected = []
    ids = {"SPX", "SP400", "SP600"} if expanded else {"SPX"}
    for frame in frames:
        membership = [r for r in frame.membership or [] if r.get("index_id", "SPX") in ids]
        if membership:
            f = copy.copy(frame)
            f.membership = membership
            selected.append(f)
    return selected


def report_results(path):
    results = {}
    for family in PRIMARY_HORIZONS:
        file = Path(path) / family / "confirmation" / "result.json"
        if file.exists():
            results[family] = json.loads(file.read_text())
    adjusted = holm_adjust({k: v["paired"]["p_value"] for k, v in results.items()})
    return {
        family: {
            "adjusted_p": adjusted[family],
            "verdict": (
                "exploratory_only"
                if results[family].get("exploratory")
                else verdict(results[family], adjusted[family], True)
            )
            if family in results
            else "unexecuted",
        }
        for family in PRIMARY_HORIZONS
    }


def run(args):
    inputs, manifest = load_snapshot(args.snapshot)
    sources = manifest["metadata"]["sources"]
    exploratory = getattr(args, "exploratory", False)
    if not exploratory:
        for foundation in ("price_share_basis", "membership", "terminal_outcomes"):
            source_ready(sources, foundation)
    family, horizon = args.family, PRIMARY_HORIZONS[args.family]
    if not exploratory and family not in ("stress", "combined"):
        source_ready(sources, family)
    if not exploratory and family == "universe":
        source_ready(sources, "historical_sectors")
    cols = list(PACKS.get(family, []))
    if family == "analyst" and sources["analyst"].get("contributor_history") is True:
        cols += BREADTH_FEATURES
    if family == "combined" and exploratory:
        packs = getattr(args, "combined_packs", None)
        if not packs or len(set(packs)) < 2:
            raise ValueError("Exploratory combined run requires explicit preselected packs")
        cols = [c for k in sorted(set(packs)) for c in PACKS[k]]
    elif family == "combined":
        evidence = report_results(args.output)
        winners = [
            k
            for k in ("news", "stress", "macro")
            if evidence[k]["verdict"]
            in ("historical_incremental_evidence", "strong_historical_incremental_evidence")
        ]
        if len(winners) < 2:
            raise ValueError("combined comparison requires at least two supported 6M feature packs")
        cols = [c for k in winners for c in PACKS[k]]
    root = Path(args.output) / "exploratory" if exploratory else Path(args.output)
    output = root / family / args.phase
    if output.exists():
        raise ValueError("registered comparison already exists; do not overwrite/retry silently")
    if args.phase == "confirmation":
        screening = json.loads((output.parent / "screening" / "result.json").read_text())
        if verdict(screening, 1.0) != "confirm":
            raise ValueError("single-seed screen did not earn confirmation")
        registration = json.loads((output.parent / "screening" / "manifest.json").read_text())
        if registration["input_sha256"] != manifest["input_sha256"]:
            raise ValueError("screening and confirmation snapshots differ")
        if registration["feature_cols"] != cols:
            raise ValueError("feature subset changed after screening")
        if (
            registration["code"]["implementation_sha256"]
            != code_identity()["implementation_sha256"]
        ):
            raise ValueError(
                "implementation changed after screening; record a new program revision"
            )
    spec = PRODUCTION_HORIZON_SPECS[horizon]
    base_cols = list(spec.feature_cols)
    frames = select_frames(inputs["frames"])
    expanded = select_frames(inputs["frames"], expanded=True) if family == "universe" else frames
    if not frames:
        raise ValueError("no point-in-time S&P 500 frames")
    market = build_universe_return_map(frames, membership_filter=True)
    grid = build_calendar_grid(frames)
    print(f"Preparing {family}/{horizon} baseline panel ({len(frames)} securities)", flush=True)
    baseline = prepare_panel(
        frames, grid, rank_cols=base_cols, membership_filter=True, market_returns_override=market
    )
    print(f"Preparing candidate panel ({len(expanded)} securities)", flush=True)
    candidate = prepare_panel(
        expanded,
        grid,
        rank_cols=base_cols + cols,
        membership_filter=True,
        macro=inputs["macro"],
        market_returns_override=market,
        allow_provisional_estimates=exploratory and family == "analyst",
        allow_provisional_news=exploratory and bool(set(cols) & set(PACKS["news"])),
    )
    for c in cols:
        if c not in candidate or not np.isfinite(candidate[c].to_numpy(dtype=float)).any():
            raise ValueError(f"blocked: no observed values for {c}")
    exploratory_report = None
    if exploratory:
        from backend.ml.exploratory import exploratory_coverage
        from scripts.audit_paired_coverage import assert_paired_panels

        if family != "universe":
            controls = paired_control_columns(baseline, base_cols, cols)
            assert_paired_panels(baseline, candidate, controls)
        exploratory_report = exploratory_coverage(baseline, manifest["metadata"], horizon)
        write_json_new(
            output.parent / f"{args.phase}_exploratory_preflight.json", exploratory_report
        )
        if exploratory_report["usable_rows"] == 0:
            raise ValueError("No usable exploratory evaluation rows")
        print(
            f"Exploratory coverage: {exploratory_report['usable_fraction']:.2%} "
            f"({exploratory_report['status']})",
            flush=True,
        )
    # Widening changes training ranks/targets. Evaluation stays on the original
    # S&P 500 outcomes, sectors, and size control, attached AFTER predictions.
    control = baseline.set_index(["date", "ticker_id"])
    n_seeds = 8 if args.phase == "confirmation" else 1
    cfg = spec.lgb_cfg if n_seeds == 8 else replace(spec.lgb_cfg, objective="regression")
    common = dict(
        horizon=horizon,
        lgb_cfg=cfg,
        wf_cfg=WalkForwardConfig(max_train_months=spec.max_train_months),
        seed=1337,
        target_mode=spec.target_mode,
        n_seeds=n_seeds,
        compute_sector_ic=True,
        smooth_span=spec.smooth_span,
        knife_lambda=spec.knife_lambda,
        winsorize_pct=spec.winsorize_pct,
        vol_gate=spec.vol_gate,
        max_test_date=SELECTION_CUTOFFS[horizon],
        label_realized_before=date(2024, 1, 1),
        return_records=True,
        log=print,
    )
    output.mkdir(parents=True)
    if exploratory_report:
        write_json_new(output / "exploratory_coverage.json", exploratory_report)
    write_json_new(
        output / "coverage.json",
        {
            "baseline": coverage_report(baseline, base_cols),
            "candidate": coverage_report(candidate, base_cols + cols),
        },
    )
    selection = candidate[
        (candidate["date"] <= SELECTION_CUTOFFS[horizon])
        & (candidate[f"label_end_{horizon}"] < date(2024, 1, 1))
    ]
    if cols:
        diagnostics = feature_diagnostics(
            selection, horizon, candidate_cols=cols, existing_cols=base_cols
        )
        write_json_new(output / "diagnostics.json", diagnostics)
    write_json_new(
        output / "manifest.json",
        {
            "registered_at": datetime.now(UTC),
            "code": code_identity(),
            "input_sha256": manifest["input_sha256"],
            "family": family,
            "primary_horizon": horizon,
            "phase": args.phase,
            "exploratory": exploratory,
            "source_statuses": {k: v.get("status") for k, v in sources.items()},
            "feature_cols": cols,
            "baseline_spec": spec,
            "fit_configuration": {k: v for k, v in common.items() if k != "log"},
            "registry": PRIMARY_HORIZONS,
            "coverage": {c: float(candidate[c].notna().mean()) for c in cols},
        },
    )
    a = walk_forward_ic(baseline, feature_cols=base_cols, **common)
    b = walk_forward_ic(candidate, feature_cols=base_cols + cols, **common)
    if family == "universe":
        write_json_new(output / "expanded_universe_predictions.json", b["prediction_records"])
        write_json_new(output / "expanded_universe_summary.json", b["sector_summary"])
        by_date = {r["date"]: r for r in b["prediction_records"]}
        restricted = []
        for original in a["prediction_records"]:
            rec = by_date.get(original["date"])
            if rec is None:
                raise ValueError("expanded model omitted an original evaluation date")
            positions = {int(tid): i for i, tid in enumerate(rec["ticker_ids"])}
            if not set(map(int, original["ticker_ids"])) <= set(positions):
                raise ValueError("expanded model omitted an original evaluation security")
            idx = [positions[int(tid)] for tid in original["ticker_ids"]]
            restricted.append(
                {
                    k: (v[idx] if isinstance(v, np.ndarray) else v)
                    for k, v in rec.items()
                    if k != "risk"
                }
            )
        b["prediction_records"] = restricted
    for result in (a, b):
        for rec in result["prediction_records"]:
            lookup = control.loc[[(rec["date"], int(tid)) for tid in rec["ticker_ids"]]]
            rec["r"] = lookup[f"r_{horizon}"].to_numpy(dtype=float)
            rec["size"] = lookup["log_market_cap"].to_numpy(dtype=float)
            rec["sector"] = lookup["sector"].to_numpy()
    write_json_new(output / "baseline_predictions.json", a["prediction_records"])
    write_json_new(output / "candidate_predictions.json", b["prediction_records"])
    comparison = paired_comparison(a["prediction_records"], b["prediction_records"], horizon)
    if exploratory:
        comparison.update(
            exploratory=True,
            source_certified=False,
            interpretation=(
                "Hypothesis exploration on frozen provisional inputs; "
                "no supported-improvement or production-promotion claim."
            ),
        )
    write_json_new(output / "result.json", comparison)
    print(
        json.dumps(
            comparison if exploratory else report_results(args.output), default=str, indent=2
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--output", required=True)
    snap.add_argument("--source-report", required=True)
    snap.add_argument("--include-research", action="store_true")
    snap.add_argument("--refresh-frame-cache", action="store_true")
    fit = sub.add_parser("run")
    fit.add_argument("--snapshot", required=True)
    fit.add_argument("--family", choices=PRIMARY_HORIZONS, required=True)
    fit.add_argument("--phase", choices=("screening", "confirmation"), default="screening")
    fit.add_argument("--output", required=True)
    fit.add_argument("--exploratory", action="store_true")
    fit.add_argument("--combined-packs", nargs="+", choices=("news", "stress", "macro"))
    report = sub.add_parser("report")
    report.add_argument("--output", required=True)
    args = p.parse_args()
    if args.command == "snapshot":
        asyncio.run(snapshot(args))
    elif args.command == "run":
        run(args)
    else:
        print(json.dumps(report_results(args.output), indent=2))


if __name__ == "__main__":
    main()
