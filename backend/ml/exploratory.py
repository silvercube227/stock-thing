"""Coverage evidence for explicitly exploratory, frozen paired experiments."""

import math
from datetime import date

from backend.ingestion.calendar import HORIZON_TRADING_DAYS, shift_trading_days
from backend.ml.gbm_baseline import (
    PRODUCTION_HORIZON_SPECS,
    WalkForwardConfig,
    select_walk_forward_samples,
    walk_forward_folds,
)
from backend.ml.research import SELECTION_CUTOFFS


def member(intervals, day):
    return any(
        date.fromisoformat(str(r["valid_from"])) <= day
        and (r["valid_to"] is None or day < date.fromisoformat(str(r["valid_to"])))
        for r in intervals
        if r.get("index_id", "SPX") == "SPX"
    )


def exploratory_coverage(panel, metadata, horizon):
    policy = metadata.get("exploratory")
    if not policy or not policy.get("intended_cohort") or not policy.get("limitations"):
        raise ValueError("Exploratory runs require frozen cohort, exclusions and limitations")
    spec = PRODUCTION_HORIZON_SPECS[horizon]
    cfg = WalkForwardConfig(max_train_months=spec.max_train_months)
    steps = HORIZON_TRADING_DAYS[horizon]
    cohort = policy["intended_cohort"]
    if len({r["ticker_id"] for r in cohort}) != len(cohort):
        raise ValueError("Duplicate intended cohort identity")
    rows = []
    anchor = date(2023, 9, 29)
    observation_end = date.fromisoformat(str(policy.get("observation_end", max(panel.date))))
    for day, cutoff in walk_forward_folds(
        sorted(panel.date.unique()), cfg.min_train_months, max(1, math.ceil(steps / 21))
    ):
        if day > SELECTION_CUTOFFS[horizon]:
            continue
        realization = shift_trading_days(shift_trading_days(day, 1), steps)
        if realization >= date(2024, 1, 1) or realization > observation_end:
            continue
        train, test = select_walk_forward_samples(
            panel, horizon, spec.target_mode, day, cutoff, cfg, date(2024, 1, 1)
        )
        used = (
            set(map(int, test.ticker_id))
            if not train.empty and len(test) >= cfg.min_names
            else set()
        )
        intended = {r["ticker_id"]: r for r in cohort if member(r["membership"], day)}
        if not used <= intended.keys():
            raise ValueError("Usable sample contains identities outside frozen intended cohort")
        groups = {}
        for label, surviving in (("surviving", True), ("removed", False)):
            ids = {
                tid for tid, r in intended.items() if member(r["membership"], anchor) == surviving
            }
            groups[label] = dict(
                intended=len(ids), usable=len(ids & used), excluded=len(ids - used)
            )
        rows.append(
            dict(
                date=day,
                intended=len(intended),
                usable=len(used),
                excluded=len(intended) - len(used),
                by_cohort=groups,
            )
        )
    total = sum(r["intended"] for r in rows)
    usable = sum(r["usable"] for r in rows)
    coverage = usable / total if total else 0
    return dict(
        status="acceptable_exploratory_coverage" if coverage >= 0.9 else "below_exploratory_target",
        usable_fraction=coverage,
        target=0.95,
        acceptable_reference=0.9,
        intended_rows=total,
        usable_rows=usable,
        folds=rows,
        denominator=(
            "Frozen reconstructed SPX membership before exclusions, including unmapped securities."
        ),
        exclusions=policy.get("exclusions", []),
        limitations=policy["limitations"],
        cohort_anchor=str(anchor),
        source_certification=False,
    )
