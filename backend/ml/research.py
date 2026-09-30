"""Frozen long-horizon registry, matched evaluation, and local research artifacts.

No database writes or production promotions live here. Snapshots contain all
inputs, including macro vintages; a cache is never treated as a snapshot.
"""

from __future__ import annotations

import hashlib
import gzip
import json
import math
import pickle
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
import io
import zipfile
from pathlib import Path
from backend.ml.compressed_io import open_artifact

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
PRIMARY_HORIZONS = {
    "analyst": "3M",
    "news": "6M",
    "accounting": "1Y",
    "stress": "6M",
    "macro": "6M",
    "universe": "6M",
    "combined": "6M",
}
SELECTION_CUTOFFS = {"3M": date(2023, 9, 30), "6M": date(2023, 6, 30), "1Y": date(2022, 12, 31)}
BLOCK_MONTHS = {"3M": 3, "6M": 6, "1Y": 12}
MIN_DELTA_IC = 0.003


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, (date, datetime, Path)):
        return str(value)
    if isinstance(value, Decimal):
        return str(value)  # preserve exact PostgreSQL numerics in audit artifacts
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json_new(path, value):
    """Exclusive creation: rerunning cannot overwrite a registered result."""
    path = Path(path)
    serialized = json.dumps(json_value(value), sort_keys=True, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        f.write(serialized)


def code_identity():
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    # Include uncommitted implementation files, but never credentials or data.
    digest = hashlib.sha256()
    for folder in ("backend", "scripts"):
        for p in sorted((root / folder).rglob("*")):
            if p.suffix in (".py", ".sql"):
                digest.update(str(p.relative_to(root)).encode())
                digest.update(p.read_bytes())
    return {"revision": revision, "implementation_sha256": digest.hexdigest()}


def create_snapshot(path, frames, macro, metadata):
    """Create a new, content-verified snapshot. Pickles are LOCAL trusted inputs."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    payload = pickle.dumps({"frames": frames, "macro": macro}, protocol=5)
    with gzip.open(path / "inputs.pkl", 'wb', compresslevel=6) as stream:
        stream.write(payload)
    root = Path(__file__).resolve().parents[2]
    source_files = [p for folder in ('backend', 'scripts') for p in (root/folder).rglob('*')
                    if p.is_file() and p.suffix in ('.py', '.sql')]
    source_files += [root/p for p in ('pyproject.toml', 'requirements.txt', 'backend/requirements.txt')
                     if (root/p).is_file()]
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for source in sorted(source_files):
            info = zipfile.ZipInfo(str(source.relative_to(root)))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, source.read_bytes())
    source_payload = archive.getvalue()
    (path/'source.zip').write_bytes(source_payload)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc),
        "code": code_identity(),
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "source_archive_sha256": hashlib.sha256(source_payload).hexdigest(),
        "registry": PRIMARY_HORIZONS,
        "metadata": metadata,
        "n_securities": len(frames),
    }
    write_json_new(path / "manifest.json", manifest)
    return manifest


def coverage_report(panel, features=()):
    rows = []
    if panel.empty:
        return rows
    for year, group in panel.groupby(pd.to_datetime(panel["date"]).dt.year):
        item = {
            "year": int(year),
            "rows": len(group),
            "securities": int(group.ticker_id.nunique()),
            "dates": int(group.date.nunique()),
            "features": {c: float(group[c].notna().mean()) for c in features if c in group},
        }
        for h in BLOCK_MONTHS:
            if f"mask_{h}" in group:
                item[h] = {
                    "labeled_rows": int(group[f"mask_{h}"].sum()),
                    "outcomes": group[f"outcome_{h}"].value_counts().to_dict()
                    if f"outcome_{h}" in group
                    else {},
                }
        if "removed_cohort" in group:
            item["availability_by_cohort"] = {
                "removed" if removed else "surviving": {
                    c: float(cohort[c].notna().mean()) for c in features if c in cohort
                }
                for removed, cohort in group.groupby("removed_cohort")
            }
        rows.append(item)
    return rows


def load_snapshot(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("incompatible research snapshot; create a fresh snapshot")
    if manifest.get('source_archive_sha256'):
        if hashlib.sha256((path/'source.zip').read_bytes()).hexdigest() != manifest['source_archive_sha256']:
            raise ValueError('snapshot source archive hash mismatch')
    with open_artifact(path / "inputs.pkl") as stream:
        payload = stream.read()
    if hashlib.sha256(payload).hexdigest() != manifest["input_sha256"]:
        raise ValueError("snapshot input hash mismatch")
    return pickle.loads(payload), manifest


def calendar_block_summary(dates, values, block_size, reps=10000, seed=1337):
    """Moving blocks never jump a missing month. Undefined dates remain gaps."""
    pairs = sorted(zip(dates, values, strict=True), key=lambda p: pd.Timestamp(p[0]))
    months = [pd.Timestamp(d).year * 12 + pd.Timestamp(d).month for d, _ in pairs]
    if len(set(months)) != len(months):
        raise ValueError("one evaluation observation per calendar month required")
    if block_size < 1 or reps < 2:
        raise ValueError("positive block size and at least two replicates required")
    segments, current = [], []
    prev = None
    for month, (_, value) in zip(months, pairs, strict=True):
        if not np.isfinite(value) or (prev is not None and month != prev + 1):
            if current:
                segments.append(np.asarray(current))
            current = []
        if np.isfinite(value):
            current.append(float(value))
        prev = month
    if current:
        segments.append(np.asarray(current))
    a = np.concatenate(segments) if segments else np.array([])
    n = len(a)
    out = {
        "n_dates": n,
        "block_months": block_size,
        "effective_blocks": n / block_size,
        "mean_ic": float(a.mean()) if n else float("nan"),
        "ci_low": None,
        "ci_high": None,
        "p_value": None,
        "se_block": None,
        "approx_95pct_threshold_ic": None,
        "status": "insufficient",
    }
    # Short isolated segments cannot be stretched into full calendar blocks.
    if n / block_size < 5 or any(len(s) < block_size for s in segments):
        return out
    rng = np.random.default_rng(seed)
    boots = np.zeros(reps)
    # Resample each contiguous segment at its original weight, preserving gaps.
    for segment in segments:
        starts = rng.integers(
            0, len(segment) - block_size + 1, size=(reps, math.ceil(len(segment) / block_size))
        )
        idx = (starts[:, :, None] + np.arange(block_size)).reshape(reps, -1)
        boots += segment[idx[:, : len(segment)]].sum(axis=1) / n
    centered = boots - a.mean()
    se = float(boots.std(ddof=1))
    out.update(
        ci_low=float(np.quantile(boots, 0.025)),
        ci_high=float(np.quantile(boots, 0.975)),
        p_value=float((1 + (np.abs(centered) >= abs(a.mean())).sum()) / (reps + 1)),
        se_block=se,
        approx_95pct_threshold_ic=1.96 * se,
        status="estimated",
    )
    return out


def _metrics(pred, realized, sectors, size, min_names=10):
    df = pd.DataFrame({"p": pred, "r": realized, "s": sectors, "size": size})
    ics, partials, tops = [], [], []
    for _, g in df.dropna(subset=["s"]).groupby("s"):
        if len(g) < min_names:
            continue
        ic = g.p.corr(g.r, method="spearman")
        if np.isfinite(ic):
            ics.append(ic)
        # Average tied selections fractionally through percentile weights.
        k = max(1, math.ceil(0.1 * len(g)))
        cut = g.p.nlargest(k).iloc[-1]
        above, tied = g.p > cut, g.p == cut
        w = above.astype(float) + tied.astype(float) * ((k - above.sum()) / tied.sum())
        tops.append(float((w * g.r).sum() / w.sum() - g.r.mean()))
        observed = g.dropna(subset=["size"])
        if len(observed) >= min_names:
            ranks = observed[["p", "r", "size"]].rank().to_numpy()
            x = np.column_stack([np.ones(len(ranks)), ranks[:, 2]])
            residual = ranks[:, :2] - x @ np.linalg.lstsq(x, ranks[:, :2], rcond=None)[0]
            if np.all(residual.std(axis=0) > 1e-10):
                partials.append(float(np.corrcoef(residual.T)[0, 1]))
    mean = lambda a: float(np.mean(a)) if a else float("nan")
    return {
        "sector_ic": mean(ics),
        "size_neutral_ic": mean(partials),
        "sector_top_decile_excess": mean(tops),
    }


def paired_comparison(baseline, candidate, horizon, reps=10000):
    """Reject unmatched predictions instead of intersecting away missing names."""
    if len(baseline) != len(candidate):
        raise ValueError("unmatched fold count")
    dates, delta, absolute_a, absolute_b, rows = [], [], [], [], []
    for a, b in zip(baseline, candidate, strict=True):
        if str(a["date"]) != str(b["date"]):
            raise ValueError("unmatched fold dates")
        ai, bi = np.argsort(a["ticker_ids"]), np.argsort(b["ticker_ids"])
        if len(set(a["ticker_ids"])) != len(ai) or len(set(b["ticker_ids"])) != len(bi):
            raise ValueError("duplicate security in predictions")
        for key in ("ticker_ids", "r", "sector", "size", "entry_date", "label_end"):
            if key not in a and key not in b and key in ("entry_date", "label_end"):
                continue
            av, bv = np.asarray(a[key])[ai], np.asarray(b[key])[bi]
            equal = (
                np.array_equal(av, bv, equal_nan=True)
                if av.dtype.kind in "fci"
                else np.array_equal(av, bv)
            )
            if not equal:
                raise ValueError(f"unmatched {key} on {a['date']}")
        if not np.isfinite(a["pred"]).all() or not np.isfinite(b["pred"]).all():
            raise ValueError("nonfinite prediction")
        ma = _metrics(
            np.asarray(a["pred"])[ai],
            np.asarray(a["r"])[ai],
            np.asarray(a["sector"])[ai],
            np.asarray(a["size"])[ai],
        )
        mb = _metrics(
            np.asarray(b["pred"])[bi],
            np.asarray(b["r"])[bi],
            np.asarray(b["sector"])[bi],
            np.asarray(b["size"])[bi],
        )
        rows.append(
            {
                "date": a["date"],
                "baseline": ma,
                "candidate": mb,
                "delta": {k: mb[k] - ma[k] for k in ma},
            }
        )
        dates.append(a["date"])
        absolute_a.append(ma["sector_ic"])
        absolute_b.append(mb["sector_ic"])
        delta.append(mb["sector_ic"] - ma["sector_ic"])
    block = BLOCK_MONTHS[horizon]
    return {
        "horizon": horizon,
        "folds": rows,
        "paired": calendar_block_summary(dates, delta, block, reps),
        "sensitivity": calendar_block_summary(dates, delta, 2 * block, reps),
        "baseline": calendar_block_summary(dates, absolute_a, block, reps),
        "candidate": calendar_block_summary(dates, absolute_b, block, reps),
        "size_neutral_delta": float(np.mean([r["delta"]["size_neutral_ic"] for r in rows]))
        if rows
        else float("nan"),
        "top_decile_delta": float(np.mean([r["delta"]["sector_top_decile_excess"] for r in rows]))
        if rows
        else float("nan"),
    }


def holm_adjust(p_values):
    if set(p_values) - set(PRIMARY_HORIZONS):
        raise ValueError("unregistered comparison")
    values = {
        k: (p_values.get(k) if p_values.get(k) is not None else 1.0) for k in PRIMARY_HORIZONS
    }
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
        raise ValueError("invalid p-value")
    adjusted, running = {}, 0.0
    for i, (key, p) in enumerate(sorted(values.items(), key=lambda kv: kv[1])):
        running = max(running, min(1.0, (len(values) - i) * p))
        adjusted[key] = running
    return adjusted


def verdict(result, adjusted_p, confirmed=False):
    p, sensitivity = result["paired"], result["sensitivity"]
    if p.get("mean_ic") is None:
        return "inconclusive" if confirmed else "screen_failed"
    if not confirmed:
        return "confirm" if p["mean_ic"] >= MIN_DELTA_IC else "screen_failed"
    supported = (
        p["status"] == "estimated"
        and p["mean_ic"] >= MIN_DELTA_IC
        and p["ci_low"] > 0
        and adjusted_p < 0.05
        and result.get("size_neutral_delta") is not None
        and result.get("top_decile_delta") is not None
        and result["size_neutral_delta"] >= 0
        and result["top_decile_delta"] >= 0
    )
    if not supported:
        return "inconclusive"
    return (
        "strong_historical_incremental_evidence"
        if sensitivity["status"] == "estimated" and sensitivity["ci_low"] > 0
        else "historical_incremental_evidence"
    )


def append_prospective(path, record, now=None):
    now = now or datetime.now(timezone.utc)
    entry = datetime.fromisoformat(record["entry_at"])
    if entry.tzinfo is None or entry <= now:
        raise ValueError("prospective entry must be a timezone-aware future timestamp")
    if any(k in record for k in ("realized_return", "label", "outcome")):
        raise ValueError("prediction log cannot include outcomes")
    # One immutable file per prediction batch; append-only directory, no rewrite.
    write_json_new(
        Path(path) / f"{now.strftime('%Y%m%dT%H%M%S%fZ')}.json", {**record, "recorded_at": now}
    )
