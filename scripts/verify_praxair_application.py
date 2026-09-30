"""Independently check immutable post-commit snapshots and production cache."""

import argparse
import hashlib
import json
import pickle
from dataclasses import asdict
from datetime import date
from pathlib import Path

from backend.config import get_settings
from backend.ml.compressed_io import open_artifact
from backend.ml.dataset import FRAME_CACHE_VERSION
from backend.ml.research import json_value, load_snapshot, write_json_new
from scripts.signal_research import source_ready


def digest(value):
    return hashlib.sha256(
        json.dumps(json_value(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def run(parent, snapshot, production, receipt, output):
    before, _ = load_snapshot(parent)
    # Only retain per-security hashes between full snapshots, not four copies of
    # the multi-million-row price reference in memory at once.
    old_hashes, previously_empty = {}, set()
    for frame in before["frames"]:
        values = asdict(frame)
        if values["sector_history"] is None:
            previously_empty.add(frame.ticker_id)
            values["sector_history"] = []
        old_hashes[frame.ticker_id] = digest(values)
    old_macro = digest(before["macro"])
    del before, values, frame
    after, manifest = load_snapshot(snapshot)
    applied = json.loads(Path(receipt).read_text())
    if not applied["applied"] or applied["sectors_applied"] or applied["global_gates_promoted"]:
        raise ValueError("Unexpected application scope")
    b = {f.ticker_id: f for f in after["frames"]}
    if old_hashes.keys() != b.keys() or old_macro != digest(after["macro"]):
        raise ValueError("Security inventory or macro values changed")
    if digest([asdict(b[tid]) for tid in (360, 947)]) != applied["stored_frames_sha256"]:
        raise ValueError("Post-commit frames differ from in-transaction readback")
    empty_sectors = 0
    new_hashes = {}
    for tid, frame in b.items():
        right = asdict(frame)
        new_hashes[tid] = digest(right)
        if tid in previously_empty and right["sector_history"] == []:
            empty_sectors += 1
        if tid not in (360, 947) and old_hashes[tid] != new_hashes[tid]:
            raise ValueError(f"Unrelated security changed: {tid}")
    if b[947].sector_history != [] or b[360].price_history_start != date(2018, 10, 31):
        raise ValueError("Price boundary or unknown-sector policy lost")
    target_rows, successor_rows = len(b[947].prices), len(b[360].prices)
    del after, b, right, frame
    prod, prod_manifest = load_snapshot(production)
    prod_hashes = {f.ticker_id: digest(asdict(f)) for f in prod["frames"]}
    prod_rows = sum(len(f.prices) for f in prod["frames"])
    if 947 in prod_hashes:
        raise ValueError("Research-only predecessor leaked into default production universe")
    if any(value != new_hashes[tid] for tid, value in prod_hashes.items()):
        raise ValueError("Production and research frames disagree")
    del prod
    with open_artifact(get_settings().frame_cache_dir / "frames_all.pkl") as stream:
        cache = pickle.load(stream)
    if (
        cache["version"] != FRAME_CACHE_VERSION
        or {f.ticker_id: digest(asdict(f)) for f in cache["frames"]} != prod_hashes
    ):
        raise ValueError("Production cache does not match fresh snapshot")
    del cache
    for data in (manifest, prod_manifest):
        for gate in ("price_share_basis", "membership", "terminal_outcomes"):
            try:
                source_ready(data["metadata"]["sources"], gate)
            except ValueError:
                continue
            raise ValueError("Incomplete shared reference gate promoted")
    report = dict(
        status="scoped_application_verified_not_global_certification",
        input_sha256=manifest["input_sha256"],
        production_input_sha256=prod_manifest["input_sha256"],
        application_receipt_sha256=hashlib.sha256(Path(receipt).read_bytes()).hexdigest(),
        unrelated_security_data_preserved=len(old_hashes) - 2,
        empty_sectors_now_explicitly_unknown=empty_sectors,
        macro_unchanged=True,
        frame_cache_version=FRAME_CACHE_VERSION,
        cache_matches_snapshot=True,
        research_frames=len(new_hashes),
        production_frames=len(prod_hashes),
        production_price_rows=prod_rows,
        predecessor_rows=target_rows,
        successor_rows=successor_rows,
        shared_gates_still_blocked=True,
        model_fits=0,
    )
    write_json_new(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", default=".research/reference-gate-audit-2026-09-08-v2")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--production", required=True)
    p.add_argument("--receipt", default=".research/verification/praxair-reference-applied-v1.json")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run(a.parent, a.snapshot, a.production, a.receipt, a.output)
