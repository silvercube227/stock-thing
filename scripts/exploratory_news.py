"""Archive available Reuters history and freeze explicitly unvalidated news features."""

import argparse
import copy
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

from backend.ingestion.news_lseg import (
    daily_aggregate,
    fetch_window,
    open_store,
    parse_headline,
    score_archive,
    scorer_fingerprint,
    store_rows,
    utc,
)
from backend.ml.research import create_snapshot, load_snapshot, write_json_new

MODEL_REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def plan(snapshot, output):
    manifest = json.loads((Path(snapshot) / "manifest.json").read_text())
    meta = manifest["metadata"]["exploratory"]
    identities = Path(".research/verification/identity-archive.json").read_bytes()
    names = {r["RIC"]: r["Company Common Name"]
             for r in json.loads(identities)["sources"]["identity_archive"]["records"]
             if r.get("RIC")}
    excluded = {r["ticker_id"] for r in meta["exclusions"]
                if r["reason"] != "unresolved_pre_listing_history"}
    requests, aliases, skipped = [], {}, []
    for member in meta["intended_cohort"]:
        ric, tid = member["ric"], member["ticker_id"]
        if not member["mapped"] or tid in excluded:
            skipped.append(dict(ric=ric, reason="frozen_identity_exclusion_or_unmapped"))
            continue
        intervals = []
        for interval in member["membership"]:
            start = max(date(2010, 1, 1), date.fromisoformat(interval["valid_from"])
                        - timedelta(days=90))
            end = min(date(2023, 7, 1), date.fromisoformat(interval["valid_to"])
                      if interval["valid_to"] else date(2023, 7, 1))
            if start < end:
                intervals.append((start, end))
        if not intervals:
            skipped.append(dict(ric=ric, reason="no_pre_cutoff_membership"))
            continue
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        name = names.get(ric, "")
        # Explicitly provisional issuer aliases; no stock-return-selected names.
        short = re.sub(r"(?:,?\s+(?:incorporated|inc\.?|corporation|corp\.?|co\.?|"
                       r"company|plc|ltd\.?|limited|holdings?))+[.,]*$", "", name,
                       flags=re.I).strip()
        aliases[str(tid)] = [dict(alias=n, valid_from="2010-01-01", valid_to="2023-07-01")
                             for n in sorted({name, short}) if len(n) >= 3]
        for start, end in merged:
            requests.append(dict(ric=ric, ticker_id=tid, start=start.isoformat(),
                                 end=end.isoformat()))
    requests.sort(key=lambda r: digest(("exploratory-news-v1|" + r["ric"]
                                       + r["start"]).encode()))
    result = dict(status="frozen_exploratory_plan", snapshot=str(snapshot),
                  input_sha256=manifest["input_sha256"], requests=requests, aliases=aliases,
                  skipped=skipped, identity_archive_sha256=digest(identities),
                  model_revision=MODEL_REVISION,
                  limitations=[
                      "Incomplete source windows retain valid headlines and report rejected rows.",
                      "Partial-window scores use retained headlines; missing stories can bias results.",
                      "Current issuer-name aliases are provisional and may miss historical names.",
                      "FinBERT and entity/event rules lack independent human validation.",
                      "The pretrained classifier is retrospective, not historically deployed.",
                  ])
    write_json_new(output, result)
    print(dict(requests=len(requests), securities=len(aliases), skipped=len(skipped)))


def retained_rows(rows, start, end):
    """Exclude bad records individually without changing publication timestamps."""
    retained, rejected = [], Counter()
    for row in rows:
        try:
            parse_headline(row, end)
            stamp = utc(row["version_created"])
            if not start <= stamp < end:
                rejected["outside_requested_window"] += 1
                continue
        except (TypeError, ValueError, KeyError) as exc:
            rejected[str(exc)] += 1
            continue
        retained.append(row)
    return retained, rejected


def archive_window(ld, conn, request, raw_dir):
    start, end = utc(request["start"]), utc(request["end"])
    key = digest(json.dumps(request, sort_keys=True).encode())
    marker = raw_dir / (key + ".json")
    if marker.exists():
        entry = json.loads(marker.read_text())
        blob = gzip.decompress(Path(entry["artifact"]).read_bytes())
        if digest(blob) != entry["sha256"]:
            raise ValueError("News archive hash changed")
        payload = json.loads(blob)
        if payload["request"] != request:
            raise ValueError("News request changed")
        rows = payload["rows"]
    else:
        rows = fetch_window(ld, request["ric"], start, end)
        blob = json.dumps(dict(request=request, rows=rows), sort_keys=True, default=str).encode()
        artifact = raw_dir / (digest(blob) + ".json.gz")
        if not artifact.exists():
            with gzip.open(artifact, "xb") as stream:
                stream.write(blob)
        entry = dict(artifact=str(artifact), sha256=digest(blob), request=request)
        write_json_new(marker, entry)
    retained, rejected = retained_rows(rows, start, end)
    store_rows(conn, retained, request["ticker_id"], request["ric"])
    status = "partial" if rejected else "complete" if rows else "empty"
    with conn:
        conn.execute("insert or replace into coverage values (?,?,?,?,?,?)",
                     (request["ticker_id"], start.isoformat(), end.isoformat(), status,
                      len(retained), json.dumps(dict(rejected)) if rejected else None))
    return dict(request=request, status=status, retained=len(retained), raw_rows=len(rows),
                rejected=dict(rejected), archive=entry)


def fetch(plan_path, store, output, limit):
    from backend.config import get_settings
    from backend.ingestion.estimates import _open_session

    plan_bytes = Path(plan_path).read_bytes()
    frozen = json.loads(plan_bytes)
    Path(store).parent.mkdir(parents=True, exist_ok=True)
    binding = Path(store).parent / "plan.sha256"
    if binding.exists() and binding.read_text() != digest(plan_bytes):
        raise ValueError("News store is bound to a different frozen plan")
    if not binding.exists():
        binding.write_text(digest(plan_bytes))
    raw_dir = Path(store).parent / "raw"
    raw_dir.mkdir(exist_ok=True)
    conn = open_store(store)
    ld = _open_session(get_settings().lseg_app_key)
    results = []
    requests = frozen["requests"][:limit] if limit else frozen["requests"]
    try:
        for i, request in enumerate(requests):
            try:
                result = archive_window(ld, conn, request, raw_dir)
            except Exception as exc:
                result = dict(request=request, status="error", error=f"{type(exc).__name__}: {exc}")
            results.append(result)
            print(f"{i+1}/{len(requests)} {request['ric']} {result['status']} "
                  f"{result.get('retained', 0)}/{result.get('raw_rows', 0)}", flush=True)
    finally:
        ld.close_session()
        conn.close()
    write_json_new(output, dict(status="exploratory_archive_not_certified", results=results,
                               plan_sha256=digest(plan_bytes), store=str(store),
                               store_sha256=digest(Path(store).read_bytes()),
                               planned=len(frozen["requests"]), attempted=len(requests),
                               production_database_writes=0))


def prepare(plan_path, store, source_manifest, output, device):
    import torch

    torch.set_num_threads(4)
    frozen = json.loads(Path(plan_path).read_text())
    receipt_bytes = Path(source_manifest).read_bytes()
    receipt = json.loads(receipt_bytes)
    if receipt["plan_sha256"] != digest(Path(plan_path).read_bytes()):
        raise ValueError("Fetch receipt belongs to another plan")
    if receipt["attempted"] != receipt["planned"]:
        raise ValueError("Finish planned fetch attempts before the full-cohort experiment")
    conn = open_store(store)
    try:
        scored = score_archive(conn, frozen["aliases"], frozen["model_revision"],
                               exploratory=True, local_files_only=True, device=device)
        revision = "exploratory:" + scorer_fingerprint(frozen["model_revision"], frozen["aliases"])
        total_scored = conn.execute("select count(*) from scores where scorer_revision=?",
                                    (revision,)).fetchone()[0]
        daily = daily_aggregate(conn, revision)
        coverage = conn.execute("select ticker_id,start_at,end_at,status,n_rows,error "
                                "from coverage order by ticker_id,start_at").fetchall()
    finally:
        conn.close()
    inputs, manifest = load_snapshot(frozen["snapshot"])
    if manifest["input_sha256"] != frozen["input_sha256"]:
        raise ValueError("Parent snapshot changed")
    by_id, coverage_by_id = defaultdict(list), defaultdict(list)
    for row in daily:
        by_id[row["ticker_id"]].append(row)
    for tid, start, end, status, count, error in coverage:
        coverage_by_id[tid].append(dict(start_date=utc(start).date(),
                                       end_date=utc(end).date() - timedelta(days=1),
                                       status=status, retained_headlines=count,
                                       excluded_records=json.loads(error) if error else {}))
    for frame in inputs["frames"]:
        if frame.news or frame.news_coverage:
            raise ValueError("Refusing to replace existing snapshot news")
        frame.news = by_id.get(frame.ticker_id, [])
        frame.news_coverage = coverage_by_id.get(frame.ticker_id, [])
    metadata = copy.deepcopy(manifest["metadata"])
    metadata["exploratory"]["limitations"] += frozen["limitations"]
    metadata["exploratory"]["news"] = dict(
        plan_sha256=digest(Path(plan_path).read_bytes()), source_manifest=str(source_manifest),
        source_manifest_sha256=digest(receipt_bytes), newly_scored_versions=scored,
        scored_versions=total_scored, scored_store_sha256=digest(Path(store).read_bytes()),
        scorer_revision=revision, independent_validation=False, daily_rows=len(daily),
        securities_with_events=len(by_id),
        coverage_statuses=dict(Counter(r[3] for r in coverage)),
        parent_snapshot_sha256=manifest["input_sha256"],
    )
    result = create_snapshot(output, inputs["frames"], inputs["macro"], metadata)
    print(json.dumps(dict(metadata["exploratory"]["news"],
                         input_sha256=result["input_sha256"]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "fetch", "prepare"))
    parser.add_argument("--snapshot", default=".research/exploratory-reference-2026-09-19-v2")
    parser.add_argument("--plan")
    parser.add_argument("--store", default=".news_cache/exploratory-v1/news.sqlite")
    parser.add_argument("--source-manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "plan":
        plan(args.snapshot, args.output)
    elif args.command == "fetch":
        fetch(args.plan, args.store, args.output, args.limit)
    else:
        prepare(args.plan, args.store, args.source_manifest, args.output, args.device)
