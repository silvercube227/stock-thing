"""Freeze a deterministic news review corpus and export blinded human worksheets.

Sector reconstructions are sampling metadata only, pending the separate sector
gate. No model predictions, gold labels, database writes or gate promotions.
"""

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from backend.ingestion.news_lseg import ingest_window, open_store, utc
from backend.ml.research import write_json_new

SOURCES = {
    "mapping": ".research/verification/security-mapping-exact-2026-09-18-v2.json",
    "identities": ".research/verification/identity-archive.json",
    "membership": ".research/verification/membership-reconciled.json",
    "sectors": ".research/verification/sector-2018-anchor-reviewed.json",
}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def covers(row, start, end):
    return row["valid_from"] <= start and (not row["valid_to"] or row["valid_to"] >= end)


def make_plan(output):
    data = {k: json.loads(Path(p).read_text()) for k, p in SOURCES.items()}
    names = {
        r["RIC"]: r["Company Common Name"]
        for r in data["identities"]["sources"]["identity_archive"]["records"]
        if r.get("RIC")
    }
    mapping = {
        r["ric"]: r
        for r in data["mapping"]["mappings"]
        if r["status"]
        in ("identity_candidate_no_history", "identity_candidate_requires_history_review")
        and len(r["candidate_ticker_ids"]) == 1
    }
    members = defaultdict(list)
    for row in data["membership"]["intervals"]:
        members[row["security_id"]].append(row)
    eligible = defaultdict(dict)
    for year, era in [(2013, "2012-2016"), (2018, "2017-2020"), (2022, "2021+")]:
        start, end = f"{year}-06-01", f"{year}-09-01"
        for sector in data["sectors"]["results"]:
            for interval in sector.get("intervals", []):
                ric = interval["security_id"]
                identity = mapping.get(ric)
                if not identity or not names.get(ric) or not covers(interval, start, end):
                    continue
                if not any(covers(r, start, end) for r in members[ric]):
                    continue
                if identity["first_trade_date"] and identity["first_trade_date"] > start:
                    continue
                if identity["retire_date"] and identity["retire_date"] < end:
                    continue
                survives = any(covers(r, "2023-09-29", "2023-09-30") for r in members[ric])
                cohort = "surviving" if survives else "removed"
                stratum = (era, sector["description"], cohort)
                eligible[stratum][ric] = dict(
                    ric=ric,
                    ticker_id=identity["candidate_ticker_ids"][0],
                    issuer=names[ric],
                    isin=identity["isin"],
                    start=start,
                    end=end,
                    stratum=list(stratum),
                )
    requests = []
    for stratum, choices in sorted(eligible.items()):
        ordered = sorted(
            choices, key=lambda ric: digest("news-review-v1|" + "|".join(stratum) + ric)
        )
        requests.append(choices[ordered[0]])
    write_json_new(
        output,
        dict(
            status="sampling_plan_frozen_source_gates_incomplete",
            requests=requests,
            selection=(
                "One hash-selected exact-RIC identity candidate per available era/sector/cohort; "
                "June-August windows fixed before retrieval or labels."
            ),
            cohort_as_of="2023-09-29",
            source_hashes={
                p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCES.values()
            },
            eligible_counts={"|".join(k): len(v) for k, v in sorted(eligible.items())},
            limitations=[
                "Sector intervals and security mappings are review candidates, "
                "not certified research features.",
                "Finite quarter/issuer sample does not establish full news archive coverage.",
                "No post-2023 stories or future removal status used.",
            ],
            database_writes=0,
            gates_promoted=0,
        ),
    )


def fetch(plan_path, store, output):
    from backend.config import get_settings
    from backend.ingestion.estimates import _open_session

    plan = json.loads(Path(plan_path).read_text())
    Path(store).parent.mkdir(parents=True, exist_ok=True)
    conn = open_store(store)
    ld = _open_session(get_settings().lseg_app_key)
    results = []
    try:
        for i, request in enumerate(plan["requests"]):
            status = ingest_window(
                ld,
                conn,
                request["ticker_id"],
                request["ric"],
                request["start"],
                request["end"],
                raw_archive=Path(store).parent / "raw",
            )
            results.append(dict(request=request, status=status))
            print(
                f"{i + 1}/{len(plan['requests'])} {request['ric']} {request['start']} {status}",
                flush=True,
            )
    finally:
        ld.close_session()
        conn.close()
    write_json_new(
        output,
        dict(
            status="retrieved_not_certified",
            results=results,
            plan_sha256=hashlib.sha256(Path(plan_path).read_bytes()).hexdigest(),
            store_sha256=hashlib.sha256(Path(store).read_bytes()).hexdigest(),
            database_writes=0,
            gates_promoted=0,
        ),
    )


def select_sample(rows, n=400):
    """One version per story; balanced round-robin sampling and stratified split."""
    buckets, seen = defaultdict(list), set()
    for row in sorted(rows, key=lambda r: (r["version_created"], r["version_key"], r["ticker_id"])):
        if row["story_id"] in seen:
            continue
        seen.add(row["story_id"])
        buckets[tuple(row["stratum"])].append(row.copy())
    if len(seen) < n:
        raise ValueError(f"at least {n} distinct stories required, found {len(seen)}")
    for bucket in buckets.values():
        bucket.sort(key=lambda r: digest(r["version_key"]))
    selected = []
    while len(selected) < n:
        for key in sorted(buckets):
            if not buckets[key] or len(selected) >= n:
                continue
            selected.append(buckets[key].pop())
    split_counts, selected_buckets = Counter(), defaultdict(list)
    for row in selected:
        selected_buckets[tuple(row["stratum"])].append(row)
    for key, bucket in sorted(selected_buckets.items()):
        first = min(
            ("development", "validation"),
            key=lambda s: (split_counts[s], digest("|".join(key) + s)),
        )
        other = "validation" if first == "development" else "development"
        for i, row in enumerate(sorted(bucket, key=lambda r: digest(r["version_key"]))):
            split = first if i % 2 == 0 else other
            row.update(split=split, gold_relevant=None, gold_event=None, gold_sentiment=None)
            split_counts[split] += 1
    if split_counts != {"development": n // 2, "validation": n // 2}:
        raise ValueError("uneven split; refusing unlocked annotation sample")
    return selected


def export(plan_path, store, directory, include_quarantined=False):
    plan = json.loads(Path(plan_path).read_text())
    conn = open_store(store)
    rows = []
    try:
        for request in plan["requests"]:
            found = conn.execute(
                """select v.story_id,v.version_key,v.title,v.version_created
                from versions v join relevance r using(version_key)
                where r.ticker_id=? and r.ric=? and v.version_created>=? and v.version_created<?""",
                (request["ticker_id"], request["ric"], request["start"], request["end"]),
            ).fetchall()
            for story, key, title, stamp in found:
                rows.append(
                    dict(
                        story_id=story,
                        version_key=key,
                        title=title,
                        version_created=stamp,
                        **request,
                    )
                )
    finally:
        conn.close()
    raw_hashes = {}
    if include_quarantined:
        requests = {(r["ric"], r["start"], r["end"]): r for r in plan["requests"]}
        for artifact in sorted((Path(store).parent / "raw").glob("*.json.gz")):
            payload = gzip.decompress(artifact.read_bytes())
            if hashlib.sha256(payload).hexdigest() != artifact.name.removesuffix(".json.gz"):
                raise ValueError("raw news archive hash mismatch")
            raw = json.loads(payload)
            request = requests.get((raw["ric"], raw["start"][:10], raw["end"][:10]))
            if not request:
                continue
            raw_hashes[str(artifact)] = hashlib.sha256(payload).hexdigest()
            for row in raw["rows"]:
                if not row.get("story_id") or not row.get("title"):
                    raise ValueError("raw annotation record lacks identity or title")
                first, stamp = utc(row["first_created"]), utc(row["version_created"])
                if not request["start"] <= stamp.date().isoformat() < request["end"]:
                    continue
                key = digest(f"{row['story_id']}|{stamp.isoformat()}|{digest(row['title'])}")
                rows.append(
                    dict(
                        story_id=row["story_id"],
                        version_key=key,
                        title=row["title"],
                        version_created=stamp.isoformat(),
                        first_created=first.isoformat(),
                        timestamp_order_valid=stamp >= first,
                        **request,
                    )
                )
    sample = select_sample(rows)
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=False)
    for split in ("development", "validation"):
        with (path / f"{split}.csv").open("x", newline="") as stream:
            fields = [
                "version_key",
                "ric",
                "isin",
                "issuer",
                "version_created",
                "title",
                "gold_relevant",
                "gold_event",
                "gold_sentiment",
                "reviewer",
                "notes",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(r for r in sample if r["split"] == split)
    write_json_new(
        path / "locked_sample.json",
        dict(
            status="awaiting_independent_human_labels",
            rows=sample,
            raw_source_hashes=raw_hashes,
            includes_quarantined_timestamps=include_quarantined,
            source_timing_certified=False,
            plan_sha256=hashlib.sha256(Path(plan_path).read_bytes()).hexdigest(),
            store_sha256=hashlib.sha256(Path(store).read_bytes()).hexdigest(),
            worksheets={
                f"{s}.csv": hashlib.sha256((path / f"{s}.csv").read_bytes()).hexdigest()
                for s in ("development", "validation")
            },
            limitations=plan["limitations"],
            stratum_counts=dict(Counter("|".join(r["stratum"]) for r in sample)),
            scorer_predictions_present=False,
            gates_promoted=0,
        ),
    )
    (path / "README.md").write_text(
        "# Independent news review\n\n"
        "Annotate each headline relative to the named target issuer. Preserve row keys and text.\n"
        "Enter gold_relevant as true/false; gold_event as guidance/operating/other; "
        "gold_sentiment as positive/negative/neutral. Identify the human reviewer.\n\n"
        "Relevant means the headline actually attributes information to that issuer. "
        "Guidance means a forward management outlook; operating means reported business "
        "results or operations; use other otherwise. Sentiment is the implication for "
        "the target issuer, not general market tone. Use neutral for an irrelevant story. "
        "Flag ambiguity in notes for human adjudication. Do not copy classifier labels.\n\n"
        "The development and validation sets each contain 200 distinct stories. Keep "
        "validation labels sealed from rule/model tuning. No model predictions are supplied. "
        "Validation requires frozen scorer predictions before labels are examined.\n\n"
        "This is a stratified review sample, not certification of source timestamps, "
        "full archive coverage, identity continuity or historical sectors. Some raw windows "
        "have inconsistent creation/version timestamps; their text may be reviewed here "
        "but remains quarantined from feature scoring. Sampling sectors are provisional.\n"
    )
    print(
        dict(
            stories=len(sample),
            split_counts=dict(Counter(r["split"] for r in sample)),
            strata=len({tuple(r["stratum"]) for r in sample}),
            output=str(path),
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "fetch", "export"))
    parser.add_argument("--plan", required=True)
    parser.add_argument("--store", default=".news_cache/annotation-review/news.sqlite")
    parser.add_argument("--output")
    parser.add_argument(
        "--include-quarantined",
        action="store_true",
        help="Include raw rejected windows in human text review only",
    )
    args = parser.parse_args()
    if args.command == "plan":
        make_plan(args.plan)
    elif args.command == "fetch":
        fetch(args.plan, args.store, args.output)
    else:
        export(args.plan, args.store, args.output, args.include_quarantined)
