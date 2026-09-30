"""Export the locked annotation sample, validate it, score, and aggregate locally."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from backend.ingestion.news_lseg import (
    daily_aggregate,
    entity_event,
    open_store,
    score_archive,
    scorer_fingerprint,
    validate_annotations,
)
from backend.ml.research import write_json_new


def annotation_sample(conn, strata):
    buckets, seen = defaultdict(list), set()
    rows = conn.execute("""select v.story_id,v.version_key,r.ticker_id,v.title,v.version_created
        from versions v join relevance r using(version_key)
        order by v.version_created,v.version_key,r.ticker_id""").fetchall()
    for story, key, tid, title, stamp in rows:
        if story in seen:
            continue
        seen.add(story)
        meta = strata[str(tid)]
        if meta["cohort"] not in ("surviving", "removed") or not meta["sector"]:
            raise ValueError("annotation sample requires sector and removal cohort metadata")
        year = int(stamp[:4])
        era = "2012-2016" if year < 2017 else "2017-2020" if year < 2021 else "2021+"
        bucket = (era, meta["sector"], meta["cohort"])
        buckets[bucket].append(
            dict(
                story_id=story,
                version_key=key,
                ticker_id=tid,
                title=title,
                version_created=stamp,
                stratum=bucket,
            )
        )
    for bucket in buckets.values():
        bucket.sort(key=lambda r: hashlib.sha256(r["version_key"].encode()).hexdigest())
    if sum(map(len, buckets.values())) < 400:
        raise ValueError("at least 400 distinct stories required before annotation")
    if {b[2] for b in buckets} != {"surviving", "removed"}:
        raise ValueError("annotation archive lacks surviving/removed coverage")
    selected = []
    while len(selected) < 400:
        for key in sorted(buckets):
            if buckets[key] and len(selected) < 400:
                selected.append(buckets[key].pop())
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("annotate", "validate", "score", "aggregate"))
    p.add_argument("--store", required=True)
    p.add_argument("--aliases", required=True, help="dated reviewed issuer aliases as JSON")
    p.add_argument("--model-revision", required=True)
    p.add_argument("--annotations")
    p.add_argument("--strata")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    aliases = json.loads(Path(args.aliases).read_text())
    revision = scorer_fingerprint(args.model_revision, aliases)
    conn = open_store(args.store)
    try:
        if args.command == "annotate":
            if not args.strata:
                p.error("annotate requires --strata")
            sample = annotation_sample(conn, json.loads(Path(args.strata).read_text()))
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

            classifier = pipeline(
                "text-classification",
                top_k=None,
                model=AutoModelForSequenceClassification.from_pretrained(
                    "ProsusAI/finbert", revision=args.model_revision
                ),
                tokenizer=AutoTokenizer.from_pretrained(
                    "ProsusAI/finbert", revision=args.model_revision
                ),
            )
            for i, row in enumerate(sample):
                event = entity_event(
                    row["title"], row["ticker_id"], aliases, row["version_created"]
                )
                row.update(
                    split="development" if i % 2 == 0 else "validation",
                    pred_relevant=event["relevant"],
                    pred_event=event["event_type"],
                    gold_relevant=None,
                    gold_event=None,
                    gold_sentiment=None,
                    pred_sentiment="neutral",
                )
                if event["relevant"]:
                    probs = classifier(event["text"], truncation=True, max_length=512)[0]
                    row["pred_sentiment"] = max(probs, key=lambda p: p["score"])["label"].lower()
            write_json_new(args.output, {"scorer_revision": revision, "rows": sample})
        elif args.command in ("validate", "score"):
            if not args.annotations:
                p.error("validate/score require --annotations")
            annotated = json.loads(Path(args.annotations).read_text())
            annotated["rows"] = [r for r in annotated["rows"] if r["split"] == "validation"]
            if args.command == "validate":
                result = validate_annotations(annotated["rows"])
            else:
                result = {"scored": score_archive(conn, aliases, args.model_revision, annotated)}
            write_json_new(args.output, result)
        else:
            write_json_new(args.output, daily_aggregate(conn, revision))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
