"""Reconstruct reviewed RIC-level session membership; never invent DB identity maps."""
import argparse
import hashlib
import json
from pathlib import Path

from backend.ingestion.index_lseg import build_jl_intervals
from backend.ml.research import write_json_new


def run(archive, reviews, output):
    raw, review_raw = Path(archive).read_bytes(), Path(reviews).read_bytes()
    data = json.loads(raw)['sources']['membership_archive']
    reviewed = json.loads(review_raw)
    events = [dict(date=r['Date'], security_id=r['Constituent RIC'], change=r['Change'])
              for r in data['events']]
    intervals = build_jl_intervals(
        [r['RIC'] for r in data['current']], events,
        reviewed['window_start'], reviewed['observed_on'],
        reviewed_zero_duration=reviewed['zero_duration'])
    report = {'status': 'ric_intervals_reconciled_db_mapping_required',
              'archive_sha256': hashlib.sha256(raw).hexdigest(),
              'reviews_sha256': hashlib.sha256(review_raw).hexdigest(),
              'reviews': reviewed, 'intervals': intervals,
              'n_intervals': len(intervals),
              'n_distinct_rics': len({r['security_id'] for r in intervals})}
    write_json_new(output, report)
    print({k: v for k, v in report.items() if k not in ('intervals', 'reviews')})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', required=True)
    p.add_argument('--reviews', default='docs/membership_event_reviews.json')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    run(args.archive, args.reviews, args.output)
