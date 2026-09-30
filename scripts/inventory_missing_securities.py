"""Group source rows by exact RIC before planning historical-security backfills."""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path

from backend.ml.research import write_json_new


def inventory(mapping):
    grouped = defaultdict(list)
    for row in mapping['mappings']:
        grouped[row['ric']].append(row)
    result = []
    for ric, rows in sorted(grouped.items()):
        if not any(r['status']=='absent_from_database' for r in rows):
            continue
        candidates = sorted({tid for r in rows for tid in r['candidate_ticker_ids']})
        isins = sorted({r['isin'] for r in rows if r.get('isin')})
        result.append(dict(ric=ric, isins=isins,
            issuer_ciks=sorted({str(int(r['cik'])) for r in rows if r.get('cik')}),
            source_row_count=len(rows), existing_candidate_ids=candidates,
            status=('conflicting_source_rows' if candidates else
                    'requires_identifier_review' if len(isins)!=1 else
                    'missing_security_requires_backfill'),
            retire_dates=sorted({r['retire_date'] for r in rows if r.get('retire_date')})))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping',default='.research/verification/security-mapping-resolved.json')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    raw=Path(args.mapping).read_bytes()
    rows=inventory(json.loads(raw))
    counts=dict(Counter(r['status'] for r in rows))
    write_json_new(args.output,dict(source_sha256=hashlib.sha256(raw).hexdigest(),
                                   counts=counts,securities=rows,database_writes=0))
    print(counts)
