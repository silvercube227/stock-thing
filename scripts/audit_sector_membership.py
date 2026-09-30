"""Reconcile sector events against independently requested historical sets."""
import argparse
from datetime import date
import gzip
import hashlib
import json
from pathlib import Path

from backend.ingestion.index_lseg import build_jl_intervals
from backend.ml.research import write_json_new


def load_entry(entry):
    blob = gzip.decompress(Path(entry['artifact']).read_bytes())
    if hashlib.sha256(blob).hexdigest() != entry['sha256']:
        raise ValueError('sector archive hash mismatch')
    payload = json.loads(blob)
    if payload['request'] != entry['request']:
        raise ValueError('sector archive request mismatch')
    return payload


def members(records):
    values = [r['Constituent RIC'] for r in records]
    if not values or any(v in ('', '<NA>', 'nan', 'None') for v in values):
        raise ValueError('historical membership unavailable')
    if len(values) != len(set(values)):
        raise ValueError('duplicate historical constituent')
    return values


def reviewed_aliases(path):
    review = json.loads(Path(path).read_text())
    if review['scope'] != 'sector_membership_identity_only':
        raise ValueError('unexpected security-alias scope')
    aliases = {}
    for group in review['aliases']:
        rows = load_entry(group['evidence'])['records']
        if not group.get('transfer_source') or not group.get('listing_transfer_date'):
            raise ValueError('listing transfer evidence required')
        if group['security_id'] != 'ISIN:' + group['isin']:
            raise ValueError('canonical security differs from reviewed ISIN')
        for ric in group['rics']:
            hits = [r for r in rows if r.get('RIC') == ric]
            if len(hits) != 1 or any(hits[0].get(field) != group[key] for field,key in
                                    [('ISIN','isin'), ('CUSIP','cusip'), ('CIK Number','cik')]):
                raise ValueError('security alias identity is not unambiguous')
            if ric in aliases:
                raise ValueError('duplicate security alias review')
            aliases[ric] = group['security_id']
    return aliases


def canonical_membership_rows(rows, aliases):
    """Collapse only distinct reviewed RIC aliases for the same security in a set.

    Repeated exact RICs still fail. Event streams must not use this set operation.
    """
    result, seen, raw_seen, duplicates = [], {}, set(), []
    for row in rows:
        ric = row['Constituent RIC']
        if ric in raw_seen:
            raise ValueError('duplicate historical constituent')
        raw_seen.add(ric)
        security = aliases.get(ric, ric)
        if security in seen:
            if ric not in aliases or seen[security] not in aliases:
                raise ValueError('unreviewed security collision')
            duplicates.append(dict(security_id=security, rics=[seen[security], ric]))
            continue
        seen[security] = ric
        result.append(dict(row, **{'Constituent RIC': security}))
    return result, duplicates


def run(source, output, anchor='2026-09-04', snapshots=None):
    manifest = json.loads(Path(source).read_text())
    payloads = [load_entry(e) for e in manifest['entries']]
    extra_hashes = {}
    for snapshot_path in snapshots or []:
        raw = Path(snapshot_path).read_bytes()
        extra_hashes[snapshot_path] = hashlib.sha256(raw).hexdigest()
        payloads.extend(load_entry(e) for e in json.loads(raw)['entries'])
    alias_path = Path('docs/sector_security_alias_reviews.json')
    aliases = reviewed_aliases(alias_path) if alias_path.exists() else {}
    duplicate_aliases = []
    for payload in payloads:
        if payload['request']['fields'] == ['TR.IndexConstituentRIC'] and isinstance(payload['request']['universe'], str):
            payload['records'], duplicates = canonical_membership_rows(payload['records'], aliases)
            if duplicates:
                duplicate_aliases.append(dict(request=payload['request'], duplicates=duplicates))
            continue
        for row in payload['records']:
            ric = row.get('Constituent RIC')
            if ric in aliases:
                row['Constituent RIC'] = aliases[ric]
    identities = {r['Instrument']: r['Instrument Description']
                  for p in payloads if 'TR.InstrumentDescription' in p['request']['fields']
                  for r in p['records']}
    reviews = json.loads(Path('docs/membership_event_reviews.json').read_text())['zero_duration']
    sector_review_path = Path('docs/sector_event_reviews.json')
    sector_reviews = json.loads(sector_review_path.read_text()) if sector_review_path.exists() else {}
    for review in sector_reviews.get('continuity', []):
        sets = [set(members(load_entry(e)['records'])) for e in review['evidence']]
        if len(sets) != 2 or sets[0] != sets[1] or review['security_id'] not in sets[0]:
            raise ValueError('continuity evidence does not show matching before/after membership')
    results = []
    for index, description in identities.items():
        result = dict(index=index, description=description, checks=[])
        if not description.startswith('S&P 500 ') or not description.endswith('(Sector)'):
            results.append(dict(**result, status='blocked_index_identity'))
            continue
        entries = [p for p in payloads if p['request']['universe'] == index]
        snapshots = {p['request']['parameters']['SDate']: p for p in entries
                     if p['request']['fields'] == ['TR.IndexConstituentRIC']}
        event_payload = next(p for p in entries if 'IC' in p['request']['parameters'])
        try:
            events = [dict(security_id=r['Constituent RIC'], date=r['Date'], change=r['Change'])
                      for r in event_payload['records']]
            pairs = {}
            for event in events:
                pairs.setdefault((event['date'], event['security_id']), set()).add(event['change'].lower())
            applicable = [r for r in reviews
                          if pairs.get((r['date'], r['security_id'])) == {'joiner', 'leaver'}]
            applicable = [r for r in applicable if r['date'] <= anchor]
            continuity = [r for r in sector_reviews.get('continuity', [])
                          if sector_reviews.get('index') == index and r['date'] <= anchor]
            intervals = build_jl_intervals(members(snapshots[anchor]['records']), events,
                                           '2010-01-01', anchor,
                                           reviewed_zero_duration=applicable,
                                           reviewed_continuity=continuity)
        except ValueError as exc:
            results.append(dict(**result, status='blocked_reconstruction', reason=str(exc)))
            continue
        for when, snapshot in sorted(snapshots.items()):
            if when > anchor:
                continue
            try:
                expected = set(members(snapshot['records']))
            except ValueError as exc:
                result['checks'].append(dict(date=when, status='unavailable', reason=str(exc)))
                continue
            day = date.fromisoformat(when)
            actual = {r['security_id'] for r in intervals if r['valid_from'] <= day
                      and (r['valid_to'] is None or day < r['valid_to'])}
            result['checks'].append(dict(date=when, status='matched' if actual == expected else 'mismatch',
                                         expected=len(expected), reconstructed=len(actual),
                                         missing=sorted(expected-actual), extra=sorted(actual-expected)))
        result.update(status='reconstructed_requires_coverage_certification', intervals=intervals)
        results.append(result)
    write_json_new(output, dict(status='sector_history_audit_not_certified', results=results,
                                anchor=anchor, additional_snapshot_hashes=extra_hashes,
                                duplicate_alias_observations=duplicate_aliases,
                                alias_review_sha256=hashlib.sha256(alias_path.read_bytes()).hexdigest()
                                if alias_path.exists() else None,
                                sector_reviews_sha256=hashlib.sha256(sector_review_path.read_bytes()).hexdigest()
                                if sector_review_path.exists() else None,
                                archive_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                                database_writes=0))
    for result in results:
        print(result['index'], result['status'], result.get('reason', ''),
              [(r['date'], r['status']) for r in result['checks']])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='.research/verification/sector-history-archive.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--anchor', default='2026-09-04')
    parser.add_argument('--snapshots', action='append')
    args = parser.parse_args()
    run(args.source, args.output, args.anchor, args.snapshots)
