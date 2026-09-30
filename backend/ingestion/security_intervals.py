"""Dated separation of a verified predecessor from a vendor-linked successor."""
from backend.ingestion.index_lseg import day


def split_predecessor_intervals(intervals, *, predecessor, successor, effective_date, evidence):
    if predecessor == successor or not evidence:
        raise ValueError('distinct securities and transition evidence are required')
    boundary = day(effective_date)
    result = []
    for row in intervals:
        start = day(row['valid_from'])
        end = day(row['valid_to']) if row.get('valid_to') is not None else None
        if end is not None and end <= start:
            raise ValueError('invalid security interval')
        if row['security_id'] != successor or start >= boundary:
            result.append(dict(row))
            continue
        # Preserve each membership episode; never bridge a gap or a reentry.
        result.append(dict(row, security_id=predecessor, valid_to=min(end, boundary) if end else boundary,
                           transition_evidence=evidence))
        if end is None or end > boundary:
            result.append(dict(row, valid_from=boundary, transition_evidence=evidence))
    by_security = {}
    for row in result:
        by_security.setdefault(row['security_id'], []).append(row)
    for rows in by_security.values():
        rows.sort(key=lambda r: day(r['valid_from']))
        for left, right in zip(rows, rows[1:]):
            if left.get('valid_to') is None or day(left['valid_to']) > day(right['valid_from']):
                raise ValueError('predecessor/successor split creates overlapping membership')
    return result
