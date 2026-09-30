"""Strict corroboration of observed price events with independent action terms."""
from datetime import date
import math


def classify_share_action(observed_date, price_factor, records):
    """Return one source-confirmed event, or None; no nearest-date guessing."""
    d = str(observed_date)
    matches = []
    for index, r in enumerate(records):
        if r.get('Capital Change Is Rescinded') != 'False':
            continue
        ex_date = r.get('Capital Change Ex Date')
        event_date = ex_date if ex_date not in (None, '', 'NaT', '<NA>') else r.get('Capital Change Effective Date')
        if event_date != d:
            continue
        try:
            date.fromisoformat(event_date)
            price_factor = float(price_factor)
            if r.get('Adjustment Type') == 'LSEG Pricing Only':
                ratio = 1.0
                expected = 1/float(r['Adjustment Factor'])
                tolerance = 5e-4  # source price factors may be rounded independently
            elif r.get('Corporate Change Event Type') in ('Share Split', 'Share Consolidation'):
                if r.get('Adjustment Type') != 'Capital Change Type':
                    continue
                ratio = float(r['Terms New Shares'])/float(r['Terms Old Shares'])
                expected, tolerance = ratio, 1e-6
            else:
                continue
            if ratio > 0 and math.isfinite(ratio) and math.isclose(price_factor,expected,rel_tol=tolerance):
                matches.append((index, ratio))
        except (ValueError, TypeError, ZeroDivisionError, KeyError):
            continue
    return matches[0] if len(matches) == 1 else None


# Two further corroboration rules, each narrower than "nearest match". They exist
# because the strict rule above rejects two source patterns that are not
# ambiguous, only differently expressed.
PRICING_ONLY_MAX_DISAGREEMENT = 0.05   # spin-off stub valuation, not a ratio error
TERMS_OFFSET_MAX_DAYS = 4              # LSEG dates the term change, Yahoo the price step


def _event_dates(record):
    from datetime import date as _date

    out = []
    for field in ("Capital Change Ex Date", "Capital Change Effective Date"):
        try:
            out.append(_date.fromisoformat(record.get(field) or ""))
        except (TypeError, ValueError):
            continue
    return out


def classify_pricing_only(observed_date, price_factor, records):
    """A distribution that changes price but not the parent's share count.

    Every non-rescinded source record on the observed date must be
    `LSEG Pricing Only`, which is LSEG's statement that the event adjusts price
    history alone. The share ratio is then 1.0 by definition of the adjustment
    type, so the price factors are compared only as a sanity bound: for a
    spin-off the vendor and Yahoo value the distributed stub against different
    reference prices and disagree by a fraction of a percent, which the exact
    rule treats as a mismatch. Returns (indices, 1.0, disagreement) or None.
    """
    on_date, others = [], []
    for index, r in enumerate(records):
        if r.get("Capital Change Is Rescinded") != "False":
            continue
        if observed_date in _event_dates(r):
            on_date.append((index, r))
        else:
            others.append(r)
    if not on_date or not all(r.get("Adjustment Type") == "LSEG Pricing Only" for _, r in on_date):
        return None
    expected = 1.0
    for _, r in on_date:
        try:
            factor = float(r["Adjustment Factor"])
        except (TypeError, ValueError, KeyError):
            return None
        if not factor > 0:
            return None
        expected /= factor
    price_factor = float(price_factor)
    if not price_factor > 0 or not math.isfinite(expected):
        return None
    disagreement = abs(expected - price_factor) / price_factor
    if disagreement > PRICING_ONLY_MAX_DISAGREEMENT:
        return None
    return [i for i, _ in on_date], 1.0, disagreement


def classify_terms_offset(observed_date, price_factor, records):
    """A share split whose source date precedes the observed price step.

    Requires exactly one non-rescinded split or consolidation within
    TERMS_OFFSET_MAX_DAYS whose declared terms equal the observed price factor
    exactly. The ratio still comes from the terms, never from the price.
    Returns (index, ratio, offset_days) or None.
    """
    price_factor = float(price_factor)
    hits = []
    for index, r in enumerate(records):
        if r.get("Capital Change Is Rescinded") != "False":
            continue
        if r.get("Corporate Change Event Type") not in ("Share Split", "Share Consolidation"):
            continue
        if r.get("Adjustment Type") != "Capital Change Type":
            continue
        try:
            ratio = float(r["Terms New Shares"]) / float(r["Terms Old Shares"])
        except (TypeError, ValueError, ZeroDivisionError, KeyError):
            continue
        if not (ratio > 0 and math.isfinite(ratio)):
            continue
        if not math.isclose(price_factor, ratio, rel_tol=1e-6):
            continue
        offsets = [(d - observed_date).days for d in _event_dates(r)]
        near = [o for o in offsets if -TERMS_OFFSET_MAX_DAYS <= o < 0]
        if near:
            hits.append((index, ratio, min(near, key=abs)))
    return hits[0] if len(hits) == 1 else None


_NUMERIC_FIELDS = ("Adjustment Factor", "Terms New Shares", "Terms Old Shares")


def _normalized_key(record):
    """Identity of a source event, insensitive to numeric string formatting.

    The same action is returned by more than one pull with terms written as
    '1.0/9.0' in one and '1.0/9' in the other. Treating those as two events
    squares the adjustment factor and silently destroys the corroboration.
    """
    key = []
    for field, value in sorted(record.items()):
        if field in _NUMERIC_FIELDS:
            try:
                value = repr(round(float(value), 9))
            except (TypeError, ValueError):
                value = str(value)
        key.append((field, str(value)))
    return tuple(key)


def dedupe_actions(records):
    """Distinct source events, preserving first-seen order."""
    seen, out = set(), []
    for r in records:
        key = _normalized_key(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def classify_mixed_capital_change(observed_date, price_factor, records):
    """One share-changing action combined with price-only distributions.

    A reverse split executed alongside a spin-off (DowDuPont, Motorola, Leidos)
    produces a price factor that is the product of both, so neither the terms
    ratio nor the pricing-only factor matches it alone. The SHARE ratio still
    comes solely from the single capital-change record; the distributions
    contribute 1.0. The observed price factor is only checked against the
    product of every adjustment factor on the date.
    Returns (indices, ratio, disagreement) or None.
    """
    on_date = [
        (i, r) for i, r in enumerate(records)
        if r.get("Capital Change Is Rescinded") == "False" and observed_date in _event_dates(r)
    ]
    if not on_date:
        return None
    changes = [(i, r) for i, r in on_date if r.get("Adjustment Type") == "Capital Change Type"]
    pricing = [(i, r) for i, r in on_date if r.get("Adjustment Type") == "LSEG Pricing Only"]
    if len(changes) != 1 or not pricing or len(changes) + len(pricing) != len(on_date):
        return None
    record = changes[0][1]
    if record.get("Corporate Change Event Type") not in ("Share Split", "Share Consolidation"):
        return None
    try:
        ratio = float(record["Terms New Shares"]) / float(record["Terms Old Shares"])
        expected = 1.0
        for _, r in on_date:
            expected /= float(r["Adjustment Factor"])
    except (TypeError, ValueError, ZeroDivisionError, KeyError):
        return None
    price_factor = float(price_factor)
    if not (ratio > 0 and math.isfinite(ratio) and math.isfinite(expected) and price_factor > 0):
        return None
    disagreement = abs(expected - price_factor) / price_factor
    if disagreement > PRICING_ONLY_MAX_DISAGREEMENT:
        return None
    return [i for i, _ in on_date], ratio, disagreement
