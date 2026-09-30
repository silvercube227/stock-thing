"""Review source dividend rows without treating empty responses as zero dividends."""
from datetime import date
from decimal import Decimal, InvalidOperation

CASH_TYPES={'Interim','Final','Special','Extra','Capital gains'}


def review_dividends(records, price_dates):
    dates=set(price_dates)
    seen=set()
    events={}
    issues=[]
    outside=[]
    empty=0
    for row in records:
        d=row.get('Dividend Ex Date','')[:10]
        if d in ('','NaT','<NA>'):
            empty+=1
            continue
        try:date.fromisoformat(d)
        except ValueError:
            issues.append(dict(date=d,reason='invalid_ex_date'))
            continue
        if not dates or d<min(dates) or d>max(dates):
            outside.append(d)
            continue
        if d not in dates:
            issues.append(dict(date=d,reason='ex_date_has_no_price_bar'))
            continue
        if row.get('Dividend Currency')!='USD' or row.get('Dividend Type') not in CASH_TYPES:
            issues.append(dict(date=d,reason='currency_or_dividend_type_unverified'))
            continue
        try:
            amount=Decimal(row.get('Gross Dividend Amount',''))
            if not amount.is_finite() or amount<0:raise InvalidOperation
        except InvalidOperation:
            issues.append(dict(date=d,reason='invalid_or_missing_amount'))
            continue
        key=(d,row['Dividend Type'])
        identity=(*key,row.get('Dividend Pay Date'),amount)
        if identity in seen:continue
        seen.add(identity)
        if key in events:
            issues.append(dict(date=d,reason='conflicting_same_type_events'))
        else:events[key]=amount
    bad_dates={r['date'] for r in issues}
    amounts={}
    for (d,_),amount in events.items():
        if d not in bad_dates:amounts[d]=amounts.get(d,Decimal(0))+amount
    return dict(cash_by_ex_date={d:str(v) for d,v in sorted(amounts.items())},
                issues=issues,empty_rows=empty,outside_price_history=sorted(set(outside)),
                status='source_issues' if issues else 'parsed_requires_completeness_verification')
