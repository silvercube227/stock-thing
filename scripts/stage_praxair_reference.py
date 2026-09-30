"""Stage the documented PX/LIN separation offline; no source-gate promotion.

Replays native prices and membership sources, verifies merger anchors, and
exercises terminal labels. The candidate remains separate from the database and
production cache while distribution completeness and reference certification are
unresolved.
"""

import argparse
import copy
from datetime import date, timedelta
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle

from backend.ingestion.calendar import HORIZON_TRADING_DAYS, shift_trading_days, trading_days_between
from backend.ingestion.dividend_review import review_dividends
from backend.ingestion.prices_lseg import normalize_vendor_pair
from backend.ingestion.security_intervals import split_predecessor_intervals
from backend.ml.factors.pit import as_traded_closes, day, documented_targets
from backend.ml.research import create_snapshot, load_snapshot, write_json_new
from backend.ml.terminal import successor_horizon_values
from scripts.audit_vendor_adjustments import changes, load_rows

FINAL = date(2018, 10, 30)
BOUNDARY = date(2018, 10, 31)


def run(snapshot, output):
    output = Path(output)
    if output.exists():
        raise ValueError('Candidate already exists')
    inputs, manifest = load_snapshot(snapshot)
    frames = copy.deepcopy(inputs['frames'])
    by_symbol = {f.symbol:f for f in frames}
    target, successor = by_symbol['LSEG:PX.N^J18'], by_symbol['LIN']
    if target.ticker_id != 947 or target.prices or target.membership or target.security_events:
        raise ValueError('Expected the registered, unapplied predecessor')
    hashes = {}

    def read(path):
        raw = Path(path).read_bytes()
        hashes[path] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    base = '.research/verification/'
    source = read(base+'praxair-source-archived.json')['events'][0]
    content = Path(source['archived_source']).read_bytes()
    if hashlib.sha256(content).hexdigest() != source['source_sha256']:
        raise ValueError('Merger document changed')
    hashes[source['archived_source']] = source['source_sha256']
    if (source['last_trade_date'] != str(FINAL) or source['effective_date'] != str(BOUNDARY)
            or source['terms'] != dict(cash_per_share=0, successor_shares_per_share=1)):
        raise ValueError('Merger review changed')
    raw = load_rows(read(base+'praxair-raw-history.json')['securities'][0])
    adjusted = load_rows(read(base+'praxair-adjusted-history.json')['securities'][0])
    if raw['request']['adjustments'] != 'unadjusted' or adjusted['request']['adjustments'] != ['CCH','CRE','RPO','RTS']:
        raise ValueError('Unexpected adjustment basis')
    dividends = review_dividends(read(base+'praxair-dividends.json')['records'],
                                [r['Date'][:10] for r in raw['records']])
    capital = changes(raw['records'], adjusted['records'])
    if dividends['issues'] or capital['events']:
        raise ValueError('New distribution or capital adjustment requires review')
    prices = normalize_vendor_pair(target.ticker_id, raw['records'], adjusted['records'],
                                   dividends['cash_by_ex_date'])
    price_report = read('.research/praxair-normalized-candidate/report.json')
    stored = gzip.decompress(Path('.research/praxair-normalized-candidate/prices.pkl.gz').read_bytes())
    if hashlib.sha256(stored).hexdigest() != price_report['input_sha256'] or pickle.loads(stored) != prices:
        raise ValueError('Praxair normalization candidate does not reproduce')
    if prices[-1]['trade_date'] != FINAL:
        raise ValueError('Predecessor final session mismatch')
    expected = list(trading_days_between(date(2010, 1, 4), FINAL))
    if [r['trade_date'] for r in prices] != expected:
        raise ValueError('Predecessor price session gap')
    parent = read(base+'membership-reconciled.json')['intervals']
    split = split_predecessor_intervals(parent, predecessor='PX.N^J18', successor='LIN.OQ',
        effective_date=BOUNDARY, evidence=source['source_sha256'])
    dates = lambda row: dict(valid_from=day(row['valid_from']),
        valid_to=day(row['valid_to']) if row['valid_to'] else None)
    # Reconstruct from unsplit evidence rather than trusting the previous candidate.
    target.membership = [dict(**dates(r), index_id='SPX') for r in split if r['security_id'] == 'PX.N^J18']
    successor.membership = [dict(**dates(r), index_id='SPX') for r in split if r['security_id'] == 'LIN.OQ']
    if target.membership != [dict(valid_from=date(2010,1,1), valid_to=BOUNDARY, index_id='SPX')]:
        raise ValueError('Unexpected predecessor membership episodes')
    target.prices = prices
    old_successor_prices = successor.prices
    successor.prices = [p for p in old_successor_prices if p['trade_date'] >= BOUNDARY]
    # No current industry/sector backfill. Only the separate reviewed Materials
    # intervals are staged; the wider sector reconstruction remains uncertified.
    reviewed = read(base+'praxair-separated-interval-candidate.json')
    for frame, ric in [(target,'PX.N^J18'), (successor,'LIN.OQ')]:
        frame.sector_history = [dict(**dates(r), sector='Materials', industry=None)
            for r in reviewed['materials_intervals'] if r['security_id'] == ric]
        if not frame.sector_history:
            raise ValueError('Missing dated Materials candidate')
    probes = read(base+'praxair-price-transition-probe.json')['securities']
    anchor_checks = []
    for frame, ric, when in [(target,'PX.N^J18',FINAL),(successor,'LIN.OQ',BOUNDARY)]:
        native = next(float(r['TRDPRC_1']) for s in probes if s['ric'] == ric
                      for r in s['records'] if r['Date'] == str(when))
        computed = dict(zip([p['trade_date'] for p in frame.prices], as_traded_closes(frame.prices)))
        actual = computed[when]
        if not math.isclose(native, actual, rel_tol=1e-6):
            raise ValueError(f'Unverified merger price anchor: {ric}')
        anchor_checks.append(dict(ric=ric, date=when, native=native, computed=actual))
    observation_end = max(p['trade_date'] for f in frames for p in f.prices)
    ends = {shift_trading_days(p['trade_date'], steps) for p in prices
            if p['trade_date'] >= FINAL-timedelta(days=400) for steps in HORIZON_TRADING_DAYS.values()}
    ends = {d for d in ends if BOUNDARY <= d <= observation_end}
    valuation = successor_horizon_values(target_prices=prices, successor_prices=successor.prices,
        target_last_trade=FINAL, ownership_close=BOUNDARY, horizon_dates=ends,
        cash_per_share=0, exchange_ratio=1)
    if valuation['missing_horizons']:
        raise ValueError('Missing exact successor valuation sessions')
    event = dict(effective_date=BOUNDARY, event_type='acquisition', consideration_type='successor',
        proceeds_basis='adj_close', horizon_values=valuation['horizon_values'], verified=True,
        source=json.dumps(dict(source, last_trade_date=str(FINAL), stage='offline_candidate_only')))
    target.security_events = [event]
    target.removed_at = BOUNDARY
    before, after = {}, {}
    for i in range(len(prices)-1):
        if prices[i]['trade_date'] < FINAL-timedelta(days=400):
            continue
        original = documented_targets(prices, i, [], observation_end)
        corrected = documented_targets(prices, i, [event], observation_end)
        for horizon in HORIZON_TRADING_DAYS:
            before[horizon] = before.get(horizon,0)+int(original[horizon][1])
            after[horizon] = after.get(horizon,0)+int(corrected[horizon][1])
    metadata = copy.deepcopy(manifest['metadata'])
    metadata.update(stage='uncertified_praxair_reference_candidate',
                    parent_input_sha256=manifest['input_sha256'], source_hashes=hashes)
    created = create_snapshot(output, frames, inputs['macro'], metadata)
    report = dict(status='offline_candidate_not_certified', database_writes=0,
        source_hashes=hashes, parent_input_sha256=manifest['input_sha256'],
        candidate_input_sha256=created['input_sha256'], predecessor_rows=len(prices),
        removed_predecessor_bars_from_successor=sum(p['trade_date'] < BOUNDARY for p in old_successor_prices),
        anchor_checks=anchor_checks, membership=target.membership,
        terminal_horizons=len(valuation['horizon_values']), labels_without_terminal=before,
        labels_with_terminal_candidate=after,
        pending=['Praxair distribution-completeness certification',
                 'Successor price/action certification across all valuation dates',
                 'Full parent/sector reconciliation and security-identity coverage',
                 'Exact rollback rehearsal before database application'])
    write_json_new(output/'praxair-review.json', report)
    print({k:v for k,v in report.items() if k not in ('source_hashes','anchor_checks','pending')})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    run(a.snapshot, a.output)
