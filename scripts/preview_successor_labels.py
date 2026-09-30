"""Replay reviewed stock deals against an immutable snapshot; no database writes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from backend.ingestion.calendar import HORIZON_TRADING_DAYS, shift_trading_days
from backend.ml.factors.pit import day, documented_targets
from backend.ml.research import load_snapshot, write_json_new
from backend.ml.terminal import successor_horizon_values


def run(snapshot, reviews, output):
    inputs, manifest = load_snapshot(snapshot)
    frames = {f.symbol: f for f in inputs['frames']}
    raw = Path(reviews).read_bytes()
    reviews_data = json.loads(raw)
    report = dict(snapshot=snapshot, input_sha256=manifest['input_sha256'],
                  review_sha256=hashlib.sha256(raw).hexdigest(), events=[], blocked=[],
                  status='preview_only_identity_and_source_price_checks_required')
    observation_end = max(day(p['trade_date']) for f in frames.values() for p in f.prices)
    for review in reviews_data['events']:
        symbol, terms = review['symbol'], review['terms']
        if terms['consideration_type'] != 'successor':
            report['blocked'].append(dict(symbol=symbol, reason='Election/proration precision unresolved'))
            continue
        content = Path(review['archived_source']).read_bytes()
        if hashlib.sha256(content).hexdigest() != review['source_sha256']:
            raise ValueError(f'Source hash mismatch: {symbol}')
        target, successor = frames[symbol], frames[terms['successor_symbol']]
        final = day(review['last_trade_date'])
        prices = [p for p in target.prices if day(p['trade_date']) <= final]
        ends = set()
        positions = []
        for pos in range(len(prices)-1):
            entry = day(prices[pos+1]['trade_date'])
            if entry != shift_trading_days(day(prices[pos]['trade_date']), 1):
                continue
            horizons = [shift_trading_days(entry, n) for n in HORIZON_TRADING_DAYS.values()]
            if any(final < end <= observation_end for end in horizons):
                ends.update(end for end in horizons if final < end <= observation_end)
                positions.append(pos)
        valuation = successor_horizon_values(
            target_prices=prices, successor_prices=successor.prices,
            target_last_trade=final, ownership_close=review['ownership_close'],
            horizon_dates=ends, cash_per_share=terms['cash_per_share'],
            exchange_ratio=terms['exchange_ratio'])
        # Verified here describes a synthetic test event, not a DB/source certification.
        event = dict(effective_date=review['effective_date'], event_type='acquisition',
                     consideration_type='successor', proceeds_basis='adj_close',
                     source=json.dumps(review, sort_keys=True), verified=True,
                     horizon_values=valuation['horizon_values'])
        counts, samples = Counter(), []
        for pos in positions:
            labels = documented_targets(prices, pos, [event], observation_end)
            for horizon, label in labels.items():
                if label[4] == 'acquisition':
                    counts[f'{horizon}_known' if label[1] else f'{horizon}_unknown'] += 1
            if pos == positions[-1]:
                samples.append(dict(prediction_date=prices[pos]['trade_date'], labels=labels))
        report['events'].append(dict(symbol=symbol, ticker_id=target.ticker_id,
                                      successor_ticker_id=successor.ticker_id,
                                      last_trade_date=review['last_trade_date'],
                                      placeholder_or_later_bars_excluded=len(target.prices)-len(prices),
                                      valuation=valuation, label_counts=dict(counts), samples=samples))
    write_json_new(output, report)
    print({e['symbol']:e['label_counts'] for e in report['events']})
    print({'blocked':report['blocked'], 'labels_applied':0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--reviews', default='docs/terminal_event_research.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.snapshot, args.reviews, args.output)
