"""Read-only source probes. Output contains shapes/coverage, never credentials."""

from __future__ import annotations

import argparse
import json

from backend.config import get_settings
from backend.ingestion.estimates import _open_session


def price_basis_sample(ld, ric, start, end):
    import pandas as pd

    frames = []
    for adjusted in (0, 1):
        df = ld.get_history(ric, fields=[f'TR.CLOSEPRICE(Adjusted={adjusted})'],
                            interval='1d', start=start, end=end)
        if len(df.columns) != 1:
            raise ValueError(f'Expected one price field, received {list(df.columns)}')
        frames.append(df.rename(columns={df.columns[0]: f'close_adjusted_{adjusted}'}))
    return pd.concat(frames, axis=1)


def probe(only=None):
    import lseg.data as ld

    out = {"version": 1, "sources": {}}
    s = get_settings()
    try:
        _open_session(s.lseg_app_key)
    except Exception as exc:
        return {"status": "blocked", "reason": type(exc).__name__}
    if only == 'original_security_actions':
        from pathlib import Path
        from scripts.pull_corporate_actions import FIELDS

        rics = json.loads(Path('docs/original_security_rics.json').read_text())
        try:
            actions = ld.get_data(rics, FIELDS, {'SDate': '2010-01-01', 'EDate': '2026-09-04'})
            dividends = ld.get_data(rics,
                ['TR.DivExDate', 'TR.DivPayDate', 'TR.DivUnadjustedGross', 'TR.DivCurr', 'TR.DivType'],
                {'SDate': '2010-01-01', 'EDate': '2026-09-04', 'DateType': 'XD'})
            return {'sources': {
                'corporate_actions': {'requested_rics': rics, 'failures': [],
                    'records': actions.astype(str).to_dict('records')},
                'dividends': {'requested_rics': rics, 'date_type': 'XD',
                    'records': dividends.astype(str).to_dict('records')},
            }, 'status': 'observed_requires_event_and_completeness_review'}
        finally:
            ld.close_session()
    if only == 'terminal_prices':
        samples = [('AET.N^K18', '2018-11-26', '2018-11-30'),
                   ('ESRX.OQ^L18', '2018-12-18', '2018-12-24'),
                   ('SCG.N^A19', '2018-12-27', '2019-01-03'),
                   ('BMS.N^F19', '2019-06-06', '2019-06-13'),
                   ('TWX.N^F18', '2018-06-12', '2018-06-18'),
                   ('ANDV.N^J18', '2018-09-26', '2018-10-03'),
                   ('CVS.N', '2018-11-28', '2018-11-29'),
                   ('CI.N', '2018-12-20', '2018-12-21'),
                   ('D.N', '2018-12-31', '2019-01-02'),
                   ('AMCR.N', '2019-06-11', '2019-06-12'),
                   ('T.N', '2018-06-14', '2018-06-15'),
                   ('MPC.N', '2018-09-28', '2018-10-01')]
        results = []
        try:
            for ric, start, end in samples:
                try:
                    df = price_basis_sample(ld, ric, start, end)
                    results.append(dict(ric=ric, records=df.reset_index().astype(str).to_dict('records')))
                except Exception as exc:
                    results.append(dict(ric=ric, error=type(exc).__name__, detail=str(exc)[:300]))
                print(f'terminal price probe: {ric}', flush=True)
        finally:
            ld.close_session()
        return {'sources': {'terminal_prices': results}}
    if only == 'historical_pricing_service':
        results = []
        try:
            for ric,start,end in [('ACS.N^B10','2010-01-01','2010-02-10'),
                                   ('AET.N^K18','2018-11-26','2018-11-30'),
                                   ('TWX.N^F18','2018-06-12','2018-06-18'),
                                   ('BMS.N^F19','2019-06-06','2019-06-13'),
                                   ('ADNT.N','2016-10-17','2016-10-24')]:
                try:
                    df = ld.get_history(ric, fields=['OPEN_PRC','HIGH_1','LOW_1','TRDPRC_1','ACVOL_UNS'],
                                        start=start,end=end,interval='1d',adjustments='unadjusted')
                    results.append(dict(ric=ric,records=df.reset_index().astype(str).to_dict('records')))
                except Exception as exc:
                    results.append(dict(ric=ric,error=type(exc).__name__,detail=str(exc)[:300]))
                print(f'historical pricing: {ric}',flush=True)
        finally:
            ld.close_session()
        return {'sources':{'historical_pricing_service':results}}
    if only == 'corporate_actions_archive':
        from pathlib import Path
        mapping = json.loads(Path('.research/verification/security-mapping-candidates-v2.json').read_text())
        affected = {r['ticker_id'] for r in mapping['unclassified_actions']}
        rics = sorted(r['ric'] for r in mapping['mappings']
                      if r['status']=='unique_candidate_requires_history_review'
                      and r['candidate_ticker_ids'][0] in affected)
        records, failures = [], []
        try:
            for start in range(0, len(rics), 20):
                batch = rics[start:start+20]
                try:
                    df = ld.get_data(batch,
                        ['TR.CACorpActEventType', 'TR.CACorpActDesc', 'TR.CAExDate',
                         'TR.CAEffectiveDate', 'TR.CAAdjustmentFactor', 'TR.CAAdjustmentType',
                         'TR.CATermsOldShares', 'TR.CATermsNewShares', 'TR.CAIsRescinded'],
                        {'SDate': '2010-01-01', 'EDate': '2026-09-07'})
                    records.extend(df.astype(str).to_dict('records'))
                    print(f'actions: {min(start+20,len(rics))}/{len(rics)} securities queried', flush=True)
                except Exception as exc:
                    failures.append({'rics':batch,'reason':type(exc).__name__,'detail':str(exc)[:300]})
        finally:
            ld.close_session()
        return {'sources': {'corporate_actions': {'requested_rics':rics, 'records':records,
                    'failures':failures, 'status':'partial' if failures else 'observed_requires_event_review'}}}
    if only in ('identity_archive', 'lifecycle_archive'):
        from pathlib import Path
        membership = json.loads(Path('.research/verification/membership-reconciled.json').read_text())
        rics = sorted({r['security_id'] for r in membership['intervals']})
        records, failures = [], []
        try:
            for start in range(0, len(rics), 40):
                batch = rics[start:start+40]
                try:
                    fields = ['TR.RIC', 'TR.ISIN', 'TR.CUSIP', 'TR.CIKNumber',
                              'TR.CommonName', 'TR.TickerSymbol']
                    if only == 'lifecycle_archive':
                        fields += ['TR.ExchangeTicker', 'TR.FirstTradeDate', 'TR.RetireDate']
                    df = ld.get_data(batch, fields)
                    records.extend(df.astype(str).to_dict('records'))
                    print(f'identity: {min(start+40,len(rics))}/{len(rics)} RICs queried', flush=True)
                except Exception as exc:
                    failures.append({'rics': batch, 'reason': type(exc).__name__, 'detail': str(exc)[:300]})
        finally:
            ld.close_session()
        return {'sources': {'identity_archive': {'requested_rics': rics, 'records': records,
                    'failures': failures, 'status': 'partial' if failures else 'observed_requires_dated_mapping'}}}
    if only == 'membership_archive':
        from datetime import date
        from backend.ingestion.index_lseg import build_jl_intervals

        try:
            current = ld.get_data('0#.SPX', ['TR.RIC'])
            records = []
            years = {}
            for year in range(2010, date.today().year + 1):
                frame = ld.get_data('.SPX', [
                    'TR.IndexJLConstituentRIC', 'TR.IndexJLConstituentRIC.date',
                    'TR.IndexJLConstituentRIC.change'],
                    {'SDate': f'{year}-01-01',
                     'EDate': min(f'{year}-12-31', date.today().isoformat()), 'IC': 'B'})
                batch = frame.astype(str).to_dict('records')
                records.extend(batch)
                years[str(year)] = len(batch)
                print(f'membership {year}: {len(batch)} events', flush=True)
            out['sources']['membership_archive'] = {
                'status': 'observed_requires_semantics_review', 'years': years,
                'current': current.astype(str).to_dict('records'), 'events': records}
            # Raw RIC identity is diagnostic only; never write these intervals to the DB.
            try:
                events = [dict(date=r['Date'][:10], security_id=r['Constituent RIC'],
                               change=r['Change']) for r in records]
                intervals = build_jl_intervals(current['RIC'].dropna().tolist(), events,
                                              '2010-01-01', date.today())
                out['sources']['membership_archive']['raw_ric_intervals'] = len(intervals)
            except (ValueError, KeyError) as exc:
                out['sources']['membership_archive']['reconciliation_error'] = str(exc)
        except Exception as exc:
            out['sources']['membership_archive'] = {'status': 'blocked',
                                                    'reason': type(exc).__name__, 'detail': str(exc)[:300]}
        finally:
            ld.close_session()
        return out
    tests = {
        'historical_ohlcv': lambda: ld.get_history(
            'ACS.N^B10', fields=['TR.OPENPRICE(Adjusted=0)', 'TR.HIGHPRICE(Adjusted=0)',
                                 'TR.LOWPRICE(Adjusted=0)', 'TR.CLOSEPRICE(Adjusted=0)',
                                 'TR.ACCUMULATEDVOLUME'],
            start='2010-01-01', end='2010-02-10', interval='1d'),
        'security_lifecycle': lambda: ld.get_data(
            ['AABA.OQ^J19', 'ABMD.OQ^L22', 'AET.N^K18', 'AGN.N^E20',
             'PLD.N', 'PLD.N^F11', 'BRKb.N', 'BFb.N', 'BNI.N^B10', 'AAPL.O'],
            ['TR.RIC', 'TR.ISIN', 'TR.CIKNumber', 'TR.CommonName',
             'TR.ExchangeTicker', 'TR.FirstTradeDate', 'TR.RetireDate']),
        'corporate_actions': lambda: ld.get_data(
            ['AAPL.O', 'NVDA.O', 'GE.N'],
            ['TR.CACorpActEventType', 'TR.CACorpActDesc', 'TR.CAExDate',
             'TR.CAEffectiveDate', 'TR.CAAdjustmentFactor', 'TR.CAAdjustmentType',
             'TR.CATermsOldShares', 'TR.CATermsNewShares', 'TR.CAIsRescinded'],
            {'SDate': '2010-01-01', 'EDate': '2026-09-07'}),
        "price_basis_aapl": lambda: price_basis_sample(ld, 'AAPL.O', '2020-08-25', '2020-09-04'),
        "price_basis_nvda": lambda: price_basis_sample(ld, 'NVDA.O', '2024-06-04', '2024-06-14'),
        "price_basis_ge": lambda: price_basis_sample(ld, 'GE.N', '2021-07-27', '2021-08-06'),
        "security_identity": lambda: ld.get_data(
            ['EVHC.N^L16', 'EVHC.N^J18', 'AAPL.O', 'GOOG.O', 'GOOGL.O'],
            ['TR.RIC', 'TR.ISIN', 'TR.CUSIP', 'TR.CIKNumber', 'TR.CommonName']),
        "fixed_estimates": lambda: ld.get_data(
            "AAPL.O",
            ["TR.EPSMean", "TR.EPSMean.date", "TR.EPSMean.fperiod"],
            {"SDate": "2019-01-01", "EDate": "2019-06-30", "Frq": "M", "Period": "FY1"},
        ),
        "membership": lambda: ld.get_data(
            ".SPX",
            [
                "TR.IndexJLConstituentRIC",
                "TR.IndexJLConstituentRIC.date",
                "TR.IndexJLConstituentRIC.change",
            ],
            {"SDate": "2010-01-01", "EDate": "2011-01-01", "IC": "B"},
        ),
        "dead_prices": lambda: ld.get_history(
            "BNI.N^B10", fields=["TR.PriceClose", "TR.Volume"], start="2010-01-01", end="2010-02-28"
        ),
        "historical_sectors": lambda: ld.get_data(
            "AAPL.O",
            ["TR.GICSSector", "TR.GICSSector.date"],
            {"SDate": "2012-01-01", "EDate": "2019-01-01", "Frq": "FY"},
        ),
    }
    for key, call in tests.items():
        if only and only != key and not (only == 'price_basis' and key.startswith('price_basis_')):
            continue
        try:
            df = call()
            out["sources"][key] = {
                "status": "observed_requires_semantics_review",
                "rows": len(df),
                "columns": [str(c) for c in df.columns],
                "non_null": {str(c): int(df[c].notna().sum()) for c in df.columns},
                "sample": df.head(3).astype(str).to_dict("records"),
            }
            if key.startswith('price_basis_') or key in ('security_identity', 'security_lifecycle', 'corporate_actions', 'historical_ohlcv'):
                out['sources'][key]['records'] = df.reset_index().astype(str).to_dict('records')
        except Exception as exc:
            out["sources"][key] = {"status": "blocked", "reason": type(exc).__name__, 'detail': str(exc)[:300]}
    if only in ('price_basis', 'security_identity', 'security_lifecycle', 'corporate_actions', 'historical_ohlcv'):
        ld.close_session()
        return out
    try:
        from datetime import datetime, timezone

        from backend.ingestion.news_lseg import fetch_window

        rows = fetch_window(
            ld,
            "AAPL.O",
            datetime(2013, 3, 1, tzinfo=timezone.utc),
            datetime(2013, 3, 3, tzinfo=timezone.utc),
        )
        out["sources"]["news"] = {
            "status": "observed_requires_semantics_review",
            "rows": len(rows),
            "fields": sorted(rows[0]) if rows else [],
            "version_timestamps": [str(r.get("version_created")) for r in rows[:3]],
        }
    except Exception as exc:
        out["sources"]["news"] = {
            "status": "blocked",
            "reason": type(exc).__name__,
            "detail": str(exc)[:300],
        }
    finally:
        ld.close_session()
    out["sources"]["alfred"] = {
        "status": "configured_requires_vintage_probe" if s.fred_api_key else "blocked",
        "reason": "FRED_API_KEY presence only",
    }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output")
    p.add_argument(
        "--only",
        choices=("news", "fixed_estimates", "membership", "membership_archive", "dead_prices", "historical_sectors", "historical_ohlcv", "historical_pricing_service", "price_basis", "terminal_prices", "security_identity", "security_lifecycle", "corporate_actions", "identity_archive", "lifecycle_archive", "corporate_actions_archive", "original_security_actions"),
    )
    args = p.parse_args()
    result = probe(args.only)
    if args.output:
        from backend.ml.research import write_json_new

        write_json_new(args.output, result)
    if args.output and args.only in ('membership_archive', 'identity_archive', 'lifecycle_archive', 'corporate_actions_archive'):
        print(f'Full archive saved to {args.output}')
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
