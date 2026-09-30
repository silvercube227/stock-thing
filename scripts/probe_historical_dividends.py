"""Read-only historical dividend probe, explicitly filtered by ex-date."""
import argparse
import json
from pathlib import Path

from backend.config import get_settings
from backend.ingestion.estimates import _open_session
from backend.ml.research import write_json_new


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--rics',help='JSON list of exact historical RICs; defaults to the feasibility sample')
    parser.add_argument('--start',default='2010-01-01')
    parser.add_argument('--end',default='2026-09-04')
    args=parser.parse_args()
    ld=_open_session(get_settings().lseg_app_key)
    fields=['TR.DivExDate','TR.DivPayDate','TR.DivUnadjustedGross','TR.DivCurr','TR.DivType']
    parameters={'SDate':args.start,'EDate':args.end,'DateType':'XD'}
    try:
        rics=json.loads(Path(args.rics).read_text()) if args.rics else ['ADNT.N','ANF.N','BNI.N^B10','AAPL.O']
        records=[]
        for offset in range(0,len(rics),20):
            df=ld.get_data(rics[offset:offset+20],fields,parameters)
            records.extend(df.astype(str).to_dict('records'))
            print(f'dividends {min(offset+20,len(rics))}/{len(rics)}',flush=True)
        report=dict(fields=fields,parameters=parameters,
                    requested_rics=rics,records=records,
                    status='observed_requires_cash_type_and_currency_review')
        write_json_new(args.output,report)
        print({'rows':len(records),'columns':list(df.columns)})
    finally:
        ld.close_session()
