"""Probe dated sector-index membership; never backdate current GICS fields."""
import argparse

from backend.config import get_settings
from backend.ingestion.estimates import _open_session
from backend.ml.research import write_json_new


def run(output):
    ld = _open_session(get_settings().lseg_app_key)
    requests = [
        dict(name='index_identity', universe=['.SPLRCT', '.SPLRCL'],
             fields=['TR.CommonName'], parameters={}),
        dict(name='technology_2018_events', universe='.SPLRCT',
             fields=['TR.IndexJLConstituentRIC', 'TR.IndexJLConstituentRIC.date',
                     'TR.IndexJLConstituentRIC.change'],
             parameters={'SDate': '2018-01-01', 'EDate': '2018-12-31', 'IC': 'B'}),
        dict(name='communication_2018_events', universe='.SPLRCL',
             fields=['TR.IndexJLConstituentRIC', 'TR.IndexJLConstituentRIC.date',
                     'TR.IndexJLConstituentRIC.change'],
             parameters={'SDate': '2018-01-01', 'EDate': '2018-12-31', 'IC': 'B'}),
    ]
    for index in ['.SPLRCT', '.SPLRCL']:
        for date in ['2018-08-31', '2018-10-01']:
            requests.append(dict(name=f'{index}_{date}_constituents', universe=index,
                                 fields=['TR.IndexConstituentRIC'],
                                 parameters={'SDate': date}))
    results = []
    try:
        for request in requests:
            try:
                frame = ld.get_data(request['universe'], request['fields'], request['parameters'])
                result = dict(request=request, status='observed_requires_semantics_review',
                              columns=list(frame.columns), records=frame.astype(str).to_dict('records'))
            except Exception as exc:
                result = dict(request=request, status='error', error=str(exc))
            results.append(result)
            print(request['name'], result['status'], len(result.get('records', [])), flush=True)
    finally:
        ld.close_session()
    write_json_new(output, dict(status='feasibility_only_not_certified', results=results,
                                database_writes=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
