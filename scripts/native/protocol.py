"""Content-preserving adapters; no gold/reference data is accepted here."""
import copy
import json

def text(value):
    return value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,allow_nan=False)

def labels(body):
    q=body['questions']['decision']
    return list(q['criteria']) if q['type']=='choice' else ['A','B']

def options(body):
    q=body['questions']['decision']
    return list(q['criteria'].items()) if q['type']=='choice' else [('A',q['criteria']['true']),('B',q['criteria']['false'])]

def portable_view(body):
    return {'state':copy.deepcopy(body['state']),
            'question':json.dumps(body['questions']['decision'],ensure_ascii=False,allow_nan=False),
            'options':[{'id':k,'description':text(v)} for k,v in options(body)]}
