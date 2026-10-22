"""Synthetic offline tests. No model calls, real credentials, or private patient data."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tools')]
from common import rows,sha,jev_hash,save
from validate_data import validate
from prepare_inputs import prepare
from evaluate_jev import main as api_main
from report import report

def fixture(root,n=24):
    category='B_synthetic'; ds='synthetic'; base=root/category/ds
    arms=['choice_letters','choice_numbers','choice_shuffled','binary_ovr','choice_none','choice_abstain']
    gold=[{'id':f'p{i}','gold':{'label':'A' if i%2==0 else 'B'}} for i in range(n)]
    def write(p,rs):
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(''.join(json.dumps(r)+'\n' for r in rs))
    write(base/'answers.jsonl',gold); variants=[]
    for arm in arms:
        req=[];maps=[]
        for i in range(n):
            pid=f'p{i}'; candidates={k:'Synthetic candidate '+k for k in 'ABCD'}
            align={k:k for k in candidates}
            if arm=='choice_numbers':align={str(j+1):k for j,k in enumerate(candidates)}
            if arm=='choice_shuffled':align=dict(zip('ABCD','CDAB'))
            if arm=='choice_abstain':align['E']='E';candidates['E']='ABSTAIN'
            criteria={k:candidates[v] for k,v in align.items()}
            if arm=='choice_none':criteria[gold[i]['gold']['label']]='none of the above'
            mapping={'parent_id':pid,'display_to_canonical':align,'canonical_to_display':{v:k for k,v in align.items()}}
            if arm=='choice_abstain':mapping['none_id']='E'
            if arm=='binary_ovr':
                mapping={'parent_id':pid,'option_to_query_id':{k:pid+'__'+k for k in candidates}}
                qq=[(pid+'__'+k,{'type':'noul','instructions':'Judge the supplied candidate.',
                   'criteria':{'true':'Candidate '+k+' is correct.','false':'Candidate '+k+' is not correct.'}}) for k in candidates]
            else:qq=[(pid,{'type':'choice','instructions':'Select the matching synthetic candidate.','criteria':criteria})]
            for rid,q in qq:
                body={'model':'synthetic','state':{'case':'Artificial fixture '+str(i)},'questions':{'decision':q}}
                req.append({'id':rid,'parent_id':pid,'body':body,'request_sha256':jev_hash(body)})
            maps.append(mapping)
        dest=base/arm
        for name,rs in [('jev_requests.jsonl',req),('answers.jsonl',gold),('mappings.jsonl',maps)]:write(dest/name,rs)
        variants.append({'variant':arm,'parent_case_count':n,'request_count':len(req),
                         'file_sha256':{p.name:sha(p) for p in dest.iterdir()}})
    entry={'dataset':ds,'category':category,'path':category+'/'+ds,'count':n,'classes':4,
           'answers_file_sha256':sha(base/'answers.jsonl'),'variants':variants}
    save(root/'manifest.json',{'scope':'SYNTHETIC_TEST_NOT_MODEL_RESULTS','dataset_count':1,
         'unique_case_count':n,'request_count':sum(s['request_count'] for s in variants),'datasets':[entry]})

class FakeOpener:
    def __init__(self):self.calls=0
    def open(self,request,timeout):
        self.calls+=1; b=json.loads(request.data);q=b['questions']['decision']
        if q['type']=='noul':answer={'type':'noul','noul':.8 if 'Candidate A ' in q['criteria']['true'] else .1}
        else:
            keys=list(q['criteria']);ps={k:(.9 if j==0 else .1/(len(keys)-1)) for j,k in enumerate(keys)}
            answer={'type':'choice','choice':keys[0],'probabilities':ps}
        return io.BytesIO(json.dumps({'model':'synthetic-version','answers':{'decision':answer}}).encode())

class EndToEnd(unittest.TestCase):
    def test_mock_api_resume_sharding_and_four_dimensions(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'data';fixture(data)
            self.assertEqual(validate(data)['parents'],24)
            with contextlib.redirect_stdout(io.StringIO()):prepare(data,root/'native',3)
            plan=json.loads((root/'native/inputs/plan.json').read_text())
            self.assertEqual(plan['requests'],216)
            self.assertEqual(sum(s['parents'] for s in plan['shards']),24)
            a=argparse.Namespace(profile=None,provider='openrouter',model='synthetic-model',
                expected_response_model='synthetic-version',data=data,output=root/'run',workers=4,rps=1e6,
                timeout=2,case_limit=None,datasets=None,variants=None,execute=True,
                allow_external=True,allow_restricted=False,retry_length_errors=False)
            opener=FakeOpener()
            with patch.dict(os.environ,{'OPENROUTER_API_KEY':'synthetic-unit-test-only'}),patch('urllib.request.build_opener',return_value=opener),contextlib.redirect_stdout(io.StringIO()):
                api_main(a);self.assertEqual(opener.calls,216)
                api_main(a);self.assertEqual(opener.calls,216)
                r=argparse.Namespace(interface='jev',data=data,run=root/'run',name='Synthetic',output=root/'report',exclude_length_errors=False)
                self.assertTrue(report(r))
            result=json.loads((root/'report/comparison.json').read_text())['models']['Synthetic']
            self.assertAlmostEqual(result['categories']['Overall']['choice_letters']['accuracy'],.5)
            self.assertEqual(result['global_top']['binary_ovr']['5']['n'],1)
            self.assertAlmostEqual(result['global_top']['binary_ovr']['5']['coverage'],1/24)
            self.assertEqual(result['none']['total_n'],24)
            self.assertEqual(result['robustness']['choice_shuffled']['flipped'],1)
            self.assertTrue(json.loads((root/'report/audit.json').read_text())['complete'])
            # A recorded length error invalidates an OBS parent; it is not counted wrong.
            events=root/'run/synthetic/binary_ovr/events.jsonl';items=list(rows(events))
            err=dict(items[0],status='error',error_kind='context_length',error='synthetic length limit')
            with events.open('a') as f:f.write(json.dumps(err)+'\n')
            with contextlib.redirect_stdout(io.StringIO()):
                r.output=root/'failed-report';self.assertFalse(report(r))
                r.output=root/'excluded-report';r.exclude_length_errors=True;self.assertTrue(report(r))
            ex=json.loads((r.output/'comparison.json').read_text())
            self.assertFalse(ex['complete']);self.assertEqual(ex['scored_parents'],23)
            self.assertEqual(len(ex['excluded_parents']),1)
            self.assertIn('NOT_full_paper',ex['scope'])

    def test_tampered_input_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            data=Path(temp);fixture(data)
            p=next(data.rglob('choice_letters/jev_requests.jsonl'))
            p.write_text(p.read_text().replace('candidate','CANDIDATE'))
            with self.assertRaises(AssertionError):validate(data)

if __name__=='__main__':unittest.main()
