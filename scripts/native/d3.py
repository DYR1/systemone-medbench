"""One PPU, one native D3 replica; unmodified Jev state/questions, no gold."""
import argparse
import datetime
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
from common import ROOT, checkpoint_fingerprint, digest, jev_hash, parse_jev, rows, save, sha

def main(args):
    inp = ROOT/'inputs'/('pilot.jsonl' if args.pilot else f'shard{args.shard:02d}.jsonl')
    out = ROOT/'runs'/('pilot' if args.pilot else f'shard{args.shard:02d}')
    out.mkdir(parents=True, exist_ok=True)
    lock=(out/'run.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    records=list(rows(inp)); plan=json.loads((ROOT/'inputs/plan.json').read_text())
    expected=plan['pilot_sha256'] if args.pilot else plan['shards'][args.shard]['sha256']
    assert sha(inp)==expected
    modelroot=Path(args.model)
    config={'input_sha256':expected,'manifest_sha256':plan['manifest_sha256'],
        'runtime_hashes':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        'model_package_sha256':sha(modelroot/'MODEL_MANIFEST.json'),
        'checkpoint_fingerprint':checkpoint_fingerprint(modelroot),
        'expected_response_model':args.response_model,
        'versions':{n:importlib.metadata.version(n) for n in ('torch','transformers','tokenizers')},
        'decision_config_sha256':sha(modelroot/'decision_config.json'),
        'native_code_hashes':{p.name:sha(p) for p in modelroot.glob('*.py')},
        'model':str(modelroot),'protocol':'native D3.system_one prepare/run/respond; no permutation averaging; no truncation',
        'batch_size':1,'D3_FAST':os.environ.get('D3_FAST'), 'device':'cuda:0'}
    config['run_hash']=digest(config)
    if (out/'config.json').exists(): assert json.loads((out/'config.json').read_text())==config, 'Changed run fingerprints'
    save(out/'config.json',config)
    done={}
    if (out/'events.jsonl').exists():
        for e in rows(out/'events.jsonl'):
            assert e['run_hash']==config['run_hash']
            done[e['dataset'],e['variant'],e['id']]=e
    started=time.monotonic()
    def progress(stage, **extra):
        save(out/'progress.json',dict(stage=stage,pid=os.getpid(),planned=len(records),
            successful=sum(e['status']=='ok' for e in done.values()),errors=sum(e['status']!='ok' for e in done.values()),
            elapsed_seconds=time.monotonic()-started,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra))
    progress('loading')
    sys.path.insert(0,str(modelroot))
    import torch
    import transformers
    from d3_runtime import D3
    try:
        model=D3.from_pretrained(modelroot,device='cuda:0',batch_size=1,verify='fast')
        save(out/'model_info.json',{'runtime':model.runtime_info(),'torch':torch.__version__,
            'transformers':transformers.__version__,'device_name':torch.cuda.get_device_name(0),
            'decision_config':model.config,'permutation_average':model.permutation_average,
            'parameter_count':model.parameter_count()})
        assert not model.permutation_average
        progress('running')
        with (out/'events.jsonl').open('a',buffering=1) as f:
            for r in records:
                key=r['dataset'],r['variant'],r['id']
                if key in done and done[key]['status']=='ok': continue
                assert jev_hash(r['body'])==r['request_sha256']
                body=r['body']; before=json.dumps(body,ensure_ascii=False)
                ev={k:r[k] for k in ['dataset','variant','id','parent_id','request_sha256']}
                ev.update(run_hash=config['run_hash'],request=body,full_input_sent=True)
                t=time.monotonic()
                try:
                    # This is exactly D3.system_one's three native operations.
                    prepared=model.prepare(body['state'],body['questions'])
                    if 'decision' in prepared.questions:
                        rendered=model.text(body['state'],prepared.questions['decision'].rendered)
                        ev['rendered_sha256']=__import__('hashlib').sha256(rendered.encode()).hexdigest()
                        ev['input_tokens']=len(model.tokenizer.encode(rendered,add_special_tokens=False))
                    ev['input_token_ids_sha256']=digest(prepared.sequences)
                    if args.pilot and 'decision' in prepared.questions: ev['rendered_prompt']=rendered
                    probs,tokens=model.run(prepared)
                    response=model.respond(prepared,probs,tokens)
                    model.synchronize()
                    ev['response']=response
                    ev['native_probabilities']=probs
                    ans=response['answers']['decision']
                    if 'error' in ans: raise ValueError(ans['error']+': '+ans.get('message',''))
                    p,pred=parse_jev(response,body['questions']['decision'])
                    assert response['model']==args.response_model
                    assert p[pred] == max(p.values()), 'Native choice is not probability argmax'
                    ev.update(status='ok',probabilities=p,prediction=pred)
                except Exception as exc:
                    ev.update(status='error',error=str(exc),traceback=traceback.format_exc(),
                        error_kind='oom' if isinstance(exc,torch.OutOfMemoryError) else 'inference_error')
                    if isinstance(exc,torch.OutOfMemoryError): torch.cuda.empty_cache()
                assert before==json.dumps(body,ensure_ascii=False), 'Native runtime mutated request'
                ev.update(elapsed_seconds=time.monotonic()-t,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
                f.write(json.dumps(ev,ensure_ascii=False,allow_nan=False)+'\n'); done[key]=ev
                progress('running',current=list(key))
                if ev['status']!='ok' and args.pilot: raise RuntimeError('Pilot failed; inspect events.jsonl')
                if sum(e['status']!='ok' for e in list(done.values())[-5:])==5:
                    raise RuntimeError('Five consecutive errors; stop to diagnose')
        progress('complete' if all(e['status']=='ok' for e in done.values()) else 'complete_with_errors')
    except BaseException as exc:
        progress('failed',error=str(exc)); raise

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--response-model',required=True); p.add_argument('--shard',type=int,default=0); p.add_argument('--pilot',action='store_true')
    main(p.parse_args())
