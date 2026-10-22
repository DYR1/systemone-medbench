"""Portable wrapper of the original StartLuxDecision native Choice/Noul adapter."""
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
from common import ROOT, PACKAGE, checkpoint_fingerprint, digest, jev_hash, parse_jev, rows, save, sha

def main(a):
    plan=json.loads((ROOT/'inputs/plan.json').read_text())
    inp=ROOT/'inputs'/('pilot.jsonl' if a.pilot else f'shard{a.shard:02d}.jsonl')
    expected=plan['pilot_sha256'] if a.pilot else plan['shards'][a.shard]['sha256']
    assert sha(inp)==expected
    out=ROOT/'runs'/('pilot' if a.pilot else f'shard{a.shard:02d}'); out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=Path(os.environ['MEDBENCH_UPSTREAM_SOURCE']); records=list(rows(inp))
    if not (source/'startlux_decision/model.py').is_file():
        raise ValueError('--source must contain the official startlux_decision package; see docs/MODELS.md')
    cfg={'input_sha256':expected,'manifest_sha256':plan['manifest_sha256'],
         'runtime_hashes':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
         'native_code_hashes':{str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')},
         'checkpoint_fingerprint':checkpoint_fingerprint(a.model), 'model':a.model,
         'model_name':a.response_model,'protocol':'StartLuxDecision.decide native Choice/Noul; graphs=False; images=False',
         'context_limit':32768,'no_truncation':True,'batch_size':1,
         'versions':{n:importlib.metadata.version(n) for n in ['torch','transformers','tokenizers']}}
    cfg['run_hash']=digest(cfg)
    if (out/'config.json').exists(): assert json.loads((out/'config.json').read_text())==cfg
    else: save(out/'config.json',cfg)
    done={}
    if (out/'events.jsonl').exists():
        for e in rows(out/'events.jsonl'):
            assert e['run_hash']==cfg['run_hash'];done[e['dataset'],e['variant'],e['id']]=e
    t0=time.monotonic()
    def progress(stage,**extra):
        save(out/'progress.json',dict(stage=stage,pid=os.getpid(),planned=len(records),
            successful=sum(e['status']=='ok' for e in done.values()),errors=sum(e['status']!='ok' for e in done.values()),
            elapsed_seconds=time.monotonic()-t0,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra))
    progress('loading'); sys.path.insert(0,str(source))
    import torch
    from startlux_decision import StartLuxDecision, jevfmt
    torch.set_num_threads(4);torch.manual_seed(1234)
    try:
        engine=StartLuxDecision(a.model,device='cuda',graphs=False,images=False,max_length=32768)
        progress('running'); consecutive=0
        with (out/'events.jsonl').open('a',buffering=1) as f:
            for r in records:
                key=r['dataset'],r['variant'],r['id']
                if done.get(key,{}).get('status')=='ok':continue
                body=r['body'];before=jev_hash(body);assert before==r['request_sha256']
                ev={k:r[k] for k in ['dataset','variant','id','parent_id','request_sha256']}
                ev.update(run_hash=cfg['run_hash'],request=body,full_input_sent=True);start=time.monotonic()
                try:
                    row=jevfmt.from_systemone(body['state'],body['questions']['decision'],'decision')
                    ids,order=jevfmt.render_ids(row,engine.tok,max_length=32768)
                    ev.update(input_tokens=len(ids),input_token_ids_sha256=digest(ids),option_order=order)
                    if a.pilot:ev['rendered_token_ids']=ids
                    answers,usage=engine.decide(body['state'],body['questions']);torch.cuda.synchronize()
                    response={'answers':answers,'usage':usage,'model':a.response_model}; ev['response']=response
                    p,pred=parse_jev(response,body['questions']['decision'])
                    ev.update(status='ok',probabilities=p,prediction=pred);consecutive=0
                except Exception as ex:
                    ev.update(status='error',error=str(ex),traceback=traceback.format_exc(),
                        error_kind='oom' if isinstance(ex,torch.OutOfMemoryError) else 'inference_error')
                    if isinstance(ex,torch.OutOfMemoryError):torch.cuda.empty_cache()
                    consecutive+=1
                assert before==jev_hash(body)
                ev.update(elapsed_seconds=time.monotonic()-start,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
                f.write(json.dumps(ev,ensure_ascii=False,allow_nan=False)+'\n');done[key]=ev;progress('running',current=list(key))
                if ev['status']!='ok' and (a.pilot or consecutive>=3):raise RuntimeError('Pilot/repeated error; inspect logs')
        progress('complete' if all(e['status']=='ok' for e in done.values()) else 'complete_with_errors')
    except BaseException as ex:
        progress('failed',error=str(ex));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--response-model',required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--pilot',action='store_true');main(p.parse_args())
