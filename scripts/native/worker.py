"""Gold-free, append-only worker for legacy and checkpoint-native adapters."""
import argparse
import datetime
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import time
from common import ROOT,PACKAGE,checkpoint_fingerprint,digest,jev_hash,normalized,parse_jev,rows,save,sha

def main(a):
    plan=json.loads((ROOT/'inputs/plan.json').read_text())
    name='pilot' if a.pilot else f'shard{a.shard:02d}'
    inp=ROOT/'inputs'/f'{name}.jsonl'; out=ROOT/'runs'/name
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    expected=plan['pilot_sha256'] if a.pilot else plan['shards'][a.shard]['sha256']
    assert sha(inp)==expected
    source_roots=[PACKAGE/'vendor'/a.adapter,a.model/'source/src']
    if os.environ.get('MEDBENCH_UPSTREAM_SOURCE'): source_roots.append(Path(os.environ['MEDBENCH_UPSTREAM_SOURCE']))
    native_hashes={str(i)+'/'+str(p.relative_to(root)):sha(p)
                   for i,root in enumerate(source_roots) if root.exists() for p in sorted(root.rglob('*.py'))}
    versions={}
    for module in ['torch','transformers','tokenizers','accelerate','pydantic','safetensors']:
        try: versions[module]=importlib.metadata.version(module)
        except importlib.metadata.PackageNotFoundError: versions[module]=None
    cfg={'input_sha256':expected,'manifest_sha256':plan['manifest_sha256'],
         'model_name':a.response_model,'model':str(a.model),'adapter':a.adapter,
         'checkpoint':checkpoint_fingerprint(a.model),'base_checkpoint':checkpoint_fingerprint(a.base_model) if a.base_model else None,
         'native_code_sha256':native_hashes,'versions':versions,'batch_size':1,'truncation':False,
         'runtime_hashes':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}}
    cfg['run_hash']=digest(cfg)
    if (out/'config.json').exists(): assert json.loads((out/'config.json').read_text())==cfg,'Changed protocol/checkpoint; use new work directory'
    else: save(out/'config.json',cfg)
    records=list(rows(inp)); done={}; tick=time.monotonic()
    for ev in rows(out/'events.jsonl') if (out/'events.jsonl').exists() else []:
        assert ev['run_hash']==cfg['run_hash']; done[ev['dataset'],ev['variant'],ev['id']]=ev
    def progress(stage,**extra):
        save(out/'progress.json',dict(stage=stage,pid=os.getpid(),planned=len(records),
             successful=sum(e['status']=='ok' for e in done.values()),errors=sum(e['status']!='ok' for e in done.values()),
             elapsed_seconds=time.monotonic()-tick,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra))
    progress('loading'); engine=None
    try:
        import torch
        legacy=a.adapter in ['kev','semif','anyjev','gliner','openjev']
        if legacy:
            from legacy_adapters import Adapter
            engine=Adapter(a.adapter,a.model,a.base_model)
        else:
            from extra_adapters import Adapter
            engine=Adapter(a.adapter,a.model)
        progress('running'); repeated=0
        with (out/'events.jsonl').open('a',buffering=1) as f:
            for r in records:
                key=r['dataset'],r['variant'],r['id']
                if done.get(key,{}).get('status')=='ok' or done.get(key,{}).get('error_kind')=='context_length': continue
                body=r['body']; assert jev_hash(body)==r['request_sha256']
                ev={k:r[k] for k in ['dataset','variant','id','parent_id','request_sha256']}
                ev.update(run_hash=cfg['run_hash'],request=body,full_input_sent=True); start=time.monotonic()
                engine.last_render=None
                try:
                    if legacy:
                        ps,meta=engine.predict(body); ps=normalized(ps,tolerance=1e-5)
                        q=body['questions']['decision']
                        answer=({'type':'noul','noul':ps['A']} if q['type']=='noul' else
                                {'type':'choice','choice':max(ps,key=ps.get),'probabilities':ps})
                        raw={'answers':{'decision':answer}}; ev['adapter_metadata']=meta
                    else: raw=engine.predict(body)
                    torch.cuda.synchronize()
                    response=dict(raw,model=a.response_model)
                    p,pred=parse_jev(response,body['questions']['decision'],a.response_model)
                    ev.update(status='ok',response=response,native_response=raw,probabilities=p,prediction=pred)
                    repeated=0
                except Exception as ex:
                    length=ex.__class__.__name__=='LengthError' or (isinstance(ex,ValueError) and any(s in str(ex).lower() for s in ['context_length','tokens, limit','no truncation','token limit','too long','exceeds']))
                    kind='context_length' if length else ('oom' if isinstance(ex,torch.OutOfMemoryError) else 'inference_error')
                    ev.update(status='error',error_kind=kind,error=str(ex)[:2000])
                    if kind=='oom': torch.cuda.empty_cache()
                    repeated=0 if length else repeated+1
                assert jev_hash(body)==r['request_sha256'],'Adapter mutated original request'
                trace=engine.last_render
                ev.update(native_readout=trace,rendered_sha256=digest(trace) if trace is not None else None,
                          elapsed_seconds=time.monotonic()-start,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
                f.write(json.dumps(ev,ensure_ascii=False,allow_nan=False)+'\n'); done[key]=ev
                progress('running',current=list(key))
                if ev['status']!='ok' and ((a.pilot and ev['error_kind']!='context_length') or repeated>=3):
                    raise RuntimeError('Pilot/repeated inference error; inspect saved events')
        progress('complete' if len(done)==len(records) and all(e['status']=='ok' for e in done.values()) else 'complete_with_errors')
    except BaseException as ex:
        progress('failed',error=str(ex)); raise
    finally:
        if engine is not None: engine.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--adapter',required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--base-model',type=Path)
    p.add_argument('--response-model',required=True);p.add_argument('--shard',type=int,default=0)
    p.add_argument('--pilot',action='store_true');main(p.parse_args())
