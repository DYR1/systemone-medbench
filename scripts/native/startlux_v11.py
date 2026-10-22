"""One card/one official native Instant replica, without modifying Jev inputs."""
import argparse
import asyncio
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
from common import ROOT, PACKAGE, checkpoint_fingerprint, digest, jev_hash, parse_jev, rows, save, sha

CONTROLS = {'think_mode':'never','none_check':False,'allow_none':False,'return_details':True}


async def main(args):
    inp=ROOT/'inputs'/('pilot.jsonl' if args.pilot else f'shard{args.shard:02d}.jsonl')
    out=ROOT/'runs'/('pilot' if args.pilot else f'shard{args.shard:02d}')
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((ROOT/'inputs/plan.json').read_text())
    expected=plan['pilot_sha256'] if args.pilot else plan['shards'][args.shard]['sha256']
    assert sha(inp)==expected
    records=list(rows(inp)); source=PACKAGE/'vendor/startlux_v11'
    MODEL_NAME=args.response_model
    versions={n:importlib.metadata.version(n) for n in ('torch','transformers','tokenizers')}
    assert versions['transformers'].startswith('5.8.'), versions
    model_manifest=checkpoint_fingerprint(args.model)
    decision_config=json.loads((Path(args.model)/'decision_config.json').read_text())
    config={'input_sha256':expected,'manifest_sha256':plan['manifest_sha256'],
        'runtime_hashes':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        'native_code_hashes':{str(p.relative_to(source)):sha(p) for p in sorted((source/'startlux_decide').rglob('*.py'))},
        'checkpoint_fingerprint':model_manifest,
        'decision_config_sha256':sha(Path(args.model)/'decision_config.json'),
        'model':str(args.model),'model_name':MODEL_NAME,'versions':versions,
        'protocol':'official Decider.decide + TransformersBackend; BF16; bidirectional; native temperatures',
        'controls':CONTROLS,'batch_size':1,'input_truncation':False,'device':'cuda:0'}
    config['run_hash']=digest(config)
    if (out/'config.json').exists():
        assert json.loads((out/'config.json').read_text())==config,'Changed run fingerprints'
    else: save(out/'config.json',config)
    done={}
    if (out/'events.jsonl').exists():
        for e in rows(out/'events.jsonl'):
            assert e['run_hash']==config['run_hash']
            done[e['dataset'],e['variant'],e['id']]=e
    started=time.monotonic()
    def progress(stage,**extra):
        save(out/'progress.json',{'stage':stage,'pid':os.getpid(),'planned':len(records),
            'successful':sum(e['status']=='ok' for e in done.values()),
            'errors':sum(e['status']!='ok' for e in done.values()),
            'elapsed_seconds':time.monotonic()-started,
            'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra})
    progress('loading')
    sys.path.insert(0,str(source))
    import torch
    from startlux_decide.backends.transformers import TransformersBackend
    from startlux_decide.modelcfg import load_model_folder, load_tokenizer
    from startlux_decide.pipeline import Decider
    from startlux_decide.prompt import from_systemone, option_lines

    class RecordedBackend(TransformersBackend):
        # Logging only; the parent performs the entire unmodified native forward.
        traces=None
        async def readout(self,prompt,letter_ids,media=None):
            ids=self.tokenizer.encode(prompt,add_special_tokens=False)
            trace={'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                   'token_ids_sha256':digest(ids),'input_tokens':len(ids),
                   'readout_token_ids':list(letter_ids)}
            if args.pilot: trace['rendered_prompt']=prompt
            self.traces.append(trace)
            if len(ids)>self.context_length:
                raise ValueError(f'Input length {len(ids)} exceeds {self.context_length}; not truncated')
            vals=await super().readout(prompt,letter_ids,media)
            trace['letter_logits']=vals
            return vals

    engine=None
    try:
        folder=load_model_folder(args.model)
        assert folder.kind=='instant' and folder.bidirectional
        tok=load_tokenizer(args.model)
        engine=RecordedBackend(args.model,tok,bidirectional=True,device='cuda:0',dtype='bfloat16')
        engine.context_length=folder.native_context or 65536
        await engine.start()
        assert engine.check and engine.check['late_token_reach']>0
        decider=Decider(folder,engine,context_length=engine.context_length,max_images=0)
        ctl=decider.controls(CONTROLS)
        save(out/'model_info.json',{'backend':engine.describe(),'versions':versions,
            'device_name':torch.cuda.get_device_name(0),
            'decision_config':decision_config,'controls':CONTROLS})
        progress('running')
        consecutive_errors=0
        with (out/'events.jsonl').open('a',buffering=1) as f:
            for r in records:
                key=r['dataset'],r['variant'],r['id']
                if key in done and done[key]['status']=='ok': continue
                assert jev_hash(r['body'])==r['request_sha256']
                body=r['body']; question=body['questions']['decision']; before=jev_hash(body)
                ev={k:r[k] for k in ('dataset','variant','id','parent_id','request_sha256')}
                ev.update(run_hash=config['run_hash'],request=body,full_input_sent=True,controls=CONTROLS)
                t=time.monotonic(); engine.traces=[]
                try:
                    row=from_systemone(body['state'],question)
                    ids=[o['id'] for o in row['options']]
                    if question['type']=='choice':
                        assert ids==list(question['criteria'])
                    else:
                        assert ids==['true','false']
                        for o in row['options']:
                            assert o['criterion']==question['criteria'][o['id']].strip()
                    ev['native_options']=row['options']
                    if args.pilot: ev['native_option_lines']=option_lines(row)
                    result=await decider.decide(body['state'],body['questions'],ctl)
                    torch.cuda.synchronize()
                    response={'answers':result.answers,'usage':result.usage,
                              'timing_ms':result.timing_ms,'model':MODEL_NAME}
                    ev['response']=response
                    assert response['usage']['output_tokens']==0
                    assert len(engine.traces)==1,'Expected one native readout per question'
                    p,pred=parse_jev(response,question,MODEL_NAME)
                    assert max(p.values())-p[pred]<1e-12
                    assert 'none_probability' not in response['answers']['decision']
                    ev.update(status='ok',probabilities=p,prediction=pred)
                    consecutive_errors=0
                except Exception as exc:
                    ev.update(status='error',error=str(exc),traceback=traceback.format_exc(),
                              error_kind='oom' if isinstance(exc,torch.OutOfMemoryError) else 'inference_error')
                    if isinstance(exc,torch.OutOfMemoryError): torch.cuda.empty_cache()
                    consecutive_errors+=1
                assert before==jev_hash(body),'Native implementation mutated original request'
                ev.update(native_readouts=engine.traces,elapsed_seconds=time.monotonic()-t,
                    finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
                if engine.traces:
                    ev['rendered_sha256']=engine.traces[0]['prompt_sha256']
                    ev['input_token_ids_sha256']=engine.traces[0]['token_ids_sha256']
                f.write(json.dumps(ev,ensure_ascii=False,allow_nan=False)+'\n');done[key]=ev
                progress('running',current=list(key))
                if ev['status']!='ok' and (args.pilot or consecutive_errors>=3):
                    raise RuntimeError('Pilot failure or repeated inference errors; inspect events')
        progress('complete' if len(done)==len(records) and all(e['status']=='ok' for e in done.values()) else 'complete_with_errors')
    except BaseException as exc:
        progress('failed',error=str(exc));raise
    finally:
        if engine is not None: await engine.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--response-model',required=True)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--pilot',action='store_true')
    asyncio.run(main(p.parse_args()))
