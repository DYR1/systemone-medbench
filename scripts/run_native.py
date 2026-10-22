#!/usr/bin/env python3
"""Launch one GPU replica, or a selected set of independent GPU replicas."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from common import ROOT, digest, save, sha, rows

def main(a):
    plan=json.loads((a.work/'inputs/plan.json').read_text())
    devices=a.devices.split(',')
    assert all(d.isdigit() for d in devices) and len(set(devices))==len(devices)
    shards=a.shards if a.shards is not None else list(range(len(plan['shards'])))
    if a.pilot:
        assert len(devices)==1; shards=[0]
    else:
        assert len(shards)==len(devices) and len(set(shards))==len(shards)
        assert all(0<=i<len(plan['shards']) for i in shards)
    workers={'d3':'d3.py','startlux-v1':'startlux_v1.py','startlux-v11':'startlux_v11.py'}
    worker=ROOT/'scripts/native'/workers.get(a.adapter,'worker.py')
    commands=[]
    for shard,device in zip(shards,devices):
        cmd=[sys.executable,str(worker),'--model',str(a.model.resolve()),
             '--response-model',a.response_model,'--shard',str(shard)]+(['--pilot'] if a.pilot else [])
        if worker.name=='worker.py':
            cmd += ['--adapter',a.adapter]
            if a.base_model: cmd += ['--base-model',str(a.base_model.resolve())]
        commands.append({'device':device,'shard':shard,'command':cmd})
    print(json.dumps({'execute':a.execute,'commands':commands},indent=2))
    if not a.execute: return
    from native.common import checkpoint_fingerprint
    expected_source={'startlux-v1':'startlux_v1_expected_source_hashes.json','qev':'qev_expected_source_hashes.json'}.get(a.adapter)
    if expected_source:
        if not a.source: raise ValueError('This adapter requires --source; see docs/MODELS.md')
        pinned=json.loads((ROOT/'configs'/expected_source).read_text())
        for filename,expected in pinned.items():
            if not (a.source/filename).is_file() or sha(a.source/filename)!=expected:
                raise ValueError('Upstream source differs from the evaluated snapshot: '+filename)
    # Refuse to overwrite a model/protocol selection, or start on occupied cards.
    config={'adapter':a.adapter,'model':str(a.model.resolve()),'response_model':a.response_model,
            'base_model':str(a.base_model.resolve()) if a.base_model else None,
            'source':str(a.source.resolve()) if a.source else None,
            'checkpoint':checkpoint_fingerprint(a.model),
            'base_checkpoint':checkpoint_fingerprint(a.base_model) if a.base_model else None,
            'checkpoint_source_hashes':{str(p.relative_to(a.model)):sha(p) for p in sorted((a.model/'source').rglob('*.py'))},
            'external_source_hashes':{str(p.relative_to(a.source)):sha(p) for p in sorted(a.source.rglob('*.py'))} if a.source else None,
            'plan_sha256':sha(a.work/'inputs/plan.json'),
            'runtime_hashes':{p.name:sha(p) for p in sorted(worker.parent.glob('*.py'))}}
    cp=a.work/'launcher_config.json'
    if cp.exists(): assert json.loads(cp.read_text())==config,'Use a new work directory for changed model/protocol'
    else: save(cp,config)
    if not a.pilot:
        progress=json.loads((a.work/'runs/pilot/progress.json').read_text())
        passed=progress['stage']=='complete' and progress['successful']==plan['pilot_requests']
        if not passed and a.allow_pilot_length_errors:
            latest={}
            for r in rows(a.work/'runs/pilot/events.jsonl'):latest[r['dataset'],r['variant'],r['id']]=r
            expected={(r['dataset'],r['variant'],r['id']) for r in rows(a.work/'inputs/pilot.jsonl')}
            passed=(progress['stage']=='complete_with_errors' and set(latest)==expected and
                    all(r['status']=='ok' or r.get('error_kind')=='context_length' for r in latest.values()) and
                    any(r['status']=='ok' for r in latest.values()))
        assert passed, 'Pass pilot first; only explicit length errors can be acknowledged with --allow-pilot-length-errors'
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    gpu={x[0]:[int(v) for v in x[1:]] for line in inventory.splitlines() if (x:=[v.strip() for v in line.split(',')])}
    for d in devices:
        assert d in gpu and gpu[d][0]<1024 and gpu[d][1]<=5, f'GPU {d} not idle; no unrelated process will be stopped'
    logs=a.work/'logs';logs.mkdir(exist_ok=True); jobs=[]
    try:
        for item in commands:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=item['device'],MEDBENCH_RUN_ROOT=str(a.work.resolve()),
                     PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4')
            if a.adapter=='d3': env['D3_FAST']='0'
            if a.source: env['MEDBENCH_UPSTREAM_SOURCE']=str(a.source.resolve())
            log=(logs/('pilot.log' if a.pilot else f"shard{item['shard']:02d}.log")).open('a')
            proc=subprocess.Popen(item['command'],env=env,stdout=log,stderr=subprocess.STDOUT)
            jobs.append((proc,log,item))
        save(logs/('pilot_receipt.json' if a.pilot else 'launch_receipt.json'),
             {'jobs':[dict(item,pid=proc.pid) for proc,log,item in jobs]})
        codes=[proc.wait() for proc,_,_ in jobs]
        if any(codes): raise SystemExit('At least one worker failed. Inspect logs; successful requests are preserved.')
    except KeyboardInterrupt:
        for proc,_,_ in jobs:
            if proc.poll() is None: proc.terminate()
        for proc,_,_ in jobs: proc.wait()
        raise
    finally:
        for _,log,_ in jobs: log.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter',choices=['d3','startlux-v1','startlux-v11','kev','anyjev','semif','gliner','openjev','pplx-v11','pplx-v12','qev'],required=True)
    p.add_argument('--base-model',type=Path); p.add_argument('--source',type=Path)
    p.add_argument('--model',type=Path,required=True); p.add_argument('--response-model',required=True)
    p.add_argument('--work',type=Path,required=True); p.add_argument('--devices',default='0')
    p.add_argument('--shards',type=int,nargs='+');p.add_argument('--pilot',action='store_true')
    p.add_argument('--allow-pilot-length-errors',action='store_true')
    p.add_argument('--execute',action='store_true');main(p.parse_args())
