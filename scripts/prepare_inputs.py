#!/usr/bin/env python3
"""Prepare deterministic, gold-free native input shards; no inference."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
from common import ROOT, rows, sha, jev_hash, save

def prepare(data, work, shards):
    if shards < 1: raise ValueError('shards must be positive')
    m=json.loads((data/'manifest.json').read_text()); assignment={}; offset=0
    for e in m['datasets']:
        ps={r['parent_id'] for r in rows(data/e['path']/'choice_letters/jev_requests.jsonl')}
        for j,pid in enumerate(sorted(ps,key=lambda x:hashlib.sha256((e['dataset']+'\0'+x).encode()).digest())):
            assignment[e['dataset'],pid]=(offset+j)%shards
        offset+=len(ps)
    assert len(assignment)==m['unique_case_count']
    dest=work/'inputs'; dest.mkdir(parents=True,exist_ok=False)
    handles=[(dest/f'shard{i:02d}.jsonl').open('w') for i in range(shards)]
    counts=collections.Counter(); ns=[0]*shards; probes={}; seen=set(); longest=None
    try:
        for e in m['datasets']:
            for s in e['variants']:
                arm=s['variant']; p=data/e['path']/arm/'jev_requests.jsonl'
                assert sha(p)==s['file_sha256'][p.name]
                for r in rows(p):
                    assert jev_hash(r['body'])==r['request_sha256']
                    key=e['dataset'],arm,r['id']; assert key not in seen; seen.add(key)
                    out={k:r[k] for k in ['id','parent_id','body','request_sha256']}
                    out.update(dataset=e['dataset'],variant=arm,category=e['category'])
                    j=assignment[e['dataset'],r['parent_id']]
                    handles[j].write(json.dumps(out,ensure_ascii=False)+'\n'); ns[j]+=1; counts[arm]+=1
                    probes.setdefault((e['dataset'],arm),out)
                    if longest is None or len(json.dumps(r['body']))>len(json.dumps(longest['body'])): longest=out
    finally:
        for f in handles: f.close()
    assert sum(ns)==m['request_count']
    probes['longest','characters']=longest
    picked={(r['dataset'],r['variant'],r['id']):r for r in probes.values()}
    with (dest/'pilot.jsonl').open('w') as f:
        for r in picked.values(): f.write(json.dumps(r,ensure_ascii=False)+'\n')
    plan={'manifest_sha256':sha(data/'manifest.json'),'mother_items':len(assignment),'requests':sum(ns),
          'scope':m['scope'],
          'variants':dict(counts),'shards':[{'shard':i,'requests':n,
           'parents':sum(v==i for v in assignment.values()),'sha256':sha(dest/f'shard{i:02d}.jsonl')} for i,n in enumerate(ns)],
          'pilot_requests':len(picked),'pilot_sha256':sha(dest/'pilot.jsonl'),
          'source':'data/jev only, without gold','partition':'SHA256 per dataset; round-robin by mother item'}
    save(dest/'plan.json',plan); print(json.dumps(plan,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--data',type=Path,default=ROOT/'data/jev')
    p.add_argument('--work',type=Path,required=True); p.add_argument('--shards',type=int,default=16)
    a=p.parse_args(); prepare(a.data,a.work,a.shards)
