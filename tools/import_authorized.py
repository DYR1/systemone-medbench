#!/usr/bin/env python3
"""Import an already-authorized frozen paper artifact; does not obtain restricted source records."""
import argparse
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import sha,save
from validate_data import validate

def main(a):
    if not a.confirm_authorized: raise ValueError('Explicit --confirm-authorized is required')
    scope=json.loads((Path(__file__).resolve().parents[1]/'configs/paper_scope.json').read_text())
    if sha(a.source/'manifest.json')!=scope['frozen_manifest_sha256']:
        raise ValueError('Not the frozen paper Jev artifact; do not substitute an unverified reconstruction')
    validate(a.source)
    if a.output.exists(): raise ValueError('Use a new, private output directory outside this repository')
    repo=Path(__file__).resolve().parents[1]
    if a.output.resolve().is_relative_to(repo): raise ValueError('Do not import restricted data into the public repository')
    a.output.mkdir(parents=True)
    m=json.loads((a.source/'manifest.json').read_text())
    for e in m['datasets']:
        dst=a.output/e['path'];dst.mkdir(parents=True)
        shutil.copy2(a.source/e['path']/'answers.jsonl',dst/'answers.jsonl')
        for s in e['variants']:
            (dst/s['variant']).mkdir()
            for fn in s['file_sha256']:shutil.copy2(a.source/e['path']/s['variant']/fn,dst/s['variant']/fn)
    m['contains_restricted']=True; m['scope']='authorized_local_full_paper_15_datasets_8742_entries'
    m['frozen_manifest_sha256']=scope['frozen_manifest_sha256']
    save(a.output/'manifest.json',m);print(json.dumps(validate(a.output),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--confirm-authorized',action='store_true');main(p.parse_args())
