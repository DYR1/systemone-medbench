#!/usr/bin/env python3
"""Conservative pre-publication scan. Reports locations, never matching secret values."""
import argparse
import hashlib
import json
from pathlib import Path
import re
PATTERNS={
 'api_key': re.compile(r'\bsk'+r'-(?:or-v1-)?[A-Za-z0-9_-]{24,}'),
 'hf_token':re.compile(r'\bhf'+r'_[A-Za-z0-9]{24,}'),
 'github_token':re.compile(r'\bgh'+r'[pousr]_[A-Za-z0-9]{30,}'),
 'private_key':re.compile('-----BEGIN '+r'(?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
 'personal_path':re.compile(r'/(?:Users|home)/'+r'[A-Za-z0-9._-]+(?:/|\b)'),
 'internal_host':re.compile(r'yuque'+r'\.antfin\.com|/yijian'+r'[0-9]+/'),
}
BAD_NAMES={'.env','.DS_Store','id_rsa','id_ed25519','automation.toml'}
def scan(root):
    issues=[];n=0;total=0
    for p in sorted(root.rglob('*')):
        if not p.is_file():continue
        if '.git' in p.relative_to(root).parts:continue
        rel=str(p.relative_to(root));n+=1;total+=p.stat().st_size
        if p.is_symlink():issues.append({'file':rel,'kind':'symlink'});continue
        if p.name in BAD_NAMES or p.suffix in ['.pyc','.pem','.safetensors']:
            issues.append({'file':rel,'kind':'forbidden_artifact'})
        try:
            with p.open(encoding='utf-8') as f:
                for i,line in enumerate(f,1):
                    for kind,pat in PATTERNS.items():
                        if pat.search(line):issues.append({'file':rel,'line':i,'kind':kind})
        except UnicodeDecodeError:issues.append({'file':rel,'kind':'binary_requires_review'})
    return {'clean':not issues,'files':n,'bytes':total,'issues':issues,
            'note':'Pattern scan plus allowlisted assembly; not a guarantee against all personal or confidential information.'}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path);a=p.parse_args();r=scan(a.root)
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2));raise SystemExit(0 if r['clean'] else 1)
