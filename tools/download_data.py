#!/usr/bin/env python3
"""Download the public HF artifact at an explicit immutable revision, then validate."""
import argparse
import re
from pathlib import Path
from validate_data import validate

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True);p.add_argument('--revision',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not re.fullmatch('[0-9a-f]{40}',a.revision):raise ValueError('Use the published 40-character commit revision, not main')
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=a.repo,repo_type='dataset',revision=a.revision,local_dir=str(a.output),
                      allow_patterns=['data/jev/**','README.md','dataset_sources.json','paper_scope.json','LICENSES/**'])
    print(validate(a.output/'data/jev'))
