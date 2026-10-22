"""Shared helpers; run output is selected explicitly by the portable launcher."""
import importlib.util
import os
from pathlib import Path
PACKAGE = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('package_common', PACKAGE/'scripts/common.py')
_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(_module)
for _name in ['rows','sha','digest','jev_hash','save','parse_jev','normalized']:
    globals()[_name] = getattr(_module, _name)
ROOT = Path(os.environ.get('MEDBENCH_RUN_ROOT', PACKAGE/'runs/native')).resolve()

def checkpoint_fingerprint(model):
    """Metadata content hashes + weight size/mtime. Not a full weight-content hash."""
    model=Path(model)
    weights=sorted(model.glob('*.safetensors'))
    if not weights: raise ValueError('No .safetensors weights in model directory')
    import json
    for index in model.glob('*.safetensors.index.json'):
        for filename in set(json.loads(index.read_text())['weight_map'].values()):
            if not (model/filename).is_file(): raise ValueError('Missing weight shard: '+filename)
    return {'weights_stat':{p.name:{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in weights},
            'metadata_sha256':{p.name:sha(p) for p in sorted(model.iterdir())
                               if p.is_file() and p.suffix in ['.json','.py','.jinja']}}
