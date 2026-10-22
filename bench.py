#!/usr/bin/env python3
"""One entry point. `python bench.py COMMAND --help` lists that command's arguments."""
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
COMMANDS={'validate':'tools/validate_data.py','prepare':'scripts/prepare_inputs.py',
          'api':'scripts/evaluate_jev.py','native':'scripts/run_native.py',
          'report':'scripts/report.py','download':'tools/download_data.py',
          'import-authorized':'tools/import_authorized.py','scan':'tools/scan_release.py'}
if len(sys.argv)<2 or sys.argv[1] not in COMMANDS:
    print('Usage: python bench.py {'+', '.join(COMMANDS)+'} [arguments]\nAll inference commands default to dry-run; --execute is explicit.')
    raise SystemExit(0 if len(sys.argv)<2 or sys.argv[1] in ['-h','--help'] else 2)
raise SystemExit(subprocess.call([sys.executable,str(ROOT/COMMANDS[sys.argv[1]]),*sys.argv[2:]]))
