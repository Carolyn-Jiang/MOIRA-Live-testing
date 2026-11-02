"""Read-only native checkout verification; no clients or model requests."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from live_testing.native_source import verify_native_root

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--framework-dir',type=Path,default=os.environ.get('PAIR_TRADING_FIX_ROOT'))
args=parser.parse_args()
if args.framework_dir is None:parser.error('Pass --framework-dir or set PAIR_TRADING_FIX_ROOT')
print(json.dumps(verify_native_root(args.framework_dir),sort_keys=True))
