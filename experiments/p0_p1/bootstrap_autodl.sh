#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python - <<'PY'
import sys
import torch
print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if sys.version_info[:2] != (3, 10):
    raise SystemExit('Use a Python 3.10 AutoDL image for the tested dependency set')
if not torch.__version__.startswith('2.1.2'):
    raise SystemExit('Choose the tested AutoDL PyTorch 2.1.2 image')
if not torch.cuda.is_available():
    raise SystemExit('Choose an AutoDL PyTorch image with CUDA enabled')
PY

python -m pip install -r experiments/p0_p1/requirements-p0p1.txt
python -m unittest discover -s experiments/p0_p1 -p 'test_*.py' -v
python -m experiments.p0_p1.train --fleet MIX --seed 7001 \
  --stage provisional --steps 5000000 --rollout-threads 16 \
  --episode-length 700 --wandb --dry-run
