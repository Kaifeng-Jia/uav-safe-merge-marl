#!/usr/bin/env bash
set -euo pipefail

seed="${1:-7001}"
if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
  printf 'seed must be an integer\n' >&2
  exit 2
fi

export PATH=/root/miniconda3/bin:$PATH
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
manifest="experiments/p0_p1/runs/provisional/MIX/$seed/manifest.json"
diagnostic_dir="experiments/p0_p1/runs/provisional/evaluations/single_seed_$seed"

while true; do
  state="$(python - "$manifest" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.exists():
    print('pending')
else:
    code = json.loads(path.read_text()).get('exit_code')
    print('pending' if code is None else 'success' if code == 0 else 'failure')
PY
)"
  case "$state" in
    success) break ;;
    failure) printf 'Training failed; evaluation was not started.\n' >&2; exit 1 ;;
    pending) sleep 60 ;;
  esac
done

actor="$(python - "$manifest" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))['actor_checkpoint'])
PY
)"
mkdir -p "$diagnostic_dir"
for setting in 'low 30' 'med 12' 'high 5'; do
  read -r load headway <<< "$setting"
  for scene_seed in 60021 60022 60023; do
    output="$diagnostic_dir/${load}_${scene_seed}.json"
    if [[ ! -f "$output" ]]; then
      python -m experiments.p0_p1.evaluate \
        --actor "$actor" --fleet MIX --train-seed "$seed" \
        --split smoke --load "$load" --scene-seed "$scene_seed" \
        --headway "$headway" --agents 20 --episode-length 1200 \
        --output "$output" > "$diagnostic_dir/${load}_${scene_seed}.log" 2>&1
    fi
  done
done

python - "$diagnostic_dir" "$manifest" <<'PY'
import json
from pathlib import Path
from statistics import mean
import sys

directory = Path(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text())
metrics = ('S_task_percent', 'I_percent', 'C_percent', 'N_col', 'Q', 'D_bar')
summary = {
    'stage': 'provisional',
    'map': 'MIT',
    'fleet': 'MIX',
    'training_seed': manifest['training_seed'],
    'training_wall_seconds': manifest['wall_seconds'],
    'scene_seeds': [60021, 60022, 60023],
    'diagnostic_headways_seconds': {'low': 30, 'med': 12, 'high': 5},
    'means_by_load': {},
    'note': 'Ad hoc single-seed diagnostics; loads are not P0 calibrated and the MIT map is provisional.',
}
for load in ('low', 'med', 'high'):
    rows = [json.loads((directory / f'{load}_{seed}.json').read_text())
            for seed in summary['scene_seeds']]
    if any(row['actor_sha256'] != rows[0]['actor_sha256'] for row in rows):
        raise ValueError('Mixed actor checkpoints in post-run diagnostics')
    summary['means_by_load'][load] = {
        metric: mean(row['metrics'][metric] for row in rows) for metric in metrics
    }
summary['actor_sha256'] = rows[0]['actor_sha256']
(directory / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print(json.dumps(summary, indent=2, sort_keys=True))
PY
