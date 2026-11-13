#!/usr/bin/env bash
# Add ~5M environment steps from the completed seed-7001 actor/critic weights.
# This is a weights-only warm start: MIT did not save optimizer or RNG state.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
python_bin="${P0P1_PYTHON:-python}"
"$python_bin" - "$@" <<'PY'
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.p0_p1.protocol import backbone_commit, source_fingerprint

parser = argparse.ArgumentParser()
parser.add_argument('--dry-run', action='store_true')
args = parser.parse_args(sys.argv[1:])
root = Path.cwd()
parent_path = root / 'experiments/p0_p1/runs/provisional/MIX/7001/manifest.json'
parent = json.loads(parent_path.read_text())
if parent.get('exit_code') != 0 or parent.get('training_seed') != 7001 or parent.get('fleet') != 'MIX':
    raise ValueError('Parent must be the completed provisional E3/MIX seed-7001 run')
if parent['experiment_source_fingerprint'] != source_fingerprint(root):
    raise ValueError('Experiment source changed since the parent run; audit changes before warm starting')
if parent['backbone_commit'] != backbone_commit(root):
    raise ValueError('MIT backbone commit changed')
if not parent['wandb']['enabled']:
    raise ValueError('Parent run must have W&B enabled')

parent_actor = root / parent['actor_checkpoint']
model_dir = Path(os.environ.get('P0P1_PARENT_MODEL_DIR', str(parent_actor.parent))).resolve()
actor = model_dir / 'actor.pt'
critic = model_dir / 'critic.pt'
for checkpoint in (actor, critic):
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
actor_sha = sha256(actor.read_bytes()).hexdigest()
critic_sha = sha256(critic.read_bytes()).hexdigest()
expected_actor_sha = '25b3b6b8339df8b1b4e88fc86a7d499a1453b0c83eac70b6e5a81dde35fd8a41'
if actor_sha != expected_actor_sha:
    raise ValueError('Parent actor does not match the evaluated final actor')

segment_steps = 5_000_000
rollout_threads = 16
episode_length = 700
actual_steps = segment_steps // (rollout_threads * episode_length) * rollout_threads * episode_length
run_id = '7001_warmstart_5m'
experiment_name = 'p0p1_provisional_MIX_7001_warmstart_5m'
run_dir = root / 'experiments/p0_p1/runs/provisional/MIX' / run_id
if run_dir.exists() and not args.dry_run:
    raise FileExistsError(f'Warm-start run directory already exists: {run_dir}')

command = list(parent['command'])
command[0] = sys.executable
def replace(flag, value):
    index = command.index(flag)
    command[index + 1] = str(value)
replace('--num_env_steps', segment_steps)
replace('--experiment_name', experiment_name)
replace('--seed', 7001)
command.extend(['--model_dir', str(model_dir)])

manifest = {
    'stage': 'provisional', 'fleet': 'MIX', 'training_seed': 7001,
    'run_kind': 'weights_only_warm_start',
    'parent_manifest': str(parent_path.relative_to(root)),
    'parent_wandb_run_id': 'wucd7qra',
    'parent_actor_sha256': actor_sha,
    'parent_critic_sha256': critic_sha,
    'optimizer_state_restored': False,
    'rng_state_restored': False,
    'parent_environment_steps': parent['actual_environment_steps'],
    'requested_additional_environment_steps': segment_steps,
    'actual_additional_environment_steps': actual_steps,
    'cumulative_nominal_environment_steps': parent['actual_environment_steps'] + actual_steps,
    'backbone_commit': parent['backbone_commit'],
    'experiment_source_fingerprint': source_fingerprint(root),
    'map_module': parent['map_module'],
    'map_module_sha256': parent['map_module_sha256'],
    'deadline_method': 'unchanged route-distance-over-vmax provisional estimate',
    'wandb': parent['wandb'],
    'command': command,
}
if args.dry_run:
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    raise SystemExit(0)

run_dir.mkdir(parents=True)
manifest['started_utc'] = datetime.now(timezone.utc).isoformat()
manifest_path = run_dir / 'manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
start = time.monotonic()
with (run_dir / 'train.log').open('w') as log:
    result = subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
manifest['exit_code'] = result.returncode
manifest['wall_seconds'] = round(time.monotonic() - start, 3)
manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
if result.returncode == 0:
    checkpoint_root = (root / 'onpolicy/results/GraphMPE/p0p1_merge_graph/rmappo'
                       / experiment_name)
    checkpoints = list(checkpoint_root.rglob('actor.pt'))
    if len(checkpoints) != 1 or not checkpoints[0].with_name('critic.pt').is_file():
        manifest['exit_code'] = 1
        manifest['checkpoint_error'] = f'Expected one actor/critic pair; found {len(checkpoints)} actors'
    else:
        manifest['actor_checkpoint'] = str(checkpoints[0].relative_to(root))
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({key: manifest.get(key) for key in
                  ('run_kind', 'actual_additional_environment_steps', 'exit_code',
                   'wall_seconds', 'actor_checkpoint')}, indent=2))
raise SystemExit(manifest['exit_code'])
PY
