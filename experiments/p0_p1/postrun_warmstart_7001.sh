#!/usr/bin/env bash
# Evaluate the warm-started actor on exactly the same nine diagnostic scenes.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
python_bin="${P0P1_PYTHON:-python}"
"$python_bin" - <<'PY'
import contextlib
from hashlib import sha256
import io
import json
from pathlib import Path
from statistics import mean
import sys
import time

from experiments.p0_p1 import evaluate
from experiments.p0_p1.metrics import EpisodeMetrics

root = Path.cwd()
manifest_path = root / 'experiments/p0_p1/runs/provisional/MIX/7001_warmstart_5m/manifest.json'
output_dir = root / 'experiments/p0_p1/runs/provisional/evaluations/warmstart_7001_5m'
baseline_dir = root / 'experiments/p0_p1/runs/provisional/evaluations/single_seed_7001'
output_dir.mkdir(parents=True, exist_ok=True)
while True:
    state = json.loads(manifest_path.read_text())
    if state.get('exit_code') is not None:
        if state['exit_code'] != 0:
            raise RuntimeError('Warm-start training failed; no evaluation was run')
        break
    time.sleep(60)

actor = root / state['actor_checkpoint']
if not actor.is_file():
    raise FileNotFoundError(actor)
actor_sha = sha256(actor.read_bytes()).hexdigest()
original_summarize = EpisodeMetrics.summarize

def instrument(self, *args, **kwargs):
    result = original_summarize(self, *args, **kwargs)
    travel_ratio = {
        task.id: (self.arrival[task.id] - task.release_time) /
                 self.free_flow_seconds[task.id]
        for task in self.tasks if task.id in self.arrival
    }
    arrived = len(travel_ratio)
    on_time = sum(value <= 1.5 for value in travel_ratio.values())
    result['arrival_counts'] = {
        'tasks': len(self.tasks), 'arrived': arrived, 'on_time': on_time,
        'late': arrived - on_time, 'not_arrived': len(self.tasks) - arrived,
    }
    result['deadline_sensitivity'] = {
        str(multiplier): sum(value <= multiplier for value in travel_ratio.values())
        for multiplier in (1.5, 2.0, 2.5, 3.0)
    }
    return result

EpisodeMetrics.summarize = instrument
loads = [('low', 30), ('med', 12), ('high', 5)]
seeds = (60021, 60022, 60023)
summary = {
    'stage': 'provisional', 'map': 'MIT three_phase_graph_merge',
    'run_kind': 'weights_only_warm_start',
    'training_seed': 7001,
    'parent_wandb_run_id': state['parent_wandb_run_id'],
    'parent_actor_sha256': state['parent_actor_sha256'],
    'actor_sha256': actor_sha,
    'parent_environment_steps': state['parent_environment_steps'],
    'additional_environment_steps': state['actual_additional_environment_steps'],
    'scene_seeds': list(seeds),
    'diagnostic_headways_seconds': dict(loads),
    'by_load': {},
    'note': 'One seed, ad hoc loads, provisional MIT map and uncalibrated free-flow deadlines; not formal P0/P1.',
}

for load, headway in loads:
    rows = []
    baseline_rows = []
    for seed in seeds:
        output = output_dir / f'{load}_{seed}.json'
        if not output.exists():
            sys.argv = [
                'evaluate', '--actor', str(actor), '--fleet', 'MIX',
                '--train-seed', '7001', '--split', 'smoke', '--load', load,
                '--scene-seed', str(seed), '--headway', str(headway),
                '--agents', '20', '--episode-length', '1200',
                '--output', str(output),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate.main()
        row = json.loads(output.read_text())
        if row['actor_sha256'] != actor_sha:
            raise ValueError(f'Wrong actor in {output}')
        baseline = json.loads((baseline_dir / f'arrivals_{load}_{seed}.json').read_text())
        if baseline['actor_sha256'] != state['parent_actor_sha256']:
            raise ValueError(f'Wrong baseline actor for {load} {seed}')
        rows.append(row)
        baseline_rows.append(baseline)
    def total(key, source):
        return sum(row['metrics']['arrival_counts'][key] for row in source)
    metric_names = ('S_task_percent', 'I_percent', 'C_percent', 'N_col', 'D_bar')
    summary['by_load'][load] = {
        'headway_seconds': headway,
        'counts': {key: total(key, rows) for key in
                   ('tasks', 'arrived', 'on_time', 'late', 'not_arrived')},
        'baseline_counts': {key: total(key, baseline_rows) for key in
                            ('tasks', 'arrived', 'on_time', 'late', 'not_arrived')},
        'means': {key: mean(row['metrics'][key] for row in rows) for key in metric_names},
        'baseline_means': {key: mean(row['metrics'][key] for row in baseline_rows)
                           for key in metric_names},
        'deadline_sensitivity': {
            key: sum(row['metrics']['deadline_sensitivity'][key] for row in rows)
            for key in ('1.5', '2.0', '2.5', '3.0')
        },
    }
    print(load, json.dumps(summary['by_load'][load], sort_keys=True), flush=True)

(output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print(f'Wrote {output_dir / "summary.json"}', flush=True)
PY
