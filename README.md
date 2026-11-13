# UAV Safe Merge MARL

Research experiment code for multi-agent UAV coordination at corridor merges,
prepared for the Cornell Gao’s Lab project. This is a project working repository,
not an official MIT or Cornell software release.

## Status and scope

The current implementation supports **provisional P0/P1 experiments on MIT’s
single Y-shaped merge map**. It retains MIT’s 25 discrete actions (turn rate ×
longitudinal acceleration), with aircraft-specific capability limits. A policy
action lasts 1 simulation second and physics advances in 0.1-second substeps.

- **E1 / H-M:** every aircraft uses the M-type capability envelope.
- **E3 / MIX:** each task independently samples S, M, or L with equal probability.
- **P0:** baseline training, deterministic evaluation, traffic-load calibration,
  and reproducibility tools.
- **P1:** paired E1/E3 analysis of fleet heterogeneity, including low-load
  subtraction and a secondary density-matched comparison. P1 reuses E1/E3 policies.

The training/checkpoint/evaluation path has run successfully. Formal P0/P1 results
are **not complete**: the team-map integration, scripted free-flow travel-time
calibration, validation-selected load levels, and multiple training seeds remain
necessary. The map adapter currently supports the MIT two-entry, one-merge,
one-exit structure; it is not a general city-network adapter. The discontinued
city-map control abstraction is not included.

## Source and license

Based on [Jaroan/Decentralized-Corridor-MARL](https://github.com/Jaroan/Decentralized-Corridor-MARL)
at commit `45464195bd53e65bd461d769881279ff8a09c2eb`.
Upstream Git history and license notices are retained. See
[README.upstream.md](README.upstream.md) for the original project description and
citations. The repository is distributed under [GPL-3.0](LICENSE), with the
original component notices in [multiagent/LICENSE.txt](multiagent/LICENSE.txt)
and [onpolicy/LICENSE](onpolicy/LICENSE).

## Installation

The experiment environment was tested with **Python 3.10, PyTorch 2.1.2, and
PyTorch Geometric 2.5.3**. Use a full Git clone so the pinned upstream history is
available for provenance checks.

```bash
git clone https://github.com/Kaifeng-Jia/uav-safe-merge-marl.git
cd uav-safe-merge-marl
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.1.2
python -m pip install -r experiments/p0_p1/requirements-p0p1.txt
python -m unittest discover -s experiments/p0_p1 -p 'test_*.py' -v
```

For the previously used AutoDL setup, select a Python 3.10 / PyTorch 2.1.2 CUDA
image, then run `bash experiments/p0_p1/bootstrap_autodl.sh`. That script checks
the environment, installs experiment dependencies, runs tests, and prints a
training manifest; it does not start training.

## Small baseline run

Inspect the configuration first:

```bash
python -m experiments.p0_p1.train --fleet MIX --seed 5002 \
  --stage smoke --steps 11200 --rollout-threads 4 --episode-length 700 --dry-run
```

Remove `--dry-run` to train. Use a fresh seed for each run; existing run directories
are protected from overwrite. A smoke run checks the pipeline and is not evidence
of policy convergence. To evaluate the saved actor:

```bash
python -m experiments.p0_p1.evaluate \
  --actor onpolicy/results/GraphMPE/p0p1_merge_graph/rmappo/p0p1_smoke_MIX_5002/run1/models/actor.pt \
  --fleet MIX --train-seed 5002 --split smoke --load smoke \
  --scene-seed 60011 --headway 12 --agents 20 --episode-length 1200 \
  --output experiments/p0_p1/smoke_quick_baseline_5002.json
```

W&B logging is optional. Run `python -m wandb login` in your own terminal, then
add `--wandb --wandb-project YOUR_PROJECT --wandb-entity YOUR_ACCOUNT` to training.
No W&B credentials are included in this repository.

## Code guide

| Location | Purpose |
| --- | --- |
| `multiagent/custom_scenarios/p0p1_merge_graph.py` | Paired traffic, aircraft types, releases, lifecycle, and map interface |
| `multiagent/environment.py` | MIT control actions with type-specific dynamics and physical substeps |
| `onpolicy/algorithms/graph_actor_critic.py` | Graph actor/critic and neighbor-capability visibility mask |
| `experiments/p0_p1/protocol.py`, `metrics.py` | Constants, independent random streams, provenance, safety and task metrics |
| `experiments/p0_p1/train.py`, `evaluate.py`, `matrix.py` | Training, fixed-actor evaluation, and experiment scheduling |
| `experiments/p0_p1/calibration.py`, `report.py` | Load calibration and paired P1 analysis |
| `experiments/p0_p1/analysis/` | Historical follow-up/evaluation scripts for the provisional runs |

See [the experiment guide](experiments/p0_p1/README.md) for detailed methodology,
map-interface requirements, and longer-run commands. Historical scripts tied to
seed 7001 require local manifests/checkpoints that are not distributed here.

## Existing experiment records

The MIT-map runs used one training seed (7001). MIX completed two approximately
5-million-step segments. H-M completed its first segment; its second segment was
stopped by request. The second segments were **weights-only warm starts**, with
optimizer and random states reset, not exact resumptions.

The two MIX summaries are retained as provisional historical records:

- [First segment](experiments/p0_p1/RESULTS_provisional_MIX_7001.md)
- [Warm-start segment](experiments/p0_p1/RESULTS_warmstart_MIX_7001.md)

The 30/12/5-second evaluation release intervals were diagnostic choices, not
formally calibrated traffic loads. Deadlines used estimated free-flow times, so
these results do not certify the formal P0/P1 protocol or operational safety.

Experiment checkpoints, raw runs, credentials, local environments, and internal
planning documents are excluded. `model_weights/` contains the original upstream
example weights, not this project's P0/P1 policies.

## Publication changes

The publication preparation changes documentation, ignore rules, and Git-provenance
validation only. The latter now accepts experiment commits descended from the
pinned MIT base instead of requiring `HEAD` to equal that base. The source-content
fingerprint still identifies experiment changes; this publication therefore has
a new fingerprint and must not be presented as the unchanged historical run
source. Aircraft controls, rewards, map geometry, and experiment parameters were
not changed during publication preparation.
