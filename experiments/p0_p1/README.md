# P0/P1 execution plan and implementation status

Protocol reference: the project’s internal experiment plan (`exp 2.pdf`), dated
2026-09-28; that planning document is not distributed in this source repository. The MIT backbone is
`Jaroan/Decentralized-Corridor-MARL` at commit
`45464195bd53e65bd461d769881279ff8a09c2eb` (GPL-3.0).

## 中文说明：本地代码改动与小规模 baseline

MIT 代码已克隆到本机 `UAV/mit-corridor-backbone/`，所有改动都在这个副本中；
上游 GitHub 仓库和同学的地图未被改动。当前公开版本见根目录 README；
原工作副本与历史结果保留在本机。
P0 的参考条件 E3 是 MIX 机队、仅公共邻机信息、ASAP 释放；E1 是同一配置下
全部使用 M 型机的对照。P1 复用 P0 的 E1/E3 策略，不另外训练。
MIX 的机型集合是 S/M/L，各任务独立等概率抽取；单个十任务场景不强制
恰好包含三种机型。E1/H-M 和 E3/MIX 都已进行单种子预实验。

4090 上种子 7001 的 4,995,200 步预实验已完成，固定场景三档按时完成率
分别为 20.0%/30.0%/26.7%，不要求按时但须在仿真结束前到达的比例分别为
96.7%/85.0%/68.3%；详见
[`RESULTS_provisional_MIX_7001.md`](RESULTS_provisional_MIX_7001.md)。

| 文件 | 对照实验方案所做的事 |
| --- | --- |
| `multiagent/custom_scenarios/p0p1_merge_graph.py` | 继承 MIT 合流场景；生成三种机型和配对交通任务，执行计量放行、生成净空、休眠/活跃/ghost 状态；在本地观测附加自身包络，在图节点预留邻机能力及可用性掩码；补上固定目标节点和团队地图接口。MIT 地图的自由流时间仍是估算值。 |
| `multiagent/environment.py`, `multiagent/MPE_env.py` | 让一个策略动作保持 1 秒，其间执行十次 0.1 秒物理更新；按机型包络把同一个 25 档动作序号映射为转率和加速度；仅在新场景启用该环境。 |
| `onpolicy/algorithms/graph_actor_critic.py` | actor 对邻机能力应用公共信息掩码；critic 保留完整能力输入，其他网络维度不变。 |
| `onpolicy/runner/shared/graph_mpe_runner.py`, `onpolicy/envs/env_wrappers.py` | 训练时屏蔽休眠/ghost 的损失与循环状态；修复 MIT 单环境图封装返回值与 runner 不一致的问题。 |
| `experiments/p0_p1/protocol.py`, `metrics.py` | 固定尺度、三类包络、独立随机数流、训练负荷范围；按活跃物理子步计 I%，按连续子步轨迹计 5 m 碰撞，并计算任务完成、延误、吞吐和等待指标。 |
| `experiments/p0_p1/train.py`, `evaluate.py`, `matrix.py` | 记录代码/地图/配置散列；运行单种子训练、确定性单场景评估和完整种子/场景矩阵；防止覆盖已有结果。 |
| `experiments/p0_p1/calibration.py`, `report.py` | P0 在 E3 验证集上选低/中/高负荷并冻结最小效应；P1 按训练种子聚合配对场景，计算中/高负荷相对低负荷的交互，做 Holm 与 TOST；密度匹配分析只调整 E1 评价放行率。 |

已在本机完成一次受控的 E3 快速 baseline：一个训练种子、11,200 环境步、
四个 rollout workers、用时 208 秒；随后在一个固定的 20 任务场景上做确定性
评估，用时约 13 秒。训练、检查点保存和评估全链路通过；该策略只有四次
PPO 更新，场景内按时完成率为 0，因此只能证明实现可运行，不能证明控制器
已学会完成任务，也不能用于 P0/P1 科学比较。

随后将单个 E3 种子（5003）增加到 56,000 环境步、20 次 PPO 更新，
本机用时 1,113 秒。固定场景 60021 下，临时选用 30/12/5 秒放行间隔进行
三档诊断，按时完成率均为 0%，所以不能从该策略标定 P0 的负荷拐点。
对应 I% 为 20.96/30.84/59.98；所有三档都有 T_max 未完成任务。
另用双机、无交通的简易航路跟踪器做可达性检查：两机在 112/118 秒抵达
目标，均早于按当前估计 T_ff 得到的时限。这只排除了“路线根本不可达”
这一种实现故障；不等于正式的自由流时间标定，也不能证明学习设置已收敛。
这次短训结果提示需增加训练预算并检查学习曲线，暂不继续正式 P0 负荷扫描。
W&B 开关已接入单种子训练脚本；云端在线运行和模型备份已经完成。
原 actor/critic 权重的追加约 500 万步热启动训练由
`warmstart_provisional_7001.sh` 启动；优化器与随机状态无法精确恢复，
所以这是单独的 W&B run，而非原 run 的无缝延长。原模型和评估不会被覆盖。
本次热启动于 2026-10-05 18:36 UTC 在 AutoDL 4090 启动，
[W&B run](https://wandb.ai/kj366-cornell-university/p0p1_corridor/runs/o4xfq6rs)。
`watch_warmstart_checkpoints_7001.sh` 备份中途权重，
`postrun_warmstart_7001.sh` 在结束后自动重评相同的 9 个场景。
续训已正常结束，低负荷按时率从 20.0% 提高到 86.7%，但安全指标和
中/高负荷表现不一致；详见
[`RESULTS_warmstart_MIX_7001.md`](RESULTS_warmstart_MIX_7001.md)。

2026-10-06 的 E1/H-M 种子 7001 首段预实验已完成 4,995,200 环境步，
首段 [W&B run](https://wandb.ai/kj366-cornell-university/p0p1_corridor/runs/kao47v4t)。
第二段从 actor/critic 权重热启动，后按用户要求停止；它未完成原定预算。
`analysis/e1_followup_7001.py` 保留当时的编排代码，需要本地历史 manifest
和检查点才能使用。E1/E3 仍是单种子、临时 MIT 地图预实验，不构成正式
P1 Gate 1 推断。未完成的团队地图适配不属于这份有效实现。

如需再跑一个相同规模但独立的快速检查，使用新种子即可：

```bash
.venv/bin/python -m experiments.p0_p1.train --fleet MIX --seed 5002 \
  --stage smoke --steps 11200 --rollout-threads 4 --episode-length 700
.venv/bin/python -m experiments.p0_p1.evaluate \
  --actor onpolicy/results/GraphMPE/p0p1_merge_graph/rmappo/p0p1_smoke_MIX_5002/run1/models/actor.pt \
  --fleet MIX --train-seed 5002 --split smoke --load smoke \
  --scene-seed 60011 --headway 12 --agents 20 --episode-length 1200 \
  --output experiments/p0_p1/smoke_quick_baseline_5002.json
```

正式 P0/P1 仍有两项前置工作：接入满足几何约束的团队地图，并用无交通脚本
控制器计算各路线/机型的自由流时间；随后按下面的试跑结果确定正式预算。
当前地图接口限于双入口、单合流、单出口的 MIT 相位结构。P0 的收敛斜率、
置信区间精度和低负荷安全门槛仍需在正式结果上逐项审核。

## Scientific protocol

| Phase | Runs | Validation and reporting |
| --- | --- | --- |
| Engineering smoke | one matched H-M/MIX pair, tiny budget | Check shapes, releases, dynamics, graph privacy, and full train/eval path. Never include in a scientific comparison. |
| Pilot | E1=H-M/public and E3=MIX/public, three shared training seeds each, half of the proposed 5 million steps | Measure time, learning curves, seed-level SD of paired differences, plateau at 60–70% of budget; increase formal budget or seed count **before** final runs if needed. Pilot seeds excluded from all comparisons. |
| P0 | Retrain E1/E3 with at least ten new matched seeds each, common 5 million step budget unless pilot increases it | Sweep at least ten log-spaced loads on ten validation scenarios per load. Use E3's throughput/completion curve to freeze D_low/D_med/D_high. Evaluate fixed final checkpoints once on 30 untouched test scenes per load. Check convergence, precision, low-load collision-free rate and corridor compliance as in Table 10. |
| P1 | No new training | Compare E3 versus E1 on matched training and scenario seeds. Gate 1 tests four interactions: {I%, D-bar} × {medium, high}, each subtracting the low-load contrast. Report low-load learning difficulty separately, and repeat a secondary H-M analysis with density-matched release rate. Holm correction and TOST distinguish pass/no effect/uncertain. |

`train.py` launches one seed through MIT's rmappo trainer. `matrix.py`
enumerates the registered seeds and scenes, resumes completed jobs and checks
the frozen code/map fingerprint. `evaluate.py` loads a fixed actor and writes
one scenario JSON. `report.py` selects P0 loads from validation results,
freezes Gate 1's minimum effects, or computes the seed-level P1 Gate 1. The `protocol.py`
module holds frozen physics/traffic constants and independent random streams;
`metrics.py` counts physical substep exposure and continuous collision crossings.

## Current execution state

- MIT's two-entry merge map is used **only for provisional work**. It is a
  three-tube Y geometry with 1.5 u approach arms and a 2.0 u outgoing stem.
  Its corridor width is 0.3 u = 98.4 m, versus the protocol's 66 m, and its
  sharp merge lacks the required >=280 m bend-radius check. Its results cannot
  replace final P0/P1 on the team's map.
- A matched 400-environment-step H-M/MIX smoke pair completed on this Mac with
  the 500 m graph radius and three fixed corridor goal nodes, followed by
  deterministic evaluations on the same scene seed. Both policies had zero
  on-time completions; these are engineering diagnostics only. Nine
  protocol/integration tests pass.
- A longer single-seed E3 preview completed 56,000 environment steps in
  1,113 seconds. At three diagnostic headways, no task was on time, so P0 load
  calibration has not begun. A separate two-agent route-following diagnostic
  reached both goals before their provisional deadlines. The short-trained
  learned policy remains inadequate; the route is reachable.
- The one-seed 4,995,200-step RTX 4090 cloud run completed in 9 h 52 min.
  The fixed-scene results are summarized in `RESULTS_provisional_MIX_7001.md`.
- Route-specific scripted free-flow times still need to be computed on the
  frozen team map. The provisional evaluator currently uses route length over
  maximum speed and labels this in every output; D-bar, task deadlines and
  T_max are consequently provisional. The team-map adapter must supply the
  scripted travel-time table before a formal test set is opened.

## Local usage

Tested with Python 3.10, PyTorch 2.1.2, PyG 2.5.3. The workspace-local
`.venv` inherits PyTorch from a conda environment. The smaller P0/P1 dependency
set is in `requirements-p0p1.txt`; MIT's full requirements include unrelated
optimization packages.

```bash
cd uav-safe-merge-marl
.venv/bin/python -m unittest discover -s experiments/p0_p1 -p 'test_*.py' -v
.venv/bin/python -m experiments.p0_p1.matrix train --stage pilot --rollout-threads 16
```

Add `--execute` to run the six pilot jobs. They are 2.5 million steps each;
the MIT-map run is a provisional pilot only. A full MIT-map pre-experiment can
be enumerated with `matrix train --stage provisional` (20 runs, 5 million steps
each). The wrapper records a non-overwriting manifest, log and final actor
checkpoint path under `experiments/p0_p1/runs/`; MIT checkpoint files remain
under `onpolicy/results/`. Pilot seeds are 1001–1003; final seeds are
2001–2010; validation scenes are 30001–30010; test scenes are 40001–40030.
All sets are disjoint. `matrix` evaluates each final actor on ten log-spaced
headways × ten validation scenes (1000 jobs), then two fleets × ten actors ×
three frozen loads × 30 test scenes (1800 jobs).

On MIT's map, the pre-experiment runs through the same matrix with the
`provisional` stage. Complete the six pilot jobs first, inspect convergence
and runtime, then use:

```bash
python -m experiments.p0_p1.matrix train --stage provisional --execute
python -m experiments.p0_p1.matrix validation --stage provisional --execute
python -m experiments.p0_p1.report calibrate \
  experiments/p0_p1/runs/provisional/evaluations/validation --exploratory \
  --output experiments/p0_p1/runs/provisional/calibration.json
python -m experiments.p0_p1.matrix test --stage provisional \
  --loads-json experiments/p0_p1/runs/provisional/calibration.json --execute
python -m experiments.p0_p1.report gate1 \
  experiments/p0_p1/runs/provisional/evaluations/test --exploratory \
  --calibration-json experiments/p0_p1/runs/provisional/calibration.json \
  --output experiments/p0_p1/runs/provisional/gate1.json
python -m experiments.p0_p1.matrix density --stage provisional \
  --loads-json experiments/p0_p1/runs/provisional/calibration.json --execute
python -m experiments.p0_p1.report density \
  experiments/p0_p1/runs/provisional/evaluations/test/MIX \
  experiments/p0_p1/runs/provisional/evaluations/density/H-M --exploratory \
  --output experiments/p0_p1/runs/provisional/density.json
```

These results are labelled exploratory throughout. Repeat the formal sequence
below once the team's map, scripted free-flow times, and geometry checks are
frozen. Do not reuse pilot or provisional policies as formal policies.

When the team map and scripted free-flow controller are ready, replace
`your_package.your_map` below with its importable module and run the sequence:

```bash
python -m experiments.p0_p1.matrix train --stage pilot --map-module your_package.your_map --execute
# Inspect pilot convergence, paired-seed SD and wall time; freeze budget/seed count.
python -m experiments.p0_p1.matrix train --stage formal --map-module your_package.your_map --execute
python -m experiments.p0_p1.matrix validation --stage formal --map-module your_package.your_map --execute
python -m experiments.p0_p1.report calibrate experiments/p0_p1/runs/formal/evaluations/validation \
  --output experiments/p0_p1/runs/formal/calibration.json
python -m experiments.p0_p1.matrix test --stage formal --map-module your_package.your_map \
  --loads-json experiments/p0_p1/runs/formal/calibration.json --execute
python -m experiments.p0_p1.report gate1 experiments/p0_p1/runs/formal/evaluations/test \
  --calibration-json experiments/p0_p1/runs/formal/calibration.json \
  --output experiments/p0_p1/runs/formal/gate1.json
python -m experiments.p0_p1.matrix density --stage formal --map-module your_package.your_map \
  --loads-json experiments/p0_p1/runs/formal/calibration.json --execute
python -m experiments.p0_p1.report density \
  experiments/p0_p1/runs/formal/evaluations/test/MIX \
  experiments/p0_p1/runs/formal/evaluations/density/H-M \
  --output experiments/p0_p1/runs/formal/density.json
```

The 700-second training episode length must exceed the worst possible
`T_max`: for five tasks per entrance under the fixed training range, a bound
is `5.8 × 66.7 + 1.5 × max(T_ff) + 60` seconds. Increase
`--episode-length` if the team map's scripted free-flow time requires it.
The evaluator checks its episode length against each scenario's `T_max` and
refuses a shorter run. Inspect P0's learning-curve and precision checks
before interpreting Gate 1; the numerical gate report alone does not certify
P0. A zero medium-load baseline leaves its 10% minimum effect undefined and
must be resolved on validation data before opening the test split.
The density run reuses the H-M actors and all 30 test seeds, but divides the
H-M headway by the OD-weighted mean free-flow-time ratio
`(Tff_S + Tff_M + Tff_L)/(3*Tff_M)` for each scene. Its 900 evaluations are a
secondary P1 analysis; they do not alter the primary Gate 1.

For a final map, pass `--map-module your_package.your_map` to all commands.
Formal training refuses to start without it.
The module must export `install_map(scenario, world)` and
`free_flow_seconds(scenario, world, task, envelope)` using a no-traffic scripted
centerline controller, install three MIT-style
`world.tube_params` dictionaries in order (entry 0, entry 1, shared stem 2),
set `scenario.tube_width` and `scenario.progress_gain`, and set
`world.map_metadata` with:

```python
{
    "entrance_count": 2,
    "merge_count": 1,
    "min_bend_radius_m": 280,  # or greater
    "min_approach_m": 500,     # >=417 required
    "corridor_width_m": 66,
    "spawn_length_m": 300,     # or greater
}
```

The adapter is validated at scenario creation. Place geometry in the module;
if it reads separate files, export `MAP_DATA_FILES` containing their paths so
the map fingerprint includes their contents. The current MIT reward logic
supports exactly two entry branches and one outgoing stem. A more complex
team-map topology requires extending the adapter and MIT phase/goal transition
before final training. Geometry, route free-flow times, load grid and the map
module version are frozen before opening the test split.

## AutoDL pilot target

Start with one CUDA-capable 24 GB RTX 4090 instance with at least 16 vCPU and
64 GB RAM (or a 3090 with the same CPU/memory allocation). In AutoDL's base
image selector choose PyTorch 2.1.2, Python 3.10, CUDA 11.8, then install
`requirements-p0p1.txt`. Upload `UAV/p0p1_autodl_source.tar.gz`, extract it,
and run `bash experiments/p0_p1/bootstrap_autodl.sh` from the extracted repo
to check CUDA, dependencies and tests. Run a short 10-aircraft, 16-worker sample and record
environment steps/second, GPU memory, CPU utilization, and peak RAM before
choosing the final worker count. The code's device selection supports CUDA;
its local Apple Silicon test used CPU.

### One-seed MIT-map baseline within a 12–24 hour budget

MIT's public `three_phase.sh` requests 5 million environment steps for each
`air_taxi` run and lists seeds 0 and 1. It requests one Volta GPU and 20 CPUs
per Slurm array job, but does not report wall-clock training time. Its setting
uses five aircraft, a 120-step episode and 256 rollout workers; our provisional
P0 E3 setting uses ten aircraft, a 700-step episode, and initially 16 workers.
These runtimes cannot be equated.

After bootstrap on the RTX 4090 instance, measure throughput with a separate
short seed, then run one full E3 seed if projected training time is within the
approved 24-hour budget:

```bash
python -m wandb login
python -m experiments.p0_p1.train --fleet MIX --seed 7000 \
  --stage smoke --steps 56000 --rollout-threads 16 --episode-length 700 --wandb
python -m experiments.p0_p1.train --fleet MIX --seed 7001 \
  --stage provisional --steps 5000000 --rollout-threads 16 --episode-length 700 --wandb
```

Sign in to W&B interactively on the AutoDL instance; do not put the API key in
the command, script, source archive or chat. `--wandb` streams PPO losses,
reward, MIT per-agent diagnostics and P0/P1 progress snapshots to project
`p0p1_corridor` in the signed-in account. To use a team project, add
`--wandb-entity TEAM_NAME` to both commands. Metrics are logged every five PPO
updates, initially every 56,000 environment steps (about 7.5 minutes in the
first AutoDL benchmark); the local `train.log` is also retained. The
`p0p1/arrived_fraction`, `p0p1/on_time_fraction`, `p0p1/I_percent` and
`p0p1/N_col` curves are snapshots of the rollout environments at logging time,
not the fixed-scene P0 acceptance results. They help detect stalled learning.
If W&B is unreachable, run with `WANDB_MODE=offline` and use `wandb sync`
afterward; offline mode does not provide a live dashboard.
The dependency set pins W&B 0.30.0 so the current 86-character `wandb_v1_...`
keys can be entered at login; W&B 0.15.4 rejects them as longer than 40
characters.

The full command executes 4,995,200 steps because the trainer rounds down to
whole 16 × 700-step batches. At 58 environment steps/s it takes about 24 hours;
at 116 steps/s, about 12 hours. Our 56,000-step Mac run averaged 50.3 steps/s,
which extrapolates to about 27.6 hours. The AutoDL 4090 benchmark completed
56,000 steps in 451.69 seconds, or 124 steps/s, projecting about 11.2 hours
for a 4,995,200-step run with the same settings. This remains a projection;
monitor the full run. The smoke run uses a distinct seed because it is a capacity test,
not a scientific replicate. Report the full run as a single-seed exploratory
baseline only; assess the final actor on fixed scenes and inspect its learning
curve before interpreting completion or safety metrics.
The benchmark's fixed 20-task scene still had 0% on-time completion after
56,000 steps, so it is only a systems check.
On AutoDL, `postrun_provisional_single_seed.sh 7001` may run in a second
detached `screen` session while training continues. It waits for a successful
training manifest, then evaluates the frozen actor on three fixed scene seeds
at each of the ad hoc 30/12/5-second diagnostic headways (nine scenes total).
It writes per-scene JSON and `summary.json` under
`experiments/p0_p1/runs/provisional/evaluations/single_seed_7001/`.
These three labels are only diagnostic loads; they do not satisfy P0's
validation-set calibration or multi-seed acceptance gates.

Do not report P0/P1 effects or gate decisions from the 400-step smoke run.
