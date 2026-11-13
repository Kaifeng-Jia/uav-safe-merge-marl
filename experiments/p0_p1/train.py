"""Launch one seed of the MIT rmappo backbone with recorded P0/P1 settings."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.p0_p1.protocol import (
    COORDINATION_RADIUS, backbone_commit, configuration_hash, map_module_hash,
    source_fingerprint,
)


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fleet", choices=["H-M", "MIX"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", choices=["smoke", "pilot", "provisional", "formal"], required=True)
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--rollout-threads", type=int, default=4)
    parser.add_argument("--episode-length", type=int, default=700)
    parser.add_argument("--map-module", default=None)
    parser.add_argument("--wandb", action="store_true", help="Stream training metrics to Weights & Biases")
    parser.add_argument("--wandb-project", default="p0p1_corridor")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B team or username")
    parser.add_argument("--log-interval", type=int, default=5, help="PPO updates between metric logs")
    parser.add_argument("--dry-run", action="store_true")
    opts = parser.parse_args()
    if opts.stage == "formal" and opts.map_module is None:
        parser.error("Formal P0/P1 requires the team's frozen map module")
    if opts.log_interval < 1:
        parser.error("--log-interval must be positive")
    run_dir = ROOT / "experiments" / "p0_p1" / "runs" / opts.stage / opts.fleet / str(opts.seed)
    if run_dir.exists() and not opts.dry_run:
        parser.error(f"Run directory already exists: {run_dir}")
    experiment_name = f"p0p1_{opts.stage}_{opts.fleet.replace('-', '')}_{opts.seed}"
    command = [
        sys.executable, "-u", "onpolicy/scripts/train_mpe.py",
        "--project_name", opts.wandb_project,
        "--experiment_name", experiment_name, "--env_name", "GraphMPE",
        "--algorithm_name", "rmappo", "--scenario_name", "p0p1_merge_graph",
        "--p0p1_fleet", opts.fleet, "--seed", str(opts.seed),
        "--dynamics_type", "air_taxi", "--formation_type", "point",
        "--num_agents", "10", "--num_landmarks", "10",
        "--num_obstacles", "0", "--num_walls", "0",
        "--max_edge_dist", str(COORDINATION_RADIUS),
        "--episode_length", str(opts.episode_length),
        "--num_env_steps", str(opts.steps), "--n_rollout_threads", str(opts.rollout_threads),
        "--n_training_threads", "1", "--ppo_epoch", "10",
        "--num_mini_batch", "1", "--graph_feat_type", "relative",
        "--world_size", "5", "--collision_rew", "20", "--goal_rew", "20",
        "--formation_rew", "10", "--fair_wt", "1", "--fair_rew", "10",
        "--total_actions", "25", "--use_cent_obs", "False",
        "--use_dones", "False", "--use_popart", "--use_valuenorm",
        "--lr", "5e-4", "--critic_lr", "5e-4", "--opti_eps", "1e-5",
        "--clip_param", "0.2", "--entropy_coef", "0.02",
        "--max_grad_norm", "10", "--data_chunk_length", "10",
        "--save_interval", "25", "--log_interval", str(opts.log_interval), "--verbose",
    ]
    # MIT defined --use_wandb as store_false. Pass it only for local TensorBoard.
    if not opts.wandb:
        command += ["--use_wandb"]
    if opts.wandb_entity:
        command += ["--user_name", opts.wandb_entity]
    if opts.rollout_threads * opts.episode_length * 10 >= 16_384:
        command += ["--auto_mini_batch_size", "--target_mini_batch_size", "16384"]
    if opts.map_module:
        command += ["--p0p1_map_module", opts.map_module]
    source = backbone_commit(ROOT)
    manifest = {
        "stage": opts.stage, "fleet": opts.fleet, "training_seed": opts.seed,
        "requested_environment_steps": opts.steps,
        "actual_environment_steps": (opts.steps // opts.episode_length // opts.rollout_threads)
                                    * opts.episode_length * opts.rollout_threads,
        "backbone_commit": source,
        "experiment_source_fingerprint": source_fingerprint(ROOT),
        "map_module": opts.map_module or "MIT three_phase_graph_merge (provisional)",
        "map_module_sha256": map_module_hash(opts.map_module),
        "wandb": {"enabled": opts.wandb, "project": opts.wandb_project,
                  "entity": opts.wandb_entity, "log_interval_updates": opts.log_interval},
        "command": command,
    }
    manifest["configuration_hash"] = configuration_hash(manifest)
    if opts.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return
    run_dir.mkdir(parents=True)
    manifest["started_utc"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    started = time.monotonic()
    with (run_dir / "train.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode == 0:
        checkpoints = list((ROOT / "onpolicy" / "results" / "GraphMPE" /
                            "p0p1_merge_graph" / "rmappo" / experiment_name).rglob("actor.pt"))
        if len(checkpoints) != 1:
            result.returncode = 1
            manifest["checkpoint_error"] = f"Expected one final actor; found {len(checkpoints)}"
        else:
            manifest["actor_checkpoint"] = str(checkpoints[0].relative_to(ROOT))
    manifest["exit_code"] = result.returncode
    manifest["wall_seconds"] = round(time.monotonic() - started, 3)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: manifest[k] for k in ("fleet", "training_seed", "exit_code", "wall_seconds")}, indent=2))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
