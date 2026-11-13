"""Finish the paired MIT-map E1 preview after its first 5M-step segment.

This orchestration file is outside the frozen experiment source fingerprint.
The E1 schedule deliberately mirrors the completed E3 seed-7001 preview:
5M steps, then a weights-only 5M-step warm start.
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.p0_p1.protocol import backbone_commit, source_fingerprint


FIRST = ROOT / "experiments/p0_p1/runs/provisional/H-M/7001/manifest.json"
SECOND_DIR = ROOT / "experiments/p0_p1/runs/provisional/H-M/7001_warmstart_5m"
E3_FIRST = ROOT / "experiments/p0_p1/runs/provisional/MIX/7001/manifest.json"
E3_FINAL = ROOT / "experiments/p0_p1/runs/provisional/MIX/7001_warmstart_5m/manifest.json"
EVAL_DIR = ROOT / "experiments/p0_p1/runs/provisional/evaluations/p1_pair_7001"
PYTHON = sys.executable
LOADS = (("low", 30), ("med", 12), ("high", 5))
SCENES = (60021, 60022, 60023)


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def wait_for_first():
    while True:
        if FIRST.exists():
            state = read_json(FIRST)
            if state.get("exit_code") is not None:
                if state["exit_code"] != 0:
                    raise RuntimeError(f"E1 first segment failed: {FIRST}")
                return state
        time.sleep(60)


def replace(command, flag, value):
    command[command.index(flag) + 1] = str(value)


def comparable_command(command):
    normalized = list(command)
    for flag in ("--p0p1_fleet", "--experiment_name"):
        replace(normalized, flag, "<matched>")
    return normalized


def warm_start(first):
    e3_first = read_json(E3_FIRST)
    if first["training_seed"] != 7001 or first["fleet"] != "H-M":
        raise ValueError("Expected E1/H-M seed 7001")
    if first["actual_environment_steps"] != e3_first["actual_environment_steps"]:
        raise ValueError("E1 and E3 first-stage budgets differ")
    if first["experiment_source_fingerprint"] != e3_first["experiment_source_fingerprint"]:
        raise ValueError("E1 and E3 experiment sources differ")
    if first["experiment_source_fingerprint"] != source_fingerprint(ROOT):
        raise ValueError("Experiment source changed since E1 training started")
    if first["backbone_commit"] != backbone_commit(ROOT):
        raise ValueError("Backbone revision changed")
    if comparable_command(first["command"]) != comparable_command(e3_first["command"]):
        raise ValueError("E1 and E3 first-stage commands differ beyond fleet/name")

    original_actor = ROOT / first["actor_checkpoint"]
    original_critic = original_actor.with_name("critic.pt")
    if not original_actor.is_file() or not original_critic.is_file():
        raise FileNotFoundError("E1 first-stage actor/critic missing")
    if SECOND_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite {SECOND_DIR}")

    command = list(first["command"])
    command[0] = PYTHON
    replace(command, "--experiment_name", "p0p1_provisional_HM_7001_warmstart_5m")
    replace(command, "--num_env_steps", 5_000_000)
    command.extend(("--model_dir", str(original_actor.parent)))
    segment_steps = 5_000_000
    rollout_threads = int(command[command.index("--n_rollout_threads") + 1])
    episode_length = int(command[command.index("--episode_length") + 1])
    actual_steps = segment_steps // (rollout_threads * episode_length) * rollout_threads * episode_length
    manifest = {
        "stage": "provisional",
        "fleet": "H-M",
        "training_seed": 7001,
        "run_kind": "weights_only_warm_start",
        "parent_manifest": str(FIRST.relative_to(ROOT)),
        "parent_actor_sha256": sha256(original_actor.read_bytes()).hexdigest(),
        "parent_critic_sha256": sha256(original_critic.read_bytes()).hexdigest(),
        "optimizer_state_restored": False,
        "rng_state_restored": False,
        "parent_environment_steps": first["actual_environment_steps"],
        "requested_additional_environment_steps": segment_steps,
        "actual_additional_environment_steps": actual_steps,
        "cumulative_nominal_environment_steps": first["actual_environment_steps"] + actual_steps,
        "backbone_commit": first["backbone_commit"],
        "experiment_source_fingerprint": source_fingerprint(ROOT),
        "map_module": first["map_module"],
        "map_module_sha256": first["map_module_sha256"],
        "wandb": first["wandb"],
        "command": command,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    SECOND_DIR.mkdir(parents=True)
    manifest_path = SECOND_DIR / "manifest.json"
    write_json(manifest_path, manifest)
    print("Starting E1 weights-only warm start", flush=True)
    started = time.monotonic()
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with (SECOND_DIR / "train.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    manifest["exit_code"] = result.returncode
    manifest["wall_seconds"] = round(time.monotonic() - started, 3)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    if result.returncode == 0:
        checkpoint_root = (ROOT / "onpolicy/results/GraphMPE/p0p1_merge_graph/rmappo"
                           / "p0p1_provisional_HM_7001_warmstart_5m")
        checkpoints = list(checkpoint_root.rglob("actor.pt"))
        if len(checkpoints) != 1 or not checkpoints[0].with_name("critic.pt").is_file():
            manifest["exit_code"] = 1
            manifest["checkpoint_error"] = f"Expected one actor/critic pair; found {len(checkpoints)} actors"
        else:
            manifest["actor_checkpoint"] = str(checkpoints[0].relative_to(ROOT))
    write_json(manifest_path, manifest)
    if manifest["exit_code"] != 0:
        raise RuntimeError(f"E1 warm start failed; see {SECOND_DIR / 'train.log'}")
    return manifest


def evaluate_pair(e1_final):
    e3_final = read_json(E3_FINAL)
    if e3_final.get("exit_code") != 0:
        raise ValueError("E3 paired actor is unavailable")
    if e3_final["cumulative_nominal_environment_steps"] != e1_final["cumulative_nominal_environment_steps"]:
        raise ValueError("E1 and E3 cumulative budgets differ")
    if e3_final["experiment_source_fingerprint"] != e1_final["experiment_source_fingerprint"]:
        raise ValueError("E1 and E3 experiment sources differ")
    actors = {
        "H-M": ROOT / e1_final["actor_checkpoint"],
        "MIX": ROOT / e3_final["actor_checkpoint"],
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    for fleet, actor in actors.items():
        if not actor.is_file():
            raise FileNotFoundError(actor)
        folder = EVAL_DIR / fleet
        folder.mkdir(exist_ok=True)
        actor_hash = sha256(actor.read_bytes()).hexdigest()
        for load, headway in LOADS:
            for scene in SCENES:
                output = folder / f"{load}_{scene}.json"
                if output.exists():
                    prior = read_json(output)
                    if prior.get("actor_sha256") != actor_hash:
                        raise ValueError(f"Wrong actor in existing evaluation: {output}")
                    continue
                command = [
                    PYTHON, "-m", "experiments.p0_p1.evaluate",
                    "--actor", str(actor), "--fleet", fleet,
                    "--train-seed", "7001", "--split", "smoke",
                    "--load", load, "--scene-seed", str(scene),
                    "--headway", str(headway), "--agents", "20",
                    "--episode-length", "1200", "--output", str(output),
                ]
                with output.with_suffix(".log").open("w") as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                row = read_json(output)
                if row["actor_sha256"] != actor_hash or row["fixed_length_truncated"]:
                    raise ValueError(f"Invalid evaluation: {output}")
                print(f"Evaluated {fleet} {load} {scene}", flush=True)

    metrics = ("S_task_percent", "I_percent", "D_bar", "N_col", "C_percent")
    summary = {
        "stage": "provisional",
        "map": "MIT three_phase_graph_merge",
        "training_seed": 7001,
        "cumulative_nominal_environment_steps": e1_final["cumulative_nominal_environment_steps"],
        "loads_seconds": dict(LOADS),
        "scene_seeds": list(SCENES),
        "note": "One training seed, three reused diagnostic scenes per load, ad hoc loads, provisional free-flow estimate. Descriptive P1 pipeline check only; no formal Gate 1 inference.",
        "by_load": {},
    }
    for load, headway in LOADS:
        e1 = [read_json(EVAL_DIR / "H-M" / f"{load}_{scene}.json") for scene in SCENES]
        e3 = [read_json(EVAL_DIR / "MIX" / f"{load}_{scene}.json") for scene in SCENES]
        for a, b in zip(e1, e3):
            if (a["scene_seed"], a["headway"], a["experiment_source_fingerprint"]) != (
                b["scene_seed"], b["headway"], b["experiment_source_fingerprint"]
            ):
                raise ValueError("P1 scene/config pairing failed")
        summary["by_load"][load] = {
            "headway_seconds": headway,
            "E1_HM": {m: sum(r["metrics"][m] for r in e1) / len(e1) for m in metrics},
            "E3_MIX": {m: sum(r["metrics"][m] for r in e3) / len(e3) for m in metrics},
            "E3_minus_E1": {
                m: sum(b["metrics"][m] - a["metrics"][m] for a, b in zip(e1, e3)) / len(e1)
                for m in metrics
            },
        }
    write_json(EVAL_DIR / "summary.json", summary)
    print(f"Paired P1 preview complete: {EVAL_DIR / 'summary.json'}", flush=True)


def main():
    os.chdir(ROOT)
    first = wait_for_first()
    final = warm_start(first)
    evaluate_pair(final)


if __name__ == "__main__":
    main()
