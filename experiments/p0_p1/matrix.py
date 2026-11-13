"""Enumerate and resume the registered P0/P1 run matrix.

The default prints a compact job count and example command. --execute runs
pending jobs serially; completed outputs are checked before being skipped.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .protocol import evaluation_grid, map_module_hash, source_fingerprint


ROOT = Path(__file__).resolve().parents[2]
PILOT_SEEDS = tuple(range(1001, 1004))
FINAL_SEEDS = tuple(range(2001, 2011))
VALIDATION_SCENES = tuple(range(30001, 30011))
TEST_SCENES = tuple(range(40001, 40031))


def manifest(stage, fleet, seed):
    path = ROOT / "experiments" / "p0_p1" / "runs" / stage / fleet / str(seed) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"First train {stage}/{fleet}/{seed}: {path}")
    data = json.loads(path.read_text())
    if data.get("exit_code") != 0 or not data.get("actor_checkpoint"):
        raise ValueError(f"Training did not finish cleanly: {path}")
    return data


def training_jobs(stage, map_module, rollout_threads, episode_length):
    seeds = PILOT_SEEDS if stage == "pilot" else FINAL_SEEDS
    steps = 2_500_000 if stage == "pilot" else 5_000_000
    for seed in seeds:
        for fleet in ("H-M", "MIX"):
            command = [sys.executable, "-m", "experiments.p0_p1.train",
                       "--fleet", fleet, "--seed", str(seed), "--stage", stage,
                       "--steps", str(steps), "--rollout-threads", str(rollout_threads),
                       "--episode-length", str(episode_length)]
            if map_module:
                command += ["--map-module", map_module]
            output = (ROOT / "experiments" / "p0_p1" / "runs" / stage /
                      fleet / str(seed) / "manifest.json")
            yield command, output


def evaluation_jobs(stage, split, map_module, episode_length, loads):
    seeds = FINAL_SEEDS
    scenes = VALIDATION_SCENES if split == "validation" else TEST_SCENES
    fleets = (("MIX",) if split == "validation" else
              ("H-M",) if split == "density" else ("H-M", "MIX"))
    grid = {f"grid-{index:02d}": headway
            for index, headway in enumerate(evaluation_grid())}
    headways = grid if split == "validation" else loads
    if split in ("test", "density") and set(headways) != {"low", "med", "high"}:
        raise ValueError("Test requires frozen low/med/high headways")
    current_source = source_fingerprint(ROOT)
    current_map = map_module_hash(map_module)
    for seed in seeds:
        for fleet in fleets:
            saved = manifest(stage, fleet, seed)
            if saved["experiment_source_fingerprint"] != current_source:
                raise ValueError(f"Code changed since training {stage}/{fleet}/{seed}")
            if saved["map_module_sha256"] != current_map:
                raise ValueError(f"Map changed since training {stage}/{fleet}/{seed}")
            actor = ROOT / saved["actor_checkpoint"]
            if not actor.exists():
                raise FileNotFoundError(actor)
            for label, headway in headways.items():
                for scene in scenes:
                    reported_load = ("grid" if split == "validation" else
                                     f"density-{label}" if split == "density" else label)
                    output = (ROOT / "experiments" / "p0_p1" / "runs" / stage /
                              "evaluations" / split / fleet / str(seed) /
                              label / f"scene_{scene}.json")
                    command = [sys.executable, "-m", "experiments.p0_p1.evaluate",
                               "--actor", str(actor), "--fleet", fleet,
                               "--train-seed", str(seed), "--split", split,
                               "--load", reported_load,
                               "--scene-seed", str(scene), "--headway", str(headway),
                               "--agents", "20", "--episode-length", str(episode_length),
                               "--output", str(output)]
                    if map_module:
                        command += ["--map-module", map_module]
                    if split == "density":
                        command += ["--density-match"]
                    yield command, output


def completed(output, split, current_source, current_map):
    if not output.exists():
        return False
    data = json.loads(output.read_text())
    if split == "train":
        if data.get("exit_code") != 0 or not data.get("actor_checkpoint"):
            raise ValueError(f"Incomplete existing training output: {output}")
    else:
        if data.get("experiment_source_fingerprint") != current_source or \
                data.get("map_module_sha256") != current_map:
            raise ValueError(f"Stale existing evaluation output: {output}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["train", "validation", "test", "density"])
    parser.add_argument("--stage", choices=["pilot", "provisional", "formal"], required=True)
    parser.add_argument("--map-module")
    parser.add_argument("--rollout-threads", type=int, default=16)
    parser.add_argument("--episode-length", type=int)
    parser.add_argument("--loads-json", type=Path,
                        help="P0 calibrate output, required for test")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.stage == "formal" and not args.map_module:
        parser.error("Formal runs require --map-module")
    if args.phase != "train" and args.stage == "pilot":
        parser.error("Pilot checkpoints do not enter P0 validation or P1 testing")
    if args.phase in ("test", "density") and not args.loads_json:
        parser.error("Test and density analysis require frozen --loads-json from P0 validation")
    if args.phase == "train":
        jobs = list(training_jobs(args.stage, args.map_module, args.rollout_threads,
                                  args.episode_length or 700))
    else:
        loads = json.loads(args.loads_json.read_text())["headway_seconds"] if args.loads_json else None
        jobs = list(evaluation_jobs(args.stage, args.phase, args.map_module,
                                    args.episode_length or 1800, loads))
    print(f"{args.stage} {args.phase}: {len(jobs)} jobs")
    if jobs:
        print("Example:", " ".join(jobs[0][0]))
    if not args.execute:
        return
    current_source = source_fingerprint(ROOT)
    current_map = map_module_hash(args.map_module)
    for index, (command, output) in enumerate(jobs, start=1):
        if completed(output, args.phase, current_source, current_map):
            continue
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Job {index}/{len(jobs)} failed: {' '.join(command)}\n"
                               f"{result.stdout}\n{result.stderr}")
        if index % 25 == 0 or index == len(jobs):
            print(f"Completed {index}/{len(jobs)}")


if __name__ == "__main__":
    main()
