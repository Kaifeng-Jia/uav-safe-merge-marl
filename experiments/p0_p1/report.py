"""Freeze P0 loads from validation and report the P1 paired Gate 1 test."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .calibration import (
    LoadPoint, gate1_interaction, gate_decision, paired_interval,
    paired_seed_differences, select_loads,
)
from .protocol import configuration_hash


def read_results(paths):
    files = []
    for item in paths:
        path = Path(item)
        files.extend(sorted(path.rglob("*.json")) if path.is_dir() else [path])
    if not files:
        raise ValueError("No evaluation JSON files found")
    return [json.loads(path.read_text()) for path in files]


def calibrate(results, exploratory=False):
    if any(r["split"] != "validation" or r["fleet"] != "MIX" or r["load"] != "grid"
           for r in results):
        raise ValueError("P0 calibration needs E3 validation-grid results only")
    if not exploratory and any(r.get("provisional_map", True) for r in results):
        raise ValueError("Formal calibration requires the frozen team map")
    for field in ("backbone_commit", "experiment_source_fingerprint", "map_module_sha256"):
        if len({r.get(field) for r in results}) != 1:
            raise ValueError(f"Mixed code/map versions: {field}")
    grouped = defaultdict(list)
    for result in results:
        grouped[float(result["headway"])].append(result)
    if len(grouped) < 10:
        raise ValueError("P0 needs at least ten headway grid points")
    if not exploratory:
        reference = None
        for rows in grouped.values():
            pairs = {(r["training_seed"], r["scene_seed"]) for r in rows}
            if len(pairs) != len(rows) or len({seed for seed, _ in pairs}) < 10 or \
                    len({scene for _, scene in pairs}) < 10:
                raise ValueError("Each grid point needs ten seeds and ten unique validation scenes")
            if reference is None:
                reference = pairs
            elif pairs != reference:
                raise ValueError("Validation grid points need identical seed/scenario pairs")
            if any(r.get("fixed_length_truncated", True) for r in rows):
                raise ValueError("A validation episode was cut off before T_max")
        for seed in {r["training_seed"] for r in results}:
            actors = {r.get("actor_sha256") for r in results if r["training_seed"] == seed}
            if len(actors) != 1 or None in actors:
                raise ValueError("A training seed used multiple E3 actor checkpoints")
    points = []
    for headway, rows in grouped.items():
        success = float(np.mean([r["metrics"]["S_task_percent"] for r in rows]))
        points.append(LoadPoint(headway=headway, task_success_percent=success,
                                throughput=2 / headway * success / 100))
    loads = select_loads(points)
    medium = grouped[loads["med"]]
    baselines = {metric: float(np.mean([r["metrics"][metric] for r in medium]))
                 for metric in ("I_percent", "D_bar")}
    deltas = {metric: 0.1 * abs(value) for metric, value in baselines.items()}
    return {"headway_seconds": loads,
            "e3_medium_validation_mean": baselines,
            "gate1_delta_min": deltas,
            "validation_scenes": sorted({r["scene_seed"] for r in results}),
            "source_files": len(results),
            "configuration_hash": configuration_hash({"loads": loads, "count": len(results)})}


def gate1(results, delta_i, delta_d, exploratory=False):
    if any(r["split"] != "test" or r["load"] not in ("low", "med", "high")
           for r in results):
        raise ValueError("Gate 1 needs low/med/high test evaluations only")
    if not exploratory and any(r.get("provisional_map", True) or
                               r.get("free_flow_method") != "map_scripted_controller"
                               for r in results):
        raise ValueError("Formal Gate 1 requires the frozen map and scripted free-flow times")
    if not exploratory and any(r.get("fixed_length_truncated", True) for r in results):
        raise ValueError("A test episode was cut off before T_max")
    for load in ("low", "med", "high"):
        if len({r["headway"] for r in results if r["load"] == load}) != 1:
            raise ValueError(f"Inconsistent headway at {load} load")
    for field in ("backbone_commit", "experiment_source_fingerprint", "map_module_sha256"):
        known = {r[field] for r in results if field in r}
        if len(known) > 1:
            raise ValueError(f"Mixed code/map versions: {field}")
    if not exploratory:
        for fleet in ("H-M", "MIX"):
            for seed in {r["training_seed"] for r in results if r["fleet"] == fleet}:
                actors = {r.get("actor_sha256") for r in results
                          if r["fleet"] == fleet and r["training_seed"] == seed}
                if len(actors) != 1 or None in actors:
                    raise ValueError("A training seed used multiple actor checkpoints")
    family = []
    detail = {}
    for metric, delta in (("I_percent", delta_i), ("D_bar", delta_d)):
        by_fleet = {}
        for fleet in ("H-M", "MIX"):
            by_load = {}
            for load in ("low", "med", "high"):
                pairs = defaultdict(dict)
                for result in results:
                    if result["fleet"] == fleet and result["load"] == load:
                        seed = result["training_seed"]
                        if seed is None:
                            raise ValueError("Gate 1 needs training_seed on every result")
                        scene = result["scene_seed"]
                        if scene in pairs[seed]:
                            raise ValueError("Duplicate seed/scenario evaluation")
                        pairs[seed][scene] = result["metrics"][metric]
                by_load[load] = pairs
            by_fleet[fleet] = by_load
        diffs = {load: paired_seed_differences(by_fleet["MIX"][load],
                                                by_fleet["H-M"][load])
                 for load in ("low", "med", "high")}
        reference_seeds = set(by_fleet["MIX"]["low"])
        for fleet in ("H-M", "MIX"):
            for load in ("low", "med", "high"):
                if set(by_fleet[fleet][load]) != reference_seeds:
                    raise ValueError("All loads and fleets need the same training seeds")
                for seed in reference_seeds:
                    if set(by_fleet[fleet][load][seed]) != set(by_fleet["MIX"]["low"][seed]):
                        raise ValueError("All loads and fleets need the same scenario seeds")
        if not exploratory and (len(diffs["low"]) < 10 or
                                any(len(scenes) < 30 for scenes in by_fleet["MIX"]["low"].values())):
            raise ValueError("Formal Gate 1 needs at least ten training seeds and 30 shared test scenarios")
        interaction = gate1_interaction(diffs,
                                        {load: np.zeros_like(diffs[load]) for load in diffs})
        detail[metric] = {"low_effect": float(np.mean(diffs["low"])),
                          "med_interaction_mean": float(np.mean(interaction["med"])),
                          "high_interaction_mean": float(np.mean(interaction["high"])),
                          "med_95ci": paired_interval(interaction["med"]),
                          "high_95ci": paired_interval(interaction["high"])}
        family.extend((interaction["med"], interaction["high"]))
    decision = gate_decision(family, (delta_i, delta_i, delta_d, delta_d))
    return {"gate1": decision, "effects": detail,
            "interpretation": "exploratory" if exploratory else "preregistered"}


def density_comparison(main_results, matched_results, exploratory=False):
    """Secondary MIX versus density-matched H-M paired seed analysis."""
    if any(r["split"] != "test" or r["fleet"] != "MIX" for r in main_results):
        raise ValueError("Density analysis needs main MIX test results")
    if any(r["split"] != "density" or r["fleet"] != "H-M" or
           not r["load"].startswith("density-") for r in matched_results):
        raise ValueError("Density analysis needs matched H-M test results")
    all_results = main_results + matched_results
    if not exploratory and any(r.get("provisional_map", True) or
                               r.get("free_flow_method") != "map_scripted_controller" or
                               r.get("fixed_length_truncated", True) for r in all_results):
        raise ValueError("Formal density analysis needs the complete frozen-map test set")
    for field in ("backbone_commit", "experiment_source_fingerprint", "map_module_sha256"):
        if len({r.get(field) for r in all_results}) != 1:
            raise ValueError(f"Mixed code/map versions: {field}")
    output = {}
    reference_pairs = None
    for metric in ("I_percent", "D_bar"):
        by_load = {}
        for load in ("low", "med", "high"):
            pair = {}
            for fleet, rows, label in (("MIX", main_results, load),
                                       ("H-M", matched_results, f"density-{load}")):
                by_seed = defaultdict(dict)
                for row in rows:
                    if row["load"] == label:
                        seed, scene = row["training_seed"], row["scene_seed"]
                        if scene in by_seed[seed]:
                            raise ValueError("Duplicate matched test scene")
                        by_seed[seed][scene] = row["metrics"][metric]
                pairs = {(seed, scene) for seed, scenes in by_seed.items() for scene in scenes}
                if reference_pairs is None:
                    reference_pairs = pairs
                elif pairs != reference_pairs:
                    raise ValueError("Density analysis needs the same seed/scenario pairs across loads")
                pair[fleet] = by_seed
            by_load[load] = paired_seed_differences(pair["MIX"], pair["H-M"])
        interactions = gate1_interaction(by_load,
                                         {load: np.zeros_like(value)
                                          for load, value in by_load.items()})
        output[metric] = {
            "paired_difference": {load: {"mean": float(np.mean(values)),
                                         "ci95": paired_interval(values)}
                                  for load, values in by_load.items()},
            "interaction": {load: {"mean": float(np.mean(values)),
                                   "ci95": paired_interval(values)}
                            for load, values in interactions.items()},
        }
    ratios = [row["density_matching_ratio"] for row in matched_results]
    if any(ratio is None or ratio <= 0 for ratio in ratios):
        raise ValueError("Matched runs need a positive OD-weighted free-flow ratio")
    if not exploratory and (len({seed for seed, _ in reference_pairs}) < 10 or
                            len({scene for _, scene in reference_pairs}) < 30):
        raise ValueError("Formal density analysis needs ten seeds and 30 test scenes")
    return {"secondary_density_matching": output,
            "ratio_mean": float(np.mean(ratios)),
            "interpretation": "exploratory" if exploratory else
                              "descriptive secondary analysis; do not replace primary Gate 1"}


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    cal = subparsers.add_parser("calibrate")
    cal.add_argument("results", nargs="+")
    cal.add_argument("--exploratory", action="store_true")
    cal.add_argument("--output", type=Path, required=True)
    gate = subparsers.add_parser("gate1")
    gate.add_argument("results", nargs="+")
    gate.add_argument("--delta-i", type=float)
    gate.add_argument("--delta-d", type=float)
    gate.add_argument("--calibration-json", type=Path)
    gate.add_argument("--exploratory", action="store_true")
    gate.add_argument("--output", type=Path, required=True)
    density = subparsers.add_parser("density")
    density.add_argument("main_results", type=Path)
    density.add_argument("matched_results", type=Path)
    density.add_argument("--exploratory", action="store_true")
    density.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "density":
        payload = density_comparison(read_results([args.main_results]),
                                     read_results([args.matched_results]), args.exploratory)
    elif args.command == "calibrate":
        results = read_results(args.results)
        payload = calibrate(results, args.exploratory)
    else:
        results = read_results(args.results)
        if args.calibration_json:
            calibration = json.loads(args.calibration_json.read_text())
            deltas = calibration["gate1_delta_min"]
            args.delta_i = deltas["I_percent"]
            args.delta_d = deltas["D_bar"]
            for load, headway in calibration["headway_seconds"].items():
                values = {r["headway"] for r in results if r["load"] == load}
                if values != {headway}:
                    raise ValueError(f"Test {load} headway differs from P0 calibration")
        if args.delta_i is None or args.delta_d is None:
            parser.error("Gate 1 needs --calibration-json or both frozen --delta values")
        payload = gate1(results, args.delta_i, args.delta_d, args.exploratory)
    if args.output.exists():
        parser.error("Refusing to overwrite a frozen result file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
