"""Evaluate a frozen MIT MAPPO actor on one paired P0/P1 scenario.

The current MIT geometry has no validated route-specific scripted free-flow
controller. D_bar in this provisional evaluator uses a route-length estimate;
it must be replaced before testing on the team's final map.
"""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.p0_p1.protocol import (
    COORDINATION_RADIUS, backbone_commit, configuration_hash, map_module_hash,
    source_fingerprint,
)
from multiagent.MPE_env import GraphMPEEnv
from onpolicy.algorithms.graph_actor_critic import GR_Actor
from onpolicy.config import get_config, graph_config
from onpolicy.scripts.train_mpe import parse_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--fleet", choices=["H-M", "MIX"], required=True)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--split", choices=["smoke", "validation", "test", "density"], default="smoke")
    parser.add_argument("--load", choices=["smoke", "grid", "low", "med", "high",
                                           "density-low", "density-med", "density-high"],
                        default="smoke")
    parser.add_argument("--scene-seed", type=int, required=True)
    parser.add_argument("--headway", type=float, required=True)
    parser.add_argument("--agents", type=int, default=20)
    parser.add_argument("--episode-length", type=int, default=1200)
    parser.add_argument("--map-module", default=None)
    parser.add_argument("--density-match", action="store_true",
                        help="For H-M secondary P1 evaluation, multiply release rate by OD-weighted Tff ratio")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.output.exists():
        parser.error(f"Refusing to rerun/overwrite this scenario: {options.output}")
    if not options.actor.is_file():
        parser.error(f"Actor checkpoint does not exist: {options.actor}")
    if options.density_match and options.fleet != "H-M":
        parser.error("Density matching changes the H-M release rate only")
    mit_args = [
        "--scenario_name", "p0p1_merge_graph", "--env_name", "GraphMPE",
        "--algorithm_name", "rmappo", "--dynamics_type", "air_taxi",
        "--formation_type", "point", "--graph_feat_type", "relative",
        "--world_size", "5", "--num_agents", str(options.agents),
        "--num_landmarks", str(options.agents), "--num_obstacles", "0",
        "--num_walls", "0", "--episode_length", str(options.episode_length),
        "--max_edge_dist", str(COORDINATION_RADIUS),
        "--p0p1_fleet", options.fleet, "--p0p1_headway", str(options.headway),
        "--p0p1_scene_seed", str(options.scene_seed), "--total_actions", "25",
        "--collision_rew", "20", "--goal_rew", "20", "--formation_rew", "10",
        "--fair_wt", "1", "--fair_rew", "10", "--use_cent_obs", "False",
    ]
    if options.map_module:
        mit_args.extend(("--p0p1_map_module", options.map_module))
    args, parser = parse_args(mit_args, get_config())
    args, _ = graph_config(mit_args, parser)
    env = GraphMPEEnv(args)
    obs, ids, nodes, adjacency = env.reset()
    density_ratio = None
    if options.density_match:
        density_ratio = env.world.scenario.density_matching_ratio(env.world)
        env.world.scenario.eval_headway = options.headway / density_ratio
        obs, ids, nodes, adjacency = env.reset()
    if options.episode_length < math.ceil(env.world.scenario.tmax):
        raise ValueError(
            f"episode-length {options.episode_length} is shorter than T_max "
            f"{env.world.scenario.tmax:.1f}; increase it before evaluating"
        )
    actor = GR_Actor(args, env.observation_space[0], env.node_observation_space[0],
                     env.edge_observation_space[0], env.action_space[0])
    actor.load_state_dict(torch.load(options.actor, map_location="cpu"))
    actor.eval()
    recurrent = np.zeros((options.agents, args.recurrent_N, args.hidden_size), dtype=np.float32)
    masks = np.ones((options.agents, 1), dtype=np.float32)
    with torch.no_grad():
        for _ in range(options.episode_length):
            action, _, next_recurrent = actor(np.asarray(obs), np.asarray(nodes),
                                              np.asarray(adjacency), np.asarray(ids), recurrent,
                                              masks, deterministic=True)
            chosen = action.cpu().numpy().reshape(-1)
            one_hot = np.eye(25, dtype=np.float32)[chosen]
            obs, ids, nodes, adjacency, _, done, _ = env.step(one_hot)
            recurrent = next_recurrent.cpu().numpy()
            masks = (~np.asarray(done, dtype=bool)).astype(np.float32)[:, None]
            if np.all(done):
                break
    scene = env.world.scenario
    corridor_steps = float(np.sum(scene.steps_in_corridor))
    active_policy_steps = scene.metrics.active_agent_substeps / 10.0
    completed_speeds = [
        env.world.agents[task_id].state.p_dist * 328.08 /
        (arrival - scene.metrics.departure[task_id])
        for task_id, arrival in scene.metrics.arrival.items()
        if task_id in scene.metrics.departure and arrival > scene.metrics.departure[task_id]
    ]
    episode_metrics = scene.metrics.summarize(scene.headway, 2)
    episode_metrics.update({
        "I_corr_percent": 100.0 * float(np.sum(scene.spacing_violation)) / corridor_steps
        if corridor_steps else 0.0,
        "C_percent": 100.0 * (1.0 - float(np.sum(scene.conformance_percent)) /
                                active_policy_steps) if active_policy_steps else 0.0,
        "mean_completed_speed_mps": float(np.mean(completed_speeds))
        if completed_speeds else None,
    })
    result = {
        "provisional_map": options.map_module is None,
        "free_flow_method": ("map_scripted_controller" if options.map_module
                             else "route_distance_over_vmax_estimate"),
        "fleet": options.fleet,
        "training_seed": options.train_seed,
        "actor_sha256": sha256(options.actor.read_bytes()).hexdigest(),
        "split": options.split,
        "load": options.load,
        "scene_seed": options.scene_seed,
        "headway": scene.headway,
        "reference_headway": options.headway,
        "density_matching_ratio": density_ratio,
        "num_agents": options.agents,
        "seconds_simulated": env.world.simulation_time,
        "tmax_truncated": env.world.simulation_time >= scene.tmax and
                           len(scene.metrics.arrival) < len(scene.tasks),
        "fixed_length_truncated": env.world.simulation_time < scene.tmax and
                                  len(scene.metrics.arrival) < len(scene.tasks),
        "metrics": episode_metrics,
        "backbone_commit": backbone_commit(Path(__file__).resolve().parents[2]),
        "experiment_source_fingerprint": source_fingerprint(Path(__file__).resolve().parents[2]),
        "map_module_sha256": map_module_hash(options.map_module),
    }
    result["configuration_hash"] = configuration_hash({k: v for k, v in result.items() if k != "metrics"})
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    env.close()


if __name__ == "__main__":
    main()
