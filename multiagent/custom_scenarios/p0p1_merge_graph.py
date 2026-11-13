"""P0/P1 experiment on MIT's merge geometry.

The map hook is ``--p0p1_map_module``. A replacement module must export
``install_map(scenario, world)`` and install the same three tube dictionaries
used by MIT's merge scenario (two entries and one outgoing tube), with keys
entrance, exit, e, n, L and width. This keeps the training code unchanged when
the team's map arrives. See experiments/p0_p1/README.md for the remaining
geometry checks before that map is frozen.
"""

import importlib

import numpy as np

from experiments.p0_p1.metrics import EpisodeMetrics
from experiments.p0_p1.protocol import (
    COORDINATION_RADIUS, CORRIDOR_WIDTH, FLEETS, GOAL_RADIUS,
    MIN_SEPARATION, PHYSICS_DT, SPAWN_OFFSET, make_tasks, training_headway,
    validate_team_map,
)
from multiagent.core import Landmark
from multiagent.custom_scenarios.three_phase_graph_merge import Scenario as MITMergeScenario


class Scenario(MITMergeScenario):
    def make_world(self, args):
        self.fleet_mode = getattr(args, "p0p1_fleet", "H-M")
        self.eval_headway = getattr(args, "p0p1_headway", None)
        self.map_module = getattr(args, "p0p1_map_module", None)
        if args.num_agents % 2 or args.num_landmarks != args.num_agents:
            raise ValueError("P0/P1 needs an equal number of tasks at two entries and one goal per task")
        world = super().make_world(args)
        # MIT's merge scenario stores goals as coordinates only. Add the three
        # corridor exits as fixed graph entities, as required by the protocol.
        world.landmarks = [Landmark() for _ in world.tube_params]
        for index, landmark in enumerate(world.landmarks):
            landmark.id = index
            landmark.name = f"landmark {index}"
            landmark.global_id = len(world.agents) + index
            landmark.movable = False
            landmark.collide = False
            landmark.state.p_pos = np.asarray(
                self._goal_pos_for_tube(world.tube_params[index]), dtype=float)
            landmark.state.p_vel = np.zeros(2)
        world.cached_dist_vect = None
        world.calculate_distances()
        world.scenario = self
        world.dt = PHYSICS_DT
        world.min_dist_thresh = GOAL_RADIUS
        self.min_dist_thresh = GOAL_RADIUS
        world.coordination_range = COORDINATION_RADIUS
        self.coordination_range = COORDINATION_RADIUS
        self.max_edge_dist = COORDINATION_RADIUS
        self.separation_distance = MIN_SEPARATION
        return world

    def setup_two_tube_params(self, world):
        if self.map_module:
            module = importlib.import_module(self.map_module)
            module.install_map(self, world)
            if not hasattr(module, "free_flow_seconds"):
                raise ValueError("Custom map module must export scripted free_flow_seconds")
            self.free_flow_fn = module.free_flow_seconds
            validate_team_map(world.map_metadata)
            if world.map_metadata["entrance_count"] != 2 or world.map_metadata["merge_count"] != 1:
                raise ValueError("The current MIT phase adapter supports two entries and one merge")
        else:
            super().setup_two_tube_params(world)
            self.free_flow_fn = None

    def reset_world(self, world, num_current_episode=0):
        super().reset_world(world, num_current_episode)
        for index, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.asarray(
                self._goal_pos_for_tube(world.tube_params[index]), dtype=float)
        # The upstream scenario creates the geometry and goals. Tasks, types and
        # releases are generated independently of its placement random numbers.
        fixed_seed = getattr(self.args, "p0p1_scene_seed", None)
        seed = int(fixed_seed if fixed_seed is not None else np.random.randint(0, 2**31 - 1))
        headway = (float(self.eval_headway) if self.eval_headway is not None
                   else training_headway(seed))
        self.tasks = make_tasks(seed, self.fleet_mode, 2, self.num_agents // 2, headway)
        self.task_by_id = {task.id: task for task in self.tasks}
        self.metrics = EpisodeMetrics(self.tasks, {})
        self.headway = headway
        self._departed = set()
        for task in self.tasks:
            agent = world.agents[task.id]
            envelope = FLEETS[task.aircraft_type]
            agent.aircraft_type = task.aircraft_type
            agent.envelope = envelope
            agent.state.min_speed = envelope.v_min / 328.08
            agent.state.max_speed = envelope.v_max / 328.08
            agent.min_speed = agent.state.min_speed
            agent.max_speed = agent.state.max_speed
            agent.dormant = True
            agent.status = True
            # Keep fixed-size graph tensors while the task waits on the ground.
            agent.state.p_pos = np.array([1e4 + task.id * 10.0, 1e4], dtype=float)
            agent.state.reset_velocity(theta=0.0, speed=agent.state.min_speed)
            route = world.tube_params[task.entrance]
            self.current_tube[task.id] = task.entrance
            if self.free_flow_fn:
                seconds = float(self.free_flow_fn(self, world, task, envelope))
                if not np.isfinite(seconds) or seconds <= 0:
                    raise ValueError("Custom map free_flow_seconds must be finite and positive")
                self.metrics.free_flow_seconds[task.id] = seconds
            else:
                self.metrics.free_flow_seconds[task.id] = (
                    SPAWN_OFFSET + route["L"] + world.tube_params[2]["L"]
                ) / agent.state.max_speed
        self.tmax = (max(task.release_time for task in self.tasks)
                     + max(1.5 * value for value in self.metrics.free_flow_seconds.values())
                     + 60.0)
        world.calculate_distances()

    def graph_observation(self, agent, world):
        nodes = np.asarray([
            self._get_entity_feat_relative(agent, entity, world, 0.0)
            for entity in world.entities
        ])
        adjacency = world.cached_dist_mag.copy()
        inactive = [index for index, entity in enumerate(world.agents) if entity.status]
        adjacency[inactive, :] = 0
        adjacency[:, inactive] = 0
        # Directed goal-to-agent messages; no agent-to-goal or goal-to-goal messages.
        first_goal = len(world.agents)
        adjacency[:first_goal, first_goal:] = 0
        adjacency[first_goal:, first_goal:] = 0
        return nodes, adjacency

    def density_matching_ratio(self, world):
        """OD-weighted MIX/H-M no-traffic time ratio for the current scene."""
        travel = {}
        for aircraft_type, envelope in FLEETS.items():
            per_route = []
            for task in self.tasks:
                if self.free_flow_fn:
                    seconds = float(self.free_flow_fn(self, world, task, envelope))
                else:
                    route = world.tube_params[task.entrance]
                    seconds = (SPAWN_OFFSET + route["L"] + world.tube_params[2]["L"]) / \
                        envelope.in_units[1]
                if not np.isfinite(seconds) or seconds <= 0:
                    raise ValueError("Density matching needs positive scripted Tff for every aircraft type")
                per_route.append(seconds)
            travel[aircraft_type] = float(np.mean(per_route))
        return sum(travel.values()) / 3.0 / travel["M"]

    def launch_due(self, world):
        """ASAP release plus a FIFO, 50 m generation clearance rule."""
        blocked_entries = set()
        for task in sorted(self.tasks, key=lambda t: (t.release_time, t.id)):
            if task.id in self._departed or task.release_time > world.simulation_time + 1e-9:
                continue
            if task.entrance in blocked_entries:
                continue
            route = world.tube_params[task.entrance]
            position = route["entrance"] - SPAWN_OFFSET * route["e"] + task.lateral_offset * route["n"]
            blocked = any(not other.status and np.linalg.norm(other.state.p_pos - position) < MIN_SEPARATION
                          for other in world.agents)
            if blocked:
                blocked_entries.add(task.entrance)
                continue
            agent = world.agents[task.id]
            heading = float(np.arctan2(route["e"][1], route["e"][0]) + task.heading_offset)
            agent.state.p_pos = np.asarray(position, dtype=float)
            agent.state.reset_velocity(theta=heading, speed=agent.state.min_speed)
            agent.state.p_dist = 0.0
            agent.state.time = 0.0
            agent.t = 0.0
            agent.status = False
            agent.dormant = False
            self._departed.add(task.id)
            self.metrics.departure[task.id] = world.simulation_time
            s, _, _, _ = self._tube_coords(world, agent.state.p_pos, task.entrance)
            self.prev_proj[task.id] = s
            self.prev_goal_dist[task.id] = np.linalg.norm(agent.state.p_pos - self.landmark_poses[task.id])

    def observation(self, agent, world):
        if getattr(agent, "dormant", False):
            public = np.zeros(16, dtype=np.float32)
        else:
            public = super().observation(agent, world)
        return np.concatenate((public, np.asarray(agent.envelope.normalized, dtype=np.float32)))

    def _get_entity_feat_relative(self, agent, entity, world, fairness_param):
        public = super()._get_entity_feat_relative(agent, entity, world, fairness_param)
        # For P0/P1 the actor sees no neighbour capability. The unmasked values
        # remain in the tensor for the centralized critic; GR_Actor applies mu.
        capability = (np.asarray(entity.envelope.normalized, dtype=np.float32)
                      if hasattr(entity, "envelope") else np.zeros(5, dtype=np.float32))
        return np.concatenate((public[:-1], capability, np.array([0.0, public[-1]], dtype=np.float32)))

    def reward(self, agent, world):
        if getattr(agent, "dormant", False):
            return 0.0
        reward = super().reward(agent, world)
        if agent.status and agent.id not in self.metrics.arrival:
            self.metrics.arrival[agent.id] = world.simulation_time
        return reward

    def info_callback(self, agent, world):
        info = super().info_callback(agent, world)
        task = self.task_by_id[agent.id]
        arrival = self.metrics.arrival.get(agent.id)
        info["p0p1_active"] = float(not agent.status and not agent.dormant)
        info["p0p1_departed"] = float(agent.id in self._departed)
        info["p0p1_arrived"] = float(arrival is not None)
        info["p0p1_on_time"] = float(arrival is not None and
                                      arrival - task.release_time <=
                                      1.5 * self.metrics.free_flow_seconds[agent.id])
        info["p0p1_I_percent"] = (100.0 * self.metrics.separation_loss_substeps /
                                   self.metrics.active_agent_substeps
                                   if self.metrics.active_agent_substeps else 0.0)
        info["p0p1_N_col"] = float(self.metrics.collision_events)
        info["p0p1_tmax_truncated"] = float(world.simulation_time >= self.tmax and
                                            len(self.metrics.arrival) < len(self.tasks))
        return info
