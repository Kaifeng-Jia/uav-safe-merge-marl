import unittest

import numpy as np
import torch

from multiagent.MPE_env import GraphMPEEnv
from onpolicy.algorithms.graph_actor_critic import GR_Actor
from onpolicy.config import get_config, graph_config
from onpolicy.scripts.train_mpe import parse_args
from experiments.p0_p1.protocol import COORDINATION_RADIUS


def args_for(fleet):
    cli = ["--scenario_name", "p0p1_merge_graph", "--p0p1_fleet", fleet,
           "--p0p1_scene_seed", "202601", "--p0p1_headway", "2",
           "--num_agents", "6", "--num_landmarks", "6", "--episode_length", "30",
           "--num_env_steps", "60", "--n_rollout_threads", "1",
           "--dynamics_type", "air_taxi", "--formation_type", "point",
           "--world_size", "5", "--graph_feat_type", "relative"]
    cli += ["--max_edge_dist", str(COORDINATION_RADIUS)]
    _, parser = parse_args(cli, get_config())
    args, _ = graph_config(cli, parser)
    return args


class IntegrationTests(unittest.TestCase):
    def test_scenario_uses_mit_map_and_paired_tasks(self):
        homogeneous = GraphMPEEnv(args_for("H-M"))
        mixed = GraphMPEEnv(args_for("MIX"))
        homogeneous.reset()
        obs, _, nodes, _ = mixed.reset()
        self.assertEqual(np.shape(obs), (6, 21))
        self.assertEqual(np.shape(nodes)[1], 9)
        self.assertEqual(np.shape(nodes)[-1], 13)
        self.assertEqual(len(mixed.world.tube_params), 3)
        self.assertEqual(args_for("MIX").max_edge_dist, COORDINATION_RADIUS)
        _, _, _, adjacency = mixed.reset()
        self.assertTrue(np.all(np.asarray(adjacency)[:, :6, 6:] == 0))
        a = homogeneous.world.scenario.tasks
        b = mixed.world.scenario.tasks
        self.assertEqual([(t.release_time, t.exit, t.lateral_offset, t.heading_offset) for t in a],
                         [(t.release_time, t.exit, t.lateral_offset, t.heading_offset) for t in b])
        for _ in range(10):
            mixed.step(np.eye(25, dtype=np.float32)[[12] * 6])
        self.assertAlmostEqual(mixed.world.simulation_time, 10.0)
        self.assertGreater(len(mixed.world.scenario.metrics.departure), 0)
        self.assertTrue(all(agent.state.speed <= agent.state.max_speed + 1e-9
                            for agent in mixed.world.agents))
        homogeneous.close()
        mixed.close()

    def test_hidden_neighbour_capability_cannot_change_actor(self):
        args = args_for("MIX")
        env = GraphMPEEnv(args)
        obs, ids, nodes, adjacency = env.reset()
        actor = GR_Actor(args, env.observation_space[0], env.node_observation_space[0],
                         env.edge_observation_space[0], env.action_space[0])
        altered = np.asarray(nodes).copy()
        altered[..., 6:11] += 7.0
        recurrent = np.zeros((6, args.recurrent_N, args.hidden_size), dtype=np.float32)
        masks = np.ones((6, 1), dtype=np.float32)
        with torch.no_grad():
            base = actor(np.asarray(obs), np.asarray(nodes), np.asarray(adjacency),
                         np.asarray(ids), recurrent, masks, deterministic=True)
            changed = actor(np.asarray(obs), altered, np.asarray(adjacency),
                            np.asarray(ids), recurrent, masks, deterministic=True)
        for left, right in zip(base, changed):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        env.close()


if __name__ == "__main__":
    unittest.main()
