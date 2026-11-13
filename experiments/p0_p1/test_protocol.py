import unittest

import numpy as np

from experiments.p0_p1.calibration import gate1_interaction, paired_seed_differences
from experiments.p0_p1.metrics import EpisodeMetrics, segment_closest_distance
from experiments.p0_p1.protocol import (
    COLLISION_DISTANCE, FLEETS, MIN_SEPARATION, make_tasks, validate_team_map,
)
from experiments.p0_p1.report import gate1


class ProtocolTests(unittest.TestCase):
    def test_mean_matched_fleet_and_actions(self):
        for component in ("v_min", "v_max", "a_min", "a_max", "omega_max"):
            self.assertAlmostEqual(getattr(FLEETS["M"], component),
                                   (getattr(FLEETS["S"], component) + getattr(FLEETS["L"], component)) / 2)
        for fleet in FLEETS.values():
            self.assertEqual(fleet.primitive(12), (0, 0))
            self.assertAlmostEqual(fleet.primitive(0)[0], -fleet.omega_max)
            self.assertAlmostEqual(fleet.primitive(24)[0], fleet.omega_max)

    def test_pairing_preserves_demand_and_spawn(self):
        homogeneous = make_tasks(423, "H-M", 2, 10, 15)
        mixed = make_tasks(423, "MIX", 2, 10, 15)
        for a, b in zip(homogeneous, mixed):
            self.assertEqual((a.id, a.entrance, a.exit, a.release_time,
                              a.lateral_offset, a.heading_offset),
                             (b.id, b.entrance, b.exit, b.release_time,
                              b.lateral_offset, b.heading_offset))
        self.assertEqual({t.aircraft_type for t in homogeneous}, {"M"})
        self.assertGreater(len({t.aircraft_type for t in mixed}), 1)

    def test_segment_detects_crossing_between_substeps(self):
        self.assertLess(segment_closest_distance(np.array([-COLLISION_DISTANCE * 2, 0]),
                                                 np.array([COLLISION_DISTANCE * 2, 0])),
                        COLLISION_DISTANCE)
        self.assertGreater(MIN_SEPARATION, COLLISION_DISTANCE)

    def test_task_metrics_include_unflown_and_timeout(self):
        tasks = make_tasks(4, "H-M", 1, 2, 10)
        m = EpisodeMetrics(tasks, {t.id: 100 for t in tasks})
        m.departure[tasks[0].id] = tasks[0].release_time + 3
        m.arrival[tasks[0].id] = tasks[0].release_time + 90
        summary = m.summarize(10, 1)
        self.assertEqual(summary["S_task_percent"], 50)
        self.assertEqual(summary["N_unflown"], 1)
        self.assertAlmostEqual(summary["D_bar"], 0.2)

    def test_seed_is_unit_of_inference(self):
        condition = {1: {1: 3, 2: 5}, 2: {1: 6, 2: 8}}
        reference = {1: {1: 2, 2: 2}, 2: {1: 2, 2: 4}}
        differences = paired_seed_differences(condition, reference)
        np.testing.assert_array_equal(differences, [2, 4])
        interaction = gate1_interaction(
            {"low": np.array([2, 3]), "med": np.array([5, 7]), "high": np.array([9, 10])},
            {"low": np.array([1, 2]), "med": np.array([2, 3]), "high": np.array([4, 5])},
        )
        np.testing.assert_array_equal(interaction["med"], [2, 3])

    def test_map_geometry_gate(self):
        metadata = {"entrance_count": 2, "merge_count": 1, "min_bend_radius_m": 280,
                    "min_approach_m": 500, "corridor_width_m": 66,
                    "spawn_length_m": 300}
        validate_team_map(metadata)
        with self.assertRaises(ValueError):
            validate_team_map(dict(metadata, min_bend_radius_m=200))

    def test_gate1_uses_cross_load_seed_pairing(self):
        rows = []
        for seed in (1, 2):
            for scene in (101, 102):
                for load in ("low", "med", "high"):
                    for fleet in ("H-M", "MIX"):
                        excess = {"low": 1, "med": 3, "high": 5}[load]
                        rows.append({"split": "test", "load": load, "fleet": fleet,
                                     "training_seed": seed, "scene_seed": scene,
                                     "headway": {"low": 20, "med": 10, "high": 5}[load],
                                     "metrics": {"I_percent": 2 + (excess if fleet == "MIX" else 0),
                                                 "D_bar": 0.1 + (excess / 100 if fleet == "MIX" else 0)}})
        result = gate1(rows, 0.2, 0.01, exploratory=True)
        self.assertAlmostEqual(result["effects"]["I_percent"]["med_interaction_mean"], 2)
        self.assertAlmostEqual(result["effects"]["I_percent"]["high_interaction_mean"], 4)


if __name__ == "__main__":
    unittest.main()
