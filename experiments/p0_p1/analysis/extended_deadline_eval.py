"""Re-evaluate a frozen P0/P1 actor under a longer *analysis-only* horizon.

The original evaluator ends each scene at last release + 1.5 * max(Tff) + 60 s.
For a 2.0x or 2.5x deadline sensitivity check, that can stop a flight before
the later deadline. This wrapper extends T_max without changing the policy,
traffic, nominal free-flow estimate, or the original 1.5x success definition.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

from experiments.p0_p1 import evaluate
from experiments.p0_p1.metrics import EpisodeMetrics
from experiments.p0_p1.protocol import configuration_hash
from multiagent.custom_scenarios.p0p1_merge_graph import Scenario


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--fleet", choices=("H-M", "MIX"), required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--load", choices=("low", "med", "high"), required=True)
    parser.add_argument("--scene-seed", type=int, required=True)
    parser.add_argument("--headway", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon-multiplier", type=float, default=2.5)
    options = parser.parse_args()
    if options.horizon_multiplier < 2.5:
        parser.error("Horizon must cover every requested 2.5x deadline")
    if options.output.exists():
        parser.error(f"Refusing to overwrite {options.output}")

    original_reset = Scenario.reset_world
    original_summarize = EpisodeMetrics.summarize

    def reset_with_longer_horizon(self, world, num_current_episode=0):
        original_reset(self, world, num_current_episode)
        self.tmax = (max(task.release_time for task in self.tasks)
                     + options.horizon_multiplier *
                     max(self.metrics.free_flow_seconds.values()) + 60.0)

    def summarize_with_deadline_counts(self, *args, **kwargs):
        result = original_summarize(self, *args, **kwargs)
        ratios = [
            (self.arrival[task.id] - task.release_time) /
            self.free_flow_seconds[task.id]
            for task in self.tasks if task.id in self.arrival
        ]
        result["deadline_counts"] = {
            str(multiplier): sum(ratio <= multiplier for ratio in ratios)
            for multiplier in (1.0, 1.5, 2.0, 2.5)
        }
        result["deadline_counts"].update({
            "tasks": len(self.tasks),
            "arrived_with_extended_horizon": len(ratios),
        })
        return result

    Scenario.reset_world = reset_with_longer_horizon
    EpisodeMetrics.summarize = summarize_with_deadline_counts
    try:
        sys.argv = [
            "evaluate", "--actor", str(options.actor), "--fleet", options.fleet,
            "--train-seed", str(options.train_seed), "--split", "smoke",
            "--load", options.load, "--scene-seed", str(options.scene_seed),
            "--headway", str(options.headway), "--agents", "20",
            "--episode-length", "1200", "--output", str(options.output),
        ]
        evaluate.main()
    finally:
        Scenario.reset_world = original_reset
        EpisodeMetrics.summarize = original_summarize

    result = json.loads(options.output.read_text())
    counts = result["metrics"]["deadline_counts"]
    if counts["1.5"] != round(result["metrics"]["S_task_percent"] * counts["tasks"] / 100):
        raise ValueError("1.5x count disagrees with the unchanged S_task metric")
    result["analysis_horizon_multiplier"] = options.horizon_multiplier
    result["analysis_script_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    result["analysis_note"] = (
        "Deadline sensitivity only; MIT-map Tff is still route distance / vmax. "
        "T_max was extended so 2.5x deadlines are observable."
    )
    result["configuration_hash"] = configuration_hash(
        {key: value for key, value in result.items()
         if key not in ("metrics", "configuration_hash")}
    )
    options.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(options.load, options.scene_seed, counts, flush=True)


if __name__ == "__main__":
    main()
