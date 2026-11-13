"""P0/P1 episode metrics; no training or plotting dependency."""

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from .protocol import COLLISION_DISTANCE, MIN_SEPARATION, Task


def segment_closest_distance(relative_start: np.ndarray, relative_end: np.ndarray) -> float:
    """Closest relative distance within one 0.1-second physics step."""
    travel = relative_end - relative_start
    denom = float(np.dot(travel, travel))
    alpha = 0.0 if denom == 0 else float(np.clip(-np.dot(relative_start, travel) / denom, 0, 1))
    return float(np.linalg.norm(relative_start + alpha * travel))


@dataclass
class EpisodeMetrics:
    tasks: Sequence[Task]
    free_flow_seconds: dict[int, float]
    active_agent_substeps: int = 0
    separation_loss_substeps: int = 0
    collision_events: int = 0
    _colliding: set[tuple[int, int]] = field(default_factory=set)
    departure: dict[int, float] = field(default_factory=dict)
    arrival: dict[int, float] = field(default_factory=dict)

    def sample(self, before: dict[int, np.ndarray], after: dict[int, np.ndarray]) -> None:
        active = sorted(set(before) & set(after))
        self.active_agent_substeps += len(active)
        violations = set()
        colliding_now = set()
        for index, i in enumerate(active):
            for j in active[index + 1:]:
                separation = segment_closest_distance(before[i] - before[j], after[i] - after[j])
                if np.linalg.norm(after[i] - after[j]) < MIN_SEPARATION:
                    violations.update((i, j))
                if separation <= COLLISION_DISTANCE:
                    colliding_now.add((i, j))
        self.separation_loss_substeps += len(violations)
        self.collision_events += len(colliding_now - self._colliding)
        self._colliding = colliding_now

    def summarize(self, headway: float, entries: int) -> dict[str, float]:
        if any(t.id not in self.free_flow_seconds or self.free_flow_seconds[t.id] <= 0 for t in self.tasks):
            raise ValueError("Every task needs a positive scripted free-flow time")
        deadline_success = 0
        delays = []
        for task in self.tasks:
            free = self.free_flow_seconds[task.id]
            travel = self.arrival.get(task.id, math.inf) - task.release_time
            deadline_success += travel <= 1.5 * free
            delays.append(min(travel, 1.5 * free) / free - 1)
        total = len(self.tasks)
        departed = sum(task.id in self.departure for task in self.tasks)
        success = 100 * deadline_success / total
        waits = [self.departure[t.id] - t.release_time for t in self.tasks if t.id in self.departure]
        return {
            "I_percent": 100 * self.separation_loss_substeps / self.active_agent_substeps
            if self.active_agent_substeps else 0.0,
            "D_bar": float(np.mean(delays)),
            "S_task_percent": success,
            "S_percent": 100 * deadline_success / departed if departed else 0.0,
            "Q": entries / headway * success / 100,
            "N_col": self.collision_events,
            "N_unflown": total - departed,
            "T_wait_mean": float(np.mean(waits)) if waits else math.nan,
            "T_wait_p95": float(np.percentile(waits, 95)) if waits else math.nan,
        }
