"""P0 load selection and P1 seed-level inference.

Inputs here are validation/test summaries, never the same scenarios for both.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class LoadPoint:
    headway: float
    task_success_percent: float
    throughput: float
    entries: int = 2

    @property
    def arrival_rate(self) -> float:
        return self.entries / self.headway


def select_loads(points: Sequence[LoadPoint]) -> dict[str, float]:
    """Return headways for the three fixed loads in equation (13)."""
    ordered = sorted(points, key=lambda x: x.arrival_rate)
    if len(ordered) < 10:
        raise ValueError("P0 needs at least ten validation grid points")
    knee = next((p for p in ordered if p.task_success_percent < 95), None)
    high = next((p for p in ordered if p.throughput >= 0.95 * max(x.throughput for x in ordered)), None)
    if knee is None or high is None or knee in (ordered[0], ordered[-1]) or high in (ordered[0], ordered[-1]):
        raise ValueError("Knee or plateau is absent or at the edge: extend and rescan the grid")
    medium_rate = knee.arrival_rate
    next_rate = next(p.arrival_rate for p in ordered if p.arrival_rate > medium_rate)
    high_rate = max(high.arrival_rate, next_rate)
    low_rate = 0.5 * medium_rate
    entries = np.mean([p.arrival_rate * p.headway for p in ordered])
    if not np.allclose([p.arrival_rate * p.headway for p in ordered], entries):
        raise ValueError("Inconsistent number of entrances across validation points")
    return {"low": entries / low_rate, "med": entries / medium_rate, "high": entries / high_rate}


def paired_seed_differences(condition: Mapping[int, Mapping[int, float]],
                            reference: Mapping[int, Mapping[int, float]]) -> np.ndarray:
    """Average matched scenario differences within each training seed."""
    if set(condition) != set(reference) or len(condition) < 2:
        raise ValueError("The conditions need the same training seeds")
    result = []
    for seed in sorted(condition):
        if set(condition[seed]) != set(reference[seed]):
            raise ValueError("Each seed needs exactly the same test scenarios")
        result.append(np.mean([condition[seed][scene] - reference[seed][scene]
                               for scene in sorted(condition[seed])]))
    return np.asarray(result)


def gate1_interaction(e3_by_load: Mapping[str, np.ndarray],
                      e1_by_load: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """C(D) = [E3-E1](D) - [E3-E1](D_low), paired by seed."""
    baseline = e3_by_load["low"] - e1_by_load["low"]
    return {load: (e3_by_load[load] - e1_by_load[load]) - baseline
            for load in ("med", "high")}


def paired_interval(samples: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    values = np.asarray(samples, dtype=float)
    if len(values) < 2:
        raise ValueError("At least two independent training seeds are needed")
    mean = float(np.mean(values))
    sem = float(stats.sem(values))
    radius = float(stats.t.ppf((1 + confidence) / 2, len(values) - 1) * sem)
    return mean - radius, mean + radius


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.zeros_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[index]))
        adjusted[index] = running
    return adjusted


def gate_decision(family: Sequence[np.ndarray], minimum_effect: Sequence[float]) -> dict:
    """Positive values mean a coordination cost; TOST is a separate outcome."""
    if len(family) != len(minimum_effect):
        raise ValueError("One frozen delta_min is needed for each comparison")
    if any(delta <= 0 for delta in minimum_effect):
        raise ValueError("Every frozen minimum effect must be positive")
    p = []
    for samples in family:
        values = np.asarray(samples, dtype=float)
        if len(values) < 2 or not np.all(np.isfinite(values)):
            raise ValueError("Every comparison needs at least two finite seed effects")
        if np.all(values == values[0]):
            p.append(1.0 if values[0] == 0 else 0.0)
        else:
            p.append(float(stats.ttest_1samp(values, 0).pvalue))
    adjusted = holm_adjust(p)
    means = [float(np.mean(x)) for x in family]
    order = np.argsort(p)
    adjusted_intervals = [None] * len(family)
    for rank, index in enumerate(order):
        adjusted_intervals[index] = paired_interval(
            family[index], 1.0 - 0.05 / (len(family) - rank))
    passed = any(mean >= delta and p_adj < 0.05 and interval[0] > 0
                 for mean, delta, p_adj, interval in zip(
                     means, minimum_effect, adjusted, adjusted_intervals))
    equivalent = all(-delta < paired_interval(x, 0.90)[0] and paired_interval(x, 0.90)[1] < delta
                     for x, delta in zip(family, minimum_effect))
    return {"decision": "pass" if passed else "no_effect" if equivalent else "uncertain",
            "means": means, "holm_p": adjusted.tolist(),
            "holm_95ci": adjusted_intervals}
