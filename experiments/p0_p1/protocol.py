"""Frozen experiment constants and common-random-number traffic generation.

All distances are in experiment units (1 u = 328.08 m); time is seconds.
This module has no ML dependency so scenario files and analysis scripts use the
same definitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib import import_module
from importlib.util import find_spec
import json
from math import ceil
from pathlib import Path
import subprocess
from typing import Literal

import numpy as np


METRES_PER_UNIT = 328.08
BACKBONE_COMMIT = "45464195bd53e65bd461d769881279ff8a09c2eb"
MIN_SEPARATION = 50.0 / METRES_PER_UNIT
COLLISION_DISTANCE = 5.0 / METRES_PER_UNIT
COORDINATION_RADIUS = 500.0 / METRES_PER_UNIT
CORRIDOR_WIDTH = 66.0 / METRES_PER_UNIT
GOAL_RADIUS = 115.0 / METRES_PER_UNIT
SPAWN_OFFSET = 300.0 / METRES_PER_UNIT
POLICY_DT = 1.0
PHYSICS_DT = 0.1
TRAIN_BOTTLENECK_RANGE = (0.8, 12.0)


@dataclass(frozen=True)
class Envelope:
    v_min: float
    v_max: float
    a_min: float
    a_max: float
    omega_max: float

    @property
    def normalized(self) -> tuple[float, ...]:
        return tuple(x / abs(y) for x, y in zip(asdict(self).values(), asdict(FLEETS["S"]).values()))

    @property
    def in_units(self) -> tuple[float, ...]:
        return (self.v_min / METRES_PER_UNIT, self.v_max / METRES_PER_UNIT,
                self.a_min / METRES_PER_UNIT, self.a_max / METRES_PER_UNIT,
                self.omega_max)

    def primitive(self, action_index: int) -> tuple[float, float]:
        if not 0 <= action_index < 25:
            raise ValueError("The MIT actor has 25 discrete actions")
        turn_index, accel_index = divmod(action_index, 5)
        turn = (turn_index - 2) * self.omega_max / 2.0
        accel = (accel_index - 2) * self.a_max / (2.0 * METRES_PER_UNIT)
        return turn, accel


FLEETS = {
    "S": Envelope(5, 24, -0.75, 0.75, 0.107),
    "M": Envelope(5, 18, -0.5, 0.5, 0.075),
    "L": Envelope(5, 12, -0.25, 0.25, 0.043),
}


@dataclass(frozen=True)
class Task:
    id: int
    entrance: int
    exit: int
    aircraft_type: str
    release_time: float
    lateral_offset: float
    heading_offset: float


def independent_streams(seed: int) -> dict[str, np.random.Generator]:
    """A change in fleet cannot change releases, OD, or spawn perturbations."""
    names = ("release", "od", "type", "operator", "spawn")
    return {name: np.random.default_rng(child)
            for name, child in zip(names, np.random.SeedSequence(seed).spawn(len(names)))}


def headway_for_spacing(spacing_ratio: float, bottleneck_entries: int = 2) -> float:
    if spacing_ratio <= 0 or bottleneck_entries < 1:
        raise ValueError("spacing and entry count must be positive")
    return spacing_ratio * MIN_SEPARATION * bottleneck_entries / (18.0 / METRES_PER_UNIT)


def make_tasks(seed: int, fleet: Literal["H-M", "MIX"],
               entries: int, per_entry: int, headway: float,
               exits: int = 1) -> list[Task]:
    if entries < 1 or per_entry < 1 or exits < 1 or headway <= 0:
        raise ValueError("entries, tasks, exits and headway must be positive")
    if fleet not in ("H-M", "MIX"):
        raise ValueError(f"Unknown fleet: {fleet}")
    rng = independent_streams(seed)
    tasks = []
    for entry in range(entries):
        release = rng["release"].uniform(0, headway)
        for k in range(per_entry):
            if k:
                release += headway * rng["release"].uniform(0.8, 1.2)
            aircraft_type = "M" if fleet == "H-M" else ("S", "M", "L")[rng["type"].integers(0, 3)]
            task = Task(
                id=entry + entries * k,
                entrance=entry,
                exit=int(rng["od"].integers(0, exits)),
                aircraft_type=aircraft_type,
                release_time=float(release),
                lateral_offset=float(rng["spawn"].uniform(-CORRIDOR_WIDTH, CORRIDOR_WIDTH)),
                heading_offset=float(rng["spawn"].uniform(-np.pi / 6, np.pi / 6)),
            )
            tasks.append(task)
    return sorted(tasks, key=lambda t: t.id)


def training_headway(seed: int, bottleneck_entries: int = 2) -> float:
    lo, hi = TRAIN_BOTTLENECK_RANGE
    ratio = np.exp(np.random.default_rng(seed).uniform(np.log(lo), np.log(hi)))
    return headway_for_spacing(float(ratio), bottleneck_entries)


def configuration_hash(config: dict) -> str:
    return sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def backbone_commit(root: Path) -> str:
    """Report the MIT base, allowing experiment commits descended from it.

    Source archives have no Git metadata; their contents are still covered by
    source_fingerprint. Git checkouts must retain the pinned upstream history.
    """
    if not (root / ".git").exists():
        return BACKBONE_COMMIT
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BACKBONE_COMMIT, "HEAD"],
        cwd=root, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise ValueError(
            "Pinned MIT backbone is missing from this checkout's history. "
            "Use a full clone (or fetch the upstream history for a shallow clone)."
        )
    return BACKBONE_COMMIT


def source_fingerprint(root: Path) -> str:
    """Fingerprint local edits in addition to the pinned upstream Git commit."""
    files = [
        "multiagent/MPE_env.py", "multiagent/environment.py",
        "multiagent/custom_scenarios/p0p1_merge_graph.py",
        "onpolicy/algorithms/graph_actor_critic.py", "onpolicy/config.py",
        "onpolicy/envs/env_wrappers.py", "onpolicy/runner/shared/graph_mpe_runner.py",
    ]
    files.extend(str(path.relative_to(root))
                 for path in sorted((root / "experiments" / "p0_p1").glob("*.py")))
    digest = sha256()
    for relative in sorted(files):
        digest.update(relative.encode())
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def map_module_hash(module_name: str | None) -> str | None:
    if module_name is None:
        return None
    spec = find_spec(module_name)
    if spec is None or spec.origin is None:
        raise ValueError(f"Map module not importable: {module_name}")
    module_path = Path(spec.origin)
    digest = sha256(module_path.read_bytes())
    module = import_module(module_name)
    for entry in getattr(module, "MAP_DATA_FILES", ()):
        data_path = Path(entry)
        if not data_path.is_absolute():
            data_path = module_path.parent / data_path
        digest.update(str(entry).encode())
        digest.update(data_path.read_bytes())
    return digest.hexdigest()


def evaluation_grid(points: int = 10) -> list[float]:
    if points < 10:
        raise ValueError("P0 requires at least 10 grid points")
    lo, hi = TRAIN_BOTTLENECK_RANGE
    return [headway_for_spacing(float(r)) for r in np.geomspace(hi, lo, points)]


def provisional_episode_length(tasks: list[Task], longest_free_flow: float) -> int:
    """For a frozen map, recompute this with its scripted free-flow travel time."""
    return ceil(max(t.release_time for t in tasks) + 1.5 * longest_free_flow + 60)


def validate_team_map(metadata: dict) -> None:
    """Check the five geometry constraints before a custom map is frozen."""
    required = {"entrance_count", "merge_count", "min_bend_radius_m",
                "min_approach_m", "corridor_width_m", "spawn_length_m"}
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"Map metadata missing: {sorted(missing)}")
    if metadata["entrance_count"] < 2 or metadata["merge_count"] < 1:
        raise ValueError("Map needs multiple entrances and a merge")
    if metadata["min_bend_radius_m"] < 280:
        raise ValueError("Minimum bend radius must be at least 280 m")
    if metadata["min_approach_m"] < 417:
        raise ValueError("Approach to merge must be at least 417 m")
    if not np.isclose(metadata["corridor_width_m"], 66):
        raise ValueError("Corridor width must be 66 m")
    if metadata["spawn_length_m"] < 300:
        raise ValueError("Spawn/acceleration segment must be at least 300 m")
