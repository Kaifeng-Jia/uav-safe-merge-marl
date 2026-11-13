"""Archive step-labelled E1 checkpoints that MIT otherwise overwrites."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import time


ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "experiments/p0_p1/runs/provisional/H-M"
MODELS = ROOT / "onpolicy/results/GraphMPE/p0p1_merge_graph/rmappo"
STAGES = (
    ("7001", "p0p1_provisional_HM_7001"),
    ("7001_warmstart_5m", "p0p1_provisional_HM_7001_warmstart_5m"),
)


def watch(run_name, experiment_name):
    run_dir = RUNS / run_name
    manifest_path = run_dir / "manifest.json"
    checkpoint_root = MODELS / experiment_name
    archive = run_dir / "checkpoints"
    while not manifest_path.exists():
        time.sleep(30)
    archive.mkdir(parents=True, exist_ok=True)
    known = {json.loads(path.read_text())["actor_sha256"]
             for path in archive.glob("*/metadata.json")}
    while True:
        actors = list(checkpoint_root.rglob("actor.pt"))
        if len(actors) == 1:
            actor = actors[0]
            critic = actor.with_name("critic.pt")
            if critic.is_file():
                before = (actor.stat().st_mtime_ns, actor.stat().st_size,
                          critic.stat().st_mtime_ns, critic.stat().st_size)
                time.sleep(2)
                after = (actor.stat().st_mtime_ns, actor.stat().st_size,
                         critic.stat().st_mtime_ns, critic.stat().st_size)
                if before == after:
                    actor_bytes = actor.read_bytes()
                    critic_bytes = critic.read_bytes()
                    if (actor.stat().st_mtime_ns, actor.stat().st_size,
                        critic.stat().st_mtime_ns, critic.stat().st_size) == after:
                        digest = sha256(actor_bytes).hexdigest()
                        if digest not in known:
                            log = (run_dir / "train.log").read_text(errors="replace")
                            steps = [int(x) for x in re.findall(r"Total timesteps:\s*(\d+)", log)]
                            observed_step = steps[-1] if steps else None
                            folder = archive / f"{len(known):03d}_observed_step_{observed_step or 0}"
                            folder.mkdir()
                            (folder / "actor.pt").write_bytes(actor_bytes)
                            (folder / "critic.pt").write_bytes(critic_bytes)
                            metadata = {
                                "observed_log_step": observed_step,
                                "copied_utc": datetime.now(timezone.utc).isoformat(),
                                "actor_sha256": digest,
                                "critic_sha256": sha256(critic_bytes).hexdigest(),
                                "source_actor_path": str(actor.relative_to(ROOT)),
                            }
                            (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
                            known.add(digest)
                            print(f"{run_name} {observed_step} {digest}", flush=True)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("exit_code") is not None:
                print(f"{run_name} exit={manifest['exit_code']} snapshots={len(known)}", flush=True)
                return
        time.sleep(30)


def main():
    for run_name, experiment_name in STAGES:
        watch(run_name, experiment_name)


if __name__ == "__main__":
    main()
