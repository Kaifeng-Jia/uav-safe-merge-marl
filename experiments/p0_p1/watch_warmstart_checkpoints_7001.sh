#!/usr/bin/env bash
# Keep step-labelled copies of MIT's overwritten actor.pt/critic.pt saves.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
python_bin="${P0P1_PYTHON:-python}"
"$python_bin" - <<'PY'
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import time

root = Path.cwd()
run_dir = root / 'experiments/p0_p1/runs/provisional/MIX/7001_warmstart_5m'
manifest_path = run_dir / 'manifest.json'
checkpoint_root = (root / 'onpolicy/results/GraphMPE/p0p1_merge_graph/rmappo'
                   / 'p0p1_provisional_MIX_7001_warmstart_5m')
archive = run_dir / 'checkpoints'
archive.mkdir(parents=True, exist_ok=True)
known = {json.loads(p.read_text())['actor_sha256']
         for p in archive.glob('*/metadata.json')}
while True:
    actors = list(checkpoint_root.rglob('actor.pt'))
    if len(actors) == 1:
        actor = actors[0]
        critic = actor.with_name('critic.pt')
        if critic.is_file():
            initial = (actor.stat().st_mtime_ns, actor.stat().st_size,
                       critic.stat().st_mtime_ns, critic.stat().st_size)
            time.sleep(2)
            stable = (actor.stat().st_mtime_ns, actor.stat().st_size,
                      critic.stat().st_mtime_ns, critic.stat().st_size)
            if initial == stable:
                actor_bytes = actor.read_bytes()
                digest = sha256(actor_bytes).hexdigest()
                if digest not in known:
                    log = (run_dir / 'train.log').read_text(errors='replace')
                    steps = [int(x) for x in re.findall(r'Total timesteps:\s*(\d+)', log)]
                    observed_step = steps[-1] if steps else None
                    folder = archive / f'{len(known):03d}_observed_step_{observed_step or 0}'
                    folder.mkdir()
                    (folder / 'actor.pt').write_bytes(actor_bytes)
                    shutil.copy2(critic, folder / 'critic.pt')
                    metadata = {
                        'observed_log_step': observed_step,
                        'copied_utc': datetime.now(timezone.utc).isoformat(),
                        'actor_sha256': digest,
                        'critic_sha256': sha256((folder / 'critic.pt').read_bytes()).hexdigest(),
                        'source_actor_path': str(actor.relative_to(root)),
                    }
                    (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
                    known.add(digest)
                    print(json.dumps(metadata), flush=True)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('exit_code') is not None:
            print(f"training_exit_code={manifest['exit_code']} snapshots={len(known)}", flush=True)
            break
    time.sleep(30)
PY
