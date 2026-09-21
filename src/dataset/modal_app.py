"""Collect the corpus on Modal.

Everything runs remotely: the dev Mac has 7.7 GB free, which is not enough for
the dataset, the pretrained checkpoints, or the dreamer4 corpus. The Mac holds
only this repo (kilobytes); frames live in a Modal Volume.

    modal run src/dataset/modal_app.py --split train --shards 130 --episodes 200
    modal run src/dataset/modal_app.py::probe            # dense coverage of the rare ledge

Shards are small on purpose. A shard holds its frames uncompressed in RAM
before np.savez_compressed, and 128x128x3 is 49 KB/frame raw, so 200 episodes
(~15k frames) already peaks around 1.5 GB.
"""

from __future__ import annotations

import json
import sys as _sys
from pathlib import Path

import modal

_DIR = Path(__file__).resolve().parent

app = modal.App("pixel-world-data")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.*", "pygame==2.6.1")
    .env({"SDL_VIDEODRIVER": "dummy"})
    .add_local_dir(str(_DIR / "env"), "/root/env")
    .add_local_file(str(_DIR / "collect.py"), "/root/collect.py")
)

vol = modal.Volume.from_name("pixel-world-data", create_if_missing=True)
DATA = "/data"

# Seed blocks live in env/splits.py so scripts/tests/check_splits.py verifies the same
# numbers the collector actually uses.
_sys.path.insert(0, str(_DIR))
from env.splits import SEED_BASE  # noqa: E402


@app.function(image=image, volumes={DATA: vol}, cpu=1.0, memory=4096, timeout=3600)
def collect(split: str, shard: int, episodes: int, goal: int | None = None) -> dict:
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root")
    from collect import collect_shard

    out = Path(DATA) / split / f"shard_{shard:04d}.npz"
    meta = collect_shard(out, episodes, SEED_BASE[split] + shard, goal=goal, verbose=False)
    vol.commit()
    return meta


def _merge(metas: list[dict]) -> dict:
    total = {"shards": len(metas), "episodes": 0, "frames": 0, "bytes": 0}
    goal_eps, goal_frames = {}, {}
    for m in metas:
        total["episodes"] += m["episodes"]
        total["frames"] += m["frames"]
        total["bytes"] += m["bytes"]
        for k, v in m["goal_episodes"].items():
            goal_eps[int(k)] = goal_eps.get(int(k), 0) + v
        for k, v in m["goal_frames"].items():
            goal_frames[int(k)] = goal_frames.get(int(k), 0) + v
    total["gb"] = round(total["bytes"] / 1e9, 3)
    total["bytes_per_frame"] = round(total["bytes"] / max(total["frames"], 1), 1)
    fr = sum(goal_frames.values()) or 1
    ep = sum(goal_eps.values()) or 1
    total["task_episode_share"] = {k: round(v / ep, 4) for k, v in sorted(goal_eps.items())}
    total["task_frame_share"] = {k: round(v / fr, 4) for k, v in sorted(goal_frames.items())}
    return total


@app.function(image=image, volumes={DATA: vol}, timeout=600)
def build_manifest(split: str) -> dict:
    """Scan every shard already in the volume, not just this run's.

    Topping a split up must not silently shrink its manifest to the shards
    the latest run happened to produce.
    """
    from pathlib import Path

    vol.reload()
    d = Path(DATA) / split
    metas = [json.loads(f.read_text()) for f in sorted(d.glob("shard_*.json"))]
    summary = _merge(metas)
    (d / "manifest.json").write_text(json.dumps(summary, indent=2))
    vol.commit()
    return summary


@app.local_entrypoint()
def main(split: str = "train", shards: int = 150, episodes: int = 200,
         goal: int = -1, start: int = 0):
    """Collect shards [start, shards). `start` lets a split be topped up
    without recomputing the shards it already has."""
    g = None if goal < 0 else goal
    args = [(split, i, episodes, g) for i in range(start, shards)]
    if args:
        list(collect.starmap(args))
    print(json.dumps(build_manifest.remote(split), indent=2))


@app.local_entrypoint()
def probe():
    """Dense coverage of the rare ledge, held out from training.

    We need this precisely because the training corpus does not have it: the
    hallucination predictors have to be scored against ground truth in the
    region the model is expected to hallucinate in.
    """
    list(collect.starmap([("probe_ledge", i, 150, 3) for i in range(16)]))
    print(json.dumps(build_manifest.remote("probe_ledge"), indent=2))
