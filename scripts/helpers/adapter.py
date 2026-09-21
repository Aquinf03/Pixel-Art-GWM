"""Convert our npz shards into the layout dreamer4's loader expects.

    <data_dir>/<task>.pt                     TensorDict: episode, action, reward
    <frames_dir>/<task>/<task>_shard%04d.pt  {"frames": uint8 (N,3,128,128)}

Their preprocess writes 2048 frames per shard, so we match that. Episode ids
must be unique across the whole task, not per input shard.

Actions are written RAW, not EMA-smoothed. Smoothing was only worth doing to
make our button presses resemble DMControl's continuous statistics for
transfer; we are training from scratch, so raw discrete actions carry more
per-step information. Inference must then run --action_smooth_beta 0 so the
two ends match.

    modal run scripts/helpers/adapter.py::main --split train
"""

from __future__ import annotations

import modal

app = modal.App("pixel-world-adapter")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.8.0", "numpy==1.24.4", "tensordict==0.10.0")
)

vol = modal.Volume.from_name("pixel-world-data", create_if_missing=True)
DATA = "/data"
TASK = "pixelworld-climb"
SHARD_SIZE = 2048


@app.function(image=image, volumes={DATA: vol}, cpu=2.0, memory=8192, timeout=5400)
def convert(split: str, limit: int = 0, dest: str = "") -> dict:
    import glob
    import json
    import os

    import numpy as np
    import torch
    from tensordict import TensorDict

    vol.reload()
    src = sorted(glob.glob(f"{DATA}/{split}/shard_*.npz"))
    if limit:
        src = src[:limit]
    # `dest` lets us build a small fast split (e.g. 100 shards) for sweeps.
    # A 6-way parallel sweep against the full 679 GB corpus stalled the volume
    # outright; experiments need a dataset that fits comfortably in cache.
    out = dest or split
    out_data = f"{DATA}/d4/{out}/demos"
    out_frames = f"{DATA}/d4/{out}/frames/{TASK}"
    os.makedirs(out_data, exist_ok=True)
    os.makedirs(out_frames, exist_ok=True)

    ep_offset, shard_idx, buf, n_buf = 0, 0, [], 0
    all_ep, all_act, all_rew = [], [], []

    def flush(force: bool = False):
        nonlocal buf, n_buf, shard_idx
        while n_buf >= SHARD_SIZE or (force and n_buf > 0):
            cat = torch.cat(buf, 0) if len(buf) > 1 else buf[0]
            take = min(SHARD_SIZE, cat.shape[0])
            # .clone(), NOT .contiguous(): a prefix slice is already
            # contiguous, so .contiguous() returns a VIEW and torch.save then
            # serialises the whole underlying storage. That silently made every
            # shard 654 MB instead of 96 MB -- a 6.5x bloat on top of the
            # format's own 133x, for 679 GB total.
            torch.save({"frames": cat[:take].clone()},
                       f"{out_frames}/{TASK}_shard{shard_idx:04d}.pt")
            shard_idx += 1
            rest = cat[take:]
            buf = [rest] if rest.shape[0] else []
            n_buf = rest.shape[0]
            if force and n_buf == 0:
                break

    for k, p in enumerate(src):
        d = np.load(p)
        # Check BOTH versions. n_actions alone is not enough: the v2 and v3
        # sprites share an action space, so an art change slips straight past
        # it -- which is exactly how a half-converted mix nearly got through.
        assert int(d["n_actions"]) == 6, f"{p}: stale action space"
        assert int(d["art_version"]) == 3, f"{p}: stale sprite art"
        fr = torch.from_numpy(d["frames"]).permute(0, 3, 1, 2).contiguous()  # (N,3,H,W)
        buf.append(fr)
        n_buf += fr.shape[0]
        all_ep.append(torch.from_numpy(d["ep_id"].astype(np.int64)) + ep_offset)
        all_act.append(torch.from_numpy(d["action_vecs"].astype(np.float32)))
        all_rew.append(torch.from_numpy(d["rewards"].astype(np.float32)))
        ep_offset += int(d["ep_id"].max()) + 1
        flush()
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(src)} shards, {shard_idx} frame-shards written", flush=True)
    flush(force=True)

    ep = torch.cat(all_ep); act = torch.cat(all_act); rew = torch.cat(all_rew)
    n = ep.shape[0]
    td = TensorDict({"episode": ep, "action": act, "reward": rew}, batch_size=[n])
    torch.save(td, f"{out_data}/{TASK}.pt")

    # tasks.json -- the registration path. text_embedding is omitted on purpose:
    # wm_dataset falls back to a 512-dim zero vector, and we have one task and
    # no language conditioning to express.
    meta = {TASK: {
        "embodiment": "2D pixel-art character with 6 discrete actions: idle, "
                      "left, right, jump, crouch, use",
        "instruction": "Climb the platforms and collect the goal coin",
        "action_dim": 3,
        "max_episode_steps": 200,
    }}
    with open(f"{DATA}/d4/{out}/tasks.json", "w") as f:
        json.dump(meta, f, indent=2)
    vol.commit()

    return {"split": out, "src_from": split, "src_shards": len(src), "frames": int(n),
            "frame_shards": shard_idx, "episodes": int(ep.max()) + 1,
            "action_shape": list(act.shape), "reward_sum": float(rew.sum())}


@app.function(image=image, volumes={DATA: vol}, cpu=2.0, memory=8192, timeout=1800)
def verify(split: str) -> dict:
    """Instantiate dreamer4's own loader against the converted data.

    Converting to a format is not the same as the format being accepted, and
    the loader is where a mismatch would actually surface.
    """
    import subprocess, sys, glob, torch
    subprocess.run(["git", "clone", "-q", "https://github.com/nicklashansen/dreamer4",
                    "/opt/d4"], check=True)
    sys.path.insert(0, "/opt/d4/dreamer4")
    from wm_dataset import WorldModelDataset

    base = f"{DATA}/d4/{split}"
    ds = WorldModelDataset(
        data_dirs=[f"{base}/demos"], frames_dirs=[f"{base}/frames"],
        tasks_json=f"{base}/tasks.json", seq_len=16,
    )
    item = ds[0]
    out = {"dataset_len": len(ds), "tasks": list(ds.tasks)}
    for k, v in (item.items() if hasattr(item, "items") else []):
        if torch.is_tensor(v):
            out[f"item.{k}"] = f"{tuple(v.shape)} {v.dtype}"
    return out


@app.local_entrypoint()
def main(split: str = "train", limit: int = 0, check: int = 1, dest: str = ""):
    import json
    print(json.dumps(convert.remote(split, limit, dest), indent=2))
    if check:
        try:
            print(json.dumps(verify.remote(split), indent=2))
        except Exception as e:
            print("VERIFY FAILED:", type(e).__name__, e)
