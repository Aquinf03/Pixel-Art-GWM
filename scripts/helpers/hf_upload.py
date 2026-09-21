"""Publish the corpus to the Hugging Face Hub, straight from the Modal volume.

Uploading from the volume avoids round-tripping ~0.9 GB through a laptop with
7.6 GB free, and Modal's egress is considerably faster than a domestic link.
"""

from __future__ import annotations

import modal

app = modal.App("hf-upload")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub>=0.26", "numpy==2.*")
)
vol = modal.Volume.from_name("pixel-world-data", create_if_missing=True)
DATA = "/data"

CARD = """---
license: mit
task_categories:
  - reinforcement-learning
  - video-classification
tags:
  - world-models
  - pixel-art
  - synthetic
  - action-conditioned
size_categories:
  - 1M<n<10M
---

# Small Worlds: a generated pixel-platformer corpus

**2,025,385 frames** of a single controllable agent in a deterministic 2D
platformer, with per-frame actions, rewards, episode boundaries and task labels.
Nothing here was scraped: every frame was rendered by a simulator written for the
purpose, which makes the coverage structure a *controlled variable*.

Built for research on world models and on hallucination in generative dynamics
models. Accompanying report: *Small Worlds: Subject-Scale Failure in
Reconstruction-Trained World-Model Tokenizers*.

## Why this corpus exists

Two properties are hard to obtain from recorded gameplay and are exact here:

1. **A known finite palette.** The renderer emits exactly **21 colours**, so
   reconstruction can be scored as exact per-pixel agreement and *restricted to
   any semantic subset* — the agent's own colours, for instance. This is what
   makes the paper's central measurement possible.
2. **Coverage as a dial.** Each episode targets one coin drawn from a skewed
   distribution and ends when it is collected, so a goal is a *task* and the
   uppermost platform is visited in ~3% of episodes by construction.

The agent occupies a measured **1.30%** of frame pixels.

## Splits

| split | episodes | frames | size | purpose |
|---|---|---|---|---|
| `train` | 35,000 | 2,025,385 | 0.75 GB | training |
| `val` | 2,400 | 136,620 | 0.05 GB | evaluation |
| `test` | 2,400 | 139,157 | 0.05 GB | held out entirely |
| `probe_ledge` | 2,400 | 364,684 | 0.13 GB | 100% low-coverage region |

Seed blocks are widely separated and verified disjoint.

## Format

One `.npz` per shard, with a `.json` of per-shard statistics beside it.

| field | shape | dtype | notes |
|---|---|---|---|
| `frames` | (N, 128, 128, 3) | uint8 | native resolution, no resampling |
| `actions` | (N,) | uint8 | 0–5; 255 marks a terminal step |
| `action_vecs` | (N, 16) | float32 | zero-padded continuous encoding |
| `action_mask` | (16,) | float32 | 3 valid dimensions |
| `rewards` | (N,) | float32 | coin +1, crate +2, door +5 |
| `ep_id` | (N,) | int32 | episode boundaries |
| `goal` | (N,) | uint8 | task label; enables per-task resampling |
| `terminal` | (N,) | bool | |

Frames are stored once per episode; `frames[i+1]` is the successor within an
episode and `ep_id` marks boundaries. Flat pixel art deflates well — **368
bytes/frame** against 49,152 raw.

```python
import numpy as np
from huggingface_hub import hf_hub_download

p = hf_hub_download("REPO_ID", "train/shard_0000.npz", repo_type="dataset")
d = np.load(p)
frames, actions, rewards = d["frames"], d["action_vecs"], d["rewards"]
```

## Coverage structure

Episode share and frame share are deliberately mismatched, reproducing the
heavy-tailed structure of larger corpora:

| task | episode share | frame share | ratio |
|---|---|---|---|
| ground | 31.9% | 5.3% | 0.17 |
| P1 | 31.1% | 37.5% | 1.21 |
| P2 | 31.0% | 41.6% | 1.34 |
| **P3 (rare)** | **6.0%** | 15.5% | 2.57 |

## Behaviour policies

| policy | share |
|---|---|
| scripted (routes over a physics-derived hop graph) | 40% |
| scripted + 20% random | 25% |
| scripted + 50% random | 15% |
| uniform random | 15% |
| explorer (biased to the rare platform) | 5% |

## Measured properties

- **Redundancy**: 25.4% of frames unique (mean 3.94 repeats) → ~515k distinct frames
- **Intrinsic dimensionality**: a frame is a deterministic function of ~26 bits of state
- **Palette**: exactly 21 colours; 9 belong to the agent
- **Versioning**: every shard carries `action_space_version` and `art_version`

## Licence

MIT.
"""


@app.function(image=image, volumes={DATA: vol}, timeout=5400,
              secrets=[modal.Secret.from_name("hf-token")])
def push(repo_id: str, splits: str = "train,val,test,probe_ledge",
         private: bool = False) -> dict:
    import glob, os
    from huggingface_hub import HfApi

    vol.reload()
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    card = CARD.replace("REPO_ID", repo_id)
    open("/tmp/README.md", "w").write(card)
    api.upload_file(path_or_fileobj="/tmp/README.md", path_in_repo="README.md",
                    repo_id=repo_id, repo_type="dataset")

    counts = {}
    for split in splits.split(","):
        files = sorted(glob.glob(f"{DATA}/{split}/shard_*.npz")) \
            + sorted(glob.glob(f"{DATA}/{split}/shard_*.json")) \
            + sorted(glob.glob(f"{DATA}/{split}/manifest.json"))
        if not files:
            counts[split] = 0
            continue
        api.upload_folder(folder_path=f"{DATA}/{split}", repo_id=repo_id,
                          repo_type="dataset", path_in_repo=split,
                          allow_patterns=["*.npz", "*.json"])
        counts[split] = len(files)
        print(f"uploaded {split}: {len(files)} files", flush=True)
    return {"repo": f"https://huggingface.co/datasets/{repo_id}", "files": counts}


@app.local_entrypoint()
def main(repo_id: str = "maxmill/small-worlds-pixel-platformer",
         splits: str = "train,val,test,probe_ledge", private: bool = False):
    import json
    print(json.dumps(push.remote(repo_id, splits, private), indent=2))
