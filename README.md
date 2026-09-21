# Small Worlds

A complete action-conditioned world model of a 2D pixel-art character, trained from
scratch on self-generated data. 4.4M parameters, 17.7 MB, ~100 fps on a laptop GPU.

**Report:** [`docs.md`](docs.md)
**Dataset:** [`Aquinlabs/small-worlds-pixel-platformer`](https://huggingface.co/datasets/Aquinlabs/small-worlds-pixel-platformer)
**Model:** [`Aquinlabs/pixel-art-gwm`](https://huggingface.co/Aquinlabs/pixel-art-gwm)

## The finding

Reconstruction objectives averaged over a frame are not neutral about *what* they
preserve. When the controllable agent occupies 1.3% of the observation, such an
objective can be reduced by discarding it — and standard metrics improve while it
happens.

| tokenizer | whole-frame accuracy | **agent-only accuracy** |
|---|---|---|
| adapted general-purpose, step 1,000 | 0.4245 | 0.0151 |
| adapted general-purpose, step 3,000 | 0.7656 | **0.0010** ↓ |
| **this work** | **0.9998** | **0.9994** |

A reconstruction containing **no agent at all** scores 0.987 whole-frame.

Two changes fix it: **spatially structured latents** (the agent occupies specific
cells and cannot be marginalised) and **per-pixel classification over the known
21-colour palette** (cross-entropy cannot blur; class weights can express relevance).

## Results

| | |
|---|---|
| Tokenizer | 2,942,173 params · 0.9994 agent accuracy · 24× compression |
| Dynamics | 1,459,080 params · 0.885 rollout agent accuracy |
| Drift over 8 imagined frames | 0.999 → 0.995 (none measurable) |
| Action fidelity | left −16 px, right +35 px, idle −0.00 px |
| Speed | 101 fps M2 GPU · 26 fps M2 CPU · 93 fps A100 |
| Training compute | 2.3 GPU-hours |

## Layout

```
src/dataset/        environment and corpus collection
src/algo/           tokenizer, dynamics, Modal training, live play, aq method
scripts/tests/      split checks, probes, coverage, action measurements
scripts/helpers/    adapters, uploads, human play, shell wrappers
assets/             images: character, world, recon
docs.md             combined report
artifacts/          checkpoints, metrics, evals
```

## Reproducing

```bash
modal run src/dataset/modal_app.py::main --split train --shards 175 --episodes 200   # ~2 min
modal run src/dataset/modal_app.py::main --split val   --shards 12  --episodes 200
modal run src/algo/pal_modal.py::main --run tok --steps 1500 --char-weight 5.0    # ~5 min
modal run src/algo/pal_modal.py::dyn  --run dyn --steps 26000                     # ~40 min
modal run src/algo/pal_modal.py::check --run tok
```

Play locally with the released checkpoints:

```bash
python src/algo/play_local.py --tok artifacts/tokenizer.pt --dyn artifacts/dynamics.pt
```

**Model:** [`Aquinlabs/pixel-art-gwm`](https://huggingface.co/Aquinlabs/pixel-art-gwm)

## Dataset

2,025,385 frames across 35,000 episodes, generated (not scraped).

**Hugging Face:** [`Aquinlabs/small-worlds-pixel-platformer`](https://huggingface.co/datasets/Aquinlabs/small-worlds-pixel-platformer)
**How it was made:** [`docs.md`](docs.md)

```python
import numpy as np
from huggingface_hub import hf_hub_download
p = hf_hub_download("Aquinlabs/small-worlds-pixel-platformer",
                    "train/shard_0000.npz", repo_type="dataset")
d = np.load(p)   # frames, action_vecs, rewards, ep_id, goal, terminal
```

## Licence

Apache License 2.0. Copyright 2026 Aquin Labs Private Limited. See [`LICENSE.md`](LICENSE.md).
