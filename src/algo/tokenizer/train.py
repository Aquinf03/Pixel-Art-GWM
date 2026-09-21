"""Train the palette tokenizer. Reads our npz shards directly.

Reading npz rather than dreamer4's .pt format skips the 679 GB expansion and
the volume scan that stalled every parallel run: the whole corpus is 0.75 GB
compressed, so a shard loads in milliseconds.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

from model import (CHAR_IDX, PaletteTokenizer, class_weights, loss_fn,
                   palette_tensor, to_indices, to_rgb)


def shard_batches(shards, batch, device, pal, rng, reuse=200):
    """Yield (rgb01, idx) batches.

    `reuse` batches are drawn from each loaded shard before moving on.
    npz is compressed, so `np.load(p)["frames"]` decompresses the whole ~570 MB
    array -- picking a random shard per batch made almost every step pay that,
    which is why the first version produced no steps at all. Each shard holds
    ~11k frames, so 200 batches of 32 still samples it sparsely.
    """
    while True:
        p = shards[int(rng.integers(len(shards)))]
        fr = np.load(p)["frames"]
        for _ in range(reuse):
            i = rng.integers(0, len(fr), size=batch)
            x = torch.from_numpy(fr[i]).to(device).permute(0, 3, 1, 2)
            yield x.float() / 255.0, to_indices(x, pal)


@torch.no_grad()
def evaluate(model, shards, device, pal, n=256, batch=64):
    model.eval()
    fr = np.load(shards[0])["frames"]
    idx = np.linspace(0, len(fr) - 1, n).astype(int)
    hits = char_hits = char_tot = tot = 0
    char = torch.tensor(CHAR_IDX, device=device)
    for s in range(0, n, batch):
        x = torch.from_numpy(fr[idx[s:s + batch]]).to(device).permute(0, 3, 1, 2)
        t = to_indices(x, pal)
        pred = model(x.float() / 255.0)[0].argmax(1)
        hit = (pred == t)
        hits += hit.sum().item(); tot += t.numel()
        m = (t.unsqueeze(-1) == char.view(1, 1, 1, -1)).any(-1)
        char_hits += hit[m].sum().item(); char_tot += int(m.sum())
    model.train()
    return {"palette_acc": hits / max(tot, 1),
            "char_acc": char_hits / max(char_tot, 1),
            "char_pixels_frac": char_tot / max(tot, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/train")
    ap.add_argument("--val", default="/data/val")
    ap.add_argument("--out", default="/runs/pal")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--z-ch", type=int, default=8, dest="z_ch")
    ap.add_argument("--downs", type=int, default=3)
    ap.add_argument("--char-weight", type=float, default=1.0, dest="char_weight")
    ap.add_argument("--eval-every", type=int, default=250, dest="eval_every")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pal = palette_tensor(dev)
    tr = sorted(glob.glob(f"{a.data}/shard_*.npz"))
    va = sorted(glob.glob(f"{a.val}/shard_*.npz"))
    assert tr and va, f"no shards under {a.data} / {a.val}"
    print(f"train shards {len(tr)}  val shards {len(va)}", flush=True)

    model = PaletteTokenizer(a.width, a.z_ch, a.downs).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    z_hw = 128 // (2 ** a.downs)
    print(f"params {n_par:,}  latent {a.z_ch}x{z_hw}x{z_hw} = "
          f"{a.z_ch * z_hw * z_hw:,} numbers/frame", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    w = class_weights(a.char_weight, dev)
    gen = shard_batches(tr, a.batch, dev, pal, np.random.default_rng(0))

    Path(a.out).mkdir(parents=True, exist_ok=True)
    hist, t0 = [], time.time()
    for step in range(1, a.steps + 1):
        x, t = next(gen)
        logits, _ = model(x)
        loss = loss_fn(logits, t, w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % a.eval_every == 0 or step == a.steps:
            m = evaluate(model, va, dev, pal)
            rate = step / (time.time() - t0)
            print(f"step {step:6d} | loss={loss.item():.4f} | "
                  f"char_acc={m['char_acc']:.4f} | palette_acc={m['palette_acc']:.4f} | "
                  f"{rate:.1f} steps/s", flush=True)
            hist.append({"step": step, "loss": loss.item(), **m})
            torch.save({"model": model.state_dict(), "args": vars(a)},
                       f"{a.out}/latest.pt")
            Path(f"{a.out}/history.json").write_text(json.dumps(hist, indent=2))

    print(json.dumps({"final": hist[-1] if hist else None,
                      "params": n_par, "wall_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
