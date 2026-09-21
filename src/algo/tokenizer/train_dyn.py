"""Train the action-conditioned dynamics model on frozen tokenizer latents.

The tokenizer is frozen, as in dreamer4: dynamics learns to move through a
latent space that is not shifting under it. Stages compose sequentially, which
is why stage 1 had to be right first.

The eval is deliberately a ROLLOUT, not one-step error. One-step latent MSE can
look excellent while multi-step rollouts drift into nonsense, and drift is the
failure mode that decides whether this is playable.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

from dynamics import FlowDynamics, flow_loss, sample_next
from model import (CHAR_IDX, PaletteTokenizer, palette_tensor, to_indices,
                   to_rgb)


def load_shard(p):
    d = np.load(p)
    return d["frames"], d["action_vecs"].astype(np.float32), d["ep_id"], d["terminal"]


def windows(ep_id, terminal, ctx, horizon):
    """Valid start indices where ctx history + horizon future stay in one episode."""
    n = len(ep_id)
    ok = []
    for i in range(ctx - 1, n - horizon):
        if ep_id[i] != ep_id[i - ctx + 1] or ep_id[i] != ep_id[i + horizon]:
            continue
        if terminal[i - ctx + 1:i + horizon].any():
            continue
        ok.append(i)
    return np.array(ok, dtype=np.int64)


class Batcher:
    def __init__(self, shards, ctx, horizon, batch, device, tok, pal, rng, reuse=150):
        self.__dict__.update(locals()); del self.self
        self._load()

    def _load(self):
        p = self.shards[int(self.rng.integers(len(self.shards)))]
        self.fr, self.act, ep, term = load_shard(p)
        self.idx = windows(ep, term, self.ctx, self.horizon)
        self.left = self.reuse

    @torch.no_grad()
    def next(self):
        if self.left <= 0 or len(self.idx) < self.batch:
            self._load()
        self.left -= 1
        i = self.idx[self.rng.integers(0, len(self.idx), size=self.batch)]
        # (B, ctx+horizon) frame indices centred so position ctx-1 is "now"
        cols = np.stack([i + o for o in range(-self.ctx + 1, self.horizon + 1)], 1)
        x = torch.from_numpy(self.fr[cols]).to(self.device)          # (B,T,H,W,3)
        B, T = x.shape[:2]
        x = x.permute(0, 1, 4, 2, 3).reshape(B * T, 3, *x.shape[2:4]).float() / 255.0
        z = self.tok.encoder(x).reshape(B, T, -1, *self.tok.encoder(x[:1]).shape[-2:])
        a = torch.from_numpy(self.act[cols]).to(self.device)          # (B,T,16)
        return z, a, torch.from_numpy(self.fr[cols]).to(self.device)


@torch.no_grad()
def rollout_eval(dyn, tok, batcher, pal, horizon, steps=8, n_batches=2):
    """Imagine `horizon` steps from a real context; score the decoded frames."""
    dyn.eval()
    char = torch.tensor(CHAR_IDX, device=batcher.device)
    pal_hits = pal_tot = ch_hits = ch_tot = 0
    per_step = []
    for _ in range(n_batches):
        z, a, raw = batcher.next()
        ctx = batcher.ctx
        past = z[:, :ctx]
        accs = []
        for h in range(horizon):
            act = a[:, ctx - 1 + h]
            zn = sample_next(dyn, past, act, steps=steps)
            past = torch.cat([past[:, 1:], zn.unsqueeze(1)], 1)
            pred = to_rgb(tok.decoder(zn).argmax(1), pal)
            truth = raw[:, ctx + h].permute(0, 3, 1, 2)
            t_idx = to_indices(truth, pal)
            p_idx = to_indices((pred * 255).to(torch.uint8), pal)
            hit = (p_idx == t_idx)
            m = (t_idx.unsqueeze(-1) == char.view(1, 1, 1, -1)).any(-1)
            pal_hits += hit.sum().item(); pal_tot += hit.numel()
            ch_hits += hit[m].sum().item(); ch_tot += int(m.sum())
            accs.append(hit.float().mean().item())
        per_step.append(accs)
    dyn.train()
    return {"rollout_palette_acc": pal_hits / max(pal_tot, 1),
            "rollout_char_acc": ch_hits / max(ch_tot, 1),
            "per_step_acc": [round(float(np.mean(c)), 4) for c in zip(*per_step)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/train")
    ap.add_argument("--val", default="/data/val")
    ap.add_argument("--tok", default="/runs/pal_v2/latest.pt")
    ap.add_argument("--out", default="/runs/dyn")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=500, dest="eval_every")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pal = palette_tensor(dev)
    ck = torch.load(a.tok, map_location=dev, weights_only=False)
    ta = ck["args"]
    tok = PaletteTokenizer(ta["width"], ta["z_ch"], ta["downs"]).to(dev)
    tok.load_state_dict(ck["model"]); tok.eval()
    for p in tok.parameters():
        p.requires_grad_(False)

    tr = sorted(glob.glob(f"{a.data}/shard_*.npz"))
    va = sorted(glob.glob(f"{a.val}/shard_*.npz"))
    assert tr and va, "no shards"
    rng = np.random.default_rng(0)
    btr = Batcher(tr, a.ctx, a.horizon, a.batch, dev, tok, pal, rng)
    bva = Batcher(va, a.ctx, a.horizon, a.batch, dev, tok, pal, np.random.default_rng(1))

    dyn = FlowDynamics(ta["z_ch"], a.ctx, 16, a.width, a.depth).to(dev)
    print(f"tokenizer frozen ({sum(p.numel() for p in tok.parameters()):,} params)", flush=True)
    print(f"dynamics {sum(p.numel() for p in dyn.parameters()):,} params  "
          f"ctx={a.ctx} horizon={a.horizon}", flush=True)

    opt = torch.optim.AdamW(dyn.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)

    Path(a.out).mkdir(parents=True, exist_ok=True)
    hist, t0 = [], time.time()
    for step in range(1, a.steps + 1):
        z, act, _ = btr.next()
        past, z_next = z[:, :a.ctx], z[:, a.ctx]
        loss = flow_loss(dyn, past, z_next, act[:, a.ctx - 1])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dyn.parameters(), 1.0)
        opt.step(); sched.step()

        if step % a.eval_every == 0 or step == a.steps:
            m = rollout_eval(dyn, tok, bva, pal, a.horizon)
            print(f"step {step:6d} | loss={loss.item():.5f} | "
                  f"rollout_char={m['rollout_char_acc']:.4f} | "
                  f"rollout_pal={m['rollout_palette_acc']:.4f} | "
                  f"per_step={m['per_step_acc']} | "
                  f"{step/(time.time()-t0):.1f} steps/s", flush=True)
            hist.append({"step": step, "loss": loss.item(), **m})
            torch.save({"model": dyn.state_dict(), "args": vars(a), "tok_args": ta},
                       f"{a.out}/latest.pt")
            # Keep the BEST checkpoint too. The 26k run peaked at step 23,000
            # and we kept 26,000 -- the eval is noisy (2 batches x 16 windows),
            # so the last checkpoint is not reliably the best one.
            if m["rollout_char_acc"] >= max(x["rollout_char_acc"] for x in hist):
                torch.save({"model": dyn.state_dict(), "args": vars(a), "tok_args": ta,
                            "metrics": m, "step": step}, f"{a.out}/best.pt")
            Path(f"{a.out}/history.json").write_text(json.dumps(hist, indent=2))

    print(json.dumps({"final": hist[-1] if hist else None,
                      "wall_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
