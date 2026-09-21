"""Run the palette tokenizer on Modal.

Deliberately independent of scripts/tests/train_d4.py: no dreamer4 checkout, no 679 GB .pt
corpus, no 989-shard volume scan. It reads the compact npz shards directly,
so a container is training within seconds of starting.
"""

from __future__ import annotations

from pathlib import Path

import modal

_TOK = Path(__file__).resolve().parent / "tokenizer"

app = modal.App("pal-tokenizer")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "numpy==2.*", "pillow")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_dir(str(_TOK), "/root/tok")
)

data_vol = modal.Volume.from_name("pixel-world-data", create_if_missing=True)
run_vol = modal.Volume.from_name("d4-runs", create_if_missing=True)
DATA, RUNS = "/data", "/runs"


@app.function(image=image, volumes={DATA: data_vol, RUNS: run_vol},
              gpu="A100-40GB", cpu=8.0, memory=32768, timeout=7200)
def train(run: str = "pal", steps: int = 4000, batch: int = 32,
          char_weight: float = 1.0, width: int = 64, z_ch: int = 8,
          downs: int = 3, lr: float = 3e-4) -> dict:
    import json, os, subprocess, sys
    data_vol.reload()
    out = f"{RUNS}/{run}"
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "train.py",
           "--data", f"{DATA}/train", "--val", f"{DATA}/val", "--out", out,
           "--steps", str(steps), "--batch", str(batch), "--lr", str(lr),
           "--width", str(width), "--z-ch", str(z_ch), "--downs", str(downs),
           "--char-weight", str(char_weight)]
    print("[runner] " + " ".join(cmd[1:]), flush=True)
    p = subprocess.run(cmd, cwd="/root/tok", capture_output=False, text=True)
    run_vol.commit()
    hist = f"{out}/history.json"
    h = json.loads(open(hist).read()) if os.path.exists(hist) else []
    return {"run": run, "returncode": p.returncode,
            "final": h[-1] if h else None, "history": h[-6:]}


@app.local_entrypoint()
def main(run: str = "pal", steps: int = 4000, batch: int = 32,
         char_weight: float = 1.0, width: int = 64, z_ch: int = 8, downs: int = 3):
    import json
    r = train.remote(run, steps, batch, char_weight, width, z_ch, downs)
    print(json.dumps(r, indent=2))


@app.function(image=image, volumes={DATA: data_vol, RUNS: run_vol},
              gpu="A10G", cpu=4.0, memory=16384, timeout=1800)
def evaluate(run: str = "pal_v2", n: int = 8) -> dict:
    """Reconstruction sheet + u_r, on held-out val and on the rare ledge.

    The picture matters as much as the number: a plausible metric hid a robot
    arm in phase 2 and a missing fox this morning.
    """
    import glob, json, sys
    import numpy as np, torch
    from PIL import Image
    sys.path.insert(0, "/root/tok")
    from model import PaletteTokenizer, palette_tensor, to_indices, to_rgb, CHAR_IDX

    data_vol.reload(); run_vol.reload()
    dev = torch.device("cuda")
    ck = torch.load(f"{RUNS}/{run}/latest.pt", map_location=dev, weights_only=False)
    a = ck["args"]
    m = PaletteTokenizer(a["width"], a["z_ch"], a["downs"]).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    pal = palette_tensor(dev)

    out = {"run": run, "params": sum(p.numel() for p in m.parameters())}
    sheets = {}
    for split in ("val", "probe_ledge"):
        sh = sorted(glob.glob(f"{DATA}/{split}/shard_*.npz"))
        if not sh:
            continue
        fr = np.load(sh[0])["frames"]
        idx = np.linspace(0, len(fr) - 1, 64).astype(int)
        x = torch.from_numpy(fr[idx]).to(dev).permute(0, 3, 1, 2)
        x01 = x.float() / 255.0
        with torch.no_grad():
            rec, z = m.reconstruct(x01, pal)
            d, dn = m.u_r(x01, pal)
        t = to_indices(x, pal)
        pr = to_indices((rec * 255).to(torch.uint8), pal)
        hit = (pr == t)
        ch = torch.tensor(CHAR_IDX, device=dev)
        mask = (t.unsqueeze(-1) == ch.view(1, 1, 1, -1)).any(-1)
        out[split] = {
            "palette_acc": round(hit.float().mean().item(), 5),
            "char_acc": round(hit[mask].float().mean().item(), 5),
            "u_r": round(d.mean().item(), 4),
            "u_r_norm": round(dn.mean().item(), 4),
        }
        k = min(n, x.shape[0])
        top = (x01[:k].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        bot = (rec[:k].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        H, W = top.shape[1:3]
        sheet = np.zeros((2 * H + 12, k * W + (k + 1) * 4, 3), np.uint8) + 18
        for i in range(k):
            xo = 4 + i * (W + 4)
            sheet[4:4 + H, xo:xo + W] = top[i]
            sheet[H + 8:H + 8 + H, xo:xo + W] = bot[i]
        Image.fromarray(sheet).save(f"{RUNS}/{run}/recon_{split}.png")
        sheets[split] = f"{run}/recon_{split}.png"
    out["images"] = sheets
    run_vol.commit()
    return out


@app.local_entrypoint()
def check(run: str = "pal_v2"):
    import json
    print(json.dumps(evaluate.remote(run), indent=2))


@app.function(image=image, volumes={DATA: data_vol, RUNS: run_vol},
              gpu="A10G", cpu=4.0, memory=16384, timeout=1800)
def demo(run: str = "pal_v2") -> dict:
    """Visual demo: reconstruction, per-pixel error, and the latent space."""
    import glob, json, sys
    import numpy as np, torch
    from PIL import Image
    sys.path.insert(0, "/root/tok")
    from model import PaletteTokenizer, palette_tensor, to_indices, to_rgb, CHAR_IDX

    data_vol.reload(); run_vol.reload()
    dev = torch.device("cuda")
    ck = torch.load(f"{RUNS}/{run}/latest.pt", map_location=dev, weights_only=False)
    a = ck["args"]
    m = PaletteTokenizer(a["width"], a["z_ch"], a["downs"]).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    pal = palette_tensor(dev)
    out = f"{RUNS}/{run}"

    fr = np.load(sorted(glob.glob(f"{DATA}/val/shard_*.npz"))[0])["frames"]
    pick = np.linspace(0, len(fr) - 1, 6).astype(int)
    x = torch.from_numpy(fr[pick]).to(dev).permute(0, 3, 1, 2)
    x01 = x.float() / 255.0
    with torch.no_grad():
        z = m.encoder(x01)
        rec = to_rgb(m.decoder(z).argmax(1), pal)

    O = (x01.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    R = (rec.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    D = (np.any(O != R, -1).astype(np.uint8) * 255)

    def strip(rows, H, W, pad=4, bg=18):
        n = len(rows[0])
        sh = np.zeros((len(rows) * (H + pad) + pad, n * (W + pad) + pad, 3), np.uint8) + bg
        for r, row in enumerate(rows):
            for i, im in enumerate(row):
                im = np.dstack([im] * 3) if im.ndim == 2 else im
                sh[pad + r * (H + pad):pad + r * (H + pad) + H,
                   pad + i * (W + pad):pad + i * (W + pad) + W] = im
        return sh

    H, W = O.shape[1:3]
    Image.fromarray(strip([list(O), list(R), list(D)], H, W)).save(f"{out}/demo_recon.png")

    # latent channels for one frame, upscaled so 16x16 is legible
    zc = z[3].cpu().numpy()                       # (C,16,16)
    zc = (zc - zc.min()) / (zc.max() - zc.min() + 1e-8)  # numpy 2 removed ndarray.ptp
    tiles = [np.asarray(Image.fromarray((c * 255).astype(np.uint8)).resize((W, W), Image.NEAREST))
             for c in zc]
    Image.fromarray(strip([tiles[:4], tiles[4:8]], W, W)).save(f"{out}/demo_latent.png")

    # does the latent TRACK the character? compare two frames, one moved
    with torch.no_grad():
        za, zb = z[0:1], z[1:2]
    dz = (za - zb).abs().mean(1)[0].cpu().numpy()
    dz = (dz / (dz.max() + 1e-8) * 255).astype(np.uint8)
    dz_img = np.asarray(Image.fromarray(dz).resize((W, W), Image.NEAREST))
    Image.fromarray(strip([[O[0], O[1], np.dstack([dz_img] * 3)]], H, W)).save(f"{out}/demo_latent_delta.png")

    err = float((O != R).any(-1).mean())
    stats = {"frames": int(len(pick)),
             "input_numbers": int(3 * H * W),
             "latent_numbers": int(z[0].numel()),
             "compression": round(3 * H * W / z[0].numel(), 1),
             "wrong_pixels_frac": round(err, 6),
             "params": sum(p.numel() for p in m.parameters())}
    open(f"{out}/demo_stats.json", "w").write(json.dumps(stats, indent=2))
    run_vol.commit()
    return stats


@app.local_entrypoint()
def make_demo(run: str = "pal_v2"):
    import json
    print(json.dumps(demo.remote(run), indent=2))


@app.function(image=image, volumes={DATA: data_vol, RUNS: run_vol},
              gpu="A100-40GB", cpu=8.0, memory=32768, timeout=7200)
def train_dynamics(run: str = "dyn", steps: int = 3000, batch: int = 16,
                   ctx: int = 8, horizon: int = 8, width: int = 128,
                   depth: int = 4, lr: float = 2e-4,
                   tok_run: str = "pal_v2") -> dict:
    import json, os, subprocess, sys
    data_vol.reload(); run_vol.reload()
    out = f"{RUNS}/{run}"
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "train_dyn.py",
           "--data", f"{DATA}/train", "--val", f"{DATA}/val",
           "--tok", f"{RUNS}/{tok_run}/latest.pt", "--out", out,
           "--steps", str(steps), "--batch", str(batch), "--ctx", str(ctx),
           "--horizon", str(horizon), "--width", str(width),
           "--depth", str(depth), "--lr", str(lr)]
    print("[runner] " + " ".join(cmd[1:]), flush=True)
    p = subprocess.run(cmd, cwd="/root/tok", text=True)
    run_vol.commit()
    h = f"{out}/history.json"
    hist = json.loads(open(h).read()) if os.path.exists(h) else []
    return {"run": run, "returncode": p.returncode,
            "final": hist[-1] if hist else None, "history": hist[-4:]}


@app.local_entrypoint()
def dyn(run: str = "dyn_v1", steps: int = 3000, batch: int = 16,
        n_ctx: int = 8, horizon: int = 8, tok_run: str = "pal_v2"):
    # `ctx` is reserved by Modal's CLI (Click context), hence n_ctx.
    import json
    print(json.dumps(train_dynamics.remote(run, steps, batch, n_ctx, horizon,
                                           tok_run=tok_run), indent=2))


@app.function(image=image, volumes={DATA: data_vol, RUNS: run_vol},
              gpu="A10G", cpu=4.0, memory=16384, timeout=1800)
def rollout_demo(dyn_run: str = "dyn_v2", tok_run: str = "pal_v2",
                 horizon: int = 12) -> dict:
    """Two demos, the second being the one that matters.

    1. IMAGINATION: give the model 8 real frames, then let it imagine 12 more
       from the real actions. Compare against what actually happened.
    2. ACTION FIDELITY: same context, but press LEFT vs RIGHT. If the two
       rollouts diverge, the model is genuinely action-conditioned rather than
       replaying a memorised trajectory -- the failure the paper calls action
       marginalisation.
    """
    import glob, json, sys
    import numpy as np, torch
    from PIL import Image
    sys.path.insert(0, "/root/tok")
    from model import PaletteTokenizer, palette_tensor, to_indices, to_rgb, CHAR_IDX
    from dynamics import FlowDynamics, sample_next

    data_vol.reload(); run_vol.reload()
    dev = torch.device("cuda")
    tck = torch.load(f"{RUNS}/{tok_run}/latest.pt", map_location=dev, weights_only=False)
    ta = tck["args"]
    tok = PaletteTokenizer(ta["width"], ta["z_ch"], ta["downs"]).to(dev)
    tok.load_state_dict(tck["model"]); tok.eval()
    dck = torch.load(f"{RUNS}/{dyn_run}/latest.pt", map_location=dev, weights_only=False)
    da = dck["args"]
    dyn = FlowDynamics(ta["z_ch"], da["ctx"], 16, da["width"], da["depth"]).to(dev)
    dyn.load_state_dict(dck["model"]); dyn.eval()
    pal = palette_tensor(dev)
    ctx = da["ctx"]
    out = f"{RUNS}/{dyn_run}"

    d = np.load(sorted(glob.glob(f"{DATA}/val/shard_*.npz"))[0])
    fr, av, ep, term = d["frames"], d["action_vecs"].astype(np.float32), d["ep_id"], d["terminal"]
    start = None
    for i in range(ctx, len(fr) - horizon - 2):
        if ep[i] == ep[i - ctx] == ep[i + horizon] and not term[i - ctx:i + horizon].any():
            start = i; break
    assert start is not None, "no clean window"

    def encode(idx):
        x = torch.from_numpy(fr[idx]).to(dev).permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad():
            return tok.encoder(x)

    past0 = encode(np.arange(start - ctx, start)).unsqueeze(0)
    truth = fr[start:start + horizon]

    def roll(actions):
        past = past0.clone(); frames = []
        with torch.no_grad():
            for h in range(horizon):
                z = sample_next(dyn, past, actions[h].unsqueeze(0), steps=8)
                past = torch.cat([past[:, 1:], z.unsqueeze(1)], 1)
                frames.append((to_rgb(tok.decoder(z).argmax(1), pal)[0]
                               .permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        return np.stack(frames)

    real_a = torch.from_numpy(av[start:start + horizon]).to(dev)
    imagined = roll(real_a)

    # forced LEFT vs forced RIGHT from the same context
    def const(vec):
        a = torch.zeros(horizon, 16, device=dev); a[:, 0] = vec
        return a
    left, right = roll(const(-1.0)), roll(const(1.0))

    def strip(rows, pad=4):
        H, W = rows[0][0].shape[:2]; n = len(rows[0])
        sh = np.zeros((len(rows) * (H + pad) + pad, n * (W + pad) + pad, 3), np.uint8) + 18
        for r, row in enumerate(rows):
            for i, im in enumerate(row):
                sh[pad + r * (H + pad):pad + r * (H + pad) + H,
                   pad + i * (W + pad):pad + i * (W + pad) + W] = im
        return sh

    Image.fromarray(strip([list(truth), list(imagined)])).save(f"{out}/demo_imagine.png")
    Image.fromarray(strip([list(left), list(right)])).save(f"{out}/demo_actions.png")

    div = float(np.mean(np.any(left != right, -1)))
    t = to_indices(torch.from_numpy(truth).to(dev).permute(0, 3, 1, 2), pal)
    p = to_indices(torch.from_numpy(imagined).to(dev).permute(0, 3, 1, 2), pal)
    ch = torch.tensor(CHAR_IDX, device=dev)
    m = (t.unsqueeze(-1) == ch.view(1, 1, 1, -1)).any(-1)
    res = {"horizon": horizon,
           "imagined_palette_acc": round((p == t).float().mean().item(), 4),
           "imagined_char_acc": round((p == t)[m].float().mean().item(), 4),
           "left_vs_right_divergence": round(div, 4),
           "images": [f"{dyn_run}/demo_imagine.png", f"{dyn_run}/demo_actions.png"]}
    open(f"{out}/demo_rollout.json", "w").write(json.dumps(res, indent=2))
    run_vol.commit()
    return res


@app.local_entrypoint()
def show(dyn_run: str = "dyn_v2", horizon: int = 12):
    import json
    print(json.dumps(rollout_demo.remote(dyn_run, "pal_v2", horizon), indent=2))


@app.function(image=image, volumes={DATA: data_vol, RUNS: run_vol},
              gpu="A10G", cpu=4.0, memory=16384, timeout=1800)
def action_fidelity(dyn_run: str = "dyn_v2", tok_run: str = "pal_v2",
                    horizon: int = 12, n_starts: int = 12) -> dict:
    """Does LEFT actually move the character left?

    Divergence alone is not enough -- a model could respond to actions
    incoherently and still diverge. This tracks the character's centroid
    (found by its own palette colours) and checks the SIGN of its motion.
    That is the paper's action-fidelity check, and the failure it detects is
    action marginalisation.
    """
    import glob, json, sys
    import numpy as np, torch
    sys.path.insert(0, "/root/tok")
    from model import PaletteTokenizer, palette_tensor, to_indices, to_rgb, CHAR_IDX
    from dynamics import FlowDynamics, sample_next

    data_vol.reload(); run_vol.reload()
    dev = torch.device("cuda")
    tck = torch.load(f"{RUNS}/{tok_run}/latest.pt", map_location=dev, weights_only=False)
    ta = tck["args"]
    tok = PaletteTokenizer(ta["width"], ta["z_ch"], ta["downs"]).to(dev)
    tok.load_state_dict(tck["model"]); tok.eval()
    dck = torch.load(f"{RUNS}/{dyn_run}/latest.pt", map_location=dev, weights_only=False)
    da = dck["args"]
    dyn = FlowDynamics(ta["z_ch"], da["ctx"], 16, da["width"], da["depth"]).to(dev)
    dyn.load_state_dict(dck["model"]); dyn.eval()
    pal = palette_tensor(dev); ctx = da["ctx"]
    char = torch.tensor(CHAR_IDX, device=dev)

    d = np.load(sorted(glob.glob(f"{DATA}/val/shard_*.npz"))[0])
    fr, ep, term = d["frames"], d["ep_id"], d["terminal"]
    starts = [i for i in range(ctx, len(fr) - horizon - 2)
              if ep[i] == ep[i - ctx] == ep[i + horizon]
              and not term[i - ctx:i + horizon].any()][:n_starts * 40:40][:n_starts]

    def centroid_x(img_u8):
        idx = to_indices(img_u8, pal)
        m = (idx.unsqueeze(-1) == char.view(1, 1, 1, -1)).any(-1).float()
        xs = torch.arange(idx.shape[-1], device=dev).view(1, 1, -1)
        tot = m.sum((1, 2))
        return torch.where(tot > 0, (m * xs).sum((1, 2)) / tot.clamp_min(1),
                           torch.full_like(tot, float("nan")))

    drift = {"left": [], "right": [], "idle": []}
    for s in starts:
        x = torch.from_numpy(fr[np.arange(s - ctx, s)]).to(dev).permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad():
            past0 = tok.encoder(x).unsqueeze(0)
        for name, v in (("left", -1.0), ("right", 1.0), ("idle", 0.0)):
            past = past0.clone(); xs = []
            a = torch.zeros(1, 16, device=dev); a[0, 0] = v
            with torch.no_grad():
                for _ in range(horizon):
                    z = sample_next(dyn, past, a, steps=8)
                    past = torch.cat([past[:, 1:], z.unsqueeze(1)], 1)
                    img = (to_rgb(tok.decoder(z).argmax(1), pal) * 255).to(torch.uint8)
                    xs.append(centroid_x(img).item())
            xs = [v for v in xs if v == v]
            if len(xs) >= 2:
                drift[name].append(xs[-1] - xs[0])

    res = {"horizon": horizon, "starts": len(starts)}
    for k, v in drift.items():
        res[k] = {"mean_x_drift_px": round(float(np.mean(v)), 2),
                  "n": len(v)} if v else {"mean_x_drift_px": None, "n": 0}
    lm = res["left"]["mean_x_drift_px"]; rm = res["right"]["mean_x_drift_px"]
    res["correct_direction"] = bool(lm is not None and rm is not None and lm < 0 < rm)
    res["separation_px"] = round(rm - lm, 2) if (lm is not None and rm is not None) else None
    open(f"{RUNS}/{dyn_run}/action_fidelity.json", "w").write(json.dumps(res, indent=2))
    run_vol.commit()
    return res


@app.local_entrypoint()
def fidelity(dyn_run: str = "dyn_v2", horizon: int = 12):
    import json
    print(json.dumps(action_fidelity.remote(dyn_run, "pal_v2", horizon), indent=2))


@app.function(image=image, volumes={DATA: data_vol, RUNS: run_vol},
              gpu="A100-40GB", cpu=8.0, memory=32768, timeout=3600)
def predictors(dyn_run: str = "dyn_v2", tok_run: str = "pal_v2",
               horizon: int = 8, n_windows: int = 64, seeds: int = 4) -> dict:
    """The paper's three label-free hallucination predictors, on our model.

      u_r  ||z_hat - Enc(Dec(z_hat))||   tokenizer round trip on the PREDICTED
           latent. Nothing in training optimises it, which is why it stays
           informative.
      u_f  how far the denoiser's clean-frame prediction moves between Euler
           substeps. A well-conditioned step settles; a hallucinated one keeps
           moving.
      u_s  spread of the next latent across independent denoising seeds.

    The claim under test is that these track REALISED rollout error without
    ever seeing it. We measure both and correlate, and we also split by
    region -- the rare top ledge is where they should fire.
    """
    import glob, json, sys
    import numpy as np, torch
    sys.path.insert(0, "/root/tok")
    from model import PaletteTokenizer, palette_tensor, to_indices, to_rgb, CHAR_IDX
    from dynamics import FlowDynamics, sample_next

    data_vol.reload(); run_vol.reload()
    dev = torch.device("cuda")
    tck = torch.load(f"{RUNS}/{tok_run}/latest.pt", map_location=dev, weights_only=False)
    ta = tck["args"]
    tok = PaletteTokenizer(ta["width"], ta["z_ch"], ta["downs"]).to(dev)
    tok.load_state_dict(tck["model"]); tok.eval()
    dck = torch.load(f"{RUNS}/{dyn_run}/latest.pt", map_location=dev, weights_only=False)
    da = dck["args"]
    dyn = FlowDynamics(ta["z_ch"], da["ctx"], 16, da["width"], da["depth"]).to(dev)
    dyn.load_state_dict(dck["model"]); dyn.eval()
    pal = palette_tensor(dev); ctx = da["ctx"]

    rows = []
    for split in ("val", "probe_ledge"):
        shards = sorted(glob.glob(f"{DATA}/{split}/shard_*.npz"))
        if not shards:
            continue
        d = np.load(shards[0])
        fr, av, ep, term = d["frames"], d["action_vecs"].astype(np.float32), d["ep_id"], d["terminal"]
        ok = [i for i in range(ctx, len(fr) - horizon - 2)
              if ep[i] == ep[i - ctx] == ep[i + horizon]
              and not term[i - ctx:i + horizon].any()]
        step = max(1, len(ok) // n_windows)
        for s in ok[::step][:n_windows]:
            x = torch.from_numpy(fr[np.arange(s - ctx, s)]).to(dev).permute(0, 3, 1, 2).float() / 255.0
            with torch.no_grad():
                past = tok.encoder(x).unsqueeze(0)
            for h in range(horizon):
                a = torch.from_numpy(av[s + h]).to(dev).unsqueeze(0)
                with torch.no_grad():
                    z, trace = sample_next(dyn, past, a, steps=8, return_trace=True)
                    # u_f: movement of the clean-frame prediction across substeps
                    u_f = float(torch.stack([(trace[i + 1] - trace[i]).flatten(1).norm(dim=1)
                                             for i in range(len(trace) - 1)]).mean())
                    # u_s: spread across independent seeds
                    zs = torch.stack([sample_next(
                        dyn, past, a, steps=8,
                        generator=torch.Generator(device=dev).manual_seed(1000 + k))
                        for k in range(seeds)])
                    u_s = float(zs.std(0).mean())
                    # u_r: round trip of the PREDICTED latent
                    rec = to_rgb(tok.decoder(z).argmax(1), pal)
                    z2 = tok.encoder(rec)
                    u_r = float((z - z2).flatten(1).norm(dim=1).mean())
                    u_r_n = u_r / max(float(z.flatten(1).norm(dim=1).mean()), 1e-8)
                    # realised error, which the predictors never see
                    truth = torch.from_numpy(fr[s + h]).to(dev).permute(2, 0, 1).unsqueeze(0)
                    t_idx = to_indices(truth, pal)
                    p_idx = to_indices((rec * 255).to(torch.uint8), pal)
                    err = float((p_idx != t_idx).float().mean())
                    ch = torch.tensor(CHAR_IDX, device=dev)
                    m = (t_idx.unsqueeze(-1) == ch.view(1, 1, 1, -1)).any(-1)
                    cerr = float((p_idx != t_idx)[m].float().mean()) if m.any() else float("nan")
                past = torch.cat([past[:, 1:], z.unsqueeze(1)], 1)
                rows.append({"split": split, "h": h, "u_r": u_r, "u_r_norm": u_r_n,
                             "u_f": u_f, "u_s": u_s, "err": err, "char_err": cerr})

    def spearman(a, b):
        a, b = np.asarray(a), np.asarray(b)
        ok = ~(np.isnan(a) | np.isnan(b))
        a, b = a[ok], b[ok]
        if len(a) < 8:
            return None
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        return round(float(np.corrcoef(ra, rb)[0, 1]), 3)

    out = {"n": len(rows), "horizon": horizon, "seeds": seeds}
    for key in ("u_r", "u_r_norm", "u_f", "u_s"):
        out[f"spearman_{key}_vs_error"] = spearman([r[key] for r in rows],
                                                   [r["err"] for r in rows])
        out[f"spearman_{key}_vs_char_error"] = spearman([r[key] for r in rows],
                                                        [r["char_err"] for r in rows])
    for split in ("val", "probe_ledge"):
        sub = [r for r in rows if r["split"] == split]
        if sub:
            out[split] = {k: round(float(np.nanmean([r[k] for r in sub])), 5)
                          for k in ("u_r", "u_r_norm", "u_f", "u_s", "err", "char_err")}
            out[split]["n"] = len(sub)
    open(f"{RUNS}/{dyn_run}/predictors.json", "w").write(json.dumps(
        {"summary": out, "rows": rows[:2000]}, indent=2))
    run_vol.commit()
    return out


@app.local_entrypoint()
def preds(dyn_run: str = "dyn_v2", horizon: int = 8, n_windows: int = 64):
    import json
    print(json.dumps(predictors.remote(dyn_run, "pal_v2", horizon, n_windows), indent=2))


@app.function(image=image, volumes={DATA: data_vol, RUNS: run_vol},
              gpu="A100-40GB", cpu=8.0, memory=32768, timeout=3600)
def video_frames(dyn_run: str = "dyn_v2", tok_run: str = "pal_v2",
                 horizon: int = 60) -> dict:
    """A long rollout, saved as raw frames for encoding locally.

    Three panels: what really happened, what the model imagined from the same
    actions, and the pixels that differ. 60 frames = 4 seconds at 15 fps, which
    is 7.5x the model's 8-frame context -- so most of what you see is the model
    running on its own output.
    """
    import glob, sys
    import numpy as np, torch
    sys.path.insert(0, "/root/tok")
    from model import PaletteTokenizer, palette_tensor, to_rgb
    from dynamics import FlowDynamics, sample_next

    data_vol.reload(); run_vol.reload()
    dev = torch.device("cuda")
    tck = torch.load(f"{RUNS}/{tok_run}/latest.pt", map_location=dev, weights_only=False)
    ta = tck["args"]
    tok = PaletteTokenizer(ta["width"], ta["z_ch"], ta["downs"]).to(dev)
    tok.load_state_dict(tck["model"]); tok.eval()
    dck = torch.load(f"{RUNS}/{dyn_run}/latest.pt", map_location=dev, weights_only=False)
    da = dck["args"]
    dyn = FlowDynamics(ta["z_ch"], da["ctx"], 16, da["width"], da["depth"]).to(dev)
    dyn.load_state_dict(dck["model"]); dyn.eval()
    pal = palette_tensor(dev); ctx = da["ctx"]

    d = np.load(sorted(glob.glob(f"{DATA}/val/shard_*.npz"))[0])
    fr, av, ep, term = d["frames"], d["action_vecs"].astype(np.float32), d["ep_id"], d["terminal"]
    best, blen = None, 0
    for i in range(ctx, len(fr) - horizon - 2):
        if ep[i] == ep[i - ctx] == ep[i + horizon] and not term[i - ctx:i + horizon].any():
            best = i; break
    assert best is not None
    s = best

    x = torch.from_numpy(fr[np.arange(s - ctx, s)]).to(dev).permute(0, 3, 1, 2).float() / 255.0
    with torch.no_grad():
        past = tok.encoder(x).unsqueeze(0)
    imagined = []
    with torch.no_grad():
        for h in range(horizon):
            a = torch.from_numpy(av[s + h]).to(dev).unsqueeze(0)
            z = sample_next(dyn, past, a, steps=8)
            past = torch.cat([past[:, 1:], z.unsqueeze(1)], 1)
            img = to_rgb(tok.decoder(z).argmax(1), pal)[0]
            imagined.append((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    imagined = np.stack(imagined)
    truth = fr[s:s + horizon]
    np.savez_compressed(f"{RUNS}/{dyn_run}/video.npz",
                        truth=truth, imagined=imagined,
                        context=fr[s - ctx:s], actions=av[s:s + horizon])
    run_vol.commit()
    return {"frames": int(horizon), "context": int(ctx),
            "file": f"{dyn_run}/video.npz"}


@app.local_entrypoint()
def vid(dyn_run: str = "dyn_v2", horizon: int = 60):
    import json
    print(json.dumps(video_frames.remote(dyn_run, "pal_v2", horizon), indent=2))


@app.function(image=image, volumes={DATA: data_vol, RUNS: run_vol},
              gpu="A100-40GB", cpu=8.0, memory=32768, timeout=3600)
def scripted_rollout(dyn_run: str = "dyn_v2", tok_run: str = "pal_v2") -> dict:
    """Drive the model with a HAND-WRITTEN action script, not replayed actions.

    Replaying recorded actions shows the model can predict. A script shows
    causality: you can read the command and watch the fox obey it. Held runs
    of one action make the response legible at slow playback.
    """
    import glob, json, sys
    import numpy as np, torch
    sys.path.insert(0, "/root/tok")
    from model import PaletteTokenizer, palette_tensor, to_rgb
    from dynamics import FlowDynamics, sample_next

    # (action name, dim0 move, dim1 vertical, dim2 use, repeats)
    SCRIPT = [
        ("idle",   0.0,  0.0, 0.0, 6),
        ("right",  1.0,  0.0, 0.0, 18),
        ("idle",   0.0,  0.0, 0.0, 6),
        ("left",  -1.0,  0.0, 0.0, 18),
        ("idle",   0.0,  0.0, 0.0, 6),
        ("jump",   0.0,  1.0, 0.0, 10),
        ("right",  1.0,  0.0, 0.0, 14),
        ("use",    0.0,  0.0, 1.0, 8),
        ("crouch", 0.0, -1.0, 0.0, 10),
        ("idle",   0.0,  0.0, 0.0, 6),
        ("left",  -1.0,  0.0, 0.0, 18),
        ("jump",   0.0,  1.0, 0.0, 10),
        ("right",  1.0,  0.0, 0.0, 20),
        ("idle",   0.0,  0.0, 0.0, 10),
    ]

    data_vol.reload(); run_vol.reload()
    dev = torch.device("cuda")
    tck = torch.load(f"{RUNS}/{tok_run}/latest.pt", map_location=dev, weights_only=False)
    ta = tck["args"]
    tok = PaletteTokenizer(ta["width"], ta["z_ch"], ta["downs"]).to(dev)
    tok.load_state_dict(tck["model"]); tok.eval()
    dck = torch.load(f"{RUNS}/{dyn_run}/latest.pt", map_location=dev, weights_only=False)
    da = dck["args"]
    dyn = FlowDynamics(ta["z_ch"], da["ctx"], 16, da["width"], da["depth"]).to(dev)
    dyn.load_state_dict(dck["model"]); dyn.eval()
    pal = palette_tensor(dev); ctx = da["ctx"]

    d = np.load(sorted(glob.glob(f"{DATA}/val/shard_*.npz"))[0])
    fr, ep, term = d["frames"], d["ep_id"], d["terminal"]
    s = next(i for i in range(ctx, len(fr) - 4)
             if ep[i] == ep[i - ctx] and not term[i - ctx:i].any())
    x = torch.from_numpy(fr[np.arange(s - ctx, s)]).to(dev).permute(0, 3, 1, 2).float() / 255.0
    with torch.no_grad():
        past = tok.encoder(x).unsqueeze(0)

    frames, labels = [], []
    with torch.no_grad():
        for name, mv, vt, us, rep in SCRIPT:
            a = torch.zeros(1, 16, device=dev)
            a[0, 0], a[0, 1], a[0, 2] = mv, vt, us
            for _ in range(rep):
                z = sample_next(dyn, past, a, steps=8)
                past = torch.cat([past[:, 1:], z.unsqueeze(1)], 1)
                img = to_rgb(tok.decoder(z).argmax(1), pal)[0]
                frames.append((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                labels.append(name)
    np.savez_compressed(f"{RUNS}/{dyn_run}/scripted.npz",
                        frames=np.stack(frames), labels=np.array(labels),
                        context=fr[s - ctx:s])
    run_vol.commit()
    return {"frames": len(frames), "actions": sorted(set(labels)),
            "file": f"{dyn_run}/scripted.npz"}


@app.local_entrypoint()
def script_vid(dyn_run: str = "dyn_v2"):
    import json
    print(json.dumps(scripted_rollout.remote(dyn_run, "pal_v2"), indent=2))


@app.function(image=image, volumes={DATA: data_vol, RUNS: run_vol},
              gpu="A100-40GB", cpu=8.0, memory=32768, timeout=1800)
def benchmark(dyn_run: str = "dyn_v2", tok_run: str = "pal_v2") -> dict:
    """Raw generation speed on-GPU, isolated from network and PNG encoding."""
    import glob, sys, time
    import numpy as np, torch
    sys.path.insert(0, "/root/tok")
    from model import PaletteTokenizer, palette_tensor, to_rgb
    from dynamics import FlowDynamics, sample_next

    data_vol.reload(); run_vol.reload()
    dev = torch.device("cuda")
    tck = torch.load(f"{RUNS}/{tok_run}/latest.pt", map_location=dev, weights_only=False)
    ta = tck["args"]
    tok = PaletteTokenizer(ta["width"], ta["z_ch"], ta["downs"]).to(dev)
    tok.load_state_dict(tck["model"]); tok.eval()
    dck = torch.load(f"{RUNS}/{dyn_run}/latest.pt", map_location=dev, weights_only=False)
    da = dck["args"]
    dyn = FlowDynamics(ta["z_ch"], da["ctx"], 16, da["width"], da["depth"]).to(dev)
    dyn.load_state_dict(dck["model"]); dyn.eval()
    pal = palette_tensor(dev); ctx = da["ctx"]

    fr = np.load(sorted(glob.glob(f"{DATA}/val/shard_*.npz"))[0])["frames"]
    x = torch.from_numpy(fr[:ctx]).to(dev).permute(0, 3, 1, 2).float() / 255.0
    with torch.no_grad():
        past0 = tok.encoder(x).unsqueeze(0)
    a = torch.zeros(1, 16, device=dev); a[0, 0] = 1.0

    out = {}
    for steps in (1, 2, 4, 8):
        past = past0.clone()
        with torch.no_grad():                       # warm up
            for _ in range(3):
                sample_next(dyn, past, a, steps=steps)
        torch.cuda.synchronize(); t0 = time.time()
        n = 30
        with torch.no_grad():
            for _ in range(n):
                z = sample_next(dyn, past, a, steps=steps)
                past = torch.cat([past[:, 1:], z.unsqueeze(1)], 1)
                to_rgb(tok.decoder(z).argmax(1), pal)
        torch.cuda.synchronize()
        d = time.time() - t0
        out[f"denoise_steps_{steps}"] = {"fps": round(n / d, 1),
                                         "ms_per_frame": round(d / n * 1000, 1)}
    # batched: many independent worlds at once
    for B in (1, 8, 32):
        past = past0.repeat(B, 1, 1, 1, 1); ab = a.repeat(B, 1)
        with torch.no_grad():
            sample_next(dyn, past, ab, steps=4)
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            for _ in range(10):
                sample_next(dyn, past, ab, steps=4)
        torch.cuda.synchronize()
        d = time.time() - t0
        out[f"batch_{B}_at_4_steps"] = {"total_fps": round(10 * B / d, 1)}
    out["env_native_fps"] = 15
    return out


@app.local_entrypoint()
def bench(dyn_run: str = "dyn_v2"):
    import json
    print(json.dumps(benchmark.remote(dyn_run, "pal_v2"), indent=2))
