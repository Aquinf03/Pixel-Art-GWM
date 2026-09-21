"""Decisive base-model test, before spending real GPU money.

Issue #9 on the dreamer4 repo reports blurry dynamics rollouts even after 145k
steps on 4xA100, and the author has not replied. So before building anything on
this base we check the RELEASED checkpoint directly, on two questions:

  A. Does the frozen tokenizer survive our pixel art?  Encode one of our frames
     and decode it back. This is exactly the round trip `u_r` is built on, and
     brief section 6 predicts the encoder may snap our sprite onto something it
     knows -- perceptual hallucination.

  B. Are its dynamics rollouts sharp?  Seed from our own encoded frame and roll
     forward holding one action.

If A is poor we must finetune the tokenizer (which the repo supports as a
separate entrypoint). If B is blurry, dreamer4 is the wrong base entirely and
we pivot to from-scratch flow matching before spending real money.

    modal run scripts/tests/d4_probe.py::fetch      # pull the 0.77 GB of checkpoints once
    modal run scripts/tests/d4_probe.py::probe
"""

from __future__ import annotations

import modal

app = modal.App("d4-probe")

REPO = "https://github.com/nicklashansen/dreamer4"
COMMIT = "b8abafbf"  # pinned: last commit on main, 2026-07-09

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch==2.8.0", "torchvision==0.23.0", "numpy==1.24.4", "pillow==12.0.0",
        "aiohttp==3.13.3", "tensordict==0.10.0", "huggingface_hub", "einops",
    )
    .run_commands(
        f"git clone {REPO} /opt/dreamer4",
        f"cd /opt/dreamer4 && git checkout {COMMIT}",
    )
)

vol = modal.Volume.from_name("d4-ckpt", create_if_missing=True)
CK = "/ckpt"


@app.function(image=image, volumes={CK: vol}, timeout=1800)
def fetch():
    from huggingface_hub import hf_hub_download
    import shutil, os
    for f in ["tokenizer.pt", "dynamics.pt", "config.json"]:
        p = hf_hub_download("nicklashansen/dreamer4", f)
        shutil.copy(p, os.path.join(CK, f))
        print(f, os.path.getsize(os.path.join(CK, f)) / 1e6, "MB")
    vol.commit()


@app.function(image=image, volumes={CK: vol}, gpu="A10G", timeout=1800)
def probe():
    import sys, os, json, math
    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, "/opt/dreamer4/dreamer4")
    import interactive as I
    from model import temporal_patchify, temporal_unpatchify, pack_bottleneck_to_spatial

    frames_u8 = np.load(f"{CK}/probe_frames.npy")
    dev = torch.device("cuda")
    tok, ti = I.load_tokenizer_from_ckpt(f"{CK}/tokenizer.pt", dev)
    print("tokenizer info:", ti)
    H, W, C, patch = ti["H"], ti["W"], ti["C"], ti["patch"]

    # ---- A. tokenizer round trip on our pixel art ----
    x = torch.from_numpy(frames_u8).float().div(255).permute(0, 3, 1, 2)  # (N,C,H,W)
    x = x.to(dev)
    assert x.shape[-1] == W and x.shape[-2] == H, f"size {x.shape} vs tokenizer {H}x{W}"

    recons, psnrs = [], []
    for i in range(x.shape[0]):
        f = x[i:i + 1].unsqueeze(1)                       # (1,1,C,H,W)
        p = temporal_patchify(f, patch)
        z, _ = tok.encoder(p)
        pred = tok.decoder(z)
        rec = temporal_unpatchify(pred, H, W, C, patch)[0, 0].clamp(0, 1)
        mse = torch.mean((rec - x[i]) ** 2).item()
        psnrs.append(10 * math.log10(1.0 / max(mse, 1e-10)))
        recons.append((rec.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

    # side-by-side: original on top, reconstruction below
    N = len(recons)
    sheet = np.zeros((2 * H + 12, N * W + (N + 1) * 4, 3), np.uint8) + 18
    for i in range(N):
        xo = 4 + i * (W + 4)
        sheet[4:4 + H, xo:xo + W] = frames_u8[i]
        sheet[H + 8:H + 8 + H, xo:xo + W] = recons[i]
    Image.fromarray(sheet).save(f"{CK}/roundtrip.png")

    out = {"tokenizer": {k: str(v) for k, v in ti.items()},
           "roundtrip_psnr": [round(p, 2) for p in psnrs],
           "roundtrip_psnr_mean": round(float(np.mean(psnrs)), 2)}

    # ---- B. dynamics rollout sharpness ----
    try:
        pf = int(os.environ.get("PACKING_FACTOR", "2"))
        dyn, di = I.load_dynamics_from_ckpt(
            f"{CK}/dynamics.pt", device=dev,
            d_bottleneck=int(ti["d_bottleneck"]), n_latents=int(ti["n_latents"]),
            packing_factor=pf)
        print("dynamics info:", di)
        k_max, n_spatial = int(di["k_max"]), int(di["n_spatial"])
        sched = I.make_tau_schedule(k_max=k_max, schedule="finest")

        f0 = x[0:1].unsqueeze(1)
        z0, _ = tok.encoder(temporal_patchify(f0, patch))
        z0p = pack_bottleneck_to_spatial(z0, n_spatial=n_spatial, k=pf)[0, 0].float()

        past = z0p.unsqueeze(0).unsqueeze(0)          # (1,1,n_spatial,d)
        A = 16
        act = torch.zeros(1, 1, A, device=dev)
        mask = torch.zeros(A, device=dev); mask[:3] = 1.0
        roll = [frames_u8[0]]
        for t in range(12):
            a = torch.zeros(1, 1, A, device=dev)
            a[0, 0, 0] = 1.0                          # hold "move right"
            act = torch.cat([act, a], dim=1)
            zt = I.sample_one_timestep_packed(
                dyn, past_packed=past, k_max=k_max, sched=sched,
                actions=act, act_mask=mask, use_amp=True)
            past = torch.cat([past, zt.unsqueeze(1)], dim=1)
            fr = I.decode_single_packed_frame(
                tok.decoder, z_packed=zt[0], H=H, W=W, C=C, patch=patch,
                packing_factor=pf, d_bottleneck=int(ti["d_bottleneck"]))
            roll.append((fr.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

        cols = len(roll)
        rs = np.zeros((H + 8, cols * W + (cols + 1) * 4, 3), np.uint8) + 18
        for i, f in enumerate(roll):
            rs[4:4 + H, 4 + i * (W + 4):4 + i * (W + 4) + W] = f
        Image.fromarray(rs).save(f"{CK}/rollout.png")

        # sharpness: mean |Laplacian| -- a blurry frame has little high frequency
        def sharp(im):
            g = im.astype(np.float32).mean(-1)
            lap = (np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1]
                          - g[1:-1, :-2] - g[1:-1, 2:]))
            return float(lap.mean())
        out["sharpness_real_frame"] = round(sharp(frames_u8[0]), 2)
        out["sharpness_rollout"] = [round(sharp(f), 2) for f in roll[1:]]
        out["dynamics"] = {k: str(v) for k, v in di.items()}
    except Exception as e:
        out["dynamics_error"] = f"{type(e).__name__}: {e}"

    (open(f"{CK}/probe.json", "w")).write(json.dumps(out, indent=2))
    vol.commit()
    return out


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(probe.remote(), indent=2)[:3000])


@app.function(image=image, volumes={CK: vol}, gpu="A10G", timeout=1800)
def probe_ur():
    """Smoke-test the latent round trip -- the quantity `u_r` is built on.

    Training penalises PIXEL error (masked MSE + 0.2 * LPIPS). `u_r` is a
    different thing: ||z - Enc(Dec(z))||, how far a latent moves when decoded
    and re-encoded. Nothing optimises it, which is exactly why it can detect
    hallucination -- the training objective cannot game it.

    Measured three ways so the number has a scale:
      ours       our pixel art (out of distribution for this checkpoint)
      recon      the model's OWN reconstruction, re-fed (in-distribution)
      noise      uniform noise (a floor for "nothing like the training data")
    """
    import json, sys
    sys.path.insert(0, "/opt/dreamer4/dreamer4")
    import numpy as np, torch
    from model import temporal_patchify, temporal_unpatchify
    import interactive as I

    frames = np.load(f"{CK}/probe_frames.npy")
    dev = torch.device("cuda")
    tok, ti = I.load_tokenizer_from_ckpt(f"{CK}/tokenizer.pt", dev)
    H, W, C, patch = ti["H"], ti["W"], ti["C"], ti["patch"]
    tok.eval()

    def enc(img_bchw):
        z, _ = tok.encoder(temporal_patchify(img_bchw.unsqueeze(1), patch))
        return z

    def dec(z):
        return temporal_unpatchify(tok.decoder(z), H, W, C, patch)[:, 0].clamp(0, 1)

    def u_r(img_bchw):
        """||z - Enc(Dec(z))||, per frame, and relative to ||z||."""
        z = enc(img_bchw)
        z2 = enc(dec(z))
        d = (z - z2).flatten(1).norm(dim=1)
        n = z.flatten(1).norm(dim=1).clamp_min(1e-8)
        return d.cpu().numpy(), (d / n).cpu().numpy()

    x = torch.from_numpy(frames).float().div(255).permute(0, 3, 1, 2).to(dev)
    with torch.inference_mode():
        d_ours, r_ours = u_r(x)
        recon = dec(enc(x))
        d_rec, r_rec = u_r(recon)
        noise = torch.rand_like(x)
        d_noi, r_noi = u_r(noise)

    def stat(d, r):
        return {"u_r": round(float(d.mean()), 4), "u_r_norm": round(float(r.mean()), 4)}

    out = {"note": "u_r = ||z - Enc(Dec(z))|| in latent space; not the training loss",
           "latent_shape": list(enc(x[:1]).shape),
           "ours_pixel_art": stat(d_ours, r_ours),
           "model_own_reconstruction": stat(d_rec, r_rec),
           "uniform_noise": stat(d_noi, r_noi)}
    out["ratio_ours_over_own_recon"] = round(
        out["ours_pixel_art"]["u_r_norm"] / max(out["model_own_reconstruction"]["u_r_norm"], 1e-8), 3)
    open(f"{CK}/ur.json", "w").write(json.dumps(out, indent=2))
    vol.commit()
    return out


@app.local_entrypoint()
def ur():
    import json
    print(json.dumps(probe_ur.remote(), indent=2))
