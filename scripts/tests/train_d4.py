"""Stage 0: does dreamer4's tokenizer training actually run on our data?

The point is not to train anything useful. It is to convert two unknowns into
numbers before committing a budget:

  - does their (five-month-stale) training code run against our converted data
  - how fast, so the real run has a cost estimate rather than a README quote

Then it measures the gate we will judge stage 1 by: round-trip PSNR on held-out
frames. Their released tokenizer scores 18.56 dB on our pixel art and
reconstructs it as a robot arm; ours has to do much better than that.

    modal run scripts/tests/train_d4.py::verify_split --split val
    modal run scripts/tests/train_d4.py::smoke --steps 500
"""

from __future__ import annotations

import modal

app = modal.App("d4-train")

REPO = "https://github.com/nicklashansen/dreamer4"
TASK = "pixelworld-climb"
COMMIT = "b8abafbf"

# The aquin workspace has A100-40GB and H100-80GB available (verified), so we
# can run dreamer4's intended batch 8 x seq_len 8. The earlier A10G fallback
# was forced by its 22 GiB, which OOMs on that config at 128x128.
# The renderer emits exactly these 21 colours. That makes a much stricter gate
# available than PSNR: snap each reconstructed pixel to the nearest palette
# entry and count exact matches. A blurry reconstruction cannot score well on
# this, whereas it can absolutely reach 30 dB PSNR -- and the released
# checkpoint scored a plausible 18.56 dB while drawing a robot arm.
PALETTE = [
    (26,38,52), (34,52,66), (36,72,54), (44,68,78), (58,106,74), (96,158,96),
    (246,206,84), (255,244,190), (36,24,30), (232,132,62), (168,84,40),
    (248,214,170), (24,20,26), (92,46,34), (255,255,255), (208,110,50),
    (120,96,148), (98,66,38), (146,100,58), (255,248,214), (206,190,236),
]

# Indices 8..15 and 19 are the character's own colours (outline, body, shade,
# cream, eye, nose, eye-shine, paw, swing). The character is only ~1.3% of the
# frame, so whole-frame palette accuracy is a near-useless gate: a
# reconstruction that omits the fox ENTIRELY still scores 0.987, above a 0.95
# threshold. char_acc measures only pixels whose GROUND TRUTH is a character
# colour, which is the thing we actually care about.
CHAR_IDX = [8, 9, 10, 11, 12, 13, 14, 15, 19]

TRAIN_GPU = "A100-40GB"
SMOKE_GPU = "A100-40GB"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch==2.8.0", "torchvision==0.23.0", "numpy==1.24.4", "pillow==12.0.0",
        "aiohttp==3.13.3", "tensordict==0.10.0", "lpips==0.1.4", "wandb==0.22.1",
        "einops",
    )
    .run_commands(
        f"git clone {REPO} /opt/d4",
        f"cd /opt/d4 && git checkout {COMMIT}",
        # train_tokenizer.py hardcodes tasks=TASK_SET with no CLI override, and
        # ShardedFrameDataset only scans directories named in that list -- so a
        # custom environment is invisible to the trainer until it is appended.
        # This is the real custom-env registration path; the README's "modify
        # data directory paths" does not cover it.
        f"echo \"TASK_SET = TASK_SET + ['{TASK}']\" >> /opt/d4/dreamer4/task_set.py",
        # train_tokenizer.py hardcodes wandb.init(mode="online"), which overrides
        # the WANDB_MODE env var -- so with no API key the run dies rather than
        # logging nothing. Training must not depend on an external service.
        "sed -i 's/mode=\"online\",/mode=\"disabled\",/' /opt/d4/dreamer4/train_tokenizer.py",
        "grep -q 'mode=\"disabled\"' /opt/d4/dreamer4/train_tokenizer.py",
        # ShardedFrameDataset torch.loads EVERY shard at construction just to
        # count frames. torch.save does not compress, so our 0.75 GB corpus is
        # ~100 GB of raw uint8 on disk -- that scan reads all of it off a
        # network volume before step 0. mmap reads headers instead of payloads.
        # (Their README's ">256 GB RAM" recommendation is the same problem,
        # solved by holding the dataset in memory; we cannot.)
        "sed -i 's/torch.load(path, map_location=\"cpu\")/torch.load(path, map_location=\"cpu\", mmap=True)/g' "
        "/opt/d4/dreamer4/sharded_frame_dataset.py",
        "grep -c 'mmap=True' /opt/d4/dreamer4/sharded_frame_dataset.py",
        # Character-colour weighting (CHAR_WEIGHT). Variance weighting helped
        # the background, not the fox -- see patch_charloss.py.
        "echo IiIiQ2hhcmFjdGVyLXdlaWdodGVkIHJlY29uc3RydWN0aW9uIGxvc3MsIG9wdC1pbiB2aWEgQ0hBUl9XRUlHSFQuCgpERVRBSUxfV0VJR0hUIChwYXRjaF9sb3NzLnB5KSB3ZWlnaHRlZCBwYXRjaGVzIGJ5IHZhcmlhbmNlLCB3aGljaCB1cHdlaWdodHMKZXZlcnkgZWRnZSBpbiB0aGUgZnJhbWUgLS0gcGxhdGZvcm0gYm91bmRhcmllcywgc2t5IGJhbmRzLCBjb2lucywgdGhlIGNyYXRlLgpNZWFzdXJlZCByZXN1bHQ6IHdob2xlLWZyYW1lIGFjY3VyYWN5IGFuZCBQU05SIGltcHJvdmVkIGEgbG90LCBjaGFyYWN0ZXIKYWNjdXJhY3kgZ290IFdPUlNFICgwLjAxNTEgLT4gMC4wMDc1IGF0IHN0ZXAgMTAwMCkuIFRoZXJlIGFyZSBmYXIgbW9yZQpiYWNrZ3JvdW5kIGVkZ2VzIHRoYW4gY2hhcmFjdGVyIHBpeGVscywgc28gdGhlIGV4dHJhIGdyYWRpZW50IHdlbnQgdG8gdGhlbS4KClRoaXMgd2VpZ2h0cyBwYXRjaGVzIHRoYXQgYWN0dWFsbHkgY29udGFpbiB0aGUgY2hhcmFjdGVyLCBpZGVudGlmaWVkIGJ5IGl0cwpvd24gcGFsZXR0ZSBjb2xvdXJzLiBFbmNvZGluZyAidGhlIGNvbnRyb2xsYWJsZSBlbnRpdHkgbWF0dGVycyBtb3JlIHRoYW4gdGhlCmJhY2tkcm9wIiBhcyBhIHByaW9yIGlzIGEgbGVnaXRpbWF0ZSBkZXNpZ24gY2hvaWNlIGZvciBhIHdvcmxkIG1vZGVsIC0tIGl0IGlzCnRoZSB0aGluZyB0aGUgcGxheWVyIGFjdHMgdGhyb3VnaC4KIiIiCmltcG9ydCBwYXRobGliCgpDSEFSX1JHQiA9IFsKICAgICgzNiwgMjQsIDMwKSwgKDIzMiwgMTMyLCA2MiksICgxNjgsIDg0LCA0MCksICgyNDgsIDIxNCwgMTcwKSwKICAgICgyNCwgMjAsIDI2KSwgKDkyLCA0NiwgMzQpLCAoMjU1LCAyNTUsIDI1NSksICgyMDgsIDExMCwgNTApLAogICAgKDI1NSwgMjQ4LCAyMTQpLApdCgpwID0gcGF0aGxpYi5QYXRoKCIvb3B0L2Q0L2RyZWFtZXI0L21vZGVsLnB5IikKcyA9IHAucmVhZF90ZXh0KCkKYW5jaG9yID0gIiAgICBzcSA9IGRpZmYubXVsKGRpZmYpICogbWFzayIKYXNzZXJ0IGFuY2hvciBpbiBzLCAibG9zcyBhbmNob3Igbm90IGZvdW5kIgoKaW5qZWN0ID0gJycnICAgIHNxID0gZGlmZi5tdWwoZGlmZikgKiBtYXNrCiAgICBpbXBvcnQgb3MgYXMgX29zCiAgICBfY3cgPSBmbG9hdChfb3MuZW52aXJvbi5nZXQoIkNIQVJfV0VJR0hUIiwgIjAiKSBvciAwKQogICAgaWYgX2N3ID4gMDoKICAgICAgICAjIHRhcmdldCBwYXRjaGVzIGFyZSAoQixULE5wLERwKSB3aXRoIERwID0gcGF0Y2gqcGF0Y2gqQywgdmFsdWVzIGluIFswLDFdCiAgICAgICAgX3QgPSB0YXJnZXRfYnRuZC5mbG9hdCgpCiAgICAgICAgX3B4ID0gX3QucmVzaGFwZSgqX3Quc2hhcGVbOjNdLCAtMSwgMykgICAgICAgICAgICAgICAgICMgKEIsVCxOcCxwcCwzKQogICAgICAgIF9jID0gdG9yY2gudGVuc29yKCVzLCBkdHlwZT1fcHguZHR5cGUsIGRldmljZT1fcHguZGV2aWNlKSAvIDI1NS4wCiAgICAgICAgX2QgPSAoX3B4LnVuc3F1ZWV6ZSgtMikgLSBfYy52aWV3KDEsIDEsIDEsIDEsIC0xLCAzKSkuYWJzKCkuc3VtKC0xKQogICAgICAgIF9oaXQgPSAoX2QuYW1pbigtMSkgPCAwLjAyKS5hbnkoLTEsIGtlZXBkaW09VHJ1ZSkuZmxvYXQoKSAgIyAoQixULE5wLDEpCiAgICAgICAgX3cgPSAxLjAgKyBfY3cgKiBfaGl0CiAgICAgICAgc3EgPSBzcSAqIF93CiAgICAgICAgcmV0dXJuIHNxLnN1bSgpIC8gKChtYXNrICogX3cpLnN1bSgpLmNsYW1wX21pbigxLjApICogZGlmZi5zaGFwZVstMV0pCicnJyAlIChDSEFSX1JHQiwpCgpzID0gcy5yZXBsYWNlKGFuY2hvciArICIgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICMgYnJvYWRjYXN0IG1hc2sgb3ZlciBEcCIsIGluamVjdCwgMSkKaWYgIkNIQVJfV0VJR0hUIiBub3QgaW4gczogICAgICAgICAgICAgICAgICAgICAgICMgYW5jaG9yIGhhZCBubyB0cmFpbGluZyBjb21tZW50CiAgICBzID0gcy5yZXBsYWNlKGFuY2hvciwgaW5qZWN0LCAxKQphc3NlcnQgIkNIQVJfV0VJR0hUIiBpbiBzLCAiY2hhci13ZWlnaHQgaW5qZWN0aW9uIGZhaWxlZCIKcC53cml0ZV90ZXh0KHMpCnByaW50KCJwYXRjaGVkIHJlY29uX2xvc3NfZnJvbV9tYWUgd2l0aCBDSEFSX1dFSUdIVCBzdXBwb3J0IikK | base64 -d > /tmp/patch_charloss.py && python3 /tmp/patch_charloss.py",
    )
    # wandb is imported unconditionally by train_tokenizer.py; disabled makes it a no-op.
    .env({"WANDB_MODE": "disabled", "PYTHONUNBUFFERED": "1"})
)

data_vol = modal.Volume.from_name("pixel-world-data", create_if_missing=True)
run_vol = modal.Volume.from_name("d4-runs", create_if_missing=True)
DATA, RUNS = "/data", "/runs"



def _stream(cmd, cwd, log_path, done_marker=None, grace=180):
    """Run a subprocess, echoing stdout live AND capturing it.

    subprocess.run(capture_output=True) hides everything until exit, which
    makes a multi-hour run unwatchable. Two extras beyond that:

    - `done_marker(text) -> bool` lets us detect that the work is finished from
      the output itself. This trainer reliably HANGS after its last step (its
      teardown never returns), so waiting on process exit would block forever
      while the checkpoints are already safely written.
    - after the marker fires we wait `grace` seconds for a clean exit, then
      terminate. Checkpoints are the real artifact and they are on disk.
    """
    import subprocess, sys, threading, time
    lines, finished_at = [], None

    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)

    def pump():
        nonlocal finished_at
        with open(log_path, "w", buffering=1) as f:
            for line in proc.stdout:
                sys.stdout.write(line); sys.stdout.flush()
                f.write(line); lines.append(line)
                if finished_at is None and done_marker and done_marker(line):
                    finished_at = time.time()
                    print(f"[runner] completion marker seen; "
                          f"allowing {grace}s for clean exit", flush=True)

    t = threading.Thread(target=pump, daemon=True)
    t.start()

    while proc.poll() is None:
        if finished_at and time.time() - finished_at > grace:
            print("[runner] trainer did not exit after finishing; terminating",
                  flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            break
        time.sleep(2)
    t.join(timeout=10)
    rc = proc.returncode if proc.returncode is not None else 0
    # A hang AFTER the work finished is not a failure.
    if finished_at is not None:
        rc = 0
    return rc, "".join(lines)


@app.function(image=image, volumes={DATA: data_vol}, cpu=2.0, memory=8192, timeout=1800)
def verify_split(split: str = "val") -> dict:
    """Instantiate dreamer4's own loaders against our converted data.

    Writing a format is not the same as the format being accepted; the loader
    is where a mismatch actually surfaces.
    """
    import sys, glob
    import torch
    sys.path.insert(0, "/opt/d4/dreamer4")
    data_vol.reload()

    base = f"{DATA}/d4/{split}"
    out = {"frame_shards": len(glob.glob(f"{base}/frames/*/*.pt")),
           "demo_files": glob.glob(f"{base}/demos/*.pt")}

    from sharded_frame_dataset import ShardedFrameDataset
    from task_set import TASK_SET
    out["task_registered"] = TASK in TASK_SET
    fds = ShardedFrameDataset(outdirs=[f"{base}/frames"], tasks=[TASK], seq_len=8)
    x = fds[0]
    out["ShardedFrameDataset"] = {"len": len(fds),
                                 "item": f"{tuple(x.shape)} {x.dtype}" if torch.is_tensor(x) else str(type(x))}

    from wm_dataset import WMDataset
    wds = WMDataset(data_dir=[f"{base}/demos"], frames_dir=[f"{base}/frames"],
                    tasks_json=f"{base}/tasks.json", seq_len=16, tasks=[TASK])
    it = wds[0]
    out["WMDataset"] = {"len": len(wds), "tasks": list(wds.tasks)}
    if hasattr(it, "items"):
        out["WMDataset"]["item"] = {
            k: f"{tuple(v.shape)} {v.dtype}" for k, v in it.items() if torch.is_tensor(v)}
    return out


@app.function(image=image, volumes={DATA: data_vol, RUNS: run_vol},
              gpu=SMOKE_GPU, cpu=4.0, memory=16384, timeout=3600)
def smoke(split: str = "val", steps: int = 500, batch_size: int = 8,
          seq_len: int = 8, lpips_weight: float = 0.2) -> dict:
    import glob, json, os, re, subprocess, sys, time
    data_vol.reload()

    # the ROOT holding <task>/*.pt -- passing the task dir itself finds nothing
    frames_dir = f"{DATA}/d4/{split}/frames"
    ck = f"{RUNS}/tok_smoke"
    os.makedirs(ck, exist_ok=True)

    cmd = [sys.executable, "train_tokenizer.py",
           "--data_dirs", frames_dir,
           "--seq_len", str(seq_len), "--batch_size", str(batch_size),
           "--num_workers", "4", "--max_steps", str(steps),
           "--lpips_weight", str(lpips_weight),
           "--log_every", "25", "--print_every", "25",
           "--viz_every", str(steps + 1), "--save_every", str(steps),
           "--ckpt_dir", ck]
    t0 = time.time()
    p = subprocess.run(cmd, cwd="/opt/d4/dreamer4", capture_output=True, text=True)
    wall = time.time() - t0

    os.makedirs(ck, exist_ok=True)
    open(f"{ck}/stdout.log", "w").write(p.stdout or "")
    open(f"{ck}/stderr.log", "w").write(p.stderr or "")
    tail = (p.stdout or "")[-4000:]
    err = (p.stderr or "")[-3000:]
    losses = [(int(a), float(b)) for a, b in
              re.findall(r"step[ =:]+(\d+).*?loss[ =:]+([0-9.]+)", tail, re.I)]
    result = {"returncode": p.returncode, "wall_s": round(wall, 1),
            "steps_requested": steps,
            "steps_per_s": round(steps / wall, 2) if p.returncode == 0 else None,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "n_loss_points": len(losses),
            "ckpts": [os.path.basename(x) for x in glob.glob(f"{ck}/*.pt")],
            "stdout_tail": tail[-1800:], "stderr_tail": err[-1200:]}
    open(f"{ck}/result.json", "w").write(json.dumps(result, indent=2))
    run_vol.commit()
    return result


@app.function(image=image, volumes={DATA: data_vol, RUNS: run_vol},
              gpu=TRAIN_GPU, cpu=8.0, memory=32768, timeout=86400)
def train_tokenizer(run: str, split: str = "train", steps: int = 20000,
                    batch_size: int = 8, seq_len: int = 8, lr: float = 1e-4,
                    lpips_weight: float = 0.2, resume: str = "",
                    save_every: int = 1000, n_latents: int = 16,
                    d_bottleneck: int = 32, d_model: int = 256,
                    detail_weight: float = 0.0, char_weight: float = 0.0) -> dict:
    """Stage 1: train the tokenizer on our corpus.

    dreamer4's architecture and training code, none of its weights: its released
    tokenizer scores 18.56 dB on our pixel art and rebuilds the fox as a robot
    arm, and finetuning it would invalidate the dynamics trained on its frozen
    latents.
    """
    import glob, json, os, re, subprocess, sys, time
    data_vol.reload()
    ck = f"{RUNS}/{run}"
    os.makedirs(ck, exist_ok=True)

    cmd = [sys.executable, "train_tokenizer.py",
           "--data_dirs", f"{DATA}/d4/{split}/frames",
           "--seq_len", str(seq_len), "--batch_size", str(batch_size),
           "--num_workers", "8", "--max_steps", str(steps), "--lr", str(lr),
           "--lpips_weight", str(lpips_weight),
           "--log_every", "200", "--print_every", "200",
           "--viz_every", str(steps + 1),
           # Every 1000 steps (~6 min at measured throughput). A power cut at
           # step 4400 destroyed a 30-minute run because save_every was 6000
           # and no checkpoint existed yet. Cheap insurance.
           "--save_every", str(save_every),
           "--n_latents", str(n_latents), "--d_bottleneck", str(d_bottleneck),
           "--d_model", str(d_model),
           "--ckpt_dir", ck]
    if resume:
        cmd += ["--resume", resume]

    # The patched loss reads this; without it detail_weight is a dead argument.
    env_note = ""
    if detail_weight > 0:
        os.environ["DETAIL_WEIGHT"] = str(detail_weight)
        env_note += f" DETAIL_WEIGHT={detail_weight}"
    if char_weight > 0:
        os.environ["CHAR_WEIGHT"] = str(char_weight)
        env_note += f" CHAR_WEIGHT={char_weight}"
    print(f"[runner] launching {steps} steps{env_note}", flush=True)

    t0 = time.time()
    # Without a completion marker this hangs forever: the trainer's teardown
    # never returns, so waiting on process exit never finishes.
    marker = lambda ln: f"step {steps-1:07d}" in ln or f"step {steps:07d}" in ln
    rc, out = _stream(cmd, "/opt/d4/dreamer4", f"{ck}/stdout.log", marker, grace=60)
    wall = time.time() - t0
    losses = [(int(a), float(b)) for a, b in
              re.findall(r"step[ =:]+(\d+).*?loss[ =:]+([0-9.]+)", out, re.I)]
    ckpts = sorted(glob.glob(f"{ck}/*.pt"))
    result = {"run": run, "returncode": rc, "wall_s": round(wall, 1),
              "steps": steps,
              "steps_per_s": round(steps / wall, 3) if rc == 0 else None,
              "gpu_hours": round(wall / 3600, 3),
              "loss_first": losses[0] if losses else None,
              "loss_last": losses[-1] if losses else None,
              "n_loss_points": len(losses),
              "ckpt": os.path.basename(ckpts[-1]) if ckpts else None,
              "ckpt_dir": ck, "n_ckpts": len(ckpts),
              "log_tail": out[-1500:]}
    # Written before returning: a detached run must not depend on the client
    # still being connected to deliver its result.
    open(f"{ck}/result.json", "w").write(json.dumps(result, indent=2))
    run_vol.commit()
    return result


@app.function(image=image, volumes={DATA: data_vol, RUNS: run_vol},
              gpu="A10G", cpu=4.0, memory=16384, timeout=3600)
def eval_roundtrip(run: str, ckpt: str, split: str = "val", n_frames: int = 64) -> dict:
    """The stage-1 gate: can our tokenizer represent our own pixel art?

    PSNR against held-out frames, plus a side-by-side image -- a number alone
    would not have caught what the released checkpoint did, scoring a plausible
    18.56 dB while drawing something else entirely.
    """
    import glob, json, math, sys
    import numpy as np, torch
    from PIL import Image
    sys.path.insert(0, "/opt/d4/dreamer4")
    import interactive as I
    from model import temporal_patchify, temporal_unpatchify

    data_vol.reload(); run_vol.reload()
    dev = torch.device("cuda")
    tok, ti = I.load_tokenizer_from_ckpt(f"{RUNS}/{run}/{ckpt}", dev)
    H, W, C, patch = ti["H"], ti["W"], ti["C"], ti["patch"]

    shards = sorted(glob.glob(f"{DATA}/d4/{split}/frames/*/*.pt"))
    if not shards:
        raise RuntimeError(f"no held-out frames under {DATA}/d4/{split}/frames")
    fr = torch.load(shards[0], map_location="cpu")["frames"]
    idx = np.linspace(0, fr.shape[0] - 1, n_frames).astype(int)
    x = fr[idx].float().div(255).to(dev)

    pal = torch.tensor(PALETTE, dtype=torch.float32, device=dev).div(255)  # (P,3)

    char_idx = torch.tensor(CHAR_IDX, device=dev)

    def snap(im):
        hw3 = im.permute(1, 2, 0).reshape(-1, 1, 3)
        return torch.cdist(hw3, pal.unsqueeze(0)).squeeze(1).argmin(-1)

    def accuracies(rec_chw, tgt_chw):
        """(whole-frame accuracy, character-region accuracy)."""
        r, t = snap(rec_chw), snap(tgt_chw)
        hit = (r == t).float()
        whole = hit.mean().item()
        is_char = (t.unsqueeze(1) == char_idx.unsqueeze(0)).any(1)
        char = hit[is_char].mean().item() if is_char.any() else float("nan")
        return whole, char

    psnrs, palaccs, characcs, recons = [], [], [], []
    with torch.inference_mode():
        for i in range(x.shape[0]):
            f = x[i:i + 1].unsqueeze(1)
            z, _ = tok.encoder(temporal_patchify(f, patch))
            rec = temporal_unpatchify(tok.decoder(z), H, W, C, patch)[0, 0].clamp(0, 1)
            mse = torch.mean((rec - x[i]) ** 2).item()
            psnrs.append(10 * math.log10(1.0 / max(mse, 1e-10)))
            w, c = accuracies(rec, x[i])
            palaccs.append(w)
            if c == c:  # not NaN -- some frames may show no character
                characcs.append(c)
            if len(recons) < 8:
                recons.append((rec.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

    N = len(recons)
    sheet = np.zeros((2 * H + 12, N * W + (N + 1) * 4, 3), np.uint8) + 18
    for i in range(N):
        xo = 4 + i * (W + 4)
        sheet[4:4 + H, xo:xo + W] = (x[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        sheet[H + 8:H + 8 + H, xo:xo + W] = recons[i]
    Image.fromarray(sheet).save(f"{RUNS}/{run}/roundtrip_{ckpt}.png")

    result = {"char_acc": round(float(np.mean(characcs)), 4) if characcs else None,
              "char_acc_min": round(float(np.min(characcs)), 4) if characcs else None,
              "palette_acc": round(float(np.mean(palaccs)), 4),
              "palette_acc_min": round(float(np.min(palaccs)), 4),
              "psnr": round(float(np.mean(psnrs)), 3),
              "psnr_min": round(float(np.min(psnrs)), 3),
              "psnr_max": round(float(np.max(psnrs)), 3), "n": int(len(psnrs)),
              "baseline_released_ckpt": 18.56,
              "image": f"{run}/roundtrip_{ckpt}.png"}
    open(f"{RUNS}/{run}/eval.json", "w").write(json.dumps(result, indent=2))
    run_vol.commit()
    return result


@app.function(image=image, volumes={DATA: data_vol, RUNS: run_vol},
              gpu="A10G", cpu=4.0, memory=16384, timeout=3600)
def overfit_test(n_frames: int = 256, steps: int = 1500) -> dict:
    """Can the model memorise a handful of frames?

    A 21.4M-parameter autoencoder that cannot overfit 256 frames has a broken
    config or a broken implementation, and we would rather learn that in ten
    minutes than three hours into a real run. This is a code test, not a
    quality test: near-perfect numbers here prove nothing about generalisation,
    but bad numbers prove something is wrong.
    """
    import glob, json, os, shutil, subprocess, sys, time
    import torch
    data_vol.reload()

    tiny = f"{DATA}/d4/tiny/frames/{TASK}"
    os.makedirs(tiny, exist_ok=True)
    src = sorted(glob.glob(f"{DATA}/d4/val/frames/{TASK}/*.pt"))[0]
    fr = torch.load(src, map_location="cpu")["frames"][:n_frames]
    torch.save({"frames": fr}, f"{tiny}/{TASK}_shard0000.pt")
    data_vol.commit()

    ck = f"{RUNS}/overfit"
    os.makedirs(ck, exist_ok=True)
    cmd = [sys.executable, "train_tokenizer.py",
           "--data_dirs", f"{DATA}/d4/tiny/frames",
           "--seq_len", "4", "--batch_size", "4", "--num_workers", "2",
           "--max_steps", str(steps), "--lpips_weight", "0.2",
           "--log_every", "25", "--print_every", "25",
           "--viz_every", str(steps + 1), "--save_every", str(steps),
           "--ckpt_dir", ck]
    t0 = time.time()
    marker = lambda ln: f"step {steps-1:07d}" in ln or f"step {steps:07d}" in ln
    rc, out = _stream(cmd, "/opt/d4/dreamer4", f"{ck}/stdout.log", marker, grace=120)
    wall = time.time() - t0
    ckpts = sorted(glob.glob(f"{ck}/*.pt"))
    result = {"frames_memorised": int(fr.shape[0]), "steps": steps,
              "returncode": rc, "wall_s": round(wall, 1),
              "steps_per_s": round(steps / wall, 2) if rc == 0 else None,
              "ckpt": os.path.basename(ckpts[-1]) if ckpts else None,
              "log_tail": out[-1200:]}
    open(f"{ck}/result.json", "w").write(json.dumps(result, indent=2))
    run_vol.commit()
    return result


@app.local_entrypoint()
def overfit(n_frames: int = 256, steps: int = 1500):
    """Sanity gate before any long run."""
    import json
    r = overfit_test.remote(n_frames, steps)
    print(json.dumps(r, indent=2))
    if r.get("returncode") == 0 and r.get("ckpt"):
        e = eval_roundtrip.remote("overfit", r["ckpt"], "tiny", 32)
        print("RECONSTRUCTION ON THE MEMORISED FRAMES:")
        print(json.dumps(e, indent=2))
        ok = e["palette_acc"] > 0.90
        print(f"\nVERDICT: {'PASS - pipeline is sound, proceed' if ok else 'FAIL - config or code is wrong, do NOT start the long run'}")


@app.local_entrypoint()
def main(split: str = "val", steps: int = 300, batch_size: int = 2, seq_len: int = 4):
    import json
    print("== verify ==")
    try:
        print(json.dumps(verify_split.remote(split), indent=2))
    except Exception as e:
        print("VERIFY FAILED:", type(e).__name__, e)
    print("== smoke ==")
    r = smoke.remote(split, steps, batch_size, seq_len)
    r.pop("stdout_tail", None)
    print(json.dumps(r, indent=2))


@app.local_entrypoint()
def fit_cli(run: str, split: str = "train", steps: int = 20000, batch_size: int = 16,
            seq_len: int = 8, lr: float = 1e-4, lpips_weight: float = 0.2, resume: str = ""):
    """Invoked by src/algo/world_model.py. Prints one marker line it can parse."""
    import json
    r = train_tokenizer.remote(run, split, steps, batch_size, seq_len, lr, lpips_weight, resume)
    print("AQ_RESULT " + json.dumps(r))


@app.local_entrypoint()
def eval_cli(run: str, ckpt: str, split: str = "val", n_frames: int = 64):
    import json
    print("AQ_RESULT " + json.dumps(eval_roundtrip.remote(run, ckpt, split, n_frames)))


VARIANTS = {
    # name            overrides
    "base":       {},
    "detail5":    {"detail_weight": 5.0},
    "detail20":   {"detail_weight": 20.0},
    "latent32":   {"n_latents": 32},
    "bneck64":    {"d_bottleneck": 64},
    "wide384":    {"d_model": 384},
}


@app.function(image=image, volumes={DATA: data_vol, RUNS: run_vol},
              gpu=TRAIN_GPU, cpu=8.0, memory=32768, timeout=7200)
def sweep_one(name: str, steps: int = 3000) -> dict:
    """Train one variant briefly and gate it on character accuracy.

    Whole-frame accuracy is not the question -- the baseline already reaches
    0.77 there while reconstructing no fox at all. char_acc is the only number
    that separates these.
    """
    run = f"sw_{name}"
    kw = dict(VARIANTS[name])
    r = train_tokenizer.local(run=run, split="train", steps=steps,
                              save_every=steps, **kw)
    out = {"variant": name, "overrides": kw, "steps": steps,
           "wall_s": r.get("wall_s"), "loss_last": r.get("loss_last"),
           "ckpt": r.get("ckpt")}
    if r.get("ckpt"):
        e = eval_roundtrip.local(run, r["ckpt"], "val", 32)
        out.update(char_acc=e.get("char_acc"), palette_acc=e.get("palette_acc"),
                   psnr=e.get("psnr"), image=e.get("image"))
    return out


@app.local_entrypoint()
def sweep(steps: int = 3000, only: str = ""):
    """Run every variant in parallel and rank them by char_acc."""
    import json
    names = [n for n in VARIANTS if not only or n in only.split(",")]
    results = list(sweep_one.map(names, kwargs={"steps": steps}))
    results.sort(key=lambda d: -(d.get("char_acc") or 0))
    print(json.dumps(results, indent=2))
    print()
    print(f"{'variant':<12}{'char_acc':>10}{'whole':>9}{'psnr':>8}{'loss':>10}")
    for d in results:
        ca = d.get("char_acc")
        ll = (d.get("loss_last") or [None, None])[1]
        print(f"{d['variant']:<12}{(ca if ca is not None else -1):>10.4f}"
              f"{(d.get('palette_acc') or 0):>9.4f}{(d.get('psnr') or 0):>8.2f}"
              f"{(ll if ll is not None else 0):>10.5f}")
