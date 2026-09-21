"""aq method: the pixel-world model, trained on Modal.

aq owns the experiment record; Modal owns the compute. `fit` returns a JSON
document because aq stores checkpoints as JSON, so the weights stay on the
Modal volume and the checkpoint here is a pointer plus the numbers describing
the run. Right split for this project: the dev machine has 7.6 GB free, and
what deserves versioning is which run produced which score.

`recipe.stage` selects what gets fit:

    tokenizer  frames -> latent grid -> frames. Gated on char_acc, because
               whole-frame accuracy demonstrably passes a model containing no
               character at all (the fox is 1.3% of pixels; omitting it
               entirely still scores 0.987).
    dynamics   action-conditioned prediction over frozen tokenizer latents.
               Gated on ROLLOUT accuracy, not one-step error -- one-step can
               look excellent while multi-step rollouts drift into nonsense,
               and drift is what decides whether the thing is playable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

APP = "src/algo/pal_modal.py"
VOLUME = "d4-runs"


def _modal() -> str:
    exe = shutil.which("modal") or os.path.expanduser("~/.local/bin/modal")
    if not Path(exe).exists():
        raise SystemExit("modal CLI not found; add ~/.local/bin to PATH")
    return exe


def _run(entrypoint: str, train: Path, **kw) -> None:
    """Launch a detached Modal run. Results are read from the volume.

    --detach on the FUNCTION (not a local entrypoint) is what actually survives
    a dead client; a power cut cost us a 30-minute run before this.
    """
    cmd = [_modal(), "run", "--detach", f"{APP}::{entrypoint}"]
    for k, v in kw.items():
        if v is None or v == "":
            continue
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    subprocess.run(cmd, cwd=train, capture_output=True, text=True)


def _from_volume(run: str, name: str, tries: int = 480, every: int = 15):
    """Poll the volume for a run artefact. Long-running fits outlive the client."""
    import tempfile
    for _ in range(tries):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / name
            r = subprocess.run([_modal(), "volume", "get", VOLUME,
                                f"{run}/{name}", str(dest)],
                               capture_output=True, text=True)
            if r.returncode == 0 and dest.exists():
                return json.loads(dest.read_text())
        time.sleep(every)
    return None


def fit(src: Path, rec: dict) -> dict:
    train = Path(rec.get("_train") or ".")
    cfg = rec.get("train") or {}
    stage = rec.get("stage", "tokenizer")
    run = cfg.get("run") or f"{stage}_{time.strftime('%Y%m%d_%H%M%S')}"

    if stage == "tokenizer":
        _run("main", train, run=run, steps=cfg.get("steps", 1500),
             batch=cfg.get("batch_size", 32),
             char_weight=cfg.get("char_weight", 5.0),
             width=cfg.get("width", 64), z_ch=cfg.get("z_ch", 8),
             downs=cfg.get("downs", 3))
    elif stage == "dynamics":
        _run("dyn", train, run=run, steps=cfg.get("steps", 2000),
             batch=cfg.get("batch_size", 16), n_ctx=cfg.get("ctx", 8),
             horizon=cfg.get("horizon", 8),
             tok_run=cfg.get("tokenizer_run", "pal_v2"))
    else:
        raise SystemExit(f"unknown stage: {stage}")

    hist = _from_volume(run, "history.json")
    if not hist:
        raise SystemExit(f"{run} produced no history.json (still running, or failed)")
    last = hist[-1]
    return {"backend": "modal", "kind": f"pixel-world-{stage}", "stage": stage,
            "run": run, "ckpt": "latest.pt", "ckpt_volume": VOLUME,
            "steps": last.get("step"), "loss": last.get("loss"),
            "metrics": {k: v for k, v in last.items() if k not in ("step", "loss")},
            "n_evals": len(hist)}


def evaluate(model: dict, src: Path, rec: dict) -> tuple[float, int]:
    """Score the run's own final eval. aq compares it to recipe eval.min_score."""
    ev = rec.get("eval") or {}
    metric = ev.get("metric") or ("rollout_char_acc" if model.get("stage") == "dynamics"
                                  else "char_acc")
    m = model.get("metrics") or {}
    if metric not in m:
        raise SystemExit(f"metric {metric!r} not in run metrics: {sorted(m)}")
    return float(m[metric]), int(ev.get("n_frames", 256))


def write_inspect(train: Path, model: dict) -> str:
    m = model.get("metrics") or {}
    lines = [
        "# inspect", "",
        f"stage      : {model.get('stage')}",
        f"run        : {model.get('run')}  (volume {model.get('ckpt_volume')})",
        f"steps      : {model.get('steps')}   evals: {model.get('n_evals')}",
        f"loss       : {model.get('loss')}",
        "",
        "## metrics",
    ]
    lines += [f"{k:<22}: {v}" for k, v in sorted(m.items())]
    lines += [
        "",
        "char_acc is the gate, not whole-frame accuracy: the character is 1.3% of",
        "the frame, so a reconstruction omitting it entirely still scores 0.987.",
        "For dynamics the gate is ROLLOUT accuracy -- one-step error can look fine",
        "while multi-step rollouts drift.",
        "",
        "Look at the reconstruction image before believing any of these numbers.",
        "",
    ]
    rel = "artifacts/inspect.md"
    (train / rel).write_text("\n".join(lines), encoding="utf-8")
    return rel
