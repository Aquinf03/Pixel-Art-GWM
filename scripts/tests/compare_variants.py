"""Stack every sweep variant's reconstruction into one sheet.

The ranking table gives char_acc; this gives the picture. Both are needed --
whole-frame accuracy of 0.77 looked like progress while the fox was absent.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

VARIANTS = ["base", "detail5", "detail20", "latent32", "bneck64", "wide384"]
OUT = Path("artifacts/variants")


def fetch(name: str) -> Path | None:
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{name}.png"
    run = f"sw_{name}"
    ls = subprocess.run(["modal", "volume", "ls", "d4-runs", f"/{run}"],
                        capture_output=True, text=True)
    png = [l.split("/")[-1].strip() for l in ls.stdout.splitlines() if ".png" in l]
    if not png:
        return None
    r = subprocess.run(["modal", "volume", "get", "d4-runs",
                        f"{run}/{png[-1]}", str(dest), "--force"],
                       capture_output=True, text=True)
    return dest if dest.exists() else None


def main() -> None:
    got = [(n, fetch(n)) for n in VARIANTS]
    got = [(n, p) for n, p in got if p]
    if not got:
        print("no variant images on the volume yet")
        return
    imgs = [(n, np.asarray(Image.open(p).convert("RGB"))) for n, p in got]
    w = max(a.shape[1] for _, a in imgs)
    lab, pad = 22, 8
    h = sum(a.shape[0] + lab + pad for _, a in imgs) + pad
    sheet = np.full((h, w + 2 * pad, 3), 16, np.uint8)
    y = pad
    for _, a in imgs:
        sheet[y + lab:y + lab + a.shape[0], pad:pad + a.shape[1]] = a
        y += a.shape[0] + lab + pad
    img = Image.fromarray(sheet)
    d = ImageDraw.Draw(img)
    y = pad
    for n, a in imgs:
        d.text((pad + 2, y + 6), f"{n}  (top: truth   bottom: reconstruction)",
               fill=(240, 200, 90))
        y += a.shape[0] + lab + pad
    dest = Path("artifacts/variant_comparison.png")
    img.save(dest)
    print(f"wrote {dest}  ({len(imgs)} variants, {img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
