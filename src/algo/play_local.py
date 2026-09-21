"""Play the world model on this machine. No cloud, no network.

The whole model is 4.4M parameters / 18 MB, so it runs at ~100 fps on Apple
silicon and ~27 fps on CPU alone -- both above the 15 fps the environment was
recorded at. Nothing you see after the first frame is real; every pixel is the
model's own prediction fed back into itself.

    python src/algo/play_local.py --tok artifacts/tokenizer.pt --dyn artifacts/dynamics.pt

Keys: arrows move / jump / crouch, space = use, R = reset, Q = quit.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "cocoa")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer"))

import numpy as np
import pygame
import torch

from dynamics import FlowDynamics, sample_next
from model import PaletteTokenizer, palette_tensor, to_rgb

SCALE = 5
KEYS = {pygame.K_LEFT: "left", pygame.K_RIGHT: "right", pygame.K_UP: "jump",
        pygame.K_DOWN: "crouch", pygame.K_SPACE: "use"}
VEC = {"idle": (0, 0, 0), "left": (-1, 0, 0), "right": (1, 0, 0),
       "jump": (0, 1, 0), "crouch": (0, -1, 0), "use": (0, 0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", default="/tmp/tok.pt")
    ap.add_argument("--dyn", default="/tmp/dyn.pt")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=4, help="denoising steps per frame")
    ap.add_argument("--seed-frames", default="artifacts/video.npz")
    a = ap.parse_args()

    dev = torch.device(a.device)
    tck = torch.load(a.tok, map_location="cpu", weights_only=False)
    ta = tck["args"]
    tok = PaletteTokenizer(ta["width"], ta["z_ch"], ta["downs"]).to(dev)
    tok.load_state_dict(tck["model"]); tok.eval()
    dck = torch.load(a.dyn, map_location="cpu", weights_only=False)
    da = dck["args"]
    dyn = FlowDynamics(ta["z_ch"], da["ctx"], 16, da["width"], da["depth"]).to(dev)
    dyn.load_state_dict(dck["model"]); dyn.eval()
    pal = palette_tensor(dev)
    ctx = da["ctx"]
    n_par = sum(p.numel() for p in tok.parameters()) + sum(p.numel() for p in dyn.parameters())

    ctx_frames = np.load(a.seed_frames)["context"]          # real frames, only to start

    def fresh_past():
        x = torch.from_numpy(ctx_frames[-ctx:]).to(dev).permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad():
            return tok.encoder(x).unsqueeze(0)

    pygame.init()
    W = H = 128
    screen = pygame.display.set_mode((W * SCALE, H * SCALE + 40))
    pygame.display.set_caption(f"Pixel World -- imagined locally ({n_par:,} params, {a.device})")
    font = pygame.font.SysFont("menlo", 13) if pygame.font.get_init() else None
    clock = pygame.time.Clock()

    past = fresh_past()
    frames, t0, fps = 0, time.time(), 0.0
    running = True
    while running:
        act = "idle"
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_q:
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                past, frames = fresh_past(), 0
        held = pygame.key.get_pressed()
        for k, name in KEYS.items():
            if held[k]:
                act = name
                break

        if act != "idle":                       # the world only moves when you do
            v = VEC[act]
            a_t = torch.zeros(1, 16, device=dev)
            a_t[0, 0], a_t[0, 1], a_t[0, 2] = v
            with torch.no_grad():
                z = sample_next(dyn, past, a_t, steps=a.steps)
                past = torch.cat([past[:, 1:], z.unsqueeze(1)], 1)
                img = to_rgb(tok.decoder(z).argmax(1), pal)[0]
            arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            main.last = arr
            frames += 1
            if frames % 10 == 0:
                fps = 10 / (time.time() - t0); t0 = time.time()

        arr = getattr(main, "last", None)
        if arr is None:
            with torch.no_grad():
                arr = (to_rgb(tok.decoder(past[:, -1]).argmax(1), pal)[0]
                       .permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            main.last = arr
        surf = pygame.surfarray.make_surface(np.transpose(arr, (1, 0, 2)))
        screen.fill((13, 18, 25))
        screen.blit(pygame.transform.scale(surf, (W * SCALE, H * SCALE)), (0, 40))
        if font:
            screen.blit(font.render(f"action: {act}", True, (232, 132, 62)), (10, 8))
            screen.blit(font.render(f"frame {frames}   {fps:.0f} fps   {a.device}",
                                    True, (140, 165, 185)), (170, 8))
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
