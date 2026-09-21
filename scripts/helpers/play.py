"""Manual human play -- run this on the Mac, not Modal.

The brief is explicit that human play is what reaches the level's odd corners:
scripted policies follow the hop graph, random play flails near the floor, and
neither produces the half-committed jumps and hesitation that make a corpus
cover its own edges. This needs a real display, so it cannot run headless.

    .venv/bin/python scripts/helpers/play.py --episodes 20 --out data/human

Controls:  arrows / A D move,  W or SPACE jump,  S crouch,  J attack,
           K interact,  R restart,  Q quit.  Anything else is idle.

Every step is logged whether or not you press a key, so idling is recorded as
idling rather than dropped.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "dataset"))

# Must beat pixel_world's SDL_VIDEODRIVER=dummy default -- we want a window.
os.environ["SDL_VIDEODRIVER"] = os.environ.get("PLAY_VIDEODRIVER", "cocoa")

import numpy as np
import pygame

from collect import DEFAULT_MIX  # noqa: F401  (kept so shard schema stays aligned)
from env.actions import MASK, encode
from env.splits import HUMAN_SEED_BASE
from env.pixel_world import (
    ATTACK, COIN_SPOTS, CROUCH, IDLE, INTERACT, JUMP, LEFT, N_GOALS, RIGHT, W, H,
    PixelWorld,
)

SCALE = 4


def key_action(keys) -> int:
    if keys[pygame.K_j]:
        return ATTACK
    if keys[pygame.K_k]:
        return INTERACT
    if keys[pygame.K_w] or keys[pygame.K_SPACE] or keys[pygame.K_UP]:
        return JUMP
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        return CROUCH
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        return LEFT
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        return RIGHT
    return IDLE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", default="data/human")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--goal", type=int, default=None,
                    help="force a task; default cycles so every task gets played")
    a = ap.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((W * SCALE, H * SCALE))
    pygame.display.set_caption("pixel world -- human play")
    clock = pygame.time.Clock()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = PixelWorld(seed=0, max_steps=a.max_steps)

    F, A, R, E, G = [], [], [], [], []
    ep = 0
    quit_now = False
    while ep < a.episodes and not quit_now:
        goal = a.goal if a.goal is not None else ep % N_GOALS
        seed = HUMAN_SEED_BASE + ep
        obs = env.reset(seed, goal=goal)
        frames, acts, rews = [obs], [], []
        done = False
        while not done:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    done = quit_now = True
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_q:
                    done = quit_now = True
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                    done = True
            if done:
                break

            act = key_action(pygame.key.get_pressed())
            obs, r, done, _ = env.step(act)
            frames.append(obs)
            acts.append(act)
            rews.append(r)

            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(pygame.transform.scale(surf, (W * SCALE, H * SCALE)), (0, 0))
            pygame.display.flip()
            clock.tick(15)  # env fps -- play at the rate the model will see

        if len(acts) > 4:  # discard accidental restarts
            F.append(np.stack(frames))
            A.append(np.array(acts, dtype=np.uint8))
            R.append(np.array(rews, dtype=np.float32))
            E.append(np.full(len(frames), ep, dtype=np.int32))
            G.append(np.full(len(frames), goal, dtype=np.uint8))
            reached = "hit" if not env.s.coins[goal] else "miss"
            print(f"ep {ep:3d}  goal P{goal}  {len(acts):4d} steps  "
                  f"reward {sum(rews):5.1f}  {reached}")
        ep += 1

    pygame.quit()
    if not F:
        print("nothing recorded")
        return

    frames = np.concatenate(F)
    acts = np.concatenate([np.append(x, 255) for x in A])
    rews = np.concatenate([np.append(x, 0.0) for x in R])
    terminal = np.concatenate([np.append(np.zeros(len(x), bool), True) for x in A])
    vecs = np.stack([encode(x) if x != 255 else np.zeros(16, np.float32) for x in acts])
    dest = out / "shard_human.npz"
    np.savez_compressed(
        dest, frames=frames, actions=acts, action_vecs=vecs, action_mask=MASK,
        rewards=rews, ep_id=np.concatenate(E), goal=np.concatenate(G),
        terminal=terminal, policies=np.array(["human"] * len(F)),
        fps=15, resolution=frames.shape[1],
    )
    print(f"\nwrote {dest}  {len(frames):,} frames  {dest.stat().st_size/1e6:.1f} MB")
    print("upload with: modal volume put pixel-world-data "
          f"{dest} human/shard_human.npz")


if __name__ == "__main__":
    main()
