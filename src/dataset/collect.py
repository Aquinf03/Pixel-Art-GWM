"""Roll out episodes and write compressed shards.

Storage: frames are kept once per episode, not duplicated into (frame,
next_frame) pairs -- next_frame is frames[i+1] within an episode, and
`ep_id` marks the boundaries. Flat pixel art compresses hard, so npz
deflate does most of the work for us.

Run locally for a smoke test, or inside Modal for the real collection.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np

from env import policies
from env.actions import MASK, encode
from env.pixel_world import COIN_SPOTS, N_ACTIONS, PixelWorld

# Mix chosen so most data is competent play, with random and explorer
# providing the tails. The explorer share is small on purpose -- the high
# ledge should stay a low-coverage region.
DEFAULT_MIX = {"scripted": 0.4, "scripted_eps0.2": 0.25, "scripted_eps0.5": 0.15,
               "random": 0.15, "explorer": 0.05}


def rollout(env: PixelWorld, policy, seed: int, goal: int | None = None,
            action_repeat: int = 1):
    """One episode. Returns per-step arrays; frames has len == steps + 1.

    `action_repeat` holds each chosen action for k physics steps and records
    only the resulting frame, i.e. the model sees 15/k fps. This is the single
    biggest lever on long-horizon coherence: at 15 fps a three-second rollout
    is 45 frames against a context of 8-16, so most of it is the model eating
    its own output. At 5 fps the same three seconds is ~15 frames, roughly one
    context window. It also triples per-frame motion, making actions clearer.
    """
    obs = env.reset(seed, goal=goal)
    frames, acts, rews = [obs], [], []
    done = False
    while not done:
        a = policy(env)
        r_total = 0.0
        for _ in range(action_repeat):
            obs, r, done, _ = env.step(a, render=False)
            r_total += r
            if done:
                break
        obs = env.render()
        frames.append(obs)
        acts.append(a)
        rews.append(r_total)
    return (
        np.stack(frames),
        np.array(acts, dtype=np.uint8),
        np.array(rews, dtype=np.float32),
        env.s.goal,
        not env.s.coins[env.s.goal],   # goal reached?
    )


def collect_shard(out: Path, n_episodes: int, seed0: int, mix=None,
                  max_steps: int = 200, goal: int | None = None,
                  action_repeat: int = 1, verbose: bool = True) -> dict:
    """Write one shard. `goal` forces every episode onto one task -- used to
    build the ledge probe set, where we need dense coverage of the rare
    region precisely because the training corpus does not have it."""
    mix = mix or DEFAULT_MIX
    names, probs = list(mix), np.array(list(mix.values()), dtype=float)
    probs = probs / probs.sum()

    rng = np.random.default_rng(seed0)
    env = PixelWorld(seed=seed0, max_steps=max_steps)

    F, A, R, E, P, G = [], [], [], [], [], []
    goal_eps, goal_hit = {}, {}
    t0 = time.time()

    for i in range(n_episodes):
        seed = seed0 * 100_003 + i
        pname = names[int(rng.choice(len(names), p=probs))]
        pol = policies.build(pname, np.random.default_rng(seed ^ 0x9E3779B9))
        frames, acts, rews, ep_goal, hit = rollout(env, pol, seed, goal=goal,
                                                   action_repeat=action_repeat)

        F.append(frames)
        A.append(acts)
        R.append(rews)
        E.append(np.full(len(frames), i, dtype=np.int32))
        G.append(np.full(len(frames), ep_goal, dtype=np.uint8))
        P.append(pname)
        # Per-task counts: this is the coverage structure the whole project
        # is about, so it is recorded per shard rather than recomputed later.
        goal_eps[ep_goal] = goal_eps.get(ep_goal, 0) + 1
        goal_hit[ep_goal] = goal_hit.get(ep_goal, 0) + int(hit)
        if verbose and (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n_episodes} episodes  ({time.time() - t0:.1f}s)", flush=True)

    frames = np.concatenate(F)
    # Actions/rewards are one shorter per episode than frames; pad the
    # terminal step with a sentinel so every array shares one index space.
    acts = np.concatenate([np.append(a, 255) for a in A])
    rews = np.concatenate([np.append(r, 0.0) for r in R])
    eps = np.concatenate(E)
    goals = np.concatenate(G)
    terminal = np.concatenate([np.append(np.zeros(len(a), bool), True) for a in A])

    out.parent.mkdir(parents=True, exist_ok=True)
    vecs = np.stack([encode(a) if a != 255 else np.zeros(16, np.float32) for a in acts])
    np.savez_compressed(
        out,
        frames=frames, actions=acts, action_vecs=vecs, action_mask=MASK,
        rewards=rews, ep_id=eps, goal=goals, terminal=terminal,
        policies=np.array(P), fps=15 // action_repeat,
        action_repeat=action_repeat, resolution=frames.shape[1],
        n_actions=N_ACTIONS, action_space_version=2, art_version=3,
    )

    meta = {
        "shard": out.name,
        "action_space_version": 2,
        "action_repeat": action_repeat,
        "fps": 15 // action_repeat,
        "art_version": 3,
        "episodes": n_episodes,
        "frames": int(len(frames)),
        "steps": int((~terminal).sum()),
        "bytes": out.stat().st_size,
        "bytes_per_frame": round(out.stat().st_size / len(frames), 1),
        "reward_total": float(rews.sum()),
        "goal_episodes": {int(k): v for k, v in sorted(goal_eps.items())},
        "goal_frames": {int(k): int((goals == k).sum()) for k in sorted(goal_eps)},
        "goal_success": {int(k): round(goal_hit[k] / v, 3) for k, v in sorted(goal_eps.items())},
        "policy_counts": {n: P.count(n) for n in names},
        "seconds": round(time.time() - t0, 1),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/shard_000.npz")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--goal", type=int, default=None, help="force every episode onto one task")
    ap.add_argument("--action-repeat", type=int, default=1, dest="action_repeat")
    a = ap.parse_args()
    meta = collect_shard(Path(a.out), a.episodes, a.seed, max_steps=a.max_steps,
                         goal=a.goal, action_repeat=a.action_repeat)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
