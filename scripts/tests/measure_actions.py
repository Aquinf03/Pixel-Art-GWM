"""How distinguishable is each action, from the frames alone?

Action marginalisation -- the model ignoring what you press -- is the failure
mode the paper traces to the dynamics component. It is far likelier when an
action barely changes the picture: if pressing a button moves 40 pixels of a
16,384-pixel frame, the model can predict well while ignoring it entirely.

So from real sampled states we branch on every action, render each outcome,
and measure how far apart those outcomes are. Actions whose branches collapse
onto each other are unlearnable no matter how the vector is encoded, and are
the ones worth cutting or amplifying.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "dataset"))

from env import policies
from env.pixel_world import ACTION_NAMES, N_ACTIONS, PixelWorld

MIX = ["scripted", "scripted_eps0.2", "scripted_eps0.5", "random", "explorer"]


def sample_states(n_states: int, seed: int, horizon: int):
    """Snapshot states from real rollouts, so we branch where the data lives."""
    rng = np.random.default_rng(seed)
    out = []
    ep = 0
    while len(out) < n_states:
        env = PixelWorld(seed=seed * 7919 + ep, max_steps=200)
        pol = policies.build(MIX[ep % len(MIX)], np.random.default_rng(seed + ep))
        env.reset(seed * 7919 + ep)
        for t in range(200):
            _, _, d, _ = env.step(pol(env), render=False)
            if t > 3 and rng.random() < 0.25:
                out.append((env.s.goal, copy.deepcopy(env.s)))
                if len(out) >= n_states:
                    break
            if d:
                break
        ep += 1
    return out[:n_states]


def branch(env: PixelWorld, state, action: int, horizon: int) -> np.ndarray:
    """Restore a state, hold one action for `horizon` steps, return the frame."""
    env.s = copy.deepcopy(state)
    obs = None
    for _ in range(horizon):
        obs, _, _, _ = env.step(action)
    return obs.astype(np.int16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=int, default=400)
    ap.add_argument("--horizon", type=int, default=1, help="steps the action is held")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    env = PixelWorld(seed=0, max_steps=10**9)
    states = sample_states(a.states, a.seed, a.horizon)

    # frames[i][s] = outcome of action i from state s
    frames = [[branch(env, st, i, a.horizon) for _, st in states] for i in range(N_ACTIONS)]

    npix = frames[0][0].size / 3
    sep = np.zeros((N_ACTIONS, N_ACTIONS))
    for i, j in itertools.combinations(range(N_ACTIONS), 2):
        d = [np.any(frames[i][s] != frames[j][s], axis=-1).sum() / npix
             for s in range(len(states))]
        sep[i, j] = sep[j, i] = float(np.mean(d))

    print(f"{a.states} states, action held {a.horizon} step(s)")
    print("\nPairwise separability -- mean % of pixels differing between the")
    print("outcomes of two actions from the SAME state. Near 0 = indistinguishable.\n")
    hdr = "".join(f"{n[:5]:>8}" for n in ACTION_NAMES)
    print(f"{'':>9}{hdr}")
    for i in range(N_ACTIONS):
        row = "".join(f"{sep[i, j] * 100:>8.2f}" for j in range(N_ACTIONS))
        print(f"{ACTION_NAMES[i]:>9}{row}")

    print("\nPer-action distinguishability (mean separability vs all others):")
    order = np.argsort(-sep.mean(1))
    for i in order:
        m = sep[i].mean() * 100
        bar = "#" * int(m * 8)
        print(f"  {ACTION_NAMES[i]:>9} {m:6.2f}%  {bar}")

    worst = min(itertools.combinations(range(N_ACTIONS), 2), key=lambda p: sep[p])
    print(f"\nleast separable pair: {ACTION_NAMES[worst[0]]} vs "
          f"{ACTION_NAMES[worst[1]]} ({sep[worst] * 100:.2f}%)")


if __name__ == "__main__":
    main()
