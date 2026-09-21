"""Data-collection policies.

Three sources, mirroring the mixed-quality splits dreamer4 collects with:
scripted (competent), noisy-scripted (competent but perturbed), and random.
Manual human play comes later -- that is what reaches the level's odd
corners, and it needs a real display.

The scripted policy routes over the hop graph in env/nav.py and reads env
state directly. That is deliberate: it is a privileged data collector, in
the same role as the TD-MPC2 experts dreamer4's corpus was collected with,
not a policy we intend to learn from.
"""

from __future__ import annotations

import numpy as np

from .nav import nav
from .pixel_world import (
    COIN_PLATFORM, COIN_SPOTS, CRATE_SPOT, CROUCH, DOOR_SPOT, IDLE,
    JUMP, LEFT, MOVE_SPEED, N_ACTIONS, PLAYER_W, RIGHT, USE,
)

STUCK_LIMIT = 12  # steps without moving before we shake loose


class RandomPolicy:
    name = "random"

    def __init__(self, rng):
        self.rng = rng

    def __call__(self, env) -> int:
        return int(self.rng.integers(0, N_ACTIONS))


class _Router:
    """Shared movement core: get to a target platform, then to a target x."""

    def __init__(self, rng):
        self.rng = rng
        self.hold = 0        # direction held while airborne, set when we jump
        self._last_x = None
        self._stuck = 0

    def _hold_action(self) -> int:
        return {-1: LEFT, 0: IDLE, 1: RIGHT}[self.hold]

    def move(self, env, dst_platform: int, dst_x: float) -> int:
        # A policy that idles in place emits dead frames, which is worse than
        # useless as training data -- it teaches the model that nothing moves.
        x = env.s.x
        self._stuck = self._stuck + 1 if self._last_x == x else 0
        self._last_x = x
        if self._stuck >= STUCK_LIMIT:
            self._stuck = 0
            return int(self.rng.integers(0, N_ACTIONS))

        here = env.platform_under()
        if here is None:              # airborne -- keep steering the arc
            return self._hold_action()

        if here == dst_platform:
            self.hold = 0
            cx = x + PLAYER_W / 2
            if abs(dst_x - cx) < 2.0:
                return IDLE
            return RIGHT if dst_x > cx else LEFT

        route = nav().route(here, dst_platform)
        if not route:                 # stranded; fall back to wandering
            return int(self.rng.integers(0, N_ACTIONS))

        hop = route[0]
        if hop.lo - MOVE_SPEED <= x <= hop.hi + MOVE_SPEED:
            self.hold = hop.hold
            return JUMP
        return RIGHT if hop.launch_x > x else LEFT


class ScriptedPolicy:
    """Competent play: climb to each coin in turn, break the crate, exit."""

    def __init__(self, rng, epsilon: float = 0.0):
        self.rng = rng
        self.epsilon = epsilon
        self.router = _Router(rng)
        self.name = "scripted" if epsilon == 0 else f"scripted_eps{epsilon:g}"

    def __call__(self, env) -> int:
        if self.epsilon and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, N_ACTIONS))
        return self._greedy(env)

    def _greedy(self, env) -> int:
        s = env.s
        # Head for this episode's goal coin. Coins passed on the way are
        # collected incidentally, which is where the off-goal coverage
        # comes from.
        if s.coins[s.goal]:
            g = s.goal
            return self.router.move(env, COIN_PLATFORM[g], COIN_SPOTS[g][0])
        if s.crate_ok:
            if env.near(CRATE_SPOT):
                return USE
            return self.router.move(env, 0, CRATE_SPOT[0])
        if not s.door_open:
            if env.near(DOOR_SPOT):
                return USE
            return self.router.move(env, 0, DOOR_SPOT[0])
        return IDLE


class ExplorerPolicy:
    """Climbs toward the high ledge with heavy noise. Deliberately a small
    share of the mix -- the point of the experiment is that P4 stays
    under-covered, so this provides a thin tail, not solid coverage."""

    name = "explorer"

    def __init__(self, rng, epsilon: float = 0.35):
        self.rng = rng
        self.epsilon = epsilon
        self.router = _Router(rng)

    def __call__(self, env) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, N_ACTIONS))
        top = len(COIN_SPOTS) - 1
        return self.router.move(env, COIN_PLATFORM[top], COIN_SPOTS[top][0])


def build(name: str, rng):
    if name == "random":
        return RandomPolicy(rng)
    if name == "scripted":
        return ScriptedPolicy(rng)
    if name.startswith("scripted_eps"):
        return ScriptedPolicy(rng, float(name.removeprefix("scripted_eps")))
    if name == "explorer":
        return ExplorerPolicy(rng)
    raise ValueError(f"unknown policy: {name}")
