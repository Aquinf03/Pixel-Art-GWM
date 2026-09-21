"""Derive the level's hop graph by simulating the real physics.

Hand-computed jump arcs are exactly the kind of thing that goes quietly
stale when a constant changes, so nothing here is asserted: for every
platform we try every launch x and every hold direction, run the actual
step() loop, and record where we land. The scripted policy then routes over
whatever graph falls out.

Cheap enough (a few thousand short rollouts) to just run at import.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .pixel_world import (
    JUMP, LEFT, PLATFORMS, PLAYER_H, PLAYER_W, RIGHT, PixelWorld,
)

MAX_AIR_STEPS = 60


@dataclass(frozen=True)
class Hop:
    src: int
    dst: int
    launch_x: float   # centre of the widest band of launch points that works
    lo: float         # band bounds -- the policy commits anywhere inside,
    hi: float         # which must be wider than MOVE_SPEED or it oscillates
    hold: int         # -1 left, 0 none, +1 right


def _place(env: PixelWorld, platform: int, x: float) -> None:
    top = PLATFORMS[platform][1]
    s = env.s
    s.x, s.y = float(x), float(top - PLAYER_H)
    s.vx = s.vy = 0.0
    s.on_ground, s.crouching, s.attack_timer = True, False, 0


def _simulate(env: PixelWorld, src: int, x: float, hold: int) -> int | None:
    """Jump from (src, x) holding `hold`. Returns the platform landed on."""
    _place(env, src, x)
    env.step(JUMP, render=False)
    act = {(-1): LEFT, 0: 0, 1: RIGHT}[hold]
    for _ in range(MAX_AIR_STEPS):
        env.step(act, render=False)
        if env.s.on_ground:
            return env.platform_under()
    return None


def _widest_run(xs: list[int], step: int = 1) -> tuple[float, float, float]:
    """Longest run of launch xs spaced `step` apart -> (centre, lo, hi).

    `step` must match the scan stride, or a coarse scan reports every band as
    a single point and the policy oscillates in front of every jump.
    """
    best = run = [xs[0]]
    for x in xs[1:]:
        run = run + [x] if x == run[-1] + step else [x]
        if len(run) > len(best):
            best = run
    return (best[0] + best[-1]) / 2.0, float(best[0]), float(best[-1])


def build_hops(step: int = 1) -> list[Hop]:
    """Scan every launch x at 1px resolution and keep the widest working band.

    A single launch point is not enough: the player moves MOVE_SPEED px per
    step, so a band narrower than that can be stepped straight over, and the
    policy oscillates in front of the jump forever.
    """
    env = PixelWorld(seed=0, max_steps=10**9)
    found: dict[tuple[int, int, int], list[int]] = {}
    for src, (px, _, pw, _) in enumerate(PLATFORMS):
        for x in range(int(px), int(px + pw - PLAYER_W) + 1, step):
            for hold in (-1, 0, 1):
                dst = _simulate(env, src, float(x), hold)
                if dst is not None and dst != src:
                    found.setdefault((src, dst, hold), []).append(x)

    # One edge per (src, dst): the hold direction with the widest band.
    best: dict[tuple[int, int], Hop] = {}
    for (src, dst, hold), xs in found.items():
        centre, lo, hi = _widest_run(sorted(xs), step)
        if (src, dst) not in best or (hi - lo) > (best[(src, dst)].hi - best[(src, dst)].lo):
            best[(src, dst)] = Hop(src, dst, centre, lo, hi, hold)
    return list(best.values())


class Nav:
    def __init__(self, step: int = 1):
        self.hops = build_hops(step)
        self.adj: dict[int, list[Hop]] = {}
        for h in self.hops:
            self.adj.setdefault(h.src, []).append(h)

    def route(self, src: int, dst: int) -> list[Hop]:
        """Shortest hop sequence from src to dst ([] if same, None if none)."""
        if src == dst:
            return []
        seen, q = {src}, deque([(src, [])])
        while q:
            node, path = q.popleft()
            for h in self.adj.get(node, []):
                if h.dst in seen:
                    continue
                if h.dst == dst:
                    return path + [h]
                seen.add(h.dst)
                q.append((h.dst, path + [h]))
        return None

    def reachable(self, src: int = 0) -> set[int]:
        seen, q = {src}, deque([src])
        while q:
            for h in self.adj.get(q.popleft(), []):
                if h.dst not in seen:
                    seen.add(h.dst)
                    q.append(h.dst)
        return seen


_NAV: Nav | None = None


def nav() -> Nav:
    global _NAV
    if _NAV is None:
        _NAV = Nav()
    return _NAV
