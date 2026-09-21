"""Pixel World: a small 2D platformer for world-model data collection.

One controllable sprite, 7 discrete actions, deterministic given a seed.
Renders 128x128 RGB at native resolution -- every pixel drawn once, no
upscaling, so the art stays crisp and the tokenizer sees real pixel art
rather than interpolation blur.

The API is reset/step so this same object can serve as the live simulator
that hallucination-guided collection drives online (phase 5).

Layout note: the level has a deliberate low-coverage corner -- the high
ledge at the top right is reachable only by a precise jump from platform B,
so scripted policies almost never go there. That is the region where we
expect the hallucination predictors to spike.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Must be set before pygame imports SDL; keeps this importable on a headless box.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame

from . import sprite

W = H = 128
FPS = 15  # matches MMBench2

# --- actions -------------------------------------------------------------
# `use` is context-sensitive: it breaks a crate or opens a door depending on
# what you are standing next to, and always draws its swing. The old separate
# `interact` measured 0.00% separability against `idle` -- it rendered nothing
# except in the one frame it opened the door, so no model could ever learn it.
IDLE, LEFT, RIGHT, JUMP, CROUCH, USE = range(6)
N_ACTIONS = 6
ACTION_NAMES = ["idle", "left", "right", "jump", "crouch", "use"]

# --- physics (per step, i.e. per 1/15 s) ---------------------------------
GRAVITY = 0.55
MOVE_SPEED = 3.6       # 2.6x the old 1.6 -- displacement is what makes
                       # left/right distinguishable frame to frame
JUMP_VELOCITY = -6.6   # rise ~40px, clears the 28px tier spacing
TERMINAL_VY = 7.0
REACH = 14.0  # how close the player must be to use a crate / door

# Sprite area sets a hard ceiling on how different two actions' outcomes can
# look. 14x20 is 1.71% of the frame (was 0.71%), measured 2.6x separability.
PLAYER_W, PLAYER_H, CROUCH_H = 14, 20, 12

# --- palette (mossy grove; deliberately not an Atari palette) -------------
SKY_TOP = (26, 38, 52)
SKY_MID = (34, 52, 66)
SKY_LOW = (44, 68, 78)
MOSS = (58, 106, 74)
MOSS_TOP = (96, 158, 96)
MOSS_DARK = (36, 72, 54)
PLAYER_BODY = (232, 132, 62)
PLAYER_DARK = (168, 84, 40)
PLAYER_SKIN = (248, 206, 158)
EYE = (28, 26, 34)
COIN = (246, 206, 84)
COIN_SHINE = (255, 244, 190)
CRATE = (146, 100, 58)
CRATE_DARK = (98, 66, 38)
DOOR = (120, 96, 148)
DOOR_OPEN = (206, 190, 236)
SLASH = (255, 248, 214)
ENEMY_BODY = (150, 92, 200)
ENEMY_SHADE = (108, 62, 152)
MOSS_MOVING = (126, 176, 126)
# Never appears in the palette, so it is safe as a transparency key.
_COLORKEY = (255, 0, 255)

# Zig-zag staircase. Reachability is NOT asserted here -- env/nav.py derives
# the hop graph by simulating the real physics, so tweaking these numbers can
# never silently leave a platform stranded.
GROUND, P1, P2, P3 = 0, 1, 2, 3
# Re-searched after the sprite and speed changes invalidated the old graph
# (~8k candidates, scored on measured properties -- env/nav.py derives the hop
# graph by simulating the physics, so neither is assumed): every platform
# reachable via a 3-hop route to P3, and every launch band on that route at
# least 24px wide. Band width is the one that bites -- a band narrower than
# MOVE_SPEED can be stepped over, and the policy oscillates in front of the
# jump forever.
PLATFORMS = [
    (0, 112, 128, 16),   # ground
    (56, 84, 38, 4),     # P1
    (13, 56, 38, 4),     # P2
    (83, 28, 38, 4),     # P3 -- the rare high ledge
]

# One coin per platform, resting on the surface. The P4 coin is the one we
# expect to stay under-covered.
COIN_SPOTS = [(12, 106), (75, 78), (32, 50), (102, 22)]
COIN_PLATFORM = [GROUND, P1, P2, P3]
CRATE_SPOT = (96, 104)
DOOR_SPOT = (118, 102)

# Each episode targets one coin. A goal is a "task" in the paper's sense, so
# the coverage-aware-sampling mitigation (resample uniform across tasks
# rather than frames) applies directly. The skew is what makes the top ledge
# a low-coverage region by construction rather than by accident.
# --- entities the player does not control -------------------------------
# Everything above is either static or driven by the action. These two are
# not: their motion has to be INFERRED from the frames. That is the part of a
# world model that is actually hard, and v1 did not test it at all.

# (platform index, x span, speed px/step) -- patrols back and forth on a ledge
ENEMY_PATROLS = [(1, (58, 92), 1.4), (0, (18, 62), 1.9)]

# (index into PLATFORMS, axis, amplitude, speed) -- P2 slides horizontally
MOVING_PLATFORM = (2, "x", 11.0, 0.9)

GOAL_WEIGHTS = (0.32, 0.31, 0.31, 0.06)
N_GOALS = len(GOAL_WEIGHTS)


@dataclass
class State:
    x: float = 16.0
    y: float = 99.0
    vx: float = 0.0
    vy: float = 0.0
    on_ground: bool = True
    facing: int = 1
    crouching: bool = False
    attack_timer: int = 0
    walk_phase: int = 0
    coins: list = field(default_factory=list)   # True == still collectable
    crate_ok: bool = True
    door_open: bool = False
    steps: int = 0
    goal: int = 0                               # which coin this episode wants
    enemies: list = field(default_factory=list) # [x, dir, phase] per patrol
    plat_t: float = 0.0                         # phase of the moving platform


class PixelWorld:
    """Deterministic 2D platformer. obs is uint8 (128, 128, 3)."""

    _SPRITE_CACHE: dict = {}

    def __init__(self, seed: int = 0, max_steps: int = 200):
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self._surf = pygame.Surface((W, H))
        self.s = State()
        self.reset(seed)

    # -- core ------------------------------------------------------------
    def reset(self, seed: int | None = None, goal: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if goal is None:
            goal = int(self.rng.choice(N_GOALS, p=GOAL_WEIGHTS))
        # Small start jitter so the model cannot memorise one opening frame.
        self.s = State(
            x=float(self.rng.integers(4, 22)),
            y=99.0,
            coins=[True] * len(COIN_SPOTS),
            goal=int(goal),
            enemies=[[float(lo + (hi - lo) * self.rng.random()),
                      1.0 if self.rng.random() < 0.5 else -1.0, 0]
                     for _, (lo, hi), _ in ENEMY_PATROLS],
            plat_t=float(self.rng.random() * 6.283),
        )
        return self.render()

    def step(self, action: int, render: bool = True):
        """render=False skips rasterising -- nav/search run physics only."""
        s, reward = self.s, 0.0
        action = int(action)

        s.crouching = action == CROUCH and s.on_ground
        if action == LEFT:
            s.vx, s.facing = -MOVE_SPEED, -1
        elif action == RIGHT:
            s.vx, s.facing = MOVE_SPEED, 1
        else:
            s.vx = 0.0
        if action == JUMP and s.on_ground:
            s.vy, s.on_ground = JUMP_VELOCITY, False
        if action == USE:
            s.attack_timer = 3

        self._advance_entities()
        self._physics()

        s.walk_phase = (s.walk_phase + 1) % 8 if s.vx else 0
        s.attack_timer = max(0, s.attack_timer - 1)

        reward += self._collect_coins()
        if action == USE:
            if s.crate_ok and self.near(CRATE_SPOT):
                s.crate_ok, reward = False, reward + 2.0
            elif not s.door_open and self.near(DOOR_SPOT):
                s.door_open, reward = True, reward + 5.0

        s.steps += 1
        # Episode ends on the goal coin, not on clearing the level: that is
        # what keeps each episode inside its own task's region.
        done = s.steps >= self.max_steps or not s.coins[s.goal]
        obs = self.render() if render else None
        return obs, reward, done, {"pos": (s.x, s.y)}

    def _advance_entities(self) -> None:
        """Uncontrolled motion: patrols bounce, the moving platform oscillates."""
        s = self.s
        for i, (_, (lo, hi), spd) in enumerate(ENEMY_PATROLS):
            e = s.enemies[i]
            e[0] += spd * e[1]
            if e[0] <= lo:
                e[0], e[1] = lo, 1.0
            elif e[0] >= hi:
                e[0], e[1] = hi, -1.0
            e[2] = (e[2] + 1) % 8
        s.plat_t += MOVING_PLATFORM[3] * 0.1

    def moving_platform_rect(self):
        """PLATFORMS[i] displaced by the current oscillation."""
        import math
        i, axis, amp, _ = MOVING_PLATFORM
        px, py, pw, ph = PLATFORMS[i]
        d = amp * math.sin(self.s.plat_t)
        return (px + d, py, pw, ph) if axis == "x" else (px, py + d, pw, ph)

    # -- physics helpers --------------------------------------------------
    def _rect(self, x=None, y=None) -> pygame.Rect:
        """Player AABB. Crouching shrinks from the top, so the feet stay put."""
        s = self.s
        h = CROUCH_H if s.crouching else PLAYER_H
        return pygame.Rect(
            int(x if x is not None else s.x),
            int(y if y is not None else s.y) + (PLAYER_H - h),
            PLAYER_W,
            h,
        )

    def _platforms_now(self):
        """PLATFORMS with the moving one at its current position."""
        out = []
        mi = MOVING_PLATFORM[0]
        for i, p in enumerate(PLATFORMS):
            if i == mi:
                x, y, w, h = self.moving_platform_rect()
                out.append((int(x), int(y), int(w), int(h)))
            else:
                out.append(p)
        return out

    def _hits(self, r: pygame.Rect):
        """Highest platform the rect overlaps, or None."""
        hit = None
        for p in self._platforms_now():
            if r.colliderect(pygame.Rect(*p)) and (hit is None or p[1] < hit[1]):
                hit = p
        return hit

    def _physics(self) -> None:
        s = self.s
        if s.vx:
            nx = min(max(s.x + s.vx, 0.0), float(W - PLAYER_W))
            if self._hits(self._rect(x=nx)) is None:
                s.x = nx

        s.vy = min(s.vy + GRAVITY, TERMINAL_VY)
        ny = s.y + s.vy
        hit = self._hits(self._rect(y=ny))
        if hit is None:
            s.y, s.on_ground = ny, False
            return
        px, py, ph = hit[0], hit[1], hit[3]
        if s.vy > 0:                      # landed
            s.y, s.on_ground = float(py - PLAYER_H), True
        else:                             # bonked the underside
            h = CROUCH_H if s.crouching else PLAYER_H
            s.y = float(py + ph - (PLAYER_H - h))
        s.vy = 0.0

    def platform_under(self) -> int | None:
        """Index of the platform we are standing on, or None if airborne."""
        if not self.s.on_ground:
            return None
        feet = self._rect().move(0, 1)
        for i, p in enumerate(self._platforms_now()):
            if feet.colliderect(pygame.Rect(*p)):
                return i
        return None

    def near(self, spot) -> bool:
        cx, cy = self.s.x + PLAYER_W / 2, self.s.y + PLAYER_H / 2
        return abs(cx - spot[0]) < REACH and abs(cy - spot[1]) < REACH

    def _collect_coins(self) -> float:
        got = 0.0
        pr = self._rect()
        for i, alive in enumerate(self.s.coins):
            if alive and pr.colliderect(pygame.Rect(COIN_SPOTS[i][0] - 2, COIN_SPOTS[i][1] - 2, 6, 6)):
                self.s.coins[i], got = False, got + 1.0
        return got

    # -- rendering --------------------------------------------------------
    def render(self) -> np.ndarray:
        s, surf = self.s, self._surf
        surf.fill(SKY_TOP)
        pygame.draw.rect(surf, SKY_MID, (0, 42, W, 44))
        pygame.draw.rect(surf, SKY_LOW, (0, 86, W, 42))

        mi = MOVING_PLATFORM[0]
        for i, (px, py, pw, ph) in enumerate(self._platforms_now()):
            top = MOSS_MOVING if i == mi else MOSS_TOP
            pygame.draw.rect(surf, MOSS, (px, py, pw, ph))
            pygame.draw.rect(surf, top, (px, py, pw, 2))
            pygame.draw.rect(surf, MOSS_DARK, (px, py + ph - 1, pw, 1))

        for i, alive in enumerate(s.coins):
            if alive:
                cx, cy = COIN_SPOTS[i]
                pygame.draw.rect(surf, COIN, (cx - 2, cy - 2, 4, 4))
                pygame.draw.rect(surf, COIN_SHINE, (cx - 1, cy - 2, 1, 2))

        if s.crate_ok:
            cx, cy = CRATE_SPOT
            pygame.draw.rect(surf, CRATE, (cx - 4, cy - 4, 8, 8))
            pygame.draw.rect(surf, CRATE_DARK, (cx - 4, cy - 1, 8, 1))
            pygame.draw.rect(surf, CRATE_DARK, (cx - 1, cy - 4, 1, 8))

        dx, dy = DOOR_SPOT
        pygame.draw.rect(surf, DOOR_OPEN if s.door_open else DOOR, (dx - 4, dy - 6, 8, 12))
        if not s.door_open:
            pygame.draw.rect(surf, COIN, (dx + 1, dy, 1, 1))  # handle

        self._draw_enemies(surf)
        self._draw_player(surf)
        # pygame is (w, h); numpy wants (h, w) row-major.
        return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2)).copy()

    _ENEMY_CACHE: dict = {}

    @classmethod
    def _enemy_surface(cls, facing_right: bool, phase: int):
        key = (facing_right, phase >= 4)
        got = cls._ENEMY_CACHE.get(key)
        if got is not None:
            return got
        rows = sprite.enemy_frame(facing_right, phase)
        surf = pygame.Surface((sprite.ENEMY_W, sprite.ENEMY_H))
        surf.fill(_COLORKEY); surf.set_colorkey(_COLORKEY)
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                if ch != ".":
                    surf.set_at((x, y), sprite.ENEMY_PALETTE[ch])
        cls._ENEMY_CACHE[key] = surf
        return surf

    def _draw_enemies(self, surf) -> None:
        plats = self._platforms_now()
        for i, (pi, _, _) in enumerate(ENEMY_PATROLS):
            x, d, phase = self.s.enemies[i]
            top = plats[pi][1]
            surf.blit(self._enemy_surface(d > 0, int(phase)),
                      (int(x), int(top - sprite.ENEMY_H)))

    def _draw_player(self, surf) -> None:
        """Blit the authored sprite (env/sprite.py) for the current pose.

        Surfaces are cached per (pose, phase, facing): the art is a pixel map,
        and rasterising it per-pixel on every one of two million frames would
        dominate collection time.
        """
        s = self.s
        r = self._rect()
        if s.crouching:
            pose, phase = "crouch", 0
        elif s.attack_timer:
            pose, phase = "use", 0
        elif not s.on_ground:
            pose, phase = "air", 0
        elif s.vx:
            pose, phase = "walk", 0 if s.walk_phase < 4 else 4
        else:
            pose, phase = "stand", 0

        # sprite.LEAN is 0: the art is a side profile with the mass pushed
        # forward and the tail trailing, so mirroring alone moves 71% of the
        # sprite's pixels. It stays configurable in case a pose ever needs it.
        lean = sprite.LEAN if s.facing > 0 else -sprite.LEAN
        surf.blit(self._sprite_surface(pose, phase, s.facing > 0),
                  (r.x + sprite.OFFSET_X + lean, r.bottom - self._sprite_h(pose)))
        if s.attack_timer:
            sx = r.right + 3 if s.facing > 0 else r.x - 11
            pygame.draw.rect(surf, SLASH, (sx, r.y + 5, 8, 6))

    @staticmethod
    def _sprite_h(pose: str) -> int:
        return len(sprite.frame(pose))

    @classmethod
    def _sprite_surface(cls, pose: str, phase: int, facing_right: bool):
        key = (pose, phase, facing_right)
        cached = cls._SPRITE_CACHE.get(key)
        if cached is not None:
            return cached
        rows = sprite.frame(pose, phase)
        if not facing_right:
            rows = [row[::-1] for row in rows]
        surf = pygame.Surface((sprite.ART_W, len(rows)))
        surf.fill(_COLORKEY)
        surf.set_colorkey(_COLORKEY)
        for y, row in enumerate(rows):
            for x, ch in enumerate(row):
                if ch != ".":
                    surf.set_at((x, y), sprite.PALETTE[ch])
        cls._SPRITE_CACHE[key] = surf
        return surf
