"""Hand-authored sprite art for the character.

Drawn as pixel maps rather than composed from rectangles: at 14x20 every pixel
is a deliberate choice, and rectangle stacking cannot express an outline, a
shaded belly, or a tail. The art is 18 wide so the tail and snout can overhang
the 14-wide collision box, which is centred in it.

Facing left is the mirror of facing right, so only one direction is authored.
The tail trails and the snout leads, which is what makes facing readable from
appearance instead of from which way the sprite last moved.

Key:  . transparent   O outline   B body   D shade   C cream   E eye
      W eye-shine     N nose      P paw
"""

from __future__ import annotations

ART_W, ART_H = 18, 20
OFFSET_X = -2  # art is 18 wide, collision box 14 -> centre it
LEAN = 0       # the art is directional on its own; no shift needed

PALETTE = {
    "O": (36, 24, 30),
    "B": (232, 132, 62),
    "D": (168, 84, 40),
    "C": (248, 214, 170),
    "E": (24, 20, 26),
    "W": (255, 255, 255),
    "N": (92, 46, 34),
    "P": (208, 110, 50),
}

# rows 0-16 are shared by every upright pose; only the legs differ
_UPPER = [
    "........OO...O....",
    ".......ODDO.ODO...",
    "......ODDDOODDO...",
    "......OBBBBBBBO...",
    ".....OBBBBBBBBBO..",
    ".....OBBBWEBBBBO..",
    ".....OBBBWEBBCCCO.",
    ".....OBBBBBBBCCNO.",
    "..OO.OBBBBBBBCCO..",
    ".ODDO.OBBBBBBBO...",
    ".ODDDO.OBBBBBO....",
    "..ODDDOOBBBBBOO...",
    "...ODDDBBBBCCCBO..",
    "...ODDDBBBCCCCBO..",
    "....ODDBBBCCCCBO..",
    ".....OBBBBBCCCBO..",
    ".....OBBBBBBBBBO..",
]

_LEGS = {
    "stand": [".....OBBOOOOBBBO..",
              ".....OBBO..OBBBO..",
              "....OOPO...OOPPO.."],
    "walk_a": ["....OBBOOOOOBBBO..",
               "...OBBO.....OBBBO.",
               "..OOPO.......OPPO."],
    "walk_b": [".....OBBOOOBBBO...",
               ".....OBBO.OBBBO...",
               ".....OOPO..OPPO..."],
    "air":   ["....OBBOOOOOBBBO..",
              "...OBBO....OBBBO..",
              "..OOPO......OPPO.."],
}

CROUCH = [
    "........OO...O....",
    ".......ODDO.ODO...",
    "......ODDDOODDO...",
    "......OBBBBBBBO...",
    ".....OBBBBBBBBBO..",
    ".....OBBBWEBBBBO..",
    ".....OBBBWEBBCCCO.",
    ".....OBBBBBBBCCNO.",
    "..OO.OBBBBBBBCCO..",
    ".ODDOOOBBBBBBBO...",
    ".ODDDDBBBBCCCBO...",
    "..ODDDBBBCCCCBO...",
    "...OOBBBBBCCCBO...",
    "....OOOOOOOOOO....",
]

# front paw thrust forward on the leading side
USE_ARM = {6: ".....OBBBWEBBCCCOP", 7: ".....OBBBBBBBCCNOP"}


def frame(pose: str, walk_phase: int = 0) -> list[str]:
    if pose == "crouch":
        return CROUCH
    if pose == "air":
        legs = _LEGS["air"]
    elif pose == "walk":
        legs = _LEGS["walk_a" if walk_phase < 4 else "walk_b"]
    else:
        legs = _LEGS["stand"]
    rows = list(_UPPER) + list(legs)
    if pose == "use":
        for i, r in USE_ARM.items():
            rows[i] = r
    return rows


def _check() -> None:
    bad: list[str] = []
    for name, rows in [("upper", _UPPER), ("crouch", CROUCH),
                       *[(k, v) for k, v in _LEGS.items()]]:
        for i, r in enumerate(rows):
            if len(r) != ART_W:
                bad.append(f"{name}[{i}] = {len(r)}")
    for i, r in USE_ARM.items():
        if len(r) != ART_W:
            bad.append(f"use[{i}] = {len(r)}")
    if bad:
        raise AssertionError(f"rows not {ART_W} wide: " + ", ".join(bad))
    assert len(_UPPER) + 3 == ART_H, "upper + legs must be ART_H tall"


_check()


# --- entities the player does NOT control -------------------------------
# Their motion has to be inferred from the frames rather than read off the
# action, which is the part that makes a world model non-trivial.

ENEMY_W, ENEMY_H = 12, 10
ENEMY_PALETTE = {
    "O": (36, 24, 30),      # outline (shared)
    "P": (150, 92, 200),    # body
    "Q": (108, 62, 152),    # shade
    "E": (24, 20, 26),      # eye
    "W": (255, 255, 255),   # eye white
}
_ENEMY = [
    "...OOOO...OO",
    "..OPPPPO.OQQ",
    ".OPWEPPPOQQQ",
    "OPPWEPPPPOQQ",
    "OPPPPPPPPPOQ",
    "OPQPPPPPPPO.",
    "OPPPPPPPPO..",
    ".OPPPPPPO...",
    "..OO..OO....",
    "..O....O....",
]

PLATFORM_MOVING = (126, 176, 126)   # a moving ledge reads differently from a static one


def enemy_frame(facing_right: bool = True, phase: int = 0) -> list[str]:
    rows = list(_ENEMY)
    if phase >= 4:                      # simple two-frame shuffle
        rows[-2], rows[-1] = rows[-1], rows[-2]
    return rows if facing_right else [r[::-1] for r in rows]


def _check_enemy() -> None:
    for i, r in enumerate(_ENEMY):
        assert len(r) == ENEMY_W, f"enemy row {i} is {len(r)}, expected {ENEMY_W}"
    assert len(_ENEMY) == ENEMY_H, f"enemy is {len(_ENEMY)} tall, expected {ENEMY_H}"


_check_enemy()
