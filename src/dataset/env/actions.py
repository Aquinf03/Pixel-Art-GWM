"""Action encoding: 7 discrete buttons <-> MMBench2's 16-dim continuous format.

MMBench2 zero-pads every action to 16 dims and carries a per-dimension
validity mask, so a new environment slots in by claiming the first few dims
and masking the rest. We claim 4:

    0  move      -1 left, 0 none, +1 right
    1  vertical  +1 jump, -1 crouch, 0 none
    2  use       0 / 1

The mapping is invertible, which matters for the action-shuffle and
action-inversion checks the paper uses to detect action marginalisation.
"""

from __future__ import annotations

import numpy as np

from .pixel_world import CROUCH, IDLE, JUMP, LEFT, RIGHT, USE

ACTION_DIM = 16
VALID_DIMS = 3
MASK = np.array([1] * VALID_DIMS + [0] * (ACTION_DIM - VALID_DIMS), dtype=np.float32)

_TO_VEC = {
    IDLE:   (0.0,  0.0, 0.0),
    LEFT:   (-1.0, 0.0, 0.0),
    RIGHT:  (1.0,  0.0, 0.0),
    JUMP:   (0.0,  1.0, 0.0),
    CROUCH: (0.0, -1.0, 0.0),
    USE:    (0.0,  0.0, 1.0),
}
_FROM_VEC = {v: k for k, v in _TO_VEC.items()}


def encode(action: int) -> np.ndarray:
    """Discrete button -> (16,) float32, zero-padded."""
    v = np.zeros(ACTION_DIM, dtype=np.float32)
    v[:VALID_DIMS] = _TO_VEC[int(action)]
    return v


def decode(vec: np.ndarray) -> int:
    """(16,) float32 -> nearest discrete button. Exact for encode() output."""
    key = tuple(float(x) for x in np.round(np.asarray(vec)[:VALID_DIMS], 3))
    if key in _FROM_VEC:
        return _FROM_VEC[key]
    known = np.stack([encode(a)[:VALID_DIMS] for a in range(len(_TO_VEC))])
    return int(np.argmin(np.linalg.norm(known - np.asarray(vec)[:VALID_DIMS], axis=1)))


def invert(action: int) -> int:
    """Semantic opposite -- used by the action-inversion hallucination probe."""
    return {LEFT: RIGHT, RIGHT: LEFT, JUMP: CROUCH, CROUCH: JUMP}.get(int(action), int(action))
