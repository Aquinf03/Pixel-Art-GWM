"""Verify the splits cannot share an episode seed.

Cheap to check, expensive to get wrong: a train seed leaking into the test
split would silently inflate every rollout metric we report later.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "dataset"))

from env.splits import EPISODES, HUMAN_SEED_BASE, SEED_BASE, SHARDS


def seeds(split: str) -> set[int]:
    base, out = SEED_BASE[split], set()
    for shard in range(SHARDS[split]):
        seed0 = base + shard
        out.update(seed0 * 100_003 + i for i in range(EPISODES[split]))
    return out


def main() -> None:
    sets = {s: seeds(s) for s in SHARDS}
    sets["human"] = set(range(HUMAN_SEED_BASE, HUMAN_SEED_BASE + 500))
    for s, v in sets.items():
        print(f"{s:12s} {len(v):>8,} seeds   range [{min(v):,} .. {max(v):,}]")
    print()
    names, bad = list(sets), False
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            n = len(sets[a] & sets[b])
            if n:
                bad = True
                print(f"OVERLAP {a} n {b}: {n} shared seeds")
    print("all splits disjoint" if not bad else "SPLITS OVERLAP -- fix before training")


if __name__ == "__main__":
    main()
