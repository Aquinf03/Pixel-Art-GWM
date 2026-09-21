"""Split definitions. Shared by the Modal app and the disjointness check so
the two cannot drift apart -- if they did, the check would be verifying a
layout the collector no longer uses.

Seed blocks are far apart: collect_shard derives per-episode seeds as
seed0 * 100_003 + i, so bases this widely spaced cannot collide.
"""

SEED_BASE = {"train": 0, "val": 2_000_000, "test": 3_000_000, "probe_ledge": 4_000_000}
HUMAN_SEED_BASE = 500_000_000_000  # above every other block; 900_000 collided with train shard 9

SHARDS = {"train": 175, "val": 12, "test": 12, "probe_ledge": 16}
EPISODES = {"train": 200, "val": 200, "test": 200, "probe_ledge": 150}
