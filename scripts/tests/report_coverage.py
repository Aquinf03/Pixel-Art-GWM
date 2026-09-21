"""Report the corpus coverage structure.

Coverage is the whole subject of the paper this project is built on, so it is
worth reading straight off the corpus rather than inferring it. Pulls each
split's manifest out of the Modal Volume and prints episode share against
frame share -- the mismatch between those two is the structure that makes
hallucination predictable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

VOLUME = "pixel-world-data"
SPLITS = ["train", "val", "test", "probe_ledge"]
TASKS = ["P0 ground", "P1", "P2", "P3 ledge"]


def fetch(split: str, dest: Path) -> dict | None:
    out = dest / f"{split}.json"
    r = subprocess.run(
        ["modal", "volume", "get", VOLUME, f"{split}/manifest.json", str(out)],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not out.exists():
        return None
    return json.loads(out.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    a = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        dest = Path(td)
        for split in a.splits:
            m = fetch(split, dest)
            if m is None:
                print(f"{split}: no manifest (not collected yet)\n")
                continue
            print(f"== {split} ==")
            print(f"   {m['shards']} shards  {m['episodes']:,} episodes  "
                  f"{m['frames']:,} frames  {m['gb']} GB  "
                  f"({m['bytes_per_frame']} B/frame)")
            eps = m.get("task_episode_share", {})
            frm = m.get("task_frame_share", {})
            if eps:
                print(f"   {'task':<10}{'episodes':>10}{'frames':>10}{'ratio':>8}")
                for k in sorted(eps, key=int):
                    e, f = eps[k], frm.get(k, 0.0)
                    ratio = f / e if e else 0.0
                    print(f"   {TASKS[int(k)]:<10}{e:>9.1%}{f:>10.1%}{ratio:>8.2f}")
                print("   ratio > 1 means the task eats more frames than its "
                      "episode share -- long episodes.")
            print()


if __name__ == "__main__":
    main()
