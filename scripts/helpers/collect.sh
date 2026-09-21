#!/usr/bin/env bash
# Local shard collection -- smoke tests only; the real corpus lives on Modal.
# aq runs .py tools with the kernel python, which has no pygame, so this is .sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec .venv/bin/python src/dataset/collect.py "$@"
