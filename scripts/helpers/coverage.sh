#!/usr/bin/env bash
# Report the corpus coverage structure -- the number this project is about.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
exec .venv/bin/python scripts/tests/report_coverage.py "$@"
