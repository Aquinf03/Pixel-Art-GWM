#!/usr/bin/env bash
# Collect a split on Modal. Usage: aq tool modal_collect -- --split train --shards 130
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
exec modal run src/dataset/modal_app.py::main "$@"
