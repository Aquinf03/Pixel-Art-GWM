#!/usr/bin/env bash
# Staged run with gates. Each stage only starts if the previous one passed --
# the stages compose sequentially, so a bad tokenizer would silently waste the
# dynamics run.
#
#   ./run_stages.sh                 # stage 1: tokenizer train + eval
#   ./run_stages.sh /path/to.log    # wait for stage-0 VERDICT: PASS, then stage 1
set -uo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"

STAGE0_OUT="${1:-}"

if [ -n "$STAGE0_OUT" ]; then
  echo "=== STAGE 0: overfit sanity gate ==="
  for i in $(seq 1 240); do
    grep -q "VERDICT" "$STAGE0_OUT" 2>/dev/null && break
    sleep 15
  done
  grep -E "palette_acc|psnr\"|VERDICT|returncode|steps_per_s" "$STAGE0_OUT" 2>/dev/null | tail -12

  if ! grep -q "VERDICT: PASS" "$STAGE0_OUT" 2>/dev/null; then
    echo
    echo "GATE FAILED or timed out -- stage 1 NOT started."
    echo "A 21.4M model that cannot memorise 256 frames has a broken config;"
    echo "running 60k steps would waste hours proving the same thing."
    exit 1
  fi
  echo
fi

echo "=== STAGE 1: tokenizer, 60k steps, checkpoints every 6k ==="; date
aq train . 2>&1 | tail -50
echo
echo "=== STAGE 1 GATE: palette accuracy on held-out val ==="; date
aq eval . 2>&1 | tail -30
echo
echo "=== run record ==="
aq status . 2>&1 | head -20
