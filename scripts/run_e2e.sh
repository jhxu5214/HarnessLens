#!/usr/bin/env bash
# Run one full auto-iteration: baseline -> discovery -> experience -> analyzer
# -> candidate -> paired TRAIN rollout -> promotion -> submission.
#
#   scripts/run_e2e.sh --run-id my-run --cell retail --harness opencode
#
# The run is resumable: re-invoking with the same --run-id continues from the
# last checkpoint under runs/train/<run-id>/. Every flag is forwarded to
# run_e2e.py verbatim, so `--help` shows the full list.
#
# Set SKIP_PREFLIGHT=1 to bypass scripts/check_env.py.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$( [[ -x "$ROOT/.venv/bin/python" ]] && echo "$ROOT/.venv/bin/python" || echo python3 )}"

# Pull --cell/--harness out of the argument list so preflight checks the right
# thing; both are still forwarded to run_e2e.py below.
cell="retail"
harness="opencode"
previous=""
for arg in "$@"; do
  case "$previous" in
    --cell) cell="$arg" ;;
    --harness) harness="$arg" ;;
  esac
  case "$arg" in
    --cell=*) cell="${arg#*=}" ;;
    --harness=*) harness="${arg#*=}" ;;
  esac
  previous="$arg"
done

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  if ! "$PYTHON" scripts/check_env.py --cell "$cell" --harness "$harness"; then
    echo >&2
    echo "error: preflight failed. Fix the items above or set SKIP_PREFLIGHT=1." >&2
    exit 1
  fi
  echo
fi

# terminal-bench rolls out inside containers that reach the provider through a
# local clash proxy; run_e2e.py configures it, but the controller expects the
# proxy binary to be reachable.
exec "$PYTHON" run_e2e.py "$@"
