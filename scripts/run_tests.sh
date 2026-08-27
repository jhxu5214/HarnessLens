#!/usr/bin/env bash
# Run the offline test suite. Tests that need a real agent runtime or provider
# are excluded; pass --live to include them.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$( [[ -x "$ROOT/.venv/bin/python" ]] && echo "$ROOT/.venv/bin/python" || echo python3 )}"

LIVE_TESTS=(
  tests/test_harness_editor_live.py
  tests/test_harness_query_live.py
  tests/test_native_candidate_rollout_live.py
)

args=()
if [[ "${1:-}" == "--live" ]]; then
  shift
else
  for path in "${LIVE_TESTS[@]}"; do
    args+=("--ignore=$path")
  done
fi

exec "$PYTHON" -m pytest "${args[@]}" "$@"
