#!/usr/bin/env bash
# Blind-TEST baseline for the terminal-bench cell.
#
# Task containers reach the provider through a local clash proxy, and `clashctl`
# is normally a shell function rather than an executable — hence the interactive
# subshell, which sources the user's rc file before turning the proxy on.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export ROOT
export HARNESSLENS_PYTHON="${PYTHON:-$( [[ -x "$ROOT/.venv/bin/python" ]] && echo "$ROOT/.venv/bin/python" || echo python )}"
exec bash -ic '
  clashctl on
  export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
  exec "$HARNESSLENS_PYTHON" "$ROOT/run_test_baseline.py" "$@"
' bash "$@"
