#!/usr/bin/env bash
# Create the orchestrator virtualenv and install HarnessLens' own dependencies.
#
# This does NOT install the benchmarks or the agent runtimes: those are large
# external checkouts with their own virtualenvs. See docs/benchmarks.md and
# run scripts/check_env.py afterwards to see what is still missing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-$ROOT/.venv}"

if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "error: HarnessLens needs Python 3.11 or newer (tomllib, PEP 604 unions)." >&2
  echo "       Found: $("$PYTHON" --version 2>&1)" >&2
  exit 1
fi

if [[ ! -d "$VENV" ]]; then
  echo "==> creating virtualenv at $VENV"
  "$PYTHON" -m venv "$VENV"
fi

echo "==> installing HarnessLens and dependencies"
"$VENV/bin/pip" install --upgrade pip >/dev/null
"$VENV/bin/pip" install -e ".[tau2,dev]"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "==> wrote .env from .env.example — fill in DEEPSEEK_API_KEY before running"
fi

echo
echo "==> done. Next steps:"
echo "    1. edit .env and set DEEPSEEK_API_KEY"
echo "    2. provide the benchmark checkouts under third_party/ (docs/benchmarks.md)"
echo "    3. $VENV/bin/python scripts/check_env.py --cell retail"
