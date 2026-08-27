from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

CODEX_HOOK_CONTEXT_PATH = ".codex/harness-hook-context.md"


def session_context(payload: Mapping[str, Any]) -> str:
    cwd = Path(str(payload.get("cwd") or os.getcwd())).resolve()
    path = cwd / CODEX_HOOK_CONTEXT_PATH
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        payload = {}
    context = session_context(payload if isinstance(payload, Mapping) else {})
    if not context:
        return 0
    cwd = Path(str(payload.get("cwd") or os.getcwd())).resolve()
    observation = cwd / ".codex" / "harness-hook-observation.json"
    observation.parent.mkdir(parents=True, exist_ok=True)
    observation.write_text(
        json.dumps(
            {
                "hook_event_name": str(payload.get("hook_event_name") or ""),
                "session_id": str(payload.get("session_id") or ""),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": context,
                },
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
