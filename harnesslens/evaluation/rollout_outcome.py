from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def provider_trace_error(path: str | Path) -> str:
    """Return an infrastructure error when a model call did not complete."""
    trace = Path(path)
    if not trace.is_file() or trace.stat().st_size <= 0:
        return "provider API trace is missing"
    completed_calls = 0
    try:
        lines = trace.read_text(encoding="utf-8", errors="replace").splitlines()
        for line_number, raw in enumerate(lines, 1):
            if not raw.strip():
                continue
            event = json.loads(raw)
            if not isinstance(event, Mapping):
                return f"provider API trace line {line_number} is not an object"
            if event.get("event") == "upstream_retry":
                continue
            if not isinstance(event.get("request"), Mapping):
                return f"provider API trace line {line_number} is missing its request"
            completed_calls += 1
            status = _status_code(event.get("upstream_status"))
            if status >= 400:
                detail = _provider_error_detail(event)
                return f"provider request failed with HTTP {status}: {detail}".rstrip()
            if event.get("response") is None:
                detail = _provider_error_detail(event)
                return f"provider request returned no response: {detail}".rstrip()
    except (OSError, json.JSONDecodeError) as exc:
        return f"provider API trace is unreadable: {type(exc).__name__}: {exc}"
    return "" if completed_calls else "provider API trace contains no completed calls"


def harness_execution_error(stdout: str, stderr: str) -> str:
    """Detect harness-level model failures hidden behind a zero process exit."""
    for raw in str(stdout or "").splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        failure = _stop_reason_error(event)
        if failure:
            return failure
    return ""


def _stop_reason_error(value: Any) -> str:
    if isinstance(value, Mapping):
        if str(value.get("stopReason") or value.get("stop_reason") or "").lower() == "error":
            detail = str(value.get("errorMessage") or value.get("error") or "").strip()
            return f"harness model response has stopReason=error: {detail}".rstrip()
        for child in value.values():
            failure = _stop_reason_error(child)
            if failure:
                return failure
    elif isinstance(value, list):
        for child in value:
            failure = _stop_reason_error(child)
            if failure:
                return failure
    return ""


def _status_code(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _provider_error_detail(event: Mapping[str, Any]) -> str:
    raw = event.get("error")
    if isinstance(raw, Mapping):
        return str(raw.get("code") or raw.get("type") or raw.get("message") or raw)
    text = str(raw or "").strip()
    if text:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return text[:500]
        error = payload.get("error") if isinstance(payload, Mapping) else None
        if isinstance(error, Mapping):
            return str(error.get("code") or error.get("type") or error.get("message") or error)
        return text[:500]
    return "upstream request failed"
