"""Driver for the native agent harnesses (opencode, codex, pi).

Spawns a harness binary under a captured environment, runs the provider proxy
beside it, and parses whatever session format that harness emits back into
``HarnessTurn`` records.

This began as private code inside ``claw_eval``, which ``bird_eval`` imported
sixteen names out of — so a BIRD run silently depended on a file named after a
different benchmark. The Claw-Eval cells have since been removed; BIRD is the
only consumer today, but the layer stays separate because it is about driving a
harness, not about any one benchmark. ``runtime_files()`` lists it for the bird
kind, since this file decides how a BIRD rollout behaves.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_OPENCODE_PREFIX = "~/.opencode"


def default_opencode_prefix() -> Path:
    """Filesystem prefix of a local OpenCode install (``OPENCODE_PREFIX``)."""
    return Path(os.environ.get("OPENCODE_PREFIX") or DEFAULT_OPENCODE_PREFIX).expanduser()

@dataclass(frozen=True)
class HarnessTurn:
    text: str
    stdout: str
    stderr: str
    session_id: str | None
    returncode: int | None
    elapsed_s: float

def start_harness_proxy(
    *,
    repo_root: Path,
    harness: str,
    socket_path: Path,
    artifact_root: Path,
) -> tuple[subprocess.Popen[str] | None, int | None]:
    if harness in {"opencode", "pi"}:
        return None, None
    # opencode and pi returned above; codex is the only harness left that needs
    # a translating proxy in front of the provider.
    command = [
        sys.executable,
        str(
            Path(__file__).resolve().parents[1]
            / "infrastructure"
            / "codex_responses_proxy.py"
        ),
        "--log-file",
        str(artifact_root / "codex_proxy_usage.jsonl"),
        "--context-log",
        str(artifact_root / "codex_api_calls.jsonl"),
        "--tau2-socket",
        str(socket_path),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ,
        start_new_session=True,
    )
    assert process.stdout is not None
    line = process.stdout.readline().strip()
    if not line.startswith("PORT="):
        stderr = process.stderr.read() if process.stderr else ""
        stop_process(process)
        raise RuntimeError(f"{harness} proxy failed: {line} {stderr[-1000:]}")
    return process, int(line.split("=", 1)[1])

def apply_manifest_prompt(system_prompt: str, manifest: Mapping[str, Any]) -> str:
    additions = [
        str(value).strip()
        for key in ("instructions", "prompt_appends")
        for value in manifest.get(key) or []
        if str(value).strip()
    ]
    return system_prompt + ("\n\n" + "\n\n".join(additions) if additions else "")

def run_process(
    command: Sequence[str], *, cwd: Path, env: Mapping[str, str], timeout_s: int
) -> tuple[str, str, int | None]:
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env=dict(env),
            start_new_session=True,
        )
        stdout, stderr = process.communicate(timeout=timeout_s)
        return stdout or "", stderr or "", process.returncode
    except subprocess.TimeoutExpired as exc:
        stop_process(process)
        stdout = (
            exc.stdout.decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode(errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        return stdout, stderr + f"\nTIMEOUT after {timeout_s}s", None

def parse_opencode_text(stdout: str) -> str:
    texts: list[str] = []
    for raw in stdout.splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if event.get("type") in {"text", "assistant"}:
            value = event.get("content", event.get("text", ""))
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())
        part = event.get("part")
        if isinstance(part, Mapping) and part.get("type") == "text":
            value = part.get("text")
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())
    return "\n".join(dict.fromkeys(texts))

def parse_opencode_session(stdout: str) -> str | None:
    for raw in stdout.splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        value = event.get("sessionID") or event.get("session_id")
        if value:
            return str(value)
    return None

def parse_codex_session(stdout: str) -> str | None:
    for raw in stdout.splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        value = event.get("thread_id") or event.get("session_id")
        if value:
            return str(value)
    return None

def last_json(stdout: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdout)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    for raw in reversed(stdout.splitlines()):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}

def drop_unsupported_socks_proxy(env: Any) -> None:
    for name in ("ALL_PROXY", "all_proxy"):
        if (
            str(env.get(name) or "")
            .lower()
            .startswith(("socks://", "socks4://", "socks5://", "socks5h://"))
        ):
            env.pop(name, None)

def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tasks = len(records)
    trials = sum(len(record.get("rewards") or []) for record in records)
    successes = sum(
        float(reward) >= 1.0
        for record in records
        for reward in record.get("rewards") or []
    )
    pass_at_1 = successes / trials if trials else 0.0
    return {
        "task_count": tasks,
        "trial_count": trials,
        "trial_success_count": successes,
        "trial_success_rate": pass_at_1,
        "pass_at_1": pass_at_1,
        "pass_at_2": (
            sum(
                any(float(value) >= 1.0 for value in (record.get("rewards") or [])[:2])
                for record in records
            )
            / tasks
            if tasks
            else 0.0
        ),
        "worker_error_count": sum(
            len(record.get("worker_errors") or []) for record in records
        ),
    }

def normalize_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "config_patch": dict(value.get("config_patch") or {}),
        "files": list(value.get("files") or []),
        "instructions": list(value.get("instructions") or []),
        "prompt_appends": list(value.get("prompt_appends") or []),
        "tool_desc_patches": dict(value.get("tool_desc_patches") or {}),
        "_workspace": dict(value.get("_workspace") or {}),
    }

def deep_merge(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result

def stable_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))

def opencode_binary() -> Path:
    candidates = [
        Path(str(os.environ.get("HAI_OPENCODE_BIN") or "")),
        default_opencode_prefix() / "bin" / "opencode",
    ]
    located = shutil.which("opencode")
    if located:
        candidates.insert(0, Path(located))
    for candidate in candidates:
        if str(candidate) and candidate.is_file():
            return candidate.resolve()
    raise RuntimeError("opencode executable is unavailable")

def stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
