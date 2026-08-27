from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from harnesslens.core.budget import CreationBudget
from harnesslens.infrastructure.process_isolation import (
    bubblewrap_command,
    isolated_child_env,
)
from harnesslens.evaluation.rollout_outcome import provider_trace_error


DEFAULT_OPENCODE_PREFIX = "~/.opencode"
DEFAULT_UPSTREAM_BASE_URL = "https://api.deepseek.com/v1"


_BUILTIN_TOOLS = (
    "read",
    "glob",
    "grep",
    "list",
    "write",
    "edit",
    "apply_patch",
    "patch",
    "bash",
    "webfetch",
    "websearch",
    "task",
    "skill",
    "lsp",
    "question",
    "todowrite",
    "todoread",
)


@dataclass(frozen=True)
class OpenCodeRunResult:
    call_id: str
    outcome: str
    returncode: int
    stdout_path: str
    stderr_path: str
    api_trace_path: str
    usage: Mapping[str, int]
    validation_error: str = ""


class OpenCodeIntelligentAdapter:
    def __init__(
        self,
        *,
        model: str,
        context_limit: int,
        output_limit: int,
        max_steps: int,
        timeout_s: int,
        workspace_root: str | Path,
        model_options: Mapping[str, Any] | None = None,
        allowed_builtin_tools: tuple[str, ...] = (),
    ) -> None:
        self.model = str(model)
        self.context_limit = int(context_limit)
        self.output_limit = int(output_limit)
        self.max_steps = int(max_steps)
        self.timeout_s = int(timeout_s)
        self.workspace_root = Path(workspace_root).resolve()
        self.model_options = dict(model_options or {})
        self.allowed_builtin_tools = tuple(str(item) for item in allowed_builtin_tools)
        if set(self.allowed_builtin_tools) - set(_BUILTIN_TOOLS):
            raise ValueError("unsupported OpenCode built-in tool")
        if min(self.context_limit, self.output_limit, self.max_steps, self.timeout_s) <= 0:
            raise ValueError("OpenCode limits must be positive")
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        if self.workspace_root.is_symlink():
            raise ValueError("OpenCode workspace root cannot be a symlink")

    def run(
        self,
        *,
        prompt: str,
        workspace: str | Path,
        call_id: str,
        budget: CreationBudget,
        max_steps: int | None = None,
        output_validator: Callable[[str], Any] | None = None,
        working_directory: str | Path | None = None,
    ) -> OpenCodeRunResult:
        steps = self.max_steps if max_steps is None else int(max_steps)
        if not 1 <= steps <= self.max_steps:
            raise ValueError("max_steps is outside the configured limit")
        reservation = steps * (self.context_limit + self.output_limit)
        record = budget.reserve(
            call_id,
            reservation,
            metadata={"harness": "opencode", "model": self.model, "max_steps": steps},
        )
        if record.get("status") != "reserved":
            raise ValueError(f"durable OpenCode call ID is already {record.get('status')}")
        budget.claim_launch(call_id)
        proxy: subprocess.Popen[str] | None = None
        run_root: Path | None = None
        api_trace: Path | None = None
        try:
            run_root = _validated_workspace(self.workspace_root, Path(workspace))
            run_root.mkdir(parents=True, exist_ok=True)
            run_cwd = _validated_workspace(
                self.workspace_root,
                Path(working_directory) if working_directory is not None else run_root,
            )
            run_cwd.mkdir(parents=True, exist_ok=True)
            api_trace = run_root / "api_calls.jsonl"
            proxy, port = _start_proxy(run_root=run_root, call_id=call_id)
            config_path = write_opencode_config(
                run_root,
                model=self.model,
                context_limit=self.context_limit,
                output_limit=self.output_limit,
                max_steps=steps,
                model_options=self.model_options,
                allowed_builtin_tools=self.allowed_builtin_tools,
                base_url=f"http://127.0.0.1:{port}/v1",
            )
            env = {
                **os.environ,
                "XDG_DATA_HOME": str(run_root / ".oc_data"),
                "XDG_CONFIG_HOME": str(run_root / ".oc_config"),
                "XDG_STATE_HOME": str(run_root / ".oc_state"),
                "OPENCODE_CONFIG": str(config_path),
                "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
                "OPENCODE_DISABLE_AUTOCOMPACT": "1",
                "OPENCODE_DISABLE_PRUNE": "1",
            }
            _prepend_opencode_path(env)
            executable = shutil.which("opencode", path=env.get("PATH"))
            if not executable:
                raise RuntimeError("OpenCode executable is unavailable")
            command = [
                executable,
                "run",
                "-m",
                self.model,
                "--pure",
                "--auto",
                "--title",
                str(call_id),
                "--dir",
                str(run_cwd),
                "--format",
                "json",
            ]
            if working_directory is not None:
                env = isolated_child_env(
                    env,
                    overrides={
                        "DEEPSEEK_API_KEY": "editor-local-proxy",
                        "HOME": str(run_root / ".home"),
                        "XDG_DATA_HOME": str(run_root / ".oc_data"),
                        "XDG_CONFIG_HOME": str(run_root / ".oc_config"),
                        "XDG_STATE_HOME": str(run_root / ".oc_state"),
                        "OPENCODE_CONFIG": str(config_path),
                        "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
                        "OPENCODE_DISABLE_AUTOCOMPACT": "1",
                        "OPENCODE_DISABLE_PRUNE": "1",
                    },
                )
                command = bubblewrap_command(
                    command,
                    writable_root=run_root,
                    working_directory=run_cwd,
                    read_only_roots=(Path(executable).resolve().parents[4],),
                )
            (run_root / "invocation.json").write_text(
                json.dumps(
                    {
                        "command": command,
                        "cwd": str(run_cwd),
                        "model": self.model,
                        "max_steps": steps,
                        "api_trace_path": str(api_trace),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=str(run_cwd),
                start_new_session=True,
            )
        except Exception as exc:
            _stop_process(proxy)
            budget.refund_before_launch(call_id, reason=str(exc))
            if isinstance(exc, (ValueError, RuntimeError)):
                raise
            raise RuntimeError(f"failed to launch HarnessLens OpenCode call: {exc}") from exc
        budget.mark_launched(call_id)
        try:
            stdout, stderr, outcome = _stream_process(
                process,
                prompt=str(prompt),
                timeout_s=self.timeout_s,
            )
        finally:
            _stop_process(proxy)
        assert run_root is not None and api_trace is not None
        stdout_path = run_root / "opencode.stdout"
        stderr_path = run_root / "opencode.stderr"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        if outcome == "completed" and int(process.returncode or 0) != 0:
            outcome = "nonzero_exit"
        validation_error = ""
        if outcome == "completed" and output_validator is not None:
            try:
                output_validator(stdout)
            except Exception as exc:
                outcome = "malformed_output"
                validation_error = str(exc)
        trace_error = validate_api_trace(api_trace)
        if trace_error:
            outcome = "incomplete_trace"
            validation_error = trace_error
        usage = collect_opencode_usage(stdout) or {}
        budget.settle(
            call_id,
            usage=usage,
            outcome=outcome,
            usage_complete=outcome in {"completed", "malformed_output"} and bool(usage),
        )
        return OpenCodeRunResult(
            call_id=str(call_id),
            outcome=outcome,
            returncode=int(process.returncode or 0),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            api_trace_path=str(api_trace),
            usage=usage,
            validation_error=validation_error,
        )


def write_opencode_config(
    workspace: Path,
    *,
    model: str,
    context_limit: int,
    output_limit: int,
    max_steps: int,
    model_options: Mapping[str, Any] | None = None,
    allowed_builtin_tools: tuple[str, ...] = (),
    base_url: str | None = None,
) -> Path:
    if "/" not in model:
        raise ValueError("OpenCode model must use provider/model format")
    provider_id, model_id = model.split("/", 1)
    allowed = set(allowed_builtin_tools)
    provider = {
        "models": {
            model_id: {
                "limit": {"context": int(context_limit), "output": int(output_limit)},
                **({"options": dict(model_options)} if model_options else {}),
            }
        },
        "npm": "@ai-sdk/openai-compatible",
        "name": "DeepSeek",
        "options": {
            "baseURL": str(
                base_url
                or os.environ.get("DEEPSEEK_BASE_URL")
                or os.environ.get("DEEPSEEK_URL")
                or DEFAULT_UPSTREAM_BASE_URL
            ).rstrip("/"),
            "apiKey": "{env:DEEPSEEK_API_KEY}",
        },
    }
    config = {
        "$schema": "https://opencode.ai/config.json",
        "autoupdate": False,
        "share": "disabled",
        "enabled_providers": [provider_id],
        "provider": {provider_id: provider},
        "model": model,
        "snapshot": False,
        "agent": {"build": {"steps": int(max_steps)}},
        "tools": {name: name in allowed for name in _BUILTIN_TOOLS},
        "permission": {
            **{name: "allow" if name in allowed else "deny" for name in _BUILTIN_TOOLS},
            "external_directory": "deny",
        },
    }
    path = workspace / "opencode.json"
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return path


def validate_api_trace(path: str | Path) -> str:
    error = provider_trace_error(path)
    return error.replace("provider API trace", "OpenCode API trace")


def collect_opencode_usage(stdout: str) -> dict[str, int] | None:
    usage = {"total": 0, "input": 0, "output": 0, "reasoning": 0, "cache_read": 0, "cache_write": 0}
    found = False
    for raw in stdout.splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping) or event.get("type") != "step_finish":
            continue
        part = event.get("part") if isinstance(event.get("part"), Mapping) else {}
        tokens = part.get("tokens") if isinstance(part.get("tokens"), Mapping) else None
        if not tokens or "total" not in tokens:
            return None
        cache = tokens.get("cache") if isinstance(tokens.get("cache"), Mapping) else {}
        usage["total"] += int(tokens.get("total") or 0)
        usage["input"] += int(tokens.get("input") or 0)
        usage["output"] += int(tokens.get("output") or 0)
        usage["reasoning"] += int(tokens.get("reasoning") or 0)
        usage["cache_read"] += int(cache.get("read") or 0)
        usage["cache_write"] += int(cache.get("write") or 0)
        found = True
    return usage if found else None


def _start_proxy(*, run_root: Path, call_id: str) -> tuple[subprocess.Popen[str], int]:
    script = (
        Path(__file__).resolve().parents[1]
        / "infrastructure"
        / "chat_completions_proxy.py"
    )
    key = str(os.environ.get("DEEPSEEK_API_KEY") or "")
    if not script.is_file() or not key:
        raise RuntimeError("OpenCode trace proxy prerequisites are unavailable")
    seed = int.from_bytes(
        hashlib.sha256(f"harnesslens-intelligent\0{call_id}".encode()).digest()[:4], "big"
    ) & 0x7FFFFFFF
    process = subprocess.Popen(
        [
            sys.executable,
            str(script),
            "--log-file",
            str(run_root / "api_calls.jsonl"),
            "--agent-seed",
            str(seed),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(os.environ),
    )
    ready = process.stdout.readline().strip() if process.stdout else ""
    if not ready.startswith("PORT="):
        error = process.stderr.read()[:500] if process.stderr else ""
        _stop_process(process)
        raise RuntimeError(f"OpenCode trace proxy failed: {ready} {error}")
    return process, int(ready.split("=", 1)[1])


def _stream_process(
    process: subprocess.Popen[str], *, prompt: str, timeout_s: int
) -> tuple[str, str, str]:
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    readers = [
        threading.Thread(target=_drain, args=(process.stdout, stdout_parts), daemon=True),
        threading.Thread(target=_drain, args=(process.stderr, stderr_parts), daemon=True),
    ]
    for reader in readers:
        reader.start()
    if process.stdin:
        process.stdin.write(prompt)
        process.stdin.close()
    outcome = "completed"
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        outcome = "timeout"
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
    for reader in readers:
        reader.join(timeout=10)
    return "".join(stdout_parts), "".join(stderr_parts), outcome


def _drain(stream: Any, destination: list[str]) -> None:
    if stream is None:
        return
    try:
        for chunk in iter(stream.readline, ""):
            destination.append(str(chunk))
    except (OSError, ValueError):
        return


def _stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    try:
        process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    except ProcessLookupError:
        return


def _validated_workspace(root: Path, requested: Path) -> Path:
    absolute = requested.absolute()
    try:
        relative = absolute.relative_to(root.absolute())
    except ValueError as exc:
        raise ValueError("OpenCode workspace is outside its configured root") from exc
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.exists() and cursor.is_symlink():
            raise ValueError("OpenCode workspace contains a symlink")
    resolved = absolute.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError("OpenCode workspace is outside its configured root")
    return resolved


def _prepend_opencode_path(env: dict[str, str]) -> None:
    bin_dir = str(
        Path(env.get("OPENCODE_PREFIX") or DEFAULT_OPENCODE_PREFIX).expanduser() / "bin"
    )
    path = env.get("PATH", "")
    if bin_dir not in path.split(":"):
        env["PATH"] = f"{bin_dir}:{path}" if path else bin_dir
