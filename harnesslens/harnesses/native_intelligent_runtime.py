from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from harnesslens.core.config import pi_binary, pi_install_root
from harnesslens.core.budget import CreationBudget
from harnesslens.infrastructure.process_isolation import (
    bubblewrap_command,
    isolated_child_env,
    node_runtime_root,
)


DEFAULT_CODEX_MODELS_CACHE = "~/.codex/models_cache.json"


def _codex_models_cache() -> Path:
    """Codex model-catalogue cache seeded into each sandboxed CODEX_HOME."""
    return Path(
        os.environ.get("HAI_CODEX_MODELS_CACHE") or DEFAULT_CODEX_MODELS_CACHE
    ).expanduser()


_PI_EDITOR_TOOLS = {"read", "edit", "write", "grep", "find", "ls"}
_CODEX_EDITOR_TOOLS = {"read", "edit", "write", "apply_patch", "bash"}
_MAX_DIAGNOSTIC_STDOUT_BYTES = 1024 * 1024


@dataclass(frozen=True)
class NativeRunResult:
    call_id: str
    outcome: str
    returncode: int
    stdout_path: str
    stderr_path: str
    api_trace_path: str
    usage: Mapping[str, int]
    validation_error: str = ""


class NativeIntelligentAdapter:
    def __init__(
        self,
        *,
        harness: str,
        model: str,
        context_limit: int,
        output_limit: int,
        max_steps: int,
        timeout_s: int,
        workspace_root: str | Path,
        model_options: Mapping[str, Any] | None = None,
        allowed_builtin_tools: tuple[str, ...] = (),
    ) -> None:
        normalized = str(harness).strip().lower().replace("-", "_")
        if normalized not in {"pi", "codex"}:
            raise ValueError(f"unsupported native intelligent harness: {harness}")
        normalized_tools = tuple(str(item) for item in allowed_builtin_tools)
        allowed_tools = _PI_EDITOR_TOOLS if normalized == "pi" else _CODEX_EDITOR_TOOLS
        if set(normalized_tools) - allowed_tools:
            raise ValueError(f"unsupported {normalized} intelligent tools")
        self.harness = normalized
        self.model = str(model)
        self.context_limit = int(context_limit)
        self.output_limit = int(output_limit)
        self.max_steps = int(max_steps)
        self.timeout_s = int(timeout_s)
        self.workspace_root = Path(workspace_root).resolve()
        self.model_options = dict(model_options or {})
        self.allowed_builtin_tools = normalized_tools
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        *,
        prompt: str,
        workspace: str | Path,
        call_id: str,
        budget: CreationBudget,
        max_steps: int | None = None,
        output_validator: Any = None,
        working_directory: str | Path | None = None,
    ) -> NativeRunResult:
        del output_validator
        steps = self.max_steps if max_steps is None else int(max_steps)
        if not 1 <= steps <= self.max_steps:
            raise ValueError("max_steps is outside the configured limit")
        reservation = steps * (self.context_limit + self.output_limit)
        record = budget.reserve(
            call_id,
            reservation,
            metadata={"harness": self.harness, "model": self.model, "max_steps": steps},
        )
        if record.get("status") != "reserved":
            raise ValueError(
                f"durable {self.harness} call ID is already {record.get('status')}"
            )
        budget.claim_launch(call_id)
        run_root = _validated_workspace(self.workspace_root, Path(workspace))
        run_root.mkdir(parents=True, exist_ok=True)
        run_cwd = _validated_workspace(
            self.workspace_root,
            Path(working_directory) if working_directory is not None else run_root,
        )
        run_cwd.mkdir(parents=True, exist_ok=True)
        proxy: subprocess.Popen[str] | None = None
        process: subprocess.Popen[str] | None = None
        raw_stdout_handle: io.TextIOWrapper | None = None
        raw_stderr_handle: io.TextIOWrapper | None = None
        raw_stdout_path = run_root / f".{self.harness}.raw.stdout"
        raw_stderr_path = run_root / f".{self.harness}.raw.stderr"
        stdin_prompt: str | None = None
        try:
            if self.harness == "pi":
                proxy, command, env, api_trace, config_path = self._prepare_pi(
                    run_root, prompt
                )
                stdin_prompt = prompt
            else:
                proxy, command, env, api_trace, config_path = self._prepare_codex(
                    run_root,
                    prompt,
                    editor_root=(run_cwd if working_directory is not None else None),
                )
                stdin_prompt = prompt
            if working_directory is not None:
                runtime_roots: list[Path] = []
                if self.harness == "pi":
                    runtime_roots.append(pi_install_root())
                node_root = node_runtime_root()
                if node_root is not None:
                    runtime_roots.append(node_root)
                required_env = {
                    key: value
                    for key, value in env.items()
                    if key
                    in {
                        "CODEX_HOME",
                        "HOME",
                        "PI_CODING_AGENT_DIR",
                        "PI_OFFLINE",
                        "PI_TELEMETRY",
                        "XDG_CACHE_HOME",
                        "XDG_CONFIG_HOME",
                        "XDG_DATA_HOME",
                    }
                }
                if self.harness == "codex":
                    required_env["OPENAI_API_KEY"] = "editor-local-proxy"
                env = isolated_child_env(env, overrides=required_env)
                command = bubblewrap_command(
                    command,
                    writable_root=run_root,
                    working_directory=run_cwd,
                    read_only_roots=runtime_roots,
                )
            (run_root / "invocation.json").write_text(
                json.dumps(
                    {
                        "command": command,
                        "cwd": str(run_cwd),
                        "harness": self.harness,
                        "model": self.model,
                        "max_steps": steps,
                        "api_trace_path": str(api_trace),
                        "config_path": str(config_path),
                        "prompt_transport": (
                            "stdin" if stdin_prompt is not None else "argv"
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            raw_stdout_handle = raw_stdout_path.open("w", encoding="utf-8")
            raw_stderr_handle = raw_stderr_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                stdin=(subprocess.PIPE if stdin_prompt is not None else None),
                stdout=(subprocess.PIPE if self.harness == "pi" else raw_stdout_handle),
                stderr=raw_stderr_handle,
                text=True,
                env=env,
                cwd=str(run_cwd),
                start_new_session=True,
            )
        except Exception as exc:
            _close_capture_handles(raw_stdout_handle, raw_stderr_handle)
            _stop(proxy)
            budget.refund_before_launch(call_id, reason=str(exc))
            raise
        budget.mark_launched(call_id)
        outcome = "completed"
        returned_stdout = ""
        returned_stderr = ""
        capture_thread: threading.Thread | None = None
        try:
            if self.harness == "pi":
                if process.stdout is None or process.stdin is None:
                    raise RuntimeError("Pi process capture pipes are unavailable")
                capture_thread = threading.Thread(
                    target=_capture_pi_terminal_events,
                    args=(process.stdout, raw_stdout_handle),
                    daemon=True,
                )
                capture_thread.start()
                process.stdin.write(stdin_prompt or "")
                process.stdin.close()
                process.wait(timeout=self.timeout_s)
            else:
                stdout, stderr = process.communicate(
                    input=stdin_prompt,
                    timeout=self.timeout_s,
                )
                returned_stdout = _text(stdout)
                returned_stderr = _text(stderr)
        except subprocess.TimeoutExpired as exc:
            outcome = "timeout"
            _kill_group(process)
            returned_stdout = _text(exc.stdout)
            returned_stderr = _text(exc.stderr)
        finally:
            if capture_thread is not None:
                capture_thread.join(timeout=10)
            _close_capture_handles(raw_stdout_handle, raw_stderr_handle)
            _stop(proxy)
        raw_stdout = returned_stdout or _read_capture(raw_stdout_path)
        stderr = returned_stderr or _read_capture(raw_stderr_path)
        raw_stdout_path.unlink(missing_ok=True)
        raw_stderr_path.unlink(missing_ok=True)
        parse_stdout = raw_stdout
        if self.harness == "pi":
            parse_stdout = compact_pi_stdout(raw_stdout)
        elif self.harness == "codex" and outcome == "completed":
            last_message_path = run_root / "last_message.txt"
            if last_message_path.is_file():
                last_message = last_message_path.read_text(
                    encoding="utf-8", errors="replace"
                )
                if last_message.strip():
                    parse_stdout = last_message
            (run_root / "codex.events.jsonl").write_text(raw_stdout, encoding="utf-8")
        stdout_path = run_root / f"{self.harness}.stdout"
        stderr_path = run_root / f"{self.harness}.stderr"
        stdout_path.write_text(parse_stdout, encoding="utf-8")
        stderr_path.write_text(stderr or "", encoding="utf-8")
        returncode = int(process.returncode or 0)
        if outcome == "completed" and returncode != 0:
            outcome = "nonzero_exit"
        usage = collect_native_usage(self.harness, raw_stdout)
        trace_complete = api_trace.is_file() and api_trace.stat().st_size > 0
        if outcome == "completed" and not trace_complete:
            outcome = "incomplete_trace"
        budget.settle(
            call_id,
            usage=usage,
            outcome=outcome,
            usage_complete=outcome == "completed" and bool(usage),
        )
        return NativeRunResult(
            call_id=str(call_id),
            outcome=outcome,
            returncode=returncode,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            api_trace_path=str(api_trace),
            usage=usage,
            validation_error="" if trace_complete else "complete API trace is missing",
        )

    def _prepare_pi(
        self, run_root: Path, prompt: str
    ) -> tuple[subprocess.Popen[str], list[str], dict[str, str], Path, Path]:
        del prompt
        executable = _pi_binary()
        api_trace = run_root / "api_calls.jsonl"
        proxy, port = _start_chat_proxy(api_trace, call_id=run_root.name)
        home = run_root / ".pi_home"
        home.mkdir(parents=True, exist_ok=True)
        config_path = home / "models.json"
        config_path.write_text(
            json.dumps(
                {
                    "providers": {
                        "harness-query": {
                            "baseUrl": f"http://127.0.0.1:{port}/v1",
                            "api": "openai-completions",
                            "apiKey": "probe",
                            "compat": {
                                "supportsDeveloperRole": False,
                                "supportsReasoningEffort": False,
                            },
                            "models": [
                                {
                                    "id": "deepseek-v4-flash",
                                    "reasoning": True,
                                    "contextWindow": self.context_limit,
                                    "maxTokens": self.output_limit,
                                }
                            ],
                        }
                    }
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        command = [
            str(executable),
            "--mode",
            "json",
            "--print",
            "--no-session",
            "--offline",
            "--provider",
            "harness-query",
            "--model",
            "deepseek-v4-flash",
            "--thinking",
            "high",
        ]
        if self.allowed_builtin_tools:
            command.extend(["--tools", ",".join(self.allowed_builtin_tools)])
        else:
            command.append("--no-tools")
        env = {
            **os.environ,
            "HOME": str(run_root / ".home"),
            "PI_CODING_AGENT_DIR": str(home),
            "PI_OFFLINE": "1",
            "PI_TELEMETRY": "0",
        }
        return proxy, command, env, api_trace, config_path

    def _prepare_codex(
        self,
        run_root: Path,
        prompt: str,
        *,
        editor_root: Path | None = None,
    ) -> tuple[subprocess.Popen[str], list[str], dict[str, str], Path, Path]:
        del prompt
        executable = shutil.which("codex")
        if not executable:
            raise RuntimeError("codex executable is unavailable")
        usage_log = run_root / "proxy_usage.jsonl"
        api_trace = run_root / "api_calls.jsonl"
        proxy, port = (
            _start_responses_proxy(
                usage_log,
                api_trace,
                workspace_editor_root=editor_root,
            )
            if editor_root is not None
            else _start_responses_proxy(usage_log, api_trace)
        )
        codex_home = run_root / ".codex_home"
        codex_home.mkdir(parents=True, exist_ok=True)
        config_path = codex_home / "config.toml"
        tools_enabled = bool(self.allowed_builtin_tools)
        shell_enabled = "bash" in self.allowed_builtin_tools
        shell_flag = "true" if shell_enabled else "false"
        sandbox_mode = "workspace-write" if tools_enabled else "read-only"
        config_path.write_text(
            f"""model = "gpt-5.4"
model_provider = "deepseek"
model_reasoning_effort = "high"
sandbox_mode = "{sandbox_mode}"
approval_policy = "never"
web_search = "disabled"

[model_providers.deepseek]
name = "DeepSeek via local Responses proxy"
base_url = "http://127.0.0.1:{port}/v1"
env_key = "OPENAI_API_KEY"
wire_api = "responses"

[features]
apps = false
plugins = false
shell_tool = {shell_flag}
unified_exec = false

""",
            encoding="utf-8",
        )
        source_cache = _codex_models_cache()
        if source_cache.is_file():
            shutil.copy2(source_cache, codex_home / "models_cache.json")
        last_message = run_root / "last_message.txt"
        command = [
            executable,
            "exec",
            "--strict-config",
            "--skip-git-repo-check",
            "--ignore-rules",
            "--ephemeral",
            "--json",
            "--output-last-message",
            str(last_message),
            "-",
        ]
        env = {
            **os.environ,
            "OPENAI_API_KEY": str(os.environ.get("DEEPSEEK_API_KEY") or ""),
            "CODEX_HOME": str(codex_home),
            "HOME": str(run_root / ".home"),
            "XDG_CONFIG_HOME": str(run_root / ".config"),
            "XDG_DATA_HOME": str(run_root / ".local" / "share"),
            "XDG_CACHE_HOME": str(run_root / ".cache"),
        }
        return proxy, command, env, api_trace, config_path


def collect_native_usage(harness: str, stdout: str) -> dict[str, int]:
    usage: dict[str, int] = {}
    for raw in io.StringIO(stdout):
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping):
            continue
        if harness == "pi" and event.get("type") == "message_end":
            message = (
                event.get("message")
                if isinstance(event.get("message"), Mapping)
                else {}
            )
            if message.get("role") != "assistant" or not isinstance(
                message.get("usage"), Mapping
            ):
                continue
            raw_usage = message["usage"]
            usage = {
                "total": int(raw_usage.get("totalTokens") or 0),
                "input": int(raw_usage.get("input") or 0),
                "output": int(raw_usage.get("output") or 0),
                "reasoning": int(raw_usage.get("reasoning") or 0),
                "cache_read": int(raw_usage.get("cacheRead") or 0),
                "cache_write": int(raw_usage.get("cacheWrite") or 0),
            }
        if harness == "codex" and event.get("type") == "turn.completed":
            raw_usage = (
                event.get("usage") if isinstance(event.get("usage"), Mapping) else {}
            )
            usage = {
                "total": int(raw_usage.get("input_tokens") or 0)
                + int(raw_usage.get("output_tokens") or 0),
                "input": int(raw_usage.get("input_tokens") or 0),
                "output": int(raw_usage.get("output_tokens") or 0),
                "reasoning": int(raw_usage.get("reasoning_output_tokens") or 0),
                "cache_read": int(raw_usage.get("cached_input_tokens") or 0),
                "cache_write": 0,
            }
    return usage


def compact_pi_stdout(stdout: str) -> str:
    final_assistant_event: Mapping[str, Any] | None = None
    for raw in io.StringIO(str(stdout)):
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping) or event.get("type") != "message_end":
            continue
        message = event.get("message")
        if isinstance(message, Mapping) and message.get("role") == "assistant":
            final_assistant_event = event
    if final_assistant_event is not None:
        return json.dumps(final_assistant_event, ensure_ascii=False) + "\n"
    return _bounded_utf8_tail(str(stdout), _MAX_DIAGNOSTIC_STDOUT_BYTES)


def _capture_pi_terminal_events(
    stream: io.TextIOBase,
    destination: io.TextIOBase,
) -> None:
    """Drop Pi's cumulative per-token snapshots while the child is running."""

    retained_types = {"message_end", "agent_end", "error"}
    for raw in stream:
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(event, Mapping) and str(event.get("type") or "") in retained_types:
            destination.write(json.dumps(event, ensure_ascii=False) + "\n")
            destination.flush()


def _bounded_utf8_tail(text: str, max_bytes: int) -> str:
    tail = str(text)[-int(max_bytes) :]
    encoded = tail.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return tail
    return encoded[-max_bytes:].decode("utf-8", errors="replace")


def _pi_binary() -> Path:
    return pi_binary()


def _start_chat_proxy(path: Path, *, call_id: str) -> tuple[subprocess.Popen[str], int]:
    script = (
        Path(__file__).resolve().parents[1]
        / "infrastructure"
        / "chat_completions_proxy.py"
    )
    key = str(os.environ.get("DEEPSEEK_API_KEY") or "")
    if not script.is_file() or not key:
        raise RuntimeError("Pi trace proxy prerequisites are unavailable")
    seed = (
        int.from_bytes(hashlib.sha256(call_id.encode()).digest()[:4], "big")
        & 0x7FFFFFFF
    )
    process = subprocess.Popen(
        [
            sys.executable,
            str(script),
            "--log-file",
            str(path),
            "--agent-seed",
            str(seed),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    ready = process.stdout.readline().strip() if process.stdout else ""
    if not ready.startswith("PORT="):
        error = process.stderr.read()[-1000:] if process.stderr else ""
        _stop(process)
        raise RuntimeError(f"Pi trace proxy failed: {ready} {error}")
    return process, int(ready.split("=", 1)[1])


def _start_responses_proxy(
    usage_log: Path,
    context_log: Path,
    *,
    workspace_editor_root: Path | None = None,
) -> tuple[subprocess.Popen[str], int]:
    script = (
        Path(__file__).resolve().parents[1]
        / "infrastructure"
        / "codex_responses_proxy.py"
    )
    key = str(os.environ.get("DEEPSEEK_API_KEY") or "")
    if not key:
        raise RuntimeError("Codex trace proxy prerequisites are unavailable")
    command = [
        sys.executable,
        str(script),
        "--log-file",
        str(usage_log),
        "--context-log",
        str(context_log),
    ]
    if workspace_editor_root is not None:
        command.extend(
            ["--workspace-editor-root", str(workspace_editor_root.resolve())]
        )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    ready = process.stdout.readline().strip() if process.stdout else ""
    if not ready.startswith("PORT="):
        error = process.stderr.read()[-1000:] if process.stderr else ""
        _stop(process)
        raise RuntimeError(f"Codex Responses proxy failed: {ready} {error}")
    return process, int(ready.split("=", 1)[1])


def _validated_workspace(root: Path, workspace: Path) -> Path:
    resolved = workspace.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("native workspace must be under workspace_root") from exc
    return resolved


def _kill_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def _stop(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def _close_capture_handles(*handles: io.TextIOWrapper | None) -> None:
    for handle in handles:
        if handle is not None and not handle.closed:
            handle.flush()
            handle.close()


def _read_capture(path: Path) -> str:
    return (
        path.read_text(encoding="utf-8", errors="replace")
        if path.is_file()
        else ""
    )


def _text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value or "")
