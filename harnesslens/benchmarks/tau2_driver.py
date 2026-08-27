"""Shared driver for running tau2 (τ²-bench) trials.

The opencode, codex and pi tau2 runners all need the same things: a configured
tau2 environment, agent-side tool definitions built from a domain policy, a user
simulator socket, per-trial workspace bookkeeping and trial-row I/O. That is
what lives here.

This file used to be called ``claude_code_tau2.py``: the claude-code runner was
written first and everything else grew inside it. That runner has since been
removed along with the harness, and what stayed is the part every remaining
runner imports. ``runtime_files()`` lists this module for the tau2 kind, because
it decides how a retail or banking rollout behaves.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
import uuid
import fcntl
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from harnesslens.core.config import provider_base_url
from harnesslens.core.artifacts import write_json
from harnesslens.benchmarks.benchmark_splits import BenchmarkSplit
from harnesslens.evaluation.rollout_bridge import RolloutRequest, RolloutResponse, TrainRolloutRecord


DEFAULT_SYSTEM_PROMPT = (
    "You are a customer service agent. Help the user with their request by calling "
    "the provided tools. Be helpful and thorough. Call at most one tool at a time."
)
MODEL = "deepseek-v4-flash"
USER_MODEL = "openai/deepseek-v4-flash"
TAU2_SOCKET_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class Tau2Limits:
    max_conversation_turns: int
    timeout_per_turn_s: int
    max_tool_calls_per_turn: int
    max_tool_calls: int
    group_timeout_s: int
    timeout_retries_per_turn: int = 1










def _is_clean_first_turn_timeout(
    turn_errors: Sequence[Mapping[str, Any]],
    total_tool_calls: int,
    messages: Sequence[Any],
) -> bool:
    if total_tool_calls:
        return False
    if not any(
        int(error.get("turn") or 0) == 0
        and "TIMEOUT" in str(error.get("stderr") or "")
        and not bool(error.get("retryable"))
        for error in turn_errors
    ):
        return False
    assistant_text = [
        str(getattr(message, "content", "") or "").strip()
        for message in messages
        if str(getattr(message, "role", "") or "") == "assistant"
    ]
    return not any(text for text in assistant_text[1:])


def _configure_tau2_deepseek_env() -> None:
    api_base = provider_base_url()
    os.environ["OPENAI_API_KEY"] = str(os.environ.get("DEEPSEEK_API_KEY") or "")
    os.environ["OPENAI_API_BASE"] = api_base
    os.environ["OPENAI_BASE_URL"] = api_base


def _tau2_llm_settings() -> dict[str, str]:
    api_base = provider_base_url()
    return {
        "model": str(os.environ.get("HAI_TAU2_LLM_MODEL") or USER_MODEL),
        "api_base": api_base,
        "api_key": str(os.environ["DEEPSEEK_API_KEY"]),
    }


def _configure_tau2_llm_runtime() -> dict[str, str]:
    settings = _tau2_llm_settings()
    try:
        import tau2.config as tau2_config
        from tau2.evaluator import evaluator_nl_assertions

        tau2_config.DEFAULT_LLM_NL_ASSERTIONS = settings["model"]
        tau2_config.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {
            **dict(tau2_config.DEFAULT_LLM_NL_ASSERTIONS_ARGS),
            "api_base": settings["api_base"],
            "api_key": settings["api_key"],
        }
        evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = settings["model"]
        evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS = dict(
            tau2_config.DEFAULT_LLM_NL_ASSERTIONS_ARGS
        )
    except Exception:  # noqa: BLE001
        pass
    return settings


def _tau2_env_kwargs(retrieval_config: str | None, task_obj: Any | None = None) -> dict[str, Any]:
    if not retrieval_config:
        return {}
    kwargs: dict[str, Any] = {"retrieval_variant": retrieval_config}
    if task_obj is not None:
        kwargs["task"] = task_obj
    return kwargs


def _tau2_env_for_task(domain: str, retrieval_config: str | None, task_obj: Any) -> Any:
    from tau2 import registry

    constructor = registry.get_env_constructor(domain)
    return constructor(solo_mode=False, **_tau2_env_kwargs(retrieval_config, task_obj))


def _load_tau2_task(domain: str, task_id: str, *, task_split: str) -> Any:
    from tau2.runner import load_tasks

    split_order = [task_split]
    for fallback in ("train", "test", "base"):
        if fallback not in split_order:
            split_order.append(fallback)
    load_errors: dict[str, str] = {}
    tasks: list[Any] = []
    for split_name in split_order:
        try:
            loaded = load_tasks(domain, task_split_name=split_name)
        except Exception as exc:  # noqa: BLE001
            load_errors[split_name] = f"{type(exc).__name__}: {exc}"
            continue
        tasks.extend(loaded)
        for task in loaded:
            if str(task.id) == str(task_id):
                return task
    raise ValueError(
        f"Task {task_id} not found in {domain} "
        f"(requested split={task_split}, tried={split_order}, load_errors={load_errors})"
    )


def _agent_tool_definitions(env_for_policy: Any) -> list[dict[str, Any]]:
    definitions = []
    try:
        for tool in env_for_policy.get_tools():
            fn = tool.openai_schema.get("function", {})
            definitions.append(
                {
                    "name": fn.get("name"),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }
            )
    except Exception:  # noqa: BLE001
        return []
    return definitions


def _stable_simulation_seed(domain: str, task_id: str, trial: int) -> int:
    payload = f"{domain}\0{task_id}\0{int(trial)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def _serialize_reward_info(reward_info: Any, *, evaluator_model: str) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema": "tau2.reward_info.v1",
        "available": reward_info is not None,
        "source": "online_evaluator",
        "complete": reward_info is not None,
        "model": evaluator_model,
    }
    if reward_info is None:
        return base
    if hasattr(reward_info, "model_dump"):
        payload = reward_info.model_dump(mode="json")
    elif isinstance(reward_info, Mapping):
        payload = dict(reward_info)
    else:
        payload = {
            key: value
            for key, value in vars(reward_info).items()
            if not key.startswith("_")
        }
    base.update(payload)
    return base




def _start_tau2_server(
    *,
    repo_root: Path,
    domain: str,
    task_id: str,
    socket_path: str,
    log_file: Path,
    max_steps: int,
    retrieval_config: str | None,
    tool_desc_patches: Mapping[str, Any] | None = None,
) -> subprocess.Popen[str]:
    command = [
        str(repo_root / "third_party" / "tau3-bench" / ".venv" / "bin" / "python3"),
        str(repo_root / "harnesslens" / "tau2_mcp_server.py"),
        "server",
        "--domain",
        domain,
        "--task-id",
        str(task_id),
        "--socket",
        socket_path,
        "--log-file",
        str(log_file),
        "--max-steps",
        str(max_steps),
    ]
    if retrieval_config:
        command.extend(["--retrieval-config", retrieval_config])
    if tool_desc_patches:
        patch_path = log_file.with_name(f"{log_file.stem}.tool_patches.json")
        patch_path.write_text(
            json.dumps(dict(tool_desc_patches), ensure_ascii=False),
            encoding="utf-8",
        )
        command.extend(["--tool-desc-patches", str(patch_path)])
    env = {
        **os.environ,
        "HAI_REPO_ROOT": str(repo_root),
        "PYTHONPATH": os.pathsep.join(
            [
                str(repo_root),
                str(repo_root / "third_party" / "tau3-bench" / "src"),
                os.environ.get("PYTHONPATH", ""),
            ]
        ),
    }
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    line = process.stdout.readline().strip()
    if line != "READY":
        stderr = process.stderr.read()
        raise RuntimeError(f"tau2 MCP server failed: {line} {stderr[:1000]}")
    return process






def _execute_user_tool_calls(socket_path: str, tool_calls: Sequence[Any]) -> list[Any]:
    from tau2.data_model.message import ToolMessage

    messages = []
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(TAU2_SOCKET_TIMEOUT_S)
    try:
        sock.connect(socket_path)
        for tool_call in tool_calls:
            try:
                request = json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 999,
                        "method": "tools/call",
                        "params": {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                            "_requestor": "user",
                        },
                    }
                ).encode()
                sock.sendall(struct.pack(">I", len(request)) + request)
                response = json.loads(_recv_socket(sock).decode())
                result = response.get("result", {}).get("content", [{}])[0].get("text", "")
            except Exception as exc:  # noqa: BLE001
                result = f"Error: {exc}"
            messages.append(
                ToolMessage(role="tool", id=tool_call.id, content=result, requestor="user")
            )
    finally:
        sock.close()
    return messages


def _reset_tool_step_window(socket_path: str) -> None:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(TAU2_SOCKET_TIMEOUT_S)
    try:
        sock.connect(socket_path)
        request = json.dumps(
            {"jsonrpc": "2.0", "id": 1001, "method": "harness/reset_step_window"}
        ).encode()
        sock.sendall(struct.pack(">I", len(request)) + request)
        _recv_socket(sock)
    except Exception:  # noqa: BLE001
        return
    finally:
        sock.close()


def _recv_socket(sock: socket.socket) -> bytes:
    raw_len = b""
    while len(raw_len) < 4:
        chunk = sock.recv(4 - len(raw_len))
        if not chunk:
            raise ConnectionError("socket closed while reading frame length")
        raw_len += chunk
    length = struct.unpack(">I", raw_len)[0]
    data = b""
    while len(data) < length:
        chunk = sock.recv(min(65536, length - len(data)))
        if not chunk:
            raise ConnectionError("socket closed while reading frame body")
        data += chunk
    return data


def _parse_agent_text(stdout: str) -> str:
    text = ""
    for raw in stdout.splitlines():
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, Mapping) and obj.get("result"):
            text = str(obj.get("result") or "")
    try:
        obj = json.loads(stdout)
        if isinstance(obj, Mapping) and obj.get("result"):
            text = str(obj.get("result") or "")
    except json.JSONDecodeError:
        pass
    return text






def _cleanup_trial_workspace(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:  # noqa: BLE001
        pass


@contextmanager
def _request_file_lock(request_root: Path):
    lock_path = request_root / ".run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            lock_file.seek(0)
            lock_file.truncate()
            lock_file.write(f"{os.getpid()}\n")
            lock_file.flush()
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)




def _read_calls(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return list(payload) if isinstance(payload, list) else []


def _serialize_message(message: Any) -> dict[str, Any]:
    role = getattr(message, "role", "")
    payload = {"role": str(role), "content": getattr(message, "content", None) or ""}
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in tool_calls
        ]
    if getattr(message, "id", None):
        payload["tool_call_id"] = getattr(message, "id")
    return payload


def _stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            process.kill()
    except Exception:  # noqa: BLE001
        try:
            process.terminate()
        except Exception:  # noqa: BLE001
            pass


def _summarize(records: Sequence[TrainRolloutRecord]) -> dict[str, Any]:
    n_tasks = len(records)
    trial_count = sum(len(record.rewards) for record in records)
    pass_at_1 = sum(
        float(reward) >= 1.0 for record in records for reward in record.rewards
    )
    pass_at_2 = sum(1 for record in records if max(record.rewards or (0.0,)) >= 1)
    return {
        "task_count": n_tasks,
        "pass_at_1_count": pass_at_1,
        "pass_at_2_count": pass_at_2,
        "pass_at_1": pass_at_1 / trial_count if trial_count else 0.0,
        "pass_at_2": pass_at_2 / n_tasks if n_tasks else 0.0,
        "mean_reward": (
            sum(sum(record.rewards) for record in records)
            / max(1, sum(len(record.rewards) for record in records))
        ),
    }


def _summarize_cache(path: Path) -> dict[str, Any]:
    rows = []
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        pass
    total = sum(int(row.get("total_input") or 0) for row in rows)
    read = sum(int(row.get("cache_read_input_tokens") or 0) for row in rows)
    create = sum(int(row.get("cache_creation_input_tokens") or 0) for row in rows)
    return {
        "call_count": len(rows),
        "total_input_tokens": total,
        "cache_read_input_tokens": read,
        "cache_creation_input_tokens": create,
        "cache_hit_rate": read / total if total else 0.0,
    }


def _load_cached_response(path: Path) -> RolloutResponse | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = tuple(
        TrainRolloutRecord(
            task_id=str(item["task_id"]),
            rewards=tuple(float(value) for value in item.get("rewards", [])),
            harness_version=str(item.get("harness_version") or "v0"),
            trajectory_paths=tuple(str(value) for value in item.get("trajectory_paths", [])),
            worker_errors=tuple(item.get("worker_errors") or ()),
            trial_summaries=tuple(item.get("trial_summaries") or ()),
        )
        for item in payload.get("records", [])
    )
    return RolloutResponse(
        request_id=str(payload["request"]["request_id"]),
        harness_version=str(payload["request"]["harness_version"]),
        budget_spent=int(payload.get("budget_spent") or 0),
        budget_remaining=int(payload.get("budget_remaining") or 0),
        trajectory_root=str(path.parent / "trajectories"),
        summary_json=str(path.parent / "summary.json"),
        metadata_json=str(path),
        metrics=dict(payload.get("metrics") or {}),
        per_task=dict(payload.get("per_task") or {}),
        records=records,
        scope="TEST",
    )


def _trial_path(trajectory_root: Path, task_id: str, trial: int) -> Path:
    return trajectory_root / str(task_id) / f"trial_{int(trial) + 1:04d}.jsonl"


def _write_trial_row(
    trajectory_root: Path, task_id: str, trial: int, row: Mapping[str, Any]
) -> Path:
    path = _trial_path(trajectory_root, task_id, trial)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _load_existing_trial_rows(
    trajectory_root: Path, task_id: str, repeats: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in range(int(repeats)):
        path = _trial_path(trajectory_root, task_id, trial)
        try:
            loaded = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError):
            continue
        if not loaded or not isinstance(loaded[-1], dict):
            continue
        row = dict(loaded[-1])
        if str(row.get("task_id")) != str(task_id):
            continue
        if row.get("error") or not row.get("termination"):
            continue
        row["trial"] = int(row.get("trial") or trial)
        rows.append(row)
    return rows


def _fs_tag(domain: str, task_id: str, trial: int) -> str:
    raw = f"{domain}_{task_id}_t{trial}"
    clean = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)
    return clean[:120] + "_" + hashlib.sha1(raw.encode()).hexdigest()[:8]


def _sort_key(value: str) -> tuple[int, Any]:
    text = str(value)
    if text.isdigit():
        return (0, int(text))
    return (1, text)
