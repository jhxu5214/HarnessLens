from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from harnesslens.benchmarks.benchmark_splits import BenchmarkSplit
from harnesslens.harnesses.channel_preflight import (
    build_runtime_load_report,
    refresh_runtime_load_report,
)
from harnesslens.benchmarks.tau2_driver import (
    DEFAULT_SYSTEM_PROMPT,
    Tau2Limits,
    _agent_tool_definitions,
    _cleanup_trial_workspace,
    _configure_tau2_deepseek_env,
    _configure_tau2_llm_runtime,
    _execute_user_tool_calls,
    _fs_tag,
    _load_tau2_task,
    _read_calls,
    _request_file_lock,
    _reset_tool_step_window,
    _serialize_message,
    _serialize_reward_info,
    _stable_simulation_seed,
    _start_tau2_server,
    _stop_process,
    _tau2_env_for_task,
    _tau2_env_kwargs,
)
from harnesslens.harnesses.candidate_config_runtime import (
    compile_opencode_agent_definitions,
    relocate_opencode_instruction_paths,
)
from harnesslens.harnesses.harness_workspace import normalize_workspace_snapshot
from harnesslens.harnesses.native_candidate_runtime import (
    CANDIDATE_WORKSPACE_KEY,
    candidate_system_prompt,
    materialize_project_files,
)
from harnesslens.harnesses.opencode_harness import normalize_opencode_manifest
from harnesslens.harnesses.opencode_runtime import validate_api_trace
from harnesslens.benchmarks.pi_tau2 import _run_native_tau2_batch_locked
from harnesslens.core.profiles import (
    DEFAULT_OPENCODE_CONTEXT_LIMIT,
    DEFAULT_OUTPUT_LIMIT,
)
from harnesslens.evaluation.rollout_bridge import RolloutRequest, RolloutResponse


DEFAULT_OPENCODE_PREFIX = "~/.opencode"


def _default_opencode_prefix() -> Path:
    """Filesystem prefix of a local OpenCode install (``OPENCODE_PREFIX``)."""
    return Path(os.environ.get("OPENCODE_PREFIX") or DEFAULT_OPENCODE_PREFIX).expanduser()


MODEL = "deepseek-v4-flash"
OPENCODE_MODEL = f"deepseek/{MODEL}"


def _opencode_turn_retry_attempts() -> int:
    try:
        attempts = int(os.environ.get("HAI_OPENCODE_TURN_RETRY_ATTEMPTS", "0"))
    except ValueError:
        attempts = 0
    return max(0, min(attempts, 3))


def _is_retryable_empty_turn(
    *,
    stderr: str,
    returncode: int | None,
    agent_text: str,
    new_calls: Sequence[Mapping[str, Any]],
) -> bool:
    """Retry only before an assistant turn can have caused side effects."""
    if agent_text.strip() or new_calls:
        return False
    error = str(stderr or "").lower()
    if returncode is None and "timeout" in error:
        return True
    return any(
        marker in error
        for marker in (
            "http 429",
            "insufficient_quota",
            "limit_burst_rate",
            "http 500",
            "http 502",
            "http 503",
            "connection reset",
            "connection refused",
            "stream disconnected",
        )
    )


def run_opencode_tau2_test_baseline(
    *,
    repo_root: str | Path,
    run_root: str | Path,
    split: BenchmarkSplit,
    request: RolloutRequest,
    retrieval_config: str | None,
    limits: Tau2Limits,
    harness_manifest: Mapping[str, Any] | None = None,
) -> RolloutResponse:
    root = Path(repo_root).resolve()
    _configure_tau2_deepseek_env()
    request_root = (
        Path(run_root).resolve()
        / "rollout_artifacts"
        / request.run_id
        / request.request_id
    )
    request_root.mkdir(parents=True, exist_ok=True)
    with _request_file_lock(request_root):
        return _run_native_tau2_batch_locked(
            root=root,
            request_root=request_root,
            split=split,
            request=request,
            retrieval_config=retrieval_config,
            limits=limits,
            harness_manifest=harness_manifest,
            trial_runner=run_single_opencode_tau2_trial,
            harness_name="opencode",
            usage_summarizer=_summarize_opencode_usage,
            trajectory_retention=(
                "harnesslens_opencode_trial_jsonl_and_api_sidecars_retained_"
                "workspaces_cleaned"
            ),
            api_trace_required=True,
        )


def run_single_opencode_tau2_trial(
    *,
    repo_root: Path,
    request_root: Path,
    domain: str,
    task_id: str,
    trial: int,
    retrieval_config: str | None,
    limits: Tau2Limits,
    harness_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    tau2_src = repo_root / "third_party" / "tau3-bench" / "src"
    if str(tau2_src) not in sys.path:
        sys.path.insert(0, str(tau2_src))
    os.environ.setdefault(
        "TAU2_DATA_DIR", str(repo_root / "third_party" / "tau3-bench" / "data")
    )
    from tau2.data_model.message import (
        AssistantMessage,
        MultiToolMessage,
        ToolCall,
        ToolMessage,
        UserMessage,
    )
    from tau2.data_model.simulation import SimulationRun, TerminationReason
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.orchestrator.orchestrator import CommunicationMode
    from tau2.user.user_simulator import UserSimulator

    llm_settings = _configure_tau2_llm_runtime()
    work_dir = request_root / "workspaces" / _fs_tag(domain, task_id, trial)
    runtime_root = _opencode_runtime_root(request_root, domain, task_id, trial)
    runtime_cwd = runtime_root / "project"
    runtime_home = runtime_root / "home"
    work_dir.mkdir(parents=True, exist_ok=True)
    runtime_cwd.mkdir(parents=True, exist_ok=True)
    simulation_seed = _stable_simulation_seed(domain, str(task_id), int(trial))
    task_obj = _load_tau2_task(domain, task_id, task_split="test")
    env_for_policy = _tau2_env_for_task(domain, retrieval_config, task_obj)
    domain_policy = str(getattr(env_for_policy, "policy", "") or "")
    manifest = _opencode_manifest(harness_manifest)
    full_prompt = DEFAULT_SYSTEM_PROMPT + (
        "\n\n" + domain_policy if domain_policy else ""
    )
    candidate_append = "\n\n".join(
        str(item).strip()
        for item in [*manifest["instructions"], *manifest["prompt_appends"]]
        if str(item).strip()
    )
    agent_tool_defs = _agent_tool_definitions(env_for_policy)
    tag = _fs_tag(domain, task_id, trial)
    socket_path = f"/tmp/harnesslens_tau2_opencode_{tag[-18:]}.sock"
    calls_log = work_dir / "calls.json"
    api_trace = request_root / f"api_calls_{tag}.jsonl"
    calls_log.unlink(missing_ok=True)
    api_trace.unlink(missing_ok=True)
    server_proc = _start_tau2_server(
        repo_root=repo_root,
        domain=domain,
        task_id=task_id,
        socket_path=socket_path,
        log_file=calls_log,
        max_steps=int(limits.max_tool_calls_per_turn),
        retrieval_config=(retrieval_config if domain == "banking_knowledge" else None),
        tool_desc_patches=manifest["tool_desc_patches"],
    )
    proxy_proc: subprocess.Popen[str] | None = None
    messages: list[Any] = []
    total_tool_calls = 0
    prev_call_count = 0
    all_call_strs: list[str] = []
    calls_per_turn: list[list[str]] = []
    turn_errors: list[dict[str, Any]] = []
    stdout_events: list[list[dict[str, Any]]] = []
    session_id: str | None = None
    started = time.time()
    last_error = ""
    try:
        proxy_proc, proxy_port = _start_opencode_proxy(
            api_trace=api_trace,
            agent_seed=simulation_seed,
        )
        config_path = _write_opencode_project(
            repo_root=repo_root,
            runtime_cwd=runtime_cwd,
            runtime_home=runtime_home,
            socket_path=socket_path,
            proxy_port=proxy_port,
            system_prompt=full_prompt,
            max_steps=int(limits.max_tool_calls_per_turn),
            harness_manifest=manifest,
        )
        model_context = build_runtime_load_report(
            harness="opencode",
            project_root=runtime_cwd,
            home_root=runtime_home / ".config" / "opencode",
            manifest=manifest,
            tool_definitions=agent_tool_defs,
        )
        try:
            user_tools = (
                list(
                    env_for_policy.get_user_tools(
                        include=getattr(task_obj, "user_tools", None)
                    )
                )
                if hasattr(env_for_policy, "get_user_tools")
                else None
            )
        except (AttributeError, ValueError):
            user_tools = None
        user_sim = UserSimulator(
            llm=llm_settings["model"],
            instructions=str(task_obj.user_scenario),
            tools=user_tools if user_tools else None,
            llm_args={
                "api_base": f"http://127.0.0.1:{proxy_port}/usersim/v1",
                "api_key": "harnesslens-local-proxy",
                "temperature": 0.3,
                "extra_body": {"reasoning_effort": "high"},
            },
        )
        user_sim.set_seed(simulation_seed)
        user_state = user_sim.get_init_state()
        agent_msg = AssistantMessage.text("Hi! How can I help you today?")
        messages.append(agent_msg)
        for turn in range(int(limits.max_conversation_turns)):
            user_msg, user_state = user_sim.generate_next_message(
                agent_msg, state=user_state
            )
            if not (user_msg.content or "").strip() and not user_msg.is_tool_call():
                fixed = UserMessage(role="user", content="Please go ahead and help me.")
                if user_state.messages and user_state.messages[-1] is user_msg:
                    user_state.messages[-1] = fixed
                user_msg = fixed
            messages.append(user_msg)
            user_text = user_msg.content or ""
            if _user_stopped(user_text):
                break
            while hasattr(user_msg, "is_tool_call") and user_msg.is_tool_call():
                tool_messages = _execute_user_tool_calls(
                    socket_path, user_msg.tool_calls or []
                )
                messages.extend(tool_messages)
                feedback = (
                    tool_messages[0]
                    if len(tool_messages) == 1
                    else MultiToolMessage(role="tool", tool_messages=tool_messages)
                )
                user_msg, user_state = user_sim.generate_next_message(
                    feedback, state=user_state
                )
                if not (user_msg.content or "").strip() and not user_msg.is_tool_call():
                    fixed = UserMessage(
                        role="user", content="I've completed the checks."
                    )
                    if user_state.messages and user_state.messages[-1] is user_msg:
                        user_state.messages[-1] = fixed
                    user_msg = fixed
                messages.append(user_msg)
                user_text = user_msg.content or ""
                if _user_stopped(user_text):
                    break
            if _user_stopped(user_text) or total_tool_calls >= int(
                limits.max_tool_calls
            ):
                break
            _reset_tool_step_window(socket_path)
            turn_retry_events: list[dict[str, Any]] = []
            max_attempts = 1 + _opencode_turn_retry_attempts()
            for attempt in range(max_attempts):
                turn_result = _run_opencode_turn(
                    runtime_cwd=runtime_cwd,
                    runtime_home=runtime_home,
                    config_path=config_path,
                    user_text=user_text,
                    session_id=session_id,
                    timeout_s=int(limits.timeout_per_turn_s),
                )
                stdout_events.append(_json_lines(turn_result[0]))
                calls = _read_calls(calls_log)
                new_calls = [
                    call
                    for call in calls[prev_call_count:]
                    if call.get("requestor", "assistant") == "assistant"
                ]
                retryable = _is_retryable_empty_turn(
                    stderr=turn_result[1],
                    returncode=turn_result[2],
                    agent_text=turn_result[4],
                    new_calls=new_calls,
                )
                if not retryable or attempt + 1 >= max_attempts:
                    break
                delay_s = float(2**attempt)
                turn_retry_events.append(
                    {
                        "turn": turn,
                        "attempt": attempt + 1,
                        "delay_s": delay_s,
                        "error": str(turn_result[1]).strip()[-500:],
                    }
                )
                time.sleep(delay_s)
            session_id = turn_result[3] or session_id
            prev_call_count = len(calls)
            if turn_result[1].strip() or turn_result[2] != 0:
                error = (
                    turn_result[1].strip()[-1000:] or f"opencode_exit_{turn_result[2]}"
                )
                turn_errors.append(
                    {"turn": turn, "error": error, "retry_attempts": turn_retry_events}
                )
                last_error = error
            agent_text = turn_result[4]
            if new_calls:
                tool_calls = [
                    ToolCall(
                        id=f"call_{total_tool_calls + index}",
                        name=str(call.get("name") or ""),
                        arguments=dict(call.get("arguments") or {}),
                        requestor="assistant",
                    )
                    for index, call in enumerate(new_calls)
                ]
                messages.append(
                    AssistantMessage(
                        role="assistant",
                        content=agent_text or None,
                        tool_calls=tool_calls,
                    )
                )
                for index, call in enumerate(new_calls):
                    messages.append(
                        ToolMessage(
                            role="tool",
                            id=f"call_{total_tool_calls + index}",
                            content=str(call.get("result") or ""),
                            requestor="assistant",
                        )
                    )
                turn_calls = [str(call.get("call_str") or "") for call in new_calls]
                all_call_strs.extend(turn_calls)
                calls_per_turn.append(turn_calls)
                total_tool_calls += len(new_calls)
            else:
                calls_per_turn.append([])
            if agent_text:
                agent_msg = AssistantMessage.text(agent_text)
                messages.append(agent_msg)
            else:
                agent_msg = AssistantMessage.text("")
                break

        elapsed = time.time() - started
        has_stop = any(
            isinstance(message, UserMessage) and _user_stopped(message.content or "")
            for message in messages
        )
        termination = (
            TerminationReason.USER_STOP if has_stop else TerminationReason.MAX_STEPS
        )
        now = datetime.now(timezone.utc).isoformat()
        simulation = SimulationRun(
            id=f"opencode-{tag}",
            task_id=str(task_id),
            timestamp=now,
            start_time=now,
            end_time=now,
            duration=elapsed,
            termination_reason=termination,
            messages=messages,
        )
        try:
            reward_info = evaluate_simulation(
                simulation=simulation,
                task=task_obj,
                evaluation_type=EvaluationType.ALL,
                solo_mode=False,
                domain=domain,
                mode=CommunicationMode.HALF_DUPLEX,
                env_kwargs=_tau2_env_kwargs(retrieval_config, task_obj),
            )
            reward = float(reward_info.reward if reward_info else 0.0)
            evaluation = _serialize_reward_info(
                reward_info, evaluator_model=llm_settings["model"]
            )
        except Exception as exc:  # noqa: BLE001
            reward = 0.0
            last_error = f"eval {type(exc).__name__}: {exc}"
            evaluation = {
                "schema": "tau2.reward_info.v1",
                "available": False,
                "source": "online_evaluator",
                "complete": False,
                "reason": "evaluation_error",
                "error": str(exc)[:300],
                "model": llm_settings["model"],
            }
        trace_error = validate_api_trace(api_trace)
        if trace_error:
            last_error = trace_error
        refresh_runtime_load_report(
            model_context,
            project_root=runtime_cwd,
            home_root=runtime_home / ".config" / "opencode",
        )
        return {
            "task_id": str(task_id),
            "domain": domain,
            "trial": int(trial),
            "pairing_slot": int(trial),
            "simulation_seed": simulation_seed,
            "harness": "opencode",
            "model": MODEL,
            "reward": reward,
            "evaluation": evaluation,
            "runtime_models": {
                "target_agent": MODEL,
                "user_simulator": llm_settings["model"],
                "evaluator": llm_settings["model"],
            },
            "n_messages": len(messages),
            "n_tool_calls": total_tool_calls,
            "tool_calls": all_call_strs,
            "calls_per_turn": calls_per_turn,
            "messages": [_serialize_message(message) for message in messages],
            "system_prompt": full_prompt,
            "candidate_system_prompt_append": candidate_append,
            "tool_definitions": agent_tool_defs,
            "model_context": model_context,
            "retrieval_config": retrieval_config,
            "duration": elapsed,
            "termination": str(termination),
            "error": last_error,
            "api_calls_jsonl": str(api_trace.resolve()),
            "raw": {
                "runtime_cwd": str(runtime_cwd),
                "calls": _read_calls(calls_log),
                "session_id": session_id,
                "stdout_events": stdout_events,
                "turn_errors": turn_errors,
            },
        }
    finally:
        _stop_process(proxy_proc)
        _stop_process(server_proc)
        Path(socket_path).unlink(missing_ok=True)
        if not _keep_trial_workspace():
            _cleanup_trial_workspace(work_dir)
            _cleanup_trial_workspace(runtime_root)


def _start_opencode_proxy(
    *, api_trace: Path, agent_seed: int
) -> tuple[subprocess.Popen[str], int]:
    script = (
        Path(__file__).resolve().parents[1]
        / "infrastructure"
        / "chat_completions_proxy.py"
    )
    key = str(os.environ.get("DEEPSEEK_API_KEY") or "")
    if not script.is_file() or not key:
        raise RuntimeError("OpenCode Tau2 proxy prerequisites are unavailable")
    process = subprocess.Popen(
        [
            sys.executable,
            str(script),
            "--log-file",
            str(api_trace),
            "--agent-seed",
            str(int(agent_seed)),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(os.environ),
        start_new_session=True,
    )
    ready = process.stdout.readline().strip() if process.stdout else ""
    if not ready.startswith("PORT="):
        error = process.stderr.read()[:500] if process.stderr else ""
        _stop_process(process)
        raise RuntimeError(f"OpenCode Tau2 proxy failed: {ready} {error}")
    return process, int(ready.split("=", 1)[1])


def _run_opencode_turn(
    *,
    runtime_cwd: Path,
    runtime_home: Path,
    config_path: Path,
    user_text: str,
    session_id: str | None,
    timeout_s: int,
) -> tuple[str, str, int | None, str | None, str]:
    binary = _opencode_binary()
    command = [
        str(binary),
        "run",
        "-m",
        OPENCODE_MODEL,
        "--pure",
        "--auto",
        "--dir",
        str(runtime_cwd),
        "--format",
        "json",
    ]
    if session_id:
        command.extend(["--session", session_id])
    command.append(str(user_text))
    env = _opencode_env(runtime_home=runtime_home, config_path=config_path)
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=str(runtime_cwd),
            start_new_session=True,
        )
        stdout, stderr = process.communicate(timeout=int(timeout_s))
        returncode: int | None = process.returncode
    except subprocess.TimeoutExpired as exc:
        _stop_process(process)
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
        stderr += f"\nTIMEOUT after {timeout_s}s"
        returncode = None
    return (
        stdout or "",
        stderr or "",
        returncode,
        _parse_opencode_session(stdout or ""),
        _parse_opencode_text(stdout or ""),
    )


def _opencode_env(*, runtime_home: Path, config_path: Path) -> dict[str, str]:
    locations = {
        "XDG_CONFIG_HOME": runtime_home / ".config",
        "XDG_DATA_HOME": runtime_home / ".local" / "share",
        "XDG_CACHE_HOME": runtime_home / ".cache",
        "XDG_STATE_HOME": runtime_home / ".local" / "state",
        "TMPDIR": runtime_home / "tmp",
    }
    runtime_home.mkdir(parents=True, exist_ok=True)
    for path in locations.values():
        path.mkdir(parents=True, exist_ok=True)
    for name in (
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "XDG_STATE_HOME",
    ):
        (locations[name] / "opencode").mkdir(parents=True, exist_ok=True)
    return {
        **os.environ,
        "DEEPSEEK_API_KEY": "harnesslens-local-proxy",
        "HOME": str(runtime_home),
        **{name: str(path) for name, path in locations.items()},
        "OPENCODE_CONFIG": str(config_path),
        "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
        "OPENCODE_DISABLE_AUTOCOMPACT": "1",
        "OPENCODE_DISABLE_PRUNE": "1",
    }


def _opencode_binary() -> Path:
    configured = str(os.environ.get("HAI_OPENCODE_BIN") or "").strip()
    candidates = [
        Path(configured).expanduser() if configured else None,
        _default_opencode_prefix() / "bin" / "opencode",
    ]
    located = shutil.which("opencode")
    if located:
        candidates.insert(0, Path(located))
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    raise RuntimeError("opencode executable is unavailable")


def _opencode_runtime_root(
    request_root: Path, domain: str, task_id: str, trial: int
) -> Path:
    identity = f"{request_root.resolve()}\0{domain}\0{task_id}\0{int(trial)}"
    digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
    base = Path(
        os.environ.get("HAI_OPENCODE_RUNTIME_ROOT") or "/tmp/harnesslens_opencode_runtime"
    )
    return base / digest


def _keep_trial_workspace() -> bool:
    return str(os.environ.get("HAI_KEEP_TRAJECTORY_WORKSPACE") or "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _parse_opencode_text(stdout: str) -> str:
    texts: list[str] = []
    for event in _json_lines(stdout):
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


def _parse_opencode_session(stdout: str) -> str | None:
    for event in _json_lines(stdout):
        value = event.get("sessionID") or event.get("session_id")
        if value:
            return str(value)
    return None


def _json_lines(stdout: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in stdout.splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(event, Mapping):
            rows.append(dict(event))
    return rows


def _summarize_opencode_usage(request_root: Path) -> dict[str, Any]:
    calls = 0
    input_tokens = 0
    output_tokens = 0
    by_role: dict[str, int] = {}
    for path in request_root.glob("api_calls_*.jsonl"):
        for row in _json_lines(path.read_text(encoding="utf-8", errors="replace")):
            calls += 1
            role = str(row.get("role") or "unknown")
            by_role[role] = by_role.get(role, 0) + 1
            usage = row.get("usage") if isinstance(row.get("usage"), Mapping) else {}
            input_tokens += int(
                usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            )
            output_tokens += int(
                usage.get("completion_tokens") or usage.get("output_tokens") or 0
            )
    return {
        "call_count": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "by_role": by_role,
    }


def _user_stopped(text: str) -> bool:
    return "###STOP###" in text or "###OUT-OF-SCOPE###" in text


def _write_opencode_project(
    *,
    repo_root: Path,
    runtime_cwd: Path,
    runtime_home: Path,
    socket_path: str,
    proxy_port: int,
    system_prompt: str,
    max_steps: int,
    harness_manifest: Mapping[str, Any] | None,
) -> Path:
    manifest = _opencode_manifest(harness_manifest)
    runtime_cwd.mkdir(parents=True, exist_ok=True)
    runtime_home.mkdir(parents=True, exist_ok=True)
    materialize_project_files(
        runtime_cwd,
        manifest,
        home_root=runtime_home / ".config" / "opencode",
    )

    candidate_config: dict[str, Any] = {}
    for path in _candidate_config_paths(runtime_home, runtime_cwd):
        if path.is_file():
            candidate_config = _deep_merge(candidate_config, _read_json_object(path))
    candidate_config = _deep_merge(
        candidate_config,
        {"agent": compile_opencode_agent_definitions(runtime_cwd)},
    )
    candidate_config = _deep_merge(candidate_config, manifest["config_patch"])
    candidate_config = relocate_opencode_instruction_paths(
        candidate_config,
        project_root=runtime_cwd,
    )

    candidate_prompt = str(
        (
            (candidate_config.get("agent") or {}).get("build") or {}
            if isinstance(candidate_config.get("agent"), Mapping)
            else {}
        ).get("prompt")
        or ""
    )
    prompt = candidate_system_prompt(system_prompt, manifest)
    prompt = "\n\n".join(
        item.strip()
        for item in (prompt, *(str(value) for value in manifest["instructions"]))
        if item.strip()
    )
    prompt = "\n\n".join(
        item.strip() for item in (prompt, candidate_prompt) if item.strip()
    )
    bridge = [
        str(repo_root / "third_party" / "tau3-bench" / ".venv" / "bin" / "python3"),
        str(
            repo_root / "harnesslens" / "tau2_mcp_server.py"
        ),
        "bridge",
        "--socket",
        str(socket_path),
    ]
    fixed = {
        "$schema": "https://opencode.ai/config.json",
        "autoupdate": False,
        "share": "disabled",
        "snapshot": False,
        "model": OPENCODE_MODEL,
        "small_model": OPENCODE_MODEL,
        "enabled_providers": ["deepseek"],
        "provider": {
            "deepseek": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "DeepSeek",
                "options": {
                    "baseURL": f"http://127.0.0.1:{int(proxy_port)}/v1",
                    "apiKey": "{env:DEEPSEEK_API_KEY}",
                    "timeout": 600000,
                    "chunkTimeout": 60000,
                },
                "models": {
                    MODEL: {
                        "name": "DeepSeek V4 Flash",
                        "limit": {
                            "context": DEFAULT_OPENCODE_CONTEXT_LIMIT,
                            "output": DEFAULT_OUTPUT_LIMIT,
                        },
                    }
                },
            }
        },
        "agent": {"build": {"steps": int(max_steps), "prompt": prompt}},
        "tools": {
            "bash": False,
            "read": False,
            "write": False,
            "edit": False,
            "glob": False,
            "grep": False,
            "list": False,
            "apply_patch": False,
            "webfetch": False,
            "websearch": False,
            "skill": True,
            "todowrite": False,
        },
        "permission": {
            "*": "allow",
            "bash": "deny",
            "external_directory": "deny",
            "webfetch": "deny",
            "websearch": "deny",
        },
        "mcp": {
            "tau2": {
                "type": "local",
                "enabled": True,
                "command": bridge,
            }
        },
    }
    config = _deep_merge(candidate_config, fixed)
    control_root = runtime_home / ".hai"
    control_root.mkdir(parents=True, exist_ok=True)
    config_path = control_root / "opencode.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return config_path


def _opencode_manifest(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    source = dict(raw or {})
    workspace = normalize_workspace_snapshot(source.pop(CANDIDATE_WORKSPACE_KEY, None))
    normalized = normalize_opencode_manifest(source)
    return {
        "config_patch": dict(normalized["config_patch"]),
        "files": list(normalized["files"]),
        "instructions": list(normalized["instructions"]),
        "prompt_appends": list(normalized["prompt_appends"]),
        "tool_desc_patches": dict(normalized["tool_desc_patches"]),
        CANDIDATE_WORKSPACE_KEY: workspace,
    }


def _candidate_config_paths(runtime_home: Path, runtime_cwd: Path) -> tuple[Path, ...]:
    return (
        runtime_home / ".config" / "opencode" / "config.json",
        runtime_home / ".config" / "opencode" / "opencode.json",
        runtime_home / "opencode.json",
        runtime_cwd / "opencode.json",
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid candidate OpenCode config {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"candidate OpenCode config must be an object: {path}")
    return dict(payload)


def _deep_merge(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(left)
    for key, value in right.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[str(key)] = _deep_merge(result[key], value)
        else:
            result[str(key)] = value
    return result
