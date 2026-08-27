from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from harnesslens.core.artifacts import write_json
from harnesslens.harnesses.channel_preflight import (
    build_runtime_load_report,
    refresh_runtime_load_report,
)
from harnesslens.harnesses.candidate_config_runtime import (
    load_toml_configs,
    merge_candidate_config,
)
from harnesslens.benchmarks.benchmark_splits import BenchmarkSplit
from harnesslens.benchmarks.tau2_driver import (
    DEFAULT_SYSTEM_PROMPT,
    MODEL,
    Tau2Limits,
    _agent_tool_definitions,
    _cleanup_trial_workspace,
    _configure_tau2_deepseek_env,
    _configure_tau2_llm_runtime,
    _execute_user_tool_calls,
    _fs_tag,
    _is_clean_first_turn_timeout,
    _load_cached_response,
    _load_existing_trial_rows,
    _load_tau2_task,
    _parse_agent_text,
    _read_calls,
    _request_file_lock,
    _reset_tool_step_window,
    _serialize_message,
    _serialize_reward_info,
    _sort_key,
    _stable_simulation_seed,
    _start_tau2_server,
    _stop_process,
    _summarize,
    _summarize_cache,
    _tau2_env_for_task,
    _tau2_env_kwargs,
    _trial_path,
    _write_trial_row,
)
from harnesslens.evaluation.rollout_bridge import (
    RolloutRequest,
    RolloutResponse,
    TrainRolloutRecord,
)
from harnesslens.harnesses.native_candidate_runtime import (
    codex_hook_declared,
    codex_project_trust_config,
    prepare_codex_project_hooks,
    materialize_project_files,
    native_manifest,
    render_toml,
)


DEFAULT_CODEX_MODELS_CACHE = "~/.codex/models_cache.json"


def _codex_models_cache() -> Path:
    """Codex model-catalogue cache seeded into each sandboxed CODEX_HOME."""
    return Path(
        os.environ.get("HAI_CODEX_MODELS_CACHE") or DEFAULT_CODEX_MODELS_CACHE
    ).expanduser()


def run_codex_tau2_test_baseline(
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
    output_root = Path(run_root).resolve()
    request_root = (
        output_root / "rollout_artifacts" / request.run_id / request.request_id
    )
    request_root.mkdir(parents=True, exist_ok=True)
    with _request_file_lock(request_root):
        return _run_codex_tau2_test_baseline_locked(
            root=root,
            request_root=request_root,
            split=split,
            request=request,
            retrieval_config=retrieval_config,
            limits=limits,
            harness_manifest=harness_manifest,
        )


def _run_codex_tau2_test_baseline_locked(
    *,
    root: Path,
    request_root: Path,
    split: BenchmarkSplit,
    request: RolloutRequest,
    retrieval_config: str | None,
    limits: Tau2Limits,
    harness_manifest: Mapping[str, Any] | None,
) -> RolloutResponse:
    metadata_path = request_root / "metadata.json"
    cached = _load_cached_response(metadata_path)
    if (
        cached is not None
        and int(cached.metrics.get("infrastructure_failure_count", 0) or 0) == 0
    ):
        return cached
    trajectory_root = request_root / "trajectories"
    started = time.time()
    requested_jobs = [
        (str(task_id), trial)
        for task_id, repeats in request.task_repeats.items()
        for trial in range(int(repeats))
    ]
    records_by_task: dict[str, list[dict[str, Any]]] = {
        str(task_id): _load_existing_trial_rows(
            trajectory_root, str(task_id), int(repeats)
        )
        for task_id, repeats in request.task_repeats.items()
    }
    cached_trials = {
        (task_id, int(row.get("trial") or 0))
        for task_id, rows in records_by_task.items()
        for row in rows
    }
    jobs = [
        (task_id, trial)
        for task_id, trial in requested_jobs
        if (task_id, int(trial)) not in cached_trials
    ]
    if jobs:
        with ThreadPoolExecutor(max_workers=int(request.max_concurrency)) as pool:
            futures = {
                pool.submit(
                    run_single_codex_tau2_trial,
                    repo_root=root,
                    request_root=request_root,
                    domain=split.cell,
                    task_id=task_id,
                    trial=trial,
                    retrieval_config=retrieval_config,
                    limits=limits,
                    harness_manifest=harness_manifest,
                ): (task_id, trial)
                for task_id, trial in jobs
            }
            for future in as_completed(futures, timeout=int(limits.group_timeout_s)):
                task_id, trial = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "task_id": task_id,
                        "trial": trial,
                        "reward": 0.0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                _write_trial_row(trajectory_root, str(task_id), int(trial), result)
                records_by_task[str(task_id)].append(result)
                done = sum(len(items) for items in records_by_task.values())
                if done % 10 == 0 or done == len(requested_jobs):
                    print(
                        f"[codex/{split.cell}/{request.run_id}] done {done}/{len(requested_jobs)}",
                        flush=True,
                    )

    records: list[TrainRolloutRecord] = []
    per_task: dict[str, dict[str, Any]] = {}
    for task_id in sorted(records_by_task, key=_sort_key):
        rows = sorted(
            records_by_task[task_id], key=lambda item: int(item.get("trial") or 0)
        )
        paths: list[str] = []
        rewards: list[float] = []
        errors: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            trial = int(row.get("trial") or index)
            trial_path = _trial_path(trajectory_root, task_id, trial)
            _write_trial_row(trajectory_root, task_id, trial, row)
            paths.append(str(trial_path.resolve()))
            reward = float(row.get("reward", 0.0) or 0.0)
            rewards.append(reward)
            if row.get("error"):
                errors.append({"trial": trial, "error": str(row.get("error"))})
            summaries.append(
                {
                    "trial": trial,
                    "pairing_slot": trial,
                    "simulation_seed": row.get("simulation_seed"),
                    "reward": reward,
                    "n_messages": int(row.get("n_messages") or 0),
                    "n_tool_calls": int(row.get("n_tool_calls") or 0),
                    "termination": str(row.get("termination") or ""),
                    "error": str(row.get("error") or "")[:300],
                    "meta_path": "",
                }
            )
        record = TrainRolloutRecord(
            task_id=task_id,
            rewards=tuple(rewards),
            harness_version=request.harness_version,
            trajectory_paths=tuple(paths),
            worker_errors=tuple(errors),
            trial_summaries=tuple(summaries),
        )
        records.append(record)
        per_task[task_id] = {
            "repeats": len(rows),
            "rewards": rewards,
            "trajectory_paths": paths,
            "worker_errors": errors,
            "trial_summaries": summaries,
        }

    requested = len(requested_jobs)
    infrastructure = sum(1 for record in records for err in record.worker_errors if err)
    charged = max(0, requested - infrastructure)
    metrics = {
        **_summarize(records),
        "requested_trial_count": requested,
        "resumed_trial_count": requested - len(jobs),
        "missing_trial_count": len(jobs),
        "infrastructure_failure_count": infrastructure,
        "charged_trial_count": charged,
        "workspace_cleanup_enabled": True,
        "trajectory_retention": "harnesslens_codex_trial_jsonl_and_api_sidecars_retained_workspaces_cleaned",
        "api_trace_required": True,
        "elapsed_s": round(time.time() - started, 1),
        "cache_usage": _summarize_codex_cache(request_root),
    }
    payload = {
        "request": request.to_dict(),
        "budget_spent": charged,
        "budget_remaining": max(0, requested - charged),
        "per_task": per_task,
        "records": [record.to_dict() for record in records],
        "metrics": metrics,
    }
    summary_path = request_root / "summary.json"
    write_json(metadata_path, payload)
    write_json(
        summary_path,
        {
            "request_id": request.request_id,
            "harness_version": request.harness_version,
            "budget_spent": charged,
            "budget_remaining": max(0, requested - charged),
            "metrics": metrics,
            "per_task": per_task,
        },
    )
    return RolloutResponse(
        request_id=request.request_id,
        harness_version=request.harness_version,
        budget_spent=charged,
        budget_remaining=max(0, requested - charged),
        trajectory_root=str(trajectory_root),
        summary_json=str(summary_path),
        metadata_json=str(metadata_path),
        metrics=metrics,
        per_task=per_task,
        records=tuple(records),
        scope=request.scope,
    )


def run_single_codex_tau2_trial(
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
    work_dir.mkdir(parents=True, exist_ok=True)
    runtime_cwd = _codex_runtime_cwd(work_dir)
    simulation_seed = _stable_simulation_seed(domain, str(task_id), int(trial))
    task_obj = _load_tau2_task(domain, task_id, task_split="test")
    env_for_policy = _tau2_env_for_task(domain, retrieval_config, task_obj)
    domain_policy = str(getattr(env_for_policy, "policy", "") or "")
    manifest = native_manifest("codex", harness_manifest)
    full_prompt = DEFAULT_SYSTEM_PROMPT + (
        "\n\n" + domain_policy if domain_policy else ""
    )
    agent_tool_defs = _agent_tool_definitions(env_for_policy)
    socket_path = f"/tmp/harnesslens_tau2_codex_{work_dir.name[-20:]}.sock"
    calls_log = work_dir / "calls.json"
    calls_log.unlink(missing_ok=True)
    server_proc = _start_tau2_server(
        repo_root=repo_root,
        domain=domain,
        task_id=task_id,
        socket_path=socket_path,
        log_file=calls_log,
        max_steps=int(limits.max_tool_calls_per_turn),
        retrieval_config=retrieval_config if domain == "banking_knowledge" else None,
    )
    proxy_proc, proxy_port = _start_codex_proxy(
        repo_root,
        request_root,
        socket_path=socket_path,
        log_tag=work_dir.name,
        tool_desc_patches=manifest["tool_desc_patches"],
    )
    api_calls_path = request_root / f"api_calls_{work_dir.name}.jsonl"
    messages: list[Any] = []
    total_tool_calls = 0
    prev_call_count = 0
    all_call_strs: list[str] = []
    calls_per_turn: list[list[str]] = []
    stdout_paths: list[str] = []
    stderr_paths: list[str] = []
    turn_errors: list[dict[str, Any]] = []
    first_turn = True
    try:
        codex_home = _setup_codex_home(
            runtime_cwd,
            proxy_port,
            socket_path,
            repo_root,
            system_prompt=full_prompt,
            harness_manifest=manifest,
        )
        model_context = build_runtime_load_report(
            harness="codex",
            project_root=runtime_cwd,
            home_root=codex_home,
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
                "api_base": llm_settings["api_base"],
                "api_key": llm_settings["api_key"],
                "temperature": 0.3,
                "extra_body": {"reasoning_effort": "high"},
            },
        )
        user_sim.set_seed(simulation_seed)
        user_state = user_sim.get_init_state()
        agent_msg = AssistantMessage.text("Hi! How can I help you today?")
        messages.append(agent_msg)
        started = time.time()
        last_error = ""
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
            if "###STOP###" in user_text or "###OUT-OF-SCOPE###" in user_text:
                break
            while hasattr(user_msg, "is_tool_call") and user_msg.is_tool_call():
                user_tool_calls = user_msg.tool_calls or []
                tool_messages = _execute_user_tool_calls(socket_path, user_tool_calls)
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
                if "###STOP###" in user_text or "###OUT-OF-SCOPE###" in user_text:
                    break
            if "###STOP###" in user_text or "###OUT-OF-SCOPE###" in user_text:
                break
            if total_tool_calls >= int(limits.max_tool_calls):
                break
            _reset_tool_step_window(socket_path)
            stdout, stderr, agent_text = _run_codex_turn(
                runtime_cwd=runtime_cwd,
                proxy_port=proxy_port,
                user_text=user_text,
                first_turn=first_turn,
                timeout_s=int(limits.timeout_per_turn_s),
            )
            first_turn = False
            stdout_path = work_dir / f"stdout_t{turn}.txt"
            stderr_path = work_dir / f"stderr_t{turn}.txt"
            stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
            stderr_path.write_text(stderr, encoding="utf-8", errors="replace")
            stdout_paths.append(str(stdout_path))
            stderr_paths.append(str(stderr_path))
            calls = _read_calls(calls_log)
            new_calls = [
                call
                for call in calls[prev_call_count:]
                if call.get("requestor", "assistant") == "assistant"
            ]
            prev_call_count = len(calls)
            if stderr.strip():
                turn_errors.append(
                    {
                        "turn": turn,
                        "attempt": 0,
                        "retryable": False,
                        "stderr": stderr.strip()[-1000:],
                    }
                )
            if new_calls:
                tc_objs = [
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
                        role="assistant", content=agent_text or None, tool_calls=tc_objs
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
                turn_call_strs = [str(call.get("call_str") or "") for call in new_calls]
                all_call_strs.extend(turn_call_strs)
                calls_per_turn.append(turn_call_strs)
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
            isinstance(message, UserMessage)
            and (
                "###STOP###" in (message.content or "")
                or "###OUT-OF-SCOPE###" in (message.content or "")
            )
            for message in messages
        )
        termination = (
            TerminationReason.USER_STOP if has_stop else TerminationReason.MAX_STEPS
        )
        now = datetime.now(timezone.utc).isoformat()
        simulation = SimulationRun(
            id=f"codex-{work_dir.name}",
            task_id=str(task_id),
            timestamp=now,
            start_time=now,
            end_time=now,
            duration=elapsed,
            termination_reason=termination,
            messages=messages,
        )
        if _is_clean_first_turn_timeout(turn_errors, total_tool_calls, messages):
            reward = 0.0
            last_error = "first_turn_timeout"
            evaluation = {
                "schema": "tau2.reward_info.v1",
                "available": False,
                "source": "short_circuit",
                "complete": True,
                "reason": "first_turn_timeout_without_tool_calls",
                "model": llm_settings["model"],
            }
        else:
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
        refresh_runtime_load_report(
            model_context,
            project_root=runtime_cwd,
            home_root=codex_home,
        )
        return {
            "task_id": str(task_id),
            "domain": domain,
            "trial": int(trial),
            "pairing_slot": int(trial),
            "simulation_seed": simulation_seed,
            "harness": "codex",
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
            "tool_definitions": agent_tool_defs,
            "model_context": model_context,
            "retrieval_config": retrieval_config,
            "duration": elapsed,
            "termination": str(termination),
            "error": last_error,
            "api_calls_jsonl": str(api_calls_path),
            "raw": {
                "workdir": str(work_dir),
                "runtime_cwd": str(runtime_cwd),
                "calls": _read_calls(calls_log),
                "stdout_paths": stdout_paths,
                "stderr_paths": stderr_paths,
                "turn_errors": turn_errors,
            },
        }
    finally:
        _stop_process(proxy_proc)
        _stop_process(server_proc)
        Path(socket_path).unlink(missing_ok=True)
        _cleanup_trial_workspace(work_dir)
        _cleanup_trial_workspace(runtime_cwd)


def _start_codex_proxy(
    repo_root: Path,
    request_root: Path,
    *,
    socket_path: str | None = None,
    log_tag: str | None = None,
    tool_desc_patches: Mapping[str, Any] | None = None,
) -> tuple[subprocess.Popen[str], int]:
    cache_log = request_root / (
        f"cache_log_{log_tag}.jsonl" if log_tag else "cache_log.jsonl"
    )
    context_log = request_root / (
        f"api_calls_{log_tag}.jsonl" if log_tag else "api_calls.jsonl"
    )
    command = [
        sys.executable,
        str(
            repo_root
            / "harnesslens"
            / "codex_responses_proxy.py"
        ),
        "--log-file",
        str(cache_log),
        "--context-log",
        str(context_log),
    ]
    if socket_path:
        command.extend(["--tau2-socket", str(socket_path)])
    if tool_desc_patches:
        patch_path = request_root / (
            f"tool_patches_{log_tag}.json" if log_tag else "tool_patches.json"
        )
        patch_path.write_text(
            json.dumps(dict(tool_desc_patches), ensure_ascii=False),
            encoding="utf-8",
        )
        command.extend(["--tool-desc-patches", str(patch_path)])
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "DEEPSEEK_API_KEY": os.environ["DEEPSEEK_API_KEY"]},
        start_new_session=True,
    )
    line = process.stdout.readline().strip()
    if not line.startswith("PORT="):
        stderr = process.stderr.read()
        raise RuntimeError(f"Codex Responses proxy failed: {line} {stderr[:1000]}")
    return process, int(line.split("=", 1)[1])


def _summarize_codex_cache(request_root: Path) -> dict[str, Any]:
    cache_paths = sorted(request_root.glob("cache_log*.jsonl"))
    if not cache_paths:
        return _summarize_cache(request_root / "cache_log.jsonl")
    rows: list[dict[str, Any]] = []
    for path in cache_paths:
        try:
            rows.extend(
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except (OSError, json.JSONDecodeError):
            continue
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


def _setup_codex_home(
    runtime_cwd: Path,
    proxy_port: int,
    socket_path: str,
    repo_root: Path,
    *,
    system_prompt: str = "",
    harness_manifest: Mapping[str, Any] | None = None,
) -> Path:
    manifest = native_manifest("codex", harness_manifest)
    config_patch = dict(manifest["config_patch"])
    runtime_home = runtime_cwd / "home"
    codex_home = runtime_home / ".codex"
    mcp_debug_log = runtime_cwd / "mcp_debug.jsonl"
    codex_home.mkdir(parents=True, exist_ok=True)
    materialize_project_files(
        runtime_cwd,
        manifest,
        home_root=codex_home,
    )
    candidate_config = load_toml_configs(
        (
            codex_home / "config.toml",
            runtime_cwd / ".codex" / "config.toml",
        )
    )
    candidate_config = merge_candidate_config(
        candidate_config,
        legacy_flat_patch=config_patch,
        fixed={},
    )
    candidate_developer = str(
        candidate_config.pop("developer_instructions", "")
    ).strip()
    developer_instructions = "\n\n".join(
        item for item in (system_prompt.strip(), candidate_developer) if item
    )
    base_config = {
        "model": "gpt-5.4",
        "model_provider": "deepseek",
        "model_reasoning_effort": "high",
        "sandbox_mode": "read-only",
        "approval_policy": "never",
        "web_search": "disabled",
        "developer_instructions": developer_instructions,
        "model_providers": {
            "deepseek": {
                "name": "DeepSeek via local Responses proxy",
                "base_url": f"http://127.0.0.1:{proxy_port}/v1",
                "env_key": "OPENAI_API_KEY",
                "wire_api": "responses",
            }
        },
        "features": {
            "apps": False,
            "enable_fanout": True,
            "multi_agent": True,
            "multi_agent_v2": True,
            "plugins": False,
            "remote_plugin": False,
            "shell_tool": False,
            "unified_exec": False,
        },
        "mcp_servers": {
            "tau2": {
                "command": str(
                    repo_root
                    / "third_party"
                    / "tau3-bench"
                    / ".venv"
                    / "bin"
                    / "python3"
                ),
                "args": [
                    str(
                        repo_root
                        / "harnesslens"
                        / "tau2_mcp_server.py"
                    ),
                    "bridge",
                    "--socket",
                    str(socket_path),
                ],
                "env": {"HAI_TAU2_MCP_DEBUG_LOG": str(mcp_debug_log)},
            }
        },
        **(
            codex_project_trust_config(runtime_cwd)
            if codex_hook_declared(manifest)
            else {}
        ),
    }
    config = merge_candidate_config(
        candidate_config,
        fixed=base_config,
    )
    (codex_home / "config.toml").write_text(render_toml(config), encoding="utf-8")
    prepare_codex_project_hooks(runtime_cwd, manifest)
    _write_codex_model_cache(codex_home)
    return codex_home


def _write_codex_model_cache(codex_home: Path) -> None:
    source = _codex_models_cache()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        template = dict((payload.get("models") or [])[0])
    except Exception:  # noqa: BLE001
        template = {
            "slug": "gpt-5.4",
            "display_name": "gpt-5.4",
            "description": "DeepSeek v4 flash via local Responses proxy.",
            "default_reasoning_level": "high",
            "supported_reasoning_levels": [
                {"effort": "high", "description": "High reasoning"}
            ],
            "shell_type": "shell_command",
            "visibility": "list",
            "supported_in_api": True,
            "supports_reasoning_summaries": False,
            "default_reasoning_summary": "none",
            "support_verbosity": False,
            "apply_patch_tool_type": "freeform",
            "web_search_tool_type": "text_and_image",
            "supports_parallel_tool_calls": True,
            "context_window": 65536,
            "effective_context_window_percent": 95,
            "experimental_supported_tools": [],
            "input_modalities": ["text"],
            "supports_search_tool": False,
        }
    template["slug"] = "gpt-5.4"
    template["display_name"] = "gpt-5.4"
    template["description"] = "DeepSeek v4 flash via local Responses proxy."
    template["supported_in_api"] = True
    template["supports_parallel_tool_calls"] = True
    template["supports_search_tool"] = False
    cache = {
        "fetched_at": "2026-07-18T00:00:00Z",
        "etag": "harnesslens-codex-deepseek-local",
        "client_version": "harnesslens",
        "models": [template],
    }
    (codex_home / "models_cache.json").write_text(
        json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_codex_turn(
    *,
    runtime_cwd: Path,
    proxy_port: int,
    user_text: str,
    first_turn: bool,
    timeout_s: int,
) -> tuple[str, str, str]:
    runtime_home = runtime_cwd / "home"
    codex_home = runtime_home / ".codex"
    runtime_cwd.mkdir(parents=True, exist_ok=True)
    last_message = runtime_cwd / "_last_message.txt"
    last_message.unlink(missing_ok=True)
    command = ["codex", "exec", "--strict-config"]
    if (runtime_cwd / ".codex" / "hooks.json").is_file():
        command.append("--dangerously-bypass-hook-trust")
    if not first_turn:
        command.extend(["resume", "--last"])
    command.extend(
        [
            "--skip-git-repo-check",
            "--ignore-rules",
            "--json",
            "--output-last-message",
            str(last_message),
            user_text,
        ]
    )
    env = {
        **{
            key: value for key, value in os.environ.items() if key != "DEEPSEEK_API_KEY"
        },
        "OPENAI_API_KEY": "harnesslens-local-proxy",
        "CODEX_HOME": str(codex_home),
        "HOME": str(runtime_home),
        "XDG_CONFIG_HOME": str(runtime_home / ".config"),
        "XDG_DATA_HOME": str(runtime_home / ".local" / "share"),
        "XDG_CACHE_HOME": str(runtime_home / ".cache"),
    }
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
        return stdout, stderr + "\nTIMEOUT", ""
    except Exception:
        _stop_process(process)
        raise
    agent_text = (
        last_message.read_text(encoding="utf-8", errors="replace").strip()
        if last_message.exists()
        else ""
    )
    if not agent_text:
        agent_text = _parse_agent_text(stdout or "")
    return stdout or "", stderr or "", agent_text


def _codex_runtime_cwd(work_dir: Path) -> Path:
    root = Path(os.environ.get("HAI_CODEX_RUNTIME_CWD_ROOT") or "/tmp/harnesslens_codex_cwd")
    return root / work_dir.name
