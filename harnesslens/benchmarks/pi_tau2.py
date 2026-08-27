from __future__ import annotations

import json
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from harnesslens.core.config import pi_binary, provider_base_url
from harnesslens.core.artifacts import write_json
from harnesslens.harnesses.channel_preflight import (
    build_runtime_load_report,
    refresh_runtime_load_report,
)
from harnesslens.harnesses.candidate_config_runtime import (
    load_json_configs,
    merge_candidate_config,
)
from harnesslens.benchmarks.benchmark_splits import BenchmarkSplit
from harnesslens.benchmarks.tau2_driver import (
    DEFAULT_SYSTEM_PROMPT,
    Tau2Limits,
    _agent_tool_definitions,
    _cleanup_trial_workspace,
    _configure_tau2_deepseek_env,
    _configure_tau2_llm_runtime,
    _execute_user_tool_calls,
    _fs_tag,
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
    candidate_system_prompt as _candidate_system_prompt,
    materialize_project_files,
    native_manifest,
)
from harnesslens.infrastructure.provider_capacity import provider_trial_slot


MODEL = "deepseek-v4-flash"


def run_pi_tau2_test_baseline(
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
        return _run_native_tau2_batch_locked(
            root=root,
            request_root=request_root,
            split=split,
            request=request,
            retrieval_config=retrieval_config,
            limits=limits,
            harness_manifest=harness_manifest,
        )


def _run_native_tau2_batch_locked(
    *,
    root: Path,
    request_root: Path,
    split: BenchmarkSplit,
    request: RolloutRequest,
    retrieval_config: str | None,
    limits: Tau2Limits,
    harness_manifest: Mapping[str, Any] | None,
    trial_runner: Callable[..., dict[str, Any]] | None = None,
    harness_name: str = "pi",
    usage_summarizer: Callable[[Path], Mapping[str, Any]] | None = None,
    trajectory_retention: str | None = None,
    api_trace_required: bool = False,
) -> RolloutResponse:
    runner = trial_runner or run_single_pi_tau2_trial
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
                    _run_with_provider_trial_slot,
                    runner,
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
                        "error": (
                            f"{type(exc).__name__}: {exc}\n"
                            f"{traceback.format_exc(limit=12)}"
                        ),
                    }
                _write_trial_row(trajectory_root, str(task_id), int(trial), result)
                records_by_task[str(task_id)].append(result)
                done = sum(len(items) for items in records_by_task.values())
                if done % 10 == 0 or done == len(requested_jobs):
                    print(
                        f"[{harness_name}/{split.cell}/{request.run_id}] "
                        f"done {done}/{len(requested_jobs)}",
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
        "trajectory_retention": trajectory_retention
        or f"harnesslens_{harness_name}_trial_jsonl_retained_workspaces_cleaned",
        "api_trace_required": bool(api_trace_required),
        "elapsed_s": round(time.time() - started, 1),
        "usage": dict((usage_summarizer or _summarize_pi_usage)(request_root)),
    }
    summary_path = request_root / "summary.json"
    payload = {
        "request": request.to_dict(),
        "budget_spent": charged,
        "budget_remaining": max(0, requested - charged),
        "per_task": per_task,
        "records": [record.to_dict() for record in records],
        "metrics": metrics,
    }
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


def _run_with_provider_trial_slot(
    runner: Callable[..., dict[str, Any]], **kwargs: Any
) -> dict[str, Any]:
    with provider_trial_slot() as capacity:
        result = dict(runner(**kwargs))
    result["provider_capacity"] = capacity
    return result


def run_single_pi_tau2_trial(
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
    runtime_cwd = _pi_runtime_cwd(work_dir)
    pi_home = runtime_cwd / "home"
    work_dir.mkdir(parents=True, exist_ok=True)
    runtime_cwd.mkdir(parents=True, exist_ok=True)
    simulation_seed = _stable_simulation_seed(domain, str(task_id), int(trial))
    task_obj = _load_tau2_task(domain, task_id, task_split="test")
    env_for_policy = _tau2_env_for_task(domain, retrieval_config, task_obj)
    domain_policy = str(getattr(env_for_policy, "policy", "") or "")
    manifest = native_manifest("pi", harness_manifest)
    full_prompt = DEFAULT_SYSTEM_PROMPT + (
        "\n\n" + domain_policy if domain_policy else ""
    )
    candidate_system_prompt_append = _candidate_system_prompt("", manifest)
    system_prompt_append = _candidate_system_prompt(full_prompt, manifest)
    agent_tool_defs = _agent_tool_definitions(env_for_policy)
    socket_path = f"/tmp/harnesslens_tau2_pi_{work_dir.name[-20:]}.sock"
    calls_log = work_dir / "calls.json"
    api_trace = request_root / f"api_calls_{_fs_tag(domain, task_id, trial)}.jsonl"
    calls_log.unlink(missing_ok=True)
    api_trace.unlink(missing_ok=True)
    server_proc = _start_tau2_server(
        repo_root=repo_root,
        domain=domain,
        task_id=task_id,
        socket_path=socket_path,
        log_file=calls_log,
        max_steps=int(limits.max_tool_calls_per_turn),
        retrieval_config=retrieval_config if domain == "banking_knowledge" else None,
    )
    messages: list[Any] = []
    total_tool_calls = 0
    prev_call_count = 0
    all_call_strs: list[str] = []
    calls_per_turn: list[list[str]] = []
    stdout_paths: list[str] = []
    stderr_paths: list[str] = []
    turn_errors: list[dict[str, Any]] = []
    first_turn = True
    started = time.time()
    last_error = ""
    pi_session: _PiRpcSession | None = None
    proxy_proc: subprocess.Popen[str] | None = None
    try:
        proxy_proc, proxy_port = _start_pi_logging_proxy(
            api_trace=api_trace,
            agent_seed=simulation_seed,
        )
        _write_pi_project(
            runtime_cwd=runtime_cwd,
            pi_home=pi_home,
            repo_root=repo_root,
            socket_path=socket_path,
            harness_manifest=manifest,
            base_url=f"http://127.0.0.1:{proxy_port}/v1",
        )
        pi_session = _PiRpcSession.start(
            repo_root=repo_root,
            runtime_cwd=runtime_cwd,
            pi_home=pi_home,
            socket_path=socket_path,
            stdout_path=work_dir / "pi_rpc.stdout.jsonl",
            stderr_path=work_dir / "pi_rpc.stderr.txt",
            system_prompt_append=system_prompt_append,
        )
        model_context = build_runtime_load_report(
            harness="pi",
            project_root=runtime_cwd,
            home_root=pi_home,
            manifest=manifest,
            tool_definitions=agent_tool_defs,
            skills_available=pi_session.skills_available(timeout_s=10),
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
            turn_result = pi_session.prompt(
                user_text, timeout_s=int(limits.timeout_per_turn_s)
            )
            first_turn = False
            stdout_path = work_dir / f"stdout_t{turn}.jsonl"
            stderr_path = work_dir / f"stderr_t{turn}.txt"
            stdout_path.write_text(
                "\n".join(
                    json.dumps(item, ensure_ascii=False) for item in turn_result.events
                )
                + "\n",
                encoding="utf-8",
            )
            stderr_path.write_text(
                turn_result.stderr, encoding="utf-8", errors="replace"
            )
            stdout_paths.append(str(stdout_path))
            stderr_paths.append(str(stderr_path))
            calls = _read_calls(calls_log)
            new_calls = [
                call
                for call in calls[prev_call_count:]
                if call.get("requestor", "assistant") == "assistant"
            ]
            prev_call_count = len(calls)
            if turn_result.error:
                turn_errors.append({"turn": turn, "error": turn_result.error[-1000:]})
                last_error = turn_result.error
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
                        role="assistant",
                        content=turn_result.text or None,
                        tool_calls=tc_objs,
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
            if turn_result.text:
                agent_msg = AssistantMessage.text(turn_result.text)
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
            id=f"pi-{work_dir.name}",
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
        refresh_runtime_load_report(
            model_context,
            project_root=runtime_cwd,
            home_root=pi_home,
        )
        return {
            "task_id": str(task_id),
            "domain": domain,
            "trial": int(trial),
            "pairing_slot": int(trial),
            "simulation_seed": simulation_seed,
            "harness": "pi",
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
            "candidate_system_prompt_append": candidate_system_prompt_append,
            "tool_definitions": agent_tool_defs,
            "model_context": model_context,
            "retrieval_config": retrieval_config,
            "duration": elapsed,
            "termination": str(termination),
            "error": last_error,
            "api_calls_jsonl": str(api_trace.resolve()),
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
        if pi_session is not None:
            pi_session.close()
        _stop_process(proxy_proc)
        _stop_process(server_proc)
        Path(socket_path).unlink(missing_ok=True)
        _cleanup_trial_workspace(runtime_cwd)
        _cleanup_trial_workspace(work_dir)


class _PiTurnResult:
    def __init__(
        self, *, text: str, events: list[dict[str, Any]], stderr: str, error: str
    ) -> None:
        self.text = text
        self.events = events
        self.stderr = stderr
        self.error = error


class _PiRpcSession:
    def __init__(
        self,
        *,
        process: subprocess.Popen[str],
        output_queue: "queue.Queue[dict[str, Any]]",
        stderr_path: Path,
        stderr_buffer: list[str],
        lock: threading.Lock,
    ) -> None:
        self.process = process
        self.output_queue = output_queue
        self.stderr_path = stderr_path
        self.stderr_buffer = stderr_buffer
        self.lock = lock
        self.next_id = 1

    @classmethod
    def start(
        cls,
        *,
        repo_root: Path,
        runtime_cwd: Path,
        pi_home: Path,
        socket_path: str,
        stdout_path: Path,
        stderr_path: Path,
        system_prompt_append: str = "",
    ) -> "_PiRpcSession":
        command = _pi_rpc_command(
            repo_root,
            runtime_cwd,
            system_prompt_append=system_prompt_append,
        )
        env = {
            **os.environ,
            "DEEPSEEK_API_KEY": os.environ["DEEPSEEK_API_KEY"],
            "OPENAI_API_KEY": os.environ["DEEPSEEK_API_KEY"],
            "HAI_TAU2_SOCKET": socket_path,
            "PI_CODING_AGENT_DIR": str(pi_home),
            "HOME": str(pi_home),
            "XDG_CONFIG_HOME": str(pi_home / ".config"),
            "XDG_DATA_HOME": str(pi_home / ".local" / "share"),
            "XDG_CACHE_HOME": str(pi_home / ".cache"),
        }
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=str(runtime_cwd),
            start_new_session=True,
        )
        out_queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
        err_buffer: list[str] = []
        lock = threading.Lock()
        _start_reader_thread(process.stdout, stdout_path, out_queue)
        _start_stderr_thread(process.stderr, stderr_path, err_buffer, lock)
        session = cls(
            process=process,
            output_queue=out_queue,
            stderr_path=stderr_path,
            stderr_buffer=err_buffer,
            lock=lock,
        )
        session._wait_started(timeout_s=30)
        return session

    def _wait_started(self, *, timeout_s: int) -> None:
        response = self._send_and_wait({"type": "get_state"}, timeout_s=timeout_s)
        if not response.get("success"):
            raise RuntimeError(
                f"pi rpc get_state failed: {response} stderr={self._stderr_tail()}"
            )

    def prompt(self, user_text: str, *, timeout_s: int) -> _PiTurnResult:
        response = self._send_and_wait(
            {"type": "prompt", "message": user_text},
            timeout_s=min(30, timeout_s),
        )
        if not response.get("success"):
            return _PiTurnResult(
                text="",
                events=[response],
                stderr=self._stderr_tail(),
                error=str(response.get("error") or response),
            )
        events: list[dict[str, Any]] = []
        error = ""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.process.poll() is not None:
                error = f"pi_rpc_exit_{self.process.returncode}"
                break
            try:
                event = self.output_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            events.append(event)
            deadline = time.time() + timeout_s
            if event.get("type") == "agent_settled":
                break
        else:
            error = "pi_turn_timeout"
        text = self._last_assistant_text(timeout_s=10) if not error else ""
        return _PiTurnResult(
            text=text, events=events, stderr=self._stderr_tail(), error=error
        )

    def skills_available(self, *, timeout_s: int) -> list[dict[str, Any]]:
        response = self._send_and_wait({"type": "get_commands"}, timeout_s=timeout_s)
        if not response.get("success"):
            return []
        data = response.get("data")
        commands = data.get("commands") if isinstance(data, Mapping) else data
        result = []
        for item in commands or []:
            if (
                not isinstance(item, Mapping)
                or str(item.get("source") or "") != "skill"
            ):
                continue
            raw_name = str(item.get("name") or "")
            name = raw_name.removeprefix("skill:")
            if name:
                result.append({"name": name, "n_calls": 0, "source": "pi_rpc"})
        return result

    def _last_assistant_text(self, *, timeout_s: int) -> str:
        response = self._send_and_wait(
            {"type": "get_last_assistant_text"},
            timeout_s=timeout_s,
        )
        if not response.get("success"):
            return ""
        data = response.get("data")
        if isinstance(data, Mapping):
            return str(data.get("text") or "").strip()
        return str(data or "").strip()

    def _send_and_wait(
        self, payload: dict[str, Any], *, timeout_s: int
    ) -> dict[str, Any]:
        with self.lock:
            req_id = self.next_id
            self.next_id += 1
            payload = {**payload, "id": str(req_id)}
            assert self.process.stdin is not None
            self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.process.poll() is not None:
                return {
                    "type": "response",
                    "id": str(req_id),
                    "success": False,
                    "error": f"pi_rpc_exit_{self.process.returncode}: {self._stderr_tail()}",
                }
            try:
                event = self.output_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if str(event.get("id") or "") != str(req_id):
                continue
            return event
        return {
            "type": "response",
            "id": str(req_id),
            "success": False,
            "error": f"timeout waiting for pi response to {payload.get('type')}",
        }

    def _stderr_tail(self) -> str:
        try:
            text = self.stderr_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        return text[-2000:]

    def close(self) -> None:
        _stop_process(self.process)


def _pi_rpc_command(
    repo_root: Path,
    runtime_cwd: Path,
    *,
    system_prompt_append: str = "",
) -> list[str]:
    command = [
        str(_pi_binary(repo_root)),
        "--mode",
        "rpc",
        "--provider",
        "deepseek",
        "--model",
        MODEL,
        "--approve",
        "--no-builtin-tools",
        "--no-session",
        "--offline",
        "-e",
        str(runtime_cwd / ".pi" / "tau2_extension.ts"),
    ]
    if str(system_prompt_append).strip():
        command.extend(["--append-system-prompt", str(system_prompt_append)])
    return command


def _start_reader_thread(
    stream: Any,
    path: Path,
    output_queue: "queue.Queue[dict[str, Any]]",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def run() -> None:
        with path.open("a", encoding="utf-8") as handle:
            for line in iter(stream.readline, ""):
                handle.write(line)
                handle.flush()
                try:
                    output_queue.put(json.loads(line))
                except json.JSONDecodeError:
                    output_queue.put({"raw": line.rstrip("\n")})

    threading.Thread(target=run, daemon=True).start()


def _start_stderr_thread(
    stream: Any,
    path: Path,
    buffer: list[str],
    lock: threading.Lock,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def run() -> None:
        with path.open("a", encoding="utf-8") as handle:
            for line in iter(stream.readline, ""):
                handle.write(line)
                handle.flush()
                with lock:
                    buffer.append(line)
                    if len(buffer) > 200:
                        del buffer[:-200]

    threading.Thread(target=run, daemon=True).start()


def _write_pi_project(
    *,
    runtime_cwd: Path,
    pi_home: Path,
    repo_root: Path,
    socket_path: str,
    harness_manifest: Mapping[str, Any] | None = None,
    base_url: str | None = None,
) -> None:
    del repo_root, socket_path
    manifest = native_manifest("pi", harness_manifest)
    pi_dir = runtime_cwd / ".pi"
    pi_dir.mkdir(parents=True, exist_ok=True)
    pi_home.mkdir(parents=True, exist_ok=True)
    materialize_project_files(runtime_cwd, manifest, home_root=pi_home)
    base_url = str(base_url or _deepseek_base_url()).rstrip("/")
    candidate_config = load_json_configs(
        (
            pi_home / "settings.json",
            pi_home / ".pi" / "settings.json",
            runtime_cwd / ".pi" / "settings.json",
        )
    )
    settings = merge_candidate_config(
        candidate_config,
        legacy_flat_patch=manifest["config_patch"],
        fixed={
            "permissions": {"allow": [], "deny": []},
            "provider": {
                "default": "deepseek",
                "providers": {
                    "deepseek": {
                        "baseURL": base_url,
                        "apiKeyEnvVar": "DEEPSEEK_API_KEY",
                        "models": [MODEL],
                    }
                },
            },
            "model": MODEL,
        },
    )
    (pi_dir / "settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (pi_home / "models.json").write_text(
        json.dumps(
            {
                "providers": {
                    "deepseek": {
                        "baseUrl": base_url,
                        "api": "openai-completions",
                        "apiKey": "harnesslens-runtime-key",
                        "compat": {
                            "supportsDeveloperRole": False,
                            "supportsReasoningEffort": False,
                        },
                        "models": [
                            {
                                "id": MODEL,
                                "reasoning": True,
                                "contextWindow": 65_536,
                                "maxTokens": 24_576,
                            }
                        ],
                    }
                }
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (pi_dir / "tau2_extension.ts").write_text(
        _pi_tau2_extension_source(manifest["tool_desc_patches"]),
        encoding="utf-8",
    )


def _pi_tau2_extension_source(
    tool_desc_patches: Mapping[str, Any] | None = None,
) -> str:
    patches = json.dumps(dict(tool_desc_patches or {}), ensure_ascii=False)
    return r"""
import { Type } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import net from "node:net";

type Json = Record<string, any>;

const harnessPatches: Json = __HARNESS_PATCHES__;

let rpcId = 1;

function rpc(method: string, params: Json = {}): Promise<Json> {
  const socketPath = process.env.HAI_TAU2_SOCKET;
  if (!socketPath) throw new Error("HAI_TAU2_SOCKET is not set");
  return new Promise((resolve, reject) => {
    const client = net.createConnection(socketPath);
    let data = Buffer.alloc(0);
    const id = rpcId++;
    client.on("connect", () => {
      const payload = Buffer.from(JSON.stringify({ jsonrpc: "2.0", id, method, params }), "utf8");
      const frame = Buffer.alloc(4 + payload.length);
      frame.writeUInt32BE(payload.length, 0);
      payload.copy(frame, 4);
      client.write(frame);
    });
    client.on("data", (chunk) => {
      data = Buffer.concat([data, chunk]);
      if (data.length < 4) return;
      const length = data.readUInt32BE(0);
      if (data.length < 4 + length) return;
      client.end();
      try {
        const parsed = JSON.parse(data.subarray(4, 4 + length).toString("utf8"));
        if (parsed.error) reject(new Error(parsed.error.message || JSON.stringify(parsed.error)));
        else resolve(parsed);
      } catch (error) {
        reject(error);
      }
    });
    client.on("error", reject);
  });
}

export default async function (pi: ExtensionAPI) {
  const listed = await rpc("tools/list", {});
  const tools = listed.result?.tools ?? [];
  for (const tool of tools) {
    const name = String(tool.name || "");
    if (!name) continue;
    const patch = harnessPatches[name] || {};
    const parameters = structuredClone(tool.inputSchema || { type: "object", properties: {} });
    for (const [parameter, description] of Object.entries(patch.params || {})) {
      if (parameters.properties?.[parameter]) {
        parameters.properties[parameter].description = String(description);
      }
    }
    pi.registerTool({
      name,
      label: name,
      description: String(patch.desc || tool.description || ""),
      parameters: Type.Unsafe(parameters),
      async execute(_toolCallId: string, args: Json) {
        const result = await rpc("tools/call", { name, arguments: args || {} });
        const content = result.result?.content ?? [];
        const text = content.map((part: Json) => String(part.text ?? "")).join("\n");
        return { content: [{ type: "text", text }], details: result.result ?? {} };
      },
    });
  }
}
""".replace(
        "__HARNESS_PATCHES__", patches
    )


def _pi_binary(repo_root: Path) -> Path:
    return pi_binary(repo_root)

def _start_pi_logging_proxy(
    *, api_trace: Path, agent_seed: int
) -> tuple[subprocess.Popen[str], int]:
    script = (
        Path(__file__).resolve().parents[1]
        / "infrastructure"
        / "chat_completions_proxy.py"
    )
    key = str(os.environ.get("DEEPSEEK_API_KEY") or "")
    if not script.is_file() or not key:
        raise RuntimeError("Pi Tau2 proxy prerequisites are unavailable")
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
    assert process.stdout is not None
    ready = process.stdout.readline().strip()
    if not ready.startswith("PORT="):
        error = process.stderr.read()[:500] if process.stderr else ""
        _stop_process(process)
        raise RuntimeError(f"Pi Tau2 logging proxy failed: {ready} {error}")
    return process, int(ready.split("=", 1)[1])


def _deepseek_base_url() -> str:
    base = provider_base_url()
    suffix = "/chat/completions"
    if base.endswith(suffix):
        base = base[: -len(suffix)]
    return base


def _pi_runtime_cwd(work_dir: Path) -> Path:
    root = Path(os.environ.get("HAI_PI_RUNTIME_CWD_ROOT") or "/tmp/harnesslens_pi_cwd")
    return root / work_dir.name


def _summarize_pi_usage(request_root: Path) -> dict[str, Any]:
    paths = sorted(request_root.glob("workspaces/**/pi_rpc.stderr.txt"))
    stderr_bytes = 0
    for path in paths:
        try:
            stderr_bytes += path.stat().st_size
        except OSError:
            continue
    return {"session_count": len(paths), "stderr_bytes": stderr_bytes}


def _debug_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)
