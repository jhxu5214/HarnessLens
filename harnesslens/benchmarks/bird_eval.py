from __future__ import annotations

import json
import os
import queue
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from harnesslens.core.config import provider_base_url
from harnesslens.benchmarks.bird_sql import grade_execution_accuracy
from harnesslens.harnesses.channel_preflight import build_runtime_load_report
from harnesslens.harnesses.candidate_config_runtime import (
    compile_opencode_agent_definitions,
    load_json_configs,
    load_toml_configs,
    merge_candidate_config,
    relocate_opencode_instruction_paths,
)
from harnesslens.harnesses.native_candidate_runtime import (
    candidate_system_prompt,
    candidate_workspace,
    codex_hook_declared,
    codex_project_trust_config,
    materialize_project_files,
    native_manifest,
    prepare_codex_project_hooks,
    render_toml,
)
from harnesslens.harnesses.harness_workspace import materialize_workspace
from harnesslens.core.profiles import (
    DEFAULT_OPENCODE_CONTEXT_LIMIT,
    DEFAULT_OUTPUT_LIMIT,
)
from harnesslens.harnesses.native_harness_driver import (
    HarnessTurn,
    apply_manifest_prompt,
    deep_merge,
    default_opencode_prefix,
    drop_unsupported_socks_proxy,
    last_json,
    normalize_manifest,
    opencode_binary,
    parse_codex_session,
    parse_opencode_session,
    parse_opencode_text,
    run_process,
    safe,
    stable_hash,
    start_harness_proxy,
    stop_process,
    summarize_records,
)


DEFAULT_CODEX_MODELS_CACHE = "~/.codex/models_cache.json"


def _codex_models_cache() -> Path:
    """Codex model-catalogue cache seeded into each sandboxed CODEX_HOME."""
    return Path(
        os.environ.get("HAI_CODEX_MODELS_CACHE") or DEFAULT_CODEX_MODELS_CACHE
    ).expanduser()


RUNTIME_SCHEMA = "harnesslens.bird-mini-dev-external-harness.v1"
SUPPORTED_BIRD_HARNESSES = ("opencode", "pi-agent", "codex")
DEFAULT_MODEL = "deepseek-v4-flash"
SOURCE_RELATIVE_PATH = Path("finetuning/inference/mini_dev_prompt.jsonl")


@dataclass(frozen=True)
class BirdLimits:
    max_steps: int = 30
    max_rounds: int = 1
    turn_timeout_s: int = 600
    group_timeout_s: int = 14_400
    query_timeout_s: int = 5
    grader_timeout_s: int = 30

    def to_dict(self) -> dict[str, int]:
        return {
            "max_steps": int(self.max_steps),
            "max_rounds": int(self.max_rounds),
            "turn_timeout_s": int(self.turn_timeout_s),
            "group_timeout_s": int(self.group_timeout_s),
            "query_timeout_s": int(self.query_timeout_s),
            "grader_timeout_s": int(self.grader_timeout_s),
        }


@dataclass(frozen=True)
class BirdTask:
    task_id: str
    question_id: int
    db_id: str
    question: str
    evidence: str
    schema: str
    gold_sql: str
    database: Path


def normalize_bird_harness(value: str) -> str:
    aliases = {
        "opencode": "opencode",
        "open-code": "opencode",
        "pi": "pi-agent",
        "pi_agent": "pi-agent",
        "pi-agent": "pi-agent",
        "codex": "codex",
    }
    normalized = aliases.get(str(value).strip().lower())
    if normalized is None:
        raise ValueError(f"unsupported BIRD harness: {value}")
    return normalized


def load_bird_tasks(repo_root: str | Path) -> dict[str, BirdTask]:
    root = Path(repo_root).resolve()
    bird_root = root / "third_party" / "bird-mini-dev"
    source = bird_root / SOURCE_RELATIVE_PATH
    if not source.is_file():
        raise ValueError(f"BIRD Mini-Dev task source is unavailable: {source}")
    tasks: dict[str, BirdTask] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if str(raw.get("difficulty") or "") != "challenging":
            continue
        question_id = int(raw["question_id"])
        task_id = f"bird_{question_id}"
        db_id = str(raw["db_id"])
        tasks[task_id] = BirdTask(
            task_id=task_id,
            question_id=question_id,
            db_id=db_id,
            question=str(raw.get("question") or ""),
            evidence=str(raw.get("evidence") or ""),
            schema=str(raw.get("schema") or ""),
            gold_sql=str(raw.get("SQL") or ""),
            database=(
                bird_root / "data" / "dev_databases" / db_id / f"{db_id}.sqlite"
            ).resolve(),
        )
    return tasks


def format_bird_prompt(task: BirdTask) -> str:
    evidence = task.evidence.strip() or "(none)"
    return (
        "Generate the SQLite query that answers the question.\n\n"
        f"Question:\n{task.question}\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"Database schema:\n{task.schema}\n\n"
        "You may call bird_execute_sql to validate a read-only query. Return the "
        "final answer as exactly one SQLite SELECT/WITH query in a ```sql``` block."
    )


def run_bird_batch(
    *,
    repo_root: str | Path,
    run_root: str | Path,
    scope: str,
    harness: str,
    harness_version: str,
    harness_manifest: Mapping[str, Any] | None,
    task_repeats: Mapping[str, int],
    pairing_offsets: Mapping[str, int],
    max_concurrency: int,
    limits: BirdLimits,
) -> dict[str, Any]:
    drop_unsupported_socks_proxy(os.environ)
    root = Path(repo_root).resolve()
    tasks = load_bird_tasks(root)
    selected = [str(task_id) for task_id in task_repeats]
    normalized_harness = normalize_bird_harness(harness)
    manifest = (
        native_manifest("codex", harness_manifest)
        if harness == "codex"
        else normalize_manifest(harness_manifest or {})
    )
    manifest_hash = stable_hash(manifest)
    output_root = Path(run_root).resolve()
    trajectory_root = output_root / "trajectories"
    trajectory_root.mkdir(parents=True, exist_ok=True)
    _preflight(tasks, selected, normalized_harness, limits)

    workers = max(1, min(int(max_concurrency), len(selected)))
    slots: queue.Queue[int] = queue.Queue()
    for slot in range(workers):
        slots.put(slot)
    records_by_task: dict[str, list[dict[str, Any]]] = {
        task_id: [] for task_id in selected
    }
    started = time.time()

    def run_task(task_id: str) -> tuple[str, list[dict[str, Any]]]:
        slot = slots.get()
        try:
            return task_id, _run_task_repeats(
                repo_root=root,
                trajectory_root=trajectory_root,
                scope=str(scope).upper(),
                harness=normalized_harness,
                harness_version=str(harness_version),
                manifest=manifest,
                manifest_hash=manifest_hash,
                task=tasks[task_id],
                repeats=int(task_repeats[task_id]),
                pairing_offset=int(pairing_offsets.get(task_id, 0)),
                worker_slot=slot,
                limits=limits,
            )
        finally:
            slots.put(slot)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_task, task_id): task_id for task_id in selected}
        for index, future in enumerate(as_completed(futures), start=1):
            task_id = futures[future]
            try:
                _task_id, rows = future.result()
            except Exception as exc:  # noqa: BLE001
                rows = [
                    _error_row(
                        task_id=task_id,
                        trial=trial,
                        pairing_slot=int(pairing_offsets.get(task_id, 0)) + trial,
                        harness=normalized_harness,
                        harness_version=str(harness_version),
                        manifest_hash=manifest_hash,
                        limits=limits,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    for trial in range(int(task_repeats[task_id]))
                ]
                for row in rows:
                    _write_trial_row(trajectory_root, task_id, int(row["trial"]), row)
            records_by_task[task_id].extend(rows)
            print(
                f"[bird/{normalized_harness}/{harness_version}] "
                f"done {index}/{len(selected)}: {task_id}",
                flush=True,
            )

    records: list[dict[str, Any]] = []
    per_task: dict[str, dict[str, Any]] = {}
    for task_id in sorted(records_by_task):
        rows = sorted(
            records_by_task[task_id], key=lambda row: int(row.get("trial") or 0)
        )
        rewards = [float(row.get("reward", 0.0) or 0.0) for row in rows]
        paths = [
            str(
                _trial_path(
                    trajectory_root, task_id, int(row.get("trial") or index)
                ).resolve()
            )
            for index, row in enumerate(rows)
        ]
        errors = [
            {"trial": row.get("trial", index), "error": str(row.get("error") or "")}
            for index, row in enumerate(rows)
            if row.get("error")
        ]
        summaries = [
            {
                "trial": row.get("trial", index),
                "pairing_slot": row.get("pairing_slot", index),
                "reward": float(row.get("reward", 0.0) or 0.0),
                "n_messages": int(row.get("n_messages") or 0),
                "n_tool_calls": int(row.get("n_tool_calls") or 0),
                "termination": str(row.get("termination") or ""),
                "error": str(row.get("error") or "")[:500],
            }
            for index, row in enumerate(rows)
        ]
        record = {
            "task_id": task_id,
            "harness_version": str(harness_version),
            "rewards": rewards,
            "trajectory_paths": paths,
            "worker_errors": errors,
            "trial_summaries": summaries,
        }
        records.append(record)
        per_task[task_id] = {
            "repeats": len(rows),
            "rewards": rewards,
            "trajectory_paths": paths,
            "worker_errors": errors,
            "trial_summaries": summaries,
        }

    all_rows = [row for rows in records_by_task.values() for row in rows]
    infrastructure = sum(bool(row.get("infrastructure_error")) for row in all_rows)
    metrics = {
        **summarize_records(records),
        "requested_trial_count": sum(int(value) for value in task_repeats.values()),
        "infrastructure_failure_count": infrastructure,
        "charged_trial_count": len(all_rows) - infrastructure,
        "elapsed_s": round(time.time() - started, 1),
        "harness": normalized_harness,
        "runtime_schema": RUNTIME_SCHEMA,
        "runtime_limits": limits.to_dict(),
        "evaluation_metric": "official_execution_accuracy_set_equality",
    }
    return {
        "trajectory_root": str(trajectory_root),
        "per_task": per_task,
        "records": records,
        "metrics": metrics,
    }


def _run_task_repeats(
    *,
    repo_root: Path,
    trajectory_root: Path,
    scope: str,
    harness: str,
    harness_version: str,
    manifest: Mapping[str, Any],
    manifest_hash: str,
    task: BirdTask,
    repeats: int,
    pairing_offset: int,
    worker_slot: int,
    limits: BirdLimits,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in range(repeats):
        cached = _load_existing_trial(
            trajectory_root=trajectory_root,
            task_id=task.task_id,
            trial=trial,
            pairing_slot=pairing_offset + trial,
            harness=harness,
            harness_version=harness_version,
            manifest_hash=manifest_hash,
            limits=limits,
        )
        if cached is not None:
            rows.append(cached)
            continue
        try:
            row = _run_single_trial(
                repo_root=repo_root,
                trajectory_root=trajectory_root,
                scope=scope,
                harness=harness,
                harness_version=harness_version,
                manifest=manifest,
                manifest_hash=manifest_hash,
                task=task,
                trial=trial,
                pairing_slot=pairing_offset + trial,
                worker_slot=worker_slot,
                limits=limits,
            )
        except Exception as exc:  # noqa: BLE001
            row = _error_row(
                task_id=task.task_id,
                trial=trial,
                pairing_slot=pairing_offset + trial,
                harness=harness,
                harness_version=harness_version,
                manifest_hash=manifest_hash,
                limits=limits,
                error=f"{type(exc).__name__}: {exc}",
            )
        _write_trial_row(trajectory_root, task.task_id, trial, row)
        rows.append(row)
    return rows


def _run_single_trial(
    *,
    repo_root: Path,
    trajectory_root: Path,
    scope: str,
    harness: str,
    harness_version: str,
    manifest: Mapping[str, Any],
    manifest_hash: str,
    task: BirdTask,
    trial: int,
    pairing_slot: int,
    worker_slot: int,
    limits: BirdLimits,
) -> dict[str, Any]:
    trial_root = (
        trajectory_root / safe(task.task_id) / f"trial_{trial + 1:04d}_artifacts"
    )
    shutil.rmtree(trial_root, ignore_errors=True)
    workspace = trial_root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    materialize_project_files(
        workspace,
        manifest,
        home_root=_candidate_home_root(trial_root, harness),
    )
    if harness == "codex":
        prepare_codex_project_hooks(workspace, manifest)
    mcp_log = trial_root / "mcp_calls.json"
    patches = trial_root / "tool_desc_patches.json"
    patches.write_text(
        json.dumps(manifest.get("tool_desc_patches") or {}, ensure_ascii=False),
        encoding="utf-8",
    )
    socket_path = _bird_mcp_socket_path(
        trajectory_root=trajectory_root,
        worker_slot=worker_slot,
        question_id=task.question_id,
        trial=trial,
    )
    server = _start_mcp_server(
        repo_root=repo_root,
        database=task.database,
        socket_path=socket_path,
        log_file=mcp_log,
        patches_path=patches,
        limits=limits,
    )
    proxy: subprocess.Popen[str] | None = None
    try:
        if harness in {"opencode", "pi-agent"}:
            proxy, proxy_port = _start_chat_logging_proxy(
                trial_root,
                harness=harness,
            )
        else:
            proxy, proxy_port = start_harness_proxy(
                repo_root=repo_root,
                harness=harness,
                socket_path=socket_path,
                artifact_root=trial_root,
            )
        system_prompt = (
            candidate_system_prompt(_system_prompt(), manifest)
            if harness == "pi-agent"
            else apply_manifest_prompt(_system_prompt(), manifest)
        )
        _prepare_harness(
            harness=harness,
            trial_root=trial_root,
            socket_path=socket_path,
            system_prompt=system_prompt,
            proxy_port=proxy_port,
            max_steps=limits.max_steps,
            manifest=manifest,
        )
        prompt = format_bird_prompt(task)
        turn = _run_harness_turn(
            harness=harness,
            trial_root=trial_root,
            workspace=workspace,
            socket_path=socket_path,
            prompt=prompt,
            timeout_s=limits.turn_timeout_s,
            proxy_port=proxy_port,
            manifest=manifest,
        )
        (trial_root / "harness_stdout.txt").write_text(turn.stdout, encoding="utf-8")
        (trial_root / "harness_stderr.txt").write_text(turn.stderr, encoding="utf-8")
        if turn.returncode not in {0, None}:
            raise RuntimeError(
                f"{harness} exited {turn.returncode}: {turn.stderr[-1000:]}"
            )
        predicted_sql = extract_sql(turn.text)
        grading = (
            grade_execution_accuracy(
                task.database,
                predicted_sql,
                task.gold_sql,
                timeout_s=limits.grader_timeout_s,
            )
            if predicted_sql
            else {
                "passed": False,
                "error": "ValueError: no SELECT/WITH query found in final response",
                "predicted_row_count": None,
                "gold_row_count": None,
                "diagnostic": {
                    "mismatch_type": "missing_final_sql",
                    "predicted_row_count": None,
                    "reference_row_count": None,
                    "predicted_column_count": None,
                    "reference_column_count": None,
                    "predicted_unique_row_count": None,
                    "reference_unique_row_count": None,
                    "duplicate_profile_mismatch": False,
                },
                "elapsed_s": 0.0,
            }
        )
        calls = _read_mcp_calls(mcp_log)
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        for call in calls:
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "tool_name": str(call.get("name") or ""),
                        "tool_arguments": dict(call.get("arguments") or {}),
                    },
                    {
                        "role": "tool",
                        "name": str(call.get("name") or ""),
                        "content": str(call.get("result") or ""),
                        "is_error": bool(call.get("is_error")),
                    },
                ]
            )
        messages.append({"role": "assistant", "content": turn.text})
        runtime_project = (
            _isolated_runtime_cwd(trial_root, "pi")
            if harness == "pi-agent"
            else (
                _isolated_runtime_cwd(trial_root, "codex")
                if harness == "codex"
                else workspace
            )
        )
        model_context = build_runtime_load_report(
            harness=harness,
            project_root=runtime_project,
            home_root=_candidate_home_root(trial_root, harness),
            manifest=manifest,
            tool_definitions=_bird_tool_definitions(),
        )
        passed = bool(grading["passed"])
        api_trace_name = {
            "opencode": "opencode_api_calls.jsonl",
            "pi-agent": "pi_api_calls.jsonl",
            "codex": "codex_api_calls.jsonl",
        }.get(harness, "")
        api_trace = trial_root / api_trace_name if api_trace_name else None
        return {
            "runtime_schema": RUNTIME_SCHEMA,
            "task_id": task.task_id,
            "question_id": task.question_id,
            "db_id": task.db_id,
            "difficulty": "challenging",
            "trial": int(trial),
            "pairing_slot": int(pairing_slot),
            "scope": scope,
            "runner": harness,
            "harness_version": harness_version,
            "manifest_sha256": manifest_hash,
            "limits": limits.to_dict(),
            "reward": float(passed),
            "task_score": float(passed),
            "passed": passed,
            "status": "completed",
            "error": "",
            "grader_error": str(grading.get("error") or ""),
            "infrastructure_error": False,
            "termination": "completed" if predicted_sql else "no_sql",
            "predicted_sql": predicted_sql,
            "predicted_row_count": grading.get("predicted_row_count"),
            "gold_row_count": grading.get("gold_row_count"),
            "grader_diagnostic": dict(grading.get("diagnostic") or {}),
            "grading_elapsed_s": grading.get("elapsed_s"),
            "n_messages": len(messages),
            "n_tool_calls": len(calls),
            "user_agent_rounds": 0,
            "user_agent_max_rounds": int(limits.max_rounds),
            "messages": messages,
            "tool_calls": [str(call.get("call_str") or "") for call in calls],
            "model_context": model_context,
            "api_calls_jsonl": (
                str(api_trace.resolve())
                if api_trace is not None
                and api_trace.is_file()
                and api_trace.stat().st_size > 0
                else ""
            ),
            "mcp_calls_json": str(mcp_log.resolve()),
            "elapsed_s": round(turn.elapsed_s, 2),
        }
    finally:
        for runtime_harness in ("pi", "codex"):
            shutil.rmtree(
                _isolated_runtime_cwd(trial_root, runtime_harness),
                ignore_errors=True,
            )
        stop_process(proxy)
        stop_process(server)
        socket_path.unlink(missing_ok=True)
        _cleanup_trial_runtime(trial_root)


def _bird_tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": "execute_sql",
            "description": (
                "Execute one read-only SQLite SELECT/WITH query against the current "
                "BIRD database. Use this to validate joins, filters, and result shape "
                "before returning the final SQL. Results are limited to 200 rows."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A single SQLite SELECT or WITH query.",
                    }
                },
                "required": ["sql"],
                "additionalProperties": False,
            },
        }
    ]


def _bird_mcp_socket_path(
    *, trajectory_root: Path, worker_slot: int, question_id: int, trial: int
) -> Path:
    rollout_identity = stable_hash(
        {
            "trajectory_root": str(trajectory_root.resolve()),
            "worker_slot": int(worker_slot),
            "question_id": int(question_id),
            "trial": int(trial),
        }
    )[:24]
    return Path("/tmp") / f"bird_harnesslens_{os.getpid()}_{rollout_identity}.sock"


def _start_chat_logging_proxy(
    trial_root: Path,
    *,
    harness: str,
) -> tuple[subprocess.Popen[str], int]:
    log_name = (
        "opencode_api_calls.jsonl" if harness == "opencode" else "pi_api_calls.jsonl"
    )
    command = [
        sys.executable,
        str(
            Path(__file__).resolve().parents[1]
            / "infrastructure"
            / "chat_completions_proxy.py"
        ),
        "--log-file",
        str(trial_root / log_name),
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
        raise RuntimeError(f"{harness} logging proxy failed: {line} {stderr[-1000:]}")
    return process, int(line.split("=", 1)[1])


def extract_sql(text: str) -> str:
    value = str(text or "").strip()
    fenced = re.findall(
        r"```(?:sql)?\s*(.*?)```", value, flags=re.IGNORECASE | re.DOTALL
    )
    candidates = [*reversed(fenced), value]
    for candidate in candidates:
        match = re.search(r"\b(select|with)\b", candidate, flags=re.IGNORECASE)
        if not match:
            continue
        statement = candidate[match.start() :].strip()
        for index, character in enumerate(statement):
            if character == ";" and sqlite3.complete_statement(statement[: index + 1]):
                return statement[: index + 1].strip()
        if candidate is not value or re.match(r"(?is)^(select|with)\b", statement):
            return statement.strip()
    return ""


def _system_prompt() -> str:
    return (
        "You are a text-to-SQL agent for the classic BIRD Mini-Dev SQLite benchmark. "
        "Use only the supplied schema, evidence, question, and the read-only SQL tool. "
        "Do not invent tables or columns. The final response must contain exactly one "
        "SELECT/WITH query in a sql code block and no alternative query."
    )


def _start_mcp_server(
    *,
    repo_root: Path,
    database: Path,
    socket_path: Path,
    log_file: Path,
    patches_path: Path,
    limits: BirdLimits,
) -> subprocess.Popen[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("bird_mcp_server.py")),
        "server",
        "--database",
        str(database),
        "--socket",
        str(socket_path),
        "--log-file",
        str(log_file),
        "--max-steps",
        str(limits.max_steps),
        "--query-timeout",
        str(limits.query_timeout_s),
        "--tool-desc-patches",
        str(patches_path),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "HAI_REPO_ROOT": str(repo_root)},
        start_new_session=True,
    )
    assert process.stdout is not None
    ready = process.stdout.readline().strip()
    if ready != "READY":
        stderr = process.stderr.read() if process.stderr else ""
        stop_process(process)
        raise RuntimeError(f"BIRD MCP server failed: {ready} {stderr[-1000:]}")
    return process


def _prepare_harness(
    *,
    harness: str,
    trial_root: Path,
    socket_path: Path,
    system_prompt: str,
    proxy_port: int | None,
    max_steps: int,
    manifest: Mapping[str, Any],
) -> None:
    if harness == "opencode":
        candidate_config = load_json_configs(
            (
                trial_root / "xdg_config" / "opencode" / "config.json",
                trial_root / "xdg_config" / "opencode" / "opencode.json",
                trial_root / "workspace" / "opencode.json",
            )
        )
        candidate_config = relocate_opencode_instruction_paths(
            candidate_config,
            project_root=trial_root / "workspace",
        )
        candidate_config = deep_merge(
            candidate_config,
            {"agent": compile_opencode_agent_definitions(trial_root / "workspace")},
        )
        config = _opencode_config(
            socket_path=socket_path,
            system_prompt=system_prompt,
            proxy_port=proxy_port,
            max_steps=max_steps,
            manifest=manifest,
            candidate_config=candidate_config,
        )
        (trial_root / "opencode.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    if harness == "pi-agent":
        (trial_root / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")
        return
    if proxy_port is None:
        raise RuntimeError("Codex proxy did not start")
    runtime_cwd = _isolated_runtime_cwd(trial_root, "codex")
    shutil.rmtree(runtime_cwd, ignore_errors=True)
    workspace = trial_root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copytree(workspace, runtime_cwd)
    _write_codex_home(
        trial_root=trial_root,
        socket_path=socket_path,
        proxy_port=proxy_port,
        system_prompt=system_prompt,
        manifest=manifest,
    )
    (trial_root / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")


def _opencode_config(
    *,
    socket_path: Path,
    system_prompt: str,
    max_steps: int,
    manifest: Mapping[str, Any],
    candidate_config: Mapping[str, Any] | None = None,
    proxy_port: int | None = None,
) -> dict[str, Any]:
    base_url = (
        f"http://127.0.0.1:{proxy_port}/v1"
        if proxy_port is not None
        else str(
            os.environ.get("DEEPSEEK_BASE_URL")
            or os.environ.get("DEEPSEEK_URL")
            or "https://api.deepseek.com"
        ).rstrip("/")
    )
    if base_url.endswith("/chat/completions"):
        base_url = base_url[: -len("/chat/completions")]
    candidate = merge_candidate_config(
        candidate_config or {},
        legacy_flat_patch=manifest.get("config_patch") or {},
        fixed={},
    )
    candidate_prompt = str(
        ((candidate.get("agent") or {}).get("build") or {}).get("prompt") or ""
    )
    prompt = "\n\n".join(
        item.strip() for item in (system_prompt, candidate_prompt) if item.strip()
    )
    fixed: dict[str, Any] = {
        "$schema": "https://opencode.ai/config.json",
        "autoupdate": False,
        "share": "disabled",
        "model": "deepseek/deepseek-v4-flash",
        "small_model": "deepseek/deepseek-v4-flash",
        "enabled_providers": ["deepseek"],
        "provider": {
            "deepseek": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "DeepSeek",
                "options": {
                    "baseURL": base_url,
                    "apiKey": "{env:DEEPSEEK_API_KEY}",
                    "timeout": 600000,
                    "chunkTimeout": 60000,
                },
                "models": {
                    DEFAULT_MODEL: {
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
        "permission": {"*": "allow"},
        "mcp": {
            "bird": {
                "type": "local",
                "enabled": True,
                "command": [
                    sys.executable,
                    str(Path(__file__).with_name("bird_mcp_server.py")),
                    "bridge",
                    "--socket",
                    str(socket_path),
                ],
            }
        },
    }
    return merge_candidate_config(candidate, fixed=fixed)


def _write_codex_home(
    *,
    trial_root: Path,
    socket_path: Path,
    proxy_port: int,
    system_prompt: str,
    manifest: Mapping[str, Any],
) -> None:
    home = trial_root / "codex_home"
    home.mkdir(parents=True, exist_ok=True)
    materialize_workspace(
        candidate_workspace(manifest),
        home_root=home,
        project_root=_isolated_runtime_cwd(trial_root, "codex"),
    )
    config_patch = dict(manifest.get("config_patch") or {})
    candidate_config = load_toml_configs(
        (
            home / "config.toml",
            _isolated_runtime_cwd(trial_root, "codex") / ".codex" / "config.toml",
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
    config = merge_candidate_config(
        candidate_config,
        fixed={
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
                "shell_tool": False,
                "unified_exec": False,
            },
            "mcp_servers": {
                "bird": {
                    "command": sys.executable,
                    "args": [
                        str(Path(__file__).with_name("bird_mcp_server.py")),
                        "bridge",
                        "--socket",
                        str(socket_path),
                    ],
                }
            },
            **(
                codex_project_trust_config(_isolated_runtime_cwd(trial_root, "codex"))
                if codex_hook_declared(manifest)
                else {}
            ),
        },
    )
    (home / "config.toml").write_text(render_toml(config), encoding="utf-8")
    source = _codex_models_cache()
    if source.is_file():
        shutil.copy2(source, home / "models_cache.json")


def _run_harness_turn(
    *,
    harness: str,
    trial_root: Path,
    workspace: Path,
    socket_path: Path,
    prompt: str,
    timeout_s: int,
    proxy_port: int | None,
    manifest: Mapping[str, Any],
) -> HarnessTurn:
    if harness == "opencode":
        return _run_opencode_turn(
            trial_root=trial_root,
            workspace=workspace,
            prompt=prompt,
            timeout_s=timeout_s,
        )
    if harness == "pi-agent":
        return _run_pi_turn(
            repo_root=Path(__file__).resolve().parents[2],
            trial_root=trial_root,
            workspace=workspace,
            socket_path=socket_path,
            prompt=prompt,
            timeout_s=timeout_s,
            manifest=manifest,
            proxy_port=proxy_port,
        )
    return _run_codex_turn(
        trial_root=trial_root,
        workspace=workspace,
        prompt=prompt,
        timeout_s=timeout_s,
    )


def _run_opencode_turn(
    *, trial_root: Path, workspace: Path, prompt: str, timeout_s: int
) -> HarnessTurn:
    command = [
        str(opencode_binary()),
        "run",
        "-m",
        "deepseek/deepseek-v4-flash",
        "--pure",
        "--auto",
        "--dir",
        str(workspace),
        "--format",
        "json",
        prompt,
    ]
    env = _harness_env(trial_root)
    env["OPENCODE_CONFIG"] = str(trial_root / "opencode.json")
    env["OPENCODE_DISABLE_AUTOCOMPACT"] = "1"
    env["OPENCODE_DISABLE_PRUNE"] = "1"
    started = time.time()
    stdout, stderr, returncode = run_process(
        command, cwd=workspace, env=env, timeout_s=timeout_s
    )
    return HarnessTurn(
        text=parse_opencode_text(stdout),
        stdout=stdout,
        stderr=stderr,
        session_id=parse_opencode_session(stdout),
        returncode=returncode,
        elapsed_s=time.time() - started,
    )



def _run_codex_turn(
    *, trial_root: Path, workspace: Path, prompt: str, timeout_s: int
) -> HarnessTurn:
    binary = shutil.which("codex")
    if not binary:
        raise RuntimeError("codex executable is unavailable")
    last_message = trial_root / "codex_last.txt"
    command = [
        binary,
        "exec",
        "--strict-config",
        "--skip-git-repo-check",
        "--json",
        "--output-last-message",
        str(last_message),
        prompt,
    ]
    if (workspace / ".codex" / "hooks.json").is_file():
        command.insert(3, "--dangerously-bypass-hook-trust")
    env = _harness_env(trial_root)
    env.pop("DEEPSEEK_API_KEY", None)
    env.update(
        {
            "OPENAI_API_KEY": "harnesslens-local-proxy",
            "CODEX_HOME": str(trial_root / "codex_home"),
        }
    )
    started = time.time()
    stdout, stderr, returncode = run_process(
        command,
        cwd=_isolated_runtime_cwd(trial_root, "codex"),
        env=env,
        timeout_s=timeout_s,
    )
    text = (
        last_message.read_text(encoding="utf-8", errors="replace").strip()
        if last_message.exists()
        else ""
    )
    return HarnessTurn(
        text=text,
        stdout=stdout,
        stderr=stderr,
        session_id=parse_codex_session(stdout),
        returncode=returncode,
        elapsed_s=time.time() - started,
    )


def _run_pi_turn(
    *,
    repo_root: Path,
    trial_root: Path,
    workspace: Path,
    socket_path: Path,
    prompt: str,
    timeout_s: int,
    manifest: Mapping[str, Any],
    proxy_port: int | None,
) -> HarnessTurn:
    from harnesslens.benchmarks.pi_tau2 import _PiRpcSession

    runtime_cwd = _isolated_runtime_cwd(trial_root, "pi")
    pi_home = trial_root / "pi_home"
    shutil.rmtree(runtime_cwd, ignore_errors=True)
    shutil.copytree(workspace, runtime_cwd, dirs_exist_ok=True)
    _write_pi_bird_project(
        runtime_cwd=runtime_cwd,
        pi_home=pi_home,
        manifest=manifest,
        proxy_port=proxy_port,
    )
    stdout_path = trial_root / "pi_rpc.stdout.jsonl"
    stderr_path = trial_root / "pi_rpc.stderr.txt"
    session = _PiRpcSession.start(
        repo_root=repo_root,
        runtime_cwd=runtime_cwd,
        pi_home=pi_home,
        socket_path=str(socket_path),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        system_prompt_append=(trial_root / "system_prompt.txt").read_text(
            encoding="utf-8"
        ),
    )
    started = time.time()
    try:
        result = session.prompt(prompt, timeout_s=timeout_s)
        return HarnessTurn(
            text=result.text,
            stdout=(
                stdout_path.read_text(encoding="utf-8", errors="replace")
                if stdout_path.exists()
                else ""
            ),
            stderr=result.stderr,
            session_id=None,
            returncode=0 if not result.error else 1,
            elapsed_s=time.time() - started,
        )
    finally:
        session.close()


def _write_pi_bird_project(
    *,
    runtime_cwd: Path,
    pi_home: Path,
    manifest: Mapping[str, Any],
    proxy_port: int | None = None,
) -> None:
    from harnesslens.benchmarks.pi_tau2 import MODEL as PI_MODEL

    pi_dir = runtime_cwd / ".pi"
    pi_dir.mkdir(parents=True, exist_ok=True)
    pi_home.mkdir(parents=True, exist_ok=True)
    materialize_workspace(
        candidate_workspace(manifest),
        home_root=pi_home,
        project_root=runtime_cwd,
    )
    base_url = (
        f"http://127.0.0.1:{int(proxy_port)}/v1"
        if proxy_port is not None
        else provider_base_url()
    ).rstrip("/")
    if base_url.endswith("/chat/completions"):
        base_url = base_url[: -len("/chat/completions")]
    candidate_config = load_json_configs(
        (
            pi_home / "settings.json",
            pi_home / ".pi" / "settings.json",
            runtime_cwd / ".pi" / "settings.json",
        )
    )
    settings = merge_candidate_config(
        candidate_config,
        legacy_flat_patch=manifest.get("config_patch") or {},
        fixed={
            "permissions": {"allow": [], "deny": []},
            "provider": {
                "default": "deepseek",
                "providers": {
                    "deepseek": {
                        "baseURL": base_url,
                        "apiKeyEnvVar": "DEEPSEEK_API_KEY",
                        "models": [PI_MODEL],
                    }
                },
            },
            "model": PI_MODEL,
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
                        "apiKey": "harnesslens-local-proxy",
                        "compat": {
                            "supportsDeveloperRole": False,
                            "supportsReasoningEffort": False,
                        },
                        "models": [
                            {
                                "id": PI_MODEL,
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
        _pi_bird_extension_source(), encoding="utf-8"
    )


def _pi_bird_extension_source() -> str:
    return r"""
import { Type } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import net from "node:net";

type Json = Record<string, any>;
let rpcId = 1;

function rpc(method: string, params: Json = {}): Promise<Json> {
  const socketPath = process.env.HAI_TAU2_SOCKET;
  if (!socketPath) throw new Error("BIRD MCP socket is not configured");
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
      } catch (error) { reject(error); }
    });
    client.on("error", reject);
  });
}

export default async function (pi: ExtensionAPI) {
  const listed = await rpc("tools/list", {});
  for (const tool of listed.result?.tools ?? []) {
    const name = String(tool.name || "");
    if (!name) continue;
    pi.registerTool({
      name,
      label: name,
      description: String(tool.description || ""),
      parameters: Type.Unsafe(tool.inputSchema || { type: "object", properties: {} }),
      async execute(_toolCallId: string, args: Json) {
        const result = await rpc("tools/call", { name, arguments: args || {} });
        const content = result.result?.content ?? [];
        const text = content.map((part: Json) => String(part.text ?? "")).join("\n");
        return { content: [{ type: "text", text }], details: result.result ?? {} };
      },
    });
  }
}
"""


def _harness_env(trial_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    shared = Path(__file__).resolve().parents[2] / ".cache" / "bird-opencode"
    config = trial_root / "xdg_config"
    npm_cache = shared / "npm"
    home = trial_root / "home"
    data = trial_root / "xdg_data"
    cache = trial_root / "xdg_cache"
    tmp = trial_root / "tmp"
    for path in (config, npm_cache, home, data, cache, tmp):
        path.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "HOME": str(home),
            "XDG_CONFIG_HOME": str(config),
            "XDG_DATA_HOME": str(data),
            "XDG_CACHE_HOME": str(cache),
            "TMPDIR": str(tmp),
            "npm_config_cache": str(npm_cache),
            "NPM_CONFIG_CACHE": str(npm_cache),
        }
    )
    bypass = [item for item in str(env.get("NO_PROXY") or "").split(",") if item]
    for item in ("127.0.0.1", "localhost", "::1", "api.deepseek.com", ".deepseek.com"):
        if item not in bypass:
            bypass.append(item)
    env["NO_PROXY"] = env["no_proxy"] = ",".join(bypass)
    return env


def _candidate_home_root(trial_root: Path, harness: str) -> Path:
    if harness == "pi-agent":
        return trial_root / "pi_home"
    if harness == "codex":
        return trial_root / "codex_home"
    if harness == "opencode":
        return trial_root / "xdg_config" / "opencode"
    return trial_root / "harness_home"


def _isolated_runtime_cwd(trial_root: Path, harness: str) -> Path:
    digest = stable_hash({"trial_root": str(trial_root.resolve())})[:20]
    return Path("/tmp") / "harnesslens_bird_runtime" / f"{safe(harness)}_{digest}"


def _cleanup_trial_runtime(trial_root: Path) -> None:
    if os.environ.get("HAI_KEEP_TRAJECTORY_WORKSPACE", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    for name in (
        "home",
        "xdg_config",
        "xdg_data",
        "xdg_cache",
        "tmp",
        "workspace",
        "pi_runtime",
        "pi_home",
        "codex_home",
    ):
        shutil.rmtree(trial_root / name, ignore_errors=True)


def _preflight(
    tasks: Mapping[str, BirdTask],
    task_ids: Sequence[str],
    harness: str,
    limits: BirdLimits,
) -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is required for BIRD Mini-Dev")
    if not 1 <= int(limits.max_steps) <= 200:
        raise ValueError("BIRD max_steps must be between 1 and 200")
    if int(limits.max_rounds) != 1:
        raise ValueError("BIRD is single-turn; max_rounds must be 1")
    for task_id in task_ids:
        if task_id not in tasks:
            raise ValueError(f"BIRD challenging task is unavailable: {task_id}")
        if not tasks[task_id].database.is_file():
            raise ValueError(f"BIRD database is unavailable: {tasks[task_id].database}")
    if harness == "opencode":
        opencode_binary()
    elif harness == "codex" and not shutil.which("codex"):
        raise RuntimeError("codex executable is unavailable")
    elif harness == "pi-agent":
        from harnesslens.benchmarks.pi_tau2 import _pi_binary

        _pi_binary(Path(__file__).resolve().parents[2])


def _read_mcp_calls(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [dict(item) for item in payload if isinstance(item, Mapping)]


def _materialize_manifest_files(workspace: Path, manifest: Mapping[str, Any]) -> None:
    for item in manifest.get("files") or []:
        if not isinstance(item, Mapping):
            continue
        raw = str(item.get("path") or "")
        path = Path(raw)
        if not raw or path.is_absolute() or ".." in path.parts:
            continue
        target = workspace / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(item.get("content") or ""), encoding="utf-8")


def _load_existing_trial(
    *,
    trajectory_root: Path,
    task_id: str,
    trial: int,
    pairing_slot: int,
    harness: str,
    harness_version: str,
    manifest_hash: str,
    limits: BirdLimits,
) -> dict[str, Any] | None:
    path = _trial_path(trajectory_root, task_id, trial)
    try:
        row = json.loads(path.read_text(encoding="utf-8").strip())
    except (OSError, json.JSONDecodeError):
        return None
    expected = {
        "runtime_schema": RUNTIME_SCHEMA,
        "task_id": task_id,
        "trial": trial,
        "pairing_slot": pairing_slot,
        "runner": harness,
        "harness_version": harness_version,
        "manifest_sha256": manifest_hash,
        "limits": limits.to_dict(),
        "status": "completed",
    }
    if not all(row.get(key) == value for key, value in expected.items()):
        return None
    if row.get("infrastructure_error") or row.get("error"):
        return None
    return dict(row)


def _error_row(
    *,
    task_id: str,
    trial: int,
    pairing_slot: int,
    harness: str,
    harness_version: str,
    manifest_hash: str,
    limits: BirdLimits,
    error: str,
) -> dict[str, Any]:
    return {
        "runtime_schema": RUNTIME_SCHEMA,
        "task_id": task_id,
        "trial": int(trial),
        "pairing_slot": int(pairing_slot),
        "runner": harness,
        "harness_version": harness_version,
        "manifest_sha256": manifest_hash,
        "limits": limits.to_dict(),
        "reward": 0.0,
        "task_score": 0.0,
        "passed": False,
        "status": "error",
        "error": str(error)[:4000],
        "infrastructure_error": True,
        "termination": "error",
        "n_messages": 0,
        "n_tool_calls": 0,
        "messages": [],
    }


def _trial_path(root: Path, task_id: str, trial: int) -> Path:
    return root / safe(task_id) / f"trial_{trial + 1:04d}.jsonl"


def _write_trial_row(
    root: Path, task_id: str, trial: int, row: Mapping[str, Any]
) -> Path:
    path = _trial_path(root, task_id, trial)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
