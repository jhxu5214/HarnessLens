from __future__ import annotations

import functools
import hashlib
import json
import os
import queue
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import yaml

from harnesslens.harnesses.candidate_config_runtime import (
    load_json_configs,
    load_toml_configs,
    merge_candidate_config,
    relocate_opencode_instruction_paths,
)
from harnesslens.harnesses.channel_preflight import build_runtime_load_report
from harnesslens.infrastructure.container_reaper import (
    ownership_labels,
    reap_orphaned_containers,
)
from harnesslens.harnesses.harness_workspace import materialize_workspace
from harnesslens.harnesses.native_candidate_runtime import (
    candidate_workspace,
    render_toml,
)
from harnesslens.evaluation.rollout_outcome import (
    harness_execution_error,
    provider_trace_error,
)
from harnesslens.infrastructure.provider_capacity import provider_trial_slot
from harnesslens.core.profiles import (
    DEFAULT_OPENCODE_CONTEXT_LIMIT,
    DEFAULT_OUTPUT_LIMIT,
)

try:
    from .terminal_cache import hash_path, locked_entry
except ImportError:  # Standalone baseline adapter copy.
    from terminal_cache import hash_path, locked_entry


DEFAULT_CLASHCTL_HOME = "~/clashctl"
DEFAULT_DOCKER_ROOT = "~/dockers"


def _default_docker_host() -> str:
    """Socket of the rootless dockerd, overridable without editing this file."""
    root = Path(os.environ.get("HAI_DOCKER_ROOT") or DEFAULT_DOCKER_ROOT).expanduser()
    return os.environ.get("HAI_DOCKER_HOST") or f"unix://{root / 'run' / 'docker.sock'}"


RUNTIME_SCHEMA = "harnesslens.terminal-bench-docker.v8"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_IMAGE_TEMPLATE = "alexgshaw/{task_id}:20251031"
PI_RUNNER_FILENAME = "pi_compact_runner.mjs"
CODEX_DISABLED_TOOLS_BY_TASK = {
    "video-processing": ("view_image",),
}
AGENT_OUTPUT_LIMIT_CHARS = 8 * 1024 * 1024
AGENT_EXIT_DRAIN_GRACE_S = 5.0
CLASH_START_CONCURRENCY = max(
    1, int(os.environ.get("TB_CLASH_START_CONCURRENCY", "2"))
)
_CLASH_START_SEMAPHORE = threading.BoundedSemaphore(CLASH_START_CONCURRENCY)
INFRA_ERROR_MARKERS = (
    "502 bad gateway",
    "502  bad gateway",
    "connection reset",
    "connection refused",
    "couldn't connect to server",
    "docker exec",
    "oci runtime",
    "no such container",
    "unable to fetch",
    "temporary failure",
    "first_event_timeout",
    "stream disconnected before completion",
    "error sending request for url",
    "request timed out",
    "authentication fails",
    "authentication_error",
    "invalid api key",
)


@dataclass(frozen=True)
class TerminalLimits:
    max_steps: int = 50
    agent_timeout_s: int = 1200
    verify_timeout_s: int = 1800
    first_event_timeout_s: int = 180
    infrastructure_retries: int = 2

    def to_dict(self) -> dict[str, int]:
        return {
            "max_steps": int(self.max_steps),
            "agent_timeout_s": int(self.agent_timeout_s),
            "verify_timeout_s": int(self.verify_timeout_s),
            "first_event_timeout_s": int(self.first_event_timeout_s),
            "infrastructure_retries": int(self.infrastructure_retries),
        }


@dataclass
class AgentProcessResult:
    returncode: int
    stdout: str
    stderr: str
    saw_event: bool
    timed_out: bool
    timeout_kind: str
    n_tool_calls: int


def run_terminal_batch(
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
    limits: TerminalLimits,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    os.environ.setdefault("DOCKER_HOST", _default_docker_host())
    for orphan in reap_orphaned_containers():
        print(
            f"[terminal] reclaimed container {orphan['name']}: {orphan['reason']}",
            flush=True,
        )
    output_root = Path(run_root).resolve()
    trajectory_root = output_root / "trajectories"
    trajectory_root.mkdir(parents=True, exist_ok=True)
    normalized_harness = normalize_harness(harness)
    manifest = normalize_manifest(harness_manifest or {})
    _terminal_preflight([str(task_id) for task_id in task_repeats])
    all_jobs = [
        (
            str(task_id),
            local_trial,
            int(pairing_offsets.get(str(task_id), 0)) + local_trial,
        )
        for task_id, repeats in task_repeats.items()
        for local_trial in range(int(repeats))
    ]
    records_by_task: dict[str, list[dict[str, Any]]] = {
        str(task_id): [] for task_id in task_repeats
    }
    jobs: list[tuple[str, int, int]] = []
    resumed_trial_count = 0
    for task_id, local_trial, pairing_slot in all_jobs:
        existing = _load_existing_trial_row(
            trajectory_root=trajectory_root,
            task_id=task_id,
            local_trial=local_trial,
            pairing_slot=pairing_slot,
            harness=normalized_harness,
            harness_version=str(harness_version),
        )
        if existing is None:
            jobs.append((task_id, local_trial, pairing_slot))
            continue
        records_by_task[task_id].append(existing)
        resumed_trial_count += 1
    started = time.time()
    if jobs:
        workers = max(1, min(int(max_concurrency), len(jobs)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _run_cached_trial,
                    repo_root=root,
                    trajectory_root=trajectory_root,
                    scope=str(scope).upper(),
                    harness=normalized_harness,
                    harness_version=str(harness_version),
                    harness_manifest=manifest,
                    task_id=task_id,
                    local_trial=local_trial,
                    pairing_slot=pairing_slot,
                    limits=limits,
                ): (task_id, local_trial, pairing_slot)
                for task_id, local_trial, pairing_slot in jobs
            }
            done = resumed_trial_count
            for future in as_completed(futures):
                task_id, local_trial, pairing_slot = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "task_id": task_id,
                        "trial": local_trial,
                        "pairing_slot": pairing_slot,
                        "reward": 0.0,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "infrastructure_error": True,
                        "verifier_completed": False,
                        "runtime_schema": RUNTIME_SCHEMA,
                    }
                _write_trial_row(trajectory_root, task_id, local_trial, row)
                records_by_task[task_id].append(row)
                done += 1
                print(
                    f"[terminal/{normalized_harness}/{harness_version}] done {done}/{len(all_jobs)}: "
                    f"{task_id} trial={local_trial} cache={bool((row.get('shared_cache') or {}).get('hit'))}",
                    flush=True,
                )

    records: list[dict[str, Any]] = []
    per_task: dict[str, dict[str, Any]] = {}
    for task_id in sorted(records_by_task):
        rows = sorted(
            records_by_task[task_id], key=lambda item: int(item.get("trial") or 0)
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
            {
                "trial": row.get("trial", index),
                "error": str(row.get("error") or row.get("termination") or ""),
            }
            for index, row in enumerate(rows)
            if str(row.get("status") or "completed") == "error" or row.get("error")
        ]
        summaries = [
            {
                "trial": row.get("trial", index),
                "pairing_slot": row.get("pairing_slot", row.get("trial", index)),
                "reward": float(row.get("reward", 0.0) or 0.0),
                "n_messages": int(
                    row.get("n_messages") or len(row.get("messages") or []) or 0
                ),
                "n_tool_calls": int(row.get("n_tool_calls") or 0),
                "termination": str(row.get("termination") or ""),
                "error": str(row.get("error") or "")[:500],
                "shared_cache_hit": bool((row.get("shared_cache") or {}).get("hit")),
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
    trial_rows = [row for rows in records_by_task.values() for row in rows]
    infrastructure = sum(bool(row.get("infrastructure_error")) for row in trial_rows)
    cache_hits = sum(
        bool((row.get("shared_cache") or {}).get("hit")) for row in trial_rows
    )
    metrics = {
        **summarize_records(records),
        "requested_trial_count": len(all_jobs),
        "resumed_trial_count": resumed_trial_count,
        "infrastructure_failure_count": infrastructure,
        "charged_trial_count": max(0, len(all_jobs) - infrastructure),
        "shared_cache_hit_count": cache_hits,
        "shared_cache_miss_count": len(all_jobs) - cache_hits,
        "elapsed_s": round(time.time() - started, 1),
        "harness": normalized_harness,
        "runtime_schema": RUNTIME_SCHEMA,
    }
    return {
        "trajectory_root": str(trajectory_root),
        "per_task": per_task,
        "records": records,
        "metrics": metrics,
    }


def _load_existing_trial_row(
    *,
    trajectory_root: Path,
    task_id: str,
    local_trial: int,
    pairing_slot: int,
    harness: str,
    harness_version: str,
) -> dict[str, Any] | None:
    path = _trial_path(trajectory_root, task_id, local_trial)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        row = json.loads(lines[-1])
    except (OSError, IndexError, json.JSONDecodeError):
        return None
    if not isinstance(row, Mapping):
        return None
    if (
        str(row.get("runtime_schema") or "") != RUNTIME_SCHEMA
        or str(row.get("task_id") or "") != str(task_id)
        or int(row.get("trial", -1)) != int(local_trial)
        or int(row.get("pairing_slot", -1)) != int(pairing_slot)
        or str(row.get("runner") or "") != harness
        or str(row.get("harness_version") or "") != str(harness_version)
        or str(row.get("status") or "") != "completed"
        or bool(row.get("infrastructure_error"))
        or not bool(row.get("verifier_completed"))
        or not _trial_clash_compatible(row)
    ):
        return None
    return dict(row)


def _run_cached_trial(
    *,
    repo_root: Path,
    trajectory_root: Path,
    scope: str,
    harness: str,
    harness_version: str,
    harness_manifest: Mapping[str, Any],
    task_id: str,
    local_trial: int,
    pairing_slot: int,
    limits: TerminalLimits,
) -> dict[str, Any]:
    if str(scope).upper() != "TRAIN":
        result = _run_trial_with_retries(
            repo_root=repo_root,
            trajectory_root=trajectory_root,
            harness=harness,
            harness_version=harness_version,
            harness_manifest=harness_manifest,
            task_id=task_id,
            local_trial=local_trial,
            pairing_slot=pairing_slot,
            limits=limits,
        )
        result["shared_cache"] = {"hit": False, "disabled": "non_train_scope"}
        return result
    cache_manifest = _cache_manifest(
        repo_root=repo_root,
        scope=scope,
        harness=harness,
        harness_manifest=harness_manifest,
        task_id=task_id,
        pairing_slot=pairing_slot,
        limits=limits,
    )
    with locked_entry(repo_root, cache_manifest) as entry:
        cached = entry.load()
        if cached is not None and _trial_clash_compatible(cached):
            cached["trial"] = int(local_trial)
            cached["pairing_slot"] = int(pairing_slot)
            cached["harness_version"] = str(harness_version)
            return cached
        last = _run_trial_with_retries(
            repo_root=repo_root,
            trajectory_root=trajectory_root,
            harness=harness,
            harness_version=harness_version,
            harness_manifest=harness_manifest,
            task_id=task_id,
            local_trial=local_trial,
            pairing_slot=pairing_slot,
            limits=limits,
        )
        if entry.store(last):
            last["shared_cache"] = {
                "hit": False,
                "key": entry.key,
                "object": str(entry.path),
            }
        return last


def _run_trial_with_retries(
    *,
    repo_root: Path,
    trajectory_root: Path,
    harness: str,
    harness_version: str,
    harness_manifest: Mapping[str, Any],
    task_id: str,
    local_trial: int,
    pairing_slot: int,
    limits: TerminalLimits,
) -> dict[str, Any]:
    last: dict[str, Any] | None = None
    for attempt in range(1 + max(0, int(limits.infrastructure_retries))):
        attempt_root = (
            trajectory_root
            / _safe(task_id)
            / f"trial_{local_trial:04d}_attempt_{attempt:02d}"
        )
        last = run_terminal_trial(
            repo_root=repo_root,
            output_root=attempt_root,
            harness=harness,
            harness_version=harness_version,
            harness_manifest=harness_manifest,
            task_id=task_id,
            trial=local_trial,
            pairing_slot=pairing_slot,
            limits=limits,
        )
        if not last.get("infrastructure_error"):
            break
    assert last is not None
    return last


def run_terminal_trial(
    *,
    repo_root: Path,
    output_root: Path,
    harness: str,
    harness_version: str,
    harness_manifest: Mapping[str, Any],
    task_id: str,
    trial: int,
    pairing_slot: int,
    limits: TerminalLimits,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    task_root = _task_root(repo_root, task_id)
    task = _load_task_definition(task_root)
    instruction = str(task["instruction"])
    prepared_instruction, system_prompt = _render_prompts(instruction, harness_manifest)
    image = str(task["image"] or _image_name(task_id))
    project = _project_name(task_id, trial)
    cid = project
    compose_file = task_root / "docker-compose.yaml"
    proxy_process: subprocess.Popen[str] | None = None
    started = time.time()
    try:
        runner_assets = _runner_assets(repo_root, harness)
        override = _write_compose_override(
            output_root=output_root,
            harness=harness,
            runner_assets=runner_assets,
        )
        if not compose_file.is_file():
            compose_file = _write_harbor_compose(
                output_root=output_root,
                cid=cid,
                image=image,
            )
        compose_env = _compose_env(output_root, cid, image)
        up = _run_compose(
            compose_file,
            project,
            override,
            ["up", "-d", "--no-build"],
            compose_env,
            600,
        )
        if up.returncode != 0:
            raise RuntimeError(f"docker compose up failed: {up.stderr[-1200:]}")
        clash_runtime = _start_container_clash(cid, output_root)
        _materialize_manifest_files(cid, output_root, harness, harness_manifest)
        if harness == "opencode":
            proxy_process, proxy_port = _start_chat_proxy(
                output_root, harness=harness
            )
            command = _prepare_opencode(
                cid,
                output_root,
                prepared_instruction,
                system_prompt,
                limits,
                harness_manifest,
                proxy_port,
            )
        elif harness == "codex":
            proxy_process, proxy_port = _start_codex_proxy(
                repo_root,
                output_root,
                tool_desc_patches=harness_manifest.get("tool_desc_patches") or {},
                disabled_tools=_disabled_codex_tools(task_id),
            )
            command = _prepare_codex(
                cid,
                output_root,
                prepared_instruction,
                system_prompt,
                proxy_port,
                harness_manifest,
            )
        elif harness == "pi":
            command = _prepare_pi(
                cid,
                output_root,
                prepared_instruction,
                system_prompt,
                limits,
                harness_manifest,
            )
        else:
            raise ValueError(f"unsupported terminal harness: {harness}")
        model_context = _terminal_runtime_load_report(
            output_root=output_root,
            harness=harness,
            manifest=harness_manifest,
        )
        with provider_trial_slot() as provider_capacity:
            agent = _run_agent_process(
                cid=cid,
                command=command,
                timeout_s=int(limits.agent_timeout_s),
                first_event_timeout_s=int(limits.first_event_timeout_s),
                max_steps=int(limits.max_steps),
            )
        post_agent_clash = (
            _container_clash_healthy(cid)
            if bool(clash_runtime.get("enabled"))
            else True
        )
        (output_root / "agent.stdout").write_text(agent.stdout, encoding="utf-8")
        (output_root / "agent.stderr").write_text(agent.stderr, encoding="utf-8")
        trace_name = (
            "codex-api-calls.jsonl"
            if harness == "codex"
            else f"{harness}-api-calls.jsonl"
        )
        provider_failure = harness_execution_error(agent.stdout, agent.stderr)
        if not provider_failure and harness != "pi":
            provider_failure = provider_trace_error(output_root / trace_name)
        if not post_agent_clash:
            provider_failure = "container_clash_exited_during_agent"
        infrastructure_error = bool(provider_failure) or _agent_infrastructure_error(agent)
        reward = 0.0
        verifier_completed = False
        verifier_timed_out = False
        if not infrastructure_error:
            reward, verifier_timed_out = _verify(
                cid, task_root, output_root, int(limits.verify_timeout_s)
            )
            verifier_completed = True
        post_verifier_clash = (
            _container_clash_healthy(cid)
            if bool(clash_runtime.get("enabled"))
            else True
        )
        if bool(clash_runtime.get("enabled")) and not post_verifier_clash:
            provider_failure = "container_clash_exited_during_verifier"
            infrastructure_error = True
            reward = 0.0
        clash_runtime = {
            **clash_runtime,
            "post_agent_verified": bool(post_agent_clash),
            "post_verifier_verified": bool(post_verifier_clash),
            "lifecycle_verified": bool(post_agent_clash and post_verifier_clash),
        }
        termination = "done"
        if agent.timed_out:
            termination = agent.timeout_kind or "agent_timeout"
        elif provider_failure:
            termination = provider_failure
        elif agent.returncode != 0:
            termination = f"agent_exit_{agent.returncode}"
        elif verifier_timed_out:
            termination = "verifier_timeout"
        row = {
            "task_id": task_id,
            "trial": int(trial),
            "pairing_slot": int(pairing_slot),
            "reward": float(reward),
            "status": "error" if infrastructure_error else "completed",
            "termination": termination,
            "error": termination if infrastructure_error else "",
            "infrastructure_error": bool(infrastructure_error),
            "verifier_completed": verifier_completed,
            "verifier_timed_out": verifier_timed_out,
            "n_messages": 3 if agent.saw_event else 0,
            "n_tool_calls": int(agent.n_tool_calls),
            "messages": (
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prepared_instruction},
                    {
                        "role": "assistant",
                        "content": (agent.stdout + agent.stderr)[-16000:],
                    },
                ]
                if agent.saw_event
                else []
            ),
            "model_context": model_context,
            "api_calls_jsonl": (
                str((output_root / trace_name).resolve())
                if (output_root / trace_name).is_file()
                else ""
            ),
            "elapsed": round(time.time() - started, 1),
            "runner": harness,
            "harness_version": str(harness_version),
            "image": image,
            "container_clash": clash_runtime,
            "provider_capacity": provider_capacity,
            "runtime_schema": RUNTIME_SCHEMA,
        }
        return row
    except Exception as exc:  # noqa: BLE001
        return {
            "task_id": task_id,
            "trial": int(trial),
            "pairing_slot": int(pairing_slot),
            "reward": 0.0,
            "status": "error",
            "termination": f"{type(exc).__name__}: {exc}",
            "error": f"{type(exc).__name__}: {exc}",
            "infrastructure_error": True,
            "verifier_completed": False,
            "n_messages": 0,
            "n_tool_calls": 0,
            "elapsed": round(time.time() - started, 1),
            "runner": harness,
            "harness_version": str(harness_version),
            "image": image,
            "runtime_schema": RUNTIME_SCHEMA,
        }
    finally:
        _stop_process(proxy_process)
        try:
            task_root = _task_root(repo_root, task_id)
            override = output_root / "docker-compose.override.harness.yaml"
            _run_compose(
                compose_file,
                project,
                override,
                ["down", "--remove-orphans"],
                _compose_env(output_root, cid, image),
                600,
            )
        except Exception:
            pass
        if os.environ.get("HAI_KEEP_TRAJECTORY_WORKSPACE", "").lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            shutil.rmtree(output_root / "compose-logs", ignore_errors=True)
            _cleanup_staged_harness_workspace(output_root)


def _cleanup_staged_harness_workspace(output_root: Path) -> None:
    for name in ("candidate-workspace", "harness-files", "codex-home"):
        shutil.rmtree(output_root / name, ignore_errors=True)


def _terminal_runtime_load_report(
    *,
    output_root: Path,
    harness: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_root = output_root / "candidate-workspace"
    return build_runtime_load_report(
        harness=harness,
        project_root=candidate_root / "project",
        home_root=candidate_root / "home",
        manifest=manifest,
        tool_definitions=(),
    )


def normalize_harness(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized == "pi_agent":
        normalized = "pi"
    if normalized not in {"opencode", "codex", "pi"}:
        raise ValueError(f"unsupported Terminal-Bench harness: {value}")
    return normalized


def normalize_manifest(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "config_patch": dict(raw.get("config_patch") or {}),
        "files": [
            dict(item) for item in raw.get("files") or [] if isinstance(item, Mapping)
        ],
        "prompt_appends": [str(item) for item in raw.get("prompt_appends") or []],
        "instructions": [str(item) for item in raw.get("instructions") or []],
        "tool_desc_patches": dict(raw.get("tool_desc_patches") or {}),
        "removals": [str(item) for item in raw.get("removals") or []],
        "replace_channels": sorted(
            str(item) for item in raw.get("replace_channels") or []
        ),
        "_workspace": dict(raw.get("_workspace") or {}),
    }


def ensure_terminal_v0(*, evidence_root: str | Path, run_id: str, harness: str) -> str:
    root = _version_harness_root(evidence_root, run_id, "v0", harness)
    root.mkdir(parents=True, exist_ok=True)
    patch = root / "patch.json"
    descriptions = root / "patch_descs.json"
    if not patch.exists():
        patch.write_text(
            json.dumps(normalize_manifest({}), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if not descriptions.exists():
        descriptions.write_text("{}\n", encoding="utf-8")
    return "v0"


def materialize_terminal_version(
    *,
    evidence_root: str | Path,
    run_id: str,
    harness: str,
    base_version: str,
    label: str,
    delta: Mapping[str, Any],
) -> str:
    ensure_terminal_v0(evidence_root=evidence_root, run_id=run_id, harness=harness)
    version = _safe(label)
    base = read_terminal_version(
        evidence_root=evidence_root,
        run_id=run_id,
        harness=harness,
        version=base_version,
    )
    merged = _merge_manifest(base, normalize_manifest(delta))
    root = _version_harness_root(evidence_root, run_id, version, harness)
    root.mkdir(parents=True, exist_ok=True)
    descriptions = dict(merged.get("tool_desc_patches") or {})
    patch = dict(merged)
    patch.pop("tool_desc_patches", None)
    (root / "patch.json").write_text(
        json.dumps(patch, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "patch_descs.json").write_text(
        json.dumps(descriptions, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    version_root = root.parent.parent
    lineage = version_root / "lineage.json"
    lineage.write_text(
        json.dumps(
            {
                "version": version,
                "parent": str(base_version),
                "harness": normalize_harness(harness),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    meta = version_root / "meta.json"
    if not meta.exists():
        meta.write_text(
            json.dumps(
                {
                    "version": version,
                    "parent": str(base_version),
                    "harness": normalize_harness(harness),
                    "status": "temporary",
                    "temporary_candidate": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return version


def read_terminal_version(
    *, evidence_root: str | Path, run_id: str, harness: str, version: str
) -> dict[str, Any]:
    root = _version_harness_root(evidence_root, run_id, version, harness)
    try:
        patch = json.loads((root / "patch.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        patch = {}
    try:
        descriptions = json.loads(
            (root / "patch_descs.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        descriptions = {}
    payload = normalize_manifest(patch if isinstance(patch, Mapping) else {})
    payload["tool_desc_patches"] = (
        dict(descriptions) if isinstance(descriptions, Mapping) else {}
    )
    return payload


def _version_harness_root(
    evidence_root: str | Path, run_id: str, version: str, harness: str
) -> Path:
    return (
        Path(evidence_root).resolve()
        / str(run_id)
        / "versions_percell"
        / "terminal_bench"
        / str(version)
        / "harness"
        / normalize_harness(harness)
    )


def _merge_manifest(
    base: Mapping[str, Any], delta: Mapping[str, Any]
) -> dict[str, Any]:
    replace = {str(item) for item in delta.get("replace_channels") or []}
    result = normalize_manifest(base)
    for channel in ("config_patch", "tool_desc_patches"):
        incoming = dict(delta.get(channel) or {})
        result[channel] = (
            incoming if channel in replace else {**result[channel], **incoming}
        )
    for channel in ("prompt_appends", "instructions"):
        incoming = [str(item) for item in delta.get(channel) or []]
        result[channel] = (
            incoming if channel in replace else [*result[channel], *incoming]
        )
    files = (
        {}
        if "files" in replace
        else {
            str(item.get("path") or ""): dict(item)
            for item in result["files"]
            if str(item.get("path") or "")
        }
    )
    for item in delta.get("files") or []:
        if isinstance(item, Mapping) and str(item.get("path") or ""):
            files[str(item["path"])] = dict(item)
    removals = {str(item) for item in delta.get("removals") or []}
    for path in removals:
        files.pop(path, None)
    result["files"] = [files[path] for path in sorted(files)]
    result["removals"] = sorted({*result["removals"], *removals})
    result["replace_channels"] = sorted(replace)
    return result


def _cache_manifest(
    *,
    repo_root: Path,
    scope: str,
    harness: str,
    harness_manifest: Mapping[str, Any],
    task_id: str,
    pairing_slot: int,
    limits: TerminalLimits,
) -> dict[str, Any]:
    task_root = _task_root(repo_root, task_id)
    assets = _runner_assets(repo_root, harness)
    return {
        "schema": RUNTIME_SCHEMA,
        "scope": str(scope).upper(),
        "benchmark": "terminal_bench",
        "harness": harness,
        "model": DEFAULT_MODEL,
        "runtime_source": _runtime_source_fingerprint(),
        "runner_assets": assets["fingerprint"],
        "task_id": task_id,
        "task_source_sha256": hash_path(task_root),
        "image": _image_name(task_id),
        "harness_manifest": normalize_manifest(harness_manifest),
        "limits": limits.to_dict(),
        "pairing_slot": int(pairing_slot),
        "network_policy": _terminal_network_policy(),
    }


def _container_clash_required() -> bool:
    return os.environ.get("TB_ENABLE_CONTAINER_CLASH", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _model_endpoint_no_proxy() -> bool:
    return os.environ.get("TB_MODEL_ENDPOINT_NO_PROXY", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _terminal_network_policy() -> dict[str, Any]:
    source = Path(
        os.environ.get("TB_CLASHCTL_HOME") or DEFAULT_CLASHCTL_HOME
    ).expanduser().resolve()
    dns_url = str(os.environ.get("TB_CLASH_DNS_URL") or "")
    model_endpoint = str(
        os.environ.get("DEEPSEEK_BASE_URL")
        or os.environ.get("DEEPSEEK_URL")
        or "https://api.deepseek.com/v1"
    )
    return {
        "container_clash_required": _container_clash_required(),
        "model_endpoint_no_proxy": _model_endpoint_no_proxy(),
        "model_endpoint_sha256": hashlib.sha256(
            model_endpoint.encode("utf-8")
        ).hexdigest(),
        "clash_runtime_sha256": _sha256(source / "resources" / "runtime.yaml"),
        "clash_dns_url_sha256": hashlib.sha256(dns_url.encode("utf-8")).hexdigest(),
    }


def _trial_clash_compatible(row: Mapping[str, Any]) -> bool:
    if not _container_clash_required():
        return True
    runtime = row.get("container_clash") or {}
    endpoint_no_proxy = runtime.get(
        "model_endpoint_no_proxy", runtime.get("deepseek_no_proxy")
    )
    return bool(
        isinstance(runtime, Mapping)
        and runtime.get("enabled") is True
        and endpoint_no_proxy is _model_endpoint_no_proxy()
        and runtime.get("lifecycle_verified") is True
    )


def _render_prompts(instruction: str, manifest: Mapping[str, Any]) -> tuple[str, str]:
    system = (
        "You are an autonomous terminal agent inside a disposable Docker container. "
        "Complete the task using shell and file tools. Work only in the observed environment, "
        "verify your work, and finish without explaining the benchmark."
    )
    appends = [
        str(item).strip()
        for item in manifest.get("prompt_appends") or []
        if str(item).strip()
    ]
    if appends:
        system = system + "\n\n" + "\n\n".join(appends)
    tool_patches = manifest.get("tool_desc_patches") or {}
    if tool_patches:
        system += "\n\nTool guidance:\n" + json.dumps(
            tool_patches, ensure_ascii=False, sort_keys=True
        )
    files = [
        {
            "path": str(item.get("path") or ""),
            "content": str(item.get("content") or ""),
        }
        for item in manifest.get("files") or []
        if str(item.get("path") or "") and str(item.get("content") or "").strip()
    ]
    if files:
        system += "\n\nActive harness files:\n" + json.dumps(
            files, ensure_ascii=False, sort_keys=True
        )
    instructions = [
        str(item).strip()
        for item in manifest.get("instructions") or []
        if str(item).strip()
    ]
    if instructions:
        system += "\n\n" + "\n\n".join(instructions)
    return instruction, system


def _task_root(repo_root: Path, task_id: str) -> Path:
    root = repo_root / "third_party" / "terminal-bench" / "original-tasks" / task_id
    legacy = (root / "task.yaml").is_file() and (
        root / "docker-compose.yaml"
    ).is_file()
    harbor = (root / "task.toml").is_file() and (
        root / "instruction.md"
    ).is_file()
    if not legacy and not harbor:
        raise FileNotFoundError(f"Terminal-Bench task is unavailable: {task_id}")
    return root


def _load_task_definition(task_root: Path) -> dict[str, str]:
    legacy = task_root / "task.yaml"
    if legacy.is_file():
        task = yaml.safe_load(legacy.read_text(encoding="utf-8")) or {}
        return {"instruction": str(task.get("instruction") or ""), "image": ""}
    config = tomllib.loads((task_root / "task.toml").read_text(encoding="utf-8"))
    environment = config.get("environment") or {}
    return {
        "instruction": (task_root / "instruction.md").read_text(encoding="utf-8"),
        "image": str(environment.get("docker_image") or ""),
    }


def _image_name(task_id: str) -> str:
    template = os.environ.get("TB_IMAGE_TEMPLATE", DEFAULT_IMAGE_TEMPLATE)
    return template.replace("{task_id}", task_id).replace(
        "{safe_task_id}", _safe(task_id)
    )


def _project_name(task_id: str, trial: int) -> str:
    safe_task = re.sub(r"[^a-z0-9_-]+", "-", _safe(task_id).lower()).strip("-_")
    return f"tb_{safe_task or 'task'}_t{trial}_{uuid.uuid4().hex[:8]}"[:63]


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-.") or "item"


def _docker() -> str:
    return os.environ.get("TB_DOCKER") or shutil.which("docker") or "docker"


def _terminal_preflight(task_ids: Sequence[str]) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    invalid_tasks = []
    for task_id in task_ids:
        try:
            task_root = _task_root(repo_root, task_id)
            _verifier_root(task_root)
        except FileNotFoundError as exc:
            invalid_tasks.append(str(exc))
    if invalid_tasks:
        raise RuntimeError(
            "Terminal-Bench task assets are incomplete: " + "; ".join(invalid_tasks)
        )
    info = subprocess.run(
        [_docker(), "info"], capture_output=True, timeout=30, check=False
    )
    if info.returncode != 0:
        raise RuntimeError(
            f"Docker is unavailable: {(info.stderr or b'').decode('utf-8', 'replace')[-1000:]}"
        )
    network = os.environ.get("TB_SHARED_NETWORK", "harnesslens-terminal-bench")
    inspected = subprocess.run(
        [_docker(), "network", "inspect", network],
        capture_output=True,
        timeout=30,
        check=False,
    )
    if inspected.returncode != 0:
        created = subprocess.run(
            [_docker(), "network", "create", network],
            capture_output=True,
            timeout=30,
            check=False,
        )
        if created.returncode != 0:
            raise RuntimeError(
                f"failed to create Terminal-Bench network: {(created.stderr or b'').decode('utf-8', 'replace')[-1000:]}"
            )
    missing = []
    for task_id in task_ids:
        task_root = _task_root(repo_root, task_id)
        image = str(_load_task_definition(task_root)["image"] or _image_name(task_id))
        result = subprocess.run(
            [_docker(), "image", "inspect", image],
            capture_output=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            missing.append(image)
    if missing:
        raise RuntimeError(
            "Terminal-Bench images are not preloaded: " + ", ".join(missing)
        )


def _container_proxy_env() -> dict[str, str]:
    result: dict[str, str] = {}
    container_clash = os.environ.get("TB_ENABLE_CONTAINER_CLASH", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    generic = (
        "http://127.0.0.1:16627"
        if container_clash
        else os.environ.get("TB_CONTAINER_PROXY_URL")
    )
    for lower, explicit in (
        (
            "http_proxy",
            generic
            if container_clash
            else os.environ.get("TB_CONTAINER_HTTP_PROXY")
            or generic
            or os.environ.get("http_proxy")
            or os.environ.get("HTTP_PROXY"),
        ),
        (
            "https_proxy",
            generic
            if container_clash
            else os.environ.get("TB_CONTAINER_HTTPS_PROXY")
            or generic
            or os.environ.get("https_proxy")
            or os.environ.get("HTTPS_PROXY"),
        ),
    ):
        if explicit:
            result[lower] = (
                explicit
                if container_clash
                else explicit.replace(
                    "127.0.0.1", "host.docker.internal"
                ).replace("localhost", "host.docker.internal")
            )
            result[lower.upper()] = result[lower]
    bypass = [
        item
        for item in str(
            os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
        ).split(",")
        if item
    ]
    model_endpoint = urlsplit(
        os.environ.get("DEEPSEEK_BASE_URL")
        or os.environ.get("DEEPSEEK_URL")
        or "https://api.deepseek.com/v1"
    ).hostname
    bypass_targets = [
        "127.0.0.1",
        "localhost",
        "::1",
        "host.docker.internal",
        "api.deepseek.com",
        ".deepseek.com",
        _container_host_proxy_address(),
    ]
    if _model_endpoint_no_proxy():
        bypass_targets.append(model_endpoint)
    for item in bypass_targets:
        if item and item not in bypass:
            bypass.append(item)
    result["no_proxy"] = result["NO_PROXY"] = ",".join(bypass)
    return result


def _start_container_clash(cid: str, output_root: Path) -> dict[str, Any]:
    if os.environ.get("TB_ENABLE_CONTAINER_CLASH", "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return {"enabled": False}
    with _CLASH_START_SEMAPHORE:
        return _start_container_clash_limited(cid, output_root)


def _start_container_clash_limited(
    cid: str, output_root: Path
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    source = Path(
        os.environ.get("TB_CLASHCTL_HOME") or DEFAULT_CLASHCTL_HOME
    ).expanduser().resolve()
    required = (
        source / "bin" / "mihomo",
        source / "resources" / "runtime.yaml",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("container Clash assets missing: " + ", ".join(missing))

    root = "/tmp/harness-clashctl"
    setup = _dexec(
        cid,
        f"rm -rf {root} && mkdir -p {root}/bin {root}/resources/dist",
        timeout=60,
        workdir="/",
    )
    if setup.returncode != 0:
        raise RuntimeError(
            "failed to prepare container Clash directory: "
            + (setup.stderr or b"").decode("utf-8", "replace")[-1000:]
        )
    _dcp_in(source / "bin" / "mihomo", cid, f"{root}/bin/mihomo")
    runtime = _prepare_container_clash_runtime(
        source / "resources" / "runtime.yaml", output_root
    )
    _dcp_in(runtime, cid, f"{root}/resources/runtime.yaml")
    for name in ("config.yaml", "mixin.yaml", "Country.mmdb", "geosite.dat", "cache.db"):
        path = source / "resources" / name
        if path.is_file():
            _dcp_in(path, cid, f"{root}/resources/{name}")

    runtime_log = "/agent-logs/container-mihomo.log"
    command = (
        f"exec {root}/bin/mihomo -d {root}/resources "
        f"-f {root}/resources/runtime.yaml >>{runtime_log} 2>&1"
    )
    launched = _dexec_detached(cid, command, workdir="/")
    if launched.returncode != 0:
        raise RuntimeError(
            "failed to launch container mihomo: "
            + (launched.stderr or b"").decode("utf-8", "replace")[-1000:]
        )
    readiness = (
        "proxy_ready() { local line=''; "
        "exec 3<>/dev/tcp/127.0.0.1/16627 || return 1; "
        "printf 'GET http://archive.ubuntu.com/ubuntu/ HTTP/1.1\\r\\n"
        "Host: archive.ubuntu.com\\r\\nConnection: close\\r\\n\\r\\n' >&3; "
        "IFS= read -r -t 10 line <&3 || true; exec 3<&-; exec 3>&-; "
        "case \"$line\" in *' 200 '*|*' 301 '*|*' 302 '*) return 0;; "
        "*) return 1;; esac; }; "
        "ready=0; for _ in $(seq 1 30); do "
        "if proxy_ready; then "
        "ready=1; break; fi; sleep 2; done; [ \"$ready\" = 1 ]"
    )
    started = _dexec(cid, readiness, timeout=90, workdir="/")
    log_path = output_root / "container-clashctl-start.log"
    log_path.write_text(
        (started.stdout or b"").decode("utf-8", "replace")
        + (started.stderr or b"").decode("utf-8", "replace"),
        encoding="utf-8",
    )
    if started.returncode != 0:
        raise RuntimeError(
            "clashctl on failed inside Terminal-Bench container: "
            + log_path.read_text(encoding="utf-8", errors="replace")[-1200:]
        )
    return {
        "enabled": True,
        "scope": "terminal_bench_task_container",
        "http_proxy": "http://127.0.0.1:16627",
        "deepseek_no_proxy": _model_endpoint_no_proxy(),
        "model_endpoint_no_proxy": _model_endpoint_no_proxy(),
        "startup_verified": True,
        "start_log": str(log_path),
        "runtime_log": str(output_root / "agent-logs" / "container-mihomo.log"),
    }


def _container_clash_healthy(cid: str) -> bool:
    result = _dexec(
        cid,
        "exec 3<>/dev/tcp/127.0.0.1/16627 && "
        "printf 'GET http://archive.ubuntu.com/ubuntu/ HTTP/1.1\\r\\n"
        "Host: archive.ubuntu.com\\r\\nConnection: close\\r\\n\\r\\n' >&3 && "
        "IFS= read -r -t 10 line <&3; case \"$line\" in "
        "*' 200 '*|*' 301 '*|*' 302 '*) exit 0;; *) exit 1;; esac",
        timeout=15,
        workdir="/",
    )
    return result.returncode == 0


def _prepare_container_clash_runtime(source: Path, output_root: Path) -> Path:
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise RuntimeError(f"container Clash runtime is not a mapping: {source}")
    runtime = dict(raw)
    runtime.pop("external-ui", None)
    runtime.pop("external-ui-url", None)
    dns = runtime.get("dns")
    if isinstance(dns, Mapping):
        patched_dns = dict(dns)
        for key in ("default-nameserver", "proxy-server-nameserver", "nameserver"):
            values = patched_dns.get(key)
            if isinstance(values, list):
                patched_dns[key] = [
                    f"tcp://{value}"
                    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", str(value))
                    else value
                    for value in values
                ]
        runtime["dns"] = patched_dns
    path = output_root / "container-clash-runtime.yaml"
    path.write_text(
        yaml.safe_dump(runtime, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _container_package_env() -> dict[str, str]:
    result = {
        key: value
        for key in (
            "PIP_INDEX_URL",
            "PIP_TRUSTED_HOST",
            "UV_DEFAULT_INDEX",
            "UV_INDEX_URL",
        )
        if (value := os.environ.get(key))
    }
    result.setdefault("UV_HTTP_RETRIES", "10")
    result.setdefault("UV_HTTP_TIMEOUT", "120")
    result.setdefault("UV_CONCURRENT_DOWNLOADS", "2")
    return result


def _write_compose_override(
    *, output_root: Path, harness: str, runner_assets: Mapping[str, Any]
) -> Path:
    environment = {
        **_container_proxy_env(),
        **_container_package_env(),
        "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY", ""),
        "HOME": "/tmp/harness-home",
        "XDG_CONFIG_HOME": "/tmp/harness-home/.config",
        "XDG_DATA_HOME": "/tmp/harness-home/.local/share",
        "XDG_CACHE_HOME": "/tmp/harness-home/.cache",
        "XDG_STATE_HOME": "/tmp/harness-home/.local/state",
    }
    override: dict[str, Any] = {
        "services": {
            "client": {
                "extra_hosts": ["host.docker.internal:host-gateway"],
                "environment": environment,
                # so a killed run's containers can be identified and reclaimed
                "labels": dict(ownership_labels()),
                "volumes": list(runner_assets.get("volumes") or []),
            }
        },
        "networks": {
            "default": {
                "external": True,
                "name": os.environ.get("TB_SHARED_NETWORK", "harnesslens-terminal-bench"),
            }
        },
    }
    path = output_root / "docker-compose.override.harness.yaml"
    path.write_text(yaml.safe_dump(override, sort_keys=True), encoding="utf-8")
    return path


def _write_harbor_compose(*, output_root: Path, cid: str, image: str) -> Path:
    compose = {
        "services": {
            "client": {
                "image": image,
                "container_name": cid,
                "command": ["sh", "-c", "sleep infinity"],
                "working_dir": "/app",
                "environment": {"TEST_DIR": "/tests"},
                "volumes": [
                    "${T_BENCH_TASK_LOGS_PATH}:/logs",
                    "${T_BENCH_TASK_AGENT_LOGS_PATH}:/agent-logs",
                ],
            }
        }
    }
    path = output_root / "docker-compose.harbor-task.yaml"
    path.write_text(yaml.safe_dump(compose, sort_keys=True), encoding="utf-8")
    return path


def _compose_env(output_root: Path, cid: str, image: str) -> dict[str, str]:
    logs = output_root / "compose-logs"
    agent_logs = output_root / "agent-logs"
    logs.mkdir(parents=True, exist_ok=True)
    agent_logs.mkdir(parents=True, exist_ok=True)
    return {
        **os.environ,
        "T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME": cid,
        "T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME": image,
        "T_BENCH_TASK_DOCKER_NAME_PREFIX": cid,
        "T_BENCH_CONTAINER_LOGS_PATH": "/logs",
        "T_BENCH_CONTAINER_AGENT_LOGS_PATH": "/agent-logs",
        "T_BENCH_TEST_DIR": "/tests",
        "T_BENCH_TASK_LOGS_PATH": str(logs.resolve()),
        "T_BENCH_TASK_AGENT_LOGS_PATH": str(agent_logs.resolve()),
    }


def _run_compose(
    compose_file: Path,
    project: str,
    override: Path,
    command: Sequence[str],
    env: Mapping[str, str],
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            _docker(),
            "compose",
            "-p",
            project,
            "-f",
            str(compose_file),
            "-f",
            str(override),
            *command,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=dict(env),
        check=False,
    )


def _dexec(
    cid: str, command: str, *, timeout: int, workdir: str = "/app"
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            [_docker(), "exec", "-w", workdir, cid, "bash", "-lc", command],
            capture_output=True,
            timeout=timeout,
            env=os.environ,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            args=exc.cmd,
            returncode=124,
            stdout=exc.stdout or b"",
            stderr=(exc.stderr or b"")
            + f"\ncommand timed out after {timeout}s".encode(),
        )


def _dexec_detached(
    cid: str, command: str, *, workdir: str = "/app"
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [_docker(), "exec", "-d", "-w", workdir, cid, "bash", "-lc", command],
        capture_output=True,
        timeout=60,
        env=os.environ,
        check=False,
    )


def _dcp_in(source: Path, cid: str, destination: str) -> None:
    result = subprocess.run(
        [_docker(), "cp", str(source), f"{cid}:{destination}"],
        capture_output=True,
        timeout=300,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"docker cp failed: {(result.stderr or b'').decode('utf-8', 'replace')[-1000:]}"
        )


def _dcp_contents(source: Path, cid: str, destination: str) -> None:
    result = subprocess.run(
        [_docker(), "cp", f"{source}/.", f"{cid}:{destination}"],
        capture_output=True,
        timeout=300,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"docker cp failed: {(result.stderr or b'').decode('utf-8', 'replace')[-1000:]}"
        )


@functools.lru_cache(maxsize=None)
def _runner_assets(repo_root: Path, harness: str) -> dict[str, Any]:
    if harness == "opencode":
        candidates = [
            *_configured_paths("TB_OPENCODE_BIN"),
            Path.home()
            / ".opencode/lib/node_modules/opencode-ai/node_modules/opencode-linux-x64-baseline/bin/opencode",
            Path.home()
            / ".opencode/lib/node_modules/opencode-ai/node_modules/opencode-linux-x64/bin/opencode",
        ]
        binary = _first_file(candidates, "OpenCode binary")
        return {
            "volumes": [f"{binary}:/opt/harness/bin/opencode:ro"],
            "fingerprint": {
                "version": _host_version([str(binary), "--version"]),
                "sha256": _sha256(binary),
            },
        }
    if harness == "codex":
        candidates = _configured_paths("TB_CODEX_BIN")
        candidates.extend(
            Path.home().glob(
                ".nvm/versions/node/*/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex"
            )
        )
        binary = _first_file(candidates, "Codex native binary")
        return {
            "volumes": [f"{binary}:/opt/harness/bin/codex:ro"],
            "fingerprint": {
                "version": _host_version([str(binary), "--version"]),
                "sha256": _sha256(binary),
            },
        }
    node_candidates = _configured_paths("TB_PI_NODE_BIN")
    if shutil.which("node"):
        node_candidates.append(Path(str(shutil.which("node"))))
    node = _first_file(node_candidates, "Node binary")
    pi_modules = _first_dir(
        [
            *_configured_paths("TB_PI_NODE_MODULES"),
            repo_root / ".pi-agent/node_modules",
            repo_root / "third_party/pi-agent/node_modules",
        ],
        "Pi node_modules",
    )
    libatomic = _first_file(
        [
            *_configured_paths("TB_PI_LIBATOMIC"),
            Path("/lib/x86_64-linux-gnu/libatomic.so.1"),
            Path("/usr/lib/x86_64-linux-gnu/libatomic.so.1"),
        ],
        "Pi libatomic.so.1",
    )
    package = pi_modules / "@earendil-works/pi-coding-agent/package.json"
    version = (
        json.loads(package.read_text(encoding="utf-8")).get("version")
        if package.is_file()
        else "unknown"
    )
    pi_runner = Path(__file__).with_name(PI_RUNNER_FILENAME)
    if not pi_runner.is_file():
        raise RuntimeError(f"Pi compact runner is unavailable: {pi_runner}")
    return {
        "volumes": [
            f"{node}:/opt/harness/bin/node:ro",
            f"{pi_modules}:/opt/harness/pi-node-modules:ro",
            f"{libatomic}:/opt/harness/lib/libatomic.so.1:ro",
            f"{pi_runner}:/opt/harness/{PI_RUNNER_FILENAME}:ro",
        ],
        "fingerprint": {
            "version": str(version),
            "node": _host_version([str(node), "--version"]),
            "package_sha256": _sha256(package),
            "libatomic_sha256": _sha256(libatomic),
            "compact_runner_sha256": _sha256(pi_runner),
        },
    }


def _configured_paths(name: str) -> list[Path]:
    value = str(os.environ.get(name) or "").strip()
    return [Path(value).expanduser()] if value else []


def _first_file(candidates: Sequence[Path], label: str) -> Path:
    for candidate in candidates:
        if str(candidate) and candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(f"{label} is unavailable")


def _first_dir(candidates: Sequence[Path], label: str) -> Path:
    for candidate in candidates:
        if str(candidate) and candidate.is_dir():
            return candidate.resolve()
    raise RuntimeError(f"{label} is unavailable")


def _sha256(path: Path) -> str:
    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _host_version(command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command), capture_output=True, text=True, timeout=30, check=False
    )
    return (result.stdout or result.stderr or "unknown").strip().splitlines()[0]


@functools.lru_cache(maxsize=1)
def _runtime_source_fingerprint() -> dict[str, str]:
    package_root = Path(__file__).resolve().parents[1]
    sources = {
        "terminal_bench.py": Path(__file__),
        "terminal_cache.py": Path(__file__).with_name("terminal_cache.py"),
        "codex_responses_proxy.py": (
            package_root / "infrastructure" / "codex_responses_proxy.py"
        ),
        PI_RUNNER_FILENAME: Path(__file__).with_name(PI_RUNNER_FILENAME),
    }
    return {name: _sha256(path) for name, path in sources.items()}


def _materialize_manifest_files(
    cid: str,
    output_root: Path,
    harness: str,
    manifest: Mapping[str, Any],
) -> None:
    root = output_root / "harness-files"
    for item in manifest.get("files") or []:
        raw = str(item.get("path") or "")
        path = Path(raw)
        if not raw or path.is_absolute() or ".." in path.parts:
            continue
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(item.get("content") or ""), encoding="utf-8")
        _dexec(
            cid,
            f"mkdir -p {shlex.quote(str(Path('/app') / path.parent))}",
            timeout=60,
            workdir="/",
        )
        _dcp_in(target, cid, str(Path("/app") / path))
    candidate_root = output_root / "candidate-workspace"
    home = candidate_root / "home"
    project = candidate_root / "project"
    materialize_workspace(
        candidate_workspace(manifest),
        home_root=home,
        project_root=project,
    )
    _dexec(cid, "mkdir -p /app", timeout=60, workdir="/")
    _dcp_contents(project, cid, "/app")
    home_destination = {
        "opencode": "/tmp/harness-home/.config/opencode",
        "pi": "/tmp/harness-home/.pi/agent",
        "codex": "/tmp/harness-home/.codex",
    }[normalize_harness(harness)]
    _dexec(
        cid,
        f"mkdir -p {shlex.quote(home_destination)}",
        timeout=60,
        workdir="/",
    )
    _dcp_contents(home, cid, home_destination)


def _prepare_opencode(
    cid: str,
    output_root: Path,
    instruction: str,
    system_prompt: str,
    limits: TerminalLimits,
    manifest: Mapping[str, Any],
    proxy_port: int | None = None,
) -> str:
    effective_instruction = str(instruction)
    if effective_instruction.lstrip().startswith("-"):
        effective_instruction = f"Task instructions:\n\n{effective_instruction}"
    candidate_root = output_root / "candidate-workspace"
    candidate = load_json_configs(
        (
            candidate_root / "home" / "config.json",
            candidate_root / "home" / "opencode.json",
            candidate_root / "project" / "opencode.json",
        )
    )
    candidate = merge_candidate_config(
        candidate,
        legacy_flat_patch=manifest.get("config_patch") or {},
        fixed={},
    )
    candidate = relocate_opencode_instruction_paths(
        candidate,
        project_root="/app",
    )
    candidate_prompt = str(
        ((candidate.get("agent") or {}).get("build") or {}).get("prompt") or ""
    )
    prompt = "\n\n".join(
        item.strip() for item in (system_prompt, candidate_prompt) if item.strip()
    )
    fixed = {
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
                    "baseURL": (
                        f"http://{_container_host_proxy_address()}:{proxy_port}/v1"
                        if proxy_port is not None
                        else os.environ.get(
                            "DEEPSEEK_BASE_URL"
                        )
                        or os.environ.get("DEEPSEEK_URL")
                        or "https://api.deepseek.com/v1"
                    ),
                    "apiKey": "{env:DEEPSEEK_API_KEY}",
                    "timeout": 600000,
                    "chunkTimeout": 60000,
                },
                "models": {
                    "deepseek-v4-flash": {
                        "name": "DeepSeek V4 Flash",
                        "limit": {
                            "context": DEFAULT_OPENCODE_CONTEXT_LIMIT,
                            "output": DEFAULT_OUTPUT_LIMIT,
                        },
                    }
                },
            }
        },
        "agent": {"build": {"steps": int(limits.max_steps), "prompt": prompt}},
        "permission": {
            "bash": "allow",
            "read": "allow",
            "write": "allow",
            "edit": "allow",
            "glob": "allow",
            "grep": "allow",
            "list": "allow",
        },
    }
    config = merge_candidate_config(candidate, fixed=fixed)
    if manifest.get("instructions"):
        instruction_path = output_root / "opencode-candidate-instructions.md"
        instruction_path.write_text(
            "\n\n".join(str(item) for item in manifest["instructions"]),
            encoding="utf-8",
        )
        _dexec(cid, "mkdir -p /app/.opencode", timeout=60, workdir="/")
        _dcp_in(instruction_path, cid, "/app/.opencode/candidate-instructions.md")
        config["instructions"] = ["/app/.opencode/candidate-instructions.md"]
    path = output_root / "opencode.json"
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    _dexec(
        cid,
        "mkdir -p /tmp/harness-home /opt/harness/bin && chmod +x /opt/harness/bin/opencode",
        timeout=60,
        workdir="/",
    )
    _dcp_in(path, cid, "/app/opencode.json")
    return (
        "cd /app && export PATH=/opt/harness/bin:$PATH && export OPENCODE_CONFIG=/app/opencode.json && "
        "export OPENCODE_DISABLE_AUTOCOMPACT=1 OPENCODE_DISABLE_PRUNE=1 && "
        f"opencode run -m deepseek/deepseek-v4-flash --pure --auto --format json {shlex.quote(effective_instruction)}"
    )


def _start_chat_proxy(
    output_root: Path, *, harness: str
) -> tuple[subprocess.Popen[str], int]:
    command = [
        sys.executable,
        str(
            Path(__file__).resolve().parents[1]
            / "infrastructure"
            / "chat_completions_proxy.py"
        ),
        "--host",
        "0.0.0.0",
        "--log-file",
        str(output_root / f"{harness}-api-calls.jsonl"),
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
        _stop_process(process)
        raise RuntimeError(f"{harness} Chat Completions proxy failed: {line} {stderr[-1000:]}")
    return process, int(line.split("=", 1)[1])


def _sanitized_opencode_config_patch(manifest: Mapping[str, Any]) -> dict[str, Any]:
    raw = manifest.get("config_patch") or {}
    if not isinstance(raw, Mapping):
        return {}
    patch: dict[str, Any] = {}
    for key in ("tools", "permission"):
        value = raw.get(key)
        if isinstance(value, Mapping):
            patch[key] = dict(value)
    for raw_key, value in raw.items():
        key = str(raw_key)
        if "." not in key:
            continue
        head, tail = key.split(".", 1)
        if head not in {"tools", "permission"} or not tail:
            continue
        patch.setdefault(head, {})[tail] = value
    return patch


def _deep_merge(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def _start_codex_proxy(
    repo_root: Path,
    output_root: Path,
    *,
    tool_desc_patches: Mapping[str, Any] | None = None,
    disabled_tools: Sequence[str] = (),
) -> tuple[subprocess.Popen[str], int]:
    command = [
        sys.executable,
        str(
            Path(__file__).resolve().parents[1]
            / "infrastructure"
            / "codex_responses_proxy.py"
        ),
        "--host",
        "0.0.0.0",
        "--log-file",
        str(output_root / "codex-proxy-usage.jsonl"),
        "--context-log",
        str(output_root / "codex-api-calls.jsonl"),
    ]
    for name in disabled_tools:
        command.extend(["--disable-tool", str(name)])
    if tool_desc_patches:
        patch_path = output_root / "codex-tool-desc-patches.json"
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
        env=os.environ,
        start_new_session=True,
    )
    assert process.stdout is not None
    line = process.stdout.readline().strip()
    if not line.startswith("PORT="):
        stderr = process.stderr.read() if process.stderr else ""
        _stop_process(process)
        raise RuntimeError(f"Codex Responses proxy failed: {line} {stderr[-1000:]}")
    return process, int(line.split("=", 1)[1])


def _disabled_codex_tools(task_id: str) -> tuple[str, ...]:
    return CODEX_DISABLED_TOOLS_BY_TASK.get(str(task_id), ())


def _prepare_codex(
    cid: str,
    output_root: Path,
    instruction: str,
    system_prompt: str,
    proxy_port: int,
    manifest: Mapping[str, Any],
) -> str:
    home = output_root / "codex-home"
    home.mkdir(parents=True, exist_ok=True)
    proxy_host = _container_host_proxy_address()
    config_patch = dict(manifest.get("config_patch") or {})
    candidate_root = output_root / "candidate-workspace"
    candidate_config = load_toml_configs(
        (
            candidate_root / "home" / "config.toml",
            candidate_root / "project" / ".codex" / "config.toml",
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
            "sandbox_mode": "danger-full-access",
            "approval_policy": "never",
            "web_search": "disabled",
            "developer_instructions": developer_instructions,
            "model_providers": {
                "deepseek": {
                    "name": "DeepSeek via host Responses proxy",
                    "base_url": f"http://{proxy_host}:{proxy_port}/v1",
                    "env_key": "OPENAI_API_KEY",
                    "wire_api": "responses",
                }
            },
        },
    )
    (home / "config.toml").write_text(render_toml(config), encoding="utf-8")
    (home / "models_cache.json").write_text(
        json.dumps(
            {
                "fetched_at": "2026-07-20T00:00:00Z",
                "etag": "harnesslens-terminal",
                "client_version": "harnesslens",
                "models": [
                    {
                        "slug": "gpt-5.4",
                        "display_name": "gpt-5.4",
                        "description": "DeepSeek v4 flash",
                        "default_reasoning_level": "high",
                        "supported_reasoning_levels": [
                            {"effort": "high", "description": "High"}
                        ],
                        "supported_in_api": True,
                        "supports_parallel_tool_calls": True,
                        "context_window": 65536,
                        "effective_context_window_percent": 95,
                        "input_modalities": ["text"],
                        "supports_search_tool": False,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    candidate_hooks = (
        output_root / "candidate-workspace" / "project" / ".codex" / "hooks.json"
    )
    setup = "mkdir -p /app /tmp/harness-home/.codex /opt/harness/bin && chmod +x /opt/harness/bin/codex"
    if candidate_hooks.is_file():
        setup += " && git init --quiet /app"
    _dexec(cid, setup, timeout=60, workdir="/")
    _dcp_contents(home, cid, "/tmp/harness-home/.codex")
    hook_flag = "--dangerously-bypass-hook-trust " if candidate_hooks.is_file() else ""
    return (
        "cd /app && export PATH=/opt/harness/bin:$PATH CODEX_HOME=/tmp/harness-home/.codex "
        "HOME=/tmp/harness-home OPENAI_API_KEY=harnesslens-local-proxy && unset DEEPSEEK_API_KEY && "
        "codex exec --skip-git-repo-check --ephemeral --json "
        f"{hook_flag}--dangerously-bypass-approvals-and-sandbox {shlex.quote(instruction)}"
    )


def _prepare_pi(
    cid: str,
    output_root: Path,
    instruction: str,
    system_prompt: str,
    limits: TerminalLimits,
    manifest: Mapping[str, Any],
) -> str:
    settings = output_root / "pi-settings.json"
    models = output_root / "pi-models.json"
    prompt = output_root / "pi-prompt.txt"
    system = output_root / "pi-system-prompt.txt"
    candidate_root = output_root / "candidate-workspace"
    candidate_config = load_json_configs(
        (
            candidate_root / "home" / "settings.json",
            candidate_root / "home" / ".pi" / "settings.json",
            candidate_root / "project" / ".pi" / "settings.json",
        )
    )
    settings.write_text(
        json.dumps(
            merge_candidate_config(
                candidate_config,
                legacy_flat_patch=manifest.get("config_patch") or {},
                fixed={
                    "defaultProvider": "deepseek",
                    "defaultModel": "deepseek-v4-flash",
                    "model": "deepseek-v4-flash",
                    "quietStartup": True,
                    "enableInstallTelemetry": False,
                },
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    base_url = str(
        os.environ.get("DEEPSEEK_BASE_URL")
        or os.environ.get("DEEPSEEK_URL")
        or "https://api.deepseek.com/v1"
    ).rstrip("/")
    if base_url.endswith("/chat/completions"):
        base_url = base_url[: -len("/chat/completions")]
    models.write_text(
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
                                "id": DEFAULT_MODEL,
                                "reasoning": True,
                                "contextWindow": 65_536,
                                "maxTokens": 24_576,
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
    prompt.write_text(instruction, encoding="utf-8")
    system.write_text(system_prompt, encoding="utf-8")
    _dexec(
        cid,
        "mkdir -p /app /tmp/harness-home/.pi/agent /opt/harness/bin && chmod +x /opt/harness/bin/node",
        timeout=60,
        workdir="/",
    )
    _dcp_in(settings, cid, "/tmp/harness-home/.pi/agent/settings.json")
    _dcp_in(models, cid, "/tmp/harness-home/.pi/agent/models.json")
    _dcp_in(prompt, cid, "/tmp/harness-home/pi-prompt.txt")
    _dcp_in(system, cid, "/tmp/harness-home/pi-system-prompt.txt")
    return (
        "cd /app && export HOME=/tmp/harness-home PI_CODING_AGENT_DIR=/tmp/harness-home/.pi/agent "
        "LD_LIBRARY_PATH=/opt/harness/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} "
        "PI_TELEMETRY=0 PI_OFFLINE=1 && "
        f"/opt/harness/bin/node /opt/harness/{PI_RUNNER_FILENAME} "
        "--prompt-file /tmp/harness-home/pi-prompt.txt "
        "--system-prompt-file /tmp/harness-home/pi-system-prompt.txt "
        f"--max-steps {int(limits.max_steps)}"
    )


def _run_agent_process(
    *,
    cid: str,
    command: str,
    timeout_s: int,
    first_event_timeout_s: int,
    max_steps: int,
) -> AgentProcessResult:
    process = subprocess.Popen(
        [_docker(), "exec", "-w", "/app", cid, "bash", "-lc", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=os.environ,
    )
    output_queue: queue.Queue[tuple[str, str | None]] = queue.Queue(maxsize=256)
    _start_reader(process.stdout, "stdout", output_queue)
    _start_reader(process.stderr, "stderr", output_queue)
    stdout: list[str] = []
    stderr: list[str] = []
    output_sizes = {"stdout": 0, "stderr": 0}
    tool_ids: set[str] = set()
    anonymous_tool_calls = 0
    saw_event = False
    open_streams = 2
    started = time.monotonic()
    timed_out = False
    timeout_kind = ""
    drain_deadline: float | None = None
    while True:
        try:
            source, line = output_queue.get(timeout=0.25)
            if line is None:
                open_streams = max(0, open_streams - 1)
            else:
                target = stdout if source == "stdout" else stderr
                output_sizes[source] = _append_bounded_output(
                    target,
                    line,
                    output_sizes[source],
                    AGENT_OUTPUT_LIMIT_CHARS,
                )
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    event = None
                if isinstance(event, Mapping):
                    saw_event = True
                    is_tool, call_id = _tool_event_identity(event)
                    if is_tool:
                        if call_id:
                            tool_ids.add(call_id)
                        else:
                            anonymous_tool_calls += 1
                    if str(event.get("type") or "") == "harness_limit":
                        timed_out = True
                        timeout_kind = str(event.get("kind") or "max_steps")
        except queue.Empty:
            pass
        elapsed = time.monotonic() - started
        tool_count = len(tool_ids) + anonymous_tool_calls
        if tool_count >= max(1, int(max_steps)) and process.poll() is None:
            timed_out, timeout_kind = True, "max_steps"
            _terminate_agent(cid, process)
            drain_deadline = time.monotonic() + AGENT_EXIT_DRAIN_GRACE_S
        elif (
            not saw_event
            and elapsed >= first_event_timeout_s
            and process.poll() is None
        ):
            timed_out, timeout_kind = True, "first_event_timeout"
            _terminate_agent(cid, process)
            drain_deadline = time.monotonic() + AGENT_EXIT_DRAIN_GRACE_S
        elif elapsed >= timeout_s and process.poll() is None:
            timed_out, timeout_kind = True, "agent_timeout"
            _terminate_agent(cid, process)
            drain_deadline = time.monotonic() + AGENT_EXIT_DRAIN_GRACE_S
        if process.poll() is not None and drain_deadline is None:
            drain_deadline = time.monotonic() + AGENT_EXIT_DRAIN_GRACE_S
        if process.poll() is not None and open_streams == 0 and output_queue.empty():
            break
        if drain_deadline is not None and time.monotonic() >= drain_deadline:
            break
    for stream in (process.stdout, process.stderr):
        try:
            if stream is not None:
                stream.close()
        except Exception:
            pass
    return AgentProcessResult(
        returncode=int(process.returncode or 0),
        stdout="".join(stdout),
        stderr="".join(stderr),
        saw_event=saw_event,
        timed_out=timed_out,
        timeout_kind=timeout_kind,
        n_tool_calls=len(tool_ids) + anonymous_tool_calls,
    )


def _append_bounded_output(chunks: list[str], line: str, size: int, limit: int) -> int:
    if size >= limit:
        return size
    remaining = limit - size
    if len(line) <= remaining:
        chunks.append(line)
        return size + len(line)
    chunks.append(line[:remaining])
    chunks.append("\n...[agent output truncated by harness]...\n")
    return limit


def _start_reader(
    stream: Any,
    source: str,
    output_queue: queue.Queue[tuple[str, str | None]],
) -> None:
    def run() -> None:
        if stream is None:
            output_queue.put((source, None))
            return
        try:
            for line in iter(stream.readline, ""):
                output_queue.put((source, line))
        finally:
            output_queue.put((source, None))

    threading.Thread(target=run, daemon=True).start()


def _tool_event_identity(event: Mapping[str, Any]) -> tuple[bool, str]:
    if isinstance(event.get("part"), Mapping):
        part = event["part"]
    elif isinstance(event.get("item"), Mapping):
        part = event["item"]
    else:
        part = event
    kind = f"{event.get('type') or ''} {part.get('type') or ''}".lower()
    is_tool = any(
        marker in kind for marker in ("tool", "command_execution", "function_call")
    )
    if not is_tool:
        return False, ""
    call_id = str(
        part.get("callID")
        or part.get("callId")
        or part.get("call_id")
        or part.get("toolCallId")
        or part.get("tool_call_id")
        or part.get("id")
        or ""
    )
    return True, call_id


def _count_tool_calls(events: Sequence[Mapping[str, Any]]) -> int:
    ids: set[str] = set()
    count = 0
    for event in events:
        is_tool, call_id = _tool_event_identity(event)
        if not is_tool:
            continue
        if call_id:
            ids.add(call_id)
        else:
            count += 1
    return len(ids) + count


def _step_limit_reached(events: Sequence[Mapping[str, Any]], max_steps: int) -> bool:
    return _count_tool_calls(events) >= max(1, int(max_steps))


def _terminate_agent(cid: str, process: subprocess.Popen[str]) -> None:
    try:
        subprocess.run(
            [
                _docker(),
                "exec",
                cid,
                "pkill",
                "-TERM",
                "-f",
                "opencode|codex|pi-coding-agent|pi_compact_runner",
            ],
            capture_output=True,
            timeout=15,
            check=False,
        )
    except Exception:
        pass
    try:
        process.terminate()
        process.wait(timeout=10)
    except Exception:
        try:
            process.kill()
            process.wait(timeout=5)
        except Exception:
            pass


def _contains_infra_error(text: str) -> bool:
    lowered = str(text).lower()
    return any(marker in lowered for marker in INFRA_ERROR_MARKERS)


def _agent_infrastructure_error(agent: AgentProcessResult) -> bool:
    combined = f"{agent.stdout}\n{agent.stderr}"
    if not agent.saw_event:
        return bool(
            agent.timed_out or agent.returncode != 0 or _contains_infra_error(combined)
        )
    if not _contains_infra_error(combined):
        return False
    for raw in agent.stdout.splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping):
            continue
        if str(event.get("type") or "") == "turn.failed":
            return True
        message = event.get("message")
        if isinstance(message, Mapping) and str(
            message.get("stopReason") or message.get("stop_reason") or ""
        ).lower() == "error":
            return True
    return False


def _container_host_proxy_address() -> str:
    configured = str(os.environ.get("TB_DOCKER_HOST_PROXY_HOST") or "").strip()
    if configured and configured not in {"host.docker.internal", "host-gateway"}:
        return configured
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("1.1.1.1", 53))
        return str(probe.getsockname()[0])
    finally:
        probe.close()


def _verify(
    cid: str, task_root: Path, output_root: Path, timeout: int
) -> tuple[float, bool]:
    verifier_root = _verifier_root(task_root)
    run_tests = verifier_root / "run-tests.sh"
    tests = verifier_root / "tests"
    if run_tests.is_file():
        script = run_tests.read_text(encoding="utf-8").replace(
            "curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh",
            "command -v uv >/dev/null 2>&1 || curl --http1.1 -LsSf https://astral.sh/uv/0.7.13/install.sh | sh",
        )
        script = re.sub(
            r"(?m)^(\s*)source\s+\$HOME/\.local/bin/env\s*$",
            r"\1export PATH=/usr/local/bin:$PATH",
            script,
        )
        script = "\n".join(
            (
                re.sub(
                    r"(?<!\S)([A-Za-z0-9][A-Za-z0-9+.-]*)=[^\s\\]+",
                    r"\1",
                    line,
                )
                if re.search(r"\bapt-get\s+install\b", line)
                else line
            )
            for line in script.splitlines()
        ) + ("\n" if script.endswith("\n") else "")
        patched = output_root / "run-tests.harness.sh"
        patched.write_text(script, encoding="utf-8")
        if re.search(r"\buv\b|astral\.sh/uv", script):
            _ensure_uv(cid)
        _dexec(cid, "rm -rf /tests && mkdir -p /tests", timeout=60, workdir="/")
        _dcp_in(patched, cid, "/tests/run-tests.sh")
        if tests.is_dir():
            _dcp_contents(tests, cid, "/tests")
        _dexec(
            cid,
            "if command -v fuser >/dev/null 2>&1; then "
            "deadline=$((SECONDS+120)); "
            "while fuser /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock >/dev/null 2>&1; do "
            "[ $SECONDS -ge $deadline ] && break; sleep 2; done; fi; "
            "if [ -d /etc/apt/apt.conf.d ]; then "
            'printf \'Acquire::Retries "5";\\nAcquire::http::Timeout "120";\\nAcquire::https::Timeout "120";\\n\' '
            "> /etc/apt/apt.conf.d/80-harness-retries; fi",
            timeout=150,
            workdir="/",
        )
        result = _dexec(
            cid,
            "chmod +x /tests/run-tests.sh && TEST_DIR=/tests bash /tests/run-tests.sh",
            timeout=timeout,
        )
        output = (result.stdout or b"").decode("utf-8", "replace") + (
            result.stderr or b""
        ).decode("utf-8", "replace")
        (output_root / "test-output.txt").write_text(output, encoding="utf-8")
        return _verifier_outcome(output, result.returncode)
    harbor_test = tests / "test.sh"
    if harbor_test.is_file():
        _dexec(
            cid,
            "rm -rf /tests /logs/verifier && mkdir -p /tests /logs/verifier",
            timeout=60,
            workdir="/",
        )
        _dcp_contents(tests, cid, "/tests")
        result = _dexec(
            cid,
            "chmod +x /tests/test.sh && bash /tests/test.sh",
            timeout=timeout,
            workdir="/app",
        )
        output = (result.stdout or b"").decode("utf-8", "replace") + (
            result.stderr or b""
        ).decode("utf-8", "replace")
        (output_root / "test-output.txt").write_text(output, encoding="utf-8")
        if result.returncode == 124:
            return 0.0, True
        reward = _dexec(
            cid,
            "if [ -s /logs/verifier/reward.json ]; then cat /logs/verifier/reward.json; "
            "elif [ -s /logs/verifier/reward.txt ]; then cat /logs/verifier/reward.txt; "
            "else exit 44; fi",
            timeout=30,
            workdir="/",
        )
        if reward.returncode != 0:
            raise RuntimeError(
                "Harbor verifier did not write /logs/verifier/reward.{json,txt}: "
                + output[-1200:]
            )
        reward_text = (reward.stdout or b"").decode("utf-8", "replace").strip()
        try:
            parsed = json.loads(reward_text)
            value = parsed.get("reward") if isinstance(parsed, Mapping) else parsed
        except json.JSONDecodeError:
            value = reward_text
        return float(value), False
    raise RuntimeError(f"task verifier is unavailable: {task_root}")


def _task_has_verifier(task_root: Path) -> bool:
    return (task_root / "run-tests.sh").is_file() or (
        task_root / "tests" / "test.sh"
    ).is_file()


def _verifier_root(task_root: Path) -> Path:
    if _task_has_verifier(task_root):
        return task_root
    repo_root = task_root.parents[3]
    fallback = repo_root / "assets" / "terminal_task_assets" / task_root.name
    if _task_has_verifier(fallback):
        return fallback
    raise FileNotFoundError(
        f"{task_root.name}: missing run-tests.sh or Harbor tests/test.sh"
    )


def _verifier_outcome(output: str, returncode: int) -> tuple[float, bool]:
    if returncode == 124:
        if "test session starts" in output.lower():
            return 0.0, True
        raise RuntimeError(f"verifier setup timeout: {output[-1200:]}")
    if (
        returncode != 0
        and _contains_infra_error(output)
        and "test session starts" not in output.lower()
    ):
        raise RuntimeError(f"verifier infrastructure failure: {output[-1200:]}")
    return _pytest_reward(output, returncode), False


def _ensure_uv(cid: str) -> None:
    check = _dexec(cid, "command -v uv && uv --version", timeout=30, workdir="/")
    if check.returncode == 0:
        return
    host_uv = shutil.which("uv")
    if not host_uv:
        raise RuntimeError("uv is unavailable on host and in task container")
    _dcp_in(Path(host_uv).resolve(), cid, "/usr/local/bin/uv")
    installed = _dexec(
        cid, "chmod +x /usr/local/bin/uv && uv --version", timeout=30, workdir="/"
    )
    if installed.returncode != 0:
        raise RuntimeError("host uv binary is incompatible with task container")


def _pytest_reward(output: str, returncode: int) -> float:
    parts = re.split(
        r"=+\s*short test summary info\s*=+", output, flags=re.IGNORECASE, maxsplit=1
    )
    if len(parts) >= 2:
        statuses = []
        for raw in parts[1].splitlines():
            first = (
                raw.strip().split(maxsplit=1)[0].strip(":").upper()
                if raw.strip()
                else ""
            )
            if first in {"PASSED", "FAILED", "ERROR", "XPASS", "XFAIL", "SKIPPED"}:
                statuses.append(first)
        if statuses:
            return (
                0.0
                if any(status in {"FAILED", "ERROR", "XPASS"} for status in statuses)
                else 1.0
            )
    return 1.0 if returncode == 0 else 0.0


def _stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            pass


def _trial_path(root: Path, task_id: str, trial: int) -> Path:
    return root / _safe(task_id) / f"trial_{trial + 1:04d}.jsonl"


def _write_trial_row(
    root: Path, task_id: str, trial: int, row: Mapping[str, Any]
) -> Path:
    path = _trial_path(root, task_id, trial)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(row), handle, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    task_count = len(records)
    trial_count = sum(len(record.get("rewards") or []) for record in records)
    successes = sum(
        float(reward or 0.0) >= 1.0
        for record in records
        for reward in record.get("rewards") or []
    )
    pass_at_1 = successes / trial_count if trial_count else 0.0
    return {
        "task_count": task_count,
        "trial_count": trial_count,
        "trial_success_count": successes,
        "trial_success_rate": pass_at_1,
        "pass_at_1": pass_at_1,
        "pass_at_2": (
            sum(
                any(
                    float(value or 0.0) >= 1.0
                    for value in (record.get("rewards") or [])[:2]
                )
                for record in records
            )
            / task_count
            if task_count
            else 0.0
        ),
        "worker_error_count": sum(
            len(record.get("worker_errors") or []) for record in records
        ),
    }
