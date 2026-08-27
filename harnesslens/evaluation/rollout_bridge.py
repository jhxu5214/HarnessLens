from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from harnesslens.benchmarks.bird_eval import BirdLimits, run_bird_batch
from harnesslens.harnesses.opencode_harness import normalize_opencode_manifest
from harnesslens.harnesses.opencode_runtime import validate_api_trace
from harnesslens.harnesses.harness_manifest import (
    empty_harness_manifest,
    merge_native_manifests,
    normalize_harness,
    normalize_native_manifest,
)
from harnesslens.harnesses.harness_workspace import (
    empty_workspace_snapshot,
    normalize_workspace_snapshot,
)
from harnesslens.harnesses.native_candidate_runtime import attach_candidate_workspace
from harnesslens.benchmarks.terminal_bench import (
    TerminalLimits,
    run_terminal_batch,
)
from harnesslens.core.train_protocol import MAX_ROLLOUT_CONCURRENCY


MAX_CONCURRENCY = MAX_ROLLOUT_CONCURRENCY
TAU2_AGENT_STEPS_PER_TURN = 10
TAU2_MAX_CONVERSATION_TURNS = 40


def tau2_timeout_per_turn_s() -> int:
    raw = str(os.environ.get("HAI_TAU2_TIMEOUT_PER_TURN_S") or "180")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "HAI_TAU2_TIMEOUT_PER_TURN_S must be a positive integer"
        ) from exc
    if value < 1:
        raise ValueError(
            "HAI_TAU2_TIMEOUT_PER_TURN_S must be a positive integer"
        )
    return value


TAU2_TIMEOUT_PER_TURN_S = tau2_timeout_per_turn_s()
TERMINAL_AGENT_STEPS = 50
TERMINAL_EXEC_TIMEOUT_S = 600
TERMINAL_VERIFY_TIMEOUT_S = 1800
BIRD_CELL = "bird_mini_dev_challenging"


class IncompleteRolloutTraceError(RuntimeError):
    def __init__(self, trajectory_path: str | Path, detail: str) -> None:
        self.trajectory_path = Path(trajectory_path).resolve()
        self.detail = str(detail)
        super().__init__(
            f"retained rollout trace is incomplete for {self.trajectory_path}: {self.detail}"
        )


@dataclass(frozen=True)
class RolloutRequest:
    request_id: str
    run_id: str
    scope: str
    harness_version: str
    task_repeats: Mapping[str, int]
    max_concurrency: int
    purpose: str
    pairing_offsets: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "run_id": self.run_id,
            "scope": self.scope,
            "harness_version": self.harness_version,
            "task_repeats": {str(key): int(value) for key, value in self.task_repeats.items()},
            "max_concurrency": int(self.max_concurrency),
            "purpose": self.purpose,
            "pairing_offsets": {
                str(key): int(value) for key, value in self.pairing_offsets.items()
            },
        }


@dataclass(frozen=True)
class TrainRolloutRecord:
    task_id: str
    rewards: tuple[float, ...]
    harness_version: str
    trajectory_paths: tuple[str, ...] = ()
    worker_errors: tuple[Any, ...] = ()
    trial_summaries: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "rewards": list(self.rewards),
            "harness_version": self.harness_version,
            "trajectory_paths": list(self.trajectory_paths),
            "worker_errors": list(self.worker_errors),
            "trial_summaries": [dict(item) for item in self.trial_summaries],
            "pass_at_1": _pass_at(self.rewards, 1),
            "pass_at_2": _pass_at(self.rewards, 2),
        }


@dataclass(frozen=True)
class RolloutResponse:
    request_id: str
    harness_version: str
    budget_spent: int
    budget_remaining: int
    trajectory_root: str
    summary_json: str
    metadata_json: str
    metrics: Mapping[str, Any]
    per_task: Mapping[str, Mapping[str, Any]]
    records: tuple[TrainRolloutRecord, ...] = ()
    scope: str = "TRAIN"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["records"] = [record.to_dict() for record in self.records]
        return payload


class CellHarnessRepository:
    def __init__(
        self,
        *,
        cell: str,
        repo_root: str | Path,
        run_id: str,
        evidence_root: str | Path,
        harness: str = "opencode",
        timeout_s: int = 1200,
    ) -> None:
        self.cell = str(cell)
        self.repo_root = Path(repo_root).resolve()
        self.run_id = str(run_id)
        self.evidence_root = Path(evidence_root).resolve()
        self.harness = normalize_harness(harness)
        self.timeout_s = int(timeout_s)

    def materialize_candidate(
        self, *, base_version: str, candidate_label: str, delta: Mapping[str, Any]
    ) -> str:
        if self.harness != "opencode":
            return self._materialize_native_candidate(
                base_version=base_version,
                candidate_label=candidate_label,
                delta=delta,
            )
        return self._materialize_opencode_candidate(
            base_version=base_version,
            candidate_label=candidate_label,
            delta=delta,
        )

    def materialize_workspace_candidate(
        self,
        *,
        base_version: str,
        candidate_label: str,
        workspace: Mapping[str, Any],
        manifest_delta: Mapping[str, Any] | None = None,
    ) -> str:
        version = _safe(candidate_label)
        root = _native_harness_root(
            self.evidence_root,
            self.run_id,
            self.cell,
            version,
            self.harness,
        )
        root.mkdir(parents=True, exist_ok=True)
        base_manifest = self.read_candidate_snapshot(base_version)
        delta = dict(manifest_delta or {})
        if delta:
            merged_manifest = (
                _merge_bird_manifest(base_manifest, _manifest_for_v1(delta))
                if self.harness == "opencode"
                else merge_native_manifests(
                    base_manifest, normalize_native_manifest(delta)
                )
            )
        else:
            merged_manifest = base_manifest
        _write_json(root / "manifest.json", merged_manifest)
        _write_json(root / "workspace.json", normalize_workspace_snapshot(workspace))
        self._write_version_metadata(version=version, parent=str(base_version))
        return version

    def _materialize_native_candidate(
        self, *, base_version: str, candidate_label: str, delta: Mapping[str, Any]
    ) -> str:
        version = _safe(candidate_label)
        base = self.read_candidate_snapshot(base_version)
        merged = merge_native_manifests(base, normalize_native_manifest(delta))
        root = _native_harness_root(
            self.evidence_root,
            self.run_id,
            self.cell,
            version,
            self.harness,
        )
        root.mkdir(parents=True, exist_ok=True)
        _write_json(root / "manifest.json", merged)
        _write_json(root / "workspace.json", self.read_workspace_snapshot(base_version))
        self._write_version_metadata(version=version, parent=str(base_version))
        return version

    def _materialize_opencode_candidate(
        self, *, base_version: str, candidate_label: str, delta: Mapping[str, Any]
    ) -> str:
        version = _safe(candidate_label)
        base = self.read_candidate_snapshot(base_version)
        merged = _merge_bird_manifest(base, _manifest_for_v1(delta))
        root = _native_harness_root(
            self.evidence_root,
            self.run_id,
            self.cell,
            version,
            self.harness,
        )
        root.mkdir(parents=True, exist_ok=True)
        _write_json(root / "manifest.json", merged)
        _write_json(root / "workspace.json", self.read_workspace_snapshot(base_version))
        self._write_version_metadata(version=version, parent=str(base_version))
        return version

    def read_candidate_snapshot(self, version: str) -> dict[str, Any]:
        path = _native_harness_root(
            self.evidence_root,
            self.run_id,
            self.cell,
            str(version),
            self.harness,
        ) / "manifest.json"
        if not path.is_file():
            if str(version) == "v0":
                return (
                    normalize_opencode_manifest({})
                    if self.harness == "opencode"
                    else empty_harness_manifest()
                )
            raise ValueError(f"missing {self.harness} candidate snapshot: {version}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        return (
            normalize_opencode_manifest(raw)
            if self.harness == "opencode"
            else normalize_native_manifest(raw)
        )

    def read_workspace_snapshot(self, version: str) -> dict[str, Any]:
        path = _native_harness_root(
            self.evidence_root,
            self.run_id,
            self.cell,
            str(version),
            self.harness,
        ) / "workspace.json"
        if not path.is_file():
            if str(version) == "v0":
                return empty_workspace_snapshot()
            raise ValueError(
                f"missing {self.harness} candidate workspace snapshot: {version}"
            )
        return normalize_workspace_snapshot(
            json.loads(path.read_text(encoding="utf-8"))
        )

    def _write_version_metadata(self, *, version: str, parent: str) -> None:
        version_root = _native_harness_root(
            self.evidence_root,
            self.run_id,
            self.cell,
            version,
            self.harness,
        ).parent.parent
        _write_json(
            version_root / "lineage.json",
            {"version": version, "parent": parent, "harness": self.harness},
        )
        _write_json(
            version_root / "meta.json",
            {
                "version": version,
                "parent": parent,
                "harness": self.harness,
                "status": "temporary",
                "temporary_candidate": True,
            },
        )

class TrainRolloutService:
    def __init__(
        self,
        *,
        cell: str,
        repo_root: str | Path,
        run_id: str,
        artifact_root: str | Path,
        train_task_ids: list[str] | tuple[str, ...],
        initial_budget: int,
        evidence_root: str | Path,
        workspace_root: str | Path,
        harness: str = "opencode",
        timeout_s: int = 7200,
        local_rootless_rollout: bool = True,
    ) -> None:
        self.cell = str(cell)
        self.repo_root = Path(repo_root).resolve()
        self.run_id = str(run_id)
        self.artifact_root = Path(artifact_root).resolve()
        self.train_task_ids = {str(item) for item in train_task_ids}
        self.remaining_budget = int(initial_budget)
        self.evidence_root = Path(evidence_root).resolve()
        self.workspace_root = Path(workspace_root).resolve()
        self.harness = normalize_harness(harness)
        self.timeout_s = int(timeout_s)
        self.local_rootless_rollout = bool(local_rootless_rollout)

    def run(self, request: RolloutRequest) -> RolloutResponse:
        self._validate(request)
        request_root = self.artifact_root / request.run_id / request.request_id
        cached = self._load_cached(request, request_root)
        if cached is not None:
            self.remaining_budget = cached.budget_remaining
            return cached
        groups: dict[int, list[str]] = {}
        for task_id, repeats in request.task_repeats.items():
            groups.setdefault(int(repeats), []).append(str(task_id))
        records: list[TrainRolloutRecord] = []
        per_task: dict[str, dict[str, Any]] = {}
        for repeats, task_ids in sorted(groups.items()):
            group_run_id = _safe(f"{request.run_id}_{request.request_id}_k{repeats}")
            try:
                raw = self._run_group(request, sorted(task_ids), repeats)
                retained = retain_trial_trajectories(
                    raw, target_root=request_root / "trajectories"
                )
                per_task.update(
                    {
                        str(key): dict(value)
                        for key, value in (retained.get("per_task") or {}).items()
                    }
                )
                for item in retained.get("records") or []:
                    records.append(_record_from_mapping(item, request.harness_version))
            finally:
                _cleanup_rollout_group_workspace(self.workspace_root / group_run_id)
        requested = sum(int(value) for value in request.task_repeats.values())
        infrastructure = _infrastructure_failure_count(records)
        # A provider/worker failure is an invalid sample, not a trace-repair
        # target. Repairing it can turn infrastructure errors into apparently
        # complete behavioral failures and incorrectly admit them to review.
        if infrastructure == 0:
            if self.cell == "terminal_bench":
                validate_terminal_rollout_trajectories(records)
            elif self.cell == BIRD_CELL:
                validate_bird_rollout_interactions(records)
            elif self.harness in {"pi", "codex"}:
                validate_native_rollout_interactions(records, harness=self.harness)
            else:
                validate_rollout_interactions(records)
        charged = max(0, requested - infrastructure)
        self.remaining_budget -= charged
        metrics = {
            **_summarize(records),
            "requested_trial_count": requested,
            "infrastructure_failure_count": infrastructure,
            "charged_trial_count": charged,
            "workspace_cleanup_enabled": os.environ.get("HAI_KEEP_TRAJECTORY_WORKSPACE", "0").lower()
            not in {"1", "true", "yes", "on"},
            "trajectory_retention": _trajectory_retention(self.cell, self.harness),
            "api_trace_required": _api_trace_required(self.cell, self.harness),
        }
        metadata_path = request_root / "metadata.json"
        summary_path = request_root / "summary.json"
        metadata = {
            "request": request.to_dict(),
            "budget_spent": charged,
            "budget_remaining": self.remaining_budget,
            "per_task": per_task,
            "records": [record.to_dict() for record in records],
            "metrics": metrics,
        }
        _write_json(metadata_path, metadata)
        _write_json(
            summary_path,
            {
                "request_id": request.request_id,
                "harness_version": request.harness_version,
                "budget_spent": charged,
                "budget_remaining": self.remaining_budget,
                "metrics": metrics,
                "per_task": per_task,
            },
        )
        return RolloutResponse(
            request_id=request.request_id,
            harness_version=request.harness_version,
            budget_spent=charged,
            budget_remaining=self.remaining_budget,
            trajectory_root=str(request_root / "trajectories"),
            summary_json=str(summary_path),
            metadata_json=str(metadata_path),
            metrics=metrics,
            per_task=per_task,
            records=tuple(records),
        )
    def recover_retained(self, request: RolloutRequest) -> RolloutResponse | None:
        self._validate(request)
        request_root = self.artifact_root / request.run_id / request.request_id
        expected_paths = {
            str(task_id): [
                request_root / "trajectories" / str(task_id) / f"trial_{index + 1:04d}.jsonl"
                for index in range(int(repeats))
            ]
            for task_id, repeats in request.task_repeats.items()
        }
        present = [path.is_file() for paths in expected_paths.values() for path in paths]
        if not any(present):
            return None
        if not all(present):
            raise RuntimeError("retained rollout batch is only partially materialized")
        records: list[TrainRolloutRecord] = []
        per_task: dict[str, dict[str, Any]] = {}
        for task_id, paths in expected_paths.items():
            rows = [
                json.loads(path.read_text(encoding="utf-8").strip()) for path in paths
            ]
            rewards = [float(row.get("reward", 0.0) or 0.0) for row in rows]
            errors = [
                {"trial": row.get("trial", index), "error": str(row.get("error"))}
                for index, row in enumerate(rows)
                if row.get("error")
            ]
            summaries = [
                {
                    "trial": row.get("trial", index),
                    "pairing_slot": row.get("pairing_slot", row.get("trial", index)),
                    "simulation_seed": row.get("simulation_seed"),
                    "reward": rewards[index],
                    "n_messages": int(
                        row.get("n_messages") or len(row.get("messages") or []) or 0
                    ),
                    "n_tool_calls": int(
                        row.get("n_tool_calls") or row.get("n_total_calls") or 0
                    ),
                    "termination": str(
                        row.get("termination") or row.get("stop_reason") or ""
                    ),
                    "error": str(row.get("error") or "")[:300],
                }
                for index, row in enumerate(rows)
            ]
            record = TrainRolloutRecord(
                task_id=task_id,
                rewards=tuple(rewards),
                harness_version=request.harness_version,
                trajectory_paths=tuple(str(path.resolve()) for path in paths),
                worker_errors=tuple(errors),
                trial_summaries=tuple(summaries),
            )
            records.append(record)
            per_task[task_id] = {
                "repeats": len(paths),
                "rewards": rewards,
                "trajectory_paths": list(record.trajectory_paths),
                "worker_errors": errors,
                "trial_summaries": summaries,
            }
        if self.cell == "terminal_bench":
            validate_terminal_rollout_trajectories(records)
        elif self.cell == BIRD_CELL:
            validate_bird_rollout_interactions(records)
        elif self.harness in {"pi", "codex"}:
            validate_native_rollout_interactions(records, harness=self.harness)
        else:
            validate_rollout_interactions(records)
        requested = sum(int(value) for value in request.task_repeats.values())
        self.remaining_budget -= requested
        metrics = {
            **_summarize(records),
            "requested_trial_count": requested,
            "infrastructure_failure_count": 0,
            "charged_trial_count": requested,
            "workspace_cleanup_enabled": True,
            "trajectory_retention": _trajectory_retention(self.cell, self.harness),
            "api_trace_required": _api_trace_required(self.cell, self.harness),
        }
        metadata_path = request_root / "metadata.json"
        summary_path = request_root / "summary.json"
        metadata = {
            "request": request.to_dict(),
            "budget_spent": requested,
            "budget_remaining": self.remaining_budget,
            "per_task": per_task,
            "records": [record.to_dict() for record in records],
            "metrics": metrics,
            "recovered_from_retained_trials": True,
        }
        _write_json(metadata_path, metadata)
        _write_json(
            summary_path,
            {
                "request_id": request.request_id,
                "harness_version": request.harness_version,
                "budget_spent": requested,
                "budget_remaining": self.remaining_budget,
                "metrics": metrics,
                "per_task": per_task,
                "recovered_from_retained_trials": True,
            },
        )
        return RolloutResponse(
            request_id=request.request_id,
            harness_version=request.harness_version,
            budget_spent=requested,
            budget_remaining=self.remaining_budget,
            trajectory_root=str(request_root / "trajectories"),
            summary_json=str(summary_path),
            metadata_json=str(metadata_path),
            metrics=metrics,
            per_task=per_task,
            records=tuple(records),
        )

    def replace_incomplete_trial(
        self,
        *,
        request: RolloutRequest,
        failure: IncompleteRolloutTraceError,
        repair_service: "TrainRolloutService",
    ) -> RolloutResponse:
        target = failure.trajectory_path
        task_id = target.parent.name
        trial_number = int(target.stem.rsplit("_", 1)[1])
        repair_request = RolloutRequest(
            request_id=f"{request.request_id}-repair-{task_id}-{trial_number:04d}",
            run_id=request.run_id,
            scope=request.scope,
            harness_version=request.harness_version,
            task_repeats={task_id: 1},
            max_concurrency=1,
            purpose="repair incomplete retained rollout interaction trace",
            pairing_offsets={
                task_id: int(request.pairing_offsets.get(task_id, 0)) + trial_number - 1
            },
        )
        repaired = repair_service.run(repair_request)
        source = Path(repaired.records[0].trajectory_paths[0])
        row = json.loads(source.read_text(encoding="utf-8").strip())
        row = _retain_api_sidecars(row, source_dir=source.parent, destination=target)
        target.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        recovered = self.recover_retained(request)
        if recovered is None:
            raise RuntimeError("repaired rollout batch could not be reconstructed")
        return recovered

    def _run_group(
        self, request: RolloutRequest, task_ids: list[str], repeats: int
    ) -> dict[str, Any]:
        group_run_id = _safe(f"{request.run_id}_{request.request_id}_k{repeats}")
        if self.cell == BIRD_CELL:
            if self.harness == "opencode":
                _ensure_bird_v0(self.evidence_root, self.run_id)
            manifest = _rollout_candidate_manifest(
                CellHarnessRepository(
                    cell=self.cell,
                    repo_root=self.repo_root,
                    run_id=self.run_id,
                    evidence_root=self.evidence_root,
                    harness=self.harness,
                ),
                request.harness_version,
            )
            return run_bird_batch(
                repo_root=self.repo_root,
                run_root=(
                    self.evidence_root
                    / group_run_id
                    / self.cell
                    / request.harness_version
                ),
                scope=request.scope,
                harness=self.harness,
                harness_version=request.harness_version,
                harness_manifest=manifest,
                task_repeats={task_id: repeats for task_id in task_ids},
                pairing_offsets={
                    task_id: int(request.pairing_offsets.get(task_id, 0))
                    for task_id in task_ids
                },
                max_concurrency=request.max_concurrency,
                limits=BirdLimits(),
            )
        if self.cell == "terminal_bench":
            manifest = _rollout_candidate_manifest(
                CellHarnessRepository(
                    cell=self.cell,
                    repo_root=self.repo_root,
                    run_id=self.run_id,
                    evidence_root=self.evidence_root,
                    harness=self.harness,
                ),
                request.harness_version,
            )
            return run_terminal_batch(
                repo_root=self.repo_root,
                run_root=self.artifact_root / "terminal_groups" / group_run_id,
                scope=request.scope,
                harness=self.harness,
                harness_version=request.harness_version,
                harness_manifest=manifest,
                task_repeats={task_id: repeats for task_id in task_ids},
                pairing_offsets={
                    task_id: int(request.pairing_offsets.get(task_id, 0))
                    for task_id in task_ids
                },
                max_concurrency=request.max_concurrency,
                limits=TerminalLimits(
                    max_steps=TERMINAL_AGENT_STEPS,
                    agent_timeout_s=TERMINAL_EXEC_TIMEOUT_S,
                    verify_timeout_s=TERMINAL_VERIFY_TIMEOUT_S,
                ),
            )
        if self.harness in {"opencode", "pi", "codex"}:
            return self._run_native_tau2_group(
                request=request,
                task_ids=task_ids,
                repeats=repeats,
                group_run_id=group_run_id,
            )
        raise ValueError(f"unsupported Tau2 harness: {self.harness}")

    def _run_native_tau2_group(
        self,
        *,
        request: RolloutRequest,
        task_ids: list[str],
        repeats: int,
        group_run_id: str,
    ) -> dict[str, Any]:
        if self.cell not in {"retail", "banking_knowledge"}:
            raise ValueError(
                f"{self.harness} rollout is not implemented for benchmark cell {self.cell}"
            )
        from harnesslens.benchmarks.tau2_driver import Tau2Limits

        manifest = _rollout_candidate_manifest(
            CellHarnessRepository(
                cell=self.cell,
                repo_root=self.repo_root,
                run_id=self.run_id,
                evidence_root=self.evidence_root,
                harness=self.harness,
            ),
            request.harness_version,
        )
        native_request = RolloutRequest(
            request_id=f"{request.request_id}-native-k{repeats}",
            run_id=group_run_id,
            scope=request.scope,
            harness_version=request.harness_version,
            task_repeats={task_id: repeats for task_id in task_ids},
            max_concurrency=request.max_concurrency,
            purpose=request.purpose,
            pairing_offsets={
                task_id: int(request.pairing_offsets.get(task_id, 0))
                for task_id in task_ids
            },
        )
        limits = Tau2Limits(
            max_conversation_turns=TAU2_MAX_CONVERSATION_TURNS,
            timeout_per_turn_s=TAU2_TIMEOUT_PER_TURN_S,
            max_tool_calls_per_turn=TAU2_AGENT_STEPS_PER_TURN,
            max_tool_calls=60,
            group_timeout_s=self.timeout_s,
        )
        payload = {
            "harness": self.harness,
            "repo_root": self.repo_root,
            "run_root": self.artifact_root / "native_groups" / group_run_id,
            "benchmark": self.cell,
            "request": native_request.to_dict(),
            "retrieval_config": (
                str(os.environ.get("HAI_TAU2_RETRIEVAL_CONFIG") or "bm25")
                if self.cell == "banking_knowledge"
                else None
            ),
            "limits": {
                "max_conversation_turns": limits.max_conversation_turns,
                "timeout_per_turn_s": limits.timeout_per_turn_s,
                "max_tool_calls_per_turn": limits.max_tool_calls_per_turn,
                "max_tool_calls": limits.max_tool_calls,
                "group_timeout_s": limits.group_timeout_s,
                "timeout_retries_per_turn": limits.timeout_retries_per_turn,
            },
            "harness_manifest": manifest,
        }
        return _run_native_tau2_worker(
            repo_root=self.repo_root,
            payload=payload,
            timeout_s=self.timeout_s,
        )

    def _load_cached(
        self, request: RolloutRequest, request_root: Path
    ) -> RolloutResponse | None:
        summary_path = request_root / "summary.json"
        metadata_path = request_root / "metadata.json"
        if not summary_path.is_file() or not metadata_path.is_file():
            return None
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("request") != request.to_dict():
            raise RuntimeError("existing rollout artifacts do not match the request")
        records = tuple(
            _record_from_mapping(item, request.harness_version)
            for item in metadata.get("records") or []
            if isinstance(item, Mapping)
        )
        return RolloutResponse(
            request_id=request.request_id,
            harness_version=request.harness_version,
            budget_spent=int(summary.get("budget_spent") or 0),
            budget_remaining=int(summary.get("budget_remaining") or self.remaining_budget),
            trajectory_root=str(request_root / "trajectories"),
            summary_json=str(summary_path),
            metadata_json=str(metadata_path),
            metrics=dict(summary.get("metrics") or {}),
            per_task=dict(summary.get("per_task") or {}),
            records=records,
        )

    def _validate(self, request: RolloutRequest) -> None:
        if request.scope != "TRAIN":
            raise ValueError("rollout only accepts TRAIN")
        if not 1 <= int(request.max_concurrency) <= MAX_CONCURRENCY:
            raise ValueError("rollout concurrency is outside the supported range")
        if set(map(str, request.task_repeats)) - self.train_task_ids:
            raise ValueError("rollout selected a task outside TRAIN")
        if set(map(str, request.pairing_offsets)) - set(map(str, request.task_repeats)):
            raise ValueError("pairing offset references an unselected task")
        if any(int(value) < 0 for value in request.pairing_offsets.values()):
            raise ValueError("pairing offsets must be nonnegative")
        if sum(int(value) for value in request.task_repeats.values()) > self.remaining_budget:
            raise ValueError("rollout request exceeds its allocated budget")


def _cleanup_rollout_group_workspace(path: Path) -> None:
    if os.environ.get("HAI_KEEP_TRAJECTORY_WORKSPACE", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    if path.exists() or path.is_symlink():
        shutil.rmtree(path)


def retain_trial_trajectories(
    payload: Mapping[str, Any], *, target_root: str | Path
) -> dict[str, Any]:
    normalized = dict(payload)
    per_task = {
        str(key): dict(value)
        for key, value in (payload.get("per_task") or {}).items()
        if isinstance(value, Mapping)
    }
    records = [dict(item) for item in payload.get("records") or [] if isinstance(item, Mapping)]
    by_task = {str(item.get("task_id")): item for item in records}
    target = Path(target_root)
    for task_id in dict.fromkeys([*per_task, *by_task]):
        task_payload = per_task.setdefault(task_id, {})
        record = by_task.get(task_id, {})
        source_paths = [
            str(item)
            for item in (record.get("trajectory_paths") or task_payload.get("trajectory_paths") or [])
        ]
        rows: list[tuple[dict[str, Any], Path]] = []
        for raw_path in source_paths:
            source = Path(raw_path)
            if not source.is_file():
                continue
            for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.strip():
                    rows.append((dict(json.loads(line)), source.parent))
        expected = int(task_payload.get("repeats") or len(record.get("rewards") or []) or len(rows))
        retained: list[str] = []
        task_root = target / _safe(task_id)
        task_root.mkdir(parents=True, exist_ok=True)
        for index in range(expected):
            row, source_dir = rows[index] if index < len(rows) else (
                {"task_id": task_id, "trial": index, "trajectory_unavailable": True},
                task_root,
            )
            destination = task_root / f"trial_{index + 1:04d}.jsonl"
            row = _retain_api_sidecars(row, source_dir=source_dir, destination=destination)
            destination.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            retained.append(str(destination))
        task_payload["source_trajectory_paths"] = source_paths
        task_payload["trajectory_paths"] = retained
        if record:
            record["trajectory_paths"] = retained
    normalized["per_task"] = per_task
    normalized["records"] = records
    normalized["trajectory_root"] = str(target)
    return normalized


def validate_rollout_interactions(records: list[TrainRolloutRecord]) -> None:
    for record in records:
        if len(record.trajectory_paths) != len(record.rewards):
            raise RuntimeError(
                f"rollout task {record.task_id} did not retain one trajectory per trial"
            )
        for trajectory_path in record.trajectory_paths:
            path = Path(trajectory_path)
            if not path.is_file():
                raise RuntimeError(f"retained rollout trajectory is missing: {path}")
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip()
            ]
            if len(rows) != 1 or not isinstance(rows[0], Mapping):
                raise RuntimeError(f"retained rollout trajectory is malformed: {path}")
            reference = rows[0].get("api_calls_jsonl")
            if not isinstance(reference, str) or not reference.strip():
                raise RuntimeError(f"retained rollout has no complete OpenCode trace: {path}")
            api_trace = Path(reference)
            if not api_trace.is_absolute():
                api_trace = path.parent / api_trace
            error = validate_api_trace(api_trace)
            if error:
                raise IncompleteRolloutTraceError(path, error)


def validate_native_rollout_interactions(
    records: list[TrainRolloutRecord], *, harness: str
) -> None:
    expected = normalize_harness(harness)
    if expected not in {"pi", "codex"}:
        raise ValueError("native rollout validation requires Pi or Codex")
    for record in records:
        if len(record.trajectory_paths) != len(record.rewards):
            raise RuntimeError(
                f"rollout task {record.task_id} did not retain one trajectory per trial"
            )
        for trajectory_path in record.trajectory_paths:
            path = Path(trajectory_path)
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip()
            ] if path.is_file() else []
            if len(rows) != 1 or not isinstance(rows[0], Mapping):
                raise RuntimeError(f"retained native rollout trajectory is malformed: {path}")
            row = rows[0]
            if row.get("error") and not row.get("messages"):
                continue
            if str(row.get("harness") or "") != expected:
                raise RuntimeError(
                    f"retained rollout used {row.get('harness')!r}, expected {expected!r}"
                )
            if not isinstance(row.get("messages"), list) or not isinstance(
                row.get("tool_definitions"), list
            ):
                raise RuntimeError(
                    f"retained {expected} rollout lacks model interaction context: {path}"
                )
            if expected in {"pi", "codex"}:
                reference = row.get("api_calls_jsonl")
                if not isinstance(reference, str) or not reference.strip():
                    raise IncompleteRolloutTraceError(
                        path, f"missing {expected.title()} API trace"
                    )
                api_trace = Path(reference)
                if not api_trace.is_absolute():
                    api_trace = path.parent / api_trace
                if not api_trace.is_file() or api_trace.stat().st_size == 0:
                    raise IncompleteRolloutTraceError(
                        path, f"missing {expected.title()} API trace"
                    )


def validate_bird_rollout_interactions(records: list[TrainRolloutRecord]) -> None:
    for record in records:
        if len(record.trajectory_paths) != len(record.rewards):
            raise RuntimeError(
                f"rollout task {record.task_id} did not retain one trajectory per trial"
            )
        for trajectory_path in record.trajectory_paths:
            path = Path(trajectory_path)
            try:
                row = json.loads(path.read_text(encoding="utf-8").strip())
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"retained BIRD trajectory is malformed: {path}") from exc
            if row.get("infrastructure_error"):
                continue
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise IncompleteRolloutTraceError(path, "missing user/assistant interaction")
            if messages[0].get("role") != "user" or messages[-1].get("role") != "assistant":
                raise IncompleteRolloutTraceError(path, "invalid interaction boundaries")


def validate_terminal_rollout_trajectories(records: list[TrainRolloutRecord]) -> None:
    for record in records:
        if len(record.trajectory_paths) != len(record.rewards):
            raise RuntimeError(
                f"terminal rollout task {record.task_id} did not retain one trajectory per trial"
            )
        for trajectory_path in record.trajectory_paths:
            path = Path(trajectory_path)
            if not path.is_file():
                raise RuntimeError(f"retained terminal trajectory is missing: {path}")
            rows = [
                json.loads(line)
                for line in path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                if line.strip()
            ]
            if len(rows) != 1 or not isinstance(rows[0], Mapping):
                raise RuntimeError(f"retained terminal trajectory is malformed: {path}")
            if str(rows[0].get("task_id") or "") != record.task_id:
                raise RuntimeError(f"retained terminal trajectory task mismatch: {path}")


def _retain_api_sidecars(value: Any, *, source_dir: Path, destination: Path) -> Any:
    copied: dict[Path, str] = {}

    def visit(item: Any) -> Any:
        if isinstance(item, Mapping):
            result = {}
            for key, child in item.items():
                if key == "api_calls_jsonl" and isinstance(child, str) and child.strip():
                    source = Path(child)
                    if not source.is_absolute():
                        source = source_dir / source
                    if source.is_file():
                        if source not in copied:
                            suffix = ".api_calls.jsonl" if not copied else f".api_calls.{len(copied):04d}.jsonl"
                            sidecar = destination.with_name(destination.stem + suffix)
                            shutil.copy2(source, sidecar)
                            copied[source] = sidecar.name
                        result[key] = copied[source]
                    else:
                        result[key] = child
                else:
                    result[str(key)] = visit(child)
            return result
        if isinstance(item, list):
            return [visit(child) for child in item]
        return item

    return visit(value)


def _rollout_candidate_manifest(
    repository: CellHarnessRepository, version: str
) -> dict[str, Any]:
    return attach_candidate_workspace(
        repository.read_candidate_snapshot(version),
        repository.read_workspace_snapshot(version),
    )


def _manifest_for_v1(delta: Mapping[str, Any]) -> dict[str, Any]:
    manifest = normalize_opencode_manifest(delta)
    replace = sorted(manifest["replace_channels"])
    return {
        **manifest,
        "replace_channels": replace,
        "replace_instructions": "instructions" in replace,
        "replace_prompt_appends": "prompt_appends" in replace,
        "replace_tool_desc_patches": "tool_desc_patches" in replace,
    }


def _run_native_tau2_worker(
    *, repo_root: Path, payload: Mapping[str, Any], timeout_s: int
) -> dict[str, Any]:
    executable = (
        repo_root / "third_party" / "tau3-bench" / ".venv" / "bin" / "python3"
    )
    script = (
        repo_root
        / "harnesslens"
        / "native_tau2_worker.py"
    )
    if not executable.is_file() or not script.is_file():
        raise RuntimeError("native Tau2 worker prerequisites are unavailable")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, default=str)
        input_path = Path(handle.name)
    env = dict(os.environ)
    pythonpath = str(repo_root)
    env["PYTHONPATH"] = (
        pythonpath
        if not env.get("PYTHONPATH")
        else pythonpath + os.pathsep + str(env["PYTHONPATH"])
    )
    try:
        process = subprocess.run(
            [str(executable), str(script), str(input_path)],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=int(timeout_s),
            check=False,
        )
    finally:
        input_path.unlink(missing_ok=True)
    if process.returncode != 0:
        raise RuntimeError(
            "native Tau2 worker failed: "
            + (process.stderr or process.stdout)[-4000:]
        )
    return _last_json(process.stdout)


def _bird_harness_root(
    evidence_root: str | Path, run_id: str, version: str
) -> Path:
    return (
        Path(evidence_root).resolve()
        / str(run_id)
        / "versions_percell"
        / BIRD_CELL
        / str(version)
        / "harness"
        / "opencode"
    )


def _native_harness_root(
    evidence_root: str | Path,
    run_id: str,
    cell: str,
    version: str,
    harness: str,
) -> Path:
    return (
        Path(evidence_root).resolve()
        / str(run_id)
        / "versions_percell"
        / str(cell)
        / str(version)
        / "harness"
        / normalize_harness(harness)
    )


def _ensure_bird_v0(evidence_root: str | Path, run_id: str) -> None:
    root = _bird_harness_root(evidence_root, run_id, "v0")
    root.mkdir(parents=True, exist_ok=True)
    patch = root / "patch.json"
    descriptions = root / "patch_descs.json"
    if not patch.exists():
        _write_json(
            patch,
            {
                "config_patch": {},
                "files": [],
                "instructions": [],
                "prompt_appends": [],
                "removals": [],
                "replace_channels": [],
            },
        )
    if not descriptions.exists():
        _write_json(descriptions, {})


def _merge_bird_manifest(
    base: Mapping[str, Any], delta: Mapping[str, Any]
) -> dict[str, Any]:
    left = normalize_opencode_manifest(base)
    right = normalize_opencode_manifest(delta)
    replace = set(right["replace_channels"])
    result: dict[str, Any] = {
        "config_patch": _deep_merge_mapping(
            {} if "config_patch" in replace else left["config_patch"],
            right["config_patch"],
        ),
        "tool_desc_patches": _deep_merge_mapping(
            {} if "tool_desc_patches" in replace else left["tool_desc_patches"],
            right["tool_desc_patches"],
        ),
        "instructions": list(
            dict.fromkeys(
                ([] if "instructions" in replace else left["instructions"])
                + right["instructions"]
            )
        ),
        "prompt_appends": list(
            dict.fromkeys(
                ([] if "prompt_appends" in replace else left["prompt_appends"])
                + right["prompt_appends"]
            )
        ),
        "removals": sorted(set(left["removals"]) | set(right["removals"])),
        "replace_channels": sorted(replace),
    }
    files = {
        str(item.get("path") or ""): dict(item)
        for item in left["files"]
        if str(item.get("path") or "")
    }
    for item in right["files"]:
        path = str(item.get("path") or "")
        if path:
            files[path] = dict(item)
    for path in right["removals"]:
        files.pop(str(path), None)
    result["files"] = [files[path] for path in sorted(files)]
    return result


def _deep_merge_mapping(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(left)
    for key, value in right.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge_mapping(result[key], value)
        else:
            result[key] = value
    return result


def _api_trace_required(cell: str, harness: str = "opencode") -> bool:
    return normalize_harness(harness) in {"opencode", "codex"} and str(cell) not in {
        "terminal_bench",
        BIRD_CELL,
    }


def _trajectory_retention(cell: str, harness: str = "opencode") -> str:
    if normalize_harness(harness) == "pi":
        return "harnesslens_native_trial_jsonl_with_messages_and_tool_context"
    if _api_trace_required(cell, harness):
        return "harnesslens_trial_jsonl_and_api_sidecars_retained"
    return "harnesslens_trial_jsonl_retained"


def _record_from_mapping(item: Mapping[str, Any], version: str) -> TrainRolloutRecord:
    return TrainRolloutRecord(
        task_id=str(item.get("task_id") or ""),
        rewards=tuple(float(value) for value in item.get("rewards") or []),
        harness_version=str(item.get("harness_version") or version),
        trajectory_paths=tuple(str(value) for value in item.get("trajectory_paths") or []),
        worker_errors=tuple(item.get("worker_errors") or []),
        trial_summaries=tuple(
            dict(value) for value in item.get("trial_summaries") or [] if isinstance(value, Mapping)
        ),
    )


def _pass_at(rewards: tuple[float, ...], count: int) -> int:
    return int(any(float(value) >= 1.0 for value in rewards[:count]))


def _summarize(records: list[TrainRolloutRecord]) -> dict[str, Any]:
    tasks = len(records)
    trials = sum(len(record.rewards) for record in records)
    successes = sum(float(value) >= 1.0 for record in records for value in record.rewards)
    pass_at_1 = successes / trials if trials else 0.0
    return {
        "task_count": tasks,
        "trial_count": trials,
        "trial_success_count": successes,
        "trial_success_rate": pass_at_1,
        "pass_at_1": pass_at_1,
        "pass_at_2": sum(_pass_at(record.rewards, 2) for record in records) / tasks if tasks else 0.0,
        "worker_error_count": sum(len(record.worker_errors) for record in records),
    }


def _infrastructure_failure_count(records: list[TrainRolloutRecord]) -> int:
    failures = 0
    for record in records:
        for index, summary in enumerate(record.trial_summaries):
            if (
                summary.get("error")
                and int(summary.get("n_messages") or 0) == 0
                and int(summary.get("n_tool_calls") or 0) == 0
            ):
                failures += 1
    return failures


def _last_json(stdout: str) -> dict[str, Any]:
    for raw in reversed(stdout.splitlines()):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise RuntimeError(f"subprocess emitted no JSON object: {stdout[-1000:]}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in str(value))
