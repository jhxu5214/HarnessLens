from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from harnesslens.core.artifacts import read_json, write_json
from harnesslens.core.budget import CreationBudget
from harnesslens.benchmarks.cell_config import benchmark_config
from harnesslens.harnesses.channel_preflight import PREFLIGHT_REPEAT_COUNT
from harnesslens.evaluation.rollout_bridge import (
    IncompleteRolloutTraceError,
    RolloutRequest,
    TAU2_TIMEOUT_PER_TURN_S,
    TrainRolloutService,
)
from harnesslens.core.train_protocol import TRAIN_ROLLOUT_REPEATS
from harnesslens.core.train_protocol import MAX_ROLLOUT_CONCURRENCY
from harnesslens.harnesses.harness_manifest import normalize_harness


ROLLOUT_RUNTIME_LIMITS = {
    "opencode_steps": 10,
    "max_turns": 40,
    "max_tool_calls": 60,
    "timeout_per_turn_s": TAU2_TIMEOUT_PER_TURN_S,
    "trial_timeout_s": 7200,
}
PAIRED_RETRY_REVIEW_RESERVE = 7
MAX_TARGETED_RETRY_ROUNDS = 3


@dataclass(frozen=True)
class RolloutResult:
    output: Mapping[str, Any]
    output_path: str


@dataclass(frozen=True)
class PairedRolloutResult:
    candidate: RolloutResult
    parent: RolloutResult | None


class RolloutModule:
    def __init__(
        self,
        *,
        repo_root: str | Path,
        run_root: str | Path,
        budget: CreationBudget,
        harness: str = "opencode",
        cell: str = "retail",
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.run_root = Path(run_root).resolve()
        self.budget = budget
        self.harness = normalize_harness(harness)
        self.config = benchmark_config(self.repo_root, cell)
        self.root = self.run_root / "rollout"
        self.root.mkdir(parents=True, exist_ok=True)

    def run_from_main(self, *, main_decision: str | Path, label: str) -> RolloutResult:
        decision = json.loads(Path(main_decision).read_text(encoding="utf-8"))
        task_ids = tuple(str(item) for item in decision["rollout_request"]["task_ids"])
        minimum_task_count = _decision_minimum_task_count(decision)
        return self.run_version(
            task_ids=task_ids,
            harness_version=str(decision["harness_version"]),
            label=label,
            purpose=str(decision["rollout_request"]["rationale"]),
            minimum_task_count=minimum_task_count,
        )

    def run_channel_preflight_from_main(
        self, *, main_decision: str | Path, label: str
    ) -> RolloutResult:
        decision = json.loads(Path(main_decision).read_text(encoding="utf-8"))
        task_ids = tuple(str(item) for item in decision["rollout_request"]["task_ids"])
        if not task_ids:
            raise ValueError("channel preflight requires one TRAIN task")
        return self.run_version(
            task_ids=task_ids[:1],
            harness_version=str(decision["harness_version"]),
            label=label,
            purpose="Verify candidate channel visibility before the full TRAIN rollout.",
            minimum_task_count=1,
            repeats=PREFLIGHT_REPEAT_COUNT,
        )

    def run_pair_from_main(
        self,
        *,
        main_decision: str | Path,
        label: str,
        pairing_offset: int = 0,
        task_ids_override: Sequence[str] | None = None,
    ) -> PairedRolloutResult:
        decision = json.loads(Path(main_decision).read_text(encoding="utf-8"))
        task_ids = tuple(
            str(item)
            for item in (
                task_ids_override
                if task_ids_override is not None
                else decision["rollout_request"]["task_ids"]
            )
        )
        parent_version = str(decision["candidate"]["parent_version"])
        purpose = str(decision["rollout_request"]["rationale"])
        minimum_task_count = _decision_minimum_task_count(decision)
        if parent_version == "v0" and int(pairing_offset) == 0:
            candidate = self.run_version(
                task_ids=task_ids,
                harness_version=str(decision["harness_version"]),
                label=label,
                purpose=purpose,
                pairing_offset=pairing_offset,
                minimum_task_count=minimum_task_count,
            )
            candidate = self._retry_invalid_trials(
                result=candidate,
                harness_version=str(decision["harness_version"]),
                label=label,
                purpose=purpose,
                minimum_task_count=minimum_task_count,
                reserve=PAIRED_RETRY_REVIEW_RESERVE,
            )
            failures = rollout_infrastructure_failures(candidate)
            if failures:
                raise RuntimeError(
                    "candidate rollout remains infrastructure-invalid after retry: "
                    f"{len(failures)} invalid trial(s)"
                )
            return PairedRolloutResult(candidate=candidate, parent=None)

        trial_count = len(task_ids) * TRAIN_ROLLOUT_REPEATS
        total_concurrency = min(MAX_ROLLOUT_CONCURRENCY, trial_count * 2)
        parent_concurrency = max(1, total_concurrency // 2)
        candidate_concurrency = max(1, total_concurrency - parent_concurrency)
        def run_exact_pair(
            *, pair_label: str, pair_offset: int
        ) -> PairedRolloutResult:
            pair_parent_label = f"{pair_label}-parent-{parent_version}"
            required_creations = sum(
                trial_count
                for rollout_label in (pair_parent_label, pair_label)
                if not (self.root / f"{rollout_label}.json").is_file()
                and _active_rollout_job_id(
                    self.budget,
                    base_job_id=f"rollout-{rollout_label}",
                )
                is None
            )
            if self.budget.status()["remaining"] < required_creations:
                raise RuntimeError("cannot fund exact paired parent and candidate rollouts")
            with ThreadPoolExecutor(max_workers=2) as executor:
                parent_future = executor.submit(
                    self.run_version,
                    task_ids=task_ids,
                    harness_version=parent_version,
                    label=pair_parent_label,
                    purpose=f"Exact parent control for {purpose}",
                    pairing_offset=pair_offset,
                    max_concurrency=parent_concurrency,
                    minimum_task_count=minimum_task_count,
                )
                candidate_future = executor.submit(
                    self.run_version,
                    task_ids=task_ids,
                    harness_version=str(decision["harness_version"]),
                    label=pair_label,
                    purpose=purpose,
                    pairing_offset=pair_offset,
                    max_concurrency=candidate_concurrency,
                    minimum_task_count=minimum_task_count,
                )
                parent = parent_future.result()
                candidate = candidate_future.result()
            return PairedRolloutResult(candidate=candidate, parent=parent)

        pair = run_exact_pair(pair_label=label, pair_offset=pairing_offset)
        failures = paired_infrastructure_failures(pair)
        if failures:
            retry_cost = len(failures)
            if self.budget.status()["remaining"] >= (
                retry_cost + PAIRED_RETRY_REVIEW_RESERVE
            ):
                parent = self._retry_invalid_trials(
                    result=pair.parent,
                    harness_version=parent_version,
                    label=f"{label}-parent-{parent_version}",
                    purpose=f"Exact parent control for {purpose}",
                    minimum_task_count=minimum_task_count,
                    reserve=0,
                )
                candidate = self._retry_invalid_trials(
                    result=pair.candidate,
                    harness_version=str(decision["harness_version"]),
                    label=label,
                    purpose=purpose,
                    minimum_task_count=minimum_task_count,
                    reserve=0,
                )
                pair = PairedRolloutResult(candidate=candidate, parent=parent)
                failures = paired_infrastructure_failures(pair)
        return annotate_paired_validity(pair, failures=failures)

    def _retry_invalid_trials(
        self,
        *,
        result: RolloutResult | None,
        harness_version: str,
        label: str,
        purpose: str,
        minimum_task_count: int,
        reserve: int,
    ) -> RolloutResult | None:
        if result is None:
            return None
        for _ in range(MAX_TARGETED_RETRY_ROUNDS):
            failures = rollout_infrastructure_failures(result)
            targets = [
                failure
                for failure in failures
                if str(failure.get("task_id") or "") != "*"
                and int(failure.get("pairing_slot", -1)) >= 0
            ]
            if len(targets) != len(failures):
                return result
            if not targets or self.budget.status()["remaining"] < len(targets) + reserve:
                return result

            def retry_target(
                target: Mapping[str, Any],
            ) -> tuple[Mapping[str, Any], RolloutResult]:
                task_id = str(target["task_id"])
                slot = int(target["pairing_slot"])
                cached = _latest_valid_trial_retry(
                    self.root, label=label, task_id=task_id, pairing_slot=slot
                )
                if cached is not None:
                    return target, cached
                retry_label = _next_trial_retry_label(self.root, label, task_id, slot)
                retry = self.run_version(
                    task_ids=(task_id,),
                    harness_version=harness_version,
                    label=retry_label,
                    purpose=(
                        "Retry one infrastructure-invalid TRAIN trial without "
                        "replacing valid evidence."
                    ),
                    pairing_offset=slot,
                    minimum_task_count=1,
                    repeats=1,
                )
                return target, retry

            # Each retry has its own task/slot and can use the normal rollout cap.
            with ThreadPoolExecutor(
                max_workers=min(MAX_ROLLOUT_CONCURRENCY, len(targets))
            ) as executor:
                repairs = list(executor.map(retry_target, targets))
            result = _merge_trial_retries(result, repairs)
        return result

    def run_version(
        self,
        *,
        task_ids: Sequence[str],
        harness_version: str,
        label: str,
        purpose: str,
        pairing_offset: int = 0,
        max_concurrency: int | None = None,
        minimum_task_count: int = 5,
        repeats: int = TRAIN_ROLLOUT_REPEATS,
    ) -> RolloutResult:
        output_path = self.root / f"{label}.json"
        if output_path.exists():
            return RolloutResult(
                json.loads(output_path.read_text(encoding="utf-8")), str(output_path)
            )
        task_ids = tuple(str(item) for item in task_ids)
        validate_rollout_task_ids(
            task_ids,
            train_task_ids=self._train_task_ids(),
            minimum_task_count=minimum_task_count,
        )
        repeats = int(repeats)
        if repeats < 1 or repeats > TRAIN_ROLLOUT_REPEATS:
            raise ValueError(
                f"rollout repeats must be between 1 and {TRAIN_ROLLOUT_REPEATS}"
            )
        trial_count = len(task_ids) * repeats
        request = RolloutRequest(
            request_id=f"{self.run_root.name}-{label}",
            run_id=self.run_root.name,
            scope="TRAIN",
            harness_version=str(harness_version),
            task_repeats={
                task_id: repeats for task_id in task_ids
            },
            max_concurrency=(
                min(MAX_ROLLOUT_CONCURRENCY, trial_count)
                if max_concurrency is None
                else min(int(max_concurrency), trial_count)
            ),
            purpose=str(purpose),
            pairing_offsets={task_id: int(pairing_offset) for task_id in task_ids},
        )
        service = TrainRolloutService(
            cell=self.config.cell,
            repo_root=self.repo_root,
            run_id=self.run_root.name,
            artifact_root=self.run_root / "rollout_artifacts",
            train_task_ids=list(self._train_task_ids()),
            initial_budget=trial_count,
            evidence_root=self.run_root / "rollout_evidence",
            workspace_root=self.run_root / "rollout_workspaces",
            timeout_s=ROLLOUT_RUNTIME_LIMITS["trial_timeout_s"],
            local_rootless_rollout=self.config.local_rootless_rollout,
            harness=self.harness,
        )
        os.environ.setdefault("HAI_MIN_FREE_GB", "0")
        os.environ.setdefault("HAI_MIN_MEM_GB", "0")
        try:
            retained = service.recover_retained(request)
        except IncompleteRolloutTraceError as failure:
            retained = self._repair_incomplete_rollout(
                label=label,
                request=request,
                service=service,
                failure=failure,
            )
        if retained is not None:
            payload = retained.to_dict()
            payload["fixed_repeats"] = repeats
            payload["requested_task_ids"] = list(task_ids)
            payload["pairing_offset"] = int(pairing_offset)
            write_json(output_path, payload)
            _settle_recovered_rollout_job(
                self.budget,
                base_job_id=f"rollout-{label}",
                output_path=output_path,
                metrics=retained.metrics,
            )
            return RolloutResult(payload, str(output_path))
        base_job_id = f"rollout-{label}"
        job_id = _active_rollout_job_id(
            self.budget,
            base_job_id=base_job_id,
        )
        resumed_launched_job = job_id is not None
        if job_id is None:
            job_id = self.budget.next_attempt_id(base_job_id)
            self.budget.reserve_job(
                job_id,
                creation_count=trial_count,
                metadata={
                    "harness": self.harness,
                    "model": "deepseek/deepseek-v4-flash",
                    "task_ids": list(task_ids),
                    "repeats": repeats,
                    **ROLLOUT_RUNTIME_LIMITS,
                },
            )
            self.budget.claim_launch(job_id)
            self.budget.mark_launched(job_id)
        try:
            response = service.run(request)
        except IncompleteRolloutTraceError as failure:
            response = self._repair_incomplete_rollout(
                label=label,
                request=request,
                service=service,
                failure=failure,
            )
        except Exception as exc:
            self.budget.settle_job(job_id, outcome="failed", details=str(exc)[:2000])
            if "resource guard blocked" in str(exc) and not any(
                (self.run_root / "rollout_artifacts").rglob("*.jsonl")
            ):
                self.budget.correct_prelaunch_failure(
                    job_id,
                    evidence={
                        "intelligent_sessions_created": 0,
                        "reason": "resource guard rejected every trial before adapter launch",
                    },
                )
            raise
        if isinstance(response, RolloutResult):
            return response
        payload = response.to_dict()
        payload["fixed_repeats"] = repeats
        payload["requested_task_ids"] = list(task_ids)
        payload["pairing_offset"] = int(pairing_offset)
        write_json(output_path, payload)
        self.budget.settle_job(
            job_id,
            outcome="completed",
            details={
                "metrics": dict(response.metrics),
                "output": str(output_path),
                "resumed_launched_job": resumed_launched_job,
            },
        )
        return RolloutResult(payload, str(output_path))

    def _repair_incomplete_rollout(
        self,
        *,
        label: str,
        request: RolloutRequest,
        service: TrainRolloutService,
        failure: IncompleteRolloutTraceError,
    ):
        seen: set[Path] = set()
        current = failure
        while True:
            if current.trajectory_path in seen:
                raise current
            seen.add(current.trajectory_path)
            repair_job = self.budget.next_attempt_id(f"rollout-repair-{label}")
            self.budget.reserve_job(
                repair_job,
                creation_count=1,
                metadata={
                    "harness": self.harness,
                    "model": "deepseek/deepseek-v4-flash",
                    "trajectory_path": str(current.trajectory_path),
                    "purpose": "repair incomplete retained rollout trace",
                    **ROLLOUT_RUNTIME_LIMITS,
                },
            )
            self.budget.claim_launch(repair_job)
            self.budget.mark_launched(repair_job)
            repair_service = TrainRolloutService(
                cell=self.config.cell,
                repo_root=self.repo_root,
                run_id=self.run_root.name,
                artifact_root=self.run_root / "rollout_repair_artifacts",
                train_task_ids=list(self._train_task_ids()),
                initial_budget=1,
                evidence_root=self.run_root / "rollout_evidence",
                workspace_root=self.run_root / "rollout_workspaces",
                timeout_s=ROLLOUT_RUNTIME_LIMITS["trial_timeout_s"],
                local_rootless_rollout=self.config.local_rootless_rollout,
                harness=self.harness,
            )
            try:
                repaired = service.replace_incomplete_trial(
                    request=request,
                    failure=current,
                    repair_service=repair_service,
                )
            except IncompleteRolloutTraceError as next_failure:
                details = {
                    "trajectory_path": str(current.trajectory_path),
                    "next_incomplete_trajectory_path": str(next_failure.trajectory_path),
                }
                if "rollout_repair_artifacts" in str(next_failure.trajectory_path):
                    self.budget.settle_job(
                        repair_job, outcome="failed", details=details
                    )
                    raise
                self.budget.settle_job(
                    repair_job,
                    outcome="completed",
                    details=details,
                )
                current = next_failure
                continue
            except Exception as exc:
                self.budget.settle_job(
                    repair_job, outcome="failed", details=str(exc)[:2000]
                )
                raise
            self.budget.settle_job(
                repair_job,
                outcome="completed",
                details={"trajectory_path": str(current.trajectory_path)},
            )
            return repaired

    def _train_task_ids(self) -> tuple[str, ...]:
        payload = json.loads(
            (self.run_root / "experience" / "baseline_source_index.json").read_text(
                encoding="utf-8"
            )
        )
        return tuple(str(item) for item in payload["task_ids"])


def paired_infrastructure_failures(
    pair: PairedRolloutResult,
) -> tuple[dict[str, Any], ...]:
    if pair.parent is None:
        return ()
    candidate_by_task = pair.candidate.output.get("per_task")
    parent_by_task = pair.parent.output.get("per_task")
    failures: list[dict[str, Any]] = []
    for side, per_task in (
        ("candidate", candidate_by_task),
        ("parent", parent_by_task),
    ):
        if not isinstance(per_task, Mapping):
            failures.append(
                {
                    "task_id": "*",
                    "pairing_slot": -1,
                    "side": side,
                    "error": "missing per_task mapping",
                }
            )
        elif not per_task:
            failures.append(
                {
                    "task_id": "*",
                    "pairing_slot": -1,
                    "side": side,
                    "error": "empty per_task mapping",
                }
            )
    if failures:
        return tuple(failures)
    assert isinstance(candidate_by_task, Mapping)
    assert isinstance(parent_by_task, Mapping)
    task_ids = sorted(set(candidate_by_task) | set(parent_by_task))
    for task_id in task_ids:
        candidate_trials = _trial_summaries_by_slot(
            candidate_by_task.get(task_id) or {}
        )
        parent_trials = _trial_summaries_by_slot(parent_by_task.get(task_id) or {})
        for slot in sorted(set(candidate_trials) | set(parent_trials)):
            for side, trials in (
                ("candidate", candidate_trials),
                ("parent", parent_trials),
            ):
                summary = trials.get(slot)
                if summary is None or _trial_is_infrastructure_failure(summary):
                    failures.append(
                        {
                            "task_id": str(task_id),
                            "pairing_slot": int(slot),
                            "side": side,
                            "error": (
                                "missing paired trial"
                                if summary is None
                                else str(summary.get("error") or "infrastructure error")
                            ),
                        }
                    )
    return tuple(failures)


def rollout_infrastructure_failures(
    result: RolloutResult,
) -> tuple[dict[str, Any], ...]:
    per_task = result.output.get("per_task")
    if not isinstance(per_task, Mapping) or not per_task:
        return (
            {
                "task_id": "*",
                "pairing_slot": -1,
                "side": "candidate",
                "error": "missing per_task mapping",
            },
        )
    failures: list[dict[str, Any]] = []
    for task_id, task in per_task.items():
        if not isinstance(task, Mapping):
            failures.append(
                {
                    "task_id": str(task_id),
                    "pairing_slot": -1,
                    "side": "candidate",
                    "error": "invalid task rollout payload",
                }
            )
            continue
        summaries = task.get("trial_summaries") or []
        for index, summary in enumerate(summaries):
            if not isinstance(summary, Mapping) or _trial_is_infrastructure_failure(summary):
                failures.append(
                    {
                        "task_id": str(task_id),
                        "pairing_slot": int(
                            summary.get("pairing_slot", summary.get("trial", index))
                            if isinstance(summary, Mapping)
                            else index
                        ),
                        "side": "candidate",
                        "error": (
                            str(summary.get("error") or "infrastructure error")
                            if isinstance(summary, Mapping)
                            else "invalid trial summary"
                        ),
                    }
                )
    return tuple(failures)


def _next_infrastructure_retry_label(root: Path, label: str) -> str:
    attempt = 1
    while (root / f"{label}-infra-retry-{attempt:02d}.json").is_file():
        attempt += 1
    return f"{label}-infra-retry-{attempt:02d}"


def _trial_retry_label(label: str, task_id: str, pairing_slot: int) -> str:
    safe_task = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(task_id)
    )
    return f"{label}-infra-trial-retry-{safe_task}-s{int(pairing_slot):02d}"


def _next_trial_retry_label(
    root: Path, label: str, task_id: str, pairing_slot: int
) -> str:
    base = _trial_retry_label(label, task_id, pairing_slot)
    if not (root / f"{base}.json").is_file():
        return base
    attempt = 2
    while (root / f"{base}-attempt-{attempt:02d}.json").is_file():
        attempt += 1
    return f"{base}-attempt-{attempt:02d}"


def _latest_valid_trial_retry(
    root: Path, *, label: str, task_id: str, pairing_slot: int
) -> RolloutResult | None:
    """Reuse a completed one-trial retry before charging another attempt."""
    base = _trial_retry_label(label, task_id, pairing_slot)
    candidates = sorted(root.glob(f"{base}*.json"), reverse=True)
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        task = (payload.get("per_task") or {}).get(str(task_id))
        if not isinstance(task, Mapping):
            continue
        summary = _trial_summary_for_slot(task, pairing_slot)
        if summary is not None and not _trial_is_infrastructure_failure(summary):
            return RolloutResult(payload, str(path))
    return None


def _merge_trial_retries(
    initial: RolloutResult,
    repairs: Sequence[tuple[Mapping[str, Any], RolloutResult]],
) -> RolloutResult:
    """Replace only invalid slots and retain every valid initial trial."""
    payload = json.loads(json.dumps(initial.output))
    per_task = payload.get("per_task")
    if not isinstance(per_task, dict):
        return initial
    retry_audit = [
        dict(item)
        for item in payload.get("infrastructure_trial_retries") or []
        if isinstance(item, Mapping)
    ]
    replacement_count = 0
    for target, retry in repairs:
        task_id = str(target["task_id"])
        slot = int(target["pairing_slot"])
        original_task = per_task.get(task_id)
        retry_task = (retry.output.get("per_task") or {}).get(task_id)
        if not isinstance(original_task, dict) or not isinstance(retry_task, Mapping):
            continue
        replacement = _trial_summary_for_slot(retry_task, slot)
        if replacement is None:
            continue
        original_summaries = list(original_task.get("trial_summaries") or [])
        index = _trial_summary_index_for_slot(original_summaries, slot)
        if index is None:
            continue
        original_summaries[index] = dict(replacement)
        original_task["trial_summaries"] = original_summaries
        for path_key in ("trajectory_paths", "source_trajectory_paths"):
            retry_paths = list(retry_task.get(path_key) or [])
            original_paths = list(original_task.get(path_key) or [])
            if retry_paths and index < len(original_paths):
                original_paths[index] = retry_paths[0]
                original_task[path_key] = original_paths
        retry_audit.append(
            {
                "task_id": task_id,
                "pairing_slot": slot,
                "retry_output": retry.output_path,
                "original_error": str(target.get("error") or ""),
            }
        )
        replacement_count += 1

    _refresh_rollout_payload_metrics(payload)
    payload["infrastructure_trial_retries"] = retry_audit
    payload["budget_spent"] = int(payload.get("budget_spent") or 0) + replacement_count
    write_json(initial.output_path, payload)
    return RolloutResult(payload, initial.output_path)


def _trial_summary_for_slot(
    task: Mapping[str, Any], pairing_slot: int
) -> Mapping[str, Any] | None:
    for index, summary in enumerate(task.get("trial_summaries") or []):
        if not isinstance(summary, Mapping):
            continue
        slot = int(summary.get("pairing_slot", summary.get("trial", index)))
        if slot == int(pairing_slot):
            return summary
    summaries = [
        summary
        for summary in task.get("trial_summaries") or []
        if isinstance(summary, Mapping)
    ]
    if len(summaries) == 1:
        # A one-trial retry is emitted as local trial/slot zero by native
        # runners; preserve the requested original slot when merging it back.
        normalized = dict(summaries[0])
        normalized["pairing_slot"] = int(pairing_slot)
        return normalized
    return None


def _trial_summary_index_for_slot(
    summaries: Sequence[Any], pairing_slot: int
) -> int | None:
    for index, summary in enumerate(summaries):
        if isinstance(summary, Mapping) and int(
            summary.get("pairing_slot", summary.get("trial", index))
        ) == int(pairing_slot):
            return index
    return None


def _refresh_rollout_payload_metrics(payload: dict[str, Any]) -> None:
    per_task = payload.get("per_task") or {}
    records: list[dict[str, Any]] = []
    rewards: list[float] = []
    failures = 0
    worker_errors = 0
    pass_at_2_total = 0
    for task_id, task in per_task.items():
        if not isinstance(task, dict):
            continue
        summaries = [
            dict(summary)
            for summary in task.get("trial_summaries") or []
            if isinstance(summary, Mapping)
        ]
        task_rewards = [float(summary.get("reward", 0.0) or 0.0) for summary in summaries]
        task["rewards"] = task_rewards
        task_errors = [
            {"trial": summary.get("trial", index), "error": str(summary.get("error") or "")}
            for index, summary in enumerate(summaries)
            if str(summary.get("error") or "").strip()
        ]
        task["worker_errors"] = task_errors
        rewards.extend(task_rewards)
        worker_errors += len(task_errors)
        failures += sum(
            _trial_is_infrastructure_failure(summary) for summary in summaries
        )
        pass_at_2_total += int(any(value >= 1.0 for value in task_rewards[:2]))
        records.append(
            {
                "task_id": str(task_id),
                "rewards": task_rewards,
                "harness_version": str(payload.get("harness_version") or ""),
                "trajectory_paths": list(task.get("trajectory_paths") or []),
                "worker_errors": task_errors,
                "trial_summaries": summaries,
                "pass_at_1": int(any(value >= 1.0 for value in task_rewards[:1])),
                "pass_at_2": int(any(value >= 1.0 for value in task_rewards[:2])),
            }
        )
    metrics = dict(payload.get("metrics") or {})
    trials = len(rewards)
    tasks = len(records)
    successes = sum(value >= 1.0 for value in rewards)
    metrics.update(
        {
            "task_count": tasks,
            "trial_count": trials,
            "trial_success_count": successes,
            "trial_success_rate": successes / trials if trials else 0.0,
            "pass_at_1": successes / trials if trials else 0.0,
            "pass_at_2": pass_at_2_total / tasks if tasks else 0.0,
            "worker_error_count": worker_errors,
            "infrastructure_failure_count": failures,
            "charged_trial_count": trials - failures,
        }
    )
    payload["metrics"] = metrics
    payload["records"] = records


def annotate_paired_validity(
    pair: PairedRolloutResult,
    *,
    failures: Sequence[Mapping[str, Any]],
) -> PairedRolloutResult:
    if pair.parent is None:
        return pair

    def annotate(result: RolloutResult) -> RolloutResult:
        payload = dict(result.output)
        metrics = dict(payload.get("metrics") or {})
        metrics["paired_infrastructure_valid"] = not failures
        metrics["paired_infrastructure_failure_count"] = len(failures)
        payload["metrics"] = metrics
        payload["paired_infrastructure_failures"] = [
            dict(item) for item in failures
        ]
        write_json(result.output_path, payload)
        return RolloutResult(payload, result.output_path)

    return PairedRolloutResult(
        candidate=annotate(pair.candidate),
        parent=annotate(pair.parent),
    )


def _trial_summaries_by_slot(task: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for index, summary in enumerate(task.get("trial_summaries") or []):
        if not isinstance(summary, Mapping):
            continue
        slot = int(summary.get("pairing_slot", summary.get("trial", index)))
        result[slot] = summary
    return result


def _trial_is_infrastructure_failure(summary: Mapping[str, Any]) -> bool:
    if not str(summary.get("error") or "").strip():
        return str(summary.get("termination") or "") in {"error", "timeout"}
    # The native batch service treats errors after an interaction as scored
    # behavioral/evaluator failures. Keep rollout-level retry semantics aligned
    # so those trajectories are not replaced as infrastructure faults.
    return (
        int(summary.get("n_messages") or 0) == 0
        and int(summary.get("n_tool_calls") or 0) == 0
    )


def validate_rollout_task_ids(
    task_ids: Sequence[str],
    *,
    train_task_ids: Sequence[str],
    minimum_task_count: int = 5,
) -> None:
    normalized = [str(item) for item in task_ids]
    minimum = int(minimum_task_count)
    if len(normalized) < minimum or len(normalized) != len(set(normalized)):
        word = "five" if minimum == 5 else str(minimum)
        raise ValueError(
            f"rollout requires at least {word} distinct task IDs"
        )
    outside = set(normalized) - set(str(item) for item in train_task_ids)
    if outside:
        raise ValueError(f"rollout tasks outside TRAIN: {sorted(outside)}")


def _decision_minimum_task_count(decision: Mapping[str, Any]) -> int:
    return 2 if str(decision.get("evaluation_mode") or "") == "residual_probe" else 5


def _settle_recovered_rollout_job(
    budget: CreationBudget,
    *,
    base_job_id: str,
    output_path: Path,
    metrics: Mapping[str, Any],
) -> None:
    state = read_json(budget.path)
    jobs = (state or {}).get("jobs") or {}
    active = [
        str(job_id)
        for job_id, item in jobs.items()
        if (
            job_id == base_job_id or str(job_id).startswith(f"{base_job_id}-retry-")
        )
        and isinstance(item, Mapping)
        and item.get("status") == "launched"
    ]
    if not active:
        return
    for job_id in active:
        budget.settle_job(
            job_id,
            outcome="completed",
            details={
                "metrics": dict(metrics),
                "output": str(output_path),
                "recovered_from_retained_trials": True,
            },
        )


def _active_rollout_job_id(
    budget: CreationBudget,
    *,
    base_job_id: str,
) -> str | None:
    state = read_json(budget.path)
    jobs = (state or {}).get("jobs") or {}
    active = [
        str(job_id)
        for job_id, item in jobs.items()
        if (
            job_id == base_job_id or str(job_id).startswith(f"{base_job_id}-retry-")
        )
        and isinstance(item, Mapping)
        and item.get("status") == "launched"
    ]
    if len(active) > 1:
        raise RuntimeError(f"multiple active rollout jobs for {base_job_id}: {active}")
    return active[0] if active else None
