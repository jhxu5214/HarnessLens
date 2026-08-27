from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from harnesslens.core.artifacts import write_json
from harnesslens.benchmarks.benchmark_splits import BenchmarkSplit, load_benchmark_split
from harnesslens.benchmarks.bird_eval import BirdLimits
from harnesslens.infrastructure.clash_proxy import configure_terminal_clash_proxy
from harnesslens.harnesses.harness_manifest import normalize_harness
from harnesslens.infrastructure.rootless_docker import (
    DEFAULT_DOCKER_HOST,
    ensure_rootless_docker,
)
from harnesslens.benchmarks.terminal_images import (
    TERMINAL_BENCH_IMAGE_TEMPLATE,
    ensure_terminal_shared_network,
    require_preloaded_terminal_images,
)
from harnesslens.evaluation.rollout_bridge import (
    MAX_CONCURRENCY,
    BIRD_CELL,
    RolloutRequest,
    RolloutResponse,
    TrainRolloutRecord,
    CellHarnessRepository,
    TrainRolloutService,
    _infrastructure_failure_count,
    _record_from_mapping,
    _summarize,
    _api_trace_required,
    _trajectory_retention,
    _write_json,
    retain_trial_trajectories,
    validate_bird_rollout_interactions,
    validate_native_rollout_interactions,
    validate_rollout_interactions,
)


# Paper results use one fresh TEST trial per task (held-out pass@1).
BASELINE_REPEATS = 1
BASELINE_MAX_CONCURRENCY = MAX_CONCURRENCY
TERMINAL_BASELINE_MAX_CONCURRENCY = 10
BASELINE_GROUP_TIMEOUT_S = 7200
TAU2_AGENT_STEPS_PER_TURN = 10
TAU2_MAX_CONVERSATION_TURNS = 40
TAU2_TIMEOUT_PER_TURN_S = 180
TERMINAL_AGENT_STEPS = 50
TERMINAL_EXEC_TIMEOUT_S = 600
TERMINAL_VERIFY_TIMEOUT_S = 1800


@dataclass(frozen=True)
class BlindTestBaselineResult:
    benchmark: str
    output_path: str
    response: RolloutResponse


@dataclass(frozen=True)
class BlindTestCandidateResult:
    benchmark: str
    output_path: str
    response: RolloutResponse
    harness_version: str


class BlindTestRolloutService(TrainRolloutService):
    """Additive TEST-only service; the retail TRAIN rollout class is unchanged."""

    def __init__(
        self,
        *,
        direct_model_network: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.direct_model_network = bool(direct_model_network)

    def run(self, request: RolloutRequest) -> RolloutResponse:
        self._validate(request)
        request_root = self.artifact_root / request.run_id / request.request_id
        cached = self._load_cached(request, request_root)
        if cached is not None and int(
            cached.metrics.get("infrastructure_failure_count", 0) or 0
        ) == 0:
            self.remaining_budget = cached.budget_remaining
            return cached
        repeats = int(next(iter(request.task_repeats.values())))
        task_ids = sorted(str(task_id) for task_id in request.task_repeats)
        records: list[TrainRolloutRecord] = []
        per_task: dict[str, dict[str, Any]] = {}
        raw = self._run_group(request, task_ids, repeats)
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
        requested = sum(int(value) for value in request.task_repeats.values())
        if self.cell == "terminal_bench":
            _validate_terminal_trajectories(records)
        elif self.cell == BIRD_CELL:
            validate_bird_rollout_interactions(records)
        elif self.harness in {"pi", "codex"}:
            validate_native_rollout_interactions(records, harness=self.harness)
        else:
            validate_rollout_interactions(records)
        infrastructure = _infrastructure_failure_count(records)
        charged = max(0, requested - infrastructure)
        self.remaining_budget -= charged
        metrics = {
            **_summarize(records),
            "requested_trial_count": requested,
            "infrastructure_failure_count": infrastructure,
            "charged_trial_count": charged,
            "workspace_cleanup_enabled": os.environ.get(
                "HAI_KEEP_TRAJECTORY_WORKSPACE", "0"
            ).lower()
            not in {"1", "true", "yes", "on"},
            "trajectory_retention": _trajectory_retention(self.cell),
            "api_trace_required": _api_trace_required(self.cell),
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

    def _validate(self, request: RolloutRequest) -> None:
        if request.scope != "TEST":
            raise ValueError("test baseline rollout only accepts TEST")
        if not request.task_repeats:
            raise ValueError("rollout must select at least one task")
        if len({int(value) for value in request.task_repeats.values()}) != 1:
            raise ValueError("TEST rollout tasks must use one shared repeat count")
        if not 1 <= int(request.max_concurrency) <= MAX_CONCURRENCY:
            raise ValueError("rollout concurrency is outside the supported range")
        if set(map(str, request.task_repeats)) - self.train_task_ids:
            raise ValueError("rollout selected a task outside TEST")
        if set(map(str, request.pairing_offsets)) - set(map(str, request.task_repeats)):
            raise ValueError("pairing offset references an unselected task")
        if any(int(value) < 0 for value in request.pairing_offsets.values()):
            raise ValueError("pairing offsets must be nonnegative")
        if sum(int(value) for value in request.task_repeats.values()) > self.remaining_budget:
            raise ValueError("rollout request exceeds its allocated budget")

    def _run_group(
        self, request: RolloutRequest, task_ids: list[str], repeats: int
    ) -> dict[str, Any]:
        return super()._run_group(request, task_ids, repeats)


def run_test_baseline(
    *,
    repo_root: str | Path,
    run_id: str,
    benchmark: str,
    retrieval_config: str = "bm25",
    task_ids: Sequence[str] | None = None,
    repeats: int = BASELINE_REPEATS,
    harness: str = "opencode",
) -> BlindTestBaselineResult:
    root = Path(repo_root).resolve()
    split = load_benchmark_split(benchmark)
    target_harness = normalize_harness(harness)
    effective_repeats = _validated_test_repeats(repeats)
    selected_test = tuple(str(task_id) for task_id in (task_ids or split.test))
    if not selected_test:
        raise ValueError("at least one TEST task must be selected")
    unknown = set(selected_test) - set(split.test)
    if unknown:
        raise ValueError("selected task is outside the canonical TEST split")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is required for a HarnessLens test baseline")
    normalized_retrieval = str(retrieval_config).strip()
    if split.cell == "banking_knowledge":
        if not normalized_retrieval:
            raise ValueError("banking retrieval_config must be non-empty")
        os.environ["HAI_TAU2_RETRIEVAL_CONFIG"] = normalized_retrieval
    image_preflight: Mapping[str, object] | None = None
    if split.cell == "terminal_bench":
        ensure_rootless_docker(
            str(os.environ.get("DOCKER_HOST") or DEFAULT_DOCKER_HOST)
        )
        os.environ["TB_IMAGE_TEMPLATE"] = TERMINAL_BENCH_IMAGE_TEMPLATE
        os.environ["TB_NO_REBUILD"] = "1"
        image_preflight = require_preloaded_terminal_images(selected_test).to_dict()
        os.environ["TB_SHARED_NETWORK"] = ensure_terminal_shared_network()
    run_root = (
        root
        / "runs"
        / "test_baselines"
        / str(run_id)
        / target_harness
    )
    output_path = run_root / "baseline_result.json"
    if output_path.exists():
        cached = json.loads(output_path.read_text(encoding="utf-8"))
        if cached.get("benchmark") != split.benchmark:
            raise RuntimeError("existing HarnessLens test baseline belongs to another benchmark")
    trial_count = len(selected_test) * effective_repeats
    service = BlindTestRolloutService(
        cell=split.cell,
        repo_root=root,
        run_id=str(run_id),
        artifact_root=run_root / "rollout_artifacts",
        train_task_ids=list(selected_test),
        initial_budget=trial_count,
        evidence_root=run_root / "rollout_evidence",
        workspace_root=run_root / "rollout_workspaces",
        timeout_s=(
            BirdLimits().group_timeout_s
            if split.cell == BIRD_CELL
            else BASELINE_GROUP_TIMEOUT_S
        ),
        local_rootless_rollout=split.local_rootless_rollout,
        direct_model_network=True,
        harness=target_harness,
    )
    os.environ.setdefault("HAI_MIN_FREE_GB", "0")
    os.environ.setdefault("HAI_MIN_MEM_GB", "0")
    os.environ.setdefault("HAI_TAU2_LLM_MODEL", "openai/deepseek-v4-flash")
    os.environ.setdefault("TB_OPENCODE_MODEL", "deepseek/deepseek-v4-flash")
    if split.cell == "terminal_bench":
        configure_terminal_clash_proxy()
    response = service.run(
        RolloutRequest(
            request_id=f"{run_id}-{target_harness}-{split.cell}-test-baseline",
            run_id=str(run_id),
            scope="TEST",
            harness_version="v0",
            task_repeats={task_id: effective_repeats for task_id in selected_test},
            max_concurrency=min(baseline_max_concurrency(split), trial_count),
            purpose=f"harnesslens_test_pass_at_{effective_repeats}_baseline",
            pairing_offsets={task_id: 0 for task_id in selected_test},
        )
    )
    payload: dict[str, Any] = {
        "schema": "harnesslens.test-baseline.v1",
        "benchmark": split.benchmark,
        "cell": split.cell,
        "scope": "TEST",
        "harness": target_harness,
        "harness_version": "v0",
        "split_fingerprint": split.fingerprint(),
        "train_task_count": len(split.train),
        "test_task_count": len(split.test),
        "selected_test_task_count": len(selected_test),
        "test_task_ids": list(selected_test),
        "repeats": effective_repeats,
        "runtime_limits": runtime_limits(split, repeats=effective_repeats),
        "model_network": "direct",
        "retrieval_config": (
            normalized_retrieval if split.cell == "banking_knowledge" else None
        ),
        "docker_host_proxy": "inherited" if split.cell == "terminal_bench" else "disabled",
        "terminal_image_preflight": image_preflight,
        "terminal_shared_network": (
            os.environ.get("TB_SHARED_NETWORK") if split.cell == "terminal_bench" else None
        ),
        "response": response.to_dict(),
    }
    write_json(output_path, payload)
    return BlindTestBaselineResult(split.benchmark, str(output_path), response)


def run_test_candidate(
    *,
    repo_root: str | Path,
    run_id: str,
    benchmark: str,
    patch_json: str | Path,
    patch_descs_json: str | Path | None = None,
    candidate_label: str = "candidate",
    base_version: str = "v0",
    retrieval_config: str = "bm25",
    task_ids: Sequence[str] | None = None,
    max_concurrency: int | None = None,
    repeats: int = BASELINE_REPEATS,
    harness: str = "opencode",
) -> BlindTestCandidateResult:
    root = Path(repo_root).resolve()
    split = load_benchmark_split(benchmark)
    target_harness = normalize_harness(harness)
    effective_repeats = _validated_test_repeats(repeats)
    selected_test = tuple(str(task_id) for task_id in (task_ids or split.test))
    if not selected_test:
        raise ValueError("at least one TEST task must be selected")
    unknown = set(selected_test) - set(split.test)
    if unknown:
        raise ValueError("selected task is outside the canonical TEST split")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is required for a HarnessLens test candidate")
    normalized_retrieval = str(retrieval_config).strip()
    if split.cell == "banking_knowledge":
        if not normalized_retrieval:
            raise ValueError("banking retrieval_config must be non-empty")
        os.environ["HAI_TAU2_RETRIEVAL_CONFIG"] = normalized_retrieval

    run_root = (
        root
        / "runs"
        / "test_candidates"
        / str(run_id)
        / target_harness
    )
    evidence_root = run_root / "rollout_evidence"
    repository = CellHarnessRepository(
        cell=split.cell,
        repo_root=root,
        run_id=str(run_id),
        evidence_root=evidence_root,
        harness=target_harness,
    )
    workspace_submission = _load_workspace_submission(
        patch_json,
        repo_root=root,
        harness=target_harness,
    )
    if workspace_submission is not None:
        if patch_descs_json is not None:
            raise ValueError(
                "patch_descs_json cannot be combined with a HarnessLens workspace submission"
            )
        harness_version = repository.materialize_workspace_candidate(
            base_version=base_version,
            candidate_label=candidate_label,
            workspace=workspace_submission["workspace"],
            manifest_delta=workspace_submission["manifest"],
        )
    else:
        delta = _load_candidate_delta(patch_json, patch_descs_json)
        harness_version = repository.materialize_candidate(
            base_version=base_version,
            candidate_label=candidate_label,
            delta=delta,
        )

    image_preflight: Mapping[str, object] | None = None
    if split.cell == "terminal_bench":
        ensure_rootless_docker(
            str(os.environ.get("DOCKER_HOST") or DEFAULT_DOCKER_HOST)
        )
        os.environ["TB_IMAGE_TEMPLATE"] = TERMINAL_BENCH_IMAGE_TEMPLATE
        os.environ["TB_NO_REBUILD"] = "1"
        image_preflight = require_preloaded_terminal_images(selected_test).to_dict()
        os.environ["TB_SHARED_NETWORK"] = ensure_terminal_shared_network()

    trial_count = len(selected_test) * effective_repeats
    effective_max_concurrency = min(
        baseline_max_concurrency(split),
        trial_count,
        int(max_concurrency) if max_concurrency is not None else trial_count,
    )
    if effective_max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    service = BlindTestRolloutService(
        cell=split.cell,
        repo_root=root,
        run_id=str(run_id),
        artifact_root=run_root / "rollout_artifacts",
        train_task_ids=list(selected_test),
        initial_budget=trial_count,
        evidence_root=evidence_root,
        workspace_root=run_root / "rollout_workspaces",
        timeout_s=(
            BirdLimits().group_timeout_s
            if split.cell == BIRD_CELL
            else BASELINE_GROUP_TIMEOUT_S
        ),
        local_rootless_rollout=split.local_rootless_rollout,
        direct_model_network=True,
        harness=target_harness,
    )
    os.environ.setdefault("HAI_MIN_FREE_GB", "0")
    os.environ.setdefault("HAI_MIN_MEM_GB", "0")
    os.environ.setdefault("HAI_TAU2_LLM_MODEL", "openai/deepseek-v4-flash")
    os.environ.setdefault("TB_OPENCODE_MODEL", "deepseek/deepseek-v4-flash")
    if split.cell == "terminal_bench":
        configure_terminal_clash_proxy()
    response = service.run(
        RolloutRequest(
            request_id=(
                f"{run_id}-{target_harness}-{split.cell}-test-{harness_version}"
            ),
            run_id=str(run_id),
            scope="TEST",
            harness_version=harness_version,
            task_repeats={task_id: effective_repeats for task_id in selected_test},
            max_concurrency=effective_max_concurrency,
            purpose=f"harnesslens_test_pass_at_{effective_repeats}_candidate",
            pairing_offsets={task_id: 0 for task_id in selected_test},
        )
    )
    output_path = run_root / "candidate_result.json"
    payload: dict[str, Any] = {
        "schema": "harnesslens.test-candidate.v1",
        "benchmark": split.benchmark,
        "cell": split.cell,
        "scope": "TEST",
        "harness": target_harness,
        "base_version": str(base_version),
        "harness_version": harness_version,
        "candidate_label": str(candidate_label),
        "patch_json": str(Path(patch_json).resolve()),
        "patch_descs_json": (
            str(Path(patch_descs_json).resolve()) if patch_descs_json else None
        ),
        "split_fingerprint": split.fingerprint(),
        "train_task_count": len(split.train),
        "test_task_count": len(split.test),
        "selected_test_task_count": len(selected_test),
        "test_task_ids": list(selected_test),
        "repeats": effective_repeats,
        "runtime_limits": runtime_limits(split, repeats=effective_repeats),
        "max_concurrency": effective_max_concurrency,
        "model_network": "direct",
        "retrieval_config": (
            normalized_retrieval if split.cell == "banking_knowledge" else None
        ),
        "docker_host_proxy": "inherited" if split.cell == "terminal_bench" else "disabled",
        "terminal_image_preflight": image_preflight,
        "terminal_shared_network": (
            os.environ.get("TB_SHARED_NETWORK") if split.cell == "terminal_bench" else None
        ),
        "response": response.to_dict(),
    }
    write_json(output_path, payload)
    return BlindTestCandidateResult(split.benchmark, str(output_path), response, harness_version)


def _load_candidate_delta(
    patch_json: str | Path, patch_descs_json: str | Path | None
) -> dict[str, Any]:
    patch_path = Path(patch_json)
    delta = json.loads(patch_path.read_text(encoding="utf-8"))
    if not isinstance(delta, dict):
        raise ValueError("patch_json must contain a JSON object")
    if patch_descs_json is not None:
        desc_path = Path(patch_descs_json)
        descs = json.loads(desc_path.read_text(encoding="utf-8"))
        if not isinstance(descs, dict):
            raise ValueError("patch_descs_json must contain a JSON object")
        merged = dict(delta)
        merged["tool_desc_patches"] = descs
        return merged
    return dict(delta)


def _load_workspace_submission(
    patch_json: str | Path,
    *,
    repo_root: str | Path,
    harness: str,
) -> dict[str, Any] | None:
    patch_path = Path(patch_json).resolve()
    payload = json.loads(patch_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("patch_json must contain a JSON object")
    if "snapshot_path" not in payload or "selected_version" not in payload:
        return None
    root = Path(repo_root).resolve()
    snapshot = Path(str(payload.get("snapshot_path") or "")).resolve()
    allowed_root = (root / "runs").resolve()
    try:
        snapshot.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError("submission snapshot must stay under runs/") from exc
    if str(payload.get("selected_version") or "") == "v0":
        query_path = patch_path.parent.parent / "discovery" / "harness_query.json"
        if query_path.is_symlink() or not query_path.is_file():
            raise ValueError("v0 submission is missing its Harness Query result")
        query = json.loads(query_path.read_text(encoding="utf-8"))
        if not isinstance(query, Mapping) or str(query.get("harness") or "") != str(harness):
            raise ValueError("submission snapshot harness mismatch")
        return {"workspace": {"schema": 1, "files": []}, "manifest": {}}
    meta_path = snapshot / "meta.json"
    native_root = snapshot / "harness" / str(harness)
    workspace_path = native_root / "workspace.json"
    manifest_path = native_root / "manifest.json"
    for required in (meta_path, workspace_path, manifest_path):
        if required.is_symlink() or not required.is_file():
            raise ValueError(f"submission snapshot is incomplete: {required}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, Mapping) or str(meta.get("harness") or "") != str(harness):
        raise ValueError("submission snapshot harness mismatch")
    if str(meta.get("version") or "") != str(payload.get("selected_version") or ""):
        raise ValueError("submission selected version mismatch")
    workspace = json.loads(workspace_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(workspace, Mapping) or not isinstance(manifest, Mapping):
        raise ValueError("submission workspace and manifest must be JSON objects")
    return {"workspace": dict(workspace), "manifest": dict(manifest)}


def split_summary(split: BenchmarkSplit) -> Mapping[str, Any]:
    return {
        "benchmark": split.benchmark,
        "cell": split.cell,
        "train_task_count": len(split.train),
        "test_task_count": len(split.test),
        "split_fingerprint": split.fingerprint(),
    }


def runtime_limits(
    split: BenchmarkSplit,
    *,
    repeats: int = BASELINE_REPEATS,
) -> Mapping[str, Any]:
    effective_repeats = _validated_test_repeats(repeats)
    if split.cell == BIRD_CELL:
        limits = BirdLimits()
        return {
            "repeats": effective_repeats,
            "max_concurrency": baseline_max_concurrency(split),
            **limits.to_dict(),
        }
    common = {
        "repeats": effective_repeats,
        "max_concurrency": baseline_max_concurrency(split),
        "group_timeout_s": BASELINE_GROUP_TIMEOUT_S,
    }
    if split.cell == "terminal_bench":
        return {
            **common,
            "agent_steps": TERMINAL_AGENT_STEPS,
            "exec_timeout_s": TERMINAL_EXEC_TIMEOUT_S,
            "verify_timeout_s": TERMINAL_VERIFY_TIMEOUT_S,
        }
    return {
        **common,
        "agent_steps_per_turn": TAU2_AGENT_STEPS_PER_TURN,
        "max_conversation_turns": TAU2_MAX_CONVERSATION_TURNS,
        "timeout_per_turn_s": TAU2_TIMEOUT_PER_TURN_S,
    }


def baseline_max_concurrency(split: BenchmarkSplit) -> int:
    return (
        TERMINAL_BASELINE_MAX_CONCURRENCY
        if split.cell == "terminal_bench"
        else BASELINE_MAX_CONCURRENCY
    )


def _validated_test_repeats(value: int) -> int:
    repeats = int(value)
    if repeats not in {1, 2}:
        raise ValueError("TEST repeats must be 1 or 2")
    return repeats


def _validate_terminal_trajectories(records: list[TrainRolloutRecord]) -> None:
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
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip()
            ]
            if len(rows) != 1 or not isinstance(rows[0], Mapping):
                raise RuntimeError(f"retained terminal trajectory is malformed: {path}")
            if str(rows[0].get("task_id") or "") != record.task_id:
                raise RuntimeError(f"retained terminal trajectory task mismatch: {path}")
