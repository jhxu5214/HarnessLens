from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from harnesslens.benchmarks.benchmark_splits import load_benchmark_split
from harnesslens.evaluation.rollout_bridge import RolloutRequest, TrainRolloutService
from harnesslens.evaluation.blind_test_eval import BlindTestRolloutService
from harnesslens.evaluation.blind_test_eval import (
    _load_workspace_submission,
    run_test_baseline,
    run_test_candidate,
    runtime_limits,
)


def _service(cls, tmp_path: Path, *, cell: str, task_ids: list[str], local_rootless: bool):
    return cls(
        cell=cell,
        repo_root=tmp_path,
        run_id="run",
        artifact_root=tmp_path / "artifacts",
        train_task_ids=task_ids,
        initial_budget=2,
        evidence_root=tmp_path / "evidence",
        workspace_root=tmp_path / "workspaces",
        local_rootless_rollout=local_rootless,
    )


@pytest.mark.parametrize(
    ("cell", "harness"),
    [
        ("retail", "opencode"),
        ("retail", "pi"),
        ("retail", "codex"),
        ("banking_knowledge", "opencode"),
        ("banking_knowledge", "pi"),
        ("banking_knowledge", "codex"),
        ("terminal_bench", "opencode"),
        ("terminal_bench", "pi"),
        ("terminal_bench", "codex"),
        ("bird_mini_dev_challenging", "opencode"),
        ("bird_mini_dev_challenging", "pi"),
        ("bird_mini_dev_challenging", "codex"),
    ],
)
def test_test_service_uses_shared_multiharness_runner(
    tmp_path, monkeypatch, cell, harness
):
    task_id = "task"
    service = BlindTestRolloutService(
        cell=cell,
        repo_root=tmp_path,
        run_id="run",
        artifact_root=tmp_path / "artifacts",
        train_task_ids=[task_id],
        initial_budget=1,
        evidence_root=tmp_path / "evidence",
        workspace_root=tmp_path / "workspaces",
        harness=harness,
        local_rootless_rollout=False,
    )
    captured = {}

    def fake_shared(self, request, task_ids, repeats):
        captured.update(
            harness=self.harness,
            cell=self.cell,
            task_ids=task_ids,
            repeats=repeats,
        )
        return {"records": [], "per_task": {}}

    monkeypatch.setattr(TrainRolloutService, "_run_group", fake_shared)

    service._run_group(
        RolloutRequest(
            request_id="request",
            run_id="run",
            scope="TEST",
            harness_version="v0",
            task_repeats={task_id: 1},
            max_concurrency=1,
            purpose="test",
        ),
        [task_id],
        1,
    )

    assert captured == {
        "harness": harness,
        "cell": cell,
        "task_ids": [task_id],
        "repeats": 1,
    }


def _request(task_id: str, *, scope: str) -> RolloutRequest:
    return RolloutRequest(
        request_id="request",
        run_id="run",
        scope=scope,
        harness_version="v0",
        task_repeats={task_id: 2},
        max_concurrency=2,
        purpose="test",
        pairing_offsets={task_id: 0},
    )


def test_owned_splits_match_the_canonical_inputs():
    repo_root = Path(__file__).resolve().parents[1]
    canonical = repo_root / "assets" / "canonical_splits"
    retail = load_benchmark_split("retail")
    banking = load_benchmark_split("banking")
    terminal = load_benchmark_split("terminal-bench")

    retail_reference_path = (
        repo_root / "third_party/tau3-bench/data/tau2/domains/retail/split_tasks.json"
    )
    if not retail_reference_path.is_file():
        pytest.skip("tau2 retail domain checkout is unavailable under third_party/")
    retail_reference = json.loads(retail_reference_path.read_text())
    banking_reference = json.loads(
        (canonical / "banking_knowledge_split.json").read_text()
    )
    terminal_payload = yaml.safe_load(
        (canonical / "terminal_bench_split_seed42_train30_test59.yaml").read_text()
    )["terminal_bench_split"]

    assert list(retail.test) == retail_reference["test"]
    assert list(banking.train) == banking_reference["train"]
    assert list(banking.test) == banking_reference["test"]
    assert list(terminal.train) == terminal_payload["train_task_names"]
    assert list(terminal.test) == terminal_payload["validation_task_names"]
    assert (len(retail.train), len(retail.test)) == (30, 40)
    assert (len(banking.train), len(banking.test)) == (30, 67)
    assert (len(terminal.train), len(terminal.test)) == (30, 59)


def test_existing_retail_rollout_service_still_rejects_test_scope(tmp_path):
    service = _service(
        TrainRolloutService,
        tmp_path,
        cell="retail",
        task_ids=["0"],
        local_rootless=True,
    )

    with pytest.raises(ValueError, match="only accepts TRAIN"):
        service._validate(_request("0", scope="TEST"))


def test_additive_test_service_accepts_only_test_scope(tmp_path):
    service = _service(
        BlindTestRolloutService,
        tmp_path,
        cell="banking_knowledge",
        task_ids=["task_001"],
        local_rootless=True,
    )

    service._validate(_request("task_001", scope="TEST"))
    with pytest.raises(ValueError, match="only accepts TEST"):
        service._validate(_request("task_001", scope="TRAIN"))


def test_test_rollout_rejects_repeat_groups_that_would_serialize(tmp_path):
    service = _service(
        BlindTestRolloutService,
        tmp_path,
        cell="banking_knowledge",
        task_ids=["task_001", "task_002"],
        local_rootless=True,
    )
    request = RolloutRequest(
        request_id="request",
        run_id="run",
        scope="TEST",
        harness_version="v0",
        task_repeats={"task_001": 1, "task_002": 2},
        max_concurrency=2,
        purpose="test",
        pairing_offsets={"task_001": 0, "task_002": 0},
    )

    with pytest.raises(ValueError, match="one shared repeat count"):
        service._validate(request)


def test_runtime_limits_are_fixed():
    assert runtime_limits(load_benchmark_split("banking")) == {
        "repeats": 1,
        "max_concurrency": 20,
        "group_timeout_s": 7200,
        "agent_steps_per_turn": 10,
        "max_conversation_turns": 40,
        "timeout_per_turn_s": 180,
    }


def test_test_rollout_submits_one_global_queue_per_repeat_count(tmp_path, monkeypatch):
    task_ids = ["task_3", "task_1", "task_2", "task_4", "task_5"]
    service = BlindTestRolloutService(
        cell="terminal_bench",
        repo_root=tmp_path,
        run_id="run",
        artifact_root=tmp_path / "artifacts",
        train_task_ids=task_ids,
        initial_budget=10,
        evidence_root=tmp_path / "evidence",
        workspace_root=tmp_path / "workspaces",
        local_rootless_rollout=False,
    )
    request = RolloutRequest(
        request_id="request",
        run_id="run",
        scope="TEST",
        harness_version="v0",
        task_repeats={task_id: 2 for task_id in task_ids},
        max_concurrency=2,
        purpose="test",
        pairing_offsets={task_id: 0 for task_id in task_ids},
    )
    calls = []

    def fake_run_group(_request, selected, repeats):
        calls.append((list(selected), repeats))
        return {
            "per_task": {task_id: {"rewards": [0, 0]} for task_id in selected},
            "records": [
                {
                    "task_id": task_id,
                    "rewards": [0, 0],
                    "trajectory_paths": [],
                    "worker_errors": [],
                    "trial_summaries": [],
                }
                for task_id in selected
            ],
        }

    monkeypatch.setattr(service, "_run_group", fake_run_group)
    monkeypatch.setattr(
        "harnesslens.evaluation.blind_test_eval.retain_trial_trajectories",
        lambda raw, target_root: raw,
    )
    monkeypatch.setattr(
        "harnesslens.evaluation.blind_test_eval._validate_terminal_trajectories",
        lambda records: None,
    )

    service.run(request)

    assert calls == [(sorted(task_ids), 2)]


def test_terminal_runtime_limits_are_bounded_by_concurrency():
    assert runtime_limits(load_benchmark_split("terminal-bench")) == {
        "repeats": 1,
        "max_concurrency": 10,
        "group_timeout_s": 7200,
        "agent_steps": 50,
        "exec_timeout_s": 600,
        "verify_timeout_s": 1800,
    }


def test_test_baseline_rejects_task_outside_canonical_test_split(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "key")

    with pytest.raises(ValueError, match="outside the canonical TEST split"):
        run_test_baseline(
            repo_root=tmp_path,
            run_id="run",
            benchmark="banking",
            task_ids=["not-a-test-task"],
        )


@pytest.mark.parametrize("harness", ["opencode", "pi", "codex"])
def test_test_baseline_threads_target_harness_to_service(
    tmp_path, monkeypatch, harness
):
    split = load_benchmark_split("retail")
    task_id = split.test[0]
    captured = {}

    class FakeService:
        def __init__(self, **kwargs):
            captured["service_harness"] = kwargs["harness"]

        def run(self, request):
            captured["request"] = request
            return SimpleNamespace(to_dict=lambda: {}, metrics={})

    monkeypatch.setenv("DEEPSEEK_API_KEY", "key")
    monkeypatch.setattr(
        "harnesslens.evaluation.blind_test_eval.BlindTestRolloutService",
        FakeService,
    )

    result = run_test_baseline(
        repo_root=tmp_path,
        run_id="run",
        benchmark="retail",
        task_ids=[task_id],
        repeats=1,
        harness=harness,
    )

    assert captured["service_harness"] == harness
    assert f"-{harness}-" in captured["request"].request_id
    assert f"/{harness}/" in result.output_path


@pytest.mark.parametrize("harness", ["opencode", "pi", "codex"])
def test_test_candidate_materializes_workspace_submission(
    tmp_path, monkeypatch, harness
):
    split = load_benchmark_split("retail")
    task_id = split.test[0]
    snapshot = (
        tmp_path
        / "runs"
        / "train"
        / "train"
        / "versions_percell"
        / "retail"
        / "candidate-01"
    )
    native = snapshot / "harness" / harness
    native.mkdir(parents=True)
    workspace = {
        "schema": 1,
        "files": [
            {
                "scope": "project",
                "path": "candidate.txt",
                "content": harness,
                "executable": False,
            }
        ],
    }
    (native / "workspace.json").write_text(json.dumps(workspace), encoding="utf-8")
    (native / "manifest.json").write_text("{}", encoding="utf-8")
    (snapshot / "meta.json").write_text(
        json.dumps({"harness": harness, "version": "candidate-01"}),
        encoding="utf-8",
    )
    submission = (
        tmp_path
        / "runs"
        / "train"
        / "train"
        / "submission"
        / "final.json"
    )
    submission.parent.mkdir(parents=True)
    submission.write_text(
        json.dumps(
            {
                "selected_version": "candidate-01",
                "snapshot_path": str(snapshot),
                "decision": "submit_candidate",
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    class FakeRepository:
        def __init__(self, **_kwargs):
            pass

        def materialize_workspace_candidate(self, **kwargs):
            captured.update(kwargs)
            return "workspace-candidate"

    class FakeService:
        def __init__(self, **_kwargs):
            pass

        def run(self, _request):
            return SimpleNamespace(to_dict=lambda: {}, metrics={})

    monkeypatch.setenv("DEEPSEEK_API_KEY", "key")
    monkeypatch.setattr(
        "harnesslens.evaluation.blind_test_eval.CellHarnessRepository",
        FakeRepository,
    )
    monkeypatch.setattr(
        "harnesslens.evaluation.blind_test_eval.BlindTestRolloutService",
        FakeService,
    )

    result = run_test_candidate(
        repo_root=tmp_path,
        run_id="test",
        benchmark="retail",
        patch_json=submission,
        task_ids=[task_id],
        repeats=1,
        harness=harness,
    )

    assert result.harness_version == "workspace-candidate"
    assert captured["workspace"] == workspace
    assert captured["manifest_delta"] == {}
    assert captured["base_version"] == "v0"


def test_workspace_submission_maps_selected_v0_to_empty_workspace(tmp_path):
    run_root = tmp_path / "runs" / "train" / "train"
    submission = run_root / "submission" / "final.json"
    submission.parent.mkdir(parents=True)
    submission.write_text(
        json.dumps(
            {
                "selected_version": "v0",
                "snapshot_path": str(run_root / "rollout_evidence" / "train" / "versions" / "v0"),
            }
        ),
        encoding="utf-8",
    )
    discovery = run_root / "discovery" / "harness_query.json"
    discovery.parent.mkdir(parents=True)
    discovery.write_text(json.dumps({"harness": "pi"}), encoding="utf-8")

    loaded = _load_workspace_submission(
        submission,
        repo_root=tmp_path,
        harness="pi",
    )

    assert loaded == {"workspace": {"schema": 1, "files": []}, "manifest": {}}


@pytest.mark.parametrize("harness", ["opencode", "pi", "codex"])
def test_test_candidate_threads_target_harness_to_repository_and_service(
    tmp_path, monkeypatch, harness
):
    split = load_benchmark_split("retail")
    task_id = split.test[0]
    patch_path = tmp_path / "patch.json"
    patch_path.write_text("{}", encoding="utf-8")
    captured = {}

    class FakeRepository:
        def __init__(self, **kwargs):
            captured["repository_harness"] = kwargs["harness"]

        def materialize_candidate(self, **kwargs):
            return "candidate-01"

    class FakeService:
        def __init__(self, **kwargs):
            captured["service_harness"] = kwargs["harness"]

        def run(self, request):
            captured["request"] = request
            return SimpleNamespace(to_dict=lambda: {}, metrics={})

    monkeypatch.setenv("DEEPSEEK_API_KEY", "key")
    monkeypatch.setattr(
        "harnesslens.evaluation.blind_test_eval.CellHarnessRepository",
        FakeRepository,
    )
    monkeypatch.setattr(
        "harnesslens.evaluation.blind_test_eval.BlindTestRolloutService",
        FakeService,
    )

    result = run_test_candidate(
        repo_root=tmp_path,
        run_id="run",
        benchmark="retail",
        patch_json=patch_path,
        task_ids=[task_id],
        repeats=1,
        harness=harness,
    )

    assert result.harness_version == "candidate-01"
    assert captured["repository_harness"] == harness
    assert captured["service_harness"] == harness
    assert f"-{harness}-" in captured["request"].request_id
    assert f"/{harness}/" in result.output_path
