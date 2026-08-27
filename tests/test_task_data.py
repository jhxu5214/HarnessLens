import json
import os
from pathlib import Path

import pytest

from harnesslens.benchmarks.cell_config import benchmark_config
from harnesslens.benchmarks.task_data import (
    BaselineDataset,
    _terminal_task_explorer_input,
    benchmark_task_explorer_input,
)
from harnesslens.evolution.baseline import (
    _semantic_fingerprint,
    _validate_baseline_runtime,
    build_baseline_fingerprint,
    ensure_baseline_event,
)


def _runtime_files(root):
    from harnesslens.evolution.baseline import BASELINE_RUNTIME_FILES

    for relative in BASELINE_RUNTIME_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")


def test_baseline_dataset_requires_thirty_paired_tasks(tmp_path):
    artifacts = []
    for task in range(30):
        for trial in range(2):
            path = tmp_path / f"{task}-{trial}.jsonl"
            path.write_text(json.dumps({"task_id": str(task), "trial": trial}) + "\n")
            artifacts.append({"path": str(path), "evidence_id": f"ev_{task}_{trial}"})
    event = tmp_path / "output.json"
    event.write_text(
        json.dumps({"agent_workspace_entry": {"trajectory_artifacts": artifacts}})
    )

    baseline = BaselineDataset.from_ingest_event(event)

    assert len(baseline.task_ids) == 30
    assert all(len(paths) == 2 for paths in baseline.trajectories_by_task.values())


def test_baseline_runtime_rejects_infrastructure_only_completion():
    payload = {
        "metrics": {
            "requested_trial_count": 30,
            "charged_trial_count": 0,
            "infrastructure_failure_count": 30,
            "worker_error_count": 30,
        }
    }

    with pytest.raises(ValueError, match="infrastructure failures"):
        _validate_baseline_runtime(payload, expected_trial_count=30)


def test_baseline_runtime_accepts_complete_charged_trials():
    payload = {
        "metrics": {
            "requested_trial_count": 30,
            "charged_trial_count": 30,
            "infrastructure_failure_count": 0,
            "worker_error_count": 0,
        }
    }

    _validate_baseline_runtime(payload, expected_trial_count=30)


def test_baseline_dataset_rejects_unpaired_tasks(tmp_path):
    path = tmp_path / "one.jsonl"
    path.write_text(json.dumps({"task_id": "1"}) + "\n")
    event = tmp_path / "output.json"
    event.write_text(
        json.dumps(
            {
                "agent_workspace_entry": {
                    "trajectory_artifacts": [{"path": str(path), "evidence_id": "ev_1"}]
                }
            }
        )
    )

    with pytest.raises(ValueError, match="30 tasks x 2"):
        BaselineDataset.from_ingest_event(event)


def test_cached_baseline_is_validated_without_rollout(tmp_path):
    _runtime_files(tmp_path)
    artifacts = []
    for task in range(30):
        for trial in range(2):
            path = tmp_path / f"{task}-{trial}.jsonl"
            path.write_text(json.dumps({"task_id": str(task), "trial": trial}) + "\n")
            artifacts.append({"path": str(path), "evidence_id": f"ev_{task}_{trial}"})
    event = tmp_path / "event.json"
    event.write_text(
        json.dumps(
            {
                "agent_workspace_entry": {"trajectory_artifacts": artifacts},
                "baseline_fingerprint": build_baseline_fingerprint(
                    tmp_path,
                    task_ids=tuple(str(task) for task in range(30)),
                ),
            }
        )
    )

    selected = ensure_baseline_event(
        repo_root=tmp_path,
        run_root=tmp_path / "run",
        baseline_event=event,
        task_ids=tuple(str(task) for task in range(30)),
    )

    assert selected == event.resolve()


def test_cached_baseline_rejects_runtime_drift(tmp_path):
    _runtime_files(tmp_path)
    artifacts = []
    for task in range(30):
        for trial in range(2):
            path = tmp_path / f"{task}-{trial}.jsonl"
            path.write_text(json.dumps({"task_id": str(task), "trial": trial}) + "\n")
            artifacts.append({"path": str(path), "evidence_id": f"ev_{task}_{trial}"})
    fingerprint = build_baseline_fingerprint(
        tmp_path,
        task_ids=tuple(str(task) for task in range(30)),
    )
    event = tmp_path / "event.json"
    event.write_text(
        json.dumps(
            {
                "agent_workspace_entry": {"trajectory_artifacts": artifacts},
                "baseline_fingerprint": fingerprint,
            }
        )
    )
    changed = tmp_path / "harnesslens/benchmarks/opencode_tau2.py"
    changed.write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint differs"):
        ensure_baseline_event(
            repo_root=tmp_path,
            run_root=tmp_path / "run",
            baseline_event=event,
            task_ids=tuple(str(task) for task in range(30)),
        )


def test_cached_baseline_explicitly_reuses_same_harness_after_runtime_drift(
    tmp_path, monkeypatch
):
    _runtime_files(tmp_path)
    artifacts = []
    for task in range(30):
        for trial in range(2):
            path = tmp_path / f"{task}-{trial}.jsonl"
            path.write_text(json.dumps({"task_id": str(task), "trial": trial}) + "\n")
            artifacts.append({"path": str(path), "evidence_id": f"ev_{task}_{trial}"})
    event = tmp_path / "event.json"
    event.write_text(
        json.dumps(
            {
                "agent_workspace_entry": {"trajectory_artifacts": artifacts},
                "baseline_fingerprint": build_baseline_fingerprint(
                    tmp_path,
                    task_ids=tuple(str(task) for task in range(30)),
                ),
            }
        )
    )
    changed = tmp_path / "harnesslens/benchmarks/opencode_tau2.py"
    changed.write_text("changed", encoding="utf-8")
    monkeypatch.setenv("HAI_BASELINE_REUSE_POLICY", "harness_only")
    run_root = tmp_path / "run"

    selected = ensure_baseline_event(
        repo_root=tmp_path,
        run_root=run_root,
        baseline_event=event,
        task_ids=tuple(str(task) for task in range(30)),
    )

    assert selected == event.resolve()
    reuse = json.loads((run_root / "baseline" / "reuse.json").read_text())
    assert reuse["policy"] == "explicit_harness_only_runtime_mismatch"
    assert reuse["baseline_event"] == str(event.resolve())


def test_harness_only_baseline_reuse_rejects_another_harness(tmp_path, monkeypatch):
    _runtime_files(tmp_path)
    artifacts = []
    for task in range(30):
        for trial in range(2):
            path = tmp_path / f"{task}-{trial}.jsonl"
            path.write_text(json.dumps({"task_id": str(task), "trial": trial}) + "\n")
            artifacts.append({"path": str(path), "evidence_id": f"ev_{task}_{trial}"})
    fingerprint = build_baseline_fingerprint(
        tmp_path,
        harness="pi",
        task_ids=tuple(str(task) for task in range(30)),
    )
    event = tmp_path / "event.json"
    event.write_text(
        json.dumps(
            {
                "agent_workspace_entry": {"trajectory_artifacts": artifacts},
                "baseline_fingerprint": fingerprint,
            }
        )
    )
    monkeypatch.setenv("HAI_BASELINE_REUSE_POLICY", "harness_only")

    with pytest.raises(ValueError, match="matching harness"):
        ensure_baseline_event(
            repo_root=tmp_path,
            run_root=tmp_path / "run",
            baseline_event=event,
            harness="opencode",
            task_ids=tuple(str(task) for task in range(30)),
        )


def test_cached_baseline_allows_orchestration_only_drift(tmp_path):
    _runtime_files(tmp_path)
    artifacts = []
    for task in range(30):
        for trial in range(2):
            path = tmp_path / f"{task}-{trial}.jsonl"
            path.write_text(json.dumps({"task_id": str(task), "trial": trial}) + "\n")
            artifacts.append({"path": str(path), "evidence_id": f"ev_{task}_{trial}"})
    fingerprint = build_baseline_fingerprint(
        tmp_path,
        task_ids=tuple(str(task) for task in range(30)),
    )
    fingerprint["runtime_file_sha256"][
        "harnesslens/evolution/baseline.py"
    ] = "legacy-orchestration-hash"
    fingerprint["runtime_file_sha256"][
        "harnesslens/evaluation/rollout_bridge.py"
    ] = "legacy-bridge-orchestration-hash"
    fingerprint["fingerprint_sha256"] = "legacy-full-hash"
    event = tmp_path / "event.json"
    event.write_text(
        json.dumps(
            {
                "agent_workspace_entry": {"trajectory_artifacts": artifacts},
                "baseline_fingerprint": fingerprint,
            }
        )
    )

    selected = ensure_baseline_event(
        repo_root=tmp_path,
        run_root=tmp_path / "run",
        baseline_event=event,
        task_ids=tuple(str(task) for task in range(30)),
    )

    assert selected == event.resolve()


@pytest.mark.parametrize(
    ("path", "old_digest", "new_digest"),
    [
        (
            "harnesslens/benchmarks/codex_tau2.py",
            "b6422f55fc7a4f96e92a06dbd47c169f3f88b2933e4f73280210067d3e1d64a9",
            "53168894b1cf8c5313a9ec00008f3d2720497d113fbebd33033f768d1163bb1e",
        ),
        (
            "harnesslens/harnesses/native_candidate_runtime.py",
            "0662381e90901111796667e2910f8b8f4ecb994f833870806d1e2cfb50157f0f",
            "74f449eb9eff2868a5057d14b8a0ab42af3b62945dfec3dfdf7234077c1e88cd",
        ),
        (
            "harnesslens/benchmarks/bird_eval.py",
            "bbe94facc6dec701e44e987c90fcdd0de904cb2ba2d4e21c48ae768a1fbf3438",
            "1a9e31382eb262c356ad96cad0f7068f0151a58798bd8f515ef698902d7fda11",
        ),
        (
            "harnesslens/benchmarks/bird_eval.py",
            "f211f0eec317c7eb3a2eb7de43b4d51f0e6c35a59295c8db864d2847ce4dec37",
            "d13de9fdea73778df2b0061ca01c830d0520d6e564daa881e0a37836e1e0a5f0",
        ),
        (
            "harnesslens/benchmarks/bird_eval.py",
            "f211f0eec317c7eb3a2eb7de43b4d51f0e6c35a59295c8db864d2847ce4dec37",
            "385cde635b07a3949c759b97aaf88e8d2cfc4d7b901d44a9a8f45a79d29d081b",
        ),
        (
            "harnesslens/benchmarks/opencode_tau2.py",
            "81419d16be285e2d837877596d007298d6def3982e917cc46b2a283ea11d638d",
            "e1e592c089bb4f7a09c7fdec23643b6fa13067434315622814469e9cc6913b17",
        ),
    ],
)
def test_baseline_allows_only_audited_candidate_runtime_transition(
    path, old_digest, new_digest
):
    old = _semantic_fingerprint({"runtime_file_sha256": {path: old_digest}})
    new = _semantic_fingerprint({"runtime_file_sha256": {path: new_digest}})
    unknown = _semantic_fingerprint({"runtime_file_sha256": {path: "unknown-drift"}})

    assert old == new
    assert unknown != new


def test_baseline_semantics_ignore_only_rollout_scheduling_width():
    old = _semantic_fingerprint(
        {"rollout": {"repeats": 2, "max_steps": 30, "max_concurrency": 20}}
    )
    new = _semantic_fingerprint(
        {"rollout": {"repeats": 2, "max_steps": 30, "max_concurrency": 2}}
    )
    changed_limit = _semantic_fingerprint(
        {"rollout": {"repeats": 2, "max_steps": 31, "max_concurrency": 2}}
    )

    assert old == new
    assert changed_limit != new


def test_uses_latest_complete_compatible_fresh_baseline(tmp_path):
    _runtime_files(tmp_path)
    fingerprint = build_baseline_fingerprint(
        tmp_path,
        task_ids=tuple(str(task) for task in range(30)),
    )
    runs_root = tmp_path / "runs" / "train"
    events = []
    for index in range(2):
        artifacts = []
        artifact_root = tmp_path / f"artifacts-{index}"
        artifact_root.mkdir()
        for task in range(30):
            for trial in range(2):
                path = artifact_root / f"{task}-{trial}.jsonl"
                path.write_text(
                    json.dumps({"task_id": str(task), "trial": trial}) + "\n"
                )
                artifacts.append(
                    {"path": str(path), "evidence_id": f"ev_{index}_{task}_{trial}"}
                )
        event = runs_root / f"old-{index}" / "baseline" / "bootstrap_event.json"
        event.parent.mkdir(parents=True)
        event.write_text(
            json.dumps(
                {
                    "agent_workspace_entry": {"trajectory_artifacts": artifacts},
                    "baseline_fingerprint": fingerprint,
                }
            )
        )
        os.utime(event, (index + 1, index + 1))
        events.append(event)

    run_root = runs_root / "current"
    selected = ensure_baseline_event(
        repo_root=tmp_path,
        run_root=run_root,
        baseline_event=None,
        task_ids=tuple(str(task) for task in range(30)),
    )

    assert selected == events[-1].resolve()
    reuse = json.loads((run_root / "baseline" / "reuse.json").read_text())
    assert reuse["baseline_event"] == str(events[-1].resolve())


def test_run_local_cached_baseline_rejects_runtime_drift(tmp_path):
    _runtime_files(tmp_path)
    artifacts = []
    for task in range(30):
        for trial in range(2):
            path = tmp_path / f"{task}-{trial}.jsonl"
            path.write_text(json.dumps({"task_id": str(task), "trial": trial}) + "\n")
            artifacts.append({"path": str(path), "evidence_id": f"ev_{task}_{trial}"})
    run_root = tmp_path / "run"
    event = run_root / "baseline" / "bootstrap_event.json"
    event.parent.mkdir(parents=True)
    event.write_text(
        json.dumps(
            {
                "agent_workspace_entry": {"trajectory_artifacts": artifacts},
                "baseline_fingerprint": build_baseline_fingerprint(
                    tmp_path,
                    task_ids=tuple(str(task) for task in range(30)),
                ),
            }
        )
    )
    changed = tmp_path / "harnesslens/benchmarks/opencode_tau2.py"
    changed.write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint differs"):
        ensure_baseline_event(
            repo_root=tmp_path,
            run_root=run_root,
            baseline_event=None,
            task_ids=tuple(str(task) for task in range(30)),
        )


def _paired_event(tmp_path, task_ids):
    artifacts = []
    for task_id in task_ids:
        for trial in range(2):
            path = tmp_path / f"{task_id}-{trial}.jsonl"
            path.write_text(
                json.dumps({"task_id": str(task_id), "trial": trial}) + "\n"
            )
            artifacts.append(
                {"path": str(path), "evidence_id": f"ev_{task_id}_{trial}"}
            )
    event = tmp_path / "paired_event.json"
    event.write_text(
        json.dumps({"agent_workspace_entry": {"trajectory_artifacts": artifacts}})
    )
    return event


def test_banking_config_and_fingerprint_are_not_retail():
    repo_root = Path(__file__).resolve().parents[1]

    config = benchmark_config(repo_root, "banking")
    fingerprint = build_baseline_fingerprint(repo_root, cell="banking")

    assert config.cell == "banking_knowledge"
    assert config.kind == "tau2"
    assert config.outcome_authority == "behavioral"
    assert len(config.train_task_ids) == 30
    assert fingerprint["benchmark"] == "tau2-banking_knowledge"
    assert fingerprint["cell"] == "banking_knowledge"
    assert any(
        "banking_knowledge/tasks.json" in path
        for path in fingerprint["runtime_file_sha256"]
    )
    assert not any("/retail/" in path for path in fingerprint["runtime_file_sha256"])


def test_banking_task_explorer_uses_banking_queries(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    config = benchmark_config(repo_root, "banking_knowledge")
    baseline = BaselineDataset.from_ingest_event(
        _paired_event(tmp_path, config.train_task_ids)
    )
    monkeypatch.setattr(
        "harnesslens.benchmarks.task_data._tau2_tool_definitions",
        lambda repo_root, *, cell: [
            {"name": f"{cell}_tool", "description": "tool", "parameters": {}}
        ],
    )

    payload = benchmark_task_explorer_input(
        repo_root=repo_root,
        baseline=baseline,
        cell="banking_knowledge",
    )

    assert payload["domain"] == "banking_knowledge"
    assert payload["benchmark_kind"] == "tau2"
    assert len(payload["tasks"]) == 30
    assert payload["tasks"][0]["task_id"].startswith("task_")
    assert "instruction" in payload["tasks"][0]["query"]
    assert payload["environment"]["tools"][0]["name"] == "banking_knowledge_tool"


def test_terminal_bench_config_is_parameterized():
    repo_root = Path(__file__).resolve().parents[1]

    config = benchmark_config(repo_root, "terminal-bench")
    fingerprint = build_baseline_fingerprint(repo_root, cell="terminal_bench")

    assert config.cell == "terminal_bench"
    assert config.kind == "terminal_bench"
    assert len(config.train_task_ids) == 30
    assert fingerprint["benchmark"] == "terminal-bench"
    assert fingerprint["cell"] == "terminal_bench"
    assert any(
        "terminal_bench.py" in path for path in fingerprint["runtime_file_sha256"]
    )


def test_terminal_task_explorer_supports_harbor_task_format(tmp_path):
    task_root = (
        tmp_path
        / "third_party"
        / "terminal-bench"
        / "original-tasks"
        / "harbor-task"
    )
    task_root.mkdir(parents=True)
    (task_root / "task.toml").write_text(
        'version = "1.0"\n[metadata]\ncategory = "software-engineering"\n',
        encoding="utf-8",
    )
    (task_root / "instruction.md").write_text(
        "Implement the requested interface.\n",
        encoding="utf-8",
    )
    baseline = BaselineDataset(
        task_ids=("harbor-task",),
        trajectory_paths=(),
        trajectories_by_task={},
        evidence_by_path={},
        source_event="event.json",
    )

    payload = _terminal_task_explorer_input(
        tmp_path,
        baseline,
        cell="terminal_bench",
    )

    assert payload["tasks"] == [
        {
            "task_id": "harbor-task",
            "query": {
                "instruction": "Implement the requested interface.\n",
                "metadata": {
                    "version": "1.0",
                    "metadata": {"category": "software-engineering"},
                },
            },
        }
    ]
