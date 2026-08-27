import json
import os
import tempfile
from pathlib import Path

import pytest

import harnesslens.evaluation.rollout_bridge as rollout_bridge
from harnesslens.benchmarks.benchmark_splits import load_benchmark_split
from harnesslens.core.config import load_repo_env
from harnesslens.harnesses.harness_workspace import normalize_workspace_snapshot
from harnesslens.evaluation.rollout_bridge import (
    RolloutRequest,
    TrainRolloutService,
    CellHarnessRepository,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


pytestmark = pytest.mark.skipif(
    os.environ.get("HAI_RUN_NATIVE_CANDIDATE_LIVE") != "1",
    reason="set HAI_RUN_NATIVE_CANDIDATE_LIVE=1 to run native candidate rollouts",
)


@pytest.mark.parametrize("harness", ["opencode", "pi", "codex"])
def test_native_candidate_snapshot_runs_in_same_harness_and_cleans_up(
    monkeypatch, harness
):
    load_repo_env(REPO_ROOT)
    assert os.environ.get("DEEPSEEK_API_KEY")
    temporary_root = None
    with tempfile.TemporaryDirectory(prefix=f"harnesslens-{harness}-candidate-live-") as raw:
        temporary_root = Path(raw)
        monkeypatch.setenv("HAI_PI_RUNTIME_CWD_ROOT", str(temporary_root / "pi-cwd"))
        monkeypatch.setenv("HAI_CODEX_RUNTIME_CWD_ROOT", str(temporary_root / "codex-cwd"))
        monkeypatch.setenv("HAI_OPENCODE_RUNTIME_ROOT", str(temporary_root / "opencode-cwd"))
        monkeypatch.setattr(rollout_bridge, "TAU2_MAX_CONVERSATION_TURNS", 2)
        monkeypatch.setattr(rollout_bridge, "TAU2_AGENT_STEPS_PER_TURN", 5)
        monkeypatch.setattr(rollout_bridge, "TAU2_TIMEOUT_PER_TURN_S", 180)
        split = load_benchmark_split("retail")
        task_id = split.train[0]
        evidence_root = temporary_root / "evidence"
        repository = CellHarnessRepository(
            cell=split.cell,
            repo_root=REPO_ROOT,
            run_id="live-run",
            evidence_root=evidence_root,
            harness=harness,
        )
        prompt_sentinel = f"{harness.upper()}_CANDIDATE_PROMPT_SENTINEL"
        project_sentinel = f"{harness.upper()}_CANDIDATE_PROJECT_SENTINEL"
        skill_root = {
            "opencode": ".opencode/skills",
            "pi": ".pi/skills",
            "codex": ".agents/skills",
        }[harness]
        home_config = {
            "opencode": (
                "config.json",
                json.dumps(
                    {
                        "agent": {"build": {"prompt": prompt_sentinel}},
                    }
                ),
            ),
            "pi": (
                "settings.json",
                json.dumps(
                    {
                        "notice": "candidate-workspace",
                        "compaction": {"enabled": False},
                    }
                ),
            ),
            "codex": (
                "config.toml",
                f'developer_instructions = "{prompt_sentinel}"\n',
            ),
        }[harness]
        workspace = {
            "schema": 1,
            "files": [
                {
                    "scope": "home",
                    "path": home_config[0],
                    "content": home_config[1],
                    "executable": False,
                },
                {
                    "scope": "project",
                    "path": "AGENTS.md",
                    "content": project_sentinel + "\n",
                    "executable": False,
                },
                {
                    "scope": "project",
                    "path": f"{skill_root}/query-probe/SKILL.md",
                    "content": (
                        "---\nname: query-probe\n"
                        f"description: {harness} candidate skill sentinel.\n"
                        "---\n\nUse exact identifiers supplied by the task.\n"
                    ),
                    "executable": False,
                },
            ],
        }
        repository.materialize_workspace_candidate(
            base_version="v0",
            candidate_label="candidate-01",
            workspace=workspace,
        )
        assert repository.read_workspace_snapshot("candidate-01") == (
            normalize_workspace_snapshot(workspace)
        )
        service = TrainRolloutService(
            cell=split.cell,
            repo_root=REPO_ROOT,
            run_id="live-run",
            artifact_root=temporary_root / "artifacts",
            train_task_ids=[task_id],
            initial_budget=1,
            evidence_root=evidence_root,
            workspace_root=temporary_root / "workspaces",
            harness=harness,
            timeout_s=900,
        )
        response = service.run(
            RolloutRequest(
                request_id="candidate-smoke",
                run_id="live-run",
                scope="TRAIN",
                harness_version="candidate-01",
                task_repeats={task_id: 1},
                max_concurrency=1,
                purpose="same-harness candidate smoke",
                pairing_offsets={task_id: 0},
            )
        )

        assert response.records and response.records[0].harness_version == "candidate-01"
        trajectory = Path(response.records[0].trajectory_paths[0])
        row = json.loads(trajectory.read_text(encoding="utf-8"))
        assert row.get("harness") == harness, row
        assert not row.get("error"), row
        assert row["messages"]
        runtime_cwd = Path(row["raw"]["runtime_cwd"])
        assert not runtime_cwd.exists()
        if harness in {"opencode", "codex"}:
            api_calls = Path(row["api_calls_jsonl"])
            if not api_calls.is_absolute():
                api_calls = trajectory.parent / api_calls
            request_text = api_calls.read_text(encoding="utf-8")
            assert prompt_sentinel in request_text
            if harness == "codex":
                assert project_sentinel in request_text
                assert "codex candidate skill sentinel" in request_text
        else:
            assert row["candidate_system_prompt_append"] == ""

    assert temporary_root is not None
    assert not temporary_root.exists()
