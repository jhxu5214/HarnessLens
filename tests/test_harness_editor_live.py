import json
import os
import tempfile
import tomllib
from pathlib import Path

import pytest

from harnesslens.core.budget import CreationBudget
from harnesslens.core.config import load_repo_env
from harnesslens.evolution.harness_editor import HarnessEditor
from harnesslens.harnesses.harness_query_adapters import harness_query_adapter


REPO_ROOT = Path(__file__).resolve().parents[1]


pytestmark = pytest.mark.skipif(
    os.environ.get("HAI_RUN_HARNESS_EDITOR_LIVE") != "1",
    reason="set HAI_RUN_HARNESS_EDITOR_LIVE=1 to run isolated target-harness editors",
)


@pytest.mark.parametrize("harness", ["opencode", "pi", "codex"])
def test_target_harness_editor_uses_real_query_and_isolated_workspace(harness):
    load_repo_env(REPO_ROOT)
    assert os.environ.get("DEEPSEEK_API_KEY")
    temporary_root = None
    with tempfile.TemporaryDirectory(prefix=f"harnesslens-{harness}-editor-live-") as raw:
        temporary_root = Path(raw)
        adapter = harness_query_adapter(harness, repo_root=REPO_ROOT)
        probe = adapter.architecture_probe()
        query = {
            "harness": harness,
            "architecture_probe": probe,
            "modifiable_modules": adapter.query_channel_inventory(probe),
            "evidence_catalog": adapter.query_evidence_catalog(probe),
        }
        budget = CreationBudget(
            temporary_root / "budget.json", total=1, baseline_used=0
        )
        editor = HarnessEditor(
            harness=harness,
            budget=budget,
            run_root=temporary_root / "editor_jobs",
            max_steps=12,
            timeout_s=900,
        )

        result = editor.edit(
            job_id="editor-live-01",
            base_workspace={"files": []},
            harness_query=query,
            problem={
                "id": "verify-before-use",
                "summary": "The agent used a returned record without checking its identifier.",
                "local_success_criteria": [
                    "A concise project rule tells the agent to verify returned identifiers."
                ],
            },
            evidence=[
                {
                    "id": "train-observation-01",
                    "scope": "TRAIN",
                    "finding": "A visible rollout continued after lookup without comparing the returned identifier.",
                }
            ],
            current_manifest={},
        )

        assert result.changes
        assert all(item["scope"] in {"home", "project"} for item in result.changes)
        if harness == "opencode":
            files = {
                (str(item["scope"]), str(item["path"])): str(item["content"])
                for item in result.snapshot["files"]
            }
            assert ("project", "opencode.json") in files, json.dumps(
                result.snapshot,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            config = json.loads(files[("project", "opencode.json")])
            instructions = config.get("instructions")
            assert isinstance(instructions, list) and instructions, config
            assert all(
                ("project", str(path)) in files
                for path in instructions
            ), result.snapshot
        elif harness == "pi":
            files = {
                (str(item["scope"]), str(item["path"])): str(item["content"])
                for item in result.snapshot["files"]
            }
            assert files.get(("project", "AGENTS.md"), "").strip(), result.snapshot
        elif harness == "codex":
            files = {
                (str(item["scope"]), str(item["path"])): str(item["content"])
                for item in result.snapshot["files"]
            }
            project_rule = files.get(("project", "AGENTS.md"), "").strip()
            config_text = files.get(("home", "config.toml"), "")
            developer_rule = (
                str(tomllib.loads(config_text).get("developer_instructions") or "").strip()
                if config_text
                else ""
            )
            assert project_rule or developer_rule, result.snapshot
        invocation = json.loads(
            (Path(result.root) / "invocation.json").read_text(encoding="utf-8")
        )
        assert Path(invocation["command"][0]).name == "bwrap"
        assert budget.status()["remaining"] == 0

    assert temporary_root is not None
    assert not temporary_root.exists()
