import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import harnesslens.evolution.harness_editor as editor_module
from harnesslens.core.budget import CreationBudget
from harnesslens.evolution.harness_editor import (
    EDITOR_TOOLS,
    HarnessEditor,
    validate_editor_candidate_artifacts,
)


class FakeEditorAdapter:
    constructed: list[dict] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.constructed.append(kwargs)

    def run(
        self,
        *,
        prompt,
        workspace,
        working_directory,
        call_id,
        budget,
        max_steps,
        output_validator,
    ):
        del prompt, max_steps, output_validator
        record = budget.reserve(call_id, metadata={"creation_count": 1})
        assert record["status"] == "reserved"
        budget.mark_launched(call_id)
        project = Path(working_directory) / "project"
        (project / ".candidate").mkdir(parents=True, exist_ok=True)
        (project / ".candidate" / "rule.md").write_text(
            "Use evidence.\n", encoding="utf-8"
        )
        root = Path(workspace)
        stdout = root / "editor.stdout"
        stderr = root / "editor.stderr"
        trace = root / "api_calls.jsonl"
        stdout.write_text(
            json.dumps({"changed_paths": ["project/.candidate/rule.md"]}),
            encoding="utf-8",
        )
        stderr.write_text("", encoding="utf-8")
        trace.write_text("{}\n", encoding="utf-8")
        budget.settle(call_id, outcome="completed", usage={})
        return SimpleNamespace(
            outcome="completed",
            validation_error="",
            stdout_path=str(stdout),
            stderr_path=str(stderr),
            api_trace_path=str(trace),
        )


@pytest.mark.parametrize("harness", ["opencode", "pi", "codex"])
def test_harness_editor_uses_target_harness_and_captures_actual_diff(
    tmp_path, monkeypatch, harness
):
    FakeEditorAdapter.constructed.clear()
    monkeypatch.setattr(editor_module, "OpenCodeIntelligentAdapter", FakeEditorAdapter)
    monkeypatch.setattr(editor_module, "NativeIntelligentAdapter", FakeEditorAdapter)
    budget = CreationBudget(tmp_path / f"{harness}-budget.json", total=2, baseline_used=0)
    editor = HarnessEditor(
        harness=harness,
        budget=budget,
        run_root=tmp_path / harness,
        max_steps=4,
    )

    result = editor.edit(
        job_id="editor-01",
        base_workspace={"files": []},
        harness_query={"modifiable_modules": [{"id": "project-instructions"}]},
        problem={"id": "problem-01", "summary": "Missed evidence"},
        evidence=[{"id": "exp-01", "finding": "Read before acting"}],
    )

    constructed = FakeEditorAdapter.constructed[-1]
    if harness != "opencode":
        assert constructed["harness"] == harness
    assert constructed["allowed_builtin_tools"] == EDITOR_TOOLS[harness]
    assert result.changes[0]["path"] == ".candidate/rule.md"
    assert result.summary == {
        "changed_paths": ["project/.candidate/rule.md"]
    }
    assert Path(result.root, "workspace.json").is_file()
    editor_input = json.loads(Path(result.root, "input.json").read_text(encoding="utf-8"))
    assert Path(editor_input["scratch_tree"]).is_dir()
    assert Path(editor_input["scratch_tree"]).parent == Path(result.root)
    assert budget.status()["remaining"] == 1


def test_harness_editor_does_not_treat_query_as_a_surface_whitelist(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(editor_module, "OpenCodeIntelligentAdapter", FakeEditorAdapter)
    budget = CreationBudget(tmp_path / "budget.json", total=1, baseline_used=0)
    editor = HarnessEditor(
        harness="opencode", budget=budget, run_root=tmp_path / "jobs", max_steps=2
    )

    result = editor.edit(
        job_id="editor-01",
        base_workspace={"files": []},
        harness_query={"modifiable_modules": []},
        problem={"id": "problem-01"},
        evidence=[],
    )

    assert result.changes[0]["path"] == ".candidate/rule.md"


def test_editor_candidate_artifacts_cannot_name_supplied_evidence():
    with pytest.raises(ValueError, match="must generalize evidence"):
        validate_editor_candidate_artifacts(
            snapshot={
                "files": [
                    {
                        "scope": "project",
                        "path": "AGENTS.md",
                        "content": "For exp-specific-case, return these exact columns.\n",
                    }
                ]
            },
            changes=[
                {"scope": "project", "path": "AGENTS.md", "change": "added"}
            ],
            evidence=[{"id": "exp-specific-case", "evidence_refs": ["ev_123"]}],
        )


def test_editor_candidate_artifacts_cannot_copy_schema_specific_identifiers():
    with pytest.raises(ValueError, match="AdmEmail1"):
        validate_editor_candidate_artifacts(
            snapshot={
                "files": [
                    {
                        "scope": "project",
                        "path": "AGENTS.md",
                        "content": "For example, select AdmEmail1 directly.\n",
                    }
                ]
            },
            changes=[
                {"scope": "project", "path": "AGENTS.md", "change": "added"}
            ],
            evidence=[
                {
                    "id": "column-shape",
                    "text": "The failed query selected AdmEmail1 and AdmEmail2.",
                }
            ],
        )


def test_editor_candidate_pi_skill_requires_frontmatter():
    with pytest.raises(ValueError, match="requires YAML frontmatter"):
        validate_editor_candidate_artifacts(
            snapshot={
                "files": [
                    {
                        "scope": "project",
                        "path": ".pi/skills/address-modification/SKILL.md",
                        "content": "# Address modification\n\nUse the confirmed address.\n",
                    }
                ]
            },
            changes=[
                {
                    "scope": "project",
                    "path": ".pi/skills/address-modification/SKILL.md",
                    "change": "added",
                }
            ],
            evidence=[],
        )


def test_editor_candidate_skill_name_matches_directory():
    with pytest.raises(ValueError, match="name must match"):
        validate_editor_candidate_artifacts(
            snapshot={
                "files": [
                    {
                        "scope": "project",
                        "path": ".pi/skills/address-modification/SKILL.md",
                        "content": (
                            "---\nname: wrong-name\n"
                            "description: Use after address confirmation.\n---\n\n"
                            "Apply the confirmed address.\n"
                        ),
                    }
                ]
            },
            changes=[
                {
                    "scope": "project",
                    "path": ".pi/skills/address-modification/SKILL.md",
                    "change": "added",
                }
            ],
            evidence=[],
        )


def test_editor_candidate_accepts_valid_pi_skill_frontmatter():
    validate_editor_candidate_artifacts(
        snapshot={
            "files": [
                {
                    "scope": "project",
                    "path": ".pi/skills/address-modification/SKILL.md",
                    "content": (
                        "---\nname: address-modification\n"
                        "description: Use after address confirmation.\n---\n\n"
                        "Apply the confirmed address.\n"
                    ),
                }
            ]
        },
        changes=[
            {
                "scope": "project",
                "path": ".pi/skills/address-modification/SKILL.md",
                "change": "added",
            }
        ],
        evidence=[],
    )
