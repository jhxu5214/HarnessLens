import json

import pytest
from pathlib import Path
from types import SimpleNamespace

from harnesslens.evolution.main_agent import (
    FINAL_MAIN_AGENT_SYSTEM,
    MAIN_AGENT_SYSTEM,
    MIN_CANDIDATE_ITERATION_CREATIONS,
    MIN_ROLLOUT_CREATIONS,
    _snapshot_manifest_path,
    _snapshot_workspace_path,
    _canonicalize_revision_manifest_delta,
    _compile_declared_config_channels,
    _final_review_attempts,
    _validate_mcp_tool_patches_against_query,
    _max_rollout_tasks,
    _rollout_task_limit,
    actual_editor_channel_diffs,
    assign_rollout_roles,
    build_rollout_budget_plan,
    build_exploration_status,
    canonicalize_main_decision,
    canonicalize_final_decision,
    defer_failure_only_portfolio,
    deterministic_reject_decision,
    _baseline_evidence_summary,
    iteration_creation_cost,
    validate_candidate_channel_combination,
    materialize_selected_problem,
    promotion_headroom_portfolio,
    prioritize_incumbent_controls,
    selectable_portfolio,
    validate_cached_main_decision,
    validate_final_decision,
    validate_main_decision,
    uncertain_recovery_task_ids,
)


def test_candidate_uses_at_most_one_global_instruction_channel():
    validate_candidate_channel_combination({"developer_instructions", "skills"})

    with pytest.raises(ValueError, match="at most one global instruction channel"):
        validate_candidate_channel_combination(
            {"developer_instructions", "project_instructions"}
        )


def test_opencode_inline_instruction_is_compiled_to_referenced_project_file():
    instruction = (
        "Return exactly one complete SQL statement and never emit </parameter>."
    )
    query = {
        "modifiable_modules": [
            {
                "id": "instructions_rules",
                "edit_contract": {
                    "scope": "project",
                    "path": "opencode.json",
                    "mechanism": "config",
                    "key": "instructions",
                },
            }
        ]
    }
    compiled = _compile_declared_config_channels(
        snapshot={
            "files": [
                {
                    "scope": "project",
                    "path": "opencode.json",
                    "content": '{"instructions":["' + instruction + '"]}\n',
                }
            ]
        },
        base_workspace={"files": []},
        harness_query=query,
        selected_candidate={
            "channel_plan": [{"channel_id": "instructions_rules"}]
        },
    )
    by_path = {item["path"]: item for item in compiled["files"]}
    config = __import__("json").loads(by_path["opencode.json"]["content"])
    instruction_path = config["instructions"][0]

    assert instruction_path.startswith(".opencode/instructions/autoiter-")
    assert by_path[instruction_path]["content"] == instruction + "\n"
    actual = actual_editor_channel_diffs(
        harness_query=query,
        selected_candidate={
            "channel_plan": [
                {"channel_id": "instructions_rules", "experience_ids": ["exp-a"]}
            ]
        },
        workspace_changes=[
            {"scope": "project", "path": "opencode.json"},
            {"scope": "project", "path": instruction_path},
        ],
        mcp_tool_patches={},
        base_workspace={"files": []},
        candidate_workspace=compiled,
    )
    assert actual == [
        {"channel_id": "instructions_rules", "experience_ids": ["exp-a"]}
    ]


@pytest.mark.parametrize("harness", ["opencode", "pi", "pi-agent", "codex"])
def test_submission_reads_manifest_and_workspace_paths(harness):
    root = Path("/tmp/version")
    normalized = "pi" if harness in {"pi", "pi-agent"} else harness

    assert _snapshot_manifest_path(root, harness) == (
        root / "harness" / normalized / "manifest.json"
    )
    assert _snapshot_workspace_path(root, harness) == (
        root / "harness" / normalized / "workspace.json"
    )


def test_invalid_cached_final_review_still_allows_fresh_retry():
    assert list(_final_review_attempts(None)) == [0, 1]
    assert list(_final_review_attempts({"decision": "reject_delta"})) == []


def test_selected_problem_is_materialized_from_editor_snapshot_not_analyzer_delta():
    calls = {}

    class Repository:
        def read_workspace_snapshot(self, version):
            calls["read_workspace"] = version
            return {"schema": 1, "files": []}

        def read_candidate_snapshot(self, version):
            calls["read_manifest"] = version
            return {"legacy": version}

        def materialize_workspace_candidate(self, **kwargs):
            calls["materialize"] = kwargs
            return kwargs["candidate_label"]

    class Editor:
        def edit(self, **kwargs):
            calls["edit"] = kwargs
            return SimpleNamespace(
                snapshot={
                    "schema": 1,
                    "files": [
                        {
                            "scope": "project",
                            "path": "AGENTS.md",
                            "content": "Use evidence.\n",
                            "executable": False,
                        }
                    ],
                },
                changes=(
                    {
                        "scope": "project",
                        "path": "AGENTS.md",
                        "change": "added",
                    },
                ),
                summary={"rationale": "Use evidence"},
                root="/tmp/editor-job",
                stdout_path="/tmp/editor.stdout",
                stderr_path="/tmp/editor.stderr",
                api_trace_path="/tmp/editor.trace",
            )

    selected = {
        "id": "problem-a",
        "objective": "Fix missed evidence.",
        "channel_plan": [],
        "manifest_delta": {"instructions": ["Analyzer-authored patch"]},
        "validation": {"local_behavior_checks": ["Uses evidence"]},
    }
    version, metadata = materialize_selected_problem(
        repository=Repository(),
        editor=Editor(),
        parent_version="accepted-v1",
        candidate_label="candidate-02",
        editor_job_id="editor-02",
        harness_query={"channels": []},
        selected_candidate=selected,
        evidence=[{"id": "exp-a", "finding": "Evidence was skipped."}],
    )

    assert version == "candidate-02"
    assert calls["edit"]["problem"] == {
        "id": "problem-a",
        "objective": "Fix missed evidence.",
        "channel_plan": [],
        "validation": {"local_behavior_checks": ["Uses evidence"]},
    }
    assert calls["edit"]["current_manifest"] == {"legacy": "accepted-v1"}
    assert calls["materialize"]["workspace"] == calls["edit"]["base_workspace"] | {
        "files": calls["materialize"]["workspace"]["files"]
    }
    assert calls["materialize"]["workspace"]["files"][0]["content"] == "Use evidence.\n"
    assert metadata["workspace_diff"][0]["path"] == "AGENTS.md"
    assert metadata["manifest_delta"] == {}
    assert metadata["workspace_delta"] == {
        "files": [
            {
                "scope": "project",
                "path": "AGENTS.md",
                "change": "added",
                "content": "Use evidence.\n",
                "executable": False,
            }
        ]
    }
    assert metadata["channel_diffs"] == [
        {
            "channel_id": "workspace_file:project:AGENTS.md",
            "experience_ids": [],
        }
    ]
    assert "Analyzer-authored patch" not in str(calls["materialize"])


def test_selected_problem_routes_controller_mcp_file_into_manifest():
    calls = {}

    class Repository:
        def read_workspace_snapshot(self, _version):
            return {"schema": 1, "files": []}

        def read_candidate_snapshot(self, _version):
            return {}

        def materialize_workspace_candidate(self, **kwargs):
            calls.update(kwargs)
            return kwargs["candidate_label"]

    class Editor:
        def edit(self, **_kwargs):
            return SimpleNamespace(
                snapshot={
                    "schema": 1,
                    "files": [
                        {
                            "scope": "project",
                            "path": ".harness-autoiter/mcp-tool-patches.json",
                            "content": '{"lookup_record":{"desc":"Use exact IDs."}}',
                            "executable": False,
                        }
                    ],
                },
                changes=(),
                summary=None,
                root="/tmp/editor-job",
                stdout_path="/tmp/editor.stdout",
                stderr_path="/tmp/editor.stderr",
                api_trace_path="/tmp/editor.trace",
            )

    _version, metadata = materialize_selected_problem(
        repository=Repository(),
        editor=Editor(),
        parent_version="v0",
        candidate_label="candidate-mcp",
        editor_job_id="editor-mcp",
        harness_query={},
        selected_candidate={"id": "mcp", "channel_plan": []},
        evidence=[],
    )

    assert calls["workspace"]["files"] == []
    assert calls["manifest_delta"] == {
        "tool_desc_patches": {"lookup_record": {"desc": "Use exact IDs."}}
    }
    assert metadata["mcp_tool_patch_count"] == 1
    assert metadata["manifest_delta"] == {
        "tool_desc_patches": {"lookup_record": {"desc": "Use exact IDs."}}
    }
    assert metadata["workspace_delta"] == {"files": []}
    assert metadata["channel_diffs"] == [
        {"channel_id": "mcp_tool_description", "experience_ids": []}
    ]


def test_selected_problem_repairs_unclassified_editor_files():
    calls = []

    class Repository:
        def read_workspace_snapshot(self, _version):
            return {"schema": 1, "files": []}

        def read_candidate_snapshot(self, _version):
            return {}

        def materialize_workspace_candidate(self, **kwargs):
            return kwargs["candidate_label"]

    class Editor:
        def edit(self, **kwargs):
            calls.append(kwargs)
            files = [
                {
                    "scope": "project",
                    "path": ".harness-autoiter/mcp-tool-patches.json",
                    "content": '{"execute":{"desc":"Use exact values."}}',
                    "executable": False,
                }
            ]
            changes = []
            if len(calls) == 1:
                files.append(
                    {
                        "scope": "project",
                        "path": "validate.tmp.py",
                        "content": "print('ok')\n",
                        "executable": False,
                    }
                )
                changes.append(
                    {
                        "scope": "project",
                        "path": "validate.tmp.py",
                        "change": "added",
                    }
                )
            return SimpleNamespace(
                snapshot={"schema": 1, "files": files},
                changes=tuple(changes),
                summary=None,
                root="/tmp/editor-job",
                stdout_path="/tmp/editor.stdout",
                stderr_path="/tmp/editor.stderr",
                api_trace_path="/tmp/editor.trace",
            )

    version, metadata = materialize_selected_problem(
        repository=Repository(),
        editor=Editor(),
        parent_version="v0",
        candidate_label="candidate-mcp",
        editor_job_id="editor-mcp",
        harness_query={},
        selected_candidate={
            "id": "mcp",
            "channel_plan": [{"channel_id": "mcp_tool_description"}],
        },
        evidence=[],
    )

    assert version == "candidate-mcp"
    assert len(calls) == 2
    assert "outside discovered channels" in calls[1]["problem"][
        "previous_materialization_error"
    ]
    assert metadata["workspace_diff"] == []
    assert metadata["channel_diffs"] == [
        {"channel_id": "mcp_tool_description", "experience_ids": []}
    ]


def test_selected_problem_compiles_nested_developer_instructions_to_declared_key():
    calls = []

    class Repository:
        def read_workspace_snapshot(self, _version):
            return {"schema": 1, "files": []}

        def read_candidate_snapshot(self, _version):
            return {}

        def materialize_workspace_candidate(self, **kwargs):
            calls.append(kwargs)
            return kwargs["candidate_label"]

    class Editor:
        def edit(self, **_kwargs):
            return SimpleNamespace(
                snapshot={
                    "schema": 1,
                    "files": [
                        {
                            "scope": "project",
                            "path": ".codex/config.toml",
                            "content": (
                                "[project]\n"
                                'developer_instructions = "Probe each interpretation."\n'
                            ),
                            "executable": False,
                        }
                    ],
                },
                changes=(
                    {
                        "scope": "project",
                        "path": ".codex/config.toml",
                        "change": "added",
                    },
                ),
                summary=None,
                root="/tmp/editor-job",
                stdout_path="/tmp/editor.stdout",
                stderr_path="/tmp/editor.stderr",
                api_trace_path="/tmp/editor.trace",
            )

    version, metadata = materialize_selected_problem(
        repository=Repository(),
        editor=Editor(),
        parent_version="v0",
        candidate_label="candidate-config",
        editor_job_id="editor-config",
        harness_query={
            "modifiable_modules": [
                {
                    "id": "developer_instructions",
                    "edit_contract": {
                        "scope": "project",
                        "path": ".codex/config.toml",
                        "mechanism": "config",
                        "key": "developer_instructions",
                    },
                }
            ]
        },
        selected_candidate={
            "id": "ambiguous-query",
            "channel_plan": [
                {
                    "channel_id": "developer_instructions",
                    "experience_ids": ["failure-a"],
                }
            ],
        },
        evidence=[],
    )

    assert version == "candidate-config"
    workspace_file = calls[0]["workspace"]["files"][0]
    assert workspace_file["content"] == (
        'developer_instructions = "Probe each interpretation."\n'
    )
    assert metadata["channel_diffs"] == [
        {
            "channel_id": "developer_instructions",
            "experience_ids": ["failure-a"],
        }
    ]


def test_editor_actual_paths_replace_declared_channel_attribution():
    query = {
        "modifiable_modules": [
            {
                "id": "project_instructions",
                "edit_contract": {
                    "scope": "project",
                    "path": "AGENTS.md",
                    "mechanism": "file",
                },
            },
            {
                "id": "default_project_trust",
                "edit_contract": {
                    "scope": "home",
                    "path": ".pi/agent/settings.json",
                    "mechanism": "config",
                },
            },
        ]
    }
    actual = actual_editor_channel_diffs(
        harness_query=query,
        selected_candidate={
            "channel_plan": [
                {
                    "channel_id": "project_instructions",
                    "experience_ids": ["exp-a"],
                }
            ]
        },
        workspace_changes=[
            {"scope": "project", "path": "AGENTS.md"},
            {"scope": "home", "path": ".pi/agent/settings.json"},
        ],
        mcp_tool_patches={},
    )

    assert actual == [
        {"channel_id": "default_project_trust", "experience_ids": ["exp-a"]},
        {"channel_id": "project_instructions", "experience_ids": ["exp-a"]},
    ]


def test_editor_mcp_attribution_excludes_inherited_parent_description():
    actual = actual_editor_channel_diffs(
        harness_query={},
        selected_candidate={
            "channel_plan": [
                {
                    "channel_id": "mcp_tool_parameter_description",
                    "experience_ids": ["exp-a"],
                }
            ]
        },
        workspace_changes=[],
        base_mcp_tool_patches={
            "execute_sql": {"desc": "Existing guidance."}
        },
        mcp_tool_patches={
            "execute_sql": {
                "params": {"sql": "Use exact columns."},
            }
        },
    )

    assert actual == [
        {
            "channel_id": "mcp_tool_parameter_description",
            "experience_ids": ["exp-a"],
        }
    ]


def test_editor_config_attribution_uses_changed_toml_key():
    query = {
        "modifiable_modules": [
            {
                "id": "developer_instructions",
                "edit_contract": {
                    "scope": "project",
                    "path": ".codex/config.toml",
                    "mechanism": "config",
                    "key": "developer_instructions",
                },
            },
            {
                "id": "personality",
                "edit_contract": {
                    "scope": "project",
                    "path": ".codex/config.toml",
                    "mechanism": "config",
                    "key": "personality",
                },
            },
        ]
    }

    actual = actual_editor_channel_diffs(
        harness_query=query,
        selected_candidate={
            "channel_plan": [
                {
                    "channel_id": "developer_instructions",
                    "experience_ids": ["exp-a"],
                }
            ]
        },
        workspace_changes=[{"scope": "project", "path": ".codex/config.toml"}],
        mcp_tool_patches={},
        base_workspace={"files": []},
        candidate_workspace={
            "files": [
                {
                    "scope": "project",
                    "path": ".codex/config.toml",
                    "content": 'developer_instructions = "Use exact columns."\n',
                }
            ]
        },
    )

    assert actual == [
        {"channel_id": "developer_instructions", "experience_ids": ["exp-a"]}
    ]


def test_editor_config_attribution_rejects_nested_wrong_toml_key():
    query = {
        "modifiable_modules": [
            {
                "id": "developer_instructions",
                "edit_contract": {
                    "scope": "project",
                    "path": ".codex/config.toml",
                    "mechanism": "config",
                    "key": "developer_instructions",
                },
            }
        ]
    }

    actual = actual_editor_channel_diffs(
        harness_query=query,
        selected_candidate={
            "channel_plan": [
                {
                    "channel_id": "developer_instructions",
                    "experience_ids": ["exp-a"],
                }
            ]
        },
        workspace_changes=[{"scope": "project", "path": ".codex/config.toml"}],
        mcp_tool_patches={},
        base_workspace={"files": []},
        candidate_workspace={
            "files": [
                {
                    "scope": "project",
                    "path": ".codex/config.toml",
                    "content": '[project]\ndeveloper_instructions = "inert"\n',
                }
            ]
        },
    )

    assert actual == [
        {
            "channel_id": "workspace_file:project:.codex/config.toml",
            "experience_ids": ["exp-a"],
        }
    ]


def test_editor_mcp_patch_must_use_query_discovered_parameter():
    query = {
        "mcp_editable_points": [
            {"id": "mcp_tool_description", "targets": ["execute_sql"]},
            {
                "id": "mcp_tool_parameter_description",
                "targets": [{"tool": "execute_sql", "parameters": ["sql"]}],
            },
        ]
    }

    _validate_mcp_tool_patches_against_query(
        {"execute_sql": {"params": {"sql": "Use one query."}}}, query
    )
    with pytest.raises(ValueError, match="execute_sql.query"):
        _validate_mcp_tool_patches_against_query(
            {"execute_sql": {"params": {"query": "Use one query."}}}, query
        )


def test_materialization_retries_same_problem_after_invalid_editor_output():
    class Repository:
        def read_workspace_snapshot(self, version):
            assert version == "v0"
            return {"files": []}

        def read_candidate_snapshot(self, version):
            assert version == "v0"
            return {}

        def materialize_workspace_candidate(self, **kwargs):
            assert kwargs["candidate_label"] == "candidate-01"
            return "candidate-01"

    class Editor:
        def __init__(self):
            self.problems = []

        def edit(self, *, job_id, problem, **kwargs):
            del kwargs
            self.problems.append((job_id, dict(problem)))
            if len(self.problems) == 1:
                raise ValueError("candidate artifacts must generalize evidence")
            return SimpleNamespace(
                snapshot={
                    "files": [
                        {
                            "scope": "project",
                            "path": "AGENTS.md",
                            "content": "Return only requested output fields.\n",
                        }
                    ]
                },
                changes=[
                    {"scope": "project", "path": "AGENTS.md", "change": "added"}
                ],
                summary={},
                root=Path("/tmp/editor"),
                stdout_path=Path("/tmp/editor/stdout"),
                stderr_path=Path("/tmp/editor/stderr"),
                api_trace_path=Path("/tmp/editor/trace"),
            )

    editor = Editor()
    version, metadata = materialize_selected_problem(
        repository=Repository(),
        editor=editor,
        parent_version="v0",
        candidate_label="candidate-01",
        editor_job_id="editor-01",
        harness_query={
            "modifiable_modules": [
                {
                    "id": "project_instructions",
                    "edit_contract": {
                        "scope": "project",
                        "path": "AGENTS.md",
                        "mechanism": "file",
                    },
                }
            ]
        },
        selected_candidate={
            "id": "candidate-a",
            "objective": "Generalize output selection.",
            "channel_plan": [
                {
                    "channel_id": "project_instructions",
                    "experience_ids": ["exp-a"],
                }
            ],
        },
        evidence=[{"id": "exp-a"}],
    )

    assert version == "candidate-01"
    assert metadata["channel_diffs"] == [
        {"channel_id": "project_instructions", "experience_ids": ["exp-a"]}
    ]
    assert editor.problems[1][0] == "editor-01-repair-02"
    assert "must generalize" in editor.problems[1][1][
        "previous_materialization_error"
    ]


def test_materialization_repairs_non_list_opencode_instructions():
    class Repository:
        def read_workspace_snapshot(self, version):
            assert version == "v0"
            return {"files": []}

        def read_candidate_snapshot(self, version):
            assert version == "v0"
            return {}

        def materialize_workspace_candidate(self, **kwargs):
            assert kwargs["candidate_label"] == "candidate-01"
            return "candidate-01"

    class Editor:
        def __init__(self):
            self.problems = []

        def edit(self, *, job_id, problem, **kwargs):
            del kwargs
            self.problems.append((job_id, dict(problem)))
            instruction_path = ".opencode/instructions/action.md"
            files = [
                {
                    "scope": "project",
                    "path": "opencode.json",
                    "content": json.dumps(
                        {
                            "instructions": (
                                "Execute confirmed actions."
                                if len(self.problems) == 1
                                else [instruction_path]
                            )
                        }
                    ),
                }
            ]
            if len(self.problems) > 1:
                files.append(
                    {
                        "scope": "project",
                        "path": instruction_path,
                        "content": "Execute confirmed actions.\n",
                    }
                )
            return SimpleNamespace(
                snapshot={"files": files},
                changes=[],
                summary={},
                root=Path("/tmp/editor"),
                stdout_path=Path("/tmp/editor/stdout"),
                stderr_path=Path("/tmp/editor/stderr"),
                api_trace_path=Path("/tmp/editor/trace"),
            )

    editor = Editor()
    _version, metadata = materialize_selected_problem(
        repository=Repository(),
        editor=editor,
        parent_version="v0",
        candidate_label="candidate-01",
        editor_job_id="editor-01",
        harness_query={
            "modifiable_modules": [
                {
                    "id": "instructions_rules",
                    "edit_contract": {
                        "scope": "project",
                        "path": "opencode.json",
                        "mechanism": "config",
                        "key": "instructions",
                    },
                }
            ]
        },
        selected_candidate={
            "id": "candidate-a",
            "objective": "Execute confirmed actions.",
            "channel_plan": [
                {
                    "channel_id": "instructions_rules",
                    "experience_ids": ["exp-a"],
                }
            ],
        },
        evidence=[{"id": "exp-a"}],
    )

    assert metadata["channel_diffs"] == [
        {"channel_id": "instructions_rules", "experience_ids": ["exp-a"]}
    ]
    assert editor.problems[1][0] == "editor-01-repair-02"
    assert "must be a list" in editor.problems[1][1][
        "previous_materialization_error"
    ]


def _decision():
    return {
        "decision": "materialize_and_rollout",
        "selected_candidate_id": "a",
        "exploration_plan": {
            "open_problem_ids": ["problem-a", "problem-b"],
            "chosen_problem_id": "problem-a",
            "budget_strategy": "Use the minimum rollout that can test the question.",
            "turning_point": "Move on if this candidate has attributed regressions.",
        },
        "rollout_request": {
            "task_ids": ["0", "1", "2", "3", "4"],
            "rationale": "Evidence-backed tasks.",
            "local_success_criteria": ["Execution follows confirmation."],
        },
    }


def test_main_decision_accepts_five_explicit_tasks_and_adapter_valid_delta():
    validate_main_decision(
        _decision(),
        train_task_ids=("0", "1", "2", "3", "4"),
        candidate_evidence={"a": ("ev0",)},
        candidate_sides={"a": "adjustment"},
        evidence_to_task={"ev0": "0"},
    )


def test_main_decision_accepts_controller_bounded_residual_probe():
    output = _decision()
    output["rollout_request"]["task_ids"] = ["0", "1", "2"]

    validate_main_decision(
        output,
        train_task_ids=("0", "1", "2", "3", "4"),
        candidate_evidence={"a": ("ev0",)},
        candidate_sides={"a": "adjustment"},
        evidence_to_task={"ev0": "0"},
        min_rollout_tasks=2,
        max_rollout_tasks=3,
    )


def test_main_decision_accepts_incumbent_direct_train_evidence():
    validate_main_decision(
        _decision(),
        train_task_ids=("0", "1", "2", "3", "4"),
        candidate_evidence={"a": ()},
        candidate_direct_tasks={"a": ("0", "1")},
        candidate_sides={"a": "incumbent"},
        evidence_to_task={},
    )


def test_main_decision_requires_exploration_plan():
    output = _decision()
    output.pop("exploration_plan")

    with pytest.raises(ValueError, match="exploration_plan"):
        validate_main_decision(
            output,
            train_task_ids=("0", "1", "2", "3", "4"),
            candidate_evidence={"a": ("ev0",)},
            candidate_sides={"a": "adjustment"},
            evidence_to_task={"ev0": "0"},
        )


def test_main_decision_requires_chosen_open_problem():
    output = _decision()
    output["exploration_plan"]["chosen_problem_id"] = "not-open"

    with pytest.raises(ValueError, match="chosen problem"):
        validate_main_decision(
            output,
            train_task_ids=("0", "1", "2", "3", "4"),
            candidate_evidence={"a": ("ev0",)},
            candidate_sides={"a": "adjustment"},
            evidence_to_task={"ev0": "0"},
        )


def test_main_decision_canonicalization_lists_chosen_problem_as_open():
    output = _decision()
    output["exploration_plan"]["open_problem_ids"] = ["problem-b"]

    canonicalize_main_decision(output)

    assert output["exploration_plan"]["open_problem_ids"] == [
        "problem-a",
        "problem-b",
    ]
    validate_main_decision(
        output,
        train_task_ids=("0", "1", "2", "3", "4"),
        candidate_evidence={"a": ("ev0",)},
        candidate_sides={"a": "adjustment"},
        evidence_to_task={"ev0": "0"},
    )


def test_main_prompt_prioritizes_breadth_before_unsupported_revision():
    assert "maximize expected final harness quality" in MAIN_AGENT_SYSTEM
    assert "several distinct, high-potential\nbehavior hypotheses" in MAIN_AGENT_SYSTEM
    assert "only after it shows attributable positive evidence" in MAIN_AGENT_SYSTEM
    assert "merely to balance candidate types" in MAIN_AGENT_SYSTEM
    assert "attributable positive evidence" in FINAL_MAIN_AGENT_SYSTEM
    assert "does not improve attributable recoveries" in MAIN_AGENT_SYSTEM
    assert "current TRAIN-accepted champion" in MAIN_AGENT_SYSTEM
    assert "pending revision competes with the other open problems" in MAIN_AGENT_SYSTEM
    assert "adds one bounded revision to the open\nportfolio" in FINAL_MAIN_AGENT_SYSTEM


def test_main_decision_does_not_require_candidate_type_quotas():
    output = _decision()
    validate_main_decision(
        output,
        train_task_ids=("0", "1", "2", "3", "4"),
        candidate_evidence={"a": ("ev0",), "b": ("ev1",)},
        candidate_sides={"a": "adjustment", "b": "reusable"},
        evidence_to_task={"ev0": "0", "ev1": "1"},
    )


def test_pending_revision_competes_with_other_hypotheses():
    portfolio = [
        {"side": "adjustment", "candidate": {"id": "new-problem"}},
        {"side": "revision", "candidate": {"id": "revision-a"}},
    ]

    available = selectable_portfolio(
        portfolio,
        attempted_candidate_ids=(),
        pending_revision_candidate_ids=("revision-a",),
    )

    assert [item["candidate"]["id"] for item in available] == [
        "new-problem",
        "revision-a",
    ]


def test_failure_only_candidates_wait_while_supported_options_remain():
    available = defer_failure_only_portfolio(
        [
            {
                "side": "adjustment",
                "candidate": {"id": "failure-only"},
                "evidence_profile": {"support_tier": "failure_only"},
            },
            {
                "side": "reusable",
                "candidate": {"id": "supported"},
                "evidence_profile": {"support_tier": "within_task_contrast"},
            },
            {
                "side": "revision",
                "candidate": {"id": "revision"},
                "evidence_profile": {"support_tier": "failure_only"},
            },
        ]
    )

    assert [item["candidate"]["id"] for item in available] == [
        "supported",
        "revision",
    ]


def test_failure_only_candidates_return_after_supported_options_are_exhausted():
    portfolio = [
        {
            "side": "adjustment",
            "candidate": {"id": "failure-a"},
            "evidence_profile": {"support_tier": "failure_only"},
        },
        {
            "side": "reusable",
            "candidate": {"id": "failure-b"},
            "evidence_profile": {"support_tier": "failure_only"},
        },
    ]

    assert defer_failure_only_portfolio(portfolio) == portfolio


def test_promotion_portfolio_requires_failed_task_headroom():
    portfolio = [
        {
            "side": "reusable",
            "candidate": {"id": "success-only"},
            "evidence_profile": {
                "all_failed_task_ids": [],
                "failed_trial_count": 0,
            },
        },
        {
            "side": "adjustment",
            "candidate": {"id": "failed-target"},
            "evidence_profile": {
                "all_failed_task_ids": ["bird_1"],
                "failed_trial_count": 1,
            },
        },
    ]

    available = promotion_headroom_portfolio(portfolio)

    assert [item["candidate"]["id"] for item in available] == ["failed-target"]


def test_reusable_candidate_gets_headroom_from_mixed_source_task():
    portfolio = [
        {
            "side": "reusable",
            "candidate": {"id": "demonstrated-recovery"},
            "evidence_profile": {
                "all_passed_task_ids": ["mixed-task"],
                "all_failed_task_ids": [],
                "support_tier": "pass_only",
            },
        },
        {
            "side": "reusable",
            "candidate": {"id": "already-solved"},
            "evidence_profile": {
                "all_passed_task_ids": ["passing-task"],
                "all_failed_task_ids": [],
                "support_tier": "pass_only",
            },
        },
    ]

    available = promotion_headroom_portfolio(
        portfolio,
        baseline_task_outcomes={
            "mixed-task": "fail",
            "passing-task": "pass",
        },
    )

    assert [item["candidate"]["id"] for item in available] == [
        "demonstrated-recovery"
    ]
    assert available[0]["conversion_task_ids"] == ["mixed-task"]


def test_main_decision_requires_a_declared_conversion_task():
    output = _decision()
    output["rollout_request"]["task_ids"] = ["0", "1", "2", "3", "4"]

    with pytest.raises(ValueError, match="conversion task"):
        validate_main_decision(
            output,
            train_task_ids=("0", "1", "2", "3", "4", "failed"),
            candidate_evidence={"a": ("ev0",)},
            candidate_sides={"a": "adjustment"},
            candidate_conversion_tasks={"a": ("failed",)},
            evidence_to_task={"ev0": "0"},
        )

    output["rollout_request"]["task_ids"][-1] = "failed"
    validate_main_decision(
        output,
        train_task_ids=("0", "1", "2", "3", "4", "failed"),
        candidate_evidence={"a": ("ev0",)},
        candidate_sides={"a": "adjustment"},
        candidate_conversion_tasks={"a": ("failed",)},
        evidence_to_task={"ev0": "0"},
    )


def test_controller_assigns_each_rollout_task_one_semantic_role():
    roles = assign_rollout_roles(
        task_ids=("convert", "control", "canary", "diagnostic"),
        conversion_task_ids=("convert",),
        direct_task_ids=("convert", "control"),
        baseline_task_outcomes={
            "convert": "fail",
            "control": "pass",
            "canary": "pass",
            "diagnostic": "fail",
        },
    )

    assert roles == {
        "conversion_tasks": ["convert"],
        "positive_controls": ["control"],
        "preservation_canaries": ["canary"],
        "diagnostic_tasks": ["diagnostic"],
    }


def test_possibly_related_pass_gain_is_an_uncertain_recovery():
    adjustment = {
        "primary_problem": {
            "task_assessments": [
                {
                    "task_id": "bird_1",
                    "status": "recovered",
                    "relation": "possibly_related",
                    "outcome_summary": {
                        "reference_pass_count": 0,
                        "candidate_pass_count": 1,
                    },
                },
                {
                    "task_id": "bird_2",
                    "status": "recovered",
                    "relation": "not_attributed",
                    "outcome_summary": {
                        "reference_pass_count": 0,
                        "candidate_pass_count": 1,
                    },
                },
            ]
        }
    }

    assert uncertain_recovery_task_ids(adjustment) == {"bird_1"}


def test_possibly_related_recovery_is_not_promotion_evidence():
    output = {
        "decision": "accept_delta",
        "selected_version": "candidate-01",
        "rationale": "One trial changed, but attribution remains uncertain.",
        "evidence": {
            "recovered_task_ids": [],
            "uncertain_recovery_task_ids": ["0"],
            "preserved_task_ids": ["0"],
            "attributable_regression_task_ids": [],
            "unresolved_findings": ["The recovery is only possibly related."],
        },
        "budget_disposition": {
            "remaining_creations_after_this_decision": 4,
            "minimum_next_rollout_creations": 10,
            "rollout_possible": False,
            "minimum_next_iteration_creations": MIN_CANDIDATE_ITERATION_CREATIONS,
            "iteration_possible": False,
        },
    }
    reusable = {
        "preservation": {
            "preserved_task_ids": ["0"],
            "attributable_regressions": [],
        }
    }
    adjustment = {
        "primary_problem": {
            "recommendation": "accept",
            "task_assessments": [
                {
                    "task_id": "0",
                    "status": "recovered",
                    "relation": "possibly_related",
                    "outcome_summary": {
                        "reference_pass_count": 0,
                        "candidate_pass_count": 1,
                    },
                }
            ],
        }
    }

    with pytest.raises(ValueError, match="attributable positive evidence"):
        validate_final_decision(
            output,
            candidate_version="candidate-01",
            task_ids=("0",),
            reusable=reusable,
            adjustment=adjustment,
            budget_status={"remaining": 5},
        )


def test_possibly_related_pass_gain_can_request_independent_confirmation():
    output = {
        "decision": "confirm_delta",
        "selected_version": "v0",
        "rationale": "A recorded pass gain exists but attribution is uncertain.",
        "evidence": {
            "recovered_task_ids": [],
            "uncertain_recovery_task_ids": ["0"],
            "preserved_task_ids": ["0"],
            "attributable_regression_task_ids": [],
            "unresolved_findings": ["Confirm the possibly-related recovery."],
        },
        "budget_disposition": {
            "remaining_creations_after_this_decision": 30,
            "minimum_next_rollout_creations": 10,
            "rollout_possible": True,
            "minimum_next_iteration_creations": MIN_CANDIDATE_ITERATION_CREATIONS,
            "iteration_possible": True,
        },
    }
    reusable = {
        "preservation": {
            "preserved_task_ids": ["0"],
            "attributable_regressions": [],
        }
    }
    adjustment = {
        "primary_problem": {
            "recommendation": "accept",
            "task_assessments": [
                {
                    "task_id": "0",
                    "status": "recovered",
                    "relation": "possibly_related",
                    "outcome_summary": {
                        "reference_pass_count": 0,
                        "candidate_pass_count": 1,
                    },
                }
            ],
        }
    }

    validate_final_decision(
        output,
        candidate_version="candidate-01",
        base_version="v0",
        task_ids=("0",),
        reusable=reusable,
        adjustment=adjustment,
        budget_status={"remaining": 31},
        remaining_after_decision=30,
        require_primary_metric_improvement=False,
    )


def test_final_review_can_route_a_post_analyzer_replan_candidate():
    output = {
        "decision": "replan_problem",
        "selected_version": "v0",
        "rationale": "The tested cause was falsified and a distinct blocker is visible.",
        "evidence": {
            "recovered_task_ids": [],
            "uncertain_recovery_task_ids": [],
            "preserved_task_ids": ["bird_1"],
            "attributable_regression_task_ids": [],
            "unresolved_findings": ["Replan around entity cardinality."],
        },
        "budget_disposition": {
            "remaining_creations_after_this_decision": 30,
            "minimum_next_rollout_creations": 10,
            "rollout_possible": True,
            "minimum_next_iteration_creations": MIN_CANDIDATE_ITERATION_CREATIONS,
            "iteration_possible": True,
        },
    }
    reusable = {
        "preservation": {
            "preserved_task_ids": ["bird_1"],
            "attributable_regressions": [],
        }
    }
    adjustment = {
        "primary_problem": {
            "recommendation": "refine",
            "task_assessments": [
                {
                    "task_id": "bird_1",
                    "status": "still_failing",
                    "relation": "not_attributed",
                }
            ],
        },
        "replan_candidate": {"id": "replan-count-distinct"},
    }

    validate_final_decision(
        output,
        candidate_version="candidate-01",
        base_version="v0",
        task_ids=("bird_1",),
        reusable=reusable,
        adjustment=adjustment,
        budget_status={"remaining": 31},
        remaining_after_decision=30,
    )


def test_promotion_metric_requires_positive_pass1_delta():
    output = {
        "decision": "accept_delta",
        "selected_version": "candidate-01",
        "rationale": "The second repeat improved.",
        "evidence": {
            "recovered_task_ids": ["0"],
            "preserved_task_ids": ["0"],
            "attributable_regression_task_ids": [],
            "unresolved_findings": [],
        },
        "budget_disposition": {
            "remaining_creations_after_this_decision": 4,
            "minimum_next_rollout_creations": 10,
            "rollout_possible": False,
            "minimum_next_iteration_creations": MIN_CANDIDATE_ITERATION_CREATIONS,
            "iteration_possible": False,
        },
    }
    reusable = {
        "preservation": {
            "preserved_task_ids": ["0"],
            "attributable_regressions": [],
        }
    }
    adjustment = {
        "primary_problem": {
            "recommendation": "accept",
            "task_assessments": [
                {
                    "task_id": "0",
                    "status": "recovered",
                    "relation": "attributed",
                    "outcome_summary": {
                        "reference_pass_count": 0,
                        "candidate_pass_count": 1,
                    },
                }
            ],
        }
    }

    validate_final_decision(
        output,
        candidate_version="candidate-01",
        task_ids=("0",),
        reusable=reusable,
        adjustment=adjustment,
        budget_status={"remaining": 5},
        promotion_metric="pass_at_1",
        rollout_metrics={"pass_at_1": 0.4, "pass_at_2": 0.8},
        reference_rollout_metrics={"pass_at_1": 0.4, "pass_at_2": 0.6},
        require_primary_metric_improvement=False,
    )

    with pytest.raises(ValueError, match="infrastructure-valid paired evidence"):
        validate_final_decision(
            output,
            candidate_version="candidate-01",
            task_ids=("0",),
            reusable=reusable,
            adjustment=adjustment,
            budget_status={"remaining": 5},
            promotion_metric="pass_at_1",
            rollout_metrics={
                "pass_at_1": 0.8,
                "trial_success_rate": 0.8,
                "paired_infrastructure_valid": False,
            },
            reference_rollout_metrics={"pass_at_1": 0.4},
        )

    with pytest.raises(ValueError, match="positive pass_at_1 delta"):
        validate_final_decision(
            output,
            candidate_version="candidate-01",
            task_ids=("0",),
            reusable=reusable,
            adjustment=adjustment,
            budget_status={"remaining": 5},
            promotion_metric="pass_at_1",
            rollout_metrics={"pass_at_1": 0.4, "pass_at_2": 0.8},
            reference_rollout_metrics={"pass_at_1": 0.4, "pass_at_2": 0.6},
        )

    validate_final_decision(
        output,
        candidate_version="candidate-01",
        task_ids=("0",),
        reusable=reusable,
        adjustment=adjustment,
        budget_status={"remaining": 5},
        promotion_metric="pass_at_2",
        rollout_metrics={"pass_at_1": 0.4, "pass_at_2": 0.8},
        reference_rollout_metrics={"pass_at_1": 0.4, "pass_at_2": 0.6},
    )


def test_attempted_revision_releases_other_hypotheses():
    portfolio = [
        {"side": "adjustment", "candidate": {"id": "new-problem"}},
        {"side": "revision", "candidate": {"id": "revision-a"}},
    ]

    available = selectable_portfolio(
        portfolio,
        attempted_candidate_ids=("revision-a",),
        pending_revision_candidate_ids=("revision-a",),
    )

    assert [item["candidate"]["id"] for item in available] == ["new-problem"]


def test_cached_main_decision_must_follow_current_review_chain():
    output = {
        "harness_version": "candidate-02",
        "candidate": {
            "label": "candidate-02",
            "parent_version": "v0",
            "source_candidate_ids": ["revision-a"],
        },
    }
    available = [{"candidate": {"id": "revision-a"}, "side": "revision"}]

    validate_cached_main_decision(
        output,
        parent_version="v0",
        candidate_label="candidate-02",
        available=available,
    )

    with pytest.raises(RuntimeError, match="current review chain"):
        validate_cached_main_decision(
            output,
            parent_version="v0",
            candidate_label="candidate-02",
            available=[{"candidate": {"id": "new-problem"}}],
        )


def test_incumbent_control_can_be_ordered_before_new_hypotheses():
    portfolio = [
        {"side": "adjustment", "candidate": {"id": "new-problem"}},
        {"side": "incumbent", "candidate": {"id": "accepted-control"}},
    ]

    ordered = prioritize_incumbent_controls(portfolio)

    assert [item["candidate"]["id"] for item in ordered] == [
        "accepted-control",
        "new-problem",
    ]


def test_main_decision_rejects_controller_repeat_override():
    output = _decision()
    output["rollout_request"]["repeats"] = 3
    with pytest.raises(ValueError, match="fixed"):
        validate_main_decision(
            output,
            train_task_ids=("0", "1", "2", "3", "4"),
            candidate_evidence={"a": ("ev0",)},
            candidate_sides={"a": "adjustment"},
            evidence_to_task={"ev0": "0"},
        )


def test_main_decision_rejects_candidate_outside_portfolio():
    output = _decision()
    output["selected_candidate_id"] = "missing"

    with pytest.raises(ValueError, match="unavailable candidate"):
        validate_main_decision(
            output,
            train_task_ids=("0", "1", "2", "3", "4"),
            candidate_evidence={"a": ("ev0",)},
            candidate_sides={"a": "adjustment"},
            evidence_to_task={"ev0": "0"},
        )


def test_final_decision_submits_only_tested_candidate_with_analyzer_evidence():
    output = {
        "decision": "accept_delta",
        "selected_version": "candidate-04",
        "rationale": "Recovered targeted behavior without attributable regression.",
        "evidence": {
            "recovered_task_ids": ["0"],
            "preserved_task_ids": ["0", "1"],
            "attributable_regression_task_ids": [],
            "unresolved_findings": ["Task 1 still has a local failure trial."],
        },
        "budget_disposition": {
            "remaining_creations_after_this_decision": 4,
            "minimum_next_rollout_creations": 10,
            "rollout_possible": False,
            "minimum_next_iteration_creations": MIN_CANDIDATE_ITERATION_CREATIONS,
            "iteration_possible": False,
        },
    }
    reusable = {
        "preservation": {
            "preserved_task_ids": ["0", "1"],
            "attributable_regressions": [],
            "candidate_recommendation": "accept",
        }
    }
    adjustment = {
        "primary_problem": {
            "recommendation": "accept",
            "task_assessments": [
                {
                    "task_id": "0",
                    "status": "stable_success",
                    "relation": "attributed",
                },
                {"task_id": "1", "status": "stable_success", "relation": "not_attributed"},
            ],
        }
    }

    validate_final_decision(
        output,
        candidate_version="candidate-04",
        task_ids=("0", "1"),
        reusable=reusable,
        adjustment=adjustment,
        budget_status={"remaining": 5},
    )


def test_final_decision_rejects_residual_probe_promotion():
    with pytest.raises(ValueError, match="not promotion eligible"):
        validate_final_decision(
            {"decision": "accept_delta", "selected_version": "candidate-01"},
            candidate_version="candidate-01",
            task_ids=(),
            reusable={},
            adjustment={},
            budget_status={"remaining": 10},
            promotion_eligible=False,
        )


def test_final_decision_rejects_unclassified_actual_workspace_change():
    with pytest.raises(ValueError, match="unclassified workspace"):
        validate_final_decision(
            {"decision": "accept_delta", "selected_version": "candidate-01"},
            candidate_version="candidate-01",
            task_ids=(),
            reusable={},
            adjustment={},
            budget_status={"remaining": 10},
            tested_candidate={
                "channel_diffs": [
                    {
                        "channel_id": "workspace_file:project:validate.py",
                        "experience_ids": ["exp-a"],
                    }
                ]
            },
        )


def test_final_decision_rejects_no_effect_acceptance():
    output = {
        "decision": "accept_delta",
        "selected_version": "candidate-04",
        "rationale": "The behavior was preserved.",
        "evidence": {
            "recovered_task_ids": [],
            "preserved_task_ids": ["0"],
            "attributable_regression_task_ids": [],
            "unresolved_findings": [],
        },
        "budget_disposition": {
            "remaining_creations_after_this_decision": 4,
            "minimum_next_rollout_creations": 10,
            "rollout_possible": False,
            "minimum_next_iteration_creations": MIN_CANDIDATE_ITERATION_CREATIONS,
            "iteration_possible": False,
        },
    }
    reusable = {
        "preservation": {
            "preserved_task_ids": ["0"],
            "attributable_regressions": [],
            "candidate_recommendation": "accept",
        }
    }
    adjustment = {
        "primary_problem": {
            "recommendation": "accept",
            "further_rollout_needed": False,
            "task_assessments": [
                {
                    "task_id": "0",
                    "status": "stable_success",
                    "relation": "not_attributed",
                }
            ],
        }
    }

    with pytest.raises(ValueError, match="attributable positive evidence"):
        validate_final_decision(
            output,
            candidate_version="candidate-04",
            task_ids=("0",),
            reusable=reusable,
            adjustment=adjustment,
            budget_status={"remaining": 5},
        )


def test_final_decision_rejects_behavioral_relabel_without_pass_gain():
    output = {
        "decision": "accept_delta",
        "selected_version": "candidate-04",
        "rationale": "Visible behavior became more consistent.",
        "evidence": {
            "recovered_task_ids": [],
            "preserved_task_ids": ["0"],
            "attributable_regression_task_ids": [],
            "unresolved_findings": [],
        },
        "budget_disposition": {
            "remaining_creations_after_this_decision": 4,
            "minimum_next_rollout_creations": 10,
            "rollout_possible": False,
            "minimum_next_iteration_creations": MIN_CANDIDATE_ITERATION_CREATIONS,
            "iteration_possible": False,
        },
    }
    reusable = {
        "preservation": {
            "preserved_task_ids": ["0"],
            "attributable_regressions": [],
            "candidate_recommendation": "accept",
        }
    }
    adjustment = {
        "primary_problem": {
            "recommendation": "accept",
            "further_rollout_needed": False,
            "task_assessments": [
                {
                    "task_id": "0",
                    "status": "stable_success",
                    "relation": "attributed",
                    "outcome_summary": {
                        "reference_pass_count": 1,
                        "reference_trial_count": 2,
                        "candidate_pass_count": 1,
                        "candidate_trial_count": 2,
                    },
                }
            ],
        }
    }

    with pytest.raises(ValueError, match="attributable positive evidence"):
        validate_final_decision(
            output,
            candidate_version="candidate-04",
            task_ids=("0",),
            reusable=reusable,
            adjustment=adjustment,
            budget_status={"remaining": 5},
        )


def test_final_decision_rejects_local_gain_without_positive_primary_metric_delta():
    output = {
        "decision": "accept_delta",
        "selected_version": "candidate-04",
        "rationale": "One target task improved.",
        "evidence": {
            "recovered_task_ids": ["0"],
            "preserved_task_ids": ["0", "1"],
            "attributable_regression_task_ids": [],
            "unresolved_findings": ["Task 1 had an unattributed raw loss."],
        },
        "budget_disposition": {
            "remaining_creations_after_this_decision": 4,
            "minimum_next_rollout_creations": 10,
            "rollout_possible": False,
            "minimum_next_iteration_creations": MIN_CANDIDATE_ITERATION_CREATIONS,
            "iteration_possible": False,
        },
    }
    reusable = {
        "preservation": {
            "preserved_task_ids": ["0", "1"],
            "attributable_regressions": [],
            "candidate_recommendation": "accept",
        }
    }
    adjustment = {
        "primary_problem": {
            "recommendation": "accept",
            "further_rollout_needed": False,
            "task_assessments": [
                {
                    "task_id": "0",
                    "status": "stable_success",
                    "relation": "attributed",
                    "outcome_summary": {
                        "reference_pass_count": 1,
                        "candidate_pass_count": 2,
                    },
                },
                {
                    "task_id": "1",
                    "status": "mixed",
                    "relation": "not_attributed",
                    "outcome_summary": {
                        "reference_pass_count": 2,
                        "candidate_pass_count": 1,
                    },
                },
            ],
        }
    }

    with pytest.raises(ValueError, match="positive pass_at_1 delta"):
        validate_final_decision(
            output,
            candidate_version="candidate-04",
            task_ids=("0", "1"),
            reusable=reusable,
            adjustment=adjustment,
            budget_status={"remaining": 5},
            promotion_metric="pass_at_1",
            rollout_metrics={"pass_at_1": 0.5},
            reference_rollout_metrics={"pass_at_1": 0.5},
        )


def test_deterministic_reject_keeps_parent_after_invalid_promotion():
    output = deterministic_reject_decision(
        base_version="incumbent-00",
        reusable={
            "preservation": {
                "preserved_task_ids": ["0", "1"],
                "attributable_regressions": [],
            }
        },
        adjustment={
            "primary_problem": {
                "task_assessments": [
                    {
                        "task_id": "0",
                        "status": "stable_success",
                        "relation": "attributed",
                        "outcome_summary": {
                            "reference_pass_count": 1,
                            "candidate_pass_count": 2,
                        },
                    }
                ]
            }
        },
        remaining=12,
        failure_reason="aggregate delta was not positive",
    )

    assert output["decision"] == "reject_delta"
    assert output["selected_version"] == "incumbent-00"
    assert output["evidence"]["recovered_task_ids"] == ["0"]
    assert output["budget_disposition"]["remaining_creations_after_this_decision"] == 12


def test_budget_exhausted_review_rejects_refine_acceptance():
    output = {
        "decision": "accept_delta",
        "selected_version": "candidate-04",
        "rationale": "No preservation regression was attributed.",
        "evidence": {
            "recovered_task_ids": ["0"],
            "preserved_task_ids": ["0", "1"],
            "attributable_regression_task_ids": [],
            "unresolved_findings": ["Task 1 still needs a bounded revision."],
        },
        "budget_disposition": {
            "remaining_creations_after_this_decision": 4,
            "minimum_next_rollout_creations": 10,
            "rollout_possible": False,
            "minimum_next_iteration_creations": MIN_CANDIDATE_ITERATION_CREATIONS,
            "iteration_possible": False,
        },
    }
    reusable = {
        "preservation": {
            "preserved_task_ids": ["0", "1"],
            "attributable_regressions": [],
            "candidate_recommendation": "accept",
        }
    }
    adjustment = {
        "primary_problem": {
            "recommendation": "refine",
            "further_rollout_needed": True,
            "task_assessments": [
                {"task_id": "0", "status": "recovered", "relation": "attributed"},
                {"task_id": "1", "status": "mixed", "relation": "attributed"},
            ],
        }
    }

    with pytest.raises(ValueError, match="failures attributed to the changed channel"):
        validate_final_decision(
            output,
            candidate_version="candidate-04",
            task_ids=("0", "1"),
            reusable=reusable,
            adjustment=adjustment,
            budget_status={"remaining": 5},
        )


def test_budget_exhausted_review_allows_gain_with_unrelated_refine_request():
    output = {
        "decision": "accept_delta",
        "selected_version": "candidate-04",
        "rationale": "The changed behavior improved without attributable regression.",
        "evidence": {
            "recovered_task_ids": ["0"],
            "preserved_task_ids": ["0", "1"],
            "attributable_regression_task_ids": [],
            "unresolved_findings": ["Task 1 remains unrelated and still failing."],
        },
        "budget_disposition": {
            "remaining_creations_after_this_decision": 4,
            "minimum_next_rollout_creations": 10,
            "rollout_possible": False,
            "minimum_next_iteration_creations": MIN_CANDIDATE_ITERATION_CREATIONS,
            "iteration_possible": False,
        },
    }
    reusable = {
        "preservation": {
            "preserved_task_ids": ["0", "1"],
            "attributable_regressions": [],
            "candidate_recommendation": "uncertain",
        }
    }
    adjustment = {
        "primary_problem": {
            "recommendation": "refine",
            "further_rollout_needed": True,
            "task_assessments": [
                {
                    "task_id": "0",
                    "status": "recovered",
                    "relation": "attributed",
                    "outcome_summary": {
                        "reference_pass_count": 0,
                        "candidate_pass_count": 2,
                    },
                },
                {
                    "task_id": "1",
                    "status": "still_failing",
                    "relation": "not_attributed",
                    "outcome_summary": {
                        "reference_pass_count": 0,
                        "candidate_pass_count": 0,
                    },
                },
            ],
        }
    }

    validate_final_decision(
        output,
        candidate_version="candidate-04",
        task_ids=("0", "1"),
        reusable=reusable,
        adjustment=adjustment,
        budget_status={"remaining": 5},
    )


def test_budget_exhausted_review_allows_accepted_gain_needing_more_confidence():
    output = {
        "decision": "accept_delta",
        "selected_version": "candidate-04",
        "rationale": "Paired evidence already shows an attributable gain without regression.",
        "evidence": {
            "recovered_task_ids": ["0"],
            "preserved_task_ids": ["0", "1"],
            "attributable_regression_task_ids": [],
            "unresolved_findings": ["More rollout would improve confidence."],
        },
        "budget_disposition": {
            "remaining_creations_after_this_decision": 4,
            "minimum_next_rollout_creations": 10,
            "rollout_possible": False,
            "minimum_next_iteration_creations": MIN_CANDIDATE_ITERATION_CREATIONS,
            "iteration_possible": False,
        },
    }
    reusable = {
        "preservation": {
            "preserved_task_ids": ["0", "1"],
            "attributable_regressions": [],
            "candidate_recommendation": "accept",
        }
    }
    adjustment = {
        "primary_problem": {
            "recommendation": "accept",
            "further_rollout_needed": True,
            "task_assessments": [
                {
                    "task_id": "0",
                    "status": "recovered",
                    "relation": "attributed",
                    "outcome_summary": {
                        "reference_pass_count": 0,
                        "candidate_pass_count": 2,
                    },
                },
                {
                    "task_id": "1",
                    "status": "stable_success",
                    "relation": "not_attributed",
                    "outcome_summary": {
                        "reference_pass_count": 2,
                        "candidate_pass_count": 2,
                    },
                },
            ],
        }
    }

    validate_final_decision(
        output,
        candidate_version="candidate-04",
        task_ids=("0", "1"),
        reusable=reusable,
        adjustment=adjustment,
        budget_status={"remaining": 5},
    )


def test_review_decision_can_fall_back_to_direct_parent():
    output = {
        "decision": "reject_delta",
        "selected_version": "accepted-v1",
        "rationale": "The new candidate regressed a preserved behavior.",
        "evidence": {
            "recovered_task_ids": [],
            "preserved_task_ids": [],
            "attributable_regression_task_ids": ["0"],
            "unresolved_findings": ["Revision still needs rollout."],
        },
        "budget_disposition": {
            "remaining_creations_after_this_decision": 4,
            "minimum_next_rollout_creations": 10,
            "rollout_possible": False,
            "minimum_next_iteration_creations": MIN_CANDIDATE_ITERATION_CREATIONS,
            "iteration_possible": False,
        },
    }
    reusable = {
        "preservation": {
            "preserved_task_ids": [],
            "attributable_regressions": [{"task_id": "0"}],
        }
    }
    adjustment = {
        "primary_problem": {
            "recommendation": "reject",
            "task_assessments": [{"task_id": "0", "status": "regressed"}],
        }
    }

    validate_final_decision(
        output,
        candidate_version="candidate-v2",
        base_version="accepted-v1",
        task_ids=("0",),
        reusable=reusable,
        adjustment=adjustment,
        budget_status={"remaining": 5},
    )


def test_final_decision_cannot_accept_only_non_attributed_recovery():
    output = {
        "decision": "accept_delta",
        "selected_version": "candidate-04",
        "rationale": "The visible recovery was unrelated, but no attributable regression occurred.",
        "evidence": {
            "recovered_task_ids": [],
            "preserved_task_ids": ["0"],
            "attributable_regression_task_ids": [],
            "unresolved_findings": ["Task 0 changed for a non-attributed reason."],
        },
        "budget_disposition": {
            "remaining_creations_after_this_decision": 4,
            "minimum_next_rollout_creations": 10,
            "rollout_possible": False,
            "minimum_next_iteration_creations": MIN_CANDIDATE_ITERATION_CREATIONS,
            "iteration_possible": False,
        },
    }
    reusable = {
        "preservation": {
            "preserved_task_ids": ["0"],
            "attributable_regressions": [],
        }
    }
    adjustment = {
        "primary_problem": {
            "recommendation": "accept",
            "task_assessments": [
                {"task_id": "0", "status": "recovered", "relation": "not_attributed"}
            ],
        }
    }

    with pytest.raises(ValueError, match="attributable positive evidence"):
        validate_final_decision(
            output,
            candidate_version="candidate-04",
            task_ids=("0",),
            reusable=reusable,
            adjustment=adjustment,
            budget_status={"remaining": 5},
        )


def test_final_decision_is_canonicalized_from_structured_inputs():
    output = {
        "decision": "accept_delta",
        "selected_version": "candidate",
        "evidence": {
            "recovered_task_ids": ["wrong"],
            "preserved_task_ids": [],
            "attributable_regression_task_ids": ["wrong"],
            "unresolved_findings": ["Keep this model judgment."],
        }
    }
    reusable = {
        "preservation": {
            "preserved_task_ids": ["2", "1"],
            "attributable_regressions": [{"task_id": "3"}],
        }
    }
    adjustment = {
        "primary_problem": {
            "task_assessments": [
                {"task_id": "0", "status": "recovered", "relation": "not_attributed"},
                {"task_id": "4", "status": "recovered", "relation": "attributed"},
            ]
        }
    }

    canonicalize_final_decision(
        output,
        candidate_version="candidate-01",
        base_version="v0",
        reusable=reusable,
        adjustment=adjustment,
    )

    assert output["selected_version"] == "candidate-01"
    assert output["evidence"] == {
        "recovered_task_ids": ["4"],
        "uncertain_recovery_task_ids": [],
        "preserved_task_ids": ["1", "2"],
        "attributable_regression_task_ids": ["3"],
        "unresolved_findings": ["Keep this model judgment."],
    }


def test_review_can_return_a_bounded_revision_for_another_rollout():
    tested_candidate = {
        "source_candidate_ids": ["reusable-exchange"],
        "channel_diffs": [
            {"channel_id": "skills", "experience_ids": ["exp-a"]},
            {"channel_id": "instructions_rules", "experience_ids": ["exp-a"]},
        ],
        "manifest_delta": {
            "config_patch": {"tools.skill": True},
            "files": [
                {
                    "path": ".opencode/skills/exchange-flow/SKILL.md",
                    "content": (
                        "---\nname: exchange-flow\n"
                        "description: Use for delivered exchanges.\n---\n\n"
                        "Execute the confirmed exchange.\n"
                    ),
                }
            ],
            "instructions": ["Discuss logistics before executing."],
        },
    }
    output = {
        "decision": "revise_delta",
        "selected_version": "wrong",
        "rationale": "Remove the attributed ambient instruction and retain the SOP.",
        "evidence": {
            "recovered_task_ids": [],
            "preserved_task_ids": [],
            "attributable_regression_task_ids": ["0"],
            "unresolved_findings": ["The revised skill still requires rollout."],
        },
        "revision_candidate": {
            "id": "model-id",
            "objective": "Retain the exchange SOP without the regressing instruction.",
            "channel_plan": [
                {
                    "channel_id": "skills",
                    "operation": "retain bounded SOP",
                    "experience_ids": ["exp-a"],
                    "rationale": "The skill was not implicated in the regression.",
                }
            ],
            "manifest_delta": {
                "config_patch": {"tools.skill": True},
                "files": tested_candidate["manifest_delta"]["files"],
            },
            "validation": {
                "local_behavior_checks": ["Confirmed exchange is executed."],
            },
        },
        "budget_disposition": {
            "remaining_creations_after_this_decision": 29,
            "minimum_next_rollout_creations": 10,
            "rollout_possible": True,
            "minimum_next_iteration_creations": MIN_CANDIDATE_ITERATION_CREATIONS,
            "iteration_possible": True,
        },
    }
    reusable = {
        "preservation": {
            "preserved_task_ids": [],
            "attributable_regressions": [{"task_id": "0"}],
        }
    }
    adjustment = {
        "primary_problem": {
            "recommendation": "reject",
            "task_assessments": [{"task_id": "0", "status": "regressed"}],
        }
    }

    canonicalize_final_decision(
        output,
        candidate_version="candidate-01",
        base_version="v0",
        reusable=reusable,
        adjustment=adjustment,
        tested_candidate=tested_candidate,
    )
    validate_final_decision(
        output,
        candidate_version="candidate-01",
        base_version="v0",
        task_ids=("0",),
        reusable=reusable,
        adjustment=adjustment,
        budget_status={"remaining": 30},
        tested_candidate=tested_candidate,
    )

    assert output["selected_version"] == "v0"
    assert output["revision_candidate"]["id"] == "revision-candidate-01"
    assert output["revision_candidate"]["origin_candidate_id"] == (
        "reusable-exchange"
    )
    assert "instructions" not in output["revision_candidate"]["manifest_delta"]

    tested_candidate["portfolio_side"] = "revision"
    with pytest.raises(ValueError, match="tested revision must be accepted or rejected"):
        validate_final_decision(
            output,
            candidate_version="candidate-01",
            base_version="v0",
            task_ids=("0",),
            reusable=reusable,
            adjustment=adjustment,
            budget_status={"remaining": 30},
            tested_candidate=tested_candidate,
        )


def test_revision_instruction_file_paths_are_canonicalized_to_instruction_delta():
    tested_candidate = {
        "source_candidate_ids": ["adjustment-fee"],
        "channel_diffs": [
            {
                "channel_id": "instructions_rules",
                "experience_ids": ["retention-offer-type-vs-annual-fee-trigger"],
            }
        ],
        "manifest_delta": {
            "instructions": ["Original annual fee instruction."],
        },
    }
    output = {
        "decision": "revise_delta",
        "selected_version": "v0",
        "rationale": "Rewrite the bounded instruction.",
        "evidence": {
            "recovered_task_ids": [],
            "preserved_task_ids": [],
            "attributable_regression_task_ids": [],
            "unresolved_findings": ["Needs another rollout."],
        },
        "revision_candidate": {
            "id": "revision",
            "objective": "Narrow annual fee trigger.",
            "channel_plan": [
                {
                    "channel_id": "instructions_rules",
                    "operation": "rewrite",
                    "experience_ids": ["retention-offer-type-vs-annual-fee-trigger"],
                    "rationale": "Same evidence, narrower trigger.",
                }
            ],
            "manifest_delta": {
                "files": [
                    {
                        "path": "instructions/annual_fee_waiver.txt",
                        "content": "Only waive annual fees for an explicit primary complaint.",
                    }
                ]
            },
            "validation": {"local_behavior_checks": ["Annual fee guard holds."]},
        },
        "budget_disposition": {
            "remaining_creations_after_this_decision": 29,
            "minimum_next_rollout_creations": 10,
            "rollout_possible": True,
            "minimum_next_iteration_creations": MIN_CANDIDATE_ITERATION_CREATIONS,
            "iteration_possible": True,
        },
    }
    reusable = {
        "preservation": {
            "preserved_task_ids": [],
            "attributable_regressions": [],
        }
    }
    adjustment = {
        "primary_problem": {
            "recommendation": "refine",
            "task_assessments": [
                {
                    "task_id": "task_043",
                    "status": "mixed",
                    "relation": "attributed",
                }
            ],
        }
    }

    canonicalize_final_decision(
        output,
        candidate_version="candidate-08",
        base_version="v0",
        reusable=reusable,
        adjustment=adjustment,
        tested_candidate=tested_candidate,
    )
    validate_final_decision(
        output,
        candidate_version="candidate-08",
        base_version="v0",
        task_ids=("task_043",),
        reusable=reusable,
        adjustment=adjustment,
        budget_status={"remaining": 30},
        tested_candidate=tested_candidate,
    )

    revision = output["revision_candidate"]
    assert revision["channel_plan"][0]["experience_ids"] == [
        "retention-offer-type-vs-annual-fee-trigger"
    ]
    assert revision["manifest_delta"] == {
        "instructions": ["Only waive annual fees for an explicit primary complaint."]
    }


@pytest.mark.parametrize(
    "path",
    (
        "tool_desc_patches",
        "tool_desc_patches.json",
        "tool_description_patches.json",
        "manifest_delta/tool_desc_patches.json",
    ),
)
def test_revision_tool_description_file_is_canonicalized_and_validated(path):
    experience_id = "transfer-fallback-guidance"
    tested_candidate = {
        "source_candidate_ids": ["adjustment-transfer"],
        "channel_diffs": [
            {
                "channel_id": "tool_description",
                "experience_ids": [experience_id],
            }
        ],
        "manifest_delta": {
            "tool_desc_patches": {
                "transfer_to_human_agents": {"desc": "Original guidance."}
            }
        },
    }
    output = {
        "decision": "revise_delta",
        "selected_version": "v0",
        "rationale": "Refine the same tool-description channel.",
        "evidence": {
            "recovered_task_ids": [],
            "preserved_task_ids": [],
            "attributable_regression_task_ids": [],
            "unresolved_findings": ["Needs another rollout."],
        },
        "revision_candidate": {
            "id": "revision",
            "objective": "Clarify transfer fallback guidance.",
            "channel_plan": [
                {
                    "channel_id": "tool_description",
                    "operation": "revise",
                    "experience_ids": [experience_id],
                    "rationale": "The same evidence supports a narrower revision.",
                }
            ],
            "manifest_delta": {
                "files": [
                    {
                        "path": path,
                        "content": (
                            '{"transfer_to_human_agents":'
                            '{"desc":"Revised fallback guidance."}}'
                        ),
                    }
                ]
            },
            "validation": {"local_behavior_checks": ["Fallback is followed."]},
        },
        "budget_disposition": {
            "remaining_creations_after_this_decision": 29,
            "minimum_next_rollout_creations": MIN_ROLLOUT_CREATIONS,
            "rollout_possible": True,
            "minimum_next_iteration_creations": MIN_CANDIDATE_ITERATION_CREATIONS,
            "iteration_possible": True,
        },
    }
    reusable = {
        "preservation": {
            "preserved_task_ids": [],
            "attributable_regressions": [],
        }
    }
    adjustment = {
        "primary_problem": {
            "recommendation": "refine",
            "task_assessments": [
                {
                    "task_id": "task_014",
                    "status": "mixed",
                    "relation": "attributed",
                }
            ],
        }
    }

    canonicalize_final_decision(
        output,
        candidate_version="candidate-01",
        base_version="v0",
        reusable=reusable,
        adjustment=adjustment,
        tested_candidate=tested_candidate,
    )
    validate_final_decision(
        output,
        candidate_version="candidate-01",
        base_version="v0",
        task_ids=("task_014",),
        reusable=reusable,
        adjustment=adjustment,
        budget_status={"remaining": 30},
        tested_candidate=tested_candidate,
    )

    revision = output["revision_candidate"]
    assert revision["channel_plan"] == [
        {
            "channel_id": "tool_description",
            "operation": "revise",
            "experience_ids": [experience_id],
            "rationale": "The same evidence supports a narrower revision.",
        }
    ]
    assert revision["manifest_delta"] == {
        "tool_desc_patches": {
            "transfer_to_human_agents": {"desc": "Revised fallback guidance."}
        }
    }


def test_revision_tool_description_file_merges_existing_parameter_patches():
    output = {
        "decision": "revise_delta",
        "revision_candidate": {
            "manifest_delta": {
                "tool_desc_patches": {
                    "lookup": {"params": {"query": "Existing query guidance."}}
                },
                "files": [
                    {
                        "path": "tool_description_patches.json",
                        "content": {
                            "lookup": {
                                "desc": "Lookup records.",
                                "params": {"limit": "Maximum records."},
                            }
                        },
                    }
                ],
            }
        },
    }

    _canonicalize_revision_manifest_delta(output["revision_candidate"])

    assert output["revision_candidate"]["manifest_delta"] == {
        "tool_desc_patches": {
            "lookup": {
                "desc": "Lookup records.",
                "params": {
                    "query": "Existing query guidance.",
                    "limit": "Maximum records.",
                },
            }
        }
    }


def test_revision_can_add_a_discovered_channel_to_make_skill_reachable():
    skill_file = {
        "path": ".opencode/skills/exchange-flow/SKILL.md",
        "content": (
            "---\nname: exchange-flow\n"
            "description: Use for delivered exchanges.\n---\n\n"
            "Execute the confirmed exchange.\n"
        ),
    }
    tested_candidate = {
        "source_candidate_ids": ["reusable-exchange"],
        "channel_diffs": [
            {"channel_id": "skills", "experience_ids": ["exp-a"]},
        ],
        "manifest_delta": {
            "config_patch": {"tools.skill": True},
            "files": [skill_file],
        },
    }
    output = {
        "decision": "revise_delta",
        "selected_version": "v0",
        "rationale": "Add a bounded trigger because the skill was available but not invoked.",
        "evidence": {
            "recovered_task_ids": [],
            "preserved_task_ids": [],
            "attributable_regression_task_ids": ["0"],
            "unresolved_findings": ["Skill invocation requires another rollout."],
        },
        "revision_candidate": {
            "id": "revision",
            "objective": "Make the existing exchange skill reachable.",
            "channel_plan": [
                {
                    "channel_id": "skills",
                    "operation": "retain SOP",
                    "experience_ids": ["exp-a"],
                    "rationale": "Retain the same domain procedure.",
                },
                {
                    "channel_id": "instructions_rules",
                    "operation": "add bounded invocation trigger",
                    "experience_ids": ["exp-a"],
                    "rationale": "Channel usage showed the skill was not invoked.",
                },
            ],
            "manifest_delta": {
                "config_patch": {"tools.skill": True},
                "files": [skill_file],
                "instructions": [
                    "For a multi-order exchange, load the matching exchange skill before planning."
                ],
            },
            "validation": {
                "local_behavior_checks": ["The exchange skill is loaded and executed."],
            },
        },
        "budget_disposition": {
            "remaining_creations_after_this_decision": 29,
            "minimum_next_rollout_creations": 10,
            "rollout_possible": True,
            "minimum_next_iteration_creations": MIN_CANDIDATE_ITERATION_CREATIONS,
            "iteration_possible": True,
        },
    }
    reusable = {
        "preservation": {
            "preserved_task_ids": [],
            "attributable_regressions": [{"task_id": "0"}],
        }
    }
    adjustment = {
        "primary_problem": {
            "recommendation": "refine",
            "task_assessments": [{"task_id": "0", "status": "regressed"}],
        }
    }

    canonicalize_final_decision(
        output,
        candidate_version="candidate-01",
        base_version="v0",
        reusable=reusable,
        adjustment=adjustment,
        tested_candidate=tested_candidate,
    )
    validate_final_decision(
        output,
        candidate_version="candidate-01",
        base_version="v0",
        task_ids=("0",),
        reusable=reusable,
        adjustment=adjustment,
        budget_status={"remaining": 30},
        tested_candidate=tested_candidate,
        revision_channel_ids={"skills", "instructions_rules"},
    )

    assert {
        item["channel_id"] for item in output["revision_candidate"]["channel_plan"]
    } == {"skills", "instructions_rules"}


def test_revision_drops_unmaterialized_noop_channel_plans():
    skill_file = {
        "path": ".opencode/skills/exchange-flow/SKILL.md",
        "content": (
            "---\nname: exchange-flow\n"
            "description: Use for delivered exchanges.\n---\n\n"
            "Execute the confirmed exchange.\n"
        ),
    }
    tested_candidate = {
        "source_candidate_ids": ["reusable-exchange"],
        "channel_diffs": [
            {"channel_id": "skills", "experience_ids": ["exp-a"]},
            {"channel_id": "instructions_rules", "experience_ids": ["exp-a"]},
        ],
        "manifest_delta": {
            "config_patch": {"tools.skill": True},
            "files": [skill_file],
            "instructions": ["Remove me if I regressed."],
        },
    }
    output = {
        "decision": "revise_delta",
        "selected_version": "v0",
        "rationale": "Keep the skill and drop nonmaterialized tool changes.",
        "evidence": {
            "recovered_task_ids": [],
            "preserved_task_ids": [],
            "attributable_regression_task_ids": ["0"],
            "unresolved_findings": ["Revision still needs rollout."],
        },
        "revision_candidate": {
            "id": "revision",
            "objective": "Retain only materialized changes.",
            "channel_plan": [
                {
                    "channel_id": "skills",
                    "operation": "keep",
                    "experience_ids": ["exp-a"],
                    "rationale": "Keep the SOP.",
                },
                {
                    "channel_id": "instructions_rules",
                    "operation": "modify",
                    "experience_ids": ["exp-a"],
                    "rationale": "Keep bounded instruction.",
                },
                {
                    "channel_id": "tool_description",
                    "operation": "none",
                    "experience_ids": ["exp-a"],
                    "rationale": "No materialized tool patch remains.",
                },
            ],
            "manifest_delta": {
                "config_patch": {"tools.skill": True},
                "files": [skill_file],
                "instructions": ["Load the exchange skill for delivered exchanges."],
            },
            "validation": {"local_behavior_checks": ["Skill is reachable."]},
        },
        "budget_disposition": {
            "remaining_creations_after_this_decision": 29,
            "minimum_next_rollout_creations": 10,
            "rollout_possible": True,
            "minimum_next_iteration_creations": MIN_CANDIDATE_ITERATION_CREATIONS,
            "iteration_possible": True,
        },
    }
    reusable = {
        "preservation": {
            "preserved_task_ids": [],
            "attributable_regressions": [{"task_id": "0"}],
        }
    }
    adjustment = {
        "primary_problem": {
            "recommendation": "refine",
            "task_assessments": [{"task_id": "0", "status": "regressed"}],
        }
    }

    canonicalize_final_decision(
        output,
        candidate_version="candidate-01",
        base_version="v0",
        reusable=reusable,
        adjustment=adjustment,
        tested_candidate=tested_candidate,
    )
    validate_final_decision(
        output,
        candidate_version="candidate-01",
        base_version="v0",
        task_ids=("0",),
        reusable=reusable,
        adjustment=adjustment,
        budget_status={"remaining": 30},
        tested_candidate=tested_candidate,
        revision_channel_ids={"skills", "instructions_rules"},
    )

    assert {
        item["channel_id"] for item in output["revision_candidate"]["channel_plan"]
    } == {"skills", "instructions_rules"}


def test_revision_canonicalizes_legacy_instruction_file_shape():
    tested_candidate = {
        "source_candidate_ids": ["adjustment-instructions"],
        "channel_diffs": [
            {"channel_id": "instructions_rules", "experience_ids": ["exp-a"]},
            {"channel_id": "instructions_rules", "experience_ids": ["exp-b"]},
        ],
        "manifest_delta": {
            "instructions": ["Original instruction A.", "Original instruction B."],
        },
    }
    output = {
        "decision": "revise_delta",
        "selected_version": "v0",
        "rationale": "Narrow the instruction without changing channels.",
        "evidence": {
            "recovered_task_ids": [],
            "preserved_task_ids": [],
            "attributable_regression_task_ids": ["0"],
            "unresolved_findings": ["Revision still needs rollout."],
        },
        "revision_candidate": {
            "id": "revision",
            "objective": "Narrow instructions.",
            "channel_plan": [
                {
                    "channel_id": "instructions_rules",
                    "operation": "update",
                    "experience_ids": ["exp-a", "exp-b"],
                    "rationale": "Both instructions remain in scope.",
                }
            ],
            "manifest_delta": {
                "files": [
                    {
                        "path": "instructions_rules",
                        "content": ["Updated instruction A.", "Updated instruction B."],
                    }
                ],
            },
            "validation": {"local_behavior_checks": ["Instructions are applied."]},
        },
        "budget_disposition": {
            "remaining_creations_after_this_decision": 29,
            "minimum_next_rollout_creations": 10,
            "rollout_possible": True,
            "minimum_next_iteration_creations": MIN_CANDIDATE_ITERATION_CREATIONS,
            "iteration_possible": True,
        },
    }
    reusable = {
        "preservation": {
            "preserved_task_ids": [],
            "attributable_regressions": [{"task_id": "0"}],
        }
    }
    adjustment = {
        "primary_problem": {
            "recommendation": "refine",
            "task_assessments": [{"task_id": "0", "status": "regressed"}],
        }
    }

    canonicalize_final_decision(
        output,
        candidate_version="candidate-01",
        base_version="v0",
        reusable=reusable,
        adjustment=adjustment,
        tested_candidate=tested_candidate,
    )
    validate_final_decision(
        output,
        candidate_version="candidate-01",
        base_version="v0",
        task_ids=("0",),
        reusable=reusable,
        adjustment=adjustment,
        budget_status={"remaining": 30},
        tested_candidate=tested_candidate,
        revision_channel_ids={"instructions_rules"},
    )

    assert output["revision_candidate"]["manifest_delta"] == {
        "instructions": ["Updated instruction A.", "Updated instruction B."]
    }


def test_revision_canonicalizes_instruction_json_string_list():
    tested_candidate = {
        "source_candidate_ids": ["adjustment-instructions"],
        "channel_diffs": [{"channel_id": "instructions_rules", "experience_ids": ["exp-a"]}],
        "manifest_delta": {
            "instructions": ["Original instruction A.", "Original instruction B."],
        },
    }
    output = {
        "decision": "revise_delta",
        "selected_version": "v0",
        "rationale": "Narrow the instruction without changing channels.",
        "evidence": {
            "recovered_task_ids": [],
            "preserved_task_ids": [],
            "attributable_regression_task_ids": ["0"],
            "unresolved_findings": ["Revision still needs rollout."],
        },
        "revision_candidate": {
            "id": "revision",
            "objective": "Narrow instructions.",
            "channel_plan": [
                {
                    "channel_id": "instructions_rules",
                    "operation": "narrow",
                    "experience_ids": ["exp-a"],
                    "rationale": "Only one instruction remains in scope.",
                }
            ],
            "manifest_delta": {
                "instructions": ['["Updated instruction A.", "Updated instruction B."]'],
            },
            "validation": {"local_behavior_checks": ["Instructions are applied."]},
        },
        "budget_disposition": {
            "remaining_creations_after_this_decision": 29,
            "minimum_next_rollout_creations": 10,
            "rollout_possible": True,
            "minimum_next_iteration_creations": MIN_CANDIDATE_ITERATION_CREATIONS,
            "iteration_possible": True,
        },
    }
    reusable = {
        "preservation": {
            "preserved_task_ids": [],
            "attributable_regressions": [{"task_id": "0"}],
        }
    }
    adjustment = {
        "primary_problem": {
            "recommendation": "refine",
            "task_assessments": [{"task_id": "0", "status": "regressed"}],
        }
    }

    canonicalize_final_decision(
        output,
        candidate_version="candidate-01",
        base_version="v0",
        reusable=reusable,
        adjustment=adjustment,
        tested_candidate=tested_candidate,
    )
    validate_final_decision(
        output,
        candidate_version="candidate-01",
        base_version="v0",
        task_ids=("0",),
        reusable=reusable,
        adjustment=adjustment,
        budget_status={"remaining": 30},
        tested_candidate=tested_candidate,
        revision_channel_ids={"instructions_rules"},
    )

    assert output["revision_candidate"]["manifest_delta"] == {
        "instructions": ["Updated instruction A.", "Updated instruction B."]
    }


def test_iteration_cost_and_task_limit_cover_the_complete_module_chain():
    assert iteration_creation_cost(5) == 20
    assert iteration_creation_cost(6) == 22
    assert _max_rollout_tasks(17) == 0
    assert _max_rollout_tasks(20) == 0
    assert _max_rollout_tasks(22) == 0
    assert _max_rollout_tasks(23) == 5
    assert _max_rollout_tasks(24) == 5
    assert _max_rollout_tasks(25) == 6
    assert _rollout_task_limit(44, unattempted_batch_count=2) == 5
    assert _rollout_task_limit(44, unattempted_batch_count=1) == 12
    assert _rollout_task_limit(45, unattempted_batch_count=2) == 6
    assert _rollout_task_limit(37, unattempted_batch_count=2) == 6


def test_rollout_task_limit_keeps_exploration_batches_small():
    assert _rollout_task_limit(90, unattempted_batch_count=3) == 6
    assert _rollout_task_limit(90, unattempted_batch_count=1) > 6


def test_budget_plan_reserves_another_screen_while_exploring():
    plan = build_rollout_budget_plan(
        remaining_creations=120,
        parent_version="incumbent-00",
        unattempted_batch_count=3,
        max_task_count=30,
    )

    assert plan["mode"] == "standard"
    assert plan["promotion_eligible"] is True
    assert plan["rollout_task_limit"] == 6
    assert plan["reserved_future_iteration_creations"] == 33


def test_budget_plan_keeps_the_last_confirmable_iteration_compact():
    plan = build_rollout_budget_plan(
        remaining_creations=88,
        parent_version="incumbent-00",
        unattempted_batch_count=3,
        max_task_count=30,
    )

    assert plan["mode"] == "standard"
    assert plan["promotion_eligible"] is True
    assert plan["rollout_task_limit"] == 6
    assert plan["reserved_future_iteration_creations"] == 0


def test_budget_plan_uses_terminal_screen_without_weakening_promotion():
    plan = build_rollout_budget_plan(
        remaining_creations=56,
        parent_version="incumbent-00",
        unattempted_batch_count=2,
        max_task_count=30,
    )

    assert plan["mode"] == "terminal_screen"
    assert plan["promotion_eligible"] is True
    assert plan["minimum_rollout_tasks"] == 5
    assert plan["rollout_task_limit"] == 6


def test_budget_plan_does_not_reserve_confirmation_when_disabled(monkeypatch):
    monkeypatch.setenv("HAI_CONFIRMATION_MODE", "off")

    plan = build_rollout_budget_plan(
        remaining_creations=56,
        parent_version="incumbent-00",
        unattempted_batch_count=2,
        max_task_count=30,
    )

    assert plan["mode"] == "standard"
    assert plan["promotion_eligible"] is True
    assert plan["rollout_task_limit"] == 6
    assert plan["confirmation_creation_cost_at_limit"] is None


def test_budget_plan_stops_when_incumbent_cannot_fund_five_task_screen():
    stopped = build_rollout_budget_plan(
        remaining_creations=24,
        parent_version="incumbent-00",
        unattempted_batch_count=2,
        max_task_count=30,
    )

    assert stopped["mode"] == "stop"
    assert stopped["rollout_task_limit"] == 0


def test_budget_plan_uses_promotable_terminal_screen_against_cached_v0():
    terminal = build_rollout_budget_plan(
        remaining_creations=24,
        parent_version="v0",
        unattempted_batch_count=2,
        max_task_count=30,
    )

    assert terminal["mode"] == "terminal_screen"
    assert terminal["promotion_eligible"] is True
    assert terminal["minimum_rollout_tasks"] == 5
    assert terminal["rollout_task_limit"] == 5


def test_exploration_status_summarizes_open_problems_and_recent_results():
    status = build_exploration_status(
        available=[
            {
                "side": "reusable",
                "candidate": {
                    "id": "candidate-a",
                    "objective": "Fix one behavior.",
                    "channel_plan": [
                        {"channel_id": "instructions_rules"},
                        {"channel_id": "tool_description"},
                    ],
                    "manifest_delta": {
                        "files": [{"path": "a", "content": "x"}],
                    },
                },
                "experience_ids": ["exp1", "exp2", "exp3", "exp4", "exp5"],
                "evidence_refs": ["ev0", "ev1"],
            }
        ],
        iteration_history=[
            {
                "candidate_id": "candidate-old",
                "candidate_version": "candidate-01",
                "review_decision": "reject_delta",
                "selected_version": "v0",
                "rollout_task_ids": ["0", "1", "2", "3", "4"],
                "rollout_metrics": {"pass_at_2": 0.2},
                "review_evidence": {
                    "recovered_task_ids": ["0"],
                    "attributable_regression_task_ids": ["1", "2"],
                },
            }
        ],
        evidence_to_task={"ev0": "0", "ev1": "1"},
        evidence_outcomes={"ev0": "fail", "ev1": "pass"},
        budget_status={"remaining": 90},
        rollout_task_limit=6,
    )

    assert status["budget"]["rollout_task_limit"] == 6
    assert status["budget"]["affordable_minimum_iterations"] == 4
    assert status["budget"]["affordable_compact_iterations"] == 3
    assert status["policy"]["prefer_small_batches"] is False
    assert status["candidate_type_counts"]["reusable"] == 1
    assert status["open_problems"][0]["problem_id"] == "candidate-a"
    assert status["open_problems"][0]["evidence_task_ids"] == ["0", "1"]
    assert status["open_problems"][0]["baseline_evidence"] == {
        "trial_count": 2,
        "failed_trial_count": 1,
        "passed_trial_count": 1,
        "task_count": 2,
        "support_tier": "mixed_evidence",
        "failure_density": 0.5,
        "all_failed_task_ids": ["0"],
        "mixed_task_ids": [],
        "all_passed_task_ids": ["1"],
    }
    assert "multi_channel_delta" in status["open_problems"][0]["risk_flags"]
    assert "large_experience_batch" in status["open_problems"][0]["risk_flags"]
    assert status["recent_results"][0]["attributable_regression_count"] == 2


def test_reusable_problem_without_passing_evidence_is_marked_high_risk():
    status = build_exploration_status(
        available=[
            {
                "side": "reusable",
                "candidate": {"id": "procedure-a", "channel_plan": []},
                "experience_ids": ["exp-a"],
                "evidence_refs": ["ev0", "ev1"],
            }
        ],
        iteration_history=[],
        evidence_to_task={"ev0": "0", "ev1": "0"},
        evidence_outcomes={"ev0": "fail", "ev1": "fail"},
        budget_status={"remaining": 40},
        rollout_task_limit=6,
    )

    assert "reusable_without_passing_evidence_support" in (
        status["open_problems"][0]["risk_flags"]
    )
    assert "failure_only_without_passing_contrast" in (
        status["open_problems"][0]["risk_flags"]
    )


def test_evidence_profile_distinguishes_within_task_contrast_from_failure_only():
    contrast = _baseline_evidence_summary(
        evidence_refs=["pass", "fail"],
        evidence_to_task={"pass": "task-a", "fail": "task-a"},
        evidence_outcomes={"pass": "pass", "fail": "fail"},
    )
    failure_only = _baseline_evidence_summary(
        evidence_refs=["fail-a", "fail-b"],
        evidence_to_task={"fail-a": "task-a", "fail-b": "task-b"},
        evidence_outcomes={"fail-a": "fail", "fail-b": "fail"},
    )

    assert contrast["support_tier"] == "within_task_contrast"
    assert contrast["mixed_task_ids"] == ["task-a"]
    assert failure_only["support_tier"] == "failure_only"
    assert failure_only["all_failed_task_ids"] == ["task-a", "task-b"]


def test_reusable_problem_with_only_passing_evidence_has_no_observed_headroom():
    status = build_exploration_status(
        available=[
            {
                "side": "reusable",
                "candidate": {"id": "procedure-a", "channel_plan": []},
                "experience_ids": ["exp-a"],
                "evidence_refs": ["ev0", "ev1"],
            }
        ],
        iteration_history=[],
        evidence_to_task={"ev0": "0", "ev1": "0"},
        evidence_outcomes={"ev0": "pass", "ev1": "pass"},
        budget_status={"remaining": 40},
        rollout_task_limit=6,
    )

    assert "no_observed_failure_headroom" in status["open_problems"][0]["risk_flags"]


def test_exploration_status_marks_revision_without_positive_parent_evidence():
    status = build_exploration_status(
        available=[
            {
                "side": "revision",
                "candidate": {"id": "revision-a", "channel_plan": []},
                "experience_ids": ["exp-a"],
                "evidence_refs": ["ev0"],
            }
        ],
        iteration_history=[
            {
                "candidate_id": "candidate-a",
                "revision_candidate_id": "revision-a",
                "review_decision": "revise_delta",
                "review_evidence": {
                    "recovered_task_ids": [],
                    "attributable_regression_task_ids": ["0"],
                },
            }
        ],
        evidence_to_task={"ev0": "0"},
        evidence_outcomes={"ev0": "fail"},
        budget_status={"remaining": 40},
        rollout_task_limit=6,
    )

    problem = status["open_problems"][0]
    assert problem["revision_parent_signal"] == {
        "attributed_recovery_count": 0,
        "attributable_regression_count": 1,
    }
    assert "revision_without_positive_parent_evidence" in problem["risk_flags"]


def test_exploration_status_exposes_revision_plateau_by_origin():
    status = build_exploration_status(
        available=[
            {
                "side": "adjustment",
                "candidate": {"id": "another-problem", "channel_plan": []},
                "experience_ids": [],
                "evidence_refs": [],
            }
        ],
        iteration_history=[
            {
                "candidate_id": "candidate-a",
                "origin_candidate_id": "candidate-a",
                "review_decision": "revise_delta",
                "review_evidence": {
                    "recovered_task_ids": ["0"],
                    "attributable_regression_task_ids": ["1"],
                },
            },
            {
                "candidate_id": "revision-a",
                "origin_candidate_id": "candidate-a",
                "review_decision": "revise_delta",
                "review_evidence": {
                    "recovered_task_ids": ["0"],
                    "attributable_regression_task_ids": ["1", "2"],
                },
            },
        ],
        evidence_to_task={},
        evidence_outcomes={},
        budget_status={"remaining": 40},
        rollout_task_limit=6,
    )

    assert status["hypothesis_history"][0] == {
        "origin_candidate_id": "candidate-a",
        "attempt_count": 2,
        "latest_recovered_count": 1,
        "latest_attributable_regression_count": 2,
        "latest_revision_improved": False,
    }


def test_main_decision_allows_more_tasks_when_the_full_iteration_is_funded():
    output = _decision()
    output["rollout_request"]["task_ids"] = ["0", "1", "2", "3", "4", "5"]
    validate_main_decision(
        output,
        train_task_ids=("0", "1", "2", "3", "4", "5"),
        candidate_evidence={"a": ("ev0",)},
        candidate_sides={"a": "adjustment"},
        evidence_to_task={"ev0": "0"},
        max_rollout_tasks=6,
    )
    actual_editor_channel_diffs,
