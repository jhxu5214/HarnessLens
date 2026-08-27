import json

import pytest

from harnesslens.harnesses.channel_preflight import (
    ChannelPreflightError,
    build_runtime_load_report,
    validate_channel_preflight,
)
from harnesslens.benchmarks.pi_tau2 import _PiRpcSession


def _decision():
    return {
        "harness_version": "candidate-01",
        "candidate": {
            "id": "candidate-a",
            "channel_diffs": [
                {"channel_id": "project_instructions"},
                {"channel_id": "skills"},
                {"channel_id": "mcp_tool_parameter_description"},
            ],
            "workspace_delta": {
                "files": [
                    {
                        "scope": "project",
                        "path": "AGENTS.md",
                        "change": "modified",
                        "content": "Always inspect account state first.\n",
                    },
                    {
                        "scope": "project",
                        "path": ".pi/skills/account-check/SKILL.md",
                        "change": "added",
                        "content": "---\nname: account-check\n---\nCheck the account.\n",
                    },
                ]
            },
            "manifest_delta": {
                "tool_desc_patches": {
                    "lookup": {
                        "params": {"account_id": "The exact account identifier."}
                    }
                }
            },
        },
    }


def _rollout(tmp_path, context):
    trajectory = tmp_path / "trial.jsonl"
    trajectory.write_text(
        json.dumps({"task_id": "1", "model_context": context}) + "\n",
        encoding="utf-8",
    )
    return {
        "per_task": {"1": {"trajectory_paths": [str(trajectory)]}},
        "summary_json": str(tmp_path / "summary.json"),
    }


def _rollout_with_request(tmp_path, context, request):
    trajectory = tmp_path / "trial.jsonl"
    sidecar = tmp_path / "trial.api_calls.jsonl"
    sidecar.write_text(
        json.dumps({"role": "agent", "request": request}) + "\n",
        encoding="utf-8",
    )
    trajectory.write_text(
        json.dumps(
            {
                "task_id": "1",
                "model_context": context,
                "api_calls_jsonl": sidecar.name,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return {"per_task": {"1": {"trajectory_paths": [str(trajectory)]}}}


def test_runtime_load_report_captures_effective_candidate_surfaces(tmp_path):
    project = tmp_path / "project"
    skill = project / ".pi" / "skills" / "account-check" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: account-check\n---\nCheck it.\n", encoding="utf-8")
    (project / "AGENTS.md").write_text(
        "Base.\n\nAlways inspect account state first.\n", encoding="utf-8"
    )

    report = build_runtime_load_report(
        harness="pi",
        project_root=project,
        home_root=tmp_path / "home",
        manifest={
            "tool_desc_patches": {
                "lookup": {"params": {"account_id": "The exact account identifier."}}
            }
        },
        tool_definitions=[
            {
                "name": "lookup",
                "description": "Lookup.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"account_id": {"type": "string"}},
                },
            }
        ],
    )

    assert report["skills_available"][0]["name"] == "account-check"
    assert "Always inspect" in report["project_instruction_files"][0]["content"]
    assert (
        report["tool_definitions"][0]["inputSchema"]["properties"]["account_id"][
            "description"
        ]
        == "The exact account identifier."
    )


def test_runtime_load_report_patches_tau2_openai_parameter_schema(tmp_path):
    project = tmp_path / "project"
    project.mkdir()

    report = build_runtime_load_report(
        harness="pi",
        project_root=project,
        home_root=tmp_path / "home",
        manifest={
            "tool_desc_patches": {
                "lookup": {"params": {"account_id": "The exact account identifier."}}
            }
        },
        tool_definitions=[
            {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"account_id": {"type": "string"}},
                },
            }
        ],
    )

    assert (
        report["tool_definitions"][0]["parameters"]["properties"]["account_id"][
            "description"
        ]
        == "The exact account identifier."
    )


def test_runtime_load_report_captures_effective_config_agents_and_hook_events(tmp_path):
    project = tmp_path / "project"
    home = tmp_path / "home"
    agent = project / ".codex" / "agents" / "reviewer.toml"
    agent.parent.mkdir(parents=True)
    agent.write_text(
        'name = "reviewer"\n'
        'description = "Inspect tool evidence before answering."\n'
        'developer_instructions = "Cross-check identifiers."\n',
        encoding="utf-8",
    )
    observation = project / ".codex" / "harness-hook-observation.json"
    observation.write_text(
        json.dumps({"hook_event_name": "SessionStart"}),
        encoding="utf-8",
    )
    home.mkdir()
    (home / "config.toml").write_text(
        'compact_prompt = "Keep unresolved identifiers."\n',
        encoding="utf-8",
    )

    report = build_runtime_load_report(
        harness="codex",
        project_root=project,
        home_root=home,
        manifest={
            "_workspace": {
                "files": [
                    {
                        "scope": "project",
                        "path": ".codex/agents/reviewer.toml",
                        "content": agent.read_text(encoding="utf-8"),
                    }
                ]
            }
        },
        tool_definitions=[],
    )

    assert report["effective_config"]["compact_prompt"] == (
        "Keep unresolved identifiers."
    )
    assert report["agent_definitions"] == [
        {
            "name": "reviewer",
            "path": ".codex/agents/reviewer.toml",
            "description": "Inspect tool evidence before answering.",
            "instructions": "Cross-check identifiers.",
        }
    ]
    assert report["runtime_events"][0]["hook_event_name"] == "SessionStart"


def test_pi_append_system_prompt_requires_final_request_visibility(tmp_path):
    sentinel = "Always reconcile the returned account identifier."
    decision = {
        "candidate": {
            "id": "pi-system",
            "channel_diffs": [{"channel_id": "system_prompt"}],
            "workspace_delta": {
                "files": [
                    {
                        "scope": "project",
                        "path": ".pi/APPEND_SYSTEM.md",
                        "change": "added",
                        "content": sentinel,
                    }
                ]
            },
            "manifest_delta": {},
        }
    }
    context = {
        "schema": "harnesslens.channel-load-report.v1",
        "harness": "pi",
        "candidate_project_files": [
            {"path": ".pi/APPEND_SYSTEM.md", "content": sentinel}
        ],
        "project_instruction_files": [],
        "skills_available": [],
        "tool_definitions": [],
    }

    report = validate_channel_preflight(
        decision=decision,
        rollout=_rollout_with_request(
            tmp_path,
            context,
            {"messages": [{"role": "system", "content": sentinel}]},
        ),
        output_path=tmp_path / "preflight.json",
    )

    assert report["passed"] is True
    assert report["request_evidence_available"] is True


def test_multiline_project_instructions_are_matched_in_native_request(tmp_path):
    instructions = (
        "# Project Instructions\n\n"
        "Return one deterministic result for singular extremum questions.\n"
    )
    decision = {
        "candidate": {
            "id": "pi-project",
            "channel_diffs": [{"channel_id": "project_instructions"}],
            "workspace_delta": {
                "files": [
                    {
                        "scope": "project",
                        "path": "AGENTS.md",
                        "change": "added",
                        "content": instructions,
                    }
                ]
            },
            "manifest_delta": {},
        }
    }
    context = {
        "schema": "harnesslens.channel-load-report.v1",
        "harness": "pi-agent",
        "project_instruction_files": [{"path": "AGENTS.md", "content": instructions}],
        "skills_available": [],
        "tool_definitions": [],
    }

    report = validate_channel_preflight(
        decision=decision,
        rollout=_rollout_with_request(
            tmp_path,
            context,
            {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "<project_instructions>\n"
                            f"{instructions}"
                            "</project_instructions>"
                        ),
                    }
                ]
            },
        ),
        output_path=tmp_path / "preflight.json",
    )

    assert report["passed"] is True
    assert report["request_evidence_available"] is True


def test_agent_definition_requires_active_request_catalog_visibility(tmp_path):
    description = "Delegate exact identifier verification to this reviewer."
    content = (
        'name = "reviewer"\n'
        f'description = "{description}"\n'
        'developer_instructions = "Verify returned identifiers."\n'
    )
    decision = {
        "candidate": {
            "id": "codex-agent",
            "channel_diffs": [{"channel_id": "agent_definitions"}],
            "workspace_delta": {
                "files": [
                    {
                        "scope": "project",
                        "path": ".codex/agents/reviewer.toml",
                        "change": "added",
                        "content": content,
                    }
                ]
            },
            "manifest_delta": {},
        }
    }
    context = {
        "schema": "harnesslens.channel-load-report.v1",
        "harness": "codex",
        "candidate_project_files": [
            {"path": ".codex/agents/reviewer.toml", "content": content}
        ],
        "agent_definitions": [
            {
                "name": "reviewer",
                "path": ".codex/agents/reviewer.toml",
                "description": description,
                "instructions": "Verify returned identifiers.",
            }
        ],
        "project_instruction_files": [],
        "skills_available": [],
        "tool_definitions": [],
    }

    report = validate_channel_preflight(
        decision=decision,
        rollout=_rollout_with_request(
            tmp_path,
            context,
            {"tools": [{"name": "spawn_agent", "description": description}]},
        ),
        output_path=tmp_path / "preflight.json",
    )

    assert report["passed"] is True


def test_codex_hook_requires_triggered_context_in_final_request(tmp_path):
    sentinel = "After resume, re-check the latest tool evidence."
    decision = {
        "candidate": {
            "id": "codex-hook",
            "channel_diffs": [{"channel_id": "hooks"}],
            "workspace_delta": {
                "files": [
                    {
                        "scope": "project",
                        "path": ".codex/harness-hook-context.md",
                        "change": "added",
                        "content": sentinel,
                    }
                ]
            },
            "manifest_delta": {},
        }
    }
    context = {
        "schema": "harnesslens.channel-load-report.v1",
        "harness": "codex",
        "candidate_project_files": [
            {"path": ".codex/harness-hook-context.md", "content": sentinel}
        ],
        "runtime_events": [{"hook_event_name": "SessionStart"}],
        "project_instruction_files": [],
        "skills_available": [],
        "tool_definitions": [],
    }

    report = validate_channel_preflight(
        decision=decision,
        rollout=_rollout_with_request(
            tmp_path,
            context,
            {"input": [{"role": "developer", "content": sentinel}]},
        ),
        output_path=tmp_path / "preflight.json",
    )

    assert report["passed"] is True
    hook_check = next(
        item for item in report["checks"] if item["channel_id"] == "hooks"
    )
    assert hook_check["event_observed"] is True


@pytest.mark.parametrize(
    ("channel_id", "harness", "path", "content", "effective"),
    [
        (
            "compaction_config",
            "pi",
            ".pi/settings.json",
            '{"compaction":{"enabled":false,"keepRecentTokens":12000}}',
            {"compaction": {"enabled": False, "keepRecentTokens": 12000}},
        ),
        (
            "compaction",
            "codex",
            ".codex/config.toml",
            'compact_prompt = "Keep unresolved evidence."\n',
            {"compact_prompt": "Keep unresolved evidence."},
        ),
    ],
)
def test_compaction_channels_require_effective_runtime_config(
    tmp_path, channel_id, harness, path, content, effective
):
    decision = {
        "candidate": {
            "id": "compaction",
            "channel_diffs": [{"channel_id": channel_id}],
            "workspace_delta": {
                "files": [
                    {
                        "scope": "project",
                        "path": path,
                        "change": "added",
                        "content": content,
                    }
                ]
            },
            "manifest_delta": {},
        }
    }
    context = {
        "schema": "harnesslens.channel-load-report.v1",
        "harness": harness,
        "effective_config": effective,
        "project_instruction_files": [],
        "candidate_project_files": [{"path": path, "content": content}],
        "skills_available": [],
        "tool_definitions": [],
    }

    report = validate_channel_preflight(
        decision=decision,
        rollout=_rollout(tmp_path, context),
        output_path=tmp_path / "preflight.json",
    )

    assert report["passed"] is True


def test_channel_preflight_accepts_visible_runtime_surfaces(tmp_path):
    context = {
        "schema": "harnesslens.channel-load-report.v1",
        "project_instruction_files": [
            {"path": "AGENTS.md", "content": "Always inspect account state first.\n"}
        ],
        "skills_available": [{"name": "account-check", "n_calls": 0}],
        "tool_definitions": [
            {
                "name": "lookup",
                "inputSchema": {
                    "properties": {
                        "account_id": {"description": "The exact account identifier."}
                    }
                },
            }
        ],
    }

    report = validate_channel_preflight(
        decision=_decision(),
        rollout=_rollout(tmp_path, context),
        output_path=tmp_path / "preflight.json",
    )

    assert report["passed"] is True


def test_opencode_instruction_rule_accepts_runtime_relocated_path_in_model_request(
    tmp_path,
):
    instruction = "Return exactly one complete SQL statement."
    instruction_path = ".opencode/instructions/autoiter-test.md"
    runtime_project = tmp_path / "runtime-project"
    config = json.dumps({"instructions": [instruction_path]}, indent=2) + "\n"
    decision = {
        "candidate": {
            "id": "sql-final-guard",
            "channel_diffs": [{"channel_id": "instructions_rules"}],
            "workspace_delta": {
                "files": [
                    {
                        "scope": "project",
                        "path": "opencode.json",
                        "change": "modified",
                        "content": config,
                    },
                    {
                        "scope": "project",
                        "path": instruction_path,
                        "change": "added",
                        "content": instruction + "\n",
                    },
                ]
            },
            "manifest_delta": {},
        }
    }
    context = {
        "schema": "harnesslens.channel-load-report.v1",
        "harness": "opencode",
        "project_root": str(runtime_project),
        "effective_config": {
            "instructions": [str((runtime_project / instruction_path).resolve())]
        },
        "candidate_project_files": [
            {
                "path": "opencode.json",
                "content": json.dumps(
                    {
                        "$schema": "https://opencode.ai/config.json",
                        "instructions": [instruction_path],
                    }
                ),
            },
            {"path": instruction_path, "content": instruction + "\n"},
        ],
        "project_instruction_files": [],
        "skills_available": [],
        "tool_definitions": [],
        "instructions": [],
        "prompt_appends": [],
    }

    report = validate_channel_preflight(
        decision=decision,
        rollout=_rollout_with_request(
            tmp_path,
            context,
            {"messages": [{"role": "system", "content": instruction}]},
        ),
        output_path=tmp_path / "preflight-instruction.json",
    )

    assert report["passed"] is True
    assert all(item["passed"] for item in report["checks"])


def test_channel_preflight_fails_closed_when_skill_telemetry_is_missing(tmp_path):
    context = {
        "schema": "harnesslens.channel-load-report.v1",
        "project_instruction_files": [
            {"path": "AGENTS.md", "content": "Always inspect account state first.\n"}
        ],
        "skills_available": [],
        "tool_definitions": [],
    }

    with pytest.raises(ChannelPreflightError, match="not available") as failure:
        validate_channel_preflight(
            decision=_decision(),
            rollout=_rollout(tmp_path, context),
            output_path=tmp_path / "preflight.json",
        )

    assert failure.value.report["passed"] is False
    assert (tmp_path / "preflight.json").is_file()


def test_instruction_channel_requires_the_runtime_materialized_file(tmp_path):
    decision = {
        "harness_version": "candidate-01",
        "candidate": {
            "id": "candidate-a",
            "channel_diffs": [{"channel_id": "developer_instructions"}],
            "workspace_delta": {
                "files": [
                    {
                        "scope": "project",
                        "path": ".codex/config.toml",
                        "change": "added",
                        "content": 'developer_instructions = "Select only requested columns."\n',
                    }
                ]
            },
            "manifest_delta": {},
        },
    }
    context = {
        "schema": "harnesslens.channel-load-report.v1",
        "project_instruction_files": [],
        "candidate_project_files": [
            {
                "path": ".codex/config.toml",
                "content": 'developer_instructions = "Select only requested columns."\n',
            }
        ],
        "skills_available": [],
        "tool_definitions": [],
    }

    report = validate_channel_preflight(
        decision=decision,
        rollout=_rollout(tmp_path, context),
        output_path=tmp_path / "preflight.json",
    )

    assert report["passed"] is True


def test_preflight_rejects_changed_channel_without_checker(tmp_path):
    decision = _decision()
    decision["candidate"]["channel_diffs"].append({"channel_id": "unknown_channel"})
    context = {
        "schema": "harnesslens.channel-load-report.v1",
        "project_instruction_files": [
            {"path": "AGENTS.md", "content": "Always inspect account state first.\n"}
        ],
        "skills_available": [{"name": "account-check", "n_calls": 0}],
        "tool_definitions": [
            {
                "name": "lookup",
                "inputSchema": {
                    "properties": {
                        "account_id": {"description": "The exact account identifier."}
                    }
                },
            }
        ],
    }

    with pytest.raises(ChannelPreflightError, match="no preflight checker") as failure:
        validate_channel_preflight(
            decision=decision,
            rollout=_rollout(tmp_path, context),
            output_path=tmp_path / "preflight.json",
        )

    assert failure.value.report["unchecked_channels"] == ["unknown_channel"]
    assert "unknown_channel" not in failure.value.report["checked_channels"]


def test_instruction_preflight_uses_retained_model_request(tmp_path):
    decision = {
        "harness_version": "candidate-01",
        "candidate": {
            "id": "candidate-a",
            "channel_diffs": [{"channel_id": "developer_instructions"}],
            "workspace_delta": {
                "files": [
                    {
                        "scope": "project",
                        "path": ".codex/config.toml",
                        "change": "added",
                        "content": 'developer_instructions = "Select only requested columns."\n',
                    }
                ]
            },
            "manifest_delta": {},
        },
    }
    context = {
        "schema": "harnesslens.channel-load-report.v1",
        "project_instruction_files": [],
        "candidate_project_files": [
            {
                "path": ".codex/config.toml",
                "content": 'developer_instructions = "Select only requested columns."\n',
            }
        ],
        "skills_available": [],
        "tool_definitions": [],
    }
    trajectory = tmp_path / "trial.jsonl"
    sidecar = tmp_path / "trial.api_calls.jsonl"
    sidecar.write_text(
        json.dumps(
            {
                "role": "agent",
                "request": {
                    "messages": [
                        {
                            "role": "developer",
                            "content": "Select only requested columns.",
                        }
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    trajectory.write_text(
        json.dumps(
            {
                "task_id": "1",
                "model_context": context,
                "api_calls_jsonl": sidecar.name,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rollout = {"per_task": {"1": {"trajectory_paths": [str(trajectory)]}}}

    report = validate_channel_preflight(
        decision=decision,
        rollout=rollout,
        output_path=tmp_path / "preflight.json",
    )

    assert report["passed"] is True
    assert report["request_evidence_available"] is True


def test_codex_instruction_preflight_parses_escaped_toml_string(tmp_path):
    instruction = (
        'When a query says "oldest and lowest salary", '
        "enumerate the interpretations first."
    )
    config_content = (
        'developer_instructions = "When a query says \\"oldest and lowest '
        'salary\\", enumerate the interpretations first."\n'
    )
    decision = {
        "harness_version": "candidate-01",
        "candidate": {
            "id": "candidate-a",
            "channel_diffs": [{"channel_id": "developer_instructions"}],
            "workspace_delta": {
                "files": [
                    {
                        "scope": "project",
                        "path": ".codex/config.toml",
                        "change": "added",
                        "content": config_content,
                    }
                ]
            },
            "manifest_delta": {},
        },
    }
    context = {
        "schema": "harnesslens.channel-load-report.v1",
        "harness": "codex",
        "project_instruction_files": [],
        "candidate_project_files": [
            {"path": ".codex/config.toml", "content": config_content}
        ],
        "skills_available": [],
        "tool_definitions": [],
    }

    report = validate_channel_preflight(
        decision=decision,
        rollout=_rollout_with_request(
            tmp_path,
            context,
            {
                "input": [
                    {"role": "developer", "content": instruction},
                ]
            },
        ),
        output_path=tmp_path / "preflight.json",
    )

    assert report["passed"] is True
    assert report["request_evidence_available"] is True


def test_codex_instruction_preflight_fails_without_request_sidecar(tmp_path):
    decision = {
        "harness_version": "candidate-01",
        "candidate": {
            "id": "candidate-a",
            "channel_diffs": [{"channel_id": "developer_instructions"}],
            "workspace_delta": {
                "files": [
                    {
                        "scope": "project",
                        "path": ".codex/config.toml",
                        "change": "added",
                        "content": 'developer_instructions = "Select only requested columns."\n',
                    }
                ]
            },
            "manifest_delta": {},
        },
    }
    context = {
        "schema": "harnesslens.channel-load-report.v1",
        "harness": "codex",
        "project_instruction_files": [],
        "candidate_project_files": [
            {
                "path": ".codex/config.toml",
                "content": 'developer_instructions = "Select only requested columns."\n',
            }
        ],
        "skills_available": [],
        "tool_definitions": [],
    }

    with pytest.raises(ChannelPreflightError, match="no model request evidence"):
        validate_channel_preflight(
            decision=decision,
            rollout=_rollout(tmp_path, context),
            output_path=tmp_path / "preflight.json",
        )


def test_pi_rpc_skill_inventory_uses_runtime_get_commands(monkeypatch):
    session = object.__new__(_PiRpcSession)
    monkeypatch.setattr(
        session,
        "_send_and_wait",
        lambda payload, timeout_s: {
            "success": True,
            "data": {
                "commands": [
                    {"name": "skill:account-check", "source": "skill"},
                    {"name": "help", "source": "extension"},
                ]
            },
        },
    )

    assert session.skills_available(timeout_s=10) == [
        {"name": "account-check", "n_calls": 0, "source": "pi_rpc"}
    ]
