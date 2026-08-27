from types import SimpleNamespace

import pytest

from harnesslens.evolution.discovery import (
    DiscoveryModule,
    _conservative_harness_query_output,
    _enforce_native_channel_admission,
    _harness_query_model_payload,
    _mcp_editable_points,
    _query_evidence_with_input_paths,
    _surface_hypothesis,
    canonicalize_harness_query_output,
    validate_harness_query,
    validate_task_explorer,
)
from harnesslens.harnesses.runner import IntelligentRunResult


@pytest.mark.parametrize(
    ("harness", "expected", "isolated"),
    [
        (
            "opencode",
            {"instructions_rules", "skills", "system_prompt", "agent_definitions"},
            {"agent_definitions"},
        ),
        (
            "pi",
            {"project_instructions", "skills", "system_prompt", "compaction_config"},
            set(),
        ),
        (
            "codex",
            {
                "developer_instructions",
                "project_instructions",
                "skills",
                "hooks",
                "compaction",
            },
            {"hooks"},
        ),
    ],
)
def test_controller_admits_verified_harness_optimization_lanes(
    harness, expected, isolated
):
    hypotheses = {
        channel_id: {
            "id": channel_id,
            "evidence_refs": [f"runtime:channel:{channel_id}"],
            "verification": {
                "runtime_observed": channel_id
                not in {"compaction", "compaction_config"}
            },
        }
        for channel_id in expected
    }
    output = {"modifiable_modules": [], "unavailable_modules": []}

    _enforce_native_channel_admission(
        output,
        hypotheses=hypotheses,
        harness=harness,
        controller_mcp_ids=set(),
    )

    modules = {item["id"]: item for item in output["modifiable_modules"]}
    assert set(modules) == expected
    assert {
        channel_id
        for channel_id, module in modules.items()
        if module["execution_lane"] == "isolated_declarative"
    } == isolated


def test_surface_hypothesis_preserves_policy_and_config_selector():
    unsupported = _surface_hypothesis(
        {"id": "personality", "status": "exists_but_unsupported"}
    )
    configurable = _surface_hypothesis(
        {
            "id": "developer_instructions",
            "status": "verified",
            "operation": {
                "kind": "harness_config_patch",
                "key": "developer_instructions",
            },
        }
    )
    workspace_config = _surface_hypothesis(
        {
            "id": "instructions_rules",
            "status": "verified",
            "operation": {
                "kind": "workspace_config",
                "scope": "project",
                "path": "opencode.json",
                "mechanism": "config",
                "key": "instructions",
            },
        }
    )

    assert unsupported["policy_status"] == "unsupported"
    assert configurable["edit_selector"] == {"key": "developer_instructions"}
    assert workspace_config["trusted_edit_contract"] == {
        "scope": "project",
        "path": "opencode.json",
        "mechanism": "config",
        "key": "instructions",
    }


def test_conservative_query_repairs_contract_from_runtime_materializer():
    module = _documented_module("instructions_rules")
    module["edit_contract"] = {
        "scope": "project",
        "path": "AGENTS.md",
        "mechanism": "file",
    }
    module["evidence_level"] = "materialized"
    module["evidence_refs"] = ["runtime:channel:instructions_rules"]
    payload = {
        "architecture_probe": {
            "harness_id": "opencode",
            "harness_version": "1",
        },
        "surface_hypotheses": [
            {
                "id": "instructions_rules",
                "trusted_edit_contract": {
                    "scope": "project",
                    "path": "opencode.json",
                    "mechanism": "config",
                    "key": "instructions",
                },
                "edit_selector": {"key": "instructions"},
                "evidence_refs": ["runtime:channel:instructions_rules"],
            }
        ],
        "evidence_catalog": [
            {
                "id": "runtime:channel:instructions_rules",
                "kind": "materializer_contract",
                "value": {
                    "kind": "workspace_config",
                    "scope": "project",
                    "path": "opencode.json",
                    "mechanism": "config",
                    "key": "instructions",
                },
            }
        ],
        "candidate_workspace_contract": {"runtime_loading": {}},
        "predefined_mcp_editable_points": [],
    }
    output = {
        "modifiable_modules": [module],
        "unavailable_modules": [],
        "mcp_editable_points": [],
        "base_profile": {},
    }

    normalized = _conservative_harness_query_output(output, payload)

    assert normalized["modifiable_modules"][0]["edit_contract"] == {
        "scope": "project",
        "path": "opencode.json",
        "mechanism": "config",
        "key": "instructions",
    }


def test_conservative_query_attaches_trusted_config_selector():
    module = _documented_module("developer_instructions")
    module["edit_contract"] = {
        "scope": "project",
        "path": ".codex/config.toml",
        "mechanism": "config",
    }
    payload = {
        "architecture_probe": {"harness_id": "codex", "harness_version": "1"},
        "surface_hypotheses": [
            {
                "id": "developer_instructions",
                "edit_selector": {"key": "developer_instructions"},
                "evidence_refs": ["docs:skills"],
            }
        ],
        "evidence_catalog": [{"id": "docs:skills", "kind": "documentation"}],
        "candidate_workspace_contract": {"runtime_loading": {}},
        "predefined_mcp_editable_points": [],
    }
    output = {
        "modifiable_modules": [module],
        "unavailable_modules": [],
        "mcp_editable_points": [],
        "base_profile": {},
    }

    normalized = _conservative_harness_query_output(output, payload)

    assert normalized["modifiable_modules"][0]["edit_contract"]["key"] == (
        "developer_instructions"
    )


class _Budget:
    def next_attempt_id(self, base_id):
        return base_id

    def reserve(self, *args, **kwargs):
        return None


def test_parallel_discovery_side_persists_before_its_sibling_finishes(
    tmp_path, monkeypatch
):
    output = {
        "categories": [
            {"id": "all", "name": "All", "purpose": "All tasks", "task_ids": ["0"]}
        ]
    }
    monkeypatch.setattr(
        "harnesslens.evolution.discovery.IntelligentHarnessRunner.run_json",
        lambda self, **kwargs: SimpleNamespace(output=output),
    )
    module = DiscoveryModule(
        repo_root=tmp_path,
        run_root=tmp_path / "run",
        budget=_Budget(),
    )

    result = module._run_task_explorer({}, ("0",))

    assert result.output == output
    assert (tmp_path / "run" / "discovery" / "task_explorer.json").is_file()


def test_harness_query_retry_receives_exact_validation_error(tmp_path, monkeypatch):
    calls = []
    previous = tmp_path / "previous.stdout"
    previous.write_text('{"modifiable_modules":[]}', encoding="utf-8")

    def run_json(_self, **kwargs):
        calls.append(kwargs["input_payload"])
        if len(calls) == 1:
            return IntelligentRunResult(
                job_id="query-1",
                harness="opencode",
                outcome="invalid",
                output=None,
                validation_error="edit_contract scope must be home or project",
                stdout_path=str(previous),
                stderr_path="",
            )
        return IntelligentRunResult(
            job_id="query-2",
            harness="opencode",
            outcome="completed",
            output={"harness": "opencode"},
            validation_error="",
            stdout_path="",
            stderr_path="",
        )

    monkeypatch.setattr(
        "harnesslens.evolution.discovery.IntelligentHarnessRunner.run_json", run_json
    )
    module = DiscoveryModule(
        repo_root=tmp_path,
        run_root=tmp_path / "run",
        budget=_Budget(),
    )

    result = module._run_harness_query(
        {"predefined_mcp_editable_points": [{"id": "mcp_tool_description"}]}
    )

    assert result.output["harness"] == "opencode"
    assert result.output["mcp_editable_points"] == [{"id": "mcp_tool_description"}]
    assert "predefined_mcp_editable_points" not in calls[0]
    assert calls[0]["controller_managed_mcp"]["connected"] is True
    assert "edit_contract scope must be home or project" in calls[1]["retry_context"]
    assert calls[1]["previous_output"] == {"modifiable_modules": []}


def test_task_explorer_validation_preserves_exact_primary_coverage():
    task_ids = tuple(str(index) for index in range(30))
    output = {
        "categories": [
            {
                "id": "a",
                "name": "A",
                "purpose": "First purpose",
                "task_ids": list(task_ids[:15]),
            },
            {
                "id": "b",
                "name": "B",
                "purpose": "Second purpose",
                "task_ids": list(task_ids[15:]),
            },
        ]
    }

    validate_task_explorer(output, task_ids)

    output["categories"][1]["task_ids"].append(task_ids[0])
    with pytest.raises(ValueError, match="exactly one"):
        validate_task_explorer(output, task_ids)


def _documented_module(channel_id="skills"):
    return {
        "id": channel_id,
        "status": "modifiable",
        "evidence_level": "documented",
        "visibility": "startup metadata and on-demand body",
        "use": "Store a task-triggered procedure.",
        "edit_contract": {
            "scope": "project",
            "path": ".agents/skills/<slug>/SKILL.md",
            "mechanism": "file",
        },
        "runtime_constraints": ["Skill discovery must be enabled."],
        "risks": "A broad trigger can affect unrelated tasks.",
        "evidence_refs": ["docs:skills"],
    }


def _unavailable_module(channel_id="compaction"):
    return {
        "id": channel_id,
        "status": "conditional",
        "reason": "The current runner disables this surface.",
        "runtime_constraints": ["Automatic compaction is disabled."],
        "evidence_refs": ["probe:runtime"],
    }


def test_harness_query_accepts_model_discovered_channel_when_grounded():
    output = {
        "modifiable_modules": [_documented_module("discovered_rules")],
        "unavailable_modules": [],
        "mcp_editable_points": [],
        "base_profile": {},
    }

    validate_harness_query(
        output,
        set(),
        evidence_ids={"docs:skills"},
    )


def test_harness_query_requires_every_surface_hint_to_be_accounted_for():
    output = {
        "modifiable_modules": [_documented_module("skills")],
        "unavailable_modules": [],
        "mcp_editable_points": [],
        "base_profile": {},
    }

    with pytest.raises(ValueError, match="missing"):
        validate_harness_query(output, {"skills", "instructions_rules"})


def test_harness_query_requires_a_workspace_edit_contract_for_modifiable_surface():
    output = {
        "modifiable_modules": [_documented_module()],
        "unavailable_modules": [],
        "mcp_editable_points": [],
        "base_profile": {},
    }
    output["modifiable_modules"][0].pop("edit_contract")

    with pytest.raises(ValueError, match="edit_contract"):
        validate_harness_query(
            output,
            {"skills"},
            evidence_ids={"docs:skills"},
        )


@pytest.mark.parametrize(
    "path", ["project/AGENTS.md", "AGENTS.md|rules.md", "AGENTS.md (preferred)"]
)
def test_harness_query_rejects_ambiguous_edit_paths(path):
    module = _documented_module()
    module["edit_contract"]["path"] = path
    output = {
        "modifiable_modules": [module],
        "unavailable_modules": [],
        "mcp_editable_points": [],
        "base_profile": {},
    }

    with pytest.raises(ValueError, match="workspace scope"):
        validate_harness_query(
            output,
            {"skills"},
            evidence_ids={"docs:skills"},
        )


def test_conditional_surface_may_preserve_a_potential_edit_path():
    conditional = _unavailable_module("compaction")
    conditional.pop("runtime_constraints")
    conditional["edit_contract"] = {
        "scope": "home",
        "path": "settings.json",
        "mechanism": "config",
    }
    output = {
        "modifiable_modules": [],
        "unavailable_modules": [conditional],
        "mcp_editable_points": [],
        "base_profile": {},
    }

    validate_harness_query(
        output,
        {"compaction"},
        evidence_ids={"probe:runtime"},
    )


def test_harness_query_runtime_observed_requires_channel_specific_trace():
    output = {
        "modifiable_modules": [_documented_module()],
        "unavailable_modules": [],
        "mcp_editable_points": [],
        "base_profile": {},
    }
    output["modifiable_modules"][0]["evidence_level"] = "runtime_observed"
    output["modifiable_modules"][0]["evidence_refs"] = ["runtime:probe:start"]
    evidence = [
        {
            "id": "runtime:probe:start",
            "kind": "request_sentinel_probe",
            "value": {"project_instructions": ["observed"]},
        }
    ]

    with pytest.raises(ValueError, match="runtime-observed evidence"):
        validate_harness_query(
            output,
            {"skills"},
            evidence_ids={"runtime:probe:start"},
            evidence_catalog=evidence,
        )


def test_harness_query_reports_all_surface_validation_errors():
    first = _documented_module("skills")
    second = _documented_module("project_instructions")
    first.pop("edit_contract")
    second.pop("edit_contract")
    output = {
        "modifiable_modules": [first, second],
        "unavailable_modules": [],
        "mcp_editable_points": [],
        "base_profile": {},
    }

    with pytest.raises(ValueError) as error:
        validate_harness_query(
            output,
            {"skills", "project_instructions"},
            evidence_ids={"docs:skills"},
        )

    assert "skills:" in str(error.value)
    assert "project_instructions:" in str(error.value)


def test_mcp_transport_adds_exact_tool_and_parameter_editable_points():
    points = _mcp_editable_points(
        {
            "environment": {
                "tool_transport": {"kind": "mcp", "server_id": "tau2"},
                "tools": [
                    {
                        "name": "modify_order",
                        "parameters": {"properties": {"order_id": {}, "item_ids": {}}},
                    }
                ],
            }
        }
    )

    assert points[0]["targets"] == ["modify_order"]
    assert points[1]["targets"] == [
        {"tool": "modify_order", "parameters": ["item_ids", "order_id"]}
    ]


def test_mcp_transport_accepts_compact_public_parameter_catalog():
    points = _mcp_editable_points(
        {
            "environment": {
                "tool_transport": {"kind": "mcp", "server_id": "bird"},
                "tools": [
                    {
                        "name": "execute_sql",
                        "parameters": {"sql": "string"},
                    }
                ],
            }
        },
        harness_id="pi",
    )

    assert points[1]["targets"] == [{"tool": "execute_sql", "parameters": ["sql"]}]
    assert points[1]["operation"]["harness"] == "pi"


def test_mcp_transport_keeps_parameterless_tools_explicit():
    points = _mcp_editable_points(
        {
            "environment": {
                "tool_transport": {"kind": "mcp", "server_id": "example"},
                "tools": [{"name": "ping", "parameters": {}}],
            }
        }
    )

    assert points[1]["targets"] == [{"tool": "ping", "parameters": []}]


def test_query_accepts_grounded_input_field_paths_as_evidence_refs():
    evidence = _query_evidence_with_input_paths(
        {
            "candidate_workspace_contract": {
                "runtime_loading": {
                    "conditional_behavior": ["Skills require read access."]
                }
            },
            "predefined_mcp_editable_points": [
                {"id": "mcp_tool_description", "targets": ["lookup"]}
            ],
        }
    )
    evidence_ids = {str(item["id"]) for item in evidence}

    assert (
        "candidate_workspace_contract:runtime_loading:conditional_behavior"
        in evidence_ids
    )
    assert "predefined_mcp_editable_points:mcp_tool_description" in evidence_ids


def test_harness_query_requires_predefined_mcp_targets_verbatim():
    required = [
        {
            "id": "mcp_tool_description",
            "base_channel_id": "tool_description",
            "server_id": "tau2",
            "targets": ["modify_order"],
        }
    ]
    output = {
        "modifiable_modules": [_documented_module("skills")],
        "unavailable_modules": [],
        "mcp_editable_points": [
            {
                "id": "mcp_tool_description",
                "base_channel_id": "tool_description",
                "server_id": "tau2",
                "targets": ["wrong_tool"],
                "status": "conditional",
                "reason": "No workspace edit path is available.",
                "runtime_constraints": ["The server binding is runner-owned."],
                "evidence_refs": ["runtime:mcp"],
            }
        ],
        "base_profile": {},
    }

    with pytest.raises(ValueError, match="targets mismatch"):
        validate_harness_query(output, {"skills"}, required)


def _grounded_query_output():
    return {
        "harness": "opencode",
        "harness_version": "1.2.3",
        "summary": "One verified startup instruction channel.",
        "modifiable_modules": [
            {
                "id": "instructions_rules",
                "status": "modifiable",
                "evidence_level": "materialized",
                "visibility": "startup",
                "use": "Concise cross-task behavior.",
                "edit_contract": {
                    "scope": "project",
                    "path": "AGENTS.md",
                    "mechanism": "file",
                },
                "runtime_constraints": ["Project instructions must be loaded."],
                "risks": "Broad wording can affect unrelated tasks.",
                "evidence_refs": ["runtime:channel:instructions_rules"],
            }
        ],
        "unavailable_modules": [],
        "mcp_editable_points": [],
        "base_profile": {"shared_rules": [], "role_isolation": "fresh workspace"},
    }


def _grounded_query_validation_kwargs():
    return {
        "expected_harness": "opencode",
        "expected_harness_version": "1.2.3",
        "evidence_ids": {"runtime:channel:instructions_rules"},
        "evidence_catalog": [
            {
                "id": "runtime:channel:instructions_rules",
                "kind": "materializer_contract",
                "value": {"kind": "project_file", "path_pattern": "AGENTS.md"},
            }
        ],
        "channel_contracts": {
            "instructions_rules": {
                "id": "instructions_rules",
            }
        },
    }


def test_harness_query_requires_matching_harness_identity_and_version():
    output = _grounded_query_output()
    output["harness"] = "codex"

    with pytest.raises(ValueError, match="harness identity"):
        validate_harness_query(
            output,
            {"instructions_rules"},
            **_grounded_query_validation_kwargs(),
        )


def test_harness_query_requires_grounded_evidence():
    output = _grounded_query_output()
    module = output["modifiable_modules"][0]
    module["evidence_refs"] = ["invented:evidence"]

    with pytest.raises(ValueError, match="evidence"):
        validate_harness_query(
            output,
            {"instructions_rules"},
            **_grounded_query_validation_kwargs(),
        )


def test_harness_query_accepts_grounded_model_assessment():
    validate_harness_query(
        _grounded_query_output(),
        {"instructions_rules"},
        **_grounded_query_validation_kwargs(),
    )


def test_legacy_materializer_does_not_prove_a_workspace_config_path():
    module = _documented_module("tool_description")
    module["evidence_level"] = "materialized"
    module["edit_contract"] = {
        "scope": "project",
        "path": "opencode.json",
        "mechanism": "config",
    }
    module["evidence_refs"] = [
        "runtime:channel:tool_description",
        "docs:tool-description",
    ]
    output = {
        "modifiable_modules": [module],
        "unavailable_modules": [],
        "mcp_editable_points": [],
        "base_profile": {},
    }
    evidence = [
        {
            "id": "runtime:channel:tool_description",
            "kind": "materializer_contract",
            "value": {
                "kind": "tool_schema_patch",
                "manifest_field": "tool_desc_patches",
            },
        },
        {
            "id": "docs:tool-description",
            "kind": "local_documentation",
            "value": "Tool descriptions are configurable in some runtimes.",
        },
    ]

    with pytest.raises(ValueError, match="not supported"):
        validate_harness_query(
            output,
            {"tool_description"},
            evidence_ids={
                "runtime:channel:tool_description",
                "docs:tool-description",
            },
            evidence_catalog=evidence,
            workspace_contract={
                "runtime_loading": {"config_paths": ["project/opencode.json"]}
            },
        )


def test_documented_evidence_does_not_bypass_workspace_grounding():
    module = _documented_module("instructions_rules")
    module["edit_contract"] = {
        "scope": "project",
        "path": "AGENTS.md",
        "mechanism": "file",
    }
    module["evidence_refs"] = ["runtime:channel:instructions_rules"]
    output = {
        "modifiable_modules": [module],
        "unavailable_modules": [],
        "mcp_editable_points": [],
        "base_profile": {},
    }

    with pytest.raises(ValueError, match="not supported"):
        validate_harness_query(
            output,
            {"instructions_rules"},
            evidence_ids={"runtime:channel:instructions_rules"},
            evidence_catalog=[
                {
                    "id": "runtime:channel:instructions_rules",
                    "kind": "materializer_contract",
                    "value": {"kind": "prompt_content"},
                }
            ],
            workspace_contract={
                "runtime_loading": {"config_paths": ["project/opencode.json"]}
            },
        )


def test_runtime_observed_workspace_contract_matches_workspace_materializer():
    module = _documented_module("instructions_rules")
    module["evidence_level"] = "runtime_observed"
    module["edit_contract"] = {
        "scope": "project",
        "path": "opencode.json",
        "mechanism": "config",
        "key": "instructions",
    }
    module["evidence_refs"] = [
        "runtime:channel:instructions_rules",
        "runtime:probe:startup",
    ]
    output = {
        "modifiable_modules": [module],
        "unavailable_modules": [],
        "mcp_editable_points": [],
        "base_profile": {},
    }
    evidence = [
        {
            "id": "runtime:channel:instructions_rules",
            "kind": "materializer_contract",
            "value": {
                "kind": "workspace_config",
                "scope": "project",
                "path": "opencode.json",
                "mechanism": "config",
                "key": "instructions",
            },
        },
        {
            "id": "runtime:probe:startup",
            "kind": "request_sentinel_probe",
            "value": {
                "observed_channels": ["instructions_rules"],
                "workspace_edit_contracts": {
                    "instructions_rules": {
                        "scope": "project",
                        "path": "opencode.json",
                        "mechanism": "config",
                        "key": "instructions",
                    }
                },
            },
        },
    ]

    validate_harness_query(
        output,
        {"instructions_rules"},
        evidence_ids={item["id"] for item in evidence},
        evidence_catalog=evidence,
    )

    module["edit_contract"]["key"] = "unobserved"
    with pytest.raises(ValueError, match="not supported"):
        validate_harness_query(
            output,
            {"instructions_rules"},
            evidence_ids={item["id"] for item in evidence},
            evidence_catalog=evidence,
        )


def test_project_file_materializer_requires_file_mechanism():
    module = _documented_module("reference_sources")
    module["evidence_level"] = "materialized"
    module["edit_contract"] = {
        "scope": "project",
        "path": "opencode.json",
        "mechanism": "config",
    }
    module["evidence_refs"] = ["runtime:channel:reference_sources"]
    output = {
        "modifiable_modules": [module],
        "unavailable_modules": [],
        "mcp_editable_points": [],
        "base_profile": {},
    }

    with pytest.raises(ValueError, match="not supported"):
        validate_harness_query(
            output,
            {"reference_sources"},
            evidence_ids={"runtime:channel:reference_sources"},
            evidence_catalog=[
                {
                    "id": "runtime:channel:reference_sources",
                    "kind": "materializer_contract",
                    "value": {
                        "kind": "project_file",
                        "path_pattern": "<relative-path>",
                    },
                }
            ],
        )


def test_harness_query_controller_owns_admitted_channel_assessment():
    output = _grounded_query_output()
    module = output["modifiable_modules"][0]
    module["evidence_level"] = "documented"
    module["evidence_refs"] = ["docs:instructions"]
    payload = {
        "architecture_probe": {
            "harness_id": "opencode",
            "harness_version": "1.2.3",
        },
        "surface_hypotheses": [{"id": "instructions_rules"}],
        "predefined_mcp_editable_points": [],
    }

    normalized = canonicalize_harness_query_output(output, payload)

    assert normalized["modifiable_modules"][0]["evidence_level"] == "materialized"
    assert normalized["modifiable_modules"][0]["evidence_refs"] == ["docs:instructions"]
    assert "operation" not in normalized["modifiable_modules"][0]


def test_harness_query_controller_attaches_config_key_to_valid_model_output():
    output = _grounded_query_output()
    output["modifiable_modules"][0]["id"] = "developer_instructions"
    output["modifiable_modules"][0]["edit_contract"] = {
        "scope": "project",
        "path": ".codex/config.toml",
        "mechanism": "config",
    }
    payload = {
        "architecture_probe": {"harness_id": "codex", "harness_version": "1"},
        "surface_hypotheses": [
            {
                "id": "developer_instructions",
                "edit_selector": {"key": "developer_instructions"},
            }
        ],
        "predefined_mcp_editable_points": [],
    }

    normalized = canonicalize_harness_query_output(output, payload)

    assert normalized["modifiable_modules"][0]["edit_contract"]["key"] == (
        "developer_instructions"
    )


def test_harness_query_controller_ignores_model_mcp_assessment():
    points = _mcp_editable_points(
        {
            "environment": {
                "tool_transport": {"kind": "mcp", "server_id": "task"},
                "tools": [{"name": "lookup", "parameters": {"properties": {}}}],
            }
        }
    )
    output = {
        "summary": "Model incorrectly assessed MCP.",
        "modifiable_modules": [{"id": "mcp_tool_description", "status": "modifiable"}],
        "unavailable_modules": [
            {"id": "mcp_tool_parameter_description", "status": "unsupported"}
        ],
        "mcp_editable_points": [
            {
                "id": "invented_mcp_point",
                "status": "unsupported",
            }
        ],
    }
    payload = {
        "architecture_probe": {"harness_id": "opencode", "harness_version": "1"},
        "predefined_mcp_editable_points": points,
    }

    normalized = canonicalize_harness_query_output(output, payload)

    assert normalized["mcp_editable_points"] == points
    assert normalized["modifiable_modules"] == []
    assert normalized["unavailable_modules"] == []
    assert normalized["native_summary"] == "Model incorrectly assessed MCP."
    assert "controller attached 2 MCP editable points" in normalized["summary"]


def test_harness_query_model_payload_hides_controller_managed_mcp_points():
    payload = {
        "predefined_mcp_editable_points": [
            {
                "id": "mcp_tool_description",
                "base_channel_id": "tool_description",
            }
        ],
        "surface_hypotheses": [
            {"id": "skills"},
            {"id": "tool_description"},
        ],
        "evidence_catalog": [
            {"id": "probe:version"},
            {"id": "runtime:mcp-workspace-bridge"},
        ],
        "public_runtime": {
            "benchmark_kind": "bird",
            "task_tool_names": ["execute_sql"],
            "tool_transport": {"kind": "mcp", "server_id": "bird"},
        },
    }

    model_payload = _harness_query_model_payload(payload)

    assert "predefined_mcp_editable_points" not in model_payload
    assert model_payload["controller_managed_mcp"]["connected"] is True
    assert model_payload["surface_hypotheses"] == [{"id": "skills"}]
    assert model_payload["evidence_catalog"] == [{"id": "probe:version"}]
    assert model_payload["public_runtime"] == {"benchmark_kind": "bird"}


def test_harness_query_fallback_only_demotes_unproven_surfaces():
    payload = {
        "architecture_probe": {
            "harness_id": "opencode",
            "harness_version": "1",
        },
        "surface_hypotheses": [
            {
                "id": "instructions_rules",
                "evidence_refs": ["runtime:channel:instructions_rules"],
            },
            {
                "id": "permissions",
                "policy_status": "forbidden",
                "evidence_refs": ["runtime:channel:permissions"],
            },
            {
                "id": "compaction_config",
                "policy_status": "conditional",
                "evidence_refs": ["runtime:channel:compaction_config"],
            },
        ],
        "evidence_catalog": [
            {
                "id": "runtime:channel:instructions_rules",
                "kind": "materializer_contract",
                "value": {"kind": "prompt_content"},
            },
            {
                "id": "runtime:channel:permissions",
                "kind": "materializer_contract",
                "value": {"kind": "harness_config_patch"},
            },
            {
                "id": "runtime:channel:compaction_config",
                "kind": "materializer_contract",
                "value": {"kind": "harness_config_patch"},
            },
        ],
        "candidate_workspace_contract": {
            "runtime_loading": {"config_paths": ["project/opencode.json"]}
        },
        "predefined_mcp_editable_points": [],
    }
    output = {
        "summary": "Unproven claims.",
        "modifiable_modules": [
            {
                "id": channel_id,
                "status": "modifiable",
                "evidence_level": "materialized",
                "visibility": "startup",
                "use": "claim",
                "edit_contract": {
                    "scope": "project",
                    "path": "opencode.json",
                    "mechanism": "config",
                },
                "runtime_constraints": ["claim"],
                "risks": "claim",
                "evidence_refs": [f"runtime:channel:{channel_id}"],
            }
            for channel_id in (
                "instructions_rules",
                "permissions",
                "compaction_config",
            )
        ],
        "unavailable_modules": [],
        "base_profile": {"shared_rules": [], "role_isolation": "fixed"},
    }

    normalized = _conservative_harness_query_output(output, payload)

    assert [item["id"] for item in normalized["modifiable_modules"]] == [
        "instructions_rules"
    ]
    unavailable = {item["id"]: item for item in normalized["unavailable_modules"]}
    assert unavailable["permissions"]["status"] == "forbidden"
    assert unavailable["compaction_config"]["status"] == "conditional"
