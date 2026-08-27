from pathlib import Path

import pytest

from harnesslens.evolution.analyzer import _validate_candidate
from harnesslens.benchmarks.cell_config import benchmark_config
from harnesslens.evolution.discovery import (
    DiscoveryModule,
    _mcp_editable_points,
    canonicalize_harness_query_output,
)
from harnesslens.harnesses.harness_query_adapters import harness_query_adapter
from harnesslens.benchmarks.task_data import BaselineDataset, benchmark_task_explorer_input


REPO_ROOT = Path(__file__).resolve().parents[1]


def _concrete_path(pattern: str) -> str:
    return (
        pattern.replace("<slug>", "query-probe")
        .replace("<name>", "query-probe")
        .replace("<relative-path>", "query-probe.md")
    )


def _minimal_delta(harness: str, channel_id: str, operation):
    kind = operation["kind"]
    field = operation.get("manifest_field")
    if kind == "prompt_content":
        return {field: ["Use the evidence-bounded query probe."]}
    if kind == "project_file":
        path = _concrete_path(operation["path_pattern"])
        content = "Use the evidence-bounded query probe."
        if path.endswith("/SKILL.md"):
            content = (
                "---\n"
                "name: query-probe\n"
                "description: Exercise one query-discovered skill channel.\n"
                "---\n\n"
                "Use the evidence-bounded query probe.\n"
            )
        elif path.endswith(".ts"):
            content = "export const queryProbe = true;\n"
        delta = {"files": [{"path": path, "content": content}]}
        if harness == "opencode" and channel_id == "skills":
            delta["config_patch"] = {"tools.skill": True}
        return delta
    if kind == "tool_schema_patch":
        patch = {"desc": "Use for one evidence-bounded query probe."}
        if operation.get("target") == "parameter_description":
            patch = {"params": {"item_id": "The exact item identifier."}}
        return {field: {"query_tool": patch}}
    if kind == "harness_config_patch":
        key = operation.get("key")
        if not key:
            suffixes = {
                ("opencode", "agent_tool_config"): "query-probe.tools",
                ("opencode", "builtin_tools"): "read",
                ("opencode", "permissions"): "read",
                ("opencode", "compaction_config"): "auto",
                ("opencode", "mcp_servers"): "query-probe",
                ("opencode", "experimental_policies"): "query-probe",
                ("pi", "compaction_config"): "enabled",
                ("codex", "mcp_servers"): "query_probe.command",
                ("codex", "tool_enablement"): "default_tools_enabled",
                ("codex", "feature_flags"): "shell_tool",
            }
            key = operation["key_prefix"] + suffixes[(harness, channel_id)]
        value = "Use the evidence-bounded query probe."
        if channel_id not in {"system_prompt", "developer_instructions"}:
            value = True
        return {field: {key: value}}
    if kind == "workspace_config":
        return {}
    raise AssertionError(f"unsupported query operation: {operation}")


def _assert_candidate_closes_contract(harness: str, contract):
    channel_id = contract["id"]
    candidate = {
        "id": f"{harness}-{channel_id}",
        "objective": f"Exercise the query-discovered {channel_id} channel.",
        "channel_plan": [
            {
                "channel_id": channel_id,
                "operation": "materialize the exact Harness Query operation",
                "experience_ids": ["exp-query-probe"],
                "rationale": "The candidate directly exercises this discovered contract.",
            }
        ],
        "manifest_delta": _minimal_delta(harness, channel_id, contract["operation"]),
        "validation": {"local_behavior_checks": ["The channel is materialized."]},
    }
    _validate_candidate(
        candidate,
        experience_ids={"exp-query-probe"},
        channel_ids={channel_id},
        channel_contracts={channel_id: contract},
        harness=harness,
    )


def _assert_grounded_inventory(adapter):
    probe = adapter.architecture_probe()
    evidence = adapter.query_evidence_catalog(probe)
    evidence_ids = {item["id"] for item in evidence}
    inventory = adapter.query_channel_inventory(probe)

    assert probe["harness_id"]
    assert probe["harness_version"]
    assert inventory
    for channel in inventory:
        assert channel["evidence_refs"]
        assert set(channel["evidence_refs"]).issubset(evidence_ids)
        if channel["status"] == "verified":
            assert channel["operation"]["kind"]
        else:
            assert "operation" not in channel
    return probe, {item["id"]: item for item in inventory}


def test_pi_query_uses_pi_native_surfaces_and_excludes_native_mcp():
    adapter = harness_query_adapter("pi", repo_root=REPO_ROOT)

    probe, channels = _assert_grounded_inventory(adapter)

    assert probe["harness_id"] == "pi"
    assert channels["project_instructions"]["operation"]["path_pattern"] == "AGENTS.md"
    assert channels["skills"]["operation"]["path_pattern"].startswith(".pi/skills/")
    assert channels["tool_enablement"]["status"] == "proposal_only"
    assert channels["mcp_servers"]["status"] == "proposal_only"


def test_codex_query_uses_codex_native_instruction_and_mcp_surfaces():
    adapter = harness_query_adapter("codex", repo_root=REPO_ROOT)

    probe, channels = _assert_grounded_inventory(adapter)

    assert probe["harness_id"] == "codex"
    assert (
        channels["developer_instructions"]["operation"]["key"]
        == "developer_instructions"
    )
    assert channels["project_instructions"]["operation"]["path_pattern"] == "AGENTS.md"
    assert (
        channels["hooks"]["operation"]["path_pattern"]
        == ".codex/harness-hook-context.md"
    )
    assert channels["user_instructions"]["status"] == "exists_but_unsupported"
    assert channels["skill_configuration"]["status"] == "forbidden"
    assert channels["mcp_servers"]["status"] == "verified"
    assert channels["mcp_servers"]["operation"]["key_prefix"] == "mcp_servers."
    assert channels["hook_event_handlers"]["status"] == "verified"
    assert channels["hook_event_handlers"]["operation"]["path_pattern"] == (
        ".codex/hooks.json"
    )
    assert channels["tool_enablement"]["status"] == "forbidden"
    assert channels["feature_flags"]["status"] == "proposal_only"
    assert channels["agent_definitions"]["status"] == "exists_but_unsupported"
    assert channels["agent_definitions"].get("operation") is None
    assert channels["plugins"]["status"] == "forbidden"
    assert channels["personality"]["status"] == "exists_but_unsupported"
    assert channels["commands"]["status"] == "exists_but_unsupported"
    assert channels["base_instructions"]["status"] == "exists_but_unsupported"
    assert channels["collaboration_mode"]["status"] == "exists_but_unsupported"
    assert channels["memory"]["status"] == "forbidden"
    assert channels["compaction"]["status"] == "verified"
    assert channels["compaction"]["operation"] == {
        "kind": "workspace_config",
        "scope": "project",
        "path": ".codex/config.toml",
        "mechanism": "config",
        "key": "compact_prompt",
    }
    assert channels["model_runtime"]["status"] == "forbidden"
    assert channels["exec_policy"]["status"] == "forbidden"
    assert channels["environment_authority"]["status"] == "forbidden"
    assert channels["session_history"]["status"] == "forbidden"
    assert channels["profiles"]["status"] == "proposal_only"

    schema = probe["app_server_schema"]
    assert "commands" in schema["external_agent_import_fields"]
    assert "subagents" in schema["external_agent_import_fields"]
    assert "baseInstructions" in schema["thread_start_fields"]
    assert "collaborationMode" in schema["turn_start_fields"]
    assert "sessionStart" in schema["hook_events"]
    assert set(schema["hook_handler_types"]) == {"agent", "command", "prompt"}


def test_codex_probe_gates_optional_native_surfaces():
    adapter = harness_query_adapter("codex", repo_root=REPO_ROOT)
    probe = adapter.architecture_probe()
    probe["feature_lines"] = [
        line
        for line in probe["feature_lines"]
        if not str(line).startswith(
            ("hooks ", "multi_agent ", "plugins ", "personality ")
        )
    ]

    channels = {item["id"] for item in adapter.query_channel_inventory(probe)}

    assert "hooks" not in channels
    assert "agent_definitions" in channels
    assert "plugins" not in channels
    assert "personality" not in channels


def test_query_adapter_accepts_pi_agent_alias():
    adapter = harness_query_adapter("pi-agent", repo_root=REPO_ROOT)

    assert adapter.architecture_probe()["harness_id"] == "pi"


def test_opencode_query_exposes_observed_project_instruction_contract():
    adapter = harness_query_adapter("opencode", repo_root=REPO_ROOT)
    inventory = {item["id"]: item for item in adapter.query_channel_inventory()}
    evidence = {
        item["id"]: item
        for item in adapter.query_evidence_catalog(adapter.architecture_probe())
    }

    contracts = evidence["runtime:probe:opencode-startup"]["value"][
        "workspace_edit_contracts"
    ]
    assert contracts["instructions_rules"] == {
        "scope": "project",
        "path": "opencode.json",
        "mechanism": "config",
        "key": "instructions",
    }
    assert inventory["instructions_rules"]["operation"] == {
        "kind": "workspace_config",
        **contracts["instructions_rules"],
    }


@pytest.mark.parametrize(
    ("harness", "evidence_id", "needle"),
    [
        ("opencode", "docs:opencode:skills", "skill"),
        ("pi", "docs:pi-skills", "skill"),
        ("codex", "docs:codex-config-reference", "config"),
    ],
)
def test_query_receives_bounded_documentation_content(harness, evidence_id, needle):
    adapter = harness_query_adapter(harness, repo_root=REPO_ROOT)
    probe = adapter.architecture_probe()
    evidence = {item["id"]: item for item in adapter.query_evidence_catalog(probe)}

    if "content" not in evidence[evidence_id]:
        # Pi ships its documentation inside the npm package rather than in
        # assets/docs_cache/, so this evidence only exists once the pi runtime
        # is installed under <repo>/.pi-agent.
        pytest.skip(
            f"{evidence_id} documentation is unavailable: "
            f"{evidence[evidence_id]['source']}"
        )
    assert needle in evidence[evidence_id]["content"].lower()
    assert isinstance(evidence[evidence_id]["truncated"], bool)


@pytest.mark.parametrize("harness", ["pi", "codex"])
def test_discovery_validates_each_native_harness_contract(tmp_path, harness):
    module = DiscoveryModule(
        repo_root=REPO_ROOT,
        run_root=tmp_path / harness,
        budget=object(),
        harness=harness,
    )
    payload = module._harness_query_input({"environment": {}})
    unavailable = [
        {
            "id": channel["id"],
            "status": (
                "forbidden"
                if channel.get("policy_status") == "forbidden"
                else "conditional"
            ),
            "reason": "This test has not established a workspace edit path.",
            "runtime_constraints": ["A runtime probe is still required."],
            "evidence_refs": [channel["evidence_refs"][0]],
        }
        for channel in payload["surface_hypotheses"]
    ]
    output = {
        "harness": payload["architecture_probe"]["harness_id"],
        "harness_version": payload["architecture_probe"]["harness_version"],
        "summary": "Grounded native harness contract.",
        "modifiable_modules": [],
        "unavailable_modules": unavailable,
        "mcp_editable_points": [],
        "base_profile": {"shared_rules": [], "role_isolation": "fresh workspace"},
    }

    canonicalize_harness_query_output(output, payload)
    module._validate_harness_output(output, payload)

    output["harness"] = "opencode"
    with pytest.raises(ValueError, match="harness identity"):
        module._validate_harness_output(output, payload)


@pytest.mark.parametrize("harness", ["pi", "codex"])
def test_native_harness_query_binds_exact_environment_tool_targets(tmp_path, harness):
    module = DiscoveryModule(
        repo_root=REPO_ROOT,
        run_root=tmp_path / harness,
        budget=object(),
        harness=harness,
    )

    payload = module._harness_query_input(
        {
            "environment": {
                "tool_transport": {"kind": "mcp", "server_id": "bank"},
                "tools": [
                    {
                        "name": "lookup_account",
                        "parameters": {
                            "properties": {"account_id": {}, "include_closed": {}}
                        },
                    }
                ],
            }
        }
    )

    points = payload["predefined_mcp_editable_points"]
    assert points[0]["targets"] == ["lookup_account"]
    assert points[0]["status"] == "modifiable"
    assert points[0]["edit_contract"]["path"] == (
        ".harness-autoiter/mcp-tool-patches.json"
    )
    assert points[0]["operation"]["harness"] == harness
    assert points[1]["targets"] == [
        {
            "tool": "lookup_account",
            "parameters": ["account_id", "include_closed"],
        }
    ]


@pytest.mark.parametrize("harness", ["opencode", "pi", "codex"])
def test_every_verified_query_channel_can_form_an_analyzer_candidate(harness):
    adapter = harness_query_adapter(harness, repo_root=REPO_ROOT)
    contracts = [
        channel
        for channel in adapter.query_channel_inventory()
        if channel["status"] == "verified"
    ]

    for contract in contracts:
        _assert_candidate_closes_contract(harness, contract)


@pytest.mark.parametrize("harness", ["opencode", "pi", "codex"])
def test_environment_mcp_points_can_form_an_analyzer_candidate(harness):
    points = _mcp_editable_points(
        {
            "environment": {
                "tool_transport": {"kind": "mcp", "server_id": "query-server"},
                "tools": [
                    {
                        "name": "query_tool",
                        "parameters": {"properties": {"item_id": {}}},
                    }
                ],
            }
        },
        harness_id=harness,
    )

    for contract in points:
        _assert_candidate_closes_contract(harness, contract)


@pytest.mark.parametrize(
    "cell",
    ["retail", "banking", "terminal", "bird"],
)
@pytest.mark.parametrize("harness", ["opencode", "pi", "codex"])
def test_harness_query_exposes_correct_editable_surfaces_for_every_task_kind(
    tmp_path, cell, harness
):
    config = benchmark_config(REPO_ROOT, cell)
    task_id = config.train_task_ids[0]
    baseline = BaselineDataset(
        task_ids=(task_id,),
        trajectory_paths=(),
        trajectories_by_task={task_id: ()},
        evidence_by_path={},
        source_event="test",
    )
    task_input = benchmark_task_explorer_input(
        repo_root=REPO_ROOT,
        baseline=baseline,
        cell=cell,
    )
    module = DiscoveryModule(
        repo_root=REPO_ROOT,
        run_root=tmp_path / harness / config.cell,
        budget=object(),
        harness=harness,
    )

    payload = module._harness_query_input(task_input)

    assert payload["architecture_probe"]["harness_id"] == harness
    assert task_input["benchmark_kind"] == config.kind
    hypotheses = payload["surface_hypotheses"]
    assert hypotheses
    assert all("status" not in item and "operation" not in item for item in hypotheses)
    workspace_contract = payload["candidate_workspace_contract"]
    assert workspace_contract["captured_scopes"]
    assert workspace_contract["runtime_loading"]["config_paths"]
    loader_ref = workspace_contract["runtime_loading_evidence_ref"]
    assert loader_ref in {item["id"] for item in payload["evidence_catalog"]}
    points = payload["predefined_mcp_editable_points"]
    if config.kind == "terminal_bench":
        assert points == []
    else:
        assert {point["id"] for point in points} == {
            "mcp_tool_description",
            "mcp_tool_parameter_description",
        }
        assert all(point["status"] == "modifiable" for point in points)
        assert all(
            point["edit_contract"]["path"] == ".harness-autoiter/mcp-tool-patches.json"
            for point in points
        )
