from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from harnesslens.harnesses.candidate_config_runtime import (
    relocate_opencode_instruction_paths,
)
from harnesslens.benchmarks.native_tau2_worker import run_payload
from harnesslens.benchmarks.opencode_tau2 import (
    _is_retryable_empty_turn,
    _opencode_env,
    _opencode_runtime_root,
    _opencode_turn_retry_attempts,
    _write_opencode_project,
)
from harnesslens.harnesses.tool_schema import patch_mcp_tool_schemas


def test_opencode_instruction_paths_cannot_escape_candidate_project(tmp_path):
    with pytest.raises(ValueError, match="stay inside the project workspace"):
        relocate_opencode_instruction_paths(
            {"instructions": ["../host-guidance.md"]},
            project_root=tmp_path,
        )


def test_opencode_turn_retry_is_bounded_and_disabled_by_default(monkeypatch):
    monkeypatch.delenv("HAI_OPENCODE_TURN_RETRY_ATTEMPTS", raising=False)
    assert _opencode_turn_retry_attempts() == 0

    monkeypatch.setenv("HAI_OPENCODE_TURN_RETRY_ATTEMPTS", "99")
    assert _opencode_turn_retry_attempts() == 3


def test_opencode_turn_retry_requires_an_empty_retryable_failure():
    assert _is_retryable_empty_turn(
        stderr="provider request failed with HTTP 429: insufficient_quota",
        returncode=1,
        agent_text="",
        new_calls=[],
    )
    assert _is_retryable_empty_turn(
        stderr="TIMEOUT after 180s",
        returncode=None,
        agent_text="",
        new_calls=[],
    )
    assert not _is_retryable_empty_turn(
        stderr="provider request failed with HTTP 429: insufficient_quota",
        returncode=1,
        agent_text="I already completed the update.",
        new_calls=[],
    )
    assert not _is_retryable_empty_turn(
        stderr="TIMEOUT after 180s",
        returncode=None,
        agent_text="",
        new_calls=[{"name": "change_user_email"}],
    )


def test_opencode_tau2_merges_candidate_config_then_restores_fixed_invariants(tmp_path):
    runtime_cwd = tmp_path / "runtime"
    runtime_home = tmp_path / "home"
    manifest = {
        "config_patch": {},
        "files": [],
        "instructions": ["Keep confirmations concise."],
        "prompt_appends": [],
        "tool_desc_patches": {},
        "_workspace": {
            "schema": 1,
            "files": [
                {
                    "scope": "home",
                    "path": "opencode.json",
                    "content": json.dumps(
                        {
                            "model": "candidate/forbidden",
                            "mcp": {
                                "candidate": {
                                    "type": "local",
                                    "command": ["./candidate-mcp"],
                                }
                            },
                        }
                    ),
                    "executable": False,
                },
                {
                    "scope": "project",
                    "path": "opencode.json",
                    "content": json.dumps(
                        {
                            "agent": {"build": {"prompt": "Candidate framing."}},
                            "instructions": ["candidate-guidance.md"],
                            "tools": {"bash": True, "read": True},
                        }
                    ),
                    "executable": False,
                },
                {
                    "scope": "project",
                    "path": "candidate-guidance.md",
                    "content": "Candidate project guidance.\n",
                    "executable": False,
                },
                {
                    "scope": "project",
                    "path": ".opencode/agents/evidence-reviewer.md",
                    "content": (
                        "---\n"
                        "description: Reviews evidence before the parent answers.\n"
                        "mode: primary\n"
                        "model: candidate/forbidden\n"
                        "permission: allow\n"
                        "---\n\n"
                        "Check the task evidence and report contradictions.\n"
                    ),
                    "executable": False,
                },
                {
                    "scope": "project",
                    "path": ".opencode/plugins/candidate.ts",
                    "content": "export const Candidate = async () => ({})\n",
                    "executable": False,
                },
            ],
        },
    }

    config_path = _write_opencode_project(
        repo_root=tmp_path,
        runtime_cwd=runtime_cwd,
        runtime_home=runtime_home,
        socket_path="/tmp/tau2.sock",
        proxy_port=12345,
        system_prompt="Fixed task policy.",
        max_steps=10,
        harness_manifest=manifest,
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert (runtime_cwd / ".opencode/plugins/candidate.ts").is_file()
    assert config["model"] == "deepseek/deepseek-v4-flash"
    assert config["enabled_providers"] == ["deepseek"]
    assert config["provider"]["deepseek"]["models"]["deepseek-v4-flash"][
        "limit"
    ] == {"context": 1_000_000, "output": 24_576}
    assert (
        config["provider"]["deepseek"]["options"]["baseURL"]
        == "http://127.0.0.1:12345/v1"
    )
    assert config["agent"]["build"]["steps"] == 10
    assert "max_subagent_depth" not in config
    assert config["agent"]["evidence-reviewer"] == {
        "description": "Reviews evidence before the parent answers.",
        "mode": "subagent",
        "prompt": "Check the task evidence and report contradictions.",
    }
    assert "Fixed task policy." in config["agent"]["build"]["prompt"]
    assert "Candidate framing." in config["agent"]["build"]["prompt"]
    assert "Keep confirmations concise." in config["agent"]["build"]["prompt"]
    assert config["instructions"] == [str(runtime_cwd / "candidate-guidance.md")]
    assert config["tools"]["bash"] is False
    assert config["tools"]["read"] is False
    assert "candidate" in config["mcp"]
    assert config["mcp"]["tau2"]["command"][-2:] == ["--socket", "/tmp/tau2.sock"]


def test_opencode_tau2_fixed_config_is_not_written_to_global_home(tmp_path):
    global_home = tmp_path / "global-codex-home"
    global_home.mkdir()
    marker = global_home / "config.toml"
    marker.write_text("unchanged\n", encoding="utf-8")

    _write_opencode_project(
        repo_root=tmp_path,
        runtime_cwd=tmp_path / "runtime",
        runtime_home=tmp_path / "isolated-home",
        socket_path="/tmp/tau2.sock",
        proxy_port=12345,
        system_prompt="Fixed task policy.",
        max_steps=10,
        harness_manifest={},
    )

    assert marker.read_text(encoding="utf-8") == "unchanged\n"


def test_tau2_mcp_applies_candidate_description_patch_without_changing_schema():
    tools = [
        {
            "name": "lookup",
            "description": "Old description",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        }
    ]

    patch_mcp_tool_schemas(
        tools,
        {
            "lookup": {
                "desc": "Use for exact account lookups.",
                "params": {"query": "Use the complete account identifier."},
            }
        },
    )

    assert tools[0]["description"] == "Use for exact account lookups."
    assert tools[0]["inputSchema"]["properties"]["query"] == {
        "type": "string",
        "description": "Use the complete account identifier.",
    }


def test_native_tau2_worker_dispatches_opencode_payload(monkeypatch, tmp_path):
    captured = {}

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(to_dict=lambda: {"ok": True})

    monkeypatch.setattr(
        "harnesslens.benchmarks.opencode_tau2.run_opencode_tau2_test_baseline",
        fake_runner,
    )
    payload = {
        "harness": "opencode",
        "repo_root": str(tmp_path),
        "run_root": str(tmp_path / "run"),
        "benchmark": "retail",
        "request": {
            "request_id": "request",
            "run_id": "run",
            "scope": "TRAIN",
            "harness_version": "candidate-01",
            "task_repeats": {"0": 1},
            "max_concurrency": 1,
            "purpose": "test",
            "pairing_offsets": {"0": 0},
        },
        "retrieval_config": None,
        "limits": {
            "max_conversation_turns": 2,
            "timeout_per_turn_s": 3,
            "max_tool_calls_per_turn": 4,
            "max_tool_calls": 5,
            "group_timeout_s": 6,
            "timeout_retries_per_turn": 1,
        },
        "harness_manifest": {"_workspace": {"schema": 1, "files": []}},
    }

    response = run_payload(payload)

    assert response.to_dict() == {"ok": True}
    assert captured["request"].scope == "TRAIN"
    assert captured["split"].cell == "retail"
    assert captured["limits"].max_tool_calls_per_turn == 4
    assert captured["harness_manifest"] == payload["harness_manifest"]


def test_opencode_tau2_creates_only_isolated_runtime_state_directories(tmp_path):
    runtime_home = tmp_path / "trial" / "home"
    config_path = runtime_home / ".hai" / "opencode.json"

    env = _opencode_env(runtime_home=runtime_home, config_path=config_path)

    assert env["HOME"] == str(runtime_home)
    assert env["OPENCODE_CONFIG"] == str(config_path)
    for name in (
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "XDG_STATE_HOME",
        "TMPDIR",
    ):
        assert Path(env[name]).is_dir()
        assert Path(env[name]).is_relative_to(runtime_home)
        if name != "TMPDIR":
            assert Path(env[name], "opencode").is_dir()


def test_opencode_tau2_runtime_root_is_short_isolated_and_request_specific(tmp_path):
    first = _opencode_runtime_root(tmp_path / "request-a", "retail", "5", 0)
    second = _opencode_runtime_root(tmp_path / "request-b", "retail", "5", 0)

    assert first.parent == Path("/tmp/harnesslens_opencode_runtime")
    assert len(first.name) == 20
    assert first != second
