from pathlib import Path

import pytest

from harnesslens.harnesses.harness_workspace import (
    MCP_PATCH_WORKSPACE_PATH,
    capture_workspace,
    diff_workspace,
    empty_workspace_snapshot,
    extract_mcp_tool_patches,
    materialize_workspace,
    normalize_workspace_snapshot,
    seed_workspace,
    workspace_digest,
)


def test_workspace_snapshot_round_trips_home_and_project_files(tmp_path: Path):
    root = tmp_path / "editor"
    (root / "home" / ".config").mkdir(parents=True)
    (root / "project" / ".harness").mkdir(parents=True)
    (root / "home" / ".config" / "settings.json").write_text(
        '{"enabled": true}\n', encoding="utf-8"
    )
    executable = root / "project" / ".harness" / "hook.sh"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)

    snapshot = capture_workspace(root)
    destination = seed_workspace(tmp_path / "copy", snapshot)

    assert capture_workspace(destination) == snapshot
    assert snapshot["files"][0]["scope"] == "home"
    assert snapshot["files"][1]["executable"] is True


def test_workspace_diff_reports_add_modify_delete_without_contents():
    base = {
        "files": [
            {"scope": "home", "path": "old.txt", "content": "old"},
            {"scope": "project", "path": "same.txt", "content": "same"},
            {"scope": "project", "path": "edit.txt", "content": "before"},
        ]
    }
    candidate = {
        "files": [
            {"scope": "project", "path": "same.txt", "content": "same"},
            {"scope": "project", "path": "edit.txt", "content": "after"},
            {"scope": "project", "path": "new.txt", "content": "new"},
        ]
    }

    changes = diff_workspace(base, candidate)

    assert [(item["path"], item["change"]) for item in changes] == [
        ("old.txt", "deleted"),
        ("edit.txt", "modified"),
        ("new.txt", "added"),
    ]
    assert all("content" not in item for item in changes)


def test_workspace_materializes_into_explicit_runtime_roots(tmp_path: Path):
    snapshot = normalize_workspace_snapshot(
        {
            "files": [
                {"scope": "home", "path": ".codex/config.toml", "content": "x = 1\n"},
                {"scope": "project", "path": "AGENTS.md", "content": "Rule.\n"},
            ]
        }
    )

    materialize_workspace(
        snapshot,
        home_root=tmp_path / "runtime-home",
        project_root=tmp_path / "runtime-project",
    )

    assert (tmp_path / "runtime-home" / ".codex" / "config.toml").is_file()
    assert (tmp_path / "runtime-project" / "AGENTS.md").is_file()


@pytest.mark.parametrize(
    "entry",
    [
        {"scope": "project", "path": "../escape", "content": "x"},
        {"scope": "host", "path": "file", "content": "x"},
        {"scope": "project", "path": "/absolute", "content": "x"},
    ],
)
def test_workspace_rejects_unsafe_or_unknown_locations(entry):
    with pytest.raises(ValueError):
        normalize_workspace_snapshot({"files": [entry]})


def test_workspace_digest_is_order_independent():
    first = {
        "files": [
            {"scope": "project", "path": "b", "content": "2"},
            {"scope": "home", "path": "a", "content": "1"},
        ]
    }
    second = {"files": list(reversed(first["files"]))}

    assert workspace_digest(first) == workspace_digest(second)
    assert workspace_digest(first) != workspace_digest(empty_workspace_snapshot())


def test_mcp_patch_bridge_is_extracted_and_removed_from_runtime_workspace():
    workspace, patches = extract_mcp_tool_patches(
        {
            "files": [
                {
                    "scope": "project",
                    "path": MCP_PATCH_WORKSPACE_PATH,
                    "content": (
                        '{"lookup_record":{"desc":"Use exact IDs.",'
                        '"params":{"record_id":"The exact record ID."}}}'
                    ),
                },
                {
                    "scope": "project",
                    "path": "AGENTS.md",
                    "content": "Keep existing behavior.\n",
                },
            ]
        }
    )

    assert patches == {
        "lookup_record": {
            "desc": "Use exact IDs.",
            "params": {"record_id": "The exact record ID."},
        }
    }
    assert [item["path"] for item in workspace["files"]] == ["AGENTS.md"]


@pytest.mark.parametrize(
    "content",
    [
        "not-json",
        "{}",
        '{"lookup_record":{"unknown":"value"}}',
        '{"lookup_record":{"params":{"record_id":""}}}',
        '{"lookup_record":{"desc":{"description":"nested"}}}',
        '{"lookup_record":{"params":{"record_id":{"description":"nested"}}}}',
    ],
)
def test_mcp_patch_bridge_rejects_invalid_payloads(content):
    with pytest.raises(ValueError):
        extract_mcp_tool_patches(
            {
                "files": [
                    {
                        "scope": "project",
                        "path": MCP_PATCH_WORKSPACE_PATH,
                        "content": content,
                    }
                ]
            }
        )
