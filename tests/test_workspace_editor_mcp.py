from pathlib import Path

from harnesslens.harnesses.workspace_editor_mcp import WorkspaceEditorServer


def test_workspace_editor_mcp_reads_and_writes_only_below_root(tmp_path: Path):
    root = tmp_path / "candidate"
    root.mkdir()
    server = WorkspaceEditorServer(root)

    written = server.call(
        "write_file", {"path": "project/AGENTS.md", "content": "Verify IDs.\n"}
    )
    read = server.call("read_file", {"path": "project/AGENTS.md"})

    assert written["content"][0]["text"] == "wrote project/AGENTS.md"
    assert read["content"][0]["text"] == "Verify IDs.\n"


def test_workspace_editor_mcp_rejects_parent_escape(tmp_path: Path):
    root = tmp_path / "candidate"
    root.mkdir()
    server = WorkspaceEditorServer(root)

    result = server.call(
        "write_file", {"path": "../leak.txt", "content": "forbidden"}
    )

    assert result["isError"] is True
    assert "escapes" in result["content"][0]["text"]
    assert not (tmp_path / "leak.txt").exists()


def test_workspace_editor_mcp_rejects_symlink_escape(tmp_path: Path):
    root = tmp_path / "candidate"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    server = WorkspaceEditorServer(root)

    result = server.call(
        "write_file", {"path": "link/leak.txt", "content": "forbidden"}
    )

    assert result["isError"] is True
    assert not (outside / "leak.txt").exists()
