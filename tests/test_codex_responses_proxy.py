import harnesslens.infrastructure.codex_responses_proxy as proxy
from harnesslens.harnesses.workspace_editor_mcp import WorkspaceEditorServer


def test_upstream_uses_current_model_endpoint(monkeypatch):
    monkeypatch.setenv(
        "DEEPSEEK_URL", "https://legacy.example.invalid/v1/chat/completions"
    )
    monkeypatch.setenv(
        "DEEPSEEK_BASE_URL", "https://relay.example.invalid/v1"
    )

    assert proxy._upstream_base_url() == (
        "https://relay.example.invalid/v1"
    )


def test_upstream_normalizes_chat_completions_suffix(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.setenv(
        "DEEPSEEK_URL", "https://gateway.example.invalid/v1/chat/completions/"
    )

    assert proxy._upstream_base_url() == "https://gateway.example.invalid/v1"


def test_translate_request_preserves_tool_search_alongside_builtin_tools(
    monkeypatch,
):
    monkeypatch.setattr(proxy, "_TAU2_SOCKET", None)
    request = {
        "instructions": "Edit the candidate.",
        "input": [{"role": "user", "content": "Write one file."}],
        "tools": [
            {
                "type": "function",
                "name": "update_plan",
                "description": "Update a plan.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "type": "tool_search",
                "description": "Discover deferred MCP tools.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        ],
    }

    translated = proxy._translate_request(request)

    names = [item["function"]["name"] for item in translated["tools"]]
    assert names == ["update_plan", "tool_search"]


def test_translate_request_filters_explicitly_disabled_tool(monkeypatch):
    monkeypatch.setattr(proxy, "_TAU2_SOCKET", None)
    monkeypatch.setattr(proxy, "_DISABLED_TOOLS", {"view_image"})

    translated = proxy._translate_request(
        {
            "input": [{"role": "user", "content": "Inspect the workspace."}],
            "tools": [
                {
                    "type": "function",
                    "name": "view_image",
                    "description": "View an image.",
                    "parameters": {"type": "object", "properties": {}},
                },
                {
                    "type": "function",
                    "name": "exec_command",
                    "description": "Run a command.",
                    "parameters": {"type": "object", "properties": {}},
                },
            ],
        }
    )

    names = [item["function"]["name"] for item in translated["tools"]]
    assert names == ["exec_command"]


def test_workspace_editor_mode_replaces_deferred_catalog_with_bounded_tools(
    tmp_path, monkeypatch
):
    root = tmp_path / "candidate"
    root.mkdir()
    monkeypatch.setattr(proxy, "_TAU2_SOCKET", None)
    monkeypatch.setattr(proxy, "_WORKSPACE_EDITOR", WorkspaceEditorServer(root))

    translated = proxy._translate_request(
        {
            "input": [{"role": "user", "content": "Write one file."}],
            "tools": [
                {
                    "type": "function",
                    "name": "update_plan",
                    "description": "Update plan.",
                    "parameters": {"type": "object", "properties": {}},
                },
                {
                    "type": "tool_search",
                    "description": "Discover tools.",
                    "parameters": {"type": "object", "properties": {}},
                },
            ],
        }
    )

    names = [item["function"]["name"] for item in translated["tools"]]
    assert names == ["list_files", "read_file", "write_file"]
    assert proxy._execute_direct_tool(
        "write_file", {"path": "project/AGENTS.md", "content": "Verify IDs.\n"}
    ) == "wrote project/AGENTS.md"
    assert (root / "project" / "AGENTS.md").read_text() == "Verify IDs.\n"
