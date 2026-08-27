from __future__ import annotations

from typing import Any


def patch_mcp_tool_schemas(
    tools: list[dict[str, Any]], patches: dict[str, Any]
) -> None:
    by_name = {str(tool.get("name") or ""): tool for tool in tools}
    for name, raw_patch in patches.items():
        tool = by_name.get(str(name))
        if tool is None or not isinstance(raw_patch, dict):
            continue
        description = str(raw_patch.get("desc") or "").strip()
        if description:
            tool["description"] = description
        params = raw_patch.get("params") or {}
        properties = (tool.get("inputSchema") or {}).get("properties") or {}
        if isinstance(params, dict) and isinstance(properties, dict):
            for param, value in params.items():
                schema = properties.get(str(param))
                if isinstance(schema, dict) and str(value).strip():
                    schema["description"] = str(value)
