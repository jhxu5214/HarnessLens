from __future__ import annotations

import copy
import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence

from harnesslens.core.artifacts import write_json
from harnesslens.harnesses.tool_schema import patch_mcp_tool_schemas


PREFLIGHT_REPEAT_COUNT = 1
PREFLIGHT_CREATION_COST = 1
_MCP_CHANNELS = frozenset(
    {
        "mcp_tool_description",
        "mcp_tool_parameter_description",
        "tool_description",
        "tool_parameter_description",
    }
)
_INSTRUCTION_CHANNELS = frozenset(
    {"developer_instructions", "instructions_rules", "system_prompt"}
)
_AGENT_CHANNELS = frozenset({"agent_definitions"})
_HOOK_CHANNELS = frozenset({"hooks"})
_EFFECTIVE_CONFIG_CHANNELS = frozenset({"compaction", "compaction_config"})
_SKILL_ROOTS = {
    "pi": (".pi/skills",),
    "pi-agent": (".pi/skills",),
    "codex": (".agents/skills",),
    "opencode": (".opencode/skills",),
}


class ChannelPreflightError(RuntimeError):
    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = dict(report)
        failures = [
            str(item.get("detail") or item.get("channel_id") or "unknown check")
            for item in self.report.get("checks") or []
            if isinstance(item, Mapping) and not item.get("passed")
        ]
        super().__init__("; ".join(failures) or "candidate channel preflight failed")


def build_runtime_load_report(
    *,
    harness: str,
    project_root: str | Path,
    home_root: str | Path | None,
    manifest: Mapping[str, Any],
    tool_definitions: Sequence[Mapping[str, Any]],
    skills_available: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Capture the candidate surfaces after the runner has materialized its runtime."""

    project = Path(project_root).resolve()
    home = Path(home_root).resolve() if home_root is not None else None
    instruction_files = []
    agents = project / "AGENTS.md"
    if agents.is_file():
        content = agents.read_text(encoding="utf-8", errors="replace")
        instruction_files.append(
            {
                "path": "AGENTS.md",
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "content": content,
            }
        )

    discovered_skills = (
        [dict(item) for item in skills_available]
        if skills_available is not None
        else _discover_skills(str(harness), project, home)
    )
    effective_tools = copy.deepcopy([dict(item) for item in tool_definitions])
    patches = dict(manifest.get("tool_desc_patches") or {})
    patch_mcp_tool_schemas(effective_tools, patches)
    _patch_openai_tool_schemas(effective_tools, patches)
    return {
        "schema": "harnesslens.channel-load-report.v1",
        "harness": str(harness),
        "project_root": str(project),
        "project_instruction_files": instruction_files,
        "candidate_project_files": _candidate_project_files(manifest, project),
        "effective_config": _effective_runtime_config(str(harness), project, home),
        "agent_definitions": _discover_agent_definitions(str(harness), project),
        "runtime_events": _discover_runtime_events(project),
        "skills_available": discovered_skills,
        "tool_definitions": effective_tools,
        "prompt_appends": [str(item) for item in manifest.get("prompt_appends") or []],
        "instructions": [str(item) for item in manifest.get("instructions") or []],
        "config_patch": dict(manifest.get("config_patch") or {}),
    }


def refresh_runtime_load_report(
    report: dict[str, Any],
    *,
    project_root: str | Path,
    home_root: str | Path | None,
) -> dict[str, Any]:
    """Refresh surfaces that can change only after the first runtime turn."""

    project = Path(project_root).resolve()
    home = Path(home_root).resolve() if home_root is not None else None
    harness = str(report.get("harness") or "")
    report["effective_config"] = _effective_runtime_config(harness, project, home)
    report["agent_definitions"] = _discover_agent_definitions(harness, project)
    report["runtime_events"] = _discover_runtime_events(project)
    return report


def validate_channel_preflight(
    *,
    decision: Mapping[str, Any],
    rollout: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    candidate = decision.get("candidate") or {}
    channels = {
        str(item.get("channel_id") or "")
        for item in candidate.get("channel_diffs") or []
        if isinstance(item, Mapping)
    }
    rows = _retained_rows(rollout)
    checks: list[dict[str, Any]] = []
    checked_channels: set[str] = set()
    if len(rows) != 1:
        checks.append(
            {
                "channel_id": "runtime",
                "passed": False,
                "detail": f"preflight requires exactly one retained trial, found {len(rows)}",
            }
        )
    else:
        context = rows[0].get("model_context")
        if (
            not isinstance(context, Mapping)
            or context.get("schema") != "harnesslens.channel-load-report.v1"
        ):
            checks.append(
                {
                    "channel_id": "runtime",
                    "passed": False,
                    "detail": "retained trial has no HarnessLens channel-load report",
                }
            )
        else:
            request_text = _model_request_text(rows[0])
            runtime_harness = str(context.get("harness") or "")
            request_required = runtime_harness in {
                "opencode",
                "pi",
                "pi-agent",
                "codex",
            } and bool(
                channels
                & (
                    _INSTRUCTION_CHANNELS
                    | _MCP_CHANNELS
                    | _AGENT_CHANNELS
                    | _HOOK_CHANNELS
                    | {"project_instructions", "skills"}
                )
            )
            if request_required and not request_text:
                checks.append(
                    {
                        "channel_id": "runtime_request",
                        "passed": False,
                        "detail": (
                            f"{runtime_harness} preflight retained no model request evidence"
                        ),
                    }
                )
            if "project_instructions" in channels:
                checks.append(
                    _check_project_instructions(
                        candidate, context, request_text=request_text
                    )
                )
                checked_channels.add("project_instructions")
            if "skills" in channels:
                checks.extend(
                    _check_skills(candidate, context, request_text=request_text)
                )
                checked_channels.add("skills")
            if channels & _MCP_CHANNELS:
                checks.extend(
                    _check_tool_patches(candidate, context, request_text=request_text)
                )
                checked_channels.update(channels & _MCP_CHANNELS)
            for channel_id in sorted(channels & _INSTRUCTION_CHANNELS):
                checks.append(
                    _check_instruction_channel(
                        channel_id,
                        candidate,
                        context,
                        request_text=request_text,
                    )
                )
                checked_channels.add(channel_id)
            for channel_id in sorted(channels & _AGENT_CHANNELS):
                checks.extend(
                    _check_agent_definitions(
                        candidate,
                        context,
                        request_text=request_text,
                    )
                )
                checked_channels.add(channel_id)
            for channel_id in sorted(channels & _HOOK_CHANNELS):
                checks.append(
                    _check_hook_context(
                        candidate,
                        context,
                        request_text=request_text,
                    )
                )
                checked_channels.add(channel_id)
            for channel_id in sorted(channels & _EFFECTIVE_CONFIG_CHANNELS):
                checks.append(
                    _check_effective_config_channel(channel_id, candidate, context)
                )
                checked_channels.add(channel_id)

    unchecked_channels = channels - checked_channels
    for channel_id in sorted(unchecked_channels):
        checks.append(
            {
                "channel_id": channel_id,
                "passed": False,
                "detail": f"changed channel {channel_id!r} has no preflight checker",
            }
        )

    report = {
        "schema": "harnesslens.channel-preflight.v1",
        "candidate_id": str(candidate.get("id") or ""),
        "harness_version": str(decision.get("harness_version") or ""),
        "changed_channels": sorted(item for item in channels if item),
        "checked_channels": sorted(item for item in checked_channels if item),
        "unchecked_channels": sorted(item for item in unchecked_channels if item),
        "request_evidence_available": bool(
            len(rows) == 1 and _model_request_text(rows[0])
        ),
        "rollout_output": str(rollout.get("summary_json") or ""),
        "checks": checks,
        "passed": (
            bool(checks)
            and not unchecked_channels
            and checked_channels == channels
            and all(bool(item["passed"]) for item in checks)
        ),
    }
    write_json(output_path, report)
    if not report["passed"]:
        raise ChannelPreflightError(report)
    return report


def _check_project_instructions(
    candidate: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    request_text: str,
) -> dict[str, Any]:
    expected = [
        str(item.get("content") or "")
        for item in (candidate.get("workspace_delta") or {}).get("files") or []
        if isinstance(item, Mapping)
        and str(item.get("scope") or "") == "project"
        and str(item.get("path") or "") == "AGENTS.md"
        and str(item.get("change") or "") != "deleted"
    ]
    loaded = "\n".join(
        str(item.get("content") or "")
        for item in context.get("project_instruction_files") or []
        if isinstance(item, Mapping)
    )
    materialized = bool(expected) and all(
        item.strip() and item.strip() in loaded for item in expected
    )
    request_visible = not request_text or all(
        item.strip() and item.strip() in request_text for item in expected
    )
    passed = materialized and request_visible
    return {
        "channel_id": "project_instructions",
        "passed": passed,
        "detail": (
            "candidate AGENTS.md content is visible in the materialized runtime"
            if passed
            else "candidate AGENTS.md content is absent from the materialized runtime"
        ),
    }


def _check_skills(
    candidate: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    request_text: str,
) -> list[dict[str, Any]]:
    expected = {
        Path(str(item.get("path") or "")).parent.name
        for item in (candidate.get("workspace_delta") or {}).get("files") or []
        if isinstance(item, Mapping)
        and str(item.get("change") or "") != "deleted"
        and str(item.get("path") or "").endswith("/SKILL.md")
    }
    available = {
        str(item.get("name") or "")
        for item in context.get("skills_available") or []
        if isinstance(item, Mapping)
    }
    descriptions = {
        Path(str(item.get("path") or "")).parent.name: _skill_description(
            str(item.get("content") or "")
        )
        for item in (candidate.get("workspace_delta") or {}).get("files") or []
        if isinstance(item, Mapping)
        and str(item.get("change") or "") != "deleted"
        and str(item.get("path") or "").endswith("/SKILL.md")
    }
    if not expected:
        return [
            {
                "channel_id": "skills",
                "passed": False,
                "detail": "skills channel changed without a candidate SKILL.md artifact",
            }
        ]
    return [
        {
            "channel_id": "skills",
            "artifact": name,
            "passed": name in available
            and (
                not request_text
                or not descriptions.get(name)
                or descriptions[name] in request_text
            ),
            "detail": (
                f"skill {name!r} is available to the runtime"
                if name in available
                else f"skill {name!r} is not available to the runtime"
            ),
        }
        for name in sorted(expected)
    ]


def _check_tool_patches(
    candidate: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    request_text: str,
) -> list[dict[str, Any]]:
    patches = (candidate.get("manifest_delta") or {}).get("tool_desc_patches") or {}
    tools = {
        str(item.get("name") or ""): item
        for item in context.get("tool_definitions") or []
        if isinstance(item, Mapping)
    }
    checks = []
    for name, raw_patch in sorted(patches.items()):
        patch = raw_patch if isinstance(raw_patch, Mapping) else {}
        tool = tools.get(str(name)) or {}
        expected_desc = str(patch.get("desc") or "").strip()
        desc_ok = (
            not expected_desc or str(tool.get("description") or "") == expected_desc
        )
        expected_params = (
            patch.get("params") if isinstance(patch.get("params"), Mapping) else {}
        )
        schema = tool.get("inputSchema") or tool.get("parameters") or {}
        properties = schema.get("properties") or {}
        params_ok = all(
            isinstance(properties.get(str(param)), Mapping)
            and str(properties[str(param)].get("description") or "") == str(value)
            for param, value in expected_params.items()
        )
        expected_values = [
            item
            for item in [
                expected_desc,
                *(str(value) for value in expected_params.values()),
            ]
            if item
        ]
        request_ok = not request_text or all(
            value in request_text for value in expected_values
        )
        passed = bool(tool) and desc_ok and params_ok and request_ok
        checks.append(
            {
                "channel_id": "mcp_tool_schema",
                "artifact": str(name),
                "passed": passed,
                "detail": (
                    f"tool schema patch for {name!r} is present in the runtime schema"
                    if passed
                    else f"tool schema patch for {name!r} is absent from the runtime schema"
                ),
            }
        )
    if not checks:
        checks.append(
            {
                "channel_id": "mcp_tool_schema",
                "passed": False,
                "detail": "MCP schema channel changed without a manifest tool patch",
            }
        )
    return checks


def _check_instruction_channel(
    channel_id: str,
    candidate: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    request_text: str,
) -> dict[str, Any]:
    expected_files = [
        item
        for item in (candidate.get("workspace_delta") or {}).get("files") or []
        if isinstance(item, Mapping)
        and str(item.get("scope") or "") == "project"
        and str(item.get("change") or "") != "deleted"
        and _instruction_file_matches(channel_id, str(item.get("path") or ""))
    ]
    loaded_files = {
        str(item.get("path") or ""): str(item.get("content") or "")
        for item in context.get("candidate_project_files") or []
        if isinstance(item, Mapping)
    }
    file_checks = [
        _instruction_file_materialized(item, loaded_files, context)
        for item in expected_files
    ]
    manifest = candidate.get("manifest_delta") or {}
    expected_text = [
        str(item).strip()
        for item in [
            *(manifest.get("instructions") or []),
            *(manifest.get("prompt_appends") or []),
        ]
        if str(item).strip()
    ]
    loaded_text = [
        str(item).strip()
        for item in [
            *(context.get("instructions") or []),
            *(context.get("prompt_appends") or []),
        ]
        if str(item).strip()
    ]
    text_ok = all(item in loaded_text for item in expected_text)
    request_values = _instruction_request_values(
        channel_id, expected_files, expected_text
    )
    request_ok = not request_text or all(
        value in request_text for value in request_values
    )
    passed = (
        bool(file_checks or expected_text)
        and all(file_checks)
        and text_ok
        and request_ok
    )
    return {
        "channel_id": channel_id,
        "passed": passed,
        "detail": (
            f"{channel_id} artifact is present in the materialized runtime"
            if passed
            else f"{channel_id} artifact is absent from the materialized runtime"
        ),
    }


def _instruction_file_matches(channel_id: str, path: str) -> bool:
    if channel_id == "developer_instructions":
        return path == ".codex/config.toml"
    if channel_id == "system_prompt":
        return path in {"opencode.json", ".pi/APPEND_SYSTEM.md"}
    if channel_id == "instructions_rules":
        return path == "opencode.json" or (
            path.endswith((".md", ".txt"))
            and not path.startswith((".opencode/skills/", ".opencode/agents/"))
        )
    return False


def _instruction_file_materialized(
    item: Mapping[str, Any],
    loaded_files: Mapping[str, str],
    context: Mapping[str, Any],
) -> bool:
    path = str(item.get("path") or "")
    content = str(item.get("content") or "").strip()
    if not content:
        return False
    if path == "opencode.json":
        try:
            expected = json.loads(content)
            loaded = json.loads(loaded_files.get(path, ""))
        except (json.JSONDecodeError, TypeError):
            return False
        expected_entries = expected.get("instructions") or []
        effective_entries = (context.get("effective_config") or {}).get(
            "instructions"
        ) or loaded.get("instructions") or []
        project_root = str(context.get("project_root") or "").strip()

        def is_effective(entry: Any) -> bool:
            if entry in effective_entries:
                return True
            path = Path(str(entry))
            if not project_root or path.is_absolute():
                return False
            relocated = str((Path(project_root) / path).resolve())
            return relocated in effective_entries

        return bool(expected_entries) and all(is_effective(entry) for entry in expected_entries)
    return content in loaded_files.get(path, "")


def _check_agent_definitions(
    candidate: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    request_text: str,
) -> list[dict[str, Any]]:
    expected_paths = {
        str(item.get("path") or "")
        for item in (candidate.get("workspace_delta") or {}).get("files") or []
        if isinstance(item, Mapping)
        and str(item.get("change") or "") != "deleted"
        and (
            re.fullmatch(r"\.opencode/agents/[^/]+\.md", str(item.get("path") or ""))
            or re.fullmatch(r"\.codex/agents/[^/]+\.toml", str(item.get("path") or ""))
        )
    }
    loaded = {
        str(item.get("path") or ""): item
        for item in context.get("agent_definitions") or []
        if isinstance(item, Mapping)
    }
    if not expected_paths:
        return [
            {
                "channel_id": "agent_definitions",
                "passed": False,
                "detail": "agent_definitions changed without a native agent artifact",
            }
        ]
    checks = []
    for path in sorted(expected_paths):
        definition = loaded.get(path) or {}
        markers = [
            str(definition.get(key) or "").strip()
            for key in ("description", "instructions")
            if str(definition.get(key) or "").strip()
        ]
        visible = (
            bool(request_text)
            and bool(markers)
            and any(marker in request_text for marker in markers)
        )
        passed = bool(definition) and visible
        checks.append(
            {
                "channel_id": "agent_definitions",
                "artifact": path,
                "passed": passed,
                "detail": (
                    f"agent definition {path!r} is exposed to the active runtime"
                    if passed
                    else f"agent definition {path!r} was not visible in an active model request"
                ),
            }
        )
    return checks


def _check_hook_context(
    candidate: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    request_text: str,
) -> dict[str, Any]:
    expected = [
        str(item.get("content") or "").strip()
        for item in (candidate.get("workspace_delta") or {}).get("files") or []
        if isinstance(item, Mapping)
        and str(item.get("path") or "") == ".codex/harness-hook-context.md"
        and str(item.get("change") or "") != "deleted"
        and str(item.get("content") or "").strip()
    ]
    loaded = {
        str(item.get("path") or ""): str(item.get("content") or "")
        for item in context.get("candidate_project_files") or []
        if isinstance(item, Mapping)
    }
    materialized = bool(expected) and all(
        item in loaded.get(".codex/harness-hook-context.md", "") for item in expected
    )
    request_visible = bool(request_text) and all(
        item in request_text for item in expected
    )
    event_observed = any(
        isinstance(item, Mapping)
        and str(item.get("hook_event_name") or "").lower()
        in {"sessionstart", "session_start"}
        for item in context.get("runtime_events") or []
    )
    passed = materialized and request_visible
    return {
        "channel_id": "hooks",
        "passed": passed,
        "event_observed": event_observed,
        "detail": (
            "harness-owned SessionStart hook injected candidate context into the model request"
            if passed
            else "candidate hook context was not injected by the SessionStart hook"
        ),
    }


def _check_effective_config_channel(
    channel_id: str,
    candidate: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    path, selector = {
        "compaction_config": (".pi/settings.json", "compaction"),
        "compaction": (".codex/config.toml", "compact_prompt"),
    }[channel_id]
    expected_content = next(
        (
            str(item.get("content") or "")
            for item in (candidate.get("workspace_delta") or {}).get("files") or []
            if isinstance(item, Mapping)
            and str(item.get("path") or "") == path
            and str(item.get("change") or "") != "deleted"
        ),
        "",
    )
    expected_config = _parse_config_text(path, expected_content)
    effective_config = context.get("effective_config") or {}
    expected_value = _nested_config_value(expected_config, selector)
    effective_value = _nested_config_value(effective_config, selector)
    passed = expected_value is not None and expected_value == effective_value
    return {
        "channel_id": channel_id,
        "passed": passed,
        "selector": selector,
        "detail": (
            f"{channel_id} is present in the effective runtime configuration"
            if passed
            else f"{channel_id} was materialized but is absent from the effective runtime configuration"
        ),
    }


def _retained_rows(rollout: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for task in (rollout.get("per_task") or {}).values():
        if not isinstance(task, Mapping):
            continue
        for raw_path in task.get("trajectory_paths") or []:
            path = Path(str(raw_path))
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        value["_trajectory_path"] = str(path.resolve())
                        rows.append(value)
    return rows


def _model_request_text(row: Mapping[str, Any]) -> str:
    reference = str(row.get("api_calls_jsonl") or "").strip()
    if not reference:
        return ""
    path = Path(reference)
    if not path.is_absolute():
        trajectory = Path(str(row.get("_trajectory_path") or ""))
        path = trajectory.parent / path
    request_strings: list[str] = []
    if not path.is_file():
        return ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("role") or "agent") == "user_sim":
            continue
        request = entry.get("request")
        if isinstance(request, Mapping):
            request_strings.extend(_nested_request_strings(request))
    return "\n".join(request_strings)


def _nested_request_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        result: list[str] = []
        for item in value.values():
            result.extend(_nested_request_strings(item))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = []
        for item in value:
            result.extend(_nested_request_strings(item))
        return result
    return []


def _instruction_request_values(
    channel_id: str,
    expected_files: Sequence[Mapping[str, Any]],
    expected_text: Sequence[str],
) -> list[str]:
    values = [str(item) for item in expected_text if str(item).strip()]
    expected_paths = {
        str(item.get("path") or "")
        for item in expected_files
        if isinstance(item, Mapping)
    }
    for item in expected_files:
        path = str(item.get("path") or "")
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if channel_id == "developer_instructions" and path.endswith("config.toml"):
            parsed = _parse_config_text(path, content)
            developer_instructions = str(
                parsed.get("developer_instructions") or ""
            ).strip()
            values.append(developer_instructions or content)
        elif path.endswith(".json"):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                values.append(content)
            else:
                values.extend(
                    value
                    for value in _json_instruction_strings(parsed)
                    if value not in expected_paths
                )
        else:
            values.append(content)
    return [value for value in values if value]


def _json_instruction_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for child in value for item in _json_instruction_strings(child)]
    if isinstance(value, Mapping):
        result = []
        for key, child in value.items():
            if str(key) in {"instructions", "prompt"}:
                result.extend(_json_instruction_strings(child))
            elif isinstance(child, Mapping):
                result.extend(_json_instruction_strings(child))
        return result
    return []


def _discover_skills(
    harness: str, project_root: Path, home_root: Path | None
) -> list[dict[str, Any]]:
    del home_root
    result = []
    for relative_root in _SKILL_ROOTS.get(harness, ()):
        root = project_root / relative_root
        for path in sorted(root.glob("*/SKILL.md")) if root.is_dir() else []:
            content = path.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"(?m)^name:\s*['\"]?([^'\"\n]+)", content)
            name = match.group(1).strip() if match else path.parent.name
            result.append({"name": name, "path": str(path.resolve()), "n_calls": 0})
    return result


def _skill_description(content: str) -> str:
    match = re.search(r"(?m)^description:\s*['\"]?([^'\"\n]+)", str(content))
    return match.group(1).strip() if match else ""


def _effective_runtime_config(
    harness: str,
    project_root: Path,
    home_root: Path | None,
) -> dict[str, Any]:
    candidates: tuple[Path, ...]
    if harness == "opencode":
        candidates = (
            *(
                (home_root.parent.parent / ".hai" / "opencode.json",)
                if home_root
                else ()
            ),
            project_root / "opencode.json",
        )
    elif harness in {"pi", "pi-agent"}:
        candidates = (project_root / ".pi" / "settings.json",)
    elif harness == "codex":
        candidates = (
            *((home_root / "config.toml",) if home_root else ()),
            project_root / ".codex" / "config.toml",
        )
    else:
        candidates = ()
    for path in candidates:
        if not path.is_file():
            continue
        parsed = _parse_config_text(path.name, path.read_text(encoding="utf-8"))
        if parsed:
            return parsed
    return {}


def _parse_config_text(path: str, content: str) -> dict[str, Any]:
    if not str(content).strip():
        return {}
    try:
        parsed = (
            tomllib.loads(content)
            if str(path).endswith(".toml")
            else json.loads(content)
        )
    except (json.JSONDecodeError, tomllib.TOMLDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _nested_config_value(config: Mapping[str, Any], selector: str) -> Any:
    current: Any = config
    for part in str(selector).rstrip(".").split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _discover_agent_definitions(
    harness: str, project_root: Path
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if harness == "opencode":
        root = project_root / ".opencode" / "agents"
        for path in sorted(root.glob("*.md")) if root.is_dir() else []:
            content = path.read_text(encoding="utf-8", errors="replace")
            frontmatter, body = _markdown_frontmatter(content)
            result.append(
                {
                    "name": path.stem,
                    "path": path.relative_to(project_root).as_posix(),
                    "description": str(frontmatter.get("description") or "").strip(),
                    "instructions": body.strip(),
                }
            )
    elif harness == "codex":
        root = project_root / ".codex" / "agents"
        for path in sorted(root.glob("*.toml")) if root.is_dir() else []:
            parsed = _parse_config_text(path.name, path.read_text(encoding="utf-8"))
            result.append(
                {
                    "name": str(parsed.get("name") or path.stem),
                    "path": path.relative_to(project_root).as_posix(),
                    "description": str(parsed.get("description") or "").strip(),
                    "instructions": str(
                        parsed.get("developer_instructions") or ""
                    ).strip(),
                }
            )
    return result


def _markdown_frontmatter(content: str) -> tuple[dict[str, str], str]:
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", content, re.S)
    if not match:
        return {}, content
    frontmatter: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip():
            frontmatter[key.strip()] = value.strip().strip("'\"")
    return frontmatter, match.group(2)


def _discover_runtime_events(project_root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for relative in (".codex/harness-hook-observation.json",):
        path = project_root / relative
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping):
            result.append({"path": relative, **dict(payload)})
    return result


def _candidate_project_files(
    manifest: Mapping[str, Any], project_root: Path
) -> list[dict[str, Any]]:
    declared = []
    workspace = manifest.get("_workspace")
    if isinstance(workspace, Mapping):
        declared.extend(
            str(item.get("path") or "")
            for item in workspace.get("files") or []
            if isinstance(item, Mapping) and str(item.get("scope") or "") == "project"
        )
    declared.extend(
        str(item.get("path") or "")
        for item in manifest.get("files") or []
        if isinstance(item, Mapping)
    )
    result = []
    for relative in sorted(set(item for item in declared if item)):
        path = (project_root / relative).resolve()
        try:
            path.relative_to(project_root)
        except ValueError:
            continue
        if path.is_file():
            content = path.read_text(encoding="utf-8", errors="replace")
            result.append(
                {
                    "path": relative,
                    "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "content": content,
                }
            )
    return result


def _patch_openai_tool_schemas(
    tools: list[dict[str, Any]], patches: Mapping[str, Any]
) -> None:
    by_name = {str(tool.get("name") or ""): tool for tool in tools}
    for name, raw_patch in patches.items():
        tool = by_name.get(str(name))
        if tool is None or not isinstance(raw_patch, Mapping):
            continue
        description = str(raw_patch.get("desc") or "").strip()
        if description:
            tool["description"] = description
        parameters = tool.get("parameters") or {}
        properties = parameters.get("properties") or {}
        raw_params = raw_patch.get("params") or {}
        if not isinstance(properties, Mapping) or not isinstance(raw_params, Mapping):
            continue
        for param, value in raw_params.items():
            schema = properties.get(str(param))
            if isinstance(schema, dict) and str(value).strip():
                schema["description"] = str(value)
