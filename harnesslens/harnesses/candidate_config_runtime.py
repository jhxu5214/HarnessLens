from __future__ import annotations

import copy
import json
import re
import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence

from harnesslens.harnesses.native_candidate_runtime import apply_flat_config


_OPENCODE_RESERVED_AGENTS = {
    "build",
    "compaction",
    "explore",
    "general",
    "plan",
    "summary",
    "title",
}


def load_json_configs(paths: Sequence[str | Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid candidate JSON config {path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"candidate JSON config must be an object: {path}")
        result = deep_merge(result, payload)
    return result


def load_toml_configs(paths: Sequence[str | Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        try:
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"invalid candidate TOML config {path}: {exc}") from exc
        result = deep_merge(result, payload)
    return result


def merge_candidate_config(
    candidate: Mapping[str, Any],
    *,
    legacy_flat_patch: Mapping[str, Any] | None = None,
    fixed: Mapping[str, Any],
) -> dict[str, Any]:
    configured = apply_flat_config(candidate, legacy_flat_patch or {})
    return deep_merge(configured, fixed)


def relocate_opencode_instruction_paths(
    candidate: Mapping[str, Any],
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Resolve candidate-owned OpenCode instruction files inside its project."""
    configured = copy.deepcopy(dict(candidate))
    if "instructions" not in configured:
        return configured
    instructions = configured.get("instructions")
    if not isinstance(instructions, list):
        raise ValueError(
            "candidate OpenCode instructions must be a list of project paths"
        )

    root = Path(project_root).resolve()
    relocated: list[str] = []
    for raw_value in instructions:
        value = str(raw_value).strip()
        path = Path(value)
        if not value or path.is_absolute() or "://" in value or ".." in path.parts:
            raise ValueError(
                "candidate OpenCode instruction paths must stay inside the project workspace"
            )
        target = (root / path).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "candidate OpenCode instruction paths must stay inside the project workspace"
            ) from exc
        relocated.append(str(target))
    configured["instructions"] = relocated
    return configured


def compile_opencode_agent_definitions(project_root: str | Path) -> dict[str, Any]:
    """Compile project agent files without admitting model, tool, or permission overrides."""
    root = Path(project_root) / ".opencode" / "agents"
    result: dict[str, Any] = {}
    for path in sorted(root.glob("*.md")) if root.is_dir() else []:
        if path.stem in _OPENCODE_RESERVED_AGENTS:
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        match = re.match(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", content, re.S)
        if not match:
            continue
        frontmatter: dict[str, str] = {}
        for line in match.group(1).splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip():
                frontmatter[key.strip()] = value.strip().strip("'\"")
        description = frontmatter.get("description", "").strip()
        if not description:
            continue
        definition: dict[str, Any] = {
            "description": description,
            "mode": "subagent",
        }
        body = match.group(2).strip()
        if body:
            definition["prompt"] = body
        result[path.stem] = definition
    return result


def deep_merge(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(left))
    for key, value in right.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[str(key)] = deep_merge(result[key], value)
        else:
            result[str(key)] = copy.deepcopy(value)
    return result
