from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Mapping


MANIFEST_FIELDS = (
    "config_patch",
    "files",
    "instructions",
    "prompt_appends",
    "tool_desc_patches",
)


def normalize_harness(harness: str) -> str:
    normalized = str(harness).strip().lower().replace("-", "_")
    if normalized == "pi_agent":
        normalized = "pi"
    if normalized not in {"opencode", "pi", "codex"}:
        raise ValueError(f"unsupported target harness: {harness}")
    return normalized


def empty_harness_manifest() -> dict[str, Any]:
    return {
        "config_patch": {},
        "files": [],
        "instructions": [],
        "prompt_appends": [],
        "tool_desc_patches": {},
    }


def normalize_native_manifest(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("native harness manifest must be an object")
    unknown = sorted(set(raw) - set(MANIFEST_FIELDS))
    if unknown:
        raise ValueError(f"unsupported native manifest fields: {unknown}")
    files = raw.get("files") or []
    config = raw.get("config_patch") or {}
    tool_patches = raw.get("tool_desc_patches") or {}
    if not isinstance(files, list) or any(not isinstance(item, Mapping) for item in files):
        raise ValueError("native manifest files must be an array of objects")
    if not isinstance(config, Mapping) or not isinstance(tool_patches, Mapping):
        raise ValueError("native config and tool patches must be objects")
    normalized_files: list[dict[str, str]] = []
    for item in files:
        path = str(item.get("path") or "")
        parsed = PurePosixPath(path)
        content = str(item.get("content") or "")
        if not path or parsed.is_absolute() or ".." in parsed.parts or "\\" in path:
            raise ValueError(f"unsafe native candidate path: {path}")
        if not content.strip():
            raise ValueError("native candidate file content must be nonempty")
        normalized_files.append({"path": path, "content": content})
    normalized = empty_harness_manifest()
    normalized["config_patch"] = dict(config)
    normalized["files"] = normalized_files
    normalized["tool_desc_patches"] = _normalize_tool_patches(tool_patches)
    for field in ("instructions", "prompt_appends"):
        value = raw.get(field) or []
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            raise ValueError(f"native manifest {field} must contain nonempty strings")
        normalized[field] = list(value)
    return normalized


def merge_native_manifests(
    base: Mapping[str, Any], delta: Mapping[str, Any]
) -> dict[str, Any]:
    left = normalize_native_manifest(base)
    right = normalize_native_manifest(delta)
    files = {
        str(item["path"]): dict(item)
        for item in left["files"]
    }
    files.update({str(item["path"]): dict(item) for item in right["files"]})
    return {
        "config_patch": _deep_merge(left["config_patch"], right["config_patch"]),
        "files": [files[path] for path in sorted(files)],
        "instructions": list(
            dict.fromkeys([*left["instructions"], *right["instructions"]])
        ),
        "prompt_appends": list(
            dict.fromkeys([*left["prompt_appends"], *right["prompt_appends"]])
        ),
        "tool_desc_patches": _deep_merge(
            left["tool_desc_patches"], right["tool_desc_patches"]
        ),
    }


def _normalize_tool_patches(raw: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for tool_name, value in raw.items():
        if not str(tool_name).strip() or not isinstance(value, Mapping):
            raise ValueError("native tool patches require named objects")
        unknown = sorted(set(value) - {"desc", "params"})
        if unknown:
            raise ValueError(f"unsupported native tool patch fields: {unknown}")
        patch: dict[str, Any] = {}
        if "desc" in value:
            description = str(value.get("desc") or "").strip()
            if not description:
                raise ValueError("native tool description patch cannot be empty")
            patch["desc"] = description
        params = value.get("params") or {}
        if not isinstance(params, Mapping) or any(
            not str(name).strip() or not str(description).strip()
            for name, description in params.items()
        ):
            raise ValueError("native parameter patches require nonempty descriptions")
        if params:
            patch["params"] = {
                str(name): str(description) for name, description in params.items()
            }
        if not patch:
            raise ValueError("native tool patch cannot be empty")
        normalized[str(tool_name)] = patch
    return normalized


def _deep_merge(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(left)
    for key, value in right.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
