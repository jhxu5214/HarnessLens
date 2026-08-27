from __future__ import annotations

import hashlib
import json
import shutil
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


WORKSPACE_SCHEMA = 1
WORKSPACE_SCOPES = ("home", "project")
MCP_PATCH_WORKSPACE_PATH = ".harness-autoiter/mcp-tool-patches.json"
DEFAULT_MAX_FILE_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 16 * 1024 * 1024


def empty_workspace_snapshot() -> dict[str, Any]:
    return {"schema": WORKSPACE_SCHEMA, "files": []}


def normalize_workspace_snapshot(
    raw: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if raw is None:
        return empty_workspace_snapshot()
    if not isinstance(raw, Mapping):
        raise ValueError("candidate workspace snapshot must be an object")
    unknown = sorted(set(raw) - {"schema", "files"})
    if unknown:
        raise ValueError(f"unsupported candidate workspace fields: {unknown}")
    schema = int(raw.get("schema") or WORKSPACE_SCHEMA)
    if schema != WORKSPACE_SCHEMA:
        raise ValueError(f"unsupported candidate workspace schema: {schema}")
    raw_files = raw.get("files") or []
    if not isinstance(raw_files, list):
        raise ValueError("candidate workspace files must be an array")
    files: dict[tuple[str, str], dict[str, Any]] = {}
    total_bytes = 0
    for item in raw_files:
        if not isinstance(item, Mapping):
            raise ValueError("candidate workspace file entries must be objects")
        unknown_file = sorted(set(item) - {"scope", "path", "content", "executable"})
        if unknown_file:
            raise ValueError(
                f"unsupported candidate workspace file fields: {unknown_file}"
            )
        scope = str(item.get("scope") or "")
        if scope not in WORKSPACE_SCOPES:
            raise ValueError(f"invalid candidate workspace scope: {scope}")
        path = _safe_relative_path(str(item.get("path") or ""))
        content = item.get("content")
        if not isinstance(content, str):
            raise ValueError("candidate workspace file content must be text")
        size = len(content.encode("utf-8"))
        if size > DEFAULT_MAX_FILE_BYTES:
            raise ValueError(f"candidate workspace file is too large: {scope}/{path}")
        total_bytes += size
        if total_bytes > DEFAULT_MAX_TOTAL_BYTES:
            raise ValueError("candidate workspace snapshot is too large")
        key = (scope, path)
        if key in files:
            raise ValueError(f"duplicate candidate workspace path: {scope}/{path}")
        files[key] = {
            "scope": scope,
            "path": path,
            "content": content,
            "executable": bool(item.get("executable", False)),
        }
    return {
        "schema": WORKSPACE_SCHEMA,
        "files": [files[key] for key in sorted(files)],
    }


def capture_workspace(root: str | Path) -> dict[str, Any]:
    workspace = Path(root).resolve()
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for scope in WORKSPACE_SCOPES:
        scope_root = workspace / scope
        if not scope_root.exists():
            continue
        if scope_root.is_symlink() or not scope_root.is_dir():
            raise ValueError(f"candidate workspace scope is not a directory: {scope}")
        for path in sorted(scope_root.rglob("*")):
            if path.is_symlink():
                raise ValueError(
                    f"candidate workspace cannot contain symlinks: {path.relative_to(workspace)}"
                )
            if path.is_dir():
                continue
            mode = path.stat().st_mode
            if not stat.S_ISREG(mode):
                raise ValueError(
                    f"candidate workspace contains a special file: {path.relative_to(workspace)}"
                )
            size = path.stat().st_size
            if size > DEFAULT_MAX_FILE_BYTES:
                raise ValueError(
                    f"candidate workspace file is too large: {path.relative_to(workspace)}"
                )
            total_bytes += size
            if total_bytes > DEFAULT_MAX_TOTAL_BYTES:
                raise ValueError("candidate workspace snapshot is too large")
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"candidate workspace file must be UTF-8 text: {path.relative_to(workspace)}"
                ) from exc
            files.append(
                {
                    "scope": scope,
                    "path": path.relative_to(scope_root).as_posix(),
                    "content": content,
                    "executable": bool(mode & stat.S_IXUSR),
                }
            )
    return normalize_workspace_snapshot(
        {"schema": WORKSPACE_SCHEMA, "files": files}
    )


def seed_workspace(
    root: str | Path, snapshot: Mapping[str, Any] | None
) -> Path:
    workspace = Path(root).resolve()
    if workspace.exists():
        shutil.rmtree(workspace)
    for scope in WORKSPACE_SCOPES:
        (workspace / scope).mkdir(parents=True, exist_ok=True)
    materialize_workspace(
        snapshot,
        home_root=workspace / "home",
        project_root=workspace / "project",
    )
    return workspace


def materialize_workspace(
    snapshot: Mapping[str, Any] | None,
    *,
    home_root: str | Path,
    project_root: str | Path,
) -> None:
    normalized = normalize_workspace_snapshot(snapshot)
    roots = {
        "home": Path(home_root).resolve(),
        "project": Path(project_root).resolve(),
    }
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    for item in normalized["files"]:
        root = roots[str(item["scope"])]
        target = (root / str(item["path"])).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"candidate workspace path escaped its scope: {target}") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(item["content"]), encoding="utf-8")
        target.chmod(0o755 if item["executable"] else 0o644)


def diff_workspace(
    base: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    left = _file_index(normalize_workspace_snapshot(base))
    right = _file_index(normalize_workspace_snapshot(candidate))
    changes: list[dict[str, Any]] = []
    for key in sorted(set(left) | set(right)):
        before = left.get(key)
        after = right.get(key)
        if before == after:
            continue
        changes.append(
            {
                "scope": key[0],
                "path": key[1],
                "change": "added" if before is None else "deleted" if after is None else "modified",
                "before_sha256": _entry_digest(before),
                "after_sha256": _entry_digest(after),
                "executable_before": (
                    bool(before["executable"]) if before is not None else None
                ),
                "executable_after": (
                    bool(after["executable"]) if after is not None else None
                ),
            }
        )
    return changes


def workspace_digest(snapshot: Mapping[str, Any] | None) -> str:
    normalized = normalize_workspace_snapshot(snapshot)
    encoded = json.dumps(
        normalized, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def extract_mcp_tool_patches(
    snapshot: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized = normalize_workspace_snapshot(snapshot)
    retained: list[dict[str, Any]] = []
    patches: dict[str, Any] = {}
    for item in normalized["files"]:
        if not (
            item["scope"] == "project"
            and item["path"] == MCP_PATCH_WORKSPACE_PATH
        ):
            retained.append(dict(item))
            continue
        try:
            raw = json.loads(str(item["content"]))
        except json.JSONDecodeError as exc:
            raise ValueError("MCP tool patch bridge must contain valid JSON") from exc
        if not isinstance(raw, Mapping):
            raise ValueError("MCP tool patch bridge must contain one JSON object")
        patches = _normalize_mcp_tool_patches(raw)
    return normalize_workspace_snapshot(
        {"schema": WORKSPACE_SCHEMA, "files": retained}
    ), patches


def _normalize_mcp_tool_patches(raw: Mapping[str, Any]) -> dict[str, Any]:
    patches: dict[str, Any] = {}
    for raw_tool, raw_patch in raw.items():
        tool = str(raw_tool).strip()
        if not tool or not isinstance(raw_patch, Mapping):
            raise ValueError("MCP tool patches require exact tool names and objects")
        unknown = set(raw_patch) - {"desc", "params"}
        if unknown:
            raise ValueError(f"unsupported MCP tool patch fields: {sorted(unknown)}")
        patch: dict[str, Any] = {}
        if "desc" in raw_patch:
            raw_description = raw_patch.get("desc")
            if not isinstance(raw_description, str):
                raise ValueError("MCP tool description patch must be text")
            description = raw_description.strip()
            if not description:
                raise ValueError("MCP tool description patch cannot be empty")
            patch["desc"] = description
        if "params" in raw_patch:
            raw_params = raw_patch.get("params")
            if not isinstance(raw_params, Mapping):
                raise ValueError("MCP parameter patches must be an object")
            if any(not isinstance(value, str) for value in raw_params.values()):
                raise ValueError("MCP parameter descriptions must be text")
            params = {
                str(name).strip(): value.strip()
                for name, value in raw_params.items()
                if str(name).strip() and value.strip()
            }
            if len(params) != len(raw_params):
                raise ValueError("MCP parameter patches require nonempty names and descriptions")
            patch["params"] = params
        if not patch:
            raise ValueError("MCP tool patch cannot be empty")
        patches[tool] = patch
    if not patches:
        raise ValueError("MCP tool patch bridge cannot be empty")
    return patches


def _safe_relative_path(raw: str) -> str:
    path = PurePosixPath(raw)
    if (
        not raw
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or "\\" in raw
    ):
        raise ValueError(f"unsafe candidate workspace path: {raw}")
    return path.as_posix()


def _file_index(snapshot: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(item["scope"]), str(item["path"])): dict(item)
        for item in snapshot["files"]
    }


def _entry_digest(entry: Mapping[str, Any] | None) -> str:
    if entry is None:
        return ""
    payload = {
        "content": str(entry["content"]),
        "executable": bool(entry["executable"]),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
