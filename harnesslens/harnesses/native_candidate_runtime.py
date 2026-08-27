from __future__ import annotations

import copy
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from harnesslens.harnesses.harness_manifest import (
    empty_harness_manifest,
    normalize_harness,
    normalize_native_manifest,
)
from harnesslens.harnesses.harness_workspace import (
    empty_workspace_snapshot,
    materialize_workspace,
    normalize_workspace_snapshot,
)


BASE_PROJECT_INSTRUCTION = "Use only the registered task tools."
CODEX_HOOK_CONTEXT_PATH = ".codex/harness-hook-context.md"
CANDIDATE_WORKSPACE_KEY = "_workspace"


def native_manifest(harness: str, raw: Mapping[str, Any] | None) -> dict[str, Any]:
    normalize_harness(harness)
    source = dict(raw or empty_harness_manifest())
    workspace = normalize_workspace_snapshot(source.pop(CANDIDATE_WORKSPACE_KEY, None))
    manifest = normalize_native_manifest(source)
    manifest[CANDIDATE_WORKSPACE_KEY] = workspace
    return manifest


def attach_candidate_workspace(
    manifest: Mapping[str, Any], workspace: Mapping[str, Any] | None
) -> dict[str, Any]:
    result = dict(manifest)
    result[CANDIDATE_WORKSPACE_KEY] = normalize_workspace_snapshot(workspace)
    return result


def candidate_workspace(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return normalize_workspace_snapshot(
        manifest.get(CANDIDATE_WORKSPACE_KEY, empty_workspace_snapshot())
    )


def materialize_project_files(
    workspace: str | Path,
    manifest: Mapping[str, Any],
    *,
    base_instruction: str = BASE_PROJECT_INSTRUCTION,
    home_root: str | Path | None = None,
) -> None:
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=True)
    if home_root is not None:
        materialize_workspace(
            candidate_workspace(manifest),
            home_root=home_root,
            project_root=root,
        )
    for item in manifest.get("files") or []:
        target = root / str(item["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(item["content"]), encoding="utf-8")
    agents = root / "AGENTS.md"
    candidate = agents.read_text(encoding="utf-8") if agents.is_file() else ""
    content = "\n\n".join(
        item.strip() for item in (base_instruction, candidate) if item.strip()
    )
    agents.write_text(content + "\n", encoding="utf-8")


def install_codex_hook_dispatcher(
    workspace: str | Path, manifest: Mapping[str, Any]
) -> bool:
    root = Path(workspace)
    if not codex_hook_declared(manifest):
        return False
    git = subprocess.run(
        ["git", "init", "--quiet", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    if git.returncode != 0:
        raise RuntimeError(
            f"failed to create isolated Codex project root: {git.stderr}"
        )
    context = root / CODEX_HOOK_CONTEXT_PATH
    if not context.is_file() or not context.read_text(encoding="utf-8").strip():
        return False
    dispatcher = Path(__file__).with_name("codex_session_hook.py").resolve()
    command = " ".join((shlex.quote(sys.executable), shlex.quote(str(dispatcher))))
    hooks_path = root / ".codex" / "hooks.json"
    hooks_path.parent.mkdir(parents=True, exist_ok=True)
    hooks_path.write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": command,
                                    "async": False,
                                }
                            ],
                        }
                    ]
                }
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return True


def prepare_codex_project_hooks(
    workspace: str | Path, manifest: Mapping[str, Any]
) -> bool:
    root = Path(workspace)
    install_codex_hook_dispatcher(root, manifest)
    hooks_path = root / ".codex" / "hooks.json"
    if not hooks_path.is_file():
        return False
    git = subprocess.run(
        ["git", "init", "--quiet", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    if git.returncode != 0:
        raise RuntimeError(
            f"failed to create isolated Codex project root: {git.stderr}"
        )
    return True


def codex_project_trust_config(workspace: str | Path) -> dict[str, Any]:
    return {
        "projects": {
            str(Path(workspace).resolve()): {"trust_level": "trusted"},
        }
    }


def codex_hook_declared(manifest: Mapping[str, Any]) -> bool:
    return any(
        isinstance(item, Mapping)
        and str(item.get("path") or "") == CODEX_HOOK_CONTEXT_PATH
        for item in manifest.get("files") or []
    )


def candidate_system_prompt(base: str, manifest: Mapping[str, Any]) -> str:
    workspace_appends = [
        str(item.get("content") or "")
        for item in candidate_workspace(manifest).get("files") or []
        if isinstance(item, Mapping)
        and str(item.get("scope") or "") == "project"
        and str(item.get("path") or "") == ".pi/APPEND_SYSTEM.md"
    ]
    workspace_skills = []
    for item in candidate_workspace(manifest).get("files") or []:
        if not isinstance(item, Mapping):
            continue
        path = str(item.get("path") or "")
        if not re.fullmatch(r"\.pi/skills/[^/]+/SKILL\.md", path):
            continue
        content = str(item.get("content") or "")
        name_match = re.search(r"(?m)^name:\s*['\"]?([^'\"\n]+)", content)
        description_match = re.search(r"(?m)^description:\s*['\"]?([^'\"\n]+)", content)
        name = name_match.group(1).strip() if name_match else ""
        description = description_match.group(1).strip() if description_match else ""
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or not description:
            continue
        workspace_skills.append(
            f'<harness_skill name="{name}" description="{description}">\n'
            f"{content.strip()}\n"
            "</harness_skill>"
        )
    return "\n\n".join(
        item.strip()
        for item in (
            str(base),
            *(str(value) for value in manifest.get("prompt_appends") or []),
            *workspace_appends,
            *workspace_skills,
        )
        if item.strip()
    )


def apply_flat_config(
    base: Mapping[str, Any], patch: Mapping[str, Any]
) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for raw_key, value in patch.items():
        parts = str(raw_key).split(".")
        if not all(parts):
            raise ValueError(f"invalid empty config path: {raw_key}")
        target = result
        for part in parts[:-1]:
            child = target.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"config path conflicts with scalar: {raw_key}")
            target = child
        target[parts[-1]] = copy.deepcopy(value)
    return result


def render_toml(config: Mapping[str, Any]) -> str:
    lines: list[str] = []

    def emit_table(path: tuple[str, ...], values: Mapping[str, Any]) -> None:
        scalars = [
            (str(key), value)
            for key, value in values.items()
            if not isinstance(value, Mapping)
        ]
        children = [
            (str(key), value)
            for key, value in values.items()
            if isinstance(value, Mapping)
        ]
        if path:
            if lines and lines[-1] != "":
                lines.append("")
            lines.append("[" + ".".join(_toml_key(item) for item in path) + "]")
        for key, value in scalars:
            lines.append(f"{_toml_key(key)} = {_toml_value(value)}")
        for key, value in children:
            emit_table((*path, key), value)

    emit_table((), config)
    return "\n".join(lines).rstrip() + "\n"


def _toml_key(value: str) -> str:
    return value if re.fullmatch(r"[A-Za-z0-9_-]+", value) else json.dumps(value)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise ValueError(f"unsupported TOML candidate value: {type(value).__name__}")
