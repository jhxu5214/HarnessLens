from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from harnesslens.core.artifacts import write_json


WORKFLOW_FINGERPRINT_SCHEMA = "harnesslens.workflow-source.v1"
WORKFLOW_FINGERPRINT_NAME = "workflow_fingerprint.json"
_PROGRESS_MARKERS = (
    "controller_state.json",
    "discovery",
    "experience",
    "analyzer",
    "main_agent",
    "rollout",
    "submission",
)


def build_workflow_fingerprint(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    package = root / "harnesslens"
    paths = [
        root / "run_e2e.py",
        *sorted(package.rglob("*.py")),
    ]
    if not paths or any(path.is_symlink() or not path.is_file() for path in paths):
        raise RuntimeError("workflow source tree is incomplete")
    digest = hashlib.sha256()
    files: dict[str, str] = {}
    for path in paths:
        relative = path.relative_to(root).as_posix()
        content = path.read_bytes()
        file_digest = hashlib.sha256(content).hexdigest()
        files[relative] = file_digest
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return {
        "schema": WORKFLOW_FINGERPRINT_SCHEMA,
        "sha256": digest.hexdigest(),
        "files": files,
    }


def establish_workflow_fingerprint(
    *, repo_root: str | Path, run_root: str | Path
) -> Mapping[str, Any]:
    run = Path(run_root).resolve()
    path = run / WORKFLOW_FINGERPRINT_NAME
    current = build_workflow_fingerprint(repo_root)
    if path.is_file():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        _validate_recorded_fingerprint(recorded, current)
        return current
    if any((run / marker).exists() for marker in _PROGRESS_MARKERS):
        raise RuntimeError(
            "existing run has no workflow fingerprint and cannot be safely resumed"
        )
    write_json(path, current)
    return current


def assert_workflow_fingerprint(
    *,
    repo_root: str | Path,
    run_root: str | Path,
    expected_sha256: str,
) -> None:
    current = build_workflow_fingerprint(repo_root)
    if str(current["sha256"]) != str(expected_sha256):
        raise RuntimeError(
            "workflow source changed during the run; start a new run ID"
        )
    path = Path(run_root).resolve() / WORKFLOW_FINGERPRINT_NAME
    if not path.is_file():
        raise RuntimeError("workflow fingerprint disappeared during the run")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    _validate_recorded_fingerprint(recorded, current)


def _validate_recorded_fingerprint(
    recorded: Any, current: Mapping[str, Any]
) -> None:
    if not isinstance(recorded, Mapping):
        raise RuntimeError("workflow fingerprint is malformed")
    if str(recorded.get("schema") or "") != WORKFLOW_FINGERPRINT_SCHEMA:
        raise RuntimeError("workflow fingerprint schema differs from this code")
    if str(recorded.get("sha256") or "") != str(current["sha256"]):
        raise RuntimeError(
            "workflow source differs from the source that created this run"
        )
