from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_LIST_ENTRIES = 2000


class WorkspaceEditorServer:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir() or self.root.is_symlink():
            raise ValueError("workspace editor root must be a real directory")

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        if "id" not in request:
            return None
        req_id = request["id"]
        method = str(request.get("method") or "")
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "bounded-workspace-editor", "version": "1"},
                "capabilities": {"tools": {}},
            }
        elif method == "tools/list":
            result = {"tools": _tools()}
        elif method == "tools/call":
            params = request.get("params") or {}
            result = self.call(
                str(params.get("name") or ""),
                dict(params.get("arguments") or {}),
            )
        elif method == "notifications/initialized":
            return None
        else:
            return _error(req_id, -32601, f"unknown method: {method}")
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def call(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        try:
            if name == "list_files":
                relative = str(arguments.get("path") or ".")
                target = self._path(relative)
                if not target.is_dir():
                    raise ValueError(f"not a directory: {relative}")
                entries = []
                for path in sorted(target.rglob("*")):
                    if path.is_symlink():
                        raise ValueError("workspace contains a symlink")
                    entries.append(
                        path.relative_to(self.root).as_posix()
                        + ("/" if path.is_dir() else "")
                    )
                    if len(entries) >= MAX_LIST_ENTRIES:
                        entries.append("...[truncated]")
                        break
                return _text("\n".join(entries) or "(empty)")
            if name == "read_file":
                relative = str(arguments.get("path") or "")
                target = self._path(relative)
                if not target.is_file() or target.is_symlink():
                    raise ValueError(f"not a regular file: {relative}")
                if target.stat().st_size > MAX_FILE_BYTES:
                    raise ValueError("file exceeds editor size limit")
                return _text(target.read_text(encoding="utf-8"))
            if name == "write_file":
                relative = str(arguments.get("path") or "")
                content = arguments.get("content")
                if not isinstance(content, str):
                    raise ValueError("content must be text")
                if len(content.encode("utf-8")) > MAX_FILE_BYTES:
                    raise ValueError("content exceeds editor size limit")
                target = self._path(relative)
                if target.exists() and (target.is_symlink() or not target.is_file()):
                    raise ValueError("target must be a regular file")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                target.chmod(0o644)
                return _text(f"wrote {target.relative_to(self.root).as_posix()}")
            raise ValueError(f"unknown workspace editor tool: {name}")
        except (OSError, UnicodeError, ValueError) as exc:
            return {
                "content": [{"type": "text", "text": f"Error: {exc}"}],
                "isError": True,
            }

    def _path(self, raw: str) -> Path:
        if not raw or "\\" in raw:
            raise ValueError("path must be a nonempty POSIX-style relative path")
        candidate = (self.root / raw).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("path escapes the candidate workspace") from exc
        current = candidate
        while current != self.root:
            if current.exists() and current.is_symlink():
                raise ValueError("symlink paths are not allowed")
            current = current.parent
        return candidate


def _tools() -> list[dict[str, Any]]:
    path = {"type": "string", "description": "Path relative to the candidate root."}
    return [
        {
            "name": "list_files",
            "description": "List files below a candidate directory.",
            "inputSchema": {
                "type": "object",
                "properties": {"path": path},
                "additionalProperties": False,
            },
        },
        {
            "name": "read_file",
            "description": "Read one UTF-8 candidate file.",
            "inputSchema": {
                "type": "object",
                "properties": {"path": path},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
        {
            "name": "write_file",
            "description": "Create or replace one UTF-8 candidate file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": path,
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    ]


def _text(value: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": str(value)}]}


def _error(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": int(code), "message": str(message)},
    }


def _read_message() -> tuple[bytes, bool] | None:
    first = sys.stdin.buffer.readline()
    if not first:
        return None
    if first.lower().startswith(b"content-length:"):
        length = int(first.split(b":", 1)[1].strip())
        while True:
            line = sys.stdin.buffer.readline()
            if not line or line in {b"\r\n", b"\n"}:
                break
        return sys.stdin.buffer.read(length), True
    return first.strip(), False


def _write_message(payload: bytes, *, framed: bool) -> None:
    if framed:
        sys.stdout.buffer.write(
            b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n"
        )
    sys.stdout.buffer.write(payload + (b"" if framed else b"\n"))
    sys.stdout.buffer.flush()


def main() -> int:
    # Exit if the orchestrator is SIGKILLed; see harnesslens/infrastructure/child_lifetime.py.
    try:
        from harnesslens.infrastructure.child_lifetime import exit_when_orphaned
    except ImportError:  # pragma: no cover - standalone/in-container execution
        try:
            from child_lifetime import exit_when_orphaned  # type: ignore
        except ImportError:
            exit_when_orphaned = None  # type: ignore[assignment]
    if exit_when_orphaned is not None:
        exit_when_orphaned()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    server = WorkspaceEditorServer(args.root)
    while True:
        incoming = _read_message()
        if incoming is None:
            return 0
        body, framed = incoming
        if not body:
            continue
        try:
            request = json.loads(body)
            response = server.handle(request)
        except Exception as exc:  # noqa: BLE001
            response = _error(None, -32603, str(exc))
        if response is not None:
            _write_message(
                json.dumps(response, ensure_ascii=False).encode("utf-8"),
                framed=framed,
            )


if __name__ == "__main__":
    os.umask(0o077)
    raise SystemExit(main())
