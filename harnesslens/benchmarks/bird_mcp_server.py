from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import threading
from pathlib import Path
from typing import Any, Mapping

try:
    from harnesslens.benchmarks.bird_sql import execute_readonly_sql
except ModuleNotFoundError:  # direct script execution by external harnesses
    from bird_sql import execute_readonly_sql


class BirdMCPServer:
    def __init__(
        self,
        *,
        database: str | Path,
        log_file: str | Path | None = None,
        max_steps: int = 30,
        query_timeout_s: float = 5.0,
        tool_desc_patches: Mapping[str, Any] | None = None,
    ) -> None:
        self.database = Path(database).resolve()
        self.log_file = Path(log_file).resolve() if log_file else None
        self.max_steps = int(max_steps)
        self.query_timeout_s = float(query_timeout_s)
        self.call_log: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        patches = dict(tool_desc_patches or {})
        self.tools = [
            _patched_tool(
                {
                    "name": "execute_sql",
                    "description": (
                        "Execute one read-only SQLite SELECT/WITH query against the "
                        "current BIRD database. Use this to validate joins, filters, "
                        "and result shape before returning the final SQL. Results are "
                        "limited to 200 rows."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "sql": {
                                "type": "string",
                                "description": "A single SQLite SELECT or WITH query.",
                            }
                        },
                        "required": ["sql"],
                        "additionalProperties": False,
                    },
                },
                patches.get("execute_sql"),
            )
        ]

    def handle_request(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        if "id" not in request:
            return None
        request_id = request.get("id")
        method = str(request.get("method") or "")
        if method == "initialize":
            result: Any = {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "bird-mini-dev-harnesslens", "version": "1.0.0"},
                "capabilities": {"tools": {}},
            }
        elif method == "tools/list":
            result = {"tools": self.tools}
        elif method == "tools/call":
            params = request.get("params") or {}
            result = self._execute_tool(
                str(params.get("name") or ""), dict(params.get("arguments") or {})
            )
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Unknown method: {method}"},
            }
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if len(self.call_log) >= self.max_steps:
                return _tool_error(
                    f"tool-call limit reached ({self.max_steps}); return final SQL"
                )
        if name != "execute_sql":
            return _tool_error(f"unknown BIRD tool: {name}")
        sql = str(arguments.get("sql") or "")
        try:
            query = execute_readonly_sql(
                self.database,
                sql,
                timeout_s=self.query_timeout_s,
                max_rows=200,
            )
            result_text = json.dumps(
                {
                    "columns": list(query.columns),
                    "rows": [list(row) for row in query.rows],
                    "truncated": query.truncated,
                    "elapsed_s": round(query.elapsed_s, 4),
                },
                ensure_ascii=False,
                default=str,
            )
            is_error = False
        except Exception as exc:  # noqa: BLE001
            result_text = f"{type(exc).__name__}: {exc}"
            is_error = True
        row = {
            "name": name,
            "arguments": {"sql": sql},
            "call_str": f"execute_sql(sql={sql!r})",
            "result": result_text,
            "is_error": is_error,
        }
        with self._lock:
            self.call_log.append(row)
            self._flush_log()
        if is_error:
            return _tool_error(result_text)
        return {"content": [{"type": "text", "text": result_text}]}

    def _flush_log(self) -> None:
        if self.log_file is None:
            return
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file.write_text(
            json.dumps(self.call_log, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _patched_tool(tool: dict[str, Any], patch: Any) -> dict[str, Any]:
    result = json.loads(json.dumps(tool))
    if isinstance(patch, str) and patch.strip():
        result["description"] = f"{result['description']}\n{patch.strip()}"
    elif isinstance(patch, Mapping):
        if str(patch.get("desc") or "").strip():
            result["description"] = (
                f"{result['description']}\n{str(patch['desc']).strip()}"
            )
        properties = result["inputSchema"].get("properties") or {}
        for name, suffix in (patch.get("params") or {}).items():
            if name in properties and str(suffix).strip():
                current = str(properties[name].get("description") or "")
                properties[name]["description"] = f"{current}\n{suffix}".strip()
    return result


def _tool_error(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": f"Error: {message}"}], "isError": True}


def _send(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def _recv(sock: socket.socket) -> bytes:
    raw_length = b""
    while len(raw_length) < 4:
        chunk = sock.recv(4 - len(raw_length))
        if not chunk:
            raise ConnectionError("socket closed")
        raw_length += chunk
    length = struct.unpack(">I", raw_length)[0]
    data = b""
    while len(data) < length:
        chunk = sock.recv(min(65_536, length - len(data)))
        if not chunk:
            raise ConnectionError("socket closed")
        data += chunk
    return data


def _handle_client(conn: socket.socket, server: BirdMCPServer) -> None:
    try:
        while True:
            response = server.handle_request(json.loads(_recv(conn)))
            if response is not None:
                _send(conn, json.dumps(response).encode())
    except (ConnectionError, json.JSONDecodeError):
        pass
    finally:
        conn.close()


def run_server(args: argparse.Namespace) -> None:
    patches = {}
    if args.tool_desc_patches:
        patches = json.loads(Path(args.tool_desc_patches).read_text(encoding="utf-8"))
    server = BirdMCPServer(
        database=args.database,
        log_file=args.log_file,
        max_steps=args.max_steps,
        query_timeout_s=args.query_timeout,
        tool_desc_patches=patches,
    )
    path = Path(args.socket)
    path.unlink(missing_ok=True)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(path))
    sock.listen(16)
    print("READY", flush=True)
    while True:
        conn, _ = sock.accept()
        threading.Thread(target=_handle_client, args=(conn, server), daemon=True).start()


def _read_stdio_message() -> tuple[bytes, bool] | None:
    first = sys.stdin.buffer.readline()
    if not first:
        return None
    if first.lower().startswith(b"content-length:"):
        headers = [first]
        while headers[-1] not in {b"\r\n", b"\n"}:
            line = sys.stdin.buffer.readline()
            if not line:
                return None
            headers.append(line)
        length = next(
            int(line.split(b":", 1)[1].strip())
            for line in headers
            if line.lower().startswith(b"content-length:")
        )
        return sys.stdin.buffer.read(length), True
    buffer = first.decode("utf-8", errors="replace")
    while True:
        try:
            json.loads(buffer.strip())
            return buffer.strip().encode(), False
        except json.JSONDecodeError:
            line = sys.stdin.buffer.readline()
            if not line:
                return None
            buffer += line.decode("utf-8", errors="replace")


def _write_stdio_message(payload: bytes, *, framed: bool) -> None:
    if framed:
        sys.stdout.buffer.write(
            b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n"
        )
        sys.stdout.buffer.write(payload)
    else:
        sys.stdout.buffer.write(payload + b"\n")
    sys.stdout.buffer.flush()


def run_bridge(args: argparse.Namespace) -> None:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(str(args.socket))
    try:
        while True:
            incoming = _read_stdio_message()
            if incoming is None:
                break
            raw, framed = incoming
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            _send(sock, raw)
            if "id" not in parsed:
                continue
            _write_stdio_message(_recv(sock), framed=framed)
    except (BrokenPipeError, ConnectionError):
        pass
    finally:
        sock.close()


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
    subparsers = parser.add_subparsers(dest="mode", required=True)
    server = subparsers.add_parser("server")
    server.add_argument("--database", required=True)
    server.add_argument("--socket", required=True)
    server.add_argument("--log-file")
    server.add_argument("--max-steps", type=int, default=30)
    server.add_argument("--query-timeout", type=float, default=5.0)
    server.add_argument("--tool-desc-patches")
    bridge = subparsers.add_parser("bridge")
    bridge.add_argument("--socket", required=True)
    args = parser.parse_args()
    if args.mode == "server":
        run_server(args)
    else:
        run_bridge(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
