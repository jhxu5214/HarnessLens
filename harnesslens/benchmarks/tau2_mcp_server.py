from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import struct
import sys
import threading
from pathlib import Path
from typing import Any

from harnesslens.harnesses.tool_schema import patch_mcp_tool_schemas


REPO_ROOT = Path(
    os.environ.get("HAI_REPO_ROOT")
    or os.environ.get("HARNESSLENS_ROOT")
    or Path(__file__).resolve().parents[2]
).resolve()
TAU2_SRC = REPO_ROOT / "third_party" / "tau3-bench" / "src"
TAU2_DATA = REPO_ROOT / "third_party" / "tau3-bench" / "data"
MAX_TOOL_RESULT = int(os.environ.get("TAU2_MAX_TOOL_RESULT", "0") or 0)
MCP_DEBUG_LOG = os.environ.get("HAI_TAU2_MCP_DEBUG_LOG")

logging.disable(logging.CRITICAL)
os.environ["LOGURU_LEVEL"] = "ERROR"
sys.path.insert(0, str(TAU2_SRC))
os.environ.setdefault("TAU2_DATA_DIR", str(TAU2_DATA))

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stderr, level="ERROR")


class Tau2MCPServer:
    def __init__(
        self,
        *,
        domain: str,
        task_id: str,
        log_file: str | None = None,
        max_steps: int | None = None,
        retrieval_config: str | None = None,
        tool_desc_patches: dict[str, Any] | None = None,
    ) -> None:
        self.domain = str(domain)
        self.task_id = str(task_id)
        self.log_file = log_file
        self.max_steps = max_steps
        self.step_window_start = 0
        self.call_log: list[dict[str, Any]] = []
        from tau2.runner import load_tasks
        from tau2.runner.build import build_environment

        tasks = []
        load_errors: dict[str, str] = {}
        for split_name in ("test", "train", "base"):
            try:
                loaded = load_tasks(self.domain, task_split_name=split_name)
            except Exception as exc:  # noqa: BLE001
                load_errors[split_name] = f"{type(exc).__name__}: {exc}"
                continue
            tasks.extend(loaded)
            if any(str(task.id) == self.task_id for task in loaded):
                break
        task_obj = next((task for task in tasks if str(task.id) == self.task_id), None)
        if task_obj is None:
            raise ValueError(
                f"Task {self.task_id} not found in {self.domain} "
                f"(load_errors={load_errors})"
            )
        env_kwargs: dict[str, Any] = {}
        if retrieval_config:
            env_kwargs["retrieval_variant"] = str(retrieval_config)
            env_kwargs["task"] = task_obj
        self.env = build_environment(self.domain, env_kwargs=env_kwargs)
        self.task_obj = task_obj

        initial_state = getattr(task_obj, "initial_state", None)
        if initial_state:
            self.env.set_state(
                initialization_data=getattr(initial_state, "initialization_data", None),
                initialization_actions=getattr(initial_state, "initialization_actions", None),
                message_history=[],
            )

        self.tools: list[dict[str, Any]] = []
        self.tool_map: dict[str, Any] = {}
        self.user_tool_map: dict[str, Any] = {}
        try:
            if hasattr(self.env, "get_user_tools"):
                for tool in self.env.get_user_tools(include=getattr(task_obj, "user_tools", None)):
                    fn = tool.openai_schema.get("function", {})
                    self.user_tool_map[str(fn["name"])] = tool
        except (AttributeError, ValueError):
            pass
        for tool in self.env.get_tools():
            schema = tool.openai_schema
            fn = schema.get("function", {})
            name = str(fn["name"])
            input_schema = fn.get("parameters") or {"type": "object", "properties": {}}
            if isinstance(input_schema, dict) and "$defs" in input_schema:
                input_schema = _inline_defs(dict(input_schema))
            self.tools.append(
                {
                    "name": name,
                    "description": str(fn.get("description") or ""),
                    "inputSchema": input_schema,
                }
            )
            self.tool_map[name] = tool
        patch_mcp_tool_schemas(self.tools, tool_desc_patches or {})

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        if "id" not in request:
            return None
        method = str(request.get("method") or "")
        req_id = request.get("id")
        _mcp_debug({"event": "request", "method": method, "id": req_id})
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": f"tau2-{self.domain}", "version": "1.0.0"},
                "capabilities": {"tools": {}},
            }
        elif method == "tools/list":
            result = {"tools": self.tools}
            _mcp_debug({"event": "tools_list", "count": len(self.tools)})
        elif method == "tools/call":
            params = request.get("params") or {}
            result = self._execute_tool(
                str(params.get("name") or ""),
                dict(params.get("arguments") or {}),
                requestor=str(params.get("_requestor") or "assistant"),
            )
        elif method == "harness/reset_step_window":
            self.step_window_start = sum(
                1
                for call in self.call_log
                if call.get("requestor", "assistant") == "assistant"
            )
            result = {"ok": True, "stepWindowStart": self.step_window_start}
        elif method == "notifications/initialized":
            return None
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown method: {method}"},
            }
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def _execute_tool(
        self, name: str, arguments: dict[str, Any], *, requestor: str
    ) -> dict[str, Any]:
        from tau2.data_model.message import ToolCall

        try:
            if requestor == "user" and name not in self.user_tool_map:
                return _tool_error(f"user tool '{name}' not found")
            if requestor == "assistant" and name not in self.tool_map:
                return _tool_error(f"agent tool '{name}' not found")
            if requestor == "assistant" and self.max_steps is not None:
                total_used = sum(
                    1
                    for call in self.call_log
                    if call.get("requestor", "assistant") == "assistant"
                )
                used = max(0, total_used - int(self.step_window_start))
                if used >= int(self.max_steps):
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"[STEP LIMIT REACHED] You have used your "
                                    f"per-turn tool-call budget ({self.max_steps}). Give "
                                    "your final response to the user now."
                                ),
                            }
                        ]
                    }
            tool = self.user_tool_map[name] if requestor == "user" else self.tool_map[name]
            response = self.env.get_response(
                ToolCall(
                    id=f"call_{len(self.call_log)}",
                    name=name,
                    arguments=arguments,
                    requestor=requestor,
                )
            )
            result = response.content if hasattr(response, "content") else str(response)
            if MAX_TOOL_RESULT and len(str(result)) > MAX_TOOL_RESULT:
                result = (
                    str(result)[:MAX_TOOL_RESULT]
                    + f"\n...[truncated {len(str(result)) - MAX_TOOL_RESULT} chars]"
                )
            self.call_log.append(
                {
                    "name": name,
                    "arguments": arguments,
                    "call_str": _call_str(name, arguments),
                    "result": result,
                    "requestor": requestor,
                }
            )
            self._flush_log()
            return {"content": [{"type": "text", "text": str(result)}]}
        except Exception as exc:  # noqa: BLE001
            message = f"Error calling {name}: {exc}"
            self.call_log.append(
                {
                    "name": name,
                    "arguments": arguments,
                    "call_str": _call_str(name, arguments),
                    "result": message,
                    "requestor": requestor,
                }
            )
            self._flush_log()
            return _tool_error(message)

    def _flush_log(self) -> None:
        if self.log_file:
            Path(self.log_file).write_text(
                json.dumps(self.call_log, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )


def _mcp_debug(payload: dict[str, Any]) -> None:
    if not MCP_DEBUG_LOG:
        return
    try:
        with Path(MCP_DEBUG_LOG).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def _tool_error(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": f"Error: {message}"}], "isError": True}


def _call_str(name: str, arguments: dict[str, Any]) -> str:
    return f"{name}({', '.join(f'{key}={value!r}' for key, value in arguments.items())})"


def _inline_defs(schema: dict[str, Any]) -> dict[str, Any]:
    defs = dict(schema.pop("$defs", {}) or {})

    def resolve(value: Any) -> Any:
        if isinstance(value, dict):
            ref = value.get("$ref")
            if isinstance(ref, str):
                key = ref.split("/")[-1]
                if key in defs:
                    return resolve(defs[key])
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        return value

    return resolve(schema)


def _send(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def _recv(sock: socket.socket) -> bytes:
    raw_len = b""
    while len(raw_len) < 4:
        chunk = sock.recv(4 - len(raw_len))
        if not chunk:
            raise ConnectionError("closed")
        raw_len += chunk
    length = struct.unpack(">I", raw_len)[0]
    data = b""
    while len(data) < length:
        chunk = sock.recv(min(65536, length - len(data)))
        if not chunk:
            raise ConnectionError("closed")
        data += chunk
    return data


def run_server(args: argparse.Namespace) -> None:
    patches: dict[str, Any] = {}
    if args.tool_desc_patches:
        payload = json.loads(
            Path(args.tool_desc_patches).read_text(encoding="utf-8")
        )
        if not isinstance(payload, dict):
            raise ValueError("Tau2 tool description patches must be an object")
        patches = payload
    server = Tau2MCPServer(
        domain=args.domain,
        task_id=args.task_id,
        log_file=args.log_file,
        max_steps=args.max_steps,
        retrieval_config=args.retrieval_config,
        tool_desc_patches=patches,
    )
    path = str(args.socket)
    Path(path).unlink(missing_ok=True)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(path)
    sock.listen(16)
    print("READY", flush=True)
    while True:
        conn, _ = sock.accept()
        threading.Thread(target=_handle_client, args=(conn, server), daemon=True).start()


def _handle_client(conn: socket.socket, server: Tau2MCPServer) -> None:
    try:
        while True:
            response = server.handle_request(json.loads(_recv(conn)))
            if response is not None:
                _send(conn, json.dumps(response).encode())
    except (ConnectionError, json.JSONDecodeError):
        pass
    finally:
        conn.close()


def _read_stdio_message() -> tuple[bytes, bool] | None:
    first = sys.stdin.buffer.readline()
    if not first:
        return None
    if first.lower().startswith(b"content-length:"):
        headers = [first]
        while True:
            line = sys.stdin.buffer.readline()
            if not line:
                return None
            headers.append(line)
            if line in {b"\r\n", b"\n"}:
                break
        length = 0
        for header in headers:
            if header.lower().startswith(b"content-length:"):
                length = int(header.split(b":", 1)[1].strip())
                break
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
        sys.stdout.buffer.write(b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n")
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
            raw_bytes, framed = incoming
            try:
                parsed = json.loads(raw_bytes.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            _send(sock, raw_bytes)
            if "id" not in parsed:
                continue
            want = parsed.get("id")
            response = _recv(sock)
            for _ in range(8):
                try:
                    if json.loads(response).get("id") == want:
                        break
                except (json.JSONDecodeError, AttributeError):
                    pass
                response = _recv(sock)
            _write_stdio_message(response, framed=framed)
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
    sub = parser.add_subparsers(dest="mode", required=True)
    server = sub.add_parser("server")
    server.add_argument("--domain", required=True)
    server.add_argument("--task-id", required=True)
    server.add_argument("--socket", required=True)
    server.add_argument("--log-file")
    server.add_argument("--max-steps", type=int)
    server.add_argument("--retrieval-config")
    server.add_argument("--tool-desc-patches")
    bridge = sub.add_parser("bridge")
    bridge.add_argument("--socket", required=True)
    args = parser.parse_args()
    if args.mode == "server":
        run_server(args)
    else:
        run_bridge(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
