from __future__ import annotations

import argparse
import copy
import http.server
import json
import os
import re
import socketserver
import ssl
import sys
import threading
import time
import urllib.request
import uuid
import socket
import struct
from typing import Any

try:
    from harnesslens.harnesses.workspace_editor_mcp import (
        WorkspaceEditorServer,
        _tools as _workspace_tool_schemas,
    )
except ImportError:
    from workspace_editor_mcp import (  # type: ignore[no-redef]
        WorkspaceEditorServer,
        _tools as _workspace_tool_schemas,
    )


DEFAULT_UPSTREAM = "https://api.deepseek.com/v1"
UPSTREAM_MODEL = "deepseek-v4-flash"
_KEY = ""
_LOG_FILE: str | None = None
_CONTEXT_LOG: str | None = None
_CALL_COUNTER = 0
_COUNTER_LOCK = threading.Lock()
_LOG_LOCK = threading.Lock()
_REASONING_CACHE: dict[str, str] = {}
_REASONING_LOCK = threading.Lock()
_TAU2_SOCKET: str | None = None
_TAU2_TOOLS_CACHE: list[dict[str, Any]] | None = None
_TAU2_TOOL_PATCHES: dict[str, Any] = {}
_TAU2_TOOLS_LOCK = threading.Lock()
_WORKSPACE_EDITOR: WorkspaceEditorServer | None = None
_DISABLED_TOOLS: set[str] = set()
_MAX_INTERNAL_TOOL_ROUNDS = 10
_DSML_TOOL_CALLS_RE = re.compile(
    r"<｜｜DSML｜｜tool_calls>(?P<body>.*?)</｜｜DSML｜｜tool_calls>",
    re.DOTALL,
)
_DSML_INVOKE_RE = re.compile(
    r"<｜｜DSML｜｜invoke\s+name=\"(?P<name>[^\"]+)\">(?P<body>.*?)</｜｜DSML｜｜invoke>",
    re.DOTALL,
)
_DSML_PARAMETER_RE = re.compile(
    r"<｜｜DSML｜｜parameter\s+name=\"(?P<name>[^\"]+)\"(?:\s+string=\"[^\"]*\")?>(?P<value>.*?)</｜｜DSML｜｜parameter>",
    re.DOTALL,
)


def _upstream_base_url() -> str:
    value = str(
        os.environ.get("DEEPSEEK_BASE_URL")
        or os.environ.get("DEEPSEEK_URL")
        or DEFAULT_UPSTREAM
    ).rstrip("/")
    if value.endswith("/chat/completions"):
        value = value[: -len("/chat/completions")]
    return value


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def _write_logs(entry: dict[str, Any], context: dict[str, Any]) -> None:
    with _LOG_LOCK:
        if _LOG_FILE:
            with open(_LOG_FILE, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        if _CONTEXT_LOG:
            with open(_CONTEXT_LOG, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(context, ensure_ascii=False, default=str) + "\n")
    sys.stderr.write(
        "[proxy] call={call} msgs={n_messages} tools={n_tools} "
        "total_in={total_input} hit={cache_read_input_tokens} "
        "out={output_tokens} rate={cache_hit_rate:.1%}\n".format(**entry)
    )
    sys.stderr.flush()


def _sse(wfile: Any, event: str, data: dict[str, Any]) -> None:
    wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
    wfile.flush()


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict)
            and part.get("type") in {"input_text", "text", "output_text"}
        )
    return str(content or "")


def _translate_request(req: dict[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    instructions = _text_content(req.get("instructions"))
    if instructions.strip():
        messages.append({"role": "system", "content": instructions})
    for item in req.get("input") or []:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user")
        if role in {"developer", "system"}:
            text = _text_content(item.get("content", ""))
            if text:
                messages.append({"role": "system", "content": text})
            continue
        if item.get("type") == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "content": str(item.get("output") or ""),
                    "tool_call_id": str(item.get("call_id") or ""),
                }
            )
        elif item.get("type") == "function_call":
            call_id = str(item.get("call_id") or item.get("id") or "")
            if not messages or messages[-1].get("role") != "assistant":
                assistant: dict[str, Any] = {"role": "assistant", "content": None, "tool_calls": []}
                with _REASONING_LOCK:
                    reasoning = _REASONING_CACHE.get(call_id)
                if reasoning:
                    assistant["reasoning_content"] = reasoning
                messages.append(assistant)
            messages[-1].setdefault("tool_calls", []).append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": str(item.get("name") or ""),
                        "arguments": str(item.get("arguments") or "{}"),
                    },
                }
            )
        else:
            text = _text_content(item.get("content", ""))
            if role == "user" and text.lstrip().startswith("<environment_context>"):
                continue
            if text:
                messages.append({"role": role, "content": text})

    tau2_tools = _tau2_chat_tools()
    workspace_tools = _workspace_editor_chat_tools()
    direct_tools = [*tau2_tools, *workspace_tools]
    tools: list[dict[str, Any]] = []
    deferred_tool_search: dict[str, Any] | None = None
    for tool in req.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        if direct_tools:
            continue
        if tool.get("type") == "tool_search":
            deferred_tool_search = {
                "type": "function",
                "function": {
                    "name": "tool_search",
                    "description": str(tool.get("description") or ""),
                    "parameters": dict(tool.get("parameters") or {}),
                },
            }
            continue
        if tool.get("type") == "function":
            name = str(tool.get("name") or "")
            params = dict(tool.get("parameters") or {})
            description = str(tool.get("description") or "")
        elif str(tool.get("type") or "").startswith("mcp"):
            name = str(tool.get("name") or tool.get("server_label") or "")
            params = dict(tool.get("parameters") or tool.get("input_schema") or {})
            description = str(tool.get("description") or "")
        else:
            continue
        if not name or name == "request_user_input" or name in _DISABLED_TOOLS:
            continue
        params.pop("additionalProperties", None)
        params.pop("strict", None)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": params,
                },
            }
        )
    if deferred_tool_search is not None:
        tools.append(deferred_tool_search)
    chat = {"model": UPSTREAM_MODEL, "messages": messages}
    if tools:
        chat["tools"] = tools
    if tau2_tools:
        existing = {tool.get("function", {}).get("name") for tool in chat.get("tools") or []}
        merged = list(chat.get("tools") or [])
        merged.extend(
            tool
            for tool in tau2_tools
            if tool.get("function", {}).get("name") not in existing
        )
        chat["tools"] = merged
    if workspace_tools:
        existing = {tool.get("function", {}).get("name") for tool in chat.get("tools") or []}
        merged = list(chat.get("tools") or [])
        merged.extend(
            tool
            for tool in workspace_tools
            if tool.get("function", {}).get("name") not in existing
        )
        chat["tools"] = merged
    if _TAU2_TOOL_PATCHES and chat.get("tools"):
        chat["tools"] = _patch_tau2_tools(
            list(chat["tools"]), _TAU2_TOOL_PATCHES
        )
    return chat


def _workspace_editor_chat_tools() -> list[dict[str, Any]]:
    if _WORKSPACE_EDITOR is None:
        return []
    return [
        {
            "type": "function",
            "function": {
                "name": str(tool["name"]),
                "description": str(tool.get("description") or ""),
                "parameters": dict(tool.get("inputSchema") or {}),
            },
        }
        for tool in _workspace_tool_schemas()
    ]


def _tau2_chat_tools() -> list[dict[str, Any]]:
    global _TAU2_TOOLS_CACHE
    if not _TAU2_SOCKET:
        return []
    with _TAU2_TOOLS_LOCK:
        if _TAU2_TOOLS_CACHE is not None:
            return list(_TAU2_TOOLS_CACHE)
        response = _tau2_rpc("tools/list", {})
        tools = []
        for tool in (response.get("result") or {}).get("tools") or []:
            if not isinstance(tool, dict):
                continue
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": str(tool.get("name") or ""),
                        "description": str(tool.get("description") or ""),
                        "parameters": dict(tool.get("inputSchema") or {}),
                    },
                }
            )
        _TAU2_TOOLS_CACHE = _patch_tau2_tools(tools, _TAU2_TOOL_PATCHES)
        return list(_TAU2_TOOLS_CACHE)


def _patch_tau2_tools(
    tools: list[dict[str, Any]], patches: dict[str, Any]
) -> list[dict[str, Any]]:
    patched = copy.deepcopy(tools)
    for tool in patched:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            continue
        patch = patches.get(str(function.get("name") or ""))
        if not isinstance(patch, dict):
            continue
        if str(patch.get("desc") or "").strip():
            function["description"] = str(patch["desc"])
        parameters = function.get("parameters")
        properties = parameters.get("properties") if isinstance(parameters, dict) else None
        if not isinstance(properties, dict):
            continue
        for name, description in (patch.get("params") or {}).items():
            parameter = properties.get(str(name))
            if isinstance(parameter, dict) and str(description).strip():
                parameter["description"] = str(description)
    return patched


def _tau2_tool_names() -> set[str]:
    return {
        str(tool.get("function", {}).get("name") or "")
        for tool in _tau2_chat_tools()
    }


def _tau2_rpc(method: str, params: dict[str, Any]) -> dict[str, Any]:
    if not _TAU2_SOCKET:
        raise RuntimeError("tau2 socket is not configured")
    payload = {
        "jsonrpc": "2.0",
        "id": f"proxy_{uuid.uuid4().hex[:8]}",
        "method": method,
        "params": params,
    }
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(_TAU2_SOCKET)
        raw = json.dumps(payload).encode()
        sock.sendall(struct.pack(">I", len(raw)) + raw)
        header = _recv_exact(sock, 4)
        length = struct.unpack(">I", header)[0]
        return json.loads(_recv_exact(sock, length).decode("utf-8", errors="replace"))


def _recv_exact(sock: socket.socket, length: int) -> bytes:
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            raise ConnectionError("closed")
        data += chunk
    return data


def _execute_tau2_tool(name: str, arguments: dict[str, Any]) -> str:
    response = _tau2_rpc(
        "tools/call",
        {"name": name, "arguments": arguments, "_requestor": "assistant"},
    )
    if response.get("error"):
        return "Error: " + str(response["error"])
    result = response.get("result") or {}
    content = result.get("content") or []
    texts = [
        str(item.get("text") or "")
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    return "\n".join(texts) if texts else str(result)


def _raw_tool_summary(req: dict[str, Any]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for tool in req.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        summary.append(
            {
                "type": tool.get("type"),
                "name": tool.get("name"),
                "server_label": tool.get("server_label"),
                "keys": sorted(str(key) for key in tool.keys()),
            }
        )
    return summary


def _response_output_items(message: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if message.get("content"):
        output.append(
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex[:8]}",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": str(message.get("content") or ""),
                        "annotations": [],
                    }
                ],
            }
        )
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        name = str(fn.get("name") or "")
        arguments = str(fn.get("arguments") or "{}")
        if name == "tool_search":
            try:
                parsed_arguments = json.loads(arguments)
                if not isinstance(parsed_arguments, dict):
                    parsed_arguments = {"query": str(parsed_arguments)}
            except Exception:
                parsed_arguments = {"query": arguments}
            output.append(
                {
                    "type": "tool_search_call",
                    "id": f"tsc_{uuid.uuid4().hex}",
                    "call_id": str(call.get("id") or ""),
                    "status": "completed",
                    "execution": "client",
                    "arguments": parsed_arguments,
                }
            )
            continue
        output.append(
            {
                "type": "function_call",
                "id": str(call.get("id") or ""),
                "call_id": str(call.get("id") or ""),
                "name": name,
                "arguments": arguments,
                "status": "completed",
            }
        )
    return output


def _coerce_dsml_tool_calls(message: dict[str, Any]) -> dict[str, Any]:
    if message.get("tool_calls"):
        return message
    content = message.get("content")
    if not isinstance(content, str) or "<｜｜DSML｜｜tool_calls>" not in content:
        return message
    tool_calls: list[dict[str, Any]] = []
    for block in _DSML_TOOL_CALLS_RE.finditer(content):
        body = block.group("body")
        for invoke in _DSML_INVOKE_RE.finditer(body):
            name = invoke.group("name").strip()
            if not name:
                continue
            arguments: dict[str, Any] = {}
            for param in _DSML_PARAMETER_RE.finditer(invoke.group("body")):
                raw_value = _unescape_dsml_text(param.group("value").strip())
                param_name = param.group("name").strip()
                arguments[param_name] = _decode_dsml_argument(param_name, raw_value)
            tool_calls.append(
                {
                    "id": f"call_dsml_{uuid.uuid4().hex[:12]}",
                    "_dsml": True,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            )
    if not tool_calls:
        return message
    cleaned = _DSML_TOOL_CALLS_RE.sub("", content).strip()
    updated = dict(message)
    updated["content"] = cleaned or None
    updated["tool_calls"] = tool_calls
    return updated


def _normalize_tau2_tool_calls(
    message: dict[str, Any], tau2_names: set[str]
) -> dict[str, Any]:
    if "call_discoverable_agent_tool" not in tau2_names:
        return message
    updated_calls: list[dict[str, Any]] = []
    changed = False
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        name = str(fn.get("name") or "")
        if name in tau2_names:
            updated_calls.append(call)
            continue
        try:
            direct_args = json.loads(str(fn.get("arguments") or "{}"))
            if not isinstance(direct_args, dict):
                direct_args = {}
        except Exception:
            direct_args = {}
        rewritten = dict(call)
        rewritten["function"] = {
            "name": "call_discoverable_agent_tool",
            "arguments": json.dumps(
                {
                    "agent_tool_name": name,
                    "arguments": json.dumps(direct_args, ensure_ascii=False),
                },
                ensure_ascii=False,
            ),
        }
        updated_calls.append(rewritten)
        changed = True
    if not changed:
        return message
    updated = dict(message)
    updated["tool_calls"] = updated_calls
    return updated


def _unescape_dsml_text(value: str) -> str:
    return (
        value.replace("&quot;", '"')
        .replace("&apos;", "'")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
    )


def _decode_dsml_argument(name: str, value: str) -> Any:
    if not value:
        return ""
    if name == "arguments":
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        if "/responses" not in self.path:
            self._forward()
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        req = json.loads(body.decode("utf-8", errors="replace"))
        raw_tool_summary = _raw_tool_summary(req)
        chat = _translate_request(req)
        try:
            completion, upstream_status, message = _complete_with_internal_tau2_tools(
                chat, raw_tool_summary
            )
        except Exception as exc:  # noqa: BLE001
            payload = json.dumps({"error": str(exc)}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        message = _normalize_tau2_tool_calls(
            _coerce_dsml_tool_calls(message), _tau2_tool_names()
        )

        reasoning = str(message.get("reasoning_content") or "")
        if reasoning and message.get("tool_calls"):
            with _REASONING_LOCK:
                for call in message.get("tool_calls") or []:
                    _REASONING_CACHE[str(call.get("id") or "")] = reasoning

        usage = dict(completion.get("usage") or {})
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        output = _response_output_items(message)
        response_obj = {
            "id": f"resp_{uuid.uuid4().hex[:12]}",
            "object": "response",
            "model": completion.get("model", ""),
            "output": output,
            "status": "completed",
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
            },
        }
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        _sse(self.wfile, "response.created", {"type": "response.created", "response": {**response_obj, "status": "in_progress", "output": []}})
        for index, item in enumerate(output):
            _sse(self.wfile, "response.output_item.added", {"type": "response.output_item.added", "output_index": index, "item": {**item, "status": "in_progress"}})
            if item["type"] == "message":
                for content_index, part in enumerate(item.get("content") or []):
                    _sse(self.wfile, "response.content_part.added", {"type": "response.content_part.added", "output_index": index, "content_index": content_index, "part": part})
                    _sse(self.wfile, "response.output_text.done", {"type": "response.output_text.done", "output_index": index, "content_index": content_index, "text": part.get("text", "")})
                    _sse(self.wfile, "response.content_part.done", {"type": "response.content_part.done", "output_index": index, "content_index": content_index, "part": part})
            else:
                if item["type"] == "function_call":
                    _sse(self.wfile, "response.function_call_arguments.delta", {"type": "response.function_call_arguments.delta", "output_index": index, "item_id": item["id"], "call_id": item["call_id"], "delta": item["arguments"]})
                    _sse(self.wfile, "response.function_call_arguments.done", {"type": "response.function_call_arguments.done", "output_index": index, "item_id": item["id"], "call_id": item["call_id"], "name": item["name"], "arguments": item["arguments"]})
            _sse(self.wfile, "response.output_item.done", {"type": "response.output_item.done", "output_index": index, "item": item})
        _sse(self.wfile, "response.completed", {"type": "response.completed", "response": response_obj})

    def _forward(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
        request = urllib.request.Request(
            f"{_upstream_base_url()}{self.path}",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {_KEY}"},
            method="POST",
        )
        try:
            with _direct_open(request, timeout=180) as response:
                payload = response.read()
                self.send_response(int(response.status))
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except Exception as exc:  # noqa: BLE001
            payload = json.dumps({"error": str(exc)}).encode()
            self.send_response(502)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def log_message(self, *_args: Any) -> None:
        return


def _complete_with_internal_tau2_tools(
    chat: dict[str, Any],
    raw_tool_summary: list[dict[str, Any]],
) -> tuple[dict[str, Any], int, dict[str, Any]]:
    tau2_names = _tau2_tool_names()
    workspace_names = {
        tool["function"]["name"] for tool in _workspace_editor_chat_tools()
    }
    direct_names = tau2_names | workspace_names
    completion: dict[str, Any] = {}
    upstream_status = 0
    for _round in range(_MAX_INTERNAL_TOOL_ROUNDS):
        completion, upstream_status = _call_upstream(chat)
        message = _normalize_tau2_tool_calls(
            _coerce_dsml_tool_calls(
                completion.get("choices", [{}])[0].get("message", {})
            ),
            tau2_names,
        )
        _log_completion(
            completion,
            upstream_status,
            chat,
            raw_tool_summary,
            normalized_message=message,
        )
        tool_calls = message.get("tool_calls") or []
        if not tool_calls or not direct_names:
            return completion, upstream_status, message
        if any(
            str((call.get("function") or {}).get("name") or "")
            not in direct_names
            for call in tool_calls
        ):
            return completion, upstream_status, message
        chat.setdefault("messages", []).append(
            {
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": [
                    {
                        "id": str(call.get("id") or ""),
                        "type": "function",
                        "function": {
                            "name": str((call.get("function") or {}).get("name") or ""),
                            "arguments": str((call.get("function") or {}).get("arguments") or "{}"),
                        },
                    }
                    for call in tool_calls
                ],
            }
        )
        for call in tool_calls:
            fn = call.get("function") or {}
            name = str(fn.get("name") or "")
            try:
                arguments = json.loads(str(fn.get("arguments") or "{}"))
                if not isinstance(arguments, dict):
                    arguments = {}
            except Exception:
                arguments = {}
            chat["messages"].append(
                {
                    "role": "tool",
                    "tool_call_id": str(call.get("id") or ""),
                    "content": _execute_direct_tool(name, arguments),
                }
            )

    final_chat = dict(chat)
    final_chat["messages"] = list(chat.get("messages") or []) + [
        {
            "role": "system",
            "content": (
                "Use the retrieved tool results above to answer the user's latest "
                "request now. Do not call more tools."
            ),
        }
    ]
    final_chat.pop("tools", None)
    completion, upstream_status = _call_upstream(final_chat)
    message = _normalize_tau2_tool_calls(
        _coerce_dsml_tool_calls(
            completion.get("choices", [{}])[0].get("message", {})
        ),
        _tau2_tool_names(),
    )
    _log_completion(
        completion,
        upstream_status,
        final_chat,
        raw_tool_summary,
        normalized_message=message,
    )
    return completion, upstream_status, message


def _execute_direct_tool(name: str, arguments: dict[str, Any]) -> str:
    if _WORKSPACE_EDITOR is not None and name in {
        tool["name"] for tool in _workspace_tool_schemas()
    }:
        result = _WORKSPACE_EDITOR.call(name, arguments)
        return "\n".join(
            str(item.get("text") or "")
            for item in result.get("content") or []
            if isinstance(item, dict)
        )
    return _execute_tau2_tool(name, arguments)


def _call_upstream(chat: dict[str, Any]) -> tuple[dict[str, Any], int]:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {_KEY}"}
    request = urllib.request.Request(
        f"{_upstream_base_url()}/chat/completions",
        data=json.dumps(chat).encode(),
        headers=headers,
        method="POST",
    )
    with _direct_open(request, timeout=180) as response:
        return json.loads(response.read()), int(response.status)


def _direct_open(request: urllib.request.Request, *, timeout: int):
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )
    return opener.open(request, timeout=timeout)


def _log_completion(
    completion: dict[str, Any],
    upstream_status: int,
    chat: dict[str, Any],
    raw_tool_summary: list[dict[str, Any]],
    normalized_message: dict[str, Any] | None = None,
) -> None:
    usage = dict(completion.get("usage") or {})
    with _COUNTER_LOCK:
        global _CALL_COUNTER
        _CALL_COUNTER += 1
        call_num = _CALL_COUNTER
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    cache_read = int(usage.get("prompt_cache_hit_tokens") or 0)
    cache_create = int(usage.get("prompt_cache_miss_tokens") or 0)
    entry = {
        "timestamp": time.time(),
        "call": call_num,
        "model": completion.get("model") or chat.get("model"),
        "input_tokens": max(0, prompt_tokens - cache_read - cache_create),
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_create,
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "total_input": prompt_tokens,
        "cache_hit_rate": cache_read / prompt_tokens if prompt_tokens else 0.0,
        "n_messages": len(chat.get("messages") or []),
        "n_tools": len(chat.get("tools") or []),
        "raw_n_tools": len(raw_tool_summary),
        "upstream_status": upstream_status,
        "stream": True,
    }
    _write_logs(
        entry,
        {
            "timestamp": entry["timestamp"],
            "call": call_num,
            "harness": "codex",
            "request": chat,
            "raw_tool_summary": raw_tool_summary,
            "response": completion,
            "normalized_message": normalized_message,
            "usage": usage,
            "upstream_status": upstream_status,
            "stream": True,
        },
    )



def _resolve_api_key(explicit: str | None) -> str:
    """Take the provider key from the environment rather than from argv.

    A key on the command line is readable by any user on the host via `ps`.
    ``--key`` is still honoured so an externally built command keeps working,
    but every caller in this repository now relies on the inherited env.
    """
    key = str(explicit or os.environ.get("DEEPSEEK_API_KEY") or "").strip()
    if not key:
        raise SystemExit(
            "provider key is unavailable: set DEEPSEEK_API_KEY in the environment"
        )
    return key


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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--key")  # deprecated: prefer DEEPSEEK_API_KEY
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--context-log")
    parser.add_argument("--tau2-socket")
    parser.add_argument("--workspace-editor-root")
    parser.add_argument("--tool-desc-patches")
    parser.add_argument("--disable-tool", action="append", default=[])
    args = parser.parse_args()
    global _KEY, _LOG_FILE, _CONTEXT_LOG, _TAU2_SOCKET, _TAU2_TOOL_PATCHES, _WORKSPACE_EDITOR
    global _DISABLED_TOOLS
    _KEY = _resolve_api_key(args.key)
    _LOG_FILE = args.log_file
    _CONTEXT_LOG = args.context_log
    _TAU2_SOCKET = args.tau2_socket
    _DISABLED_TOOLS = {
        str(name).strip() for name in args.disable_tool if str(name).strip()
    }
    _WORKSPACE_EDITOR = (
        WorkspaceEditorServer(args.workspace_editor_root)
        if args.workspace_editor_root
        else None
    )
    if args.tool_desc_patches:
        with open(args.tool_desc_patches, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("tool description patches must be an object")
        _TAU2_TOOL_PATCHES = payload
    server = ThreadingHTTPServer((str(args.host), int(args.port)), Handler)
    print(f"PORT={server.server_address[1]}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
