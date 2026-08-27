"""HarnessLens pass-through logging proxy for Chat Completions clients.

OpenCode talks the OpenAI-compatible Chat Completions API, so unlike the codex
(Responses) and claude (Anthropic) proxies this one does NO format translation —
it forwards each request verbatim to https://api.deepseek.com/v1 and relays the
response (streaming SSE or buffered JSON) back unchanged. Its sole job is to
RECORD, for every API call, the COMPLETE request the model saw (system prompt +
all messages + tool definitions) plus the assembled response and usage, so the
true model-visible context — AFTER opencode's native skill loading and plugin
post-processing of tool returns — is captured at the ground-truth boundary.

One proxy is started per isolated HarnessLens call and writes
`<trial>/api_calls.jsonl` (one JSON object per upstream call), so attribution to
(run, cell, version, task, trial) is structural (the file path) and call_index is
a clean per-trial monotonic counter.

Role tagging: the tau² user-simulator is pointed at base url `.../usersim/v1`, so
requests whose path begins with `/usersim` are logged with role="user_sim" (and
the `/usersim` prefix is stripped before forwarding); everything else is the
agent (role="agent").
"""

import http.server
import fcntl
import json
import os
import signal
import socketserver
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request

KEY = ""
LOG_FILE = None
AGENT_SEED = None
LOGGER_VERSION = "atomic-flock-v1"

_call_counter = 0
_counter_lock = threading.Lock()
_log_lock = threading.Lock()
_active_requests = 0
_active_cond = threading.Condition()
_RETRYABLE_429_CODES = {"insufficient_quota", "limit_burst_rate"}
_TRANSIENT_UPSTREAM_STATUSES = {500, 502, 503, 504}


def _provider_error_code(body):
    try:
        payload = json.loads(body.decode("utf-8", "replace"))
    except (AttributeError, json.JSONDecodeError, UnicodeDecodeError):
        return ""
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        return str(error.get("code") or error.get("type") or "")
    if isinstance(payload, dict):
        return str(payload.get("code") or payload.get("type") or "")
    return ""


def _upstream_retry_policy(status, body, attempt, retry_after=""):
    """Return a bounded retry delay, or None when the error must be relayed."""
    status = int(status)
    if status in _TRANSIENT_UPSTREAM_STATUSES:
        try:
            maximum_attempts = int(os.environ.get("HAI_PROVIDER_RETRY_ATTEMPTS", "4"))
        except ValueError:
            maximum_attempts = 4
        maximum_attempts = max(1, min(maximum_attempts, 6))
        if int(attempt) >= maximum_attempts:
            return None
        return min(5.0 * (2 ** (int(attempt) - 1)), 60.0)
    if status != 429:
        return None
    code = _provider_error_code(body)
    if code not in _RETRYABLE_429_CODES:
        return None
    try:
        maximum_attempts = int(os.environ.get("HAI_PROVIDER_RETRY_ATTEMPTS", "4"))
    except ValueError:
        maximum_attempts = 4
    maximum_attempts = max(1, min(maximum_attempts, 6))
    if int(attempt) >= maximum_attempts:
        return None
    base = 5.0 if code == "limit_burst_rate" else 30.0
    delay = min(base * (2 ** (int(attempt) - 1)), 120.0)
    try:
        delay = max(delay, float(retry_after or 0))
    except ValueError:
        pass
    return delay


def _upstream_exception_retry_delay(attempt):
    """Back off transient connection failures before returning a synthetic 502."""
    try:
        maximum_attempts = int(os.environ.get("HAI_PROVIDER_RETRY_ATTEMPTS", "4"))
    except ValueError:
        maximum_attempts = 4
    maximum_attempts = max(1, min(maximum_attempts, 6))
    if int(attempt) >= maximum_attempts:
        return None
    return min(5.0 * (2 ** (int(attempt) - 1)), 60.0)


def _provider_slot_count():
    try:
        count = int(os.environ.get("HAI_PROVIDER_MAX_CONCURRENCY", "20"))
    except ValueError:
        count = 20
    return max(1, min(count, 64))


def _acquire_provider_slot():
    root = os.environ.get("HAI_PROVIDER_SLOT_DIR", "/tmp/harnesslens-provider-slots")
    os.makedirs(root, exist_ok=True)
    deadline = time.monotonic() + 900.0
    while time.monotonic() < deadline:
        for index in range(_provider_slot_count()):
            fd = os.open(os.path.join(root, f"slot-{index:02d}.lock"), os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except BlockingIOError:
                os.close(fd)
        time.sleep(0.1)
    raise TimeoutError("timed out waiting for a shared provider concurrency slot")


def _release_provider_slot(fd):
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


DEFAULT_UPSTREAM_BASE_URL = "https://api.deepseek.com/v1"


def _upstream_base_url() -> str:
    """Resolve the upstream endpoint the way every other module does.

    DEEPSEEK_BASE_URL is the documented variable; DEEPSEEK_URL is the older
    name kept for compatibility. Requiring only the older one used to surface
    as an opaque 502 from this proxy.
    """
    return str(
        os.environ.get("DEEPSEEK_BASE_URL")
        or os.environ.get("DEEPSEEK_URL")
        or DEFAULT_UPSTREAM_BASE_URL
    ).rstrip("/")


def _upstream_url(path):
    base = _upstream_base_url()
    normalized_path = path if path.startswith("/") else f"/{path}"
    if base.endswith("/v1") and normalized_path.startswith("/v1/"):
        normalized_path = normalized_path[len("/v1"):]
    return f"{base}{normalized_path}"


def _next_call_index():
    global _call_counter
    with _counter_lock:
        _call_counter += 1
        return _call_counter


def _effective_request(role, request):
    """Inject the per-trial seed into agent calls without changing user-sim seeds."""
    if not isinstance(request, dict):
        return request
    effective = dict(request)
    if role == "agent" and AGENT_SEED is not None:
        effective["seed"] = int(AGENT_SEED)
    return effective


def _log(entry):
    if not LOG_FILE:
        return
    entry = dict(entry)
    entry.setdefault("logger_version", LOGGER_VERSION)
    payload = (json.dumps(entry, ensure_ascii=False, default=str) + "\n").encode("utf-8")
    with _log_lock:
        fd = os.open(LOG_FILE, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short write while recording OpenCode API trace")
                view = view[written:]
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(fd)


def _begin_request():
    global _active_requests
    with _active_cond:
        _active_requests += 1


def _end_request():
    global _active_requests
    with _active_cond:
        _active_requests = max(0, _active_requests - 1)
        _active_cond.notify_all()


def _graceful_shutdown(_signum, _frame):
    """Let in-flight proxy threads finish writing complete JSONL records.

    The rollout worker terminates this per-trial proxy immediately after the agent
    process exits. Without a SIGTERM handler, Python can exit while a proxy thread
    is serializing a large request/response entry, leaving a truncated JSON line.
    Downstream HarnessLens analysis requires complete API-boundary traces, so wait briefly
    for active handlers and the log lock before exiting.
    """
    deadline = time.time() + 25.0
    with _active_cond:
        while _active_requests and time.time() < deadline:
            _active_cond.wait(timeout=0.1)
    if _active_requests == 0:
        with _log_lock:
            pass
    sys.exit(0)


def _assemble_stream(buf_text):
    """Reconstruct an OpenAI chat-completion response object from a buffered SSE
    stream (sequence of `data: {chunk}` lines). Returns (response_obj, usage)."""
    content_parts = []
    reasoning_parts = []
    tool_calls = {}          # index -> {id, name, arguments}
    model = ""
    finish_reason = None
    usage = None
    role = "assistant"
    for line in buf_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if not data or data == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if chunk.get("model"):
            model = chunk["model"]
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices", []) or []:
            delta = choice.get("delta", {}) or {}
            if delta.get("role"):
                role = delta["role"]
            if delta.get("content"):
                content_parts.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])
            for tc in delta.get("tool_calls", []) or []:
                idx = tc.get("index", 0)
                slot = tool_calls.setdefault(idx, {"id": None, "name": None, "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function", {}) or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
    message = {"role": role, "content": "".join(content_parts) or None}
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = [
            {"id": tc["id"], "type": "function",
             "function": {"name": tc["name"], "arguments": tc["arguments"]}}
            for _, tc in sorted(tool_calls.items())
        ]
    resp_obj = {"model": model, "object": "chat.completion",
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}]}
    if usage is not None:
        resp_obj["usage"] = usage
    return resp_obj, usage


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        # Non-LLM passthrough (e.g. /v1/models). Not logged.
        path = self._upstream_path()
        try:
            r = urllib.request.Request(_upstream_url(path),
                                       headers={"Authorization": f"Bearer {KEY}"}, method="GET")
            with urllib.request.urlopen(r, context=ssl.create_default_context(), timeout=120) as resp:
                body = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        except Exception as e:  # noqa: BLE001
            self._error(e)

    def _role(self):
        return "user_sim" if self.path.startswith("/usersim") else "agent"

    def _upstream_path(self):
        # Strip the /usersim role-tag prefix; everything else forwards verbatim.
        p = self.path
        if p.startswith("/usersim"):
            p = p[len("/usersim"):] or "/"
        return p

    def do_POST(self):
        _begin_request()
        provider_slot = None
        try:
            provider_slot = _acquire_provider_slot()
            role = self._role()
            path = self._upstream_path()
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                req_json = json.loads(body)
            except (json.JSONDecodeError, ValueError):
                req_json = None
            req_json = _effective_request(role, req_json)
            if isinstance(req_json, dict):
                body = json.dumps(req_json, ensure_ascii=False).encode("utf-8")
            stream = bool(req_json.get("stream")) if isinstance(req_json, dict) else False
            call_index = _next_call_index()
            ts = time.time()

            hdrs = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
            upstream_attempt = 0
            while True:
                upstream_attempt += 1
                try:
                    r = urllib.request.Request(
                        _upstream_url(path), data=body, headers=hdrs, method="POST"
                    )
                    resp = urllib.request.urlopen(
                        r, context=ssl.create_default_context(), timeout=300
                    )
                    break
                except urllib.error.HTTPError as e:
                    err_body = e.read()
                    error_code = _provider_error_code(err_body)
                    delay = _upstream_retry_policy(
                        e.code,
                        err_body,
                        upstream_attempt,
                        e.headers.get("Retry-After", ""),
                    )
                    if delay is not None:
                        _log(
                            {
                                "ts": time.time(),
                                "event": "upstream_retry",
                                "call_index": call_index,
                                "harness": "opencode",
                                "role": role,
                                "upstream_status": e.code,
                                "error_code": error_code,
                                "upstream_attempt": upstream_attempt,
                                "retry_delay_s": delay,
                            }
                        )
                        time.sleep(delay)
                        continue
                    self.send_response(e.code)
                    self.send_header(
                        "Content-Type", e.headers.get("Content-Type", "application/json")
                    )
                    self.send_header("Content-Length", str(len(err_body)))
                    self.end_headers()
                    self.wfile.write(err_body)
                    _log({"ts": ts, "call_index": call_index, "harness": "opencode", "role": role,
                          "model": (req_json or {}).get("model") if isinstance(req_json, dict) else None,
                          "request": req_json, "response": None, "usage": None,
                          "stream": stream, "upstream_status": e.code,
                          "upstream_attempt_count": upstream_attempt,
                          "error": err_body.decode("utf-8", "replace")[:2000]})
                    return
                except Exception as e:  # noqa: BLE001
                    delay = _upstream_exception_retry_delay(upstream_attempt)
                    if delay is not None:
                        _log(
                            {
                                "ts": time.time(),
                                "event": "upstream_retry",
                                "call_index": call_index,
                                "harness": "opencode",
                                "role": role,
                                "upstream_status": 502,
                                "error_code": type(e).__name__,
                                "upstream_attempt": upstream_attempt,
                                "retry_delay_s": delay,
                            }
                        )
                        time.sleep(delay)
                        continue
                    self._error(e)
                    _log({"ts": ts, "call_index": call_index, "harness": "opencode", "role": role,
                          "request": req_json, "response": None, "usage": None,
                          "stream": stream, "upstream_status": 502, "error": str(e)})
                    return

            status = resp.status
            ctype = resp.headers.get("Content-Type", "")
            is_stream = stream or "text/event-stream" in ctype

            if is_stream:
                # Relay SSE chunks to the client AS THEY ARRIVE while buffering for assembly.
                self.send_response(status)
                self.send_header("Content-Type", ctype or "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                chunks = []
                try:
                    while True:
                        chunk = resp.read(8192)
                        if not chunk:
                            break
                        chunks.append(chunk)
                        try:
                            self.wfile.write(chunk)
                            self.wfile.flush()
                        except Exception:  # noqa: BLE001 — client closed; keep buffering for the log
                            pass
                finally:
                    resp.close()
                buf_text = b"".join(chunks).decode("utf-8", "replace")
                resp_obj, usage = _assemble_stream(buf_text)
                _log({"ts": ts, "call_index": call_index, "harness": "opencode", "role": role,
                      "model": resp_obj.get("model") or ((req_json or {}).get("model") if isinstance(req_json, dict) else None),
                      "request": req_json, "response": resp_obj, "usage": usage,
                      "stream": True, "upstream_status": status,
                      "upstream_attempt_count": upstream_attempt})
            else:
                resp_body = resp.read()
                resp.close()
                self.send_response(status)
                self.send_header("Content-Type", ctype or "application/json")
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
                try:
                    resp_json = json.loads(resp_body)
                except (json.JSONDecodeError, ValueError):
                    resp_json = None
                usage = resp_json.get("usage") if isinstance(resp_json, dict) else None
                _log({"ts": ts, "call_index": call_index, "harness": "opencode", "role": role,
                      "model": (resp_json or {}).get("model") if isinstance(resp_json, dict)
                               else ((req_json or {}).get("model") if isinstance(req_json, dict) else None),
                      "request": req_json, "response": resp_json, "usage": usage,
                      "stream": False, "upstream_status": status,
                      "upstream_attempt_count": upstream_attempt})
        finally:
            _release_provider_slot(provider_slot)
            _end_request()

    def _error(self, e):
        try:
            err = json.dumps({"error": str(e)}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
        except Exception:  # noqa: BLE001
            pass

    def log_message(self, *a):
        pass


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = False



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


def main():
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
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--key")  # deprecated: prefer DEEPSEEK_API_KEY
    p.add_argument("--log-file", required=True)
    p.add_argument("--agent-seed", type=int)
    a = p.parse_args()
    global KEY, LOG_FILE, AGENT_SEED
    KEY = _resolve_api_key(a.key)
    LOG_FILE = a.log_file
    AGENT_SEED = a.agent_seed
    open(LOG_FILE, "w").close()
    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)
    s = ThreadingHTTPServer((a.host, a.port), H)
    print(f"PORT={s.server_address[1]}", flush=True)
    s.serve_forever()


if __name__ == "__main__":
    main()
