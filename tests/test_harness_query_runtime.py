import json
import os
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from harnesslens.evolution.analyzer import _validate_candidate
from harnesslens.harnesses.candidate_config_runtime import (
    compile_opencode_agent_definitions,
    relocate_opencode_instruction_paths,
)
from harnesslens.harnesses.harness_query_adapters import harness_query_adapter
from harnesslens.harnesses.native_candidate_runtime import (
    install_codex_hook_dispatcher,
    render_toml,
)
from harnesslens.harnesses.opencode_harness import OpencodeHarnessAdapter


pytestmark = pytest.mark.skipif(
    os.environ.get("HAI_RUN_HARNESS_QUERY_PROBES") != "1",
    reason="set HAI_RUN_HARNESS_QUERY_PROBES=1 to run local harness request probes",
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _validate_query_derived_candidate(harness, channel_ids, delta):
    inventory = {
        item["id"]: item
        for item in harness_query_adapter(
            harness, repo_root=REPO_ROOT
        ).query_channel_inventory()
    }
    contracts = {channel_id: inventory[channel_id] for channel_id in channel_ids}
    candidate = {
        "id": f"{harness}-runtime-probe",
        "objective": "Observe query-derived candidate channels in a model request.",
        "channel_plan": [
            {
                "channel_id": channel_id,
                "operation": "materialize the exact Harness Query operation",
                "experience_ids": ["exp-runtime-probe"],
                "rationale": "This channel carries one request-visibility sentinel.",
            }
            for channel_id in channel_ids
        ],
        "manifest_delta": delta,
        "validation": {"local_behavior_checks": ["Sentinel reaches the request."]},
    }
    _validate_candidate(
        candidate,
        experience_ids={"exp-runtime-probe"},
        channel_ids=set(channel_ids),
        channel_contracts=contracts,
        harness=harness,
    )
    return candidate["manifest_delta"]


class _CaptureHandler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("content-length") or 0)
        payload = json.loads(self.rfile.read(length))
        type(self).requests.append(payload)
        request_text = json.dumps(payload, ensure_ascii=False)
        already_loaded = "HQ_SKILL_BODY_SENTINEL_1ac4" in request_text
        if "HQ_SYSTEM_SENTINEL_7f3a" in request_text and not already_loaded:
            body = (
                'data: {"id":"probe","object":"chat.completion.chunk","created":0,'
                '"model":"probe","choices":[{"index":0,"delta":{"role":"assistant",'
                '"tool_calls":[{"index":0,"id":"call_skill","type":"function",'
                '"function":{"name":"skill","arguments":"{\\"name\\":\\"query-probe\\"}"}}]},'
                '"finish_reason":null}]}\n\n'
                'data: {"id":"probe","object":"chat.completion.chunk","created":0,'
                '"model":"probe","choices":[{"index":0,"delta":{},'
                '"finish_reason":"tool_calls"}]}\n\n'
                "data: [DONE]\n\n"
            ).encode()
        else:
            body = (
                'data: {"id":"probe","object":"chat.completion.chunk","created":0,'
                '"model":"probe","choices":[{"index":0,"delta":{"role":"assistant",'
                '"content":"OK"},"finish_reason":null}]}\n\n'
                'data: {"id":"probe","object":"chat.completion.chunk","created":0,'
                '"model":"probe","choices":[{"index":0,"delta":{},'
                '"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ).encode()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        return


def _opencode_binary() -> str:
    configured = os.environ.get("OPENCODE_BIN")
    candidates = [
        configured,
        shutil.which("opencode"),
        str(
            Path(os.environ.get("OPENCODE_PREFIX") or "~/.opencode").expanduser()
            / "bin"
            / "opencode"
        ),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    pytest.skip("OpenCode binary is unavailable")


def _pi_binary() -> str:
    candidates = [
        os.environ.get("PI_AGENT_BIN"),
        str(Path(__file__).resolve().parents[1] / ".pi-agent/node_modules/.bin/pi"),
        shutil.which("pi"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    pytest.skip("Pi binary is unavailable")


def _codex_binary() -> str:
    candidate = shutil.which("codex")
    if candidate:
        return candidate
    pytest.skip("Codex binary is unavailable")


def test_opencode_query_core_channels_are_visible_in_model_request(tmp_path):
    system_sentinel = "HQ_SYSTEM_SENTINEL_7f3a"
    instruction_sentinel = "HQ_INSTRUCTION_SENTINEL_2d91"
    skill_sentinel = "HQ_SKILL_DESCRIPTION_SENTINEL_58ce"
    skill_body_sentinel = "HQ_SKILL_BODY_SENTINEL_1ac4"
    agent_description_sentinel = "HQ_SUBAGENT_DESCRIPTION_SENTINEL_6be2"
    agent_body_sentinel = "HQ_SUBAGENT_BODY_SENTINEL_f921"
    delta = {
        "config_patch": {
            "agent.build.prompt": system_sentinel,
            "tools.skill": True,
        },
        "instructions": [instruction_sentinel],
        "files": [
            {
                "path": ".opencode/skills/query-probe/SKILL.md",
                "content": (
                    "---\nname: query-probe\n"
                    f"description: {skill_sentinel}\n---\n\n{skill_body_sentinel}\n"
                ),
            },
            {
                "path": ".opencode/agents/evidence-reviewer.md",
                "content": (
                    "---\n"
                    f"description: {agent_description_sentinel}\n"
                    "mode: subagent\n"
                    "---\n\n"
                    f"{agent_body_sentinel}\n"
                ),
            },
        ],
    }
    delta = _validate_query_derived_candidate(
        "opencode",
        ("system_prompt", "instructions_rules", "skills", "agent_definitions"),
        delta,
    )
    assert OpencodeHarnessAdapter().materialized_channel_ids(delta) == {
        "system_prompt",
        "instructions_rules",
        "skills",
        "agent_definitions",
    }

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skill = workspace / delta["files"][0]["path"]
    skill.parent.mkdir(parents=True)
    skill.write_text(delta["files"][0]["content"], encoding="utf-8")
    agent = workspace / delta["files"][1]["path"]
    agent.parent.mkdir(parents=True, exist_ok=True)
    agent.write_text(delta["files"][1]["content"], encoding="utf-8")
    instruction = workspace / "AGENTS.md"
    instruction.write_text(instruction_sentinel, encoding="utf-8")

    _CaptureHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = {
            "$schema": "https://opencode.ai/config.json",
            "autoupdate": False,
            "share": "disabled",
            "model": "probe/probe",
            "small_model": "probe/probe",
            "enabled_providers": ["probe"],
            "provider": {
                "probe": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Probe",
                    "options": {
                        "baseURL": f"http://127.0.0.1:{server.server_port}/v1",
                        "apiKey": "probe",
                    },
                    "models": {
                        "probe": {
                            "name": "Probe",
                            "limit": {"context": 65536, "output": 4096},
                        }
                    },
                }
            },
            "agent": {
                "build": {
                    "steps": 2,
                    "prompt": delta["config_patch"]["agent.build.prompt"],
                },
                **compile_opencode_agent_definitions(workspace),
            },
            "instructions": ["AGENTS.md"],
            "tools": {"skill": True},
            "permission": {"skill": "allow"},
        }
        config = relocate_opencode_instruction_paths(
            config,
            project_root=workspace,
        )
        config_path = workspace / "opencode.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        runtime = tmp_path / "runtime"
        env = {
            **os.environ,
            "OPENCODE_CONFIG": str(config_path),
            "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
            "OPENCODE_DISABLE_AUTOCOMPACT": "1",
            "OPENCODE_DISABLE_PRUNE": "1",
            "XDG_DATA_HOME": str(runtime / "data"),
            "XDG_CONFIG_HOME": str(runtime / "config"),
            "XDG_STATE_HOME": str(runtime / "state"),
            "XDG_CACHE_HOME": str(runtime / "cache"),
        }
        process = subprocess.Popen(
            [
                _opencode_binary(),
                "run",
                "-m",
                "probe/probe",
                "--pure",
                "--auto",
                "--dir",
                str(workspace),
                "--format",
                "json",
                "Reply with OK.",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            request_texts = [
                json.dumps(payload, ensure_ascii=False)
                for payload in _CaptureHandler.requests
            ]
            if any(skill_body_sentinel in text for text in request_texts):
                break
            if process.poll() is not None:
                break
            time.sleep(0.1)
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=5)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert _CaptureHandler.requests
    request_texts = [
        json.dumps(payload, ensure_ascii=False) for payload in _CaptureHandler.requests
    ]
    agent_requests = [text for text in request_texts if system_sentinel in text]
    assert agent_requests, (request_texts, stdout[-2000:], stderr[-2000:])
    assert instruction_sentinel in agent_requests[0]
    assert '"name": "skill"' in agent_requests[0]
    assert skill_sentinel not in agent_requests[0]
    assert skill_body_sentinel not in agent_requests[0]
    assert agent_description_sentinel in agent_requests[0]
    assert agent_body_sentinel not in agent_requests[0]


def test_pi_query_core_channels_are_visible_in_model_request(tmp_path):
    system_sentinel = "PI_SYSTEM_SENTINEL_a112"
    instruction_sentinel = "PI_AGENTS_SENTINEL_b223"
    skill_sentinel = "PI_SKILL_DESCRIPTION_SENTINEL_c334"
    skill_body_sentinel = "PI_SKILL_BODY_SENTINEL_d445"
    delta = _validate_query_derived_candidate(
        "pi",
        ("system_prompt", "project_instructions", "skills"),
        {
            "files": [
                {"path": "AGENTS.md", "content": instruction_sentinel},
                {"path": ".pi/APPEND_SYSTEM.md", "content": system_sentinel},
                {
                    "path": ".pi/skills/query-probe/SKILL.md",
                    "content": (
                        "---\nname: query-probe\n"
                        f"description: {skill_sentinel}\n---\n\n{skill_body_sentinel}\n"
                    ),
                },
            ],
        },
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / delta["files"][0]["path"]).write_text(
        delta["files"][0]["content"], encoding="utf-8"
    )
    append_prompt = workspace / delta["files"][1]["path"]
    append_prompt.parent.mkdir(parents=True)
    append_prompt.write_text(delta["files"][1]["content"], encoding="utf-8")
    skill = workspace / delta["files"][2]["path"]
    skill.parent.mkdir(parents=True)
    skill.write_text(delta["files"][2]["content"], encoding="utf-8")

    _CaptureHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    pi_home = tmp_path / "pi-home"
    pi_home.mkdir()
    (pi_home / "models.json").write_text(
        json.dumps(
            {
                "providers": {
                    "probe": {
                        "baseUrl": f"http://127.0.0.1:{server.server_port}/v1",
                        "api": "openai-completions",
                        "apiKey": "probe",
                        "compat": {
                            "supportsDeveloperRole": False,
                            "supportsReasoningEffort": False,
                        },
                        "models": [{"id": "probe"}],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    try:
        result = subprocess.run(
            [
                _pi_binary(),
                "--mode",
                "json",
                "--print",
                "--no-session",
                "--offline",
                "--tools",
                "read",
                "--approve",
                "--provider",
                "probe",
                "--model",
                "probe",
                "Reply with OK.",
            ],
            cwd=workspace,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "HOME": str(tmp_path / "home"),
                "PI_CODING_AGENT_DIR": str(pi_home),
                "PI_OFFLINE": "1",
                "PI_TELEMETRY": "0",
            },
            timeout=60,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result.returncode == 0, result.stderr[-2000:]
    request_text = "\n".join(
        json.dumps(payload, ensure_ascii=False) for payload in _CaptureHandler.requests
    )
    assert system_sentinel in request_text
    assert instruction_sentinel in request_text
    assert skill_sentinel in request_text
    assert skill_body_sentinel not in request_text


def test_codex_query_core_channels_are_visible_in_model_request(tmp_path):
    developer_sentinel = "CODEX_DEVELOPER_SENTINEL_a91f"
    instruction_sentinel = "CODEX_AGENTS_SENTINEL_b82e"
    skill_sentinel = "CODEX_SKILL_DESCRIPTION_SENTINEL_c73d"
    skill_body_sentinel = "CODEX_SKILL_BODY_SENTINEL_d64c"
    hook_sentinel = "CODEX_SESSION_HOOK_SENTINEL_e85b"
    delta = _validate_query_derived_candidate(
        "codex",
        (
            "developer_instructions",
            "project_instructions",
            "skills",
            "hooks",
        ),
        {
            "files": [
                {"path": "AGENTS.md", "content": instruction_sentinel},
                {
                    "path": ".codex/config.toml",
                    "content": f'developer_instructions = "{developer_sentinel}"\n',
                },
                {
                    "path": ".agents/skills/query-probe/SKILL.md",
                    "content": (
                        "---\nname: query-probe\n"
                        f"description: {skill_sentinel}\n---\n\n{skill_body_sentinel}\n"
                    ),
                },
                {
                    "path": ".codex/harness-hook-context.md",
                    "content": hook_sentinel,
                },
            ],
        },
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for item in delta["files"]:
        target = workspace / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(item["content"], encoding="utf-8")
    assert install_codex_hook_dispatcher(workspace, delta)

    _CaptureHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        render_toml(
            {
                "model": "gpt-5.4",
                "model_provider": "probe",
                "model_reasoning_effort": "high",
                "developer_instructions": developer_sentinel,
                "sandbox_mode": "read-only",
                "approval_policy": "never",
                "model_providers": {
                    "probe": {
                        "name": "Probe",
                        "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                        "env_key": "OPENAI_API_KEY",
                        "wire_api": "responses",
                    }
                },
                "features": {
                    "enable_fanout": True,
                    "shell_tool": False,
                    "unified_exec": False,
                    "multi_agent": True,
                    "multi_agent_v2": True,
                },
                "projects": {str(workspace.resolve()): {"trust_level": "trusted"}},
            }
        ),
        encoding="utf-8",
    )
    source_cache = Path(
        os.environ.get("HAI_CODEX_MODELS_CACHE") or "~/.codex/models_cache.json"
    ).expanduser()
    if source_cache.is_file():
        shutil.copy2(source_cache, codex_home / "models_cache.json")
    process = None
    try:
        process = subprocess.Popen(
            [
                _codex_binary(),
                "exec",
                "--strict-config",
                "--skip-git-repo-check",
                "--ignore-rules",
                "--dangerously-bypass-hook-trust",
                "--json",
                (
                    "Use a subagent to inspect the workspace, "
                    "wait for it, then reply OK."
                ),
            ],
            cwd=workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={
                **os.environ,
                "OPENAI_API_KEY": "probe",
                "CODEX_HOME": str(codex_home),
                "HOME": str(tmp_path / "home"),
            },
            start_new_session=True,
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not _CaptureHandler.requests:
            if process.poll() is not None:
                break
            time.sleep(0.1)
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=5)
    finally:
        if process is not None and process.poll() is None:
            process.kill()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert _CaptureHandler.requests, (stdout[-2000:], stderr[-2000:])
    request_text = "\n".join(
        json.dumps(payload, ensure_ascii=False) for payload in _CaptureHandler.requests
    )
    assert developer_sentinel in request_text
    assert instruction_sentinel in request_text
    assert skill_sentinel in request_text
    assert skill_body_sentinel not in request_text
    assert hook_sentinel in request_text, (
        stdout[-8000:],
        stderr[-8000:],
        (workspace / ".codex/hooks.json").read_text(encoding="utf-8"),
        (
            (workspace / ".codex/harness-hook-observation.json").read_text(
                encoding="utf-8"
            )
            if (workspace / ".codex/harness-hook-observation.json").exists()
            else "hook-not-executed"
        ),
    )
    collaboration_tools = [
        tool
        for payload in _CaptureHandler.requests
        for tool in payload.get("tools", [])
        if str(tool.get("name") or tool.get("function", {}).get("name") or "")
        == "collaboration"
    ]
    assert collaboration_tools
