import json
import io
import queue
import threading
import time
import tomllib
from pathlib import Path

from harnesslens.infrastructure.codex_responses_proxy import (
    _patch_tau2_tools,
    _translate_request,
)
from harnesslens.benchmarks.codex_tau2 import _run_codex_turn, _setup_codex_home
from harnesslens.benchmarks.pi_tau2 import (
    _PiRpcSession,
    _candidate_system_prompt,
    _pi_rpc_command,
    _pi_tau2_extension_source,
    _write_pi_project,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_codex_turn_never_inherits_the_global_codex_home(tmp_path, monkeypatch):
    runtime_cwd = tmp_path / "trial" / "project"
    global_home = tmp_path / "global-home"
    global_config = global_home / ".codex" / "config.toml"
    global_config.parent.mkdir(parents=True)
    global_config.write_text("global-marker\n", encoding="utf-8")
    captured = {}

    class Process:
        returncode = 0

        @staticmethod
        def communicate(*, timeout):
            return "", ""

    def fake_popen(command, **kwargs):
        captured.update(command=command, **kwargs)
        return Process()

    monkeypatch.setenv("HOME", str(global_home))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("harnesslens.benchmarks.codex_tau2.subprocess.Popen", fake_popen)

    _run_codex_turn(
        runtime_cwd=runtime_cwd,
        proxy_port=12345,
        user_text="task",
        first_turn=True,
        timeout_s=10,
    )

    runtime_home = runtime_cwd / "home"
    assert captured["env"]["HOME"] == str(runtime_home)
    assert captured["env"]["CODEX_HOME"] == str(runtime_home / ".codex")
    assert captured["env"]["OPENAI_API_KEY"] == "harnesslens-local-proxy"
    assert "--strict-config" in captured["command"]
    assert "DEEPSEEK_API_KEY" not in captured["env"]
    assert Path(captured["env"]["CODEX_HOME"]) != global_config.parent
    assert global_config.read_text(encoding="utf-8") == "global-marker\n"


def test_pi_rpc_turn_timeout_measures_inactivity_not_total_turn_time(
    tmp_path, monkeypatch
):
    class Process:
        stdin = io.StringIO()

        @staticmethod
        def poll():
            return None

    events = queue.Queue()
    events.put({"type": "response", "id": "1", "success": True})
    session = _PiRpcSession(
        process=Process(),
        output_queue=events,
        stderr_path=tmp_path / "stderr.txt",
        stderr_buffer=[],
        lock=threading.Lock(),
    )
    monkeypatch.setattr(session, "_last_assistant_text", lambda **kwargs: "done")

    def emit_active_turn():
        time.sleep(0.25)
        events.put({"type": "tool_execution_start"})
        time.sleep(0.25)
        events.put({"type": "agent_settled"})

    worker = threading.Thread(target=emit_active_turn)
    worker.start()
    result = session.prompt("help", timeout_s=0.3)
    worker.join()

    assert result.error == ""
    assert result.text == "done"


def test_pi_candidate_materializes_prompt_files_config_and_tool_schema(tmp_path):
    manifest = {
        "prompt_appends": ["PI candidate prompt sentinel."],
        "files": [
            {"path": "AGENTS.md", "content": "PI project sentinel.\n"},
            {
                "path": ".pi/skills/query-probe/SKILL.md",
                "content": (
                    "---\nname: query-probe\n"
                    "description: PI skill sentinel.\n---\n\nPI skill body.\n"
                ),
            },
        ],
        "config_patch": {"compaction.enabled": False},
        "tool_desc_patches": {
            "lookup_record": {
                "desc": "Patched Pi lookup description.",
                "params": {"record_id": "Patched Pi record identifier."},
            }
        },
    }
    runtime_cwd = tmp_path / "workspace"
    pi_home = tmp_path / "home"

    _write_pi_project(
        runtime_cwd=runtime_cwd,
        pi_home=pi_home,
        repo_root=REPO_ROOT,
        socket_path="/tmp/query-probe.sock",
        harness_manifest=manifest,
    )

    assert "PI project sentinel." in (runtime_cwd / "AGENTS.md").read_text()
    assert (runtime_cwd / ".pi/skills/query-probe/SKILL.md").is_file()
    settings = json.loads((runtime_cwd / ".pi/settings.json").read_text())
    assert settings["compaction"]["enabled"] is False
    models = json.loads((pi_home / "models.json").read_text())
    assert models["providers"]["deepseek"]["api"] == "openai-completions"
    assert models["providers"]["deepseek"]["models"][0]["id"] == (
        "deepseek-v4-flash"
    )
    extension = (runtime_cwd / ".pi/tau2_extension.ts").read_text()
    assert "Patched Pi lookup description." in extension
    assert "Patched Pi record identifier." in extension
    assert _candidate_system_prompt("Base prompt.", manifest).endswith(
        "PI candidate prompt sentinel."
    )


def test_pi_rollout_keeps_query_discovered_context_and_skill_channels(tmp_path):
    command = _pi_rpc_command(
        REPO_ROOT,
        tmp_path,
        system_prompt_append="Pi native system sentinel.",
    )

    assert "--no-builtin-tools" in command
    assert "--tools" not in command
    assert command[command.index("--append-system-prompt") + 1] == (
        "Pi native system sentinel."
    )
    assert "--no-context-files" not in command
    assert "--no-extensions" not in command


def test_pi_workspace_append_system_file_is_in_native_system_prompt():
    sentinel = "PI workspace system channel sentinel."
    manifest = {
        "prompt_appends": [],
        "_workspace": {
            "schema": 1,
            "files": [
                {
                    "scope": "project",
                    "path": ".pi/APPEND_SYSTEM.md",
                    "content": sentinel,
                    "executable": False,
                }
            ],
        },
    }

    assert _candidate_system_prompt("Base prompt.", manifest) == (
        f"Base prompt.\n\n{sentinel}"
    )


def test_pi_workspace_skill_is_compiled_into_restricted_system_context():
    content = (
        "---\nname: join-check\ndescription: Verify joins before querying.\n---\n"
        "\nRun a small join sample first.\n"
    )
    manifest = {
        "_workspace": {
            "schema": 1,
            "files": [
                {
                    "scope": "project",
                    "path": ".pi/skills/join-check/SKILL.md",
                    "content": content,
                    "executable": False,
                }
            ],
        }
    }

    prompt = _candidate_system_prompt("Base prompt.", manifest)

    assert '<harness_skill name="join-check"' in prompt
    assert "Verify joins before querying." in prompt
    assert "Run a small join sample first." in prompt


def test_pi_workspace_native_settings_survive_but_fixed_runtime_wins(tmp_path):
    runtime_cwd = tmp_path / "workspace"
    pi_home = tmp_path / "home"
    manifest = {
        "_workspace": {
            "schema": 1,
            "files": [
                {
                    "scope": "home",
                    "path": "settings.json",
                    "content": json.dumps(
                        {
                            "compaction": {"enabled": False},
                            "model": "candidate-forbidden",
                            "provider": {"default": "candidate-forbidden"},
                        }
                    ),
                    "executable": False,
                }
            ],
        }
    }

    _write_pi_project(
        runtime_cwd=runtime_cwd,
        pi_home=pi_home,
        repo_root=REPO_ROOT,
        socket_path="/tmp/query-probe.sock",
        harness_manifest=manifest,
    )

    settings = json.loads((runtime_cwd / ".pi/settings.json").read_text())
    assert settings["compaction"]["enabled"] is False
    assert settings["model"] == "deepseek-v4-flash"
    assert settings["provider"]["default"] == "deepseek"


def test_codex_candidate_materializes_files_and_merged_toml_config(tmp_path):
    manifest = {
        "files": [
            {"path": "AGENTS.md", "content": "Codex project sentinel.\n"},
            {
                "path": ".agents/skills/query-probe/SKILL.md",
                "content": (
                    "---\nname: query-probe\n"
                    "description: Codex skill sentinel.\n---\n\nCodex skill body.\n"
                ),
            },
        ],
        "config_patch": {
            "developer_instructions": "Codex developer sentinel.",
            "features.shell_tool": False,
            "mcp_servers.query_probe.command": "printf",
            "mcp_servers.query_probe.args": ["ready"],
        },
        "tool_desc_patches": {},
    }
    runtime_cwd = tmp_path / "workspace"

    codex_home = _setup_codex_home(
        runtime_cwd,
        12345,
        "/tmp/query-probe.sock",
        REPO_ROOT,
        system_prompt="Tau2 base system and domain policy.",
        harness_manifest=manifest,
    )

    assert "Codex project sentinel." in (runtime_cwd / "AGENTS.md").read_text()
    assert (runtime_cwd / ".agents/skills/query-probe/SKILL.md").is_file()
    config = tomllib.loads((codex_home / "config.toml").read_text())
    assert config["developer_instructions"] == (
        "Tau2 base system and domain policy.\n\nCodex developer sentinel."
    )
    assert "disable_response_storage" not in config
    assert config["features"]["shell_tool"] is False
    assert config["features"]["enable_fanout"] is True
    assert config["features"]["multi_agent"] is True
    assert config["features"]["multi_agent_v2"] is True
    assert "orchestrator" not in config
    assert "agents" not in config
    assert config["mcp_servers"]["query_probe"] == {
        "command": "printf",
        "args": ["ready"],
    }
    assert config["mcp_servers"]["tau2"]["command"]
    assert "projects" not in config


def test_codex_hook_candidate_adds_only_exact_project_trust(tmp_path):
    runtime_cwd = tmp_path / "workspace"
    codex_home = _setup_codex_home(
        runtime_cwd,
        12345,
        "/tmp/query-probe.sock",
        REPO_ROOT,
        harness_manifest={
            "files": [
                {
                    "path": ".codex/harness-hook-context.md",
                    "content": "Keep this context visible after resume.",
                }
            ]
        },
    )

    config = tomllib.loads((codex_home / "config.toml").read_text())
    assert config["projects"] == {
        str(runtime_cwd.resolve()): {"trust_level": "trusted"}
    }
    assert (runtime_cwd / ".codex/hooks.json").is_file()


def test_codex_workspace_native_config_survives_only_inside_trial_home(tmp_path):
    runtime_cwd = tmp_path / "workspace"
    manifest = {
        "_workspace": {
            "schema": 1,
            "files": [
                {
                    "scope": "home",
                    "path": "config.toml",
                    "content": (
                        'model = "candidate-forbidden"\n'
                        'notice = "candidate-setting"\n\n'
                        '[mcp_servers.candidate]\ncommand = "printf"\n'
                    ),
                    "executable": False,
                }
            ],
        }
    }

    codex_home = _setup_codex_home(
        runtime_cwd,
        12345,
        "/tmp/query-probe.sock",
        REPO_ROOT,
        harness_manifest=manifest,
    )

    config = tomllib.loads((codex_home / "config.toml").read_text())
    assert config["notice"] == "candidate-setting"
    assert config["model"] == "gpt-5.4"
    assert config["model_provider"] == "deepseek"
    assert config["developer_instructions"] == ""
    assert config["mcp_servers"]["candidate"]["command"] == "printf"
    assert config["mcp_servers"]["tau2"]["command"]
    assert codex_home == runtime_cwd / "home" / ".codex"


def test_codex_workspace_developer_instructions_are_model_visible_config(tmp_path):
    runtime_cwd = tmp_path / "workspace"
    codex_home = _setup_codex_home(
        runtime_cwd,
        12345,
        "/tmp/query-probe.sock",
        REPO_ROOT,
        system_prompt="Fixed task policy.",
        harness_manifest={
            "_workspace": {
                "schema": 1,
                "files": [
                    {
                        "scope": "home",
                        "path": "config.toml",
                        "content": 'developer_instructions = "Candidate guidance."\n',
                        "executable": False,
                    }
                ],
            }
        },
    )

    config = tomllib.loads((codex_home / "config.toml").read_text())
    assert config["developer_instructions"] == (
        "Fixed task policy.\n\nCandidate guidance."
    )


def test_codex_proxy_patches_exact_tau2_tool_and_parameter_descriptions():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_record",
                "description": "Original.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "record_id": {"type": "string", "description": "Original ID."}
                    },
                },
            },
        }
    ]

    patched = _patch_tau2_tools(
        tools,
        {
            "lookup_record": {
                "desc": "Patched Codex lookup description.",
                "params": {"record_id": "Patched Codex record identifier."},
            }
        },
    )

    function = patched[0]["function"]
    assert function["description"] == "Patched Codex lookup description."
    assert function["parameters"]["properties"]["record_id"]["description"] == (
        "Patched Codex record identifier."
    )


def test_codex_proxy_preserves_responses_instructions_for_upstream_model():
    translated = _translate_request(
        {
            "instructions": "Top-level developer and project context.",
            "input": [
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Turn context."}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Do the task."}],
                },
            ],
        }
    )

    assert translated["messages"] == [
        {"role": "system", "content": "Top-level developer and project context."},
        {"role": "system", "content": "Turn context."},
        {"role": "user", "content": "Do the task."},
    ]
