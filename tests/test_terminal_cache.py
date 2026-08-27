from __future__ import annotations

import json
import shlex
import tomllib
from pathlib import Path
from urllib.parse import urlparse

import pytest

from harnesslens.benchmarks import terminal_bench as tb
from harnesslens.benchmarks.terminal_cache import locked_entry


def test_terminal_agent_rollout_timeout_is_twenty_minutes():
    assert tb.TerminalLimits().agent_timeout_s == 1200


def test_prepare_codex_routes_harness_prompt_to_developer_instructions(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "key")
    monkeypatch.setattr(tb, "_container_host_proxy_address", lambda: "192.0.2.10")
    setup_commands = []
    monkeypatch.setattr(
        tb,
        "_dexec",
        lambda cid, command, **kwargs: setup_commands.append(command),
    )
    monkeypatch.setattr(tb, "_dcp_contents", lambda *args, **kwargs: None)
    monkeypatch.setattr(tb, "_dcp_in", lambda *args, **kwargs: None)

    command = tb._prepare_codex(
        "container",
        tmp_path,
        "solve the task",
        "base prompt\nActive harness files: candidate guidance",
        1234,
        {"config_patch": {"developer_instructions": "candidate developer"}},
    )

    assert shlex.quote("solve the task") in command
    assert "Active harness files" not in command
    assert any("mkdir -p /app " in setup for setup in setup_commands)
    config = tomllib.loads((
        tmp_path / "codex-home" / "config.toml"
    ).read_text(encoding="utf-8"))
    assert config["model_providers"]["deepseek"]["base_url"] == (
        "http://192.0.2.10:1234/v1"
    )
    assert config["developer_instructions"] == (
        "base prompt\nActive harness files: candidate guidance\n\n"
        "candidate developer"
    )


def test_container_proxy_env_bypasses_container_reachable_codex_proxy(monkeypatch):
    monkeypatch.setattr(tb, "_container_host_proxy_address", lambda: "192.0.2.10")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    proxy_env = tb._container_proxy_env()

    assert "192.0.2.10" in proxy_env["NO_PROXY"].split(",")


def test_terminal_opencode_uses_model_endpoint_and_bypasses_proxy(tmp_path, monkeypatch):
    endpoint = "https://relay.example.invalid/v1"
    monkeypatch.setenv("DEEPSEEK_BASE_URL", endpoint)
    monkeypatch.setattr(tb, "_container_host_proxy_address", lambda: "192.0.2.10")
    monkeypatch.setattr(tb, "_dexec", lambda *args, **kwargs: None)
    monkeypatch.setattr(tb, "_dcp_contents", lambda *args, **kwargs: None)
    monkeypatch.setattr(tb, "_dcp_in", lambda *args, **kwargs: None)

    tb._prepare_opencode(
        "container",
        tmp_path,
        "solve the task",
        "system prompt",
        tb.TerminalLimits(),
        {},
    )

    config = json.loads((tmp_path / "opencode.json").read_text(encoding="utf-8"))
    assert config["provider"]["deepseek"]["options"]["baseURL"] == endpoint
    assert config["provider"]["deepseek"]["models"]["deepseek-v4-flash"][
        "limit"
    ] == {"context": 1_000_000, "output": 24_576}
    assert urlparse(endpoint).hostname in tb._container_proxy_env()["NO_PROXY"].split(",")


def test_container_clash_proxies_packages_but_bypasses_current_model_endpoint(
    monkeypatch,
):
    endpoint = "https://relay.example.invalid/v1"
    monkeypatch.setenv("TB_ENABLE_CONTAINER_CLASH", "1")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", endpoint)
    monkeypatch.setattr(tb, "_container_host_proxy_address", lambda: "192.0.2.10")

    proxy_env = tb._container_proxy_env()

    assert proxy_env["http_proxy"] == "http://127.0.0.1:16627"
    assert proxy_env["https_proxy"] == "http://127.0.0.1:16627"
    assert urlparse(endpoint).hostname in proxy_env["NO_PROXY"].split(",")


def test_aihubmix_model_endpoint_uses_container_clash(monkeypatch):
    monkeypatch.setenv("TB_ENABLE_CONTAINER_CLASH", "1")
    monkeypatch.setenv("TB_MODEL_ENDPOINT_NO_PROXY", "0")
    monkeypatch.setenv(
        "DEEPSEEK_BASE_URL", "https://aihubmix.example.invalid/v1"
    )
    monkeypatch.setattr(tb, "_container_host_proxy_address", lambda: "192.0.2.10")

    proxy_env = tb._container_proxy_env()

    assert proxy_env["https_proxy"] == "http://127.0.0.1:16627"
    assert "aihubmix.example.invalid" not in proxy_env["NO_PROXY"].split(",")
    assert tb._trial_clash_compatible(
        {
            "container_clash": {
                "enabled": True,
                "model_endpoint_no_proxy": False,
                "lifecycle_verified": True,
            }
        }
    )


def test_container_clash_execs_mihomo_as_the_container_process(tmp_path, monkeypatch):
    clash_home = tmp_path / "clashctl"
    (clash_home / "bin").mkdir(parents=True)
    (clash_home / "resources").mkdir()
    (clash_home / "bin/mihomo").write_text("binary", encoding="utf-8")
    (clash_home / "resources/runtime.yaml").write_text(
        "mixed-port: 16627\n", encoding="utf-8"
    )
    monkeypatch.setenv("TB_CLASHCTL_HOME", str(clash_home))
    exec_commands = []

    def fake_exec(_cid, command, **_kwargs):
        exec_commands.append(command)
        return type(
            "Result", (), {"returncode": 0, "stdout": b"", "stderr": b""}
        )()

    monkeypatch.setattr(tb, "_dexec", fake_exec)
    monkeypatch.setattr(tb, "_dcp_in", lambda *_args, **_kwargs: None)
    launched = []

    def fake_detached(_cid, command, **_kwargs):
        launched.append(command)
        return type(
            "Result", (), {"returncode": 0, "stdout": b"", "stderr": b""}
        )()

    monkeypatch.setattr(tb, "_dexec_detached", fake_detached)

    runtime = tb._start_container_clash_limited("container", tmp_path / "output")

    assert launched == [
        "exec /tmp/harness-clashctl/bin/mihomo "
        "-d /tmp/harness-clashctl/resources "
        "-f /tmp/harness-clashctl/resources/runtime.yaml "
        ">>/agent-logs/container-mihomo.log 2>&1"
    ]
    assert "clashctl" not in launched[0].replace("/tmp/harness-clashctl", "")
    assert any("GET http://archive.ubuntu.com/ubuntu/" in item for item in exec_commands)
    assert runtime["runtime_log"].endswith("agent-logs/container-mihomo.log")


def test_container_proxy_bypasses_deepseek_url_fallback(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.setenv(
        "DEEPSEEK_URL", "https://gateway.example.invalid/compatible-mode/v1"
    )
    monkeypatch.setattr(tb, "_container_host_proxy_address", lambda: "192.0.2.10")

    assert "gateway.example.invalid" in tb._container_proxy_env()["NO_PROXY"].split(",")


def test_terminal_opencode_routes_model_through_host_proxy(tmp_path, monkeypatch):
    monkeypatch.setattr(tb, "_container_host_proxy_address", lambda: "192.0.2.10")
    monkeypatch.setattr(tb, "_dexec", lambda *args, **kwargs: None)
    monkeypatch.setattr(tb, "_dcp_contents", lambda *args, **kwargs: None)
    monkeypatch.setattr(tb, "_dcp_in", lambda *args, **kwargs: None)

    tb._prepare_opencode(
        "container",
        tmp_path,
        "solve the task",
        "system prompt",
        tb.TerminalLimits(),
        {},
        proxy_port=4321,
    )

    config = json.loads((tmp_path / "opencode.json").read_text(encoding="utf-8"))
    assert config["provider"]["deepseek"]["options"]["baseURL"] == (
        "http://192.0.2.10:4321/v1"
    )


def test_terminal_workspace_cleanup_preserves_trajectory_artifacts(tmp_path):
    for name in ("candidate-workspace", "harness-files", "codex-home"):
        path = tmp_path / name
        path.mkdir()
        (path / "temporary").write_text("x", encoding="utf-8")
    trace = tmp_path / "agent.stdout"
    trace.write_text("event\n", encoding="utf-8")

    tb._cleanup_staged_harness_workspace(tmp_path)

    assert trace.is_file()
    assert not any(
        (tmp_path / name).exists()
        for name in ("candidate-workspace", "harness-files", "codex-home")
    )


def test_container_package_env_propagates_indexes_and_bounds_uv_concurrency(monkeypatch):
    monkeypatch.setenv("PIP_INDEX_URL", "https://mirror.example/simple")
    monkeypatch.setenv("UV_INDEX_URL", "https://mirror.example/simple")

    package_env = tb._container_package_env()

    assert package_env["PIP_INDEX_URL"] == "https://mirror.example/simple"
    assert package_env["UV_INDEX_URL"] == "https://mirror.example/simple"
    assert package_env["UV_HTTP_RETRIES"] == "10"
    assert package_env["UV_HTTP_TIMEOUT"] == "120"
    assert package_env["UV_CONCURRENT_DOWNLOADS"] == "2"


def test_prepare_opencode_prefixes_dash_prefixed_instruction(tmp_path, monkeypatch):
    monkeypatch.setattr(tb, "_dexec", lambda *args, **kwargs: None)
    monkeypatch.setattr(tb, "_dcp_in", lambda *args, **kwargs: None)

    command = tb._prepare_opencode(
        "container",
        tmp_path,
        "- solve this task",
        "system prompt",
        tb.TerminalLimits(),
        {},
    )

    assert shlex.quote("Task instructions:\n\n- solve this task") in command
    assert "--format json -- " not in command


def test_terminal_native_candidate_configs_are_merged_before_fixed_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(tb, "_container_host_proxy_address", lambda: "192.0.2.10")
    monkeypatch.setattr(tb, "_dexec", lambda *args, **kwargs: None)
    monkeypatch.setattr(tb, "_dcp_contents", lambda *args, **kwargs: None)
    monkeypatch.setattr(tb, "_dcp_in", lambda *args, **kwargs: None)
    manifest = {
        "_workspace": {
            "schema": 1,
            "files": [
                {
                    "scope": "home",
                    "path": "config.json",
                    "content": json.dumps(
                        {
                            "model": "candidate-forbidden",
                            "notice": "candidate-setting",
                            "instructions": ["candidate-guidance.md"],
                            "agent": {"build": {"steps": 999, "prompt": "candidate prompt"}},
                            "mcp": {"candidate": {"command": ["printf", "ready"]}},
                        }
                    ),
                    "executable": False,
                },
                {
                    "scope": "home",
                    "path": "config.toml",
                    "content": (
                        'model = "candidate-forbidden"\n'
                        'notice = "candidate-setting"\n'
                        'developer_instructions = "candidate prompt"\n\n'
                        '[mcp_servers.candidate]\ncommand = "printf"\n'
                    ),
                    "executable": False,
                },
                {
                    "scope": "home",
                    "path": "settings.json",
                    "content": json.dumps(
                        {
                            "model": "candidate-forbidden",
                            "compaction": {"enabled": False},
                        }
                    ),
                    "executable": False,
                },
                {
                    "scope": "project",
                    "path": "candidate-guidance.md",
                    "content": "candidate guidance\n",
                    "executable": False,
                },
                {
                    "scope": "project",
                    "path": ".codex/hooks.json",
                    "content": '{"hooks": {}}\n',
                    "executable": False,
                },
            ],
        }
    }
    tb._materialize_manifest_files("container", tmp_path, "opencode", manifest)

    tb._prepare_opencode(
        "container", tmp_path, "task", "terminal system", tb.TerminalLimits(max_steps=17), manifest
    )
    opencode = json.loads((tmp_path / "opencode.json").read_text())
    assert opencode["notice"] == "candidate-setting"
    assert opencode["model"] == "deepseek/deepseek-v4-flash"
    assert opencode["agent"]["build"] == {
        "steps": 17,
        "prompt": "terminal system\n\ncandidate prompt",
    }
    assert opencode["mcp"]["candidate"]["command"] == ["printf", "ready"]
    assert opencode["instructions"] == ["/app/candidate-guidance.md"]

    codex_command = tb._prepare_codex(
        "container", tmp_path, "task", "terminal system", 1234, manifest
    )
    codex = tomllib.loads((tmp_path / "codex-home" / "config.toml").read_text())
    assert codex["notice"] == "candidate-setting"
    assert codex["model"] == "gpt-5.4"
    assert codex["model_provider"] == "deepseek"
    assert codex["developer_instructions"] == "terminal system\n\ncandidate prompt"
    assert codex["mcp_servers"]["candidate"]["command"] == "printf"
    assert "--dangerously-bypass-hook-trust" in codex_command

    tb._prepare_pi(
        "container", tmp_path, "task", "terminal system", tb.TerminalLimits(), manifest
    )
    pi = json.loads((tmp_path / "pi-settings.json").read_text())
    assert pi["compaction"]["enabled"] is False
    assert pi["model"] == "deepseek-v4-flash"


def test_verify_hardens_package_setup_script(tmp_path, monkeypatch):
    task_root = tmp_path / "task"
    task_root.mkdir()
    output_root = tmp_path / "output"
    output_root.mkdir()
    (task_root / "run-tests.sh").write_text(
        "#!/bin/bash\n"
        "apt-get install -y curl=8.5.0-2ubuntu10.6 binutils=2.42-4ubuntu2.5\n"
        "curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh\n"
        "source $HOME/.local/bin/env\n"
        "uv run pytest\n",
        encoding="utf-8",
    )
    commands = []

    def fake_dexec(cid, command, **kwargs):
        commands.append(command)
        return type("Result", (), {"returncode": 0, "stdout": b"1 passed", "stderr": b""})()

    monkeypatch.setattr(tb, "_dexec", fake_dexec)
    monkeypatch.setattr(tb, "_dcp_in", lambda *args, **kwargs: None)
    monkeypatch.setattr(tb, "_ensure_uv", lambda cid: None)

    reward, timed_out = tb._verify("container", task_root, output_root, 30)

    patched = (output_root / "run-tests.harness.sh").read_text(encoding="utf-8")
    assert "curl binutils" in patched
    assert "curl=8.5.0" not in patched
    assert "export PATH=/usr/local/bin:$PATH" in patched
    assert "source $HOME/.local/bin/env" not in patched
    assert any("fuser /var/lib/dpkg/lock-frontend" in command for command in commands)
    assert reward == 1.0
    assert timed_out is False


def test_harbor_verifier_runs_from_task_workdir(tmp_path, monkeypatch):
    task_root = tmp_path / "task"
    tests = task_root / "tests"
    tests.mkdir(parents=True)
    (tests / "test.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    output_root = tmp_path / "output"
    output_root.mkdir()
    calls = []

    def fake_dexec(cid, command, **kwargs):
        calls.append((command, kwargs))
        stdout = b"1" if "cat /logs/verifier/reward" in command else b""
        return type(
            "Result", (), {"returncode": 0, "stdout": stdout, "stderr": b""}
        )()

    monkeypatch.setattr(tb, "_dexec", fake_dexec)
    monkeypatch.setattr(tb, "_dcp_contents", lambda *args, **kwargs: None)

    reward, timed_out = tb._verify("container", task_root, output_root, 30)

    verifier_call = next(
        item for item in calls if "bash /tests/test.sh" in item[0]
    )
    assert verifier_call[1]["workdir"] == "/app"
    assert reward == 1.0
    assert timed_out is False


def test_codex_transport_failure_after_events_is_infrastructure_error():
    agent = tb.AgentProcessResult(
        returncode=1,
        stdout=(
            '{"type":"thread.started"}\n'
            '{"type":"turn.failed","error":{"message":'
            '"stream disconnected before completion: error sending request for url"}}'
        ),
        stderr="",
        saw_event=True,
        timed_out=False,
        timeout_kind="",
        n_tool_calls=0,
    )

    assert tb._agent_infrastructure_error(agent)


def test_pi_provider_auth_failure_after_events_is_infrastructure_error():
    agent = tb.AgentProcessResult(
        returncode=1,
        stdout=(
            '{"type":"runner_start"}\n'
            '{"type":"turn_end"}\n'
            '{"type":"message_end","message":{"stopReason":"error",'
            '"errorMessage":"401 Authentication Fails: invalid api key"}}'
        ),
        stderr="",
        saw_event=True,
        timed_out=False,
        timeout_kind="",
        n_tool_calls=0,
    )

    assert tb._agent_infrastructure_error(agent)


def test_pi_terminal_does_not_require_host_proxy_trace(tmp_path, monkeypatch):
    task_root = tmp_path / "third_party/terminal-bench/original-tasks/task"
    task_root.mkdir(parents=True)
    (task_root / "task.yaml").write_text(
        "instruction: finish the task\n", encoding="utf-8"
    )
    (task_root / "docker-compose.yaml").write_text(
        "services:\n  client:\n    image: example\n", encoding="utf-8"
    )
    monkeypatch.setattr(tb, "_runner_assets", lambda *_args: {"volumes": []})
    monkeypatch.setattr(tb, "_write_compose_override", lambda **_kwargs: tmp_path / "override")
    monkeypatch.setattr(tb, "_compose_env", lambda *_args: {})
    monkeypatch.setattr(
        tb,
        "_run_compose",
        lambda *_args, **_kwargs: type(
            "Result", (), {"returncode": 0, "stderr": ""}
        )(),
    )
    monkeypatch.setattr(tb, "_start_container_clash", lambda *_args: {"enabled": True})
    monkeypatch.setattr(tb, "_container_clash_healthy", lambda *_args: True)
    monkeypatch.setattr(tb, "_materialize_manifest_files", lambda *_args: None)
    monkeypatch.setattr(tb, "_prepare_pi", lambda *_args: "pi-command")
    monkeypatch.setattr(
        tb,
        "_terminal_runtime_load_report",
        lambda **_kwargs: {"harness": "pi"},
    )
    monkeypatch.setattr(
        tb,
        "_run_agent_process",
        lambda **_kwargs: tb.AgentProcessResult(
            returncode=0,
            stdout='{"type":"agent_end"}\n',
            stderr="",
            saw_event=True,
            timed_out=False,
            timeout_kind="",
            n_tool_calls=1,
        ),
    )
    monkeypatch.setattr(tb, "_verify", lambda *_args: (1.0, False))

    row = tb.run_terminal_trial(
        repo_root=tmp_path,
        output_root=tmp_path / "output",
        harness="pi",
        harness_version="v0",
        harness_manifest={},
        task_id="task",
        trial=0,
        pairing_slot=0,
        limits=tb.TerminalLimits(),
    )

    assert row["reward"] == 1.0
    assert row["infrastructure_error"] is False
    assert row["container_clash"]["lifecycle_verified"] is True


def test_required_container_clash_rejects_old_or_startup_only_trials(monkeypatch):
    monkeypatch.setenv("TB_ENABLE_CONTAINER_CLASH", "1")

    assert not tb._trial_clash_compatible({"container_clash": {"enabled": False}})
    assert not tb._trial_clash_compatible(
        {
            "container_clash": {
                "enabled": True,
                "deepseek_no_proxy": True,
                "startup_verified": True,
            }
        }
    )
    assert tb._trial_clash_compatible(
        {
            "container_clash": {
                "enabled": True,
                "deepseek_no_proxy": True,
                "startup_verified": True,
                "lifecycle_verified": True,
            }
        }
    )


def test_terminal_step_limit_counts_unique_tool_calls():
    events = [
        {"type": "tool.started", "call_id": "call_1"},
        {"type": "tool.completed", "call_id": "call_1"},
        {
            "type": "item.completed",
            "item": {"id": "call_2", "type": "command_execution"},
        },
    ]

    assert not tb._step_limit_reached(events, 3)
    assert tb._step_limit_reached(events, 2)


def test_terminal_step_limit_deduplicates_pi_tool_execution_events():
    events = [
        {
            "type": "tool_execution_start",
            "toolCallId": "call_1",
            "toolName": "bash",
        },
        {
            "type": "tool_execution_update",
            "toolCallId": "call_1",
            "toolName": "bash",
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "call_1",
            "toolName": "bash",
        },
    ]

    assert tb._count_tool_calls(events) == 1
    assert not tb._step_limit_reached(events, 2)
    assert tb._step_limit_reached(events, 1)


def test_agent_process_compacts_memory_and_counts_tools_incrementally(tmp_path, monkeypatch):
    fake_docker = tmp_path / "fake-docker"
    fake_docker.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "for index in range(20):\n"
        "    print(json.dumps({'type': 'message_update', 'message': 'x' * 500}))\n"
        "for kind in ('start', 'update', 'end'):\n"
        "    print(json.dumps({'type': f'tool_execution_{kind}', 'toolCallId': 'call-1'}))\n"
        "print(json.dumps({'type': 'tool_execution_start', 'toolCallId': 'call-2'}))\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    monkeypatch.setattr(tb, "_docker", lambda: str(fake_docker))
    monkeypatch.setattr(tb, "AGENT_OUTPUT_LIMIT_CHARS", 1024)

    result = tb._run_agent_process(
        cid="container",
        command="ignored",
        timeout_s=10,
        first_event_timeout_s=2,
        max_steps=10,
    )

    assert result.returncode == 0
    assert result.saw_event is True
    assert result.n_tool_calls == 2
    assert "agent output truncated by harness" in result.stdout
    assert len(result.stdout) < 1100


def test_terminal_project_name_replaces_compose_invalid_dots(monkeypatch):
    monkeypatch.setattr(tb.uuid, "uuid4", lambda: type("U", (), {"hex": "12345678"})())

    project = tb._project_name("install-windows-3.11", 1)

    assert project == "tb_install-windows-3-11_t1_12345678"
    assert "." not in project


def test_shared_cache_separates_harness_and_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("HAI_TERMINAL_BENCH_CACHE", str(tmp_path / "cache"))
    base = {"task_id": "task", "trial": 0, "reward": 1.0, "status": "completed"}

    with locked_entry(tmp_path, {"scope": "TRAIN", "harness": "opencode"}) as entry:
        assert entry.load() is None
        assert entry.store(base)
    with locked_entry(tmp_path, {"scope": "TRAIN", "harness": "opencode"}) as entry:
        assert entry.load()["reward"] == 1.0
    with locked_entry(tmp_path, {"scope": "TRAIN", "harness": "codex"}) as entry:
        assert entry.load() is None
    with locked_entry(tmp_path, {"scope": "TEST", "harness": "opencode"}) as entry:
        assert entry.load() is None


def test_shared_cache_rejects_infrastructure_failures(tmp_path, monkeypatch):
    monkeypatch.setenv("HAI_TERMINAL_BENCH_CACHE", str(tmp_path / "cache"))
    row = {
        "task_id": "task",
        "trial": 0,
        "reward": 0.0,
        "status": "error",
        "infrastructure_error": True,
    }

    with locked_entry(tmp_path, {"key": "failure"}) as entry:
        assert not entry.store(row)
        assert entry.load() is None


def test_terminal_batch_reuses_completed_trial_across_run_roots(tmp_path, monkeypatch):
    monkeypatch.setenv("HAI_TERMINAL_BENCH_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(tb, "_terminal_preflight", lambda task_ids: None)
    calls = []
    monkeypatch.setattr(
        tb,
        "_cache_manifest",
        lambda **kwargs: {
            "scope": kwargs["scope"],
            "harness": kwargs["harness"],
            "task_id": kwargs["task_id"],
            "pairing_slot": kwargs["pairing_slot"],
        },
    )

    def fake_trial(**kwargs):
        calls.append(kwargs)
        return {
            "task_id": kwargs["task_id"],
            "trial": kwargs["trial"],
            "pairing_slot": kwargs["pairing_slot"],
            "reward": 1.0,
            "status": "completed",
            "verifier_completed": True,
            "n_messages": 3,
            "n_tool_calls": 1,
        }

    monkeypatch.setattr(tb, "run_terminal_trial", fake_trial)
    common = {
        "repo_root": tmp_path,
        "scope": "TRAIN",
        "harness": "opencode",
        "harness_version": "v0",
        "harness_manifest": {},
        "task_repeats": {"task": 2},
        "pairing_offsets": {"task": 0},
        "max_concurrency": 2,
        "limits": tb.TerminalLimits(),
    }

    first = tb.run_terminal_batch(run_root=tmp_path / "method-a", **common)
    second = tb.run_terminal_batch(run_root=tmp_path / "versions-current", **common)

    assert len(calls) == 2
    assert first["metrics"]["shared_cache_hit_count"] == 0
    assert second["metrics"]["shared_cache_hit_count"] == 2


def test_terminal_batch_never_reuses_or_stores_test_scope(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("HAI_TERMINAL_BENCH_CACHE", str(cache_root))
    monkeypatch.setattr(tb, "_terminal_preflight", lambda task_ids: None)
    calls = []

    def fake_trial(**kwargs):
        calls.append(kwargs)
        return {
            "task_id": kwargs["task_id"],
            "trial": kwargs["trial"],
            "pairing_slot": kwargs["pairing_slot"],
            "reward": 1.0,
            "status": "completed",
            "verifier_completed": True,
            "n_messages": 3,
            "n_tool_calls": 1,
        }

    monkeypatch.setattr(tb, "run_terminal_trial", fake_trial)
    common = {
        "repo_root": tmp_path,
        "scope": "TEST",
        "harness": "opencode",
        "harness_version": "candidate",
        "harness_manifest": {},
        "task_repeats": {"task": 2},
        "pairing_offsets": {"task": 0},
        "max_concurrency": 2,
        "limits": tb.TerminalLimits(),
    }

    first = tb.run_terminal_batch(run_root=tmp_path / "test-a", **common)
    second = tb.run_terminal_batch(run_root=tmp_path / "test-b", **common)

    assert len(calls) == 4
    assert first["metrics"]["shared_cache_hit_count"] == 0
    assert second["metrics"]["shared_cache_hit_count"] == 0
    assert not cache_root.exists()


def test_terminal_batch_resumes_completed_trials_from_same_run_root(tmp_path, monkeypatch):
    monkeypatch.setattr(tb, "_terminal_preflight", lambda task_ids: None)
    calls = []

    def fake_trial(**kwargs):
        calls.append(kwargs)
        return {
            "task_id": kwargs["task_id"],
            "trial": kwargs["trial"],
            "pairing_slot": kwargs["pairing_slot"],
            "reward": 1.0,
            "status": "completed",
            "infrastructure_error": False,
            "verifier_completed": True,
            "runner": kwargs["harness"],
            "harness_version": kwargs["harness_version"],
            "runtime_schema": tb.RUNTIME_SCHEMA,
            "n_messages": 3,
            "n_tool_calls": 1,
        }

    monkeypatch.setattr(tb, "run_terminal_trial", fake_trial)
    common = {
        "repo_root": tmp_path,
        "run_root": tmp_path / "same-test-run",
        "scope": "TEST",
        "harness": "pi-agent",
        "harness_version": "v0",
        "harness_manifest": {},
        "task_repeats": {"task": 2},
        "pairing_offsets": {"task": 0},
        "max_concurrency": 2,
        "limits": tb.TerminalLimits(),
    }

    first = tb.run_terminal_batch(**common)
    second = tb.run_terminal_batch(**common)

    assert len(calls) == 2
    assert first["metrics"]["resumed_trial_count"] == 0
    assert second["metrics"]["resumed_trial_count"] == 2
    assert second["metrics"]["requested_trial_count"] == 2


def test_terminal_batch_does_not_resume_old_runtime_schema(tmp_path, monkeypatch):
    trajectory_root = tmp_path / "trajectories"
    path = trajectory_root / "task" / "trial_0000.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "task_id": "task",
                "trial": 0,
                "pairing_slot": 0,
                "status": "completed",
                "infrastructure_error": False,
                "verifier_completed": True,
                "runner": "pi",
                "harness_version": "v0",
                "runtime_schema": "harnesslens.terminal-bench-docker.legacy",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert tb._load_existing_trial_row(
        trajectory_root=trajectory_root,
        task_id="task",
        local_trial=0,
        pairing_slot=0,
        harness="pi",
        harness_version="v0",
    ) is None


def test_verifier_timeout_is_a_scored_failure_without_infrastructure_retry():
    reward, timed_out = tb._verifier_outcome(
        "test session starts\ncollected 1 item\ncommand timed out after 1800s",
        124,
    )

    assert reward == 0.0
    assert timed_out is True


def test_verifier_setup_timeout_remains_retryable_infrastructure_failure():
    with pytest.raises(RuntimeError, match="verifier setup timeout"):
        tb._verifier_outcome("installing verifier dependencies", 124)


def test_verifier_setup_502_remains_retryable_infrastructure_failure():
    with pytest.raises(RuntimeError, match="verifier infrastructure failure"):
        tb._verifier_outcome("apt-get failed: 502 Bad Gateway", 100)


def test_pi_agent_alias_is_supported():
    assert tb.normalize_harness("pi-agent") == "pi"


def test_codex_disables_view_image_only_for_video_processing():
    assert tb._disabled_codex_tools("video-processing") == ("view_image",)
    assert tb._disabled_codex_tools("code-from-image") == ()
    assert tb._disabled_codex_tools("path-tracing") == ()


def test_prepare_pi_uses_compact_sdk_runner(tmp_path, monkeypatch):
    copied = []
    setup_commands = []
    monkeypatch.setattr(
        tb,
        "_dexec",
        lambda cid, command, **kwargs: setup_commands.append(command),
    )
    monkeypatch.setattr(
        tb,
        "_dcp_in",
        lambda source, cid, destination: copied.append((source, cid, destination)),
    )
    monkeypatch.setenv(
        "DEEPSEEK_BASE_URL", "https://relay.example.invalid/v1"
    )

    command = tb._prepare_pi(
        "container",
        tmp_path,
        "solve the task",
        "system guidance",
        tb.TerminalLimits(max_steps=7),
        {"config_patch": {"compaction.enabled": False}},
    )

    assert "/opt/harness/pi_compact_runner.mjs" in command
    assert "--max-steps 7" in command
    assert "--mode json" not in command
    assert any("mkdir -p /app " in setup for setup in setup_commands)
    assert (tmp_path / "pi-prompt.txt").read_text(encoding="utf-8") == "solve the task"
    assert (tmp_path / "pi-system-prompt.txt").read_text(encoding="utf-8") == "system guidance"
    settings = json.loads((tmp_path / "pi-settings.json").read_text())
    assert settings["compaction"]["enabled"] is False
    models = json.loads((tmp_path / "pi-models.json").read_text())
    assert models["providers"]["deepseek"]["baseUrl"] == (
        "https://relay.example.invalid/v1"
    )
    assert {destination for _, _, destination in copied} >= {
        "/tmp/harness-home/.pi/agent/models.json",
        "/tmp/harness-home/pi-prompt.txt",
        "/tmp/harness-home/pi-system-prompt.txt",
    }


def test_pi_compact_runner_loads_project_append_system_prompt():
    source = (Path(tb.__file__).with_name(tb.PI_RUNNER_FILENAME)).read_text(
        encoding="utf-8"
    )

    assert '`${cwd}/.pi/APPEND_SYSTEM.md`' in source
    assert "appendSystemPromptOverride: () => appendSystemPrompts" in source
    assert "appendSystemPromptOverride: () => []" not in source
    assert "configureHttpDispatcher();" in source


def test_empty_runner_path_override_is_ignored(monkeypatch):
    monkeypatch.delenv("TB_PI_NODE_MODULES", raising=False)

    assert tb._configured_paths("TB_PI_NODE_MODULES") == []


def test_harness_file_content_is_runner_visible():
    _, system = tb._render_prompts(
        "finish the task",
        {"files": [{"path": ".codex/skills/check/SKILL.md", "content": "verify the result"}]},
    )

    assert "Active harness files" in system
    assert "verify the result" in system


def test_candidate_instructions_never_modify_terminal_user_task():
    user, system = tb._render_prompts(
        "finish the task",
        {"instructions": ["verify before finishing"]},
    )

    assert user == "finish the task"
    assert "verify before finishing" in system


def test_terminal_runtime_load_report_captures_candidate_instruction(tmp_path):
    output_root = tmp_path / "trial"
    project = output_root / "candidate-workspace" / "project"
    instruction = project / ".opencode" / "instructions" / "early-output.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_text("Deliver a working output early.\n", encoding="utf-8")
    (project / "opencode.json").write_text(
        json.dumps({"instructions": [".opencode/instructions/early-output.md"]}),
        encoding="utf-8",
    )
    manifest = tb.normalize_manifest(
        {
            "_workspace": {
                "files": [
                    {"scope": "project", "path": "opencode.json"},
                    {
                        "scope": "project",
                        "path": ".opencode/instructions/early-output.md",
                    },
                ]
            }
        }
    )

    report = tb._terminal_runtime_load_report(
        output_root=output_root,
        harness="opencode",
        manifest=manifest,
    )

    assert report["schema"] == "harnesslens.channel-load-report.v1"
    assert report["effective_config"]["instructions"] == [
        ".opencode/instructions/early-output.md"
    ]
    assert {item["path"] for item in report["candidate_project_files"]} == {
        "opencode.json",
        ".opencode/instructions/early-output.md",
    }
