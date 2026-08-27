import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import harnesslens.harnesses.native_intelligent_runtime as native_runtime
from harnesslens.core.budget import CreationBudget
from harnesslens.harnesses.native_intelligent_runtime import NativeIntelligentAdapter
from harnesslens.harnesses.opencode_runtime import write_opencode_config
from harnesslens.core.profiles import power_profile
from harnesslens.harnesses.runner import (
    IntelligentHarnessRunner,
    _inline_controller_read_files,
    intelligent_stdout_path,
    parse_json_object,
)


def test_codex_read_role_inlines_only_run_owned_files(tmp_path):
    run_root = tmp_path / "run"
    bundle = run_root / "experience" / "bundle.json"
    bundle.parent.mkdir(parents=True)
    bundle.write_text('{"task":"visible"}\n', encoding="utf-8")

    payload = _inline_controller_read_files(
        {
            "task_bundle_paths": [str(bundle)],
            "instruction": "Use the read tool to inspect every task bundle path.",
        },
        allowed_root=run_root,
    )

    assert payload["controller_file_transport"]["mode"] == "inline"
    assert payload["controller_inlined_files"] == [
        {"path": str(bundle.resolve()), "content": '{"task":"visible"}\n'}
    ]
    assert "do not call file tools" in payload["instruction"]

    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    import pytest

    with pytest.raises(ValueError, match="outside the run root"):
        _inline_controller_read_files(
            {"source_index_path": str(outside)}, allowed_root=run_root
        )


def test_codex_native_runner_streams_large_prompt_over_stdin(tmp_path, monkeypatch):
    adapter = NativeIntelligentAdapter(
        harness="codex",
        model="deepseek-v4-flash",
        context_limit=65_536,
        output_limit=24_576,
        max_steps=60,
        timeout_s=30,
        workspace_root=tmp_path / "jobs",
        allowed_builtin_tools=("read",),
    )
    run_root = tmp_path / "jobs" / "large-prompt"

    class FinishedProxy:
        def poll(self):
            return 0

    def start_proxy(_usage_log, trace):
        trace.write_text("{}\n", encoding="utf-8")
        return FinishedProxy(), 1234

    captured = {}

    class FakeProcess:
        returncode = 0

        def communicate(self, *, input, timeout):
            captured["input"] = input
            captured["timeout"] = timeout
            (run_root / "last_message.txt").write_text(
                '{"final": true}', encoding="utf-8"
            )
            return json.dumps({"type": "turn.completed", "usage": {}}), ""

    def popen(command, **kwargs):
        captured["command"] = command
        captured["stdin"] = kwargs.get("stdin")
        captured["stdout"] = kwargs.get("stdout")
        captured["stderr"] = kwargs.get("stderr")
        return FakeProcess()

    monkeypatch.setattr(native_runtime, "_start_responses_proxy", start_proxy)
    monkeypatch.setattr(native_runtime.shutil, "which", lambda _name: "/usr/bin/codex")
    monkeypatch.setattr(subprocess, "Popen", popen)
    prompt = "x" * 3_000_000

    result = adapter.run(
        prompt=prompt,
        workspace=run_root,
        call_id="large-prompt",
        budget=CreationBudget(tmp_path / "budget.json", total=1, baseline_used=0),
    )

    assert captured["command"][-1] == "-"
    assert "--strict-config" in captured["command"]
    assert prompt not in captured["command"]
    assert captured["stdin"] is subprocess.PIPE
    assert captured["stdout"] is not subprocess.PIPE
    assert captured["stderr"] is not subprocess.PIPE
    assert captured["stdout"].closed
    assert captured["stderr"].closed
    assert captured["input"] == prompt
    assert Path(result.stdout_path).read_text(encoding="utf-8") == '{"final": true}'
    assert (run_root / "codex.events.jsonl").is_file()
    config = (run_root / ".codex_home" / "config.toml").read_text(encoding="utf-8")
    assert "shell_tool = false" in config
    assert "default_tools_enabled" not in config
    assert "disable_response_storage" not in config


def test_pi_native_runner_streams_large_prompt_over_stdin(tmp_path, monkeypatch):
    adapter = NativeIntelligentAdapter(
        harness="pi",
        model="deepseek/deepseek-v4-flash",
        context_limit=65_536,
        output_limit=24_576,
        max_steps=60,
        timeout_s=30,
        workspace_root=tmp_path / "jobs",
        allowed_builtin_tools=(),
    )
    run_root = tmp_path / "jobs" / "large-prompt"

    class FinishedProxy:
        def poll(self):
            return 0

    def start_proxy(trace, *, call_id):
        del call_id
        trace.write_text("{}\n", encoding="utf-8")
        return FinishedProxy(), 1234

    captured = {}

    class CapturedStdin(__import__("io").StringIO):
        def close(self):
            captured["input"] = self.getvalue()
            super().close()

    class FakeProcess:
        returncode = 0

        def __init__(self):
            self.stdin = CapturedStdin()
            self.stdout = __import__("io").StringIO('{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"{\\"final\\":true}"}],"usage":{"totalTokens":1}}}\n')

        def wait(self, *, timeout):
            captured["timeout"] = timeout
            return 0

    def popen(command, **kwargs):
        captured["command"] = command
        captured["stdin"] = kwargs.get("stdin")
        captured["stdout"] = kwargs.get("stdout")
        captured["stderr"] = kwargs.get("stderr")
        return FakeProcess()

    monkeypatch.setattr(native_runtime, "_start_chat_proxy", start_proxy)
    monkeypatch.setattr(native_runtime, "_pi_binary", lambda: Path("/usr/bin/pi"))
    monkeypatch.setattr(subprocess, "Popen", popen)
    prompt = "x" * 3_000_000

    result = adapter.run(
        prompt=prompt,
        workspace=run_root,
        call_id="large-prompt",
        budget=CreationBudget(tmp_path / "budget.json", total=1, baseline_used=0),
    )

    assert prompt not in captured["command"]
    assert captured["stdin"] is subprocess.PIPE
    assert captured["stdout"] is subprocess.PIPE
    assert captured["stderr"] is not subprocess.PIPE
    assert captured["stderr"].closed
    assert captured["input"] == prompt
    assert parse_json_object(Path(result.stdout_path).read_text(encoding="utf-8")) == {
        "final": True
    }
    invocation = json.loads((run_root / "invocation.json").read_text(encoding="utf-8"))
    assert invocation["prompt_transport"] == "stdin"


def test_native_stop_reaps_after_forced_kill(monkeypatch):
    signals = []

    class StubbornProcess:
        pid = 1234

        def __init__(self):
            self.wait_calls = 0

        def poll(self):
            return None

        def wait(self, *, timeout):
            assert timeout == 5
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise subprocess.TimeoutExpired("stubborn", timeout)
            return 0

    process = StubbornProcess()
    monkeypatch.setattr(
        native_runtime.os,
        "killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )

    native_runtime._stop(process)

    assert process.wait_calls == 2
    assert signals == [
        (1234, native_runtime.signal.SIGTERM),
        (1234, native_runtime.signal.SIGKILL),
    ]


def test_pi_native_runner_persists_only_final_assistant_event(tmp_path, monkeypatch):
    adapter = NativeIntelligentAdapter(
        harness="pi",
        model="deepseek-v4-flash",
        context_limit=65_536,
        output_limit=24_576,
        max_steps=60,
        timeout_s=30,
        workspace_root=tmp_path / "jobs",
        allowed_builtin_tools=(),
    )
    run_root = tmp_path / "jobs" / "compact-stdout"

    class FinishedProxy:
        def poll(self):
            return 0

    def start_proxy(trace, *, call_id):
        del call_id
        trace.write_text("{}\n", encoding="utf-8")
        return FinishedProxy(), 1234

    final_event = {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": '{"final":true}'}],
            "usage": {"totalTokens": 12, "input": 7, "output": 5},
        },
    }

    class FakeProcess:
        returncode = 0

        def __init__(self):
            import io

            self.stdin = io.StringIO()
            update = json.dumps(
                {"type": "message_update", "message": {"content": "x" * 1000}}
            )
            self.stdout = io.StringIO(
                "\n".join([update] * 3000 + [json.dumps(final_event)]) + "\n"
            )

        def wait(self, *, timeout):
            del timeout
            return 0

    monkeypatch.setattr(native_runtime, "_start_chat_proxy", start_proxy)
    monkeypatch.setattr(native_runtime, "_pi_binary", lambda: Path("/usr/bin/pi"))
    monkeypatch.setattr(subprocess, "Popen", lambda _command, **_kwargs: FakeProcess())

    result = adapter.run(
        prompt="return json",
        workspace=run_root,
        call_id="compact-stdout",
        budget=CreationBudget(tmp_path / "budget.json", total=1, baseline_used=0),
    )

    persisted = Path(result.stdout_path).read_text(encoding="utf-8")
    assert len(persisted) < 1024
    assert json.loads(persisted) == final_event
    assert parse_json_object(persisted) == {"final": True}
    assert result.usage["total"] == 12


def test_native_editor_tool_sets_are_bounded(tmp_path):
    pi = NativeIntelligentAdapter(
        harness="pi",
        model="deepseek-v4-flash",
        context_limit=100,
        output_limit=20,
        max_steps=2,
        timeout_s=30,
        workspace_root=tmp_path / "pi",
        allowed_builtin_tools=("read", "edit", "write", "grep", "find", "ls"),
    )
    codex = NativeIntelligentAdapter(
        harness="codex",
        model="deepseek-v4-flash",
        context_limit=100,
        output_limit=20,
        max_steps=2,
        timeout_s=30,
        workspace_root=tmp_path / "codex",
        allowed_builtin_tools=(),
    )

    assert "write" in pi.allowed_builtin_tools
    assert codex.allowed_builtin_tools == ()


def test_codex_editor_uses_bounded_workspace_mcp_without_shell(tmp_path, monkeypatch):
    adapter = NativeIntelligentAdapter(
        harness="codex",
        model="deepseek-v4-flash",
        context_limit=100,
        output_limit=20,
        max_steps=2,
        timeout_s=30,
        workspace_root=tmp_path / "jobs",
        allowed_builtin_tools=(),
    )
    run_root = tmp_path / "jobs" / "editor"
    candidate = run_root / "candidate"
    candidate.mkdir(parents=True)

    class FinishedProxy:
        def poll(self):
            return 0

    captured = {}

    def start_proxy(_usage_log, trace, **kwargs):
        captured.update(kwargs)
        trace.write_text("{}\n", encoding="utf-8")
        return FinishedProxy(), 1234

    monkeypatch.setattr(native_runtime, "_start_responses_proxy", start_proxy)
    monkeypatch.setattr(native_runtime.shutil, "which", lambda _name: "/usr/bin/codex")

    proxy, _command, _env, _trace, config_path = adapter._prepare_codex(
        run_root, "edit", editor_root=candidate
    )

    assert proxy.poll() == 0
    config = config_path.read_text(encoding="utf-8")
    assert "shell_tool = false" in config
    assert "default_tools_enabled" not in config
    assert "[mcp_servers.workspace_editor]" not in config
    assert captured["workspace_editor_root"] == candidate


def test_opencode_config_denies_external_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_URL", "https://example.invalid/v1")
    path = write_opencode_config(
        tmp_path,
        model="deepseek/deepseek-v4-flash",
        context_limit=100,
        output_limit=20,
        max_steps=2,
        allowed_builtin_tools=("read", "write"),
    )

    assert json.loads(path.read_text())["permission"]["external_directory"] == "deny"


def test_codex_recovery_prefers_the_final_message(tmp_path):
    (tmp_path / "codex.stdout").write_text("event stream", encoding="utf-8")
    final = tmp_path / "last_message.txt"
    final.write_text('{"ok": true}', encoding="utf-8")

    assert intelligent_stdout_path(tmp_path, "codex") == final


def test_parse_json_object_reads_opencode_text_event():
    stdout = json.dumps({"type": "text", "part": {"text": '{"ok":true,"items":[1]}'}})

    assert parse_json_object(stdout) == {"ok": True, "items": [1]}


def test_parse_json_object_recovers_one_unambiguous_fenced_object():
    stdout = json.dumps(
        {
            "type": "text",
            "part": {"text": 'Here is the result:\n```json\n{"ok": true}\n```'},
        }
    )

    assert parse_json_object(stdout) == {"ok": True}


def test_parse_json_object_does_not_choose_between_multiple_fences():
    stdout = json.dumps(
        {
            "type": "text",
            "part": {"text": '```json\n{"a": 1}\n```\n```json\n{"b": 2}\n```'},
        }
    )

    import pytest

    with pytest.raises(ValueError, match="one JSON object"):
        parse_json_object(stdout)


def test_parse_json_object_uses_final_text_after_tool_analysis():
    stdout = "\n".join(
        [
            json.dumps({"type": "text", "part": {"text": "Reading inputs."}}),
            json.dumps({"type": "tool_use", "part": {"tool": "read"}}),
            json.dumps({"type": "text", "part": {"text": '{"answer":2}'}}),
        ]
    )

    assert parse_json_object(stdout) == {"answer": 2}


def test_parse_json_object_accepts_one_object_after_brief_prose():
    stdout = json.dumps(
        {
            "type": "text",
            "part": {"text": 'Analysis complete.\n{"answer": 2, "items": [1]}'},
        }
    )

    assert parse_json_object(stdout) == {"answer": 2, "items": [1]}


def test_parse_json_object_rejects_two_embedded_objects():
    stdout = json.dumps(
        {
            "type": "text",
            "part": {"text": 'First {"answer": 1} then {"answer": 2}'},
        }
    )

    import pytest

    with pytest.raises(ValueError, match="one JSON object"):
        parse_json_object(stdout)


def test_parse_json_object_closes_only_balanced_truncated_suffix():
    truncated = '{"coverage":{"ids":["a"]},"preservation":{"decision":"accept"}'

    assert parse_json_object(truncated) == {
        "coverage": {"ids": ["a"]},
        "preservation": {"decision": "accept"},
    }


def test_parse_json_object_reads_pi_final_assistant_message():
    stdout = json.dumps(
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "x"},
                    {"type": "text", "text": '{"ok":true}'},
                ],
            },
        }
    )

    assert parse_json_object(stdout) == {"ok": True}


def test_parse_json_object_reads_codex_agent_message():
    stdout = json.dumps(
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": '{"ok":true}'},
        }
    )

    assert parse_json_object(stdout) == {"ok": True}


def test_opencode_runner_accepts_power_model_options(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_URL", "https://example.invalid/v1")
    path = write_opencode_config(
        tmp_path,
        model="deepseek/deepseek-v4-flash",
        context_limit=100,
        output_limit=20,
        max_steps=60,
        model_options={"reasoningEffort": "high"},
        allowed_builtin_tools=("read",),
    )

    config = json.loads(path.read_text())
    model = config["provider"]["deepseek"]["models"]["deepseek-v4-flash"]
    assert model["options"] == {"reasoningEffort": "high"}
    assert config["tools"]["read"] is True
    assert config["permission"]["read"] == "allow"


def test_intelligent_runner_retains_complete_interaction_manifest(
    tmp_path, monkeypatch
):
    class FakeAdapter:
        def __init__(self, **kwargs):
            assert kwargs["model"] == "deepseek/deepseek-v4-flash"

        def run(
            self, *, prompt, workspace, call_id, budget, max_steps, output_validator
        ):
            assert output_validator is None
            root = workspace
            stdout = root / "opencode.stdout"
            stderr = root / "opencode.stderr"
            api_trace = root / "api_calls.jsonl"
            (root / "opencode.json").write_text("{}", encoding="utf-8")
            (root / "invocation.json").write_text("{}", encoding="utf-8")
            stdout.write_text(
                json.dumps({"type": "text", "part": {"text": '{"ok":true}'}}),
                encoding="utf-8",
            )
            stderr.write_text("", encoding="utf-8")
            api_trace.write_text(
                json.dumps(
                    {
                        "request": {"messages": [{"role": "user", "content": prompt}]},
                        "response": {"choices": []},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return SimpleNamespace(
                outcome="completed",
                stdout_path=str(stdout),
                stderr_path=str(stderr),
                validation_error="",
                api_trace_path=str(api_trace),
            )

    monkeypatch.setattr(
        "harnesslens.harnesses.runner.OpenCodeIntelligentAdapter", FakeAdapter
    )
    budget = CreationBudget(tmp_path / "budget.json", total=120, baseline_used=60)
    runner = IntelligentHarnessRunner(
        profile=power_profile("opencode", max_steps=10),
        budget=budget,
        workspace_root=tmp_path / "jobs",
    )

    result = runner.run_json(
        job_id="task-discovery-01",
        system_prompt="Classify the tasks.",
        input_payload={"task_ids": ["0", "1"]},
    )

    manifest = json.loads(
        Path(result.interaction_manifest_path).read_text(encoding="utf-8")
    )
    assert manifest["schema"] == "harnesslens.opencode-interaction.v1"
    assert manifest["outcome"] == "completed"
    assert set(manifest["artifacts"]) == {
        "api_calls",
        "input",
        "invocation",
        "opencode_config",
        "stderr",
        "stdout",
        "submitted_prompt",
        "system_prompt",
    }
    assert all(item["sha256"] for item in manifest["artifacts"].values())
    assert (
        (tmp_path / "jobs" / "task-discovery-01" / "submitted_prompt.txt")
        .read_text(encoding="utf-8")
        .startswith("Classify the tasks.")
    )
    assert result.output == {"ok": True}
    assert json.loads(
        (tmp_path / "jobs" / "task-discovery-01" / "output.json").read_text(
            encoding="utf-8"
        )
    ) == {"ok": True}
