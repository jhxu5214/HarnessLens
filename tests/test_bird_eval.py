from __future__ import annotations

import io
import json
import random
import shutil
import sqlite3
import tomllib
from pathlib import Path

import harnesslens.benchmarks.bird_eval as bird_eval
from harnesslens.benchmarks.benchmark_splits import load_benchmark_split
from harnesslens.benchmarks.bird_eval import (
    BirdLimits,
    _bird_mcp_socket_path,
    _cleanup_trial_runtime,
    _opencode_config,
    _prepare_harness,
    _write_pi_bird_project,
    extract_sql,
    load_bird_tasks,
    normalize_bird_harness,
)
from harnesslens.benchmarks.bird_mcp_server import BirdMCPServer
from harnesslens.benchmarks.bird_sql import execute_readonly_sql, grade_execution_accuracy
from harnesslens.benchmarks.cell_config import benchmark_config
from harnesslens.evaluation.rollout_bridge import (
    BIRD_CELL,
    RolloutRequest,
    TrainRolloutService,
    CellHarnessRepository,
)
from harnesslens.benchmarks.task_data import BaselineDataset, benchmark_task_explorer_input
from harnesslens.evaluation.blind_test_eval import runtime_limits


REPO_ROOT = Path(__file__).resolve().parents[1]
BIRD_COMMIT = "b3d4bcbbae9a96934ad812551eb400c7a3b23c12"


def test_bird_pi_logging_proxy_receives_provider_key_out_of_band(
    tmp_path, monkeypatch
):
    """The proxy inherits the key; argv would expose it to every user via `ps`."""
    captured = {}

    class Process:
        stdout = io.StringIO("PORT=4321\n")
        stderr = io.StringIO()

    def fake_popen(command, **kwargs):
        captured.update(command=command, kwargs=kwargs)
        return Process()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(bird_eval.subprocess, "Popen", fake_popen)

    _process, port = bird_eval._start_chat_logging_proxy(
        tmp_path,
        harness="pi-agent",
    )

    assert "--key" not in captured["command"]
    assert "test-key" not in captured["command"]
    assert captured["kwargs"]["env"]["DEEPSEEK_API_KEY"] == "test-key"
    assert port == 4321


def test_bird_paired_rollouts_use_distinct_mcp_sockets(tmp_path):
    common = {"worker_slot": 0, "question_id": 1028, "trial": 0}

    parent = _bird_mcp_socket_path(
        trajectory_root=tmp_path / "parent-v0" / "trajectories", **common
    )
    candidate = _bird_mcp_socket_path(
        trajectory_root=tmp_path / "candidate-01" / "trajectories", **common
    )

    assert parent != candidate
    assert len(str(parent).encode()) < 108
    assert len(str(candidate).encode()) < 108


def test_bird_turn_dispatch_passes_manifest_only_to_pi(tmp_path, monkeypatch):
    calls = {}

    def fake_opencode(**kwargs):
        calls["opencode"] = kwargs
        return object()

    def fake_pi(**kwargs):
        calls["pi"] = kwargs
        return object()

    monkeypatch.setattr(bird_eval, "_run_opencode_turn", fake_opencode)
    monkeypatch.setattr(bird_eval, "_run_pi_turn", fake_pi)
    common = {
        "trial_root": tmp_path / "trial",
        "workspace": tmp_path / "workspace",
        "socket_path": tmp_path / "bird.sock",
        "prompt": "question",
        "timeout_s": 10,
        "proxy_port": None,
        "manifest": {"prompt_appends": ["candidate"]},
    }

    bird_eval._run_harness_turn(harness="opencode", **common)
    bird_eval._run_harness_turn(harness="pi-agent", **common)

    assert "manifest" not in calls["opencode"]
    assert calls["pi"]["manifest"] == common["manifest"]


def test_bird_challenging_split_is_the_hard_coded_seed_42_sample():
    split = load_benchmark_split("bird-mini-dev-challenging")
    source = (
        REPO_ROOT
        / "third_party"
        / "bird-mini-dev"
        / "finetuning"
        / "inference"
        / "mini_dev_prompt.jsonl"
    )
    rows = [
        json.loads(line) for line in source.read_text().splitlines() if line.strip()
    ]
    population = [
        f"bird_{row['question_id']}"
        for row in rows
        if row.get("difficulty") == "challenging"
    ]
    sampled = set(random.Random(42).sample(population, 30))

    assert list(split.train) == [
        task_id for task_id in population if task_id in sampled
    ]
    assert list(split.test) == [
        task_id for task_id in population if task_id not in sampled
    ]
    assert (len(split.train), len(split.test)) == (30, 72)

    payload = json.loads(
        (
            REPO_ROOT
            / "configs"
            / "bird_mini_dev_challenging_split.json"
        ).read_text()
    )
    assert payload["seed"] == 42
    assert payload["difficulty"] == "challenging"
    assert payload["population_count"] == 102
    assert payload["source_commit"] == BIRD_COMMIT


def test_bird_alias_config_and_all_challenging_databases_are_available():
    assert load_benchmark_split("bird").benchmark == "bird-mini-dev-challenging"
    config = benchmark_config(REPO_ROOT, "bird_minidev")
    tasks = load_bird_tasks(REPO_ROOT)

    assert config.kind == "bird"
    assert config.outcome_authority == "authoritative"
    assert len(config.train_task_ids) == 30
    assert len(tasks) == 102
    assert all(task.database.is_file() for task in tasks.values())
    assert "harnesslens/benchmarks/bird_eval.py" in config.runtime_files()
    assert not set(config.train_task_ids) - set(tasks)


def test_bird_explorer_exposes_public_inputs_but_not_gold_sql():
    config = benchmark_config(REPO_ROOT, "bird")
    baseline = BaselineDataset(
        task_ids=config.train_task_ids,
        trajectory_paths=(),
        trajectories_by_task={},
        evidence_by_path={},
        source_event="test",
    )
    explorer = benchmark_task_explorer_input(
        repo_root=REPO_ROOT,
        baseline=baseline,
        cell="bird",
    )

    assert explorer["benchmark_kind"] == "bird"
    assert explorer["evaluation_contract"]["outcome_authority"] == "authoritative"
    assert len(explorer["tasks"]) == 30
    assert set(explorer["tasks"][0]["query"]) == {
        "instruction",
        "evidence",
        "schema",
    }
    assert "gold_sql" in explorer["forbidden_inputs"]
    assert "SQL" not in explorer["tasks"][0]


def _database(path: Path) -> Path:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE sample (value INTEGER)")
    connection.executemany("INSERT INTO sample VALUES (?)", [(1,), (2,), (2,)])
    connection.commit()
    connection.close()
    return path


def test_bird_sql_is_read_only_and_uses_official_set_equality(tmp_path):
    database = _database(tmp_path / "sample.sqlite")
    result = execute_readonly_sql(
        database,
        "-- inspect values\nSELECT value FROM sample ORDER BY value",
        timeout_s=2,
    )
    reordered = grade_execution_accuracy(
        database,
        "SELECT value FROM sample ORDER BY value DESC",
        "SELECT value FROM sample ORDER BY value",
        timeout_s=2,
    )
    different = grade_execution_accuracy(
        database,
        "SELECT DISTINCT value FROM sample",
        "SELECT value FROM sample",
        timeout_s=2,
    )

    assert result.rows == ((1,), (2,), (2,))
    assert reordered["passed"] is True
    # BIRD's official EX converts both result lists to sets, so duplicate count is ignored.
    assert different["passed"] is True
    try:
        execute_readonly_sql(database, "DELETE FROM sample", timeout_s=2)
    except ValueError as exc:
        assert "only SELECT or WITH" in str(exc)
    else:
        raise AssertionError("write statement was accepted")


def test_bird_mcp_executes_queries_records_calls_and_enforces_step_limit(tmp_path):
    database = _database(tmp_path / "sample.sqlite")
    log = tmp_path / "calls.json"
    server = BirdMCPServer(
        database=database,
        log_file=log,
        max_steps=1,
        tool_desc_patches={"execute_sql": {"desc": "Prefer explicit joins."}},
    )
    listed = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    first = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "execute_sql",
                "arguments": {"sql": "SELECT value FROM sample"},
            },
        }
    )
    second = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "execute_sql",
                "arguments": {"sql": "SELECT 1"},
            },
        }
    )

    assert "Prefer explicit joins" in listed["result"]["tools"][0]["description"]
    assert first["result"].get("isError") is not True
    assert second["result"]["isError"] is True
    assert json.loads(log.read_text())[0]["name"] == "execute_sql"


def test_bird_sql_extraction_handles_fenced_and_bare_queries():
    assert extract_sql("Result:\n```sql\nSELECT 1;\n```") == "SELECT 1;"
    assert extract_sql("WITH x AS (SELECT 1) SELECT * FROM x") == (
        "WITH x AS (SELECT 1) SELECT * FROM x"
    )
    assert extract_sql("I cannot answer") == ""


def test_bird_harness_configs_are_mcp_only_and_cover_all_runners(tmp_path):
    config = _opencode_config(
        socket_path=tmp_path / "bird.sock",
        system_prompt="bird system",
        max_steps=30,
        manifest={},
    )
    assert config["agent"]["build"]["steps"] == 30
    assert config["mcp"]["bird"]["enabled"] is True
    assert config["tools"]["bash"] is False
    assert config["tools"]["read"] is False
    assert config["tools"]["skill"] is True
    assert config["provider"]["deepseek"]["models"]["deepseek-v4-flash"][
        "limit"
    ] == {"context": 1_000_000, "output": 24_576}

    proxied = _opencode_config(
        socket_path=tmp_path / "bird.sock",
        system_prompt="bird system",
        max_steps=30,
        manifest={},
        proxy_port=19432,
    )
    assert proxied["provider"]["deepseek"]["options"]["baseURL"] == (
        "http://127.0.0.1:19432/v1"
    )

    for harness, proxy_port, expected in (
        ("opencode", None, "opencode.json"),
        ("codex", 1234, "codex_home/config.toml"),
    ):
        trial = tmp_path / harness
        trial.mkdir()
        _prepare_harness(
            harness=harness,
            trial_root=trial,
            socket_path=tmp_path / f"{harness}.sock",
            system_prompt="bird system",
            proxy_port=proxy_port,
            max_steps=30,
            manifest={},
        )
        assert (trial / expected).is_file()
        if harness == "codex":
            parsed = tomllib.loads((trial / expected).read_text())
            assert "projects" not in parsed
            shutil.rmtree(
                bird_eval._isolated_runtime_cwd(trial, "codex"),
                ignore_errors=True,
            )

    codex_trial = tmp_path / "codex-config"
    codex_trial.mkdir()
    _prepare_harness(
        harness="codex",
        trial_root=codex_trial,
        socket_path=tmp_path / "codex-config.sock",
        system_prompt="bird system",
        proxy_port=1234,
        max_steps=30,
        manifest={
            "config_patch": {
                "developer_instructions": "candidate instruction",
                "features.shell_tool": False,
            }
        },
    )
    codex_config = (codex_trial / "codex_home" / "config.toml").read_text()
    parsed_codex = tomllib.loads(codex_config)
    assert parsed_codex["developer_instructions"] == (
        "bird system\n\ncandidate instruction"
    )
    assert "shell_tool = false" in codex_config
    shutil.rmtree(
        bird_eval._isolated_runtime_cwd(codex_trial, "codex"),
        ignore_errors=True,
    )

    pi_runtime = tmp_path / "pi-runtime"
    pi_home = tmp_path / "pi-home"
    _write_pi_bird_project(
        runtime_cwd=pi_runtime,
        pi_home=pi_home,
        manifest={"config_patch": {"compaction.enabled": False}},
        proxy_port=19432,
    )
    pi_settings = json.loads((pi_runtime / ".pi" / "settings.json").read_text())
    assert pi_settings["compaction"]["enabled"] is False
    pi_models = json.loads((pi_home / "models.json").read_text())
    assert pi_models["providers"]["deepseek"]["baseUrl"] == (
        "http://127.0.0.1:19432/v1"
    )
    assert pi_models["providers"]["deepseek"]["api"] == "openai-completions"


def test_bird_native_candidate_configs_survive_while_runtime_invariants_win(tmp_path):
    opencode = _opencode_config(
        socket_path=tmp_path / "bird.sock",
        system_prompt="bird system",
        max_steps=30,
        manifest={},
        candidate_config={
            "model": "candidate-forbidden",
            "notice": "candidate-setting",
            "agent": {"build": {"steps": 999, "prompt": "candidate prompt"}},
            "mcp": {
                "candidate": {
                    "type": "local",
                    "enabled": True,
                    "command": ["printf", "ready"],
                },
                "bird": {"command": ["candidate-forbidden"]},
            },
        },
    )
    assert opencode["notice"] == "candidate-setting"
    assert opencode["model"] == "deepseek/deepseek-v4-flash"
    assert opencode["agent"]["build"] == {
        "steps": 30,
        "prompt": "bird system\n\ncandidate prompt",
    }
    assert opencode["mcp"]["candidate"]["command"] == ["printf", "ready"]
    assert opencode["mcp"]["bird"]["command"] != ["candidate-forbidden"]

    opencode_trial = tmp_path / "opencode-native"
    workspace = opencode_trial / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "candidate-guidance.md").write_text("candidate guidance\n")
    (workspace / "opencode.json").write_text(
        json.dumps({"instructions": ["candidate-guidance.md"]}),
        encoding="utf-8",
    )
    _prepare_harness(
        harness="opencode",
        trial_root=opencode_trial,
        socket_path=tmp_path / "opencode-native.sock",
        system_prompt="bird system",
        proxy_port=None,
        max_steps=30,
        manifest={},
    )
    prepared_opencode = json.loads((opencode_trial / "opencode.json").read_text())
    assert prepared_opencode["instructions"] == [
        str(workspace / "candidate-guidance.md")
    ]

    codex_trial = tmp_path / "codex-native"
    codex_trial.mkdir()
    _prepare_harness(
        harness="codex",
        trial_root=codex_trial,
        socket_path=tmp_path / "codex-native.sock",
        system_prompt="bird system",
        proxy_port=1234,
        max_steps=30,
        manifest={
            "_workspace": {
                "schema": 1,
                "files": [
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
                    }
                ],
            }
        },
    )
    codex = tomllib.loads((codex_trial / "codex_home" / "config.toml").read_text())
    assert codex["notice"] == "candidate-setting"
    assert codex["model"] == "gpt-5.4"
    assert codex["model_provider"] == "deepseek"
    assert "disable_response_storage" not in codex
    assert codex["developer_instructions"] == "bird system\n\ncandidate prompt"
    assert codex["mcp_servers"]["candidate"]["command"] == "printf"
    assert codex["mcp_servers"]["bird"]["command"]
    assert not (tmp_path / ".codex" / "config.toml").exists()
    shutil.rmtree(
        bird_eval._isolated_runtime_cwd(codex_trial, "codex"),
        ignore_errors=True,
    )

    pi_runtime = tmp_path / "pi-native-runtime"
    pi_home = tmp_path / "pi-native-home"
    _write_pi_bird_project(
        runtime_cwd=pi_runtime,
        pi_home=pi_home,
        manifest={
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
                            }
                        ),
                        "executable": False,
                    }
                ],
            }
        },
    )
    pi = json.loads((pi_runtime / ".pi" / "settings.json").read_text())
    assert pi["compaction"]["enabled"] is False
    assert pi["model"] == "deepseek-v4-flash"


def test_bird_codex_uses_isolated_runtime_without_moving_opencode(
    tmp_path, monkeypatch
):
    trial_root = tmp_path / "trial"
    workspace = trial_root / "workspace"
    workspace.mkdir(parents=True)
    isolated = bird_eval._isolated_runtime_cwd(trial_root, "codex")
    isolated.mkdir(parents=True)
    calls = []
    monkeypatch.setenv("DEEPSEEK_API_KEY", "key")
    monkeypatch.setattr(bird_eval, "opencode_binary", lambda: "/bin/opencode")
    monkeypatch.setattr(bird_eval.shutil, "which", lambda _: "/bin/codex")

    def fake_run(command, *, cwd, env, timeout_s):
        calls.append(Path(cwd))
        return "", "", 0

    monkeypatch.setattr(bird_eval, "run_process", fake_run)

    bird_eval._run_opencode_turn(
        trial_root=trial_root,
        workspace=workspace,
        prompt="task",
        timeout_s=10,
    )
    bird_eval._run_codex_turn(
        trial_root=trial_root,
        workspace=workspace,
        prompt="task",
        timeout_s=10,
    )

    assert calls == [workspace, isolated]
    shutil.rmtree(isolated, ignore_errors=True)


def test_bird_harness_aliases_and_runtime_limits():
    assert normalize_bird_harness("opencode") == "opencode"
    assert normalize_bird_harness("codex") == "codex"
    assert normalize_bird_harness("pi") == "pi-agent"
    limits = runtime_limits(load_benchmark_split("bird"))

    assert limits == {
        "repeats": 1,
        "max_concurrency": 20,
        "max_steps": 30,
        "max_rounds": 1,
        "turn_timeout_s": 600,
        "group_timeout_s": 14400,
        "query_timeout_s": 5,
        "grader_timeout_s": 30,
    }
    assert BirdLimits().max_rounds == 1


def test_bird_grader_emits_sanitized_shape_diagnostic_without_gold_values(tmp_path):
    database = tmp_path / "shape.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE values_table(value INTEGER, label TEXT)")
    connection.executemany(
        "INSERT INTO values_table VALUES (?, ?)",
        [(1, "secret-a"), (2, "secret-b")],
    )
    connection.commit()
    connection.close()

    grading = grade_execution_accuracy(
        database,
        "SELECT value FROM values_table WHERE value = 1",
        "SELECT value, label FROM values_table",
    )

    assert grading["passed"] is False
    assert grading["diagnostic"] == {
        "mismatch_type": "column_count_mismatch",
        "predicted_row_count": 1,
        "reference_row_count": 2,
        "predicted_column_count": 1,
        "reference_column_count": 2,
        "predicted_unique_row_count": 1,
        "reference_unique_row_count": 2,
        "duplicate_profile_mismatch": False,
    }
    assert "secret" not in json.dumps(grading["diagnostic"])


def test_bird_runtime_cleanup_preserves_trace_artifacts(tmp_path, monkeypatch):
    monkeypatch.delenv("HAI_KEEP_TRAJECTORY_WORKSPACE", raising=False)
    runtime_names = (
        "home",
        "xdg_config",
        "xdg_data",
        "xdg_cache",
        "tmp",
        "workspace",
        "pi_runtime",
        "pi_home",
        "codex_home",
    )
    for name in runtime_names:
        path = tmp_path / name
        path.mkdir()
        (path / "temporary").write_text("x", encoding="utf-8")
    trace = tmp_path / "harness_stdout.txt"
    trace.write_text("retained\n", encoding="utf-8")

    _cleanup_trial_runtime(tmp_path)

    assert trace.is_file()
    assert not any((tmp_path / name).exists() for name in runtime_names)


def test_bird_candidate_repository_merges_parent_channels(tmp_path):
    repository = CellHarnessRepository(
        cell=BIRD_CELL,
        repo_root=REPO_ROOT,
        run_id="bird-repository-test",
        evidence_root=tmp_path,
    )
    first = repository.materialize_candidate(
        base_version="v0",
        candidate_label="candidate-01",
        delta={
            "instructions": ["Check joins."],
            "tool_desc_patches": {"execute_sql": {"desc": "Validate SQL."}},
        },
    )
    second = repository.materialize_candidate(
        base_version=first,
        candidate_label="candidate-02",
        delta={
            "replace_channels": ["instructions"],
            "instructions": ["Check evidence."],
            "prompt_appends": ["Return one query."],
        },
    )

    snapshot = repository.read_candidate_snapshot(second)
    assert first == "candidate-01"
    assert second == "candidate-02"
    assert snapshot["instructions"] == ["Check evidence."]
    assert snapshot["prompt_appends"] == ["Return one query."]
    assert snapshot["tool_desc_patches"]["execute_sql"]["desc"] == "Validate SQL."
    meta = json.loads(
        (
            tmp_path
            / "bird-repository-test"
            / "versions_percell"
            / BIRD_CELL
            / second
            / "meta.json"
        ).read_text()
    )
    assert meta == {
        "harness": "opencode",
        "parent": "candidate-01",
        "status": "temporary",
        "temporary_candidate": True,
        "version": "candidate-02",
    }


def test_bird_rollout_bridge_forwards_pairing_and_caps(monkeypatch, tmp_path):
    seen = {}

    def fake_run_bird_batch(**kwargs):
        seen.update(kwargs)
        return {
            "trajectory_root": "unused",
            "per_task": {},
            "records": [],
            "metrics": {},
        }

    monkeypatch.setattr(
        "harnesslens.evaluation.rollout_bridge.run_bird_batch", fake_run_bird_batch
    )
    service = TrainRolloutService(
        cell=BIRD_CELL,
        repo_root=REPO_ROOT,
        run_id="bird-rollout-test",
        artifact_root=tmp_path / "artifacts",
        train_task_ids=["bird_1476"],
        initial_budget=2,
        evidence_root=tmp_path / "evidence",
        workspace_root=tmp_path / "workspaces",
        local_rootless_rollout=False,
    )
    request = RolloutRequest(
        request_id="request",
        run_id="bird-rollout-test",
        scope="TRAIN",
        harness_version="v0",
        task_repeats={"bird_1476": 2},
        max_concurrency=2,
        purpose="test",
        pairing_offsets={"bird_1476": 4},
    )

    service._run_group(request, ["bird_1476"], 2)

    assert seen["scope"] == "TRAIN"
    assert seen["task_repeats"] == {"bird_1476": 2}
    assert seen["pairing_offsets"] == {"bird_1476": 4}
    assert seen["max_concurrency"] == 2
    assert seen["limits"] == BirdLimits()
