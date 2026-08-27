from __future__ import annotations

import json
import os
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from harnesslens.benchmarks.cell_config import benchmark_config
from harnesslens.core.train_protocol import TRAIN_ROLLOUT_REPEATS


@dataclass(frozen=True)
class BaselineDataset:
    task_ids: tuple[str, ...]
    trajectory_paths: tuple[str, ...]
    trajectories_by_task: Mapping[str, tuple[str, ...]]
    evidence_by_path: Mapping[str, str]
    source_event: str

    @classmethod
    def from_ingest_event(cls, path: str | Path) -> "BaselineDataset":
        source = Path(path).resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        entry = payload.get("agent_workspace_entry")
        if not isinstance(entry, Mapping):
            raise ValueError("baseline event is missing agent_workspace_entry")
        artifacts = entry.get("trajectory_artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError("baseline event has no retained trajectories")
        grouped: dict[str, list[str]] = {}
        evidence_by_path: dict[str, str] = {}
        ordered_paths: list[str] = []
        for artifact in artifacts:
            trajectory_path = Path(str((artifact or {}).get("path") or ""))
            if not trajectory_path.is_file():
                raise ValueError(f"baseline trajectory is unavailable: {trajectory_path}")
            rows = [
                json.loads(line)
                for line in trajectory_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                if line.strip()
            ]
            if len(rows) != 1 or not isinstance(rows[0], Mapping):
                raise ValueError(f"baseline artifact must contain one trial: {trajectory_path}")
            task_id = str(rows[0].get("task_id") or "").strip()
            if not task_id:
                raise ValueError(f"baseline trajectory has no task_id: {trajectory_path}")
            resolved = str(trajectory_path.resolve())
            grouped.setdefault(task_id, []).append(resolved)
            evidence_id = str((artifact or {}).get("evidence_id") or "").strip()
            if not evidence_id:
                raise ValueError(f"baseline artifact has no evidence_id: {trajectory_path}")
            evidence_by_path[resolved] = evidence_id
            ordered_paths.append(resolved)
        fingerprint = payload.get("baseline_fingerprint") or {}
        expected_task_ids = tuple(
            str(item) for item in (fingerprint.get("task_ids") or ())
        )
        expected_task_count = len(expected_task_ids) if expected_task_ids else 30
        if len(grouped) != expected_task_count or any(
            len(paths) != TRAIN_ROLLOUT_REPEATS for paths in grouped.values()
        ):
            raise ValueError(
                "compatible baseline must contain "
                f"{expected_task_count} tasks x {TRAIN_ROLLOUT_REPEATS} trials"
            )
        if expected_task_ids and set(grouped) != set(expected_task_ids):
            raise ValueError("baseline task IDs do not match its fingerprint")
        return cls(
            task_ids=tuple(grouped),
            trajectory_paths=tuple(ordered_paths),
            trajectories_by_task={key: tuple(value) for key, value in grouped.items()},
            evidence_by_path=evidence_by_path,
            source_event=str(source),
        )


def benchmark_task_explorer_input(
    *,
    repo_root: str | Path,
    baseline: BaselineDataset,
    cell: str,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config = benchmark_config(root, cell)
    if config.kind == "tau2":
        payload = _tau2_task_explorer_input(root, baseline, cell=config.cell)
    elif config.kind == "terminal_bench":
        payload = _terminal_task_explorer_input(root, baseline, cell=config.cell)
    elif config.kind == "bird":
        payload = _bird_task_explorer_input(root, baseline, cell=config.cell)
    else:
        raise ValueError(f"unsupported benchmark kind: {config.kind}")
    payload["evaluation_contract"] = evaluation_contract(config.outcome_authority)
    return payload


def evaluation_contract(outcome_authority: str) -> dict[str, str]:
    authority = str(outcome_authority)
    if authority == "authoritative":
        interpretation = (
            "Recorded pass/fail is authoritative for task completion. Trajectories "
            "explain the mechanism but cannot relabel a failed trial as successful."
        )
    elif authority == "behavioral":
        interpretation = (
            "Recorded pass/fail is supporting whole-task metadata. Preserve explicit "
            "visible completion disagreements for downstream review."
        )
    else:
        raise ValueError(f"unsupported outcome authority: {authority}")
    return {
        "outcome_authority": authority,
        "interpretation": interpretation,
    }


def retail_task_explorer_input(
    *,
    repo_root: str | Path,
    baseline: BaselineDataset,
) -> dict[str, Any]:
    return benchmark_task_explorer_input(
        repo_root=repo_root,
        baseline=baseline,
        cell="retail",
    )


def _tau2_task_explorer_input(
    root: Path,
    baseline: BaselineDataset,
    *,
    cell: str,
) -> dict[str, Any]:
    tasks_path = root / "third_party" / "tau3-bench" / "data" / "tau2" / "domains" / cell / "tasks.json"
    policy_path = tasks_path.parent / "policy.md"
    raw_tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    selected = set(baseline.task_ids)
    tasks: list[dict[str, Any]] = []
    for raw in raw_tasks:
        task_id = str((raw or {}).get("id") or "")
        if task_id not in selected:
            continue
        scenario = (raw or {}).get("user_scenario") or {}
        tasks.append(
            {
                "task_id": task_id,
                "query": _normalized_query(scenario.get("instructions")),
            }
        )
    by_id = {item["task_id"]: item for item in tasks}
    missing = sorted(selected - set(by_id))
    if missing:
        raise ValueError(f"TRAIN task definitions are missing: {missing}")
    ordered_tasks = [by_id[task_id] for task_id in baseline.task_ids]
    return {
        "scope": "TRAIN",
        "domain": cell,
        "benchmark_kind": "tau2",
        "tasks": ordered_tasks,
        "environment": {
            "policy": _tau2_public_policy_summary(tasks_path.parent, cell=cell),
            "tools": _tau2_tool_definitions(root, cell=cell),
            "tool_transport": {
                "kind": "mcp",
                "server_id": "tau2",
                "model_tool_prefix": "tau2_",
                "schema_patch_contract": (
                    "patch_descs[raw_tool_name].desc and "
                    "patch_descs[raw_tool_name].params[parameter] are appended by "
                    "tau2_mcp_server before tools/list reaches the rollout agent"
                ),
            },
        },
        "forbidden_inputs": [
            "trajectory",
            "reward",
            "evaluation_criteria",
            "grader",
            "reference_answer",
            "task_workspace",
        ],
    }


def _normalized_query(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        return {"instruction": value}
    if value is None:
        return {}
    return {"instruction": str(value)}


def _tau2_public_policy_summary(domain_root: Path, *, cell: str) -> str:
    policy_path = domain_root / "policy.md"
    if policy_path.is_file():
        return policy_path.read_text(encoding="utf-8")
    documents = sorted((domain_root / "documents").glob("*.json"))
    db_path = domain_root / "db.json"
    return (
        f"{cell} is a tau2 domain without a single policy.md file. Public task "
        f"knowledge is provided through the domain database"
        f"{' and ' + str(len(documents)) + ' retrieval documents' if documents else ''}. "
        "Task exploration should classify user goals from TRAIN queries and public "
        "tool definitions; it must not inspect trajectories, rewards, graders, or "
        "hidden workspace state."
        f" Database file present: {db_path.is_file()}."
    )


def _terminal_task_explorer_input(
    root: Path,
    baseline: BaselineDataset,
    *,
    cell: str,
) -> dict[str, Any]:
    selected = set(baseline.task_ids)
    tasks: list[dict[str, Any]] = []
    for task_id in baseline.task_ids:
        if task_id not in selected:
            continue
        task_root = (
            root
            / "third_party"
            / "terminal-bench"
            / "original-tasks"
            / task_id
        )
        payload = _read_terminal_task_mapping(task_root, task_id=task_id)
        tasks.append(
            {
                "task_id": task_id,
                "query": {
                    "instruction": str(
                        payload.get("instruction")
                        or payload.get("description")
                        or ""
                    ),
                    "metadata": {
                        key: value
                        for key, value in payload.items()
                        if key not in {"instruction", "description", "solution"}
                    },
                },
            }
        )
    return {
        "scope": "TRAIN",
        "domain": cell,
        "benchmark_kind": "terminal_bench",
        "tasks": tasks,
        "environment": {
            "policy": (
                "Terminal-Bench task: the agent sees the task instruction and a "
                "fresh task workspace/container. Success is determined by hidden "
                "task verification; do not inspect solution files or test answers."
            ),
            "tools": [
                {
                    "name": "bash",
                    "description": "Run shell commands in the task workspace/container.",
                    "parameters": {"command": "string"},
                },
                {
                    "name": "edit",
                    "description": "Create or modify files in the task workspace.",
                    "parameters": {"path": "string", "content": "string"},
                },
            ],
            "tool_transport": {
                "kind": "workspace_shell",
                "server_id": "terminal_bench",
                "schema_patch_contract": (
                    "Terminal-Bench does not expose MCP tau2 tool descriptions; "
                    "harness changes should target prompts, instructions, skills, "
                    "or terminal runtime guidance."
                ),
            },
        },
        "forbidden_inputs": [
            "trajectory",
            "reward",
            "evaluation_criteria",
            "grader",
            "reference_answer",
            "solution",
            "task_workspace",
        ],
    }


def _read_terminal_task_mapping(task_root: Path, *, task_id: str) -> dict[str, Any]:
    task_yaml = task_root / "task.yaml"
    if task_yaml.is_file():
        return _read_yaml_mapping(task_yaml)

    task_toml = task_root / "task.toml"
    instruction_md = task_root / "instruction.md"
    if task_toml.is_file() and instruction_md.is_file():
        payload = dict(tomllib.loads(task_toml.read_text(encoding="utf-8")))
        payload["instruction"] = instruction_md.read_text(encoding="utf-8")
        return payload

    raise ValueError(f"TRAIN task definition is missing: {task_id}")


def _bird_task_explorer_input(
    root: Path,
    baseline: BaselineDataset,
    *,
    cell: str,
) -> dict[str, Any]:
    from harnesslens.benchmarks.bird_eval import load_bird_tasks

    available = load_bird_tasks(root)
    missing = sorted(set(baseline.task_ids) - set(available))
    if missing:
        raise ValueError(f"TRAIN task definitions are missing: {missing}")
    tasks = [
        {
            "task_id": task_id,
            "query": {
                "instruction": available[task_id].question,
                "evidence": available[task_id].evidence,
                "schema": available[task_id].schema,
            },
        }
        for task_id in baseline.task_ids
    ]
    return {
        "scope": "TRAIN",
        "domain": cell,
        "benchmark_kind": "bird",
        "tasks": tasks,
        "environment": {
            "policy": (
                "BIRD Mini-Dev text-to-SQL task: generate one read-only SQLite "
                "SELECT/WITH query from the question, evidence, and public schema."
            ),
            "tools": [
                {
                    "name": "execute_sql",
                    "description": "Execute a read-only SQLite SELECT/WITH query.",
                    "parameters": {"sql": "string"},
                }
            ],
            "tool_transport": {
                "kind": "mcp",
                "server_id": "bird",
                "model_tool_prefix": "bird_",
                "schema_patch_contract": (
                    "tool_desc_patches.execute_sql.desc and params.sql are "
                    "applied before tools/list reaches the rollout agent"
                ),
            },
        },
        "forbidden_inputs": [
            "trajectory",
            "reward",
            "gold_sql",
            "grader",
            "reference_answer",
            "task_workspace",
        ],
    }



def _read_yaml_mapping(path: Path) -> Mapping[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"task yaml is not a mapping: {path}")
    return payload


def _tau2_tool_definitions(repo_root: Path, *, cell: str) -> list[dict[str, Any]]:
    python = repo_root / "third_party" / "tau3-bench" / ".venv" / "bin" / "python"
    code = r"""
import importlib
import json
import sys

cell = sys.argv[1]
module = importlib.import_module(f"tau2.domains.{cell}.environment")
get_environment = module.get_environment
if cell == "banking_knowledge":
    environment = get_environment(retrieval_variant="bm25")
else:
    environment = get_environment()
tools = []
for tool in environment.get_tools():
    function = dict(tool.openai_schema.get("function") or {})
    tools.append({
        "name": function.get("name"),
        "description": function.get("description", ""),
        "parameters": function.get("parameters", {}),
    })
print(json.dumps(tools, ensure_ascii=False))
"""
    env = {
        "PYTHONPATH": str(repo_root / "third_party" / "tau3-bench" / "src"),
        "TAU2_DATA_DIR": str(repo_root / "third_party" / "tau3-bench" / "data"),
        "HAI_TAU2_RETRIEVAL_CONFIG": os.environ.get("HAI_TAU2_RETRIEVAL_CONFIG", "bm25"),
    }

    proc = subprocess.run(
        [str(python), "-c", code, cell],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, **env},
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"failed to load {cell} public tool definitions: {proc.stderr[-2000:]}")
    for line in reversed(proc.stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, Mapping)]
    raise RuntimeError(f"{cell} environment loader did not return tool definitions")
