from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_RETAIL_TRAIN_TASK_IDS = (
    "0", "103", "105", "106", "107", "16", "20", "21", "23", "25",
    "29", "31", "34", "41", "44", "46", "59", "6", "63", "69",
    "7", "75", "8", "83", "84", "88", "91", "96", "98", "99",
)

BASELINE_RUNTIME_FILES = (
    "harnesslens/evaluation/rollout_bridge.py",
    "harnesslens/benchmarks/native_tau2_worker.py",
    "harnesslens/harnesses/native_candidate_runtime.py",
    "harnesslens/harnesses/harness_workspace.py",
    "harnesslens/harnesses/harness_manifest.py",
    "harnesslens/benchmarks/tau2_driver.py",
    "harnesslens/benchmarks/opencode_tau2.py",
    "harnesslens/benchmarks/pi_tau2.py",
    "harnesslens/benchmarks/codex_tau2.py",
    "harnesslens/infrastructure/chat_completions_proxy.py",
    "harnesslens/infrastructure/provider_capacity.py",
    "harnesslens/infrastructure/codex_responses_proxy.py",
    "harnesslens/benchmarks/tau2_mcp_server.py",
    "harnesslens/harnesses/tool_schema.py",
    "third_party/tau3-bench/data/tau2/domains/retail/tasks.json",
    "third_party/tau3-bench/data/tau2/domains/retail/policy.md",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    benchmark: str
    cell: str
    kind: str
    train_task_ids: tuple[str, ...]
    local_rootless_rollout: bool
    outcome_authority: str
    domain_files: tuple[str, ...] = ()

    def runtime_files(self) -> tuple[str, ...]:
        common = (
            "harnesslens/evaluation/rollout_bridge.py",
            "harnesslens/harnesses/native_candidate_runtime.py",
            "harnesslens/harnesses/harness_workspace.py",
        )
        if self.kind == "tau2":
            return (
                *common,
                "harnesslens/benchmarks/native_tau2_worker.py",
                "harnesslens/benchmarks/tau2_driver.py",
                "harnesslens/benchmarks/opencode_tau2.py",
                "harnesslens/benchmarks/pi_tau2.py",
                "harnesslens/benchmarks/codex_tau2.py",
                "harnesslens/infrastructure/chat_completions_proxy.py",
                "harnesslens/infrastructure/provider_capacity.py",
                "harnesslens/infrastructure/codex_responses_proxy.py",
                "harnesslens/benchmarks/tau2_mcp_server.py",
                "harnesslens/harnesses/tool_schema.py",
                *self.task_source_files(),
            )
        if self.kind == "terminal_bench":
            return (
                *common,
                "harnesslens/benchmarks/terminal_bench.py",
                "harnesslens/benchmarks/terminal_runtime_patch.py",
                "harnesslens/benchmarks/terminal_images.py",
                "harnesslens/infrastructure/rootless_docker.py",
                "harnesslens/infrastructure/provider_capacity.py",
                "configs/terminal_bench_split.json",
            )
        if self.kind == "bird":
            return (
                "harnesslens/benchmarks/bird_eval.py",
                "harnesslens/benchmarks/bird_sql.py",
                "harnesslens/benchmarks/bird_mcp_server.py",
                # BIRD drives the harness through the shared native driver, so
                # that file decides how a BIRD rollout behaves too.
                "harnesslens/harnesses/native_harness_driver.py",
                "configs/bird_mini_dev_challenging_split.json",
                *self.task_source_files(),
            )
        raise ValueError(f"unsupported benchmark kind: {self.kind}")

    def task_source_files(self) -> tuple[str, ...]:
        if self.kind == "tau2":
            return self.domain_files
        if self.kind == "terminal_bench":
            return self.domain_files
        if self.kind == "bird":
            return (
                "third_party/bird-mini-dev/finetuning/inference/mini_dev_prompt.jsonl",
            )
        raise ValueError(f"unsupported benchmark kind: {self.kind}")


_ALIASES = {
    "retail": "retail",
    "tau2-retail": "retail",
    "banking": "banking_knowledge",
    "banking_knowledge": "banking_knowledge",
    "tau2-banking_knowledge": "banking_knowledge",
    "terminal": "terminal_bench",
    "terminal_bench": "terminal_bench",
    "terminal-bench": "terminal_bench",
    "bird": "bird_mini_dev_challenging",
    "bird_minidev": "bird_mini_dev_challenging",
    "bird-mini-dev": "bird_mini_dev_challenging",
    "bird_mini_dev_challenging": "bird_mini_dev_challenging",
    "bird-mini-dev-challenging": "bird_mini_dev_challenging",
}


SUPPORTED_CELLS = ("retail", "banking", "terminal-bench", "bird")


def supported_cell_help() -> str:
    """One line naming the cells, for --cell help text and error messages."""
    return " | ".join(SUPPORTED_CELLS)


def normalize_benchmark_cell(cell: str) -> str:
    normalized = _ALIASES.get(str(cell).strip().lower())
    if normalized is None:
        raise ValueError(
            f"unknown benchmark cell {cell!r}; expected one of: "
            f"{supported_cell_help()}"
        )
    return normalized


def benchmark_cell(cell: str) -> str:
    """argparse ``type`` for --cell: accepts any alias, rejects typos at parse.

    A plain ``choices`` list would print all two dozen aliases in the usage
    line; this keeps them working while failing with a short, readable message.
    """
    import argparse

    try:
        normalize_benchmark_cell(cell)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return cell


def benchmark_config(repo_root: str | Path, cell: str) -> BenchmarkConfig:
    root = Path(repo_root).resolve()
    normalized = normalize_benchmark_cell(cell)
    if normalized == "retail":
        return BenchmarkConfig(
            benchmark="tau2-retail",
            cell="retail",
            kind="tau2",
            train_task_ids=DEFAULT_RETAIL_TRAIN_TASK_IDS,
            local_rootless_rollout=True,
            outcome_authority="behavioral",
            domain_files=(
                "third_party/tau3-bench/data/tau2/domains/retail/tasks.json",
                "third_party/tau3-bench/data/tau2/domains/retail/policy.md",
            ),
        )
    if normalized == "banking_knowledge":
        split = _load_split(root, "banking_knowledge_split.json")
        return BenchmarkConfig(
            benchmark="tau2-banking_knowledge",
            cell="banking_knowledge",
            kind="tau2",
            train_task_ids=tuple(str(item) for item in split["train"]),
            local_rootless_rollout=True,
            outcome_authority="behavioral",
            domain_files=_banking_domain_files(root),
        )
    if normalized == "terminal_bench":
        split = _load_split(root, "terminal_bench_split.json")
        train_task_ids = tuple(str(item) for item in split["train"])
        return BenchmarkConfig(
            benchmark="terminal-bench",
            cell="terminal_bench",
            kind="terminal_bench",
            train_task_ids=train_task_ids,
            local_rootless_rollout=False,
            outcome_authority="authoritative",
            domain_files=_terminal_domain_files(root, train_task_ids),
        )
    if normalized == "bird_mini_dev_challenging":
        split = _load_split(root, "bird_mini_dev_challenging_split.json")
        return BenchmarkConfig(
            benchmark="bird-mini-dev-challenging",
            cell="bird_mini_dev_challenging",
            kind="bird",
            train_task_ids=tuple(str(item) for item in split["train"]),
            local_rootless_rollout=False,
            outcome_authority="authoritative",
        )
    raise AssertionError(normalized)


def tau2_cell_config(repo_root: str | Path, cell: str) -> BenchmarkConfig:
    config = benchmark_config(repo_root, cell)
    if config.kind != "tau2":
        raise ValueError(f"not a tau2 cell: {cell}")
    return config


def _load_split(repo_root: Path, filename: str) -> dict[str, Sequence[str]]:
    candidates = (repo_root / "configs" / filename,)
    for path in candidates:
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            train = _validated_ids(payload.get("train"), label=f"{filename} TRAIN")
            test = _validated_ids(payload.get("test"), label=f"{filename} TEST")
            overlap = set(train) & set(test)
            if overlap:
                raise ValueError(f"{filename} split overlaps: {sorted(overlap)}")
            return {"train": train, "test": test}
    raise ValueError(f"split config is unavailable: {filename}")


def _banking_domain_files(repo_root: Path) -> tuple[str, ...]:
    domain = repo_root / "third_party" / "tau3-bench" / "data" / "tau2" / "domains" / "banking_knowledge"
    required = (
        "third_party/tau3-bench/data/tau2/domains/banking_knowledge/tasks.json",
        "third_party/tau3-bench/data/tau2/domains/banking_knowledge/db.json",
    )
    documents = tuple(
        str(path.relative_to(repo_root))
        for path in sorted((domain / "documents").glob("*.json"))
    )
    return (*required, *documents)


def _terminal_domain_files(
    repo_root: Path, task_ids: Sequence[str]
) -> tuple[str, ...]:
    files: list[str] = []
    for task_id in task_ids:
        task_root = (
            repo_root
            / "third_party"
            / "terminal-bench"
            / "original-tasks"
            / str(task_id)
        )
        if (task_root / "task.yaml").is_file():
            files.append(str((task_root / "task.yaml").relative_to(repo_root)))
        elif (task_root / "task.toml").is_file() and (
            task_root / "instruction.md"
        ).is_file():
            files.extend(
                str(path.relative_to(repo_root))
                for path in (task_root / "task.toml", task_root / "instruction.md")
            )
        else:
            raise ValueError(f"Terminal-Bench task definition is unavailable: {task_id}")
        if not (task_root / "run-tests.sh").is_file() and not (
            task_root / "tests" / "test.sh"
        ).is_file():
            fallback = (
                repo_root
                / "assets"
                / "terminal_task_assets"
                / str(task_id)
            )
            verifier_files = tuple(
                path
                for path in sorted(fallback.rglob("*"))
                if path.is_file() and "__pycache__" not in path.parts
            )
            if not verifier_files:
                raise ValueError(f"Terminal-Bench verifier is unavailable: {task_id}")
            files.extend(os.path.relpath(path, repo_root) for path in verifier_files)
    return tuple(files)


def _validated_ids(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} split is empty or malformed")
    normalized = tuple(str(item) for item in value)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} split contains duplicate task IDs")
    return normalized
