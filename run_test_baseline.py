#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from harnesslens.benchmarks.benchmark_splits import (
    load_benchmark_split,
    supported_test_benchmarks,
)
from harnesslens.infrastructure.rootless_docker import DEFAULT_DOCKER_HOST
from harnesslens.core.config import load_repo_env, repo_root
from harnesslens.evaluation.blind_test_eval import (
    BASELINE_MAX_CONCURRENCY,
    BASELINE_REPEATS,
    run_test_baseline,
    runtime_limits,
    split_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an additive TEST baseline without entering the controller."
    )
    parser.add_argument("--benchmark", required=True, choices=supported_test_benchmarks())
    parser.add_argument("--harness", choices=("opencode", "pi", "codex"), default="opencode")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--retrieval-config", default="bm25")
    parser.add_argument("--repeats", type=int, choices=(1, 2), default=BASELINE_REPEATS)
    parser.add_argument(
        "--task-id",
        action="append",
        help="Run pass@k for one or more canonical TEST tasks; omit for the full TEST split.",
    )
    parser.add_argument("--docker-host", default=DEFAULT_DOCKER_HOST)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    args = parser.parse_args()
    root = args.repo_root.resolve()
    load_repo_env(root)
    split = load_benchmark_split(args.benchmark)
    if split.cell == "terminal_bench":
        os.environ["DOCKER_HOST"] = args.docker_host
    if args.dry_run:
        print(
            json.dumps(
                {
                    **split_summary(split),
                    "scope": "TEST",
                    "harness": args.harness,
                    "repeats": args.repeats,
                    "trial_count": len(args.task_id or split.test) * args.repeats,
                    "selected_test_task_ids": args.task_id or list(split.test),
                    "max_concurrency": runtime_limits(split, repeats=args.repeats)["max_concurrency"],
                    "runtime_limits": runtime_limits(split, repeats=args.repeats),
                    "docker_host": (
                        os.environ.get("DOCKER_HOST", "")
                        if split.cell == "terminal_bench"
                        else ""
                    ),
                    "model_network": "direct",
                    "retrieval_config": (
                        args.retrieval_config
                        if split.cell == "banking_knowledge"
                        else None
                    ),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = run_test_baseline(
        repo_root=root,
        run_id=args.run_id,
        benchmark=args.benchmark,
        retrieval_config=args.retrieval_config,
        task_ids=args.task_id,
        repeats=args.repeats,
        harness=args.harness,
    )
    print(
        json.dumps(
            {
                "benchmark": result.benchmark,
                "harness": args.harness,
                "output": result.output_path,
                "metrics": result.response.metrics,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
