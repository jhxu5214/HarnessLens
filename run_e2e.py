#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from harnesslens.benchmarks.cell_config import benchmark_cell, benchmark_config, supported_cell_help
from harnesslens.infrastructure.clash_proxy import configure_terminal_clash_proxy
from harnesslens.core.config import load_repo_env, repo_root
from harnesslens.evolution.controller import IterationController


def configure_cell_runtime(repo_root: Path, cell: str) -> None:
    if benchmark_config(repo_root, cell).kind == "terminal_bench":
        configure_terminal_clash_proxy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-event", type=Path)
    parser.add_argument(
        "--analysis-source-run",
        help=(
            "Reuse discovery, baseline Experience, and baseline Analyzer artifacts "
            "from a run with the exact same baseline trajectories."
        ),
    )
    parser.add_argument(
        "--promotion-metric",
        choices=("pass_at_1", "pass_at_2"),
        default="pass_at_1",
        help="Primary paired metric that a candidate must strictly improve.",
    )
    parser.add_argument(
        "--incumbent-submission",
        type=Path,
        action="append",
        default=[],
        help=(
            "Prior TRAIN-accepted submission/final.json to revalidate as a normal "
            "candidate; may be repeated."
        ),
    )
    parser.add_argument("--harness", choices=("opencode", "pi", "codex"), default="opencode")
    parser.add_argument(
        "--cell",
        default="retail", type=benchmark_cell, help=f"benchmark cell: {supported_cell_help()}")
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    args = parser.parse_args()
    root = args.repo_root.resolve()
    load_repo_env(root)
    configure_cell_runtime(root, args.cell)
    result = IterationController(
        repo_root=root,
        run_id=args.run_id,
        harness=args.harness,
        cell=args.cell,
        incumbent_submissions=tuple(args.incumbent_submission),
        promotion_metric=args.promotion_metric,
        analysis_source_run=args.analysis_source_run,
    ).run(baseline_event=args.baseline_event)
    print(json.dumps({
        "selected_version": result.selected_version,
        "submission": result.submission_path,
        "budget": result.budget,
        "cell": args.cell,
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
