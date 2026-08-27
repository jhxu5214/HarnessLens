#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harnesslens.benchmarks.cell_config import benchmark_cell, supported_cell_help
from harnesslens.core.budget import CreationBudget
from harnesslens.core.config import load_repo_env, repo_root
from harnesslens.evolution.main_agent import MainAgentModule
from harnesslens.core.train_protocol import TRAIN_BASELINE_CREATIONS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tested-main-decision", required=True, type=Path)
    parser.add_argument("--post-label", required=True)
    parser.add_argument("--rollout-output", required=True, type=Path)
    parser.add_argument("--label", default="final-submission")
    parser.add_argument("--base-version", default="v0")
    parser.add_argument("--cell", default="retail", type=benchmark_cell, help=f"benchmark cell: {supported_cell_help()}")
    parser.add_argument("--harness", choices=("opencode", "pi", "codex"), default="opencode")
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    args = parser.parse_args()
    root = args.repo_root.resolve()
    load_repo_env(root)
    run_root = root / "runs" / "train" / args.run_id
    budget = CreationBudget(
        run_root / "creation_budget.json", baseline_used=TRAIN_BASELINE_CREATIONS
    )
    result = MainAgentModule(
        repo_root=root,
        run_root=run_root,
        budget=budget,
        harness=args.harness,
        cell=args.cell,
    ).finalize(
        tested_main_decision=args.tested_main_decision,
        post_label=args.post_label,
        rollout_output=args.rollout_output,
        label=args.label,
        base_version=args.base_version,
        publish=not args.no_publish,
    )
    print(json.dumps({
        "output": result.output_path,
        "selected_version": result.harness_version,
        "budget": budget.status(),
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
