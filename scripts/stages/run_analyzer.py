#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harnesslens.evolution.analyzer import AnalyzerModule
from harnesslens.core.budget import CreationBudget
from harnesslens.core.config import load_repo_env, repo_root
from harnesslens.core.train_protocol import TRAIN_BASELINE_CREATIONS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--harness", choices=("opencode", "pi", "codex"), default="opencode")
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    args = parser.parse_args()
    root = args.repo_root.resolve()
    load_repo_env(root)
    run_root = root / "runs" / "train" / args.run_id
    budget = CreationBudget(
        run_root / "creation_budget.json", baseline_used=TRAIN_BASELINE_CREATIONS
    )
    result = AnalyzerModule(
        run_root=run_root, budget=budget, harness=args.harness
    ).run(label=args.label)
    print(json.dumps({
        "reusable": result.reusable_path,
        "adjustment": result.adjustment_path,
        "budget": budget.status(),
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
