#!/usr/bin/env python3
"""Run one real task per (cell, harness) combination and print a matrix.

This is the smallest test that is still end to end: it goes through the same
TrainRolloutService dispatch a real run uses, so it exercises the benchmark
executor, the harness driver, the provider proxy and the trial-row writer.
Unit tests cannot cover any of that.

    python scripts/smoke_matrix.py --cell retail --cell bird --harness opencode

Costs one trial per combination. Terminal-Bench is excluded unless asked for by
name, because each of its tasks builds and runs a container.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harnesslens.benchmarks.benchmark_splits import load_benchmark_split  # noqa: E402
from harnesslens.benchmarks.cell_config import benchmark_config  # noqa: E402
from harnesslens.core.config import load_repo_env, repo_root  # noqa: E402
from harnesslens.evaluation.rollout_bridge import RolloutRequest, TrainRolloutService  # noqa: E402


def run_one(cell: str, harness: str, root: Path) -> dict:
    config = benchmark_config(root, cell)
    task_id = str(config.train_task_ids[0])
    started = time.time()
    with tempfile.TemporaryDirectory(prefix=f"hl-smoke-{cell}-{harness}-") as raw:
        work = Path(raw)
        service = TrainRolloutService(
            cell=config.cell,
            repo_root=root,
            run_id=f"smoke-{cell}-{harness}",
            artifact_root=work / "artifacts",
            train_task_ids=[task_id],
            initial_budget=1,
            evidence_root=work / "evidence",
            workspace_root=work / "workspaces",
            harness=harness,
            local_rootless_rollout=config.local_rootless_rollout,
        )
        request = RolloutRequest(
            request_id="smoke",
            run_id=f"smoke-{cell}-{harness}",
            scope="TRAIN",
            harness_version="v0",
            task_repeats={task_id: 1},
            max_concurrency=1,
            purpose="smoke",
            pairing_offsets={task_id: 0},
        )
        payload = service._run_group(request, [task_id], 1)
        metrics = payload.get("metrics") or {}
        rows = []
        for path in sorted(work.rglob("trial_*.jsonl")):
            line = path.read_text(encoding="utf-8").splitlines()
            if line:
                rows.append(json.loads(line[0]))
        error = next((str(r.get("error")) for r in rows if r.get("error")), "")
        return {
            "task": task_id,
            "trials": len(rows),
            "reward": [r.get("reward") for r in rows],
            "worker_errors": metrics.get("worker_error_count"),
            "seconds": round(time.time() - started),
            "error": error.splitlines()[0][:80] if error else "",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", action="append", default=[])
    parser.add_argument("--harness", action="append", default=[])
    args = parser.parse_args()
    cells = args.cell or ["retail", "banking", "bird"]
    harnesses = args.harness or ["opencode", "pi", "codex"]

    root = repo_root()
    load_repo_env(root)

    results = {}
    for cell in cells:
        for harness in harnesses:
            key = f"{cell} x {harness}"
            print(f"==> {key}", flush=True)
            try:
                results[key] = run_one(cell, harness, root)
            except Exception as exc:  # noqa: BLE001 — the report is the point
                results[key] = {"error": f"{type(exc).__name__}: {exc}"[:110]}
                traceback.print_exc(limit=3)
            print(f"    {results[key]}", flush=True)

    print("\n" + "=" * 78)
    print(f"{'combination':26s}{'trials':>7s}{'reward':>10s}{'errs':>6s}{'secs':>6s}  note")
    failed = 0
    for key, r in results.items():
        note = r.get("error", "")
        if note or r.get("worker_errors"):
            failed += 1
        print(
            f"{key:26s}{str(r.get('trials','-')):>7s}"
            f"{str(r.get('reward','-')):>10s}{str(r.get('worker_errors','-')):>6s}"
            f"{str(r.get('seconds','-')):>6s}  {note}"
        )
    print("=" * 78)
    print(f"{len(results) - failed}/{len(results)} combinations completed a real trial")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
