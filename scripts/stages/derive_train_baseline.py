#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harnesslens.core.config import repo_root
from harnesslens.core.artifacts import write_json
from harnesslens.evolution.baseline import build_baseline_fingerprint
from harnesslens.benchmarks.cell_config import benchmark_cell, benchmark_config, supported_cell_help
from harnesslens.core.train_protocol import TRAIN_ROLLOUT_REPEATS


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Derive a lower-repeat TRAIN baseline from retained rollout trials."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cell", required=True, type=benchmark_cell, help=f"benchmark cell: {supported_cell_help()}")
    parser.add_argument("--harness", choices=("opencode", "pi", "codex"), default="opencode")
    parser.add_argument(
        "--repo-root", type=Path, default=repo_root()
    )
    args = parser.parse_args()
    if TRAIN_ROLLOUT_REPEATS != 1:
        raise ValueError(
            "derive_train_baseline currently requires HAI_TRAIN_ROLLOUT_REPEATS=1"
        )

    root = args.repo_root.resolve()
    source = args.source.resolve()
    config = benchmark_config(root, args.cell)
    payload = derive_single_repeat_baseline(
        source_payload=json.loads(source.read_text(encoding="utf-8")),
        source_path=source,
        task_ids=config.train_task_ids,
        fingerprint=build_baseline_fingerprint(
            root, cell=config.cell, harness=args.harness
        ),
    )
    output = write_json(args.output.resolve(), payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "source": str(source),
                "cell": config.cell,
                "task_count": len(config.train_task_ids),
                "trial_count": payload["metrics"]["trial_count"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def derive_single_repeat_baseline(
    *,
    source_payload: Mapping[str, Any],
    source_path: Path,
    task_ids: tuple[str, ...],
    fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    entry = source_payload.get("agent_workspace_entry")
    if not isinstance(entry, Mapping):
        raise ValueError("source baseline has no agent_workspace_entry")
    artifacts = entry.get("trajectory_artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("source baseline has no trajectory artifacts")

    grouped: dict[str, list[tuple[int, int, dict[str, Any], dict[str, Any]]]] = {}
    for raw_artifact in artifacts:
        artifact = dict(raw_artifact or {})
        path = Path(str(artifact.get("path") or "")).resolve()
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ]
        if len(rows) != 1 or not isinstance(rows[0], Mapping):
            raise ValueError(f"source trajectory must contain one trial: {path}")
        row = dict(rows[0])
        task_id = str(row.get("task_id") or "")
        grouped.setdefault(task_id, []).append(
            (
                int(row.get("pairing_slot", row.get("trial", 0)) or 0),
                int(row.get("trial", 0) or 0),
                artifact,
                row,
            )
        )

    expected = set(task_ids)
    if set(grouped) != expected:
        raise ValueError("source baseline task IDs differ from the configured TRAIN split")
    selected = {
        task_id: min(grouped[task_id], key=lambda item: (item[0], item[1]))
        for task_id in task_ids
    }
    selected_artifacts = [selected[task_id][2] for task_id in task_ids]
    selected_rows = [selected[task_id][3] for task_id in task_ids]
    records = []
    per_task = {}
    for task_id, artifact, row in (
        (task_id, selected[task_id][2], selected[task_id][3]) for task_id in task_ids
    ):
        path = str(Path(str(artifact["path"])).resolve())
        reward = float(row.get("reward", 0.0) or 0.0)
        error = str(row.get("error") or "")
        summary = {
            "trial": row.get("trial", 0),
            "pairing_slot": row.get("pairing_slot", row.get("trial", 0)),
            "simulation_seed": row.get("simulation_seed"),
            "reward": reward,
            "n_messages": int(row.get("n_messages") or len(row.get("messages") or [])),
            "n_tool_calls": int(row.get("n_tool_calls") or row.get("n_total_calls") or 0),
            "termination": str(row.get("termination") or row.get("stop_reason") or ""),
            "error": error[:300],
        }
        errors = [{"trial": row.get("trial", 0), "error": error}] if error else []
        record = {
            "task_id": task_id,
            "rewards": [reward],
            "harness_version": "v0",
            "trajectory_paths": [path],
            "worker_errors": errors,
            "trial_summaries": [summary],
            "pass_at_1": int(reward >= 1.0),
            "pass_at_2": int(reward >= 1.0),
        }
        records.append(record)
        per_task[task_id] = {
            "repeats": 1,
            "rewards": [reward],
            "trajectory_paths": [path],
            "worker_errors": errors,
            "trial_summaries": [summary],
        }

    successes = sum(float(row.get("reward", 0.0) or 0.0) >= 1.0 for row in selected_rows)
    infrastructure = sum(bool(row.get("infrastructure_error")) for row in selected_rows)
    metrics = {
        "task_count": len(task_ids),
        "trial_count": len(task_ids),
        "trial_success_count": successes,
        "trial_success_rate": successes / len(task_ids),
        "pass_at_1": successes / len(task_ids),
        "pass_at_2": successes / len(task_ids),
        "worker_error_count": sum(len(record["worker_errors"]) for record in records),
        "requested_trial_count": len(task_ids),
        "infrastructure_failure_count": infrastructure,
        "charged_trial_count": len(task_ids) - infrastructure,
        "trajectory_retention": str(
            (source_payload.get("metrics") or {}).get("trajectory_retention")
            or "retained_source_trial_jsonl"
        ),
        "api_trace_required": bool(
            (source_payload.get("metrics") or {}).get("api_trace_required")
        ),
        "workspace_cleanup_enabled": True,
    }
    paths = [str(Path(str(artifact["path"])).resolve()) for artifact in selected_artifacts]
    evidence_ids = [str(artifact["evidence_id"]) for artifact in selected_artifacts]
    derived_entry = {
        "label": "harnesslens_initial_train_baseline_derived_repeat_1",
        "scope": "TRAIN",
        "harness_version": "v0",
        "budget_spent": len(task_ids),
        "budget_remaining": 0,
        "metrics": metrics,
        "trajectory_artifacts": selected_artifacts,
        "trajectory_paths": paths,
        "evidence_ids": evidence_ids,
        "derived_from_source_event": str(source_path),
    }
    return {
        "request_id": f"derived-repeat-1-{source_path.parent.parent.name}",
        "scope": "TRAIN",
        "harness_version": "v0",
        "budget_spent": len(task_ids),
        "budget_remaining": 0,
        "trajectory_root": str(Path(paths[0]).parents[2]) if paths else "",
        "metrics": metrics,
        "per_task": per_task,
        "records": records,
        "agent_workspace_entry": derived_entry,
        "baseline_creation_accounting": {
            "creation_count": len(task_ids),
            "ledger_field": "baseline_used",
            "additional_job_charge": 0,
            "execution": "reused_pairing_slot_0_from_retained_baseline",
        },
        "baseline_fingerprint": dict(fingerprint),
        "derived_from": {
            "source_event": str(source_path),
            "selection": "minimum pairing_slot then minimum trial per task",
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
