from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from harnesslens.core.artifacts import content_digest
from harnesslens.harnesses.harness_manifest import (
    empty_harness_manifest,
    normalize_harness,
    normalize_native_manifest,
)


def load_incumbent_candidate(
    submission_path: str | Path,
    *,
    cell: str,
    harness: str,
    train_task_ids: Sequence[str],
) -> dict[str, Any]:
    """Load an explicitly supplied, TRAIN-accepted submission as an initial champion."""

    source = Path(submission_path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    selected_version = str(payload.get("selected_version") or "")
    if not selected_version or selected_version == "v0":
        raise ValueError("incumbent submission must select a non-baseline version")
    raw_snapshot = Path(str(payload.get("snapshot_path") or ""))
    snapshot = (
        raw_snapshot.resolve()
        if raw_snapshot.is_absolute()
        else (source.parent / raw_snapshot).resolve()
    )
    if snapshot.name != selected_version:
        raise ValueError("incumbent snapshot does not match selected_version")
    if snapshot.parent.name != str(cell):
        raise ValueError(
            f"incumbent cell {snapshot.parent.name!r} does not match {str(cell)!r}"
        )

    normalized_harness = normalize_harness(harness)
    manifest = _read_snapshot_manifest(snapshot, normalized_harness)
    if manifest == empty_harness_manifest():
        raise ValueError("incumbent manifest is empty")

    history = payload.get("iteration_history") or []
    accepted = [
        dict(record)
        for record in history
        if isinstance(record, Mapping)
        and str(record.get("review_decision") or "") == "accept_delta"
    ]
    if not any(
        str(record.get("selected_version") or "") == selected_version
        for record in accepted
    ):
        raise ValueError("incumbent lacks TRAIN acceptance evidence for selected_version")

    allowed_tasks = {str(task_id) for task_id in train_task_ids}
    direct_task_ids = sorted(
        {
            str(task_id)
            for record in accepted
            for task_id in record.get("rollout_task_ids") or []
            if str(task_id) in allowed_tasks
        }
    )
    if not direct_task_ids:
        raise ValueError("incumbent acceptance evidence has no tasks in current TRAIN")

    channel_experiences: dict[str, set[str]] = {}
    for record in accepted:
        for diff in record.get("channel_diffs") or []:
            if not isinstance(diff, Mapping) or not str(diff.get("channel_id") or ""):
                continue
            channel_id = str(diff["channel_id"])
            channel_experiences.setdefault(channel_id, set()).update(
                str(item) for item in diff.get("experience_ids") or []
            )
    if not channel_experiences:
        raise ValueError("incumbent lacks accepted changed-channel provenance")

    evidence_fields = (
        "recovered_task_ids",
        "preserved_task_ids",
        "attributable_regression_task_ids",
    )
    prior_train_evidence = {
        field: sorted(
            {
                str(task_id)
                for record in accepted
                for task_id in (record.get("review_evidence") or {}).get(field) or []
                if str(task_id) in allowed_tasks
            }
        )
        for field in evidence_fields
    }
    digest = content_digest(
        {
            "cell": str(cell),
            "harness": normalized_harness,
            "manifest": manifest,
        }
    )[:12]
    return {
        "id": f"incumbent-{digest}",
        "objective": (
            "Revalidate a previously TRAIN-accepted cumulative harness manifest "
            "against the current evidence and runtime."
        ),
        "channel_plan": [
            {
                "channel_id": channel_id,
                "operation": "revalidate a previously TRAIN-accepted cumulative artifact",
                "experience_ids": sorted(experience_ids),
                "rationale": "Retest prior attributable TRAIN evidence under the current run.",
            }
            for channel_id, experience_ids in sorted(channel_experiences.items())
        ],
        "manifest_delta": manifest,
        "validation": {
            "local_behavior_checks": [
                "Reproduce prior attributed recoveries without attributable regression."
            ]
        },
        "_portfolio_side": "incumbent",
        "_direct_task_ids": direct_task_ids,
        "_prior_train_evidence": {
            **prior_train_evidence,
            "accepted_iteration_count": len(accepted),
            "submission_path": str(source),
            "selected_version": selected_version,
        },
    }


def _read_snapshot_manifest(snapshot: Path, harness: str) -> dict[str, Any]:
    harness_root = snapshot / "harness" / harness
    if harness == "opencode":
        patch_path = harness_root / "patch.json"
        if not patch_path.is_file():
            raise ValueError("incumbent opencode snapshot is missing patch.json")
        patch = json.loads(patch_path.read_text(encoding="utf-8"))
        descriptions_path = harness_root / "patch_descs.json"
        descriptions = (
            json.loads(descriptions_path.read_text(encoding="utf-8"))
            if descriptions_path.is_file()
            else {}
        )
        return normalize_native_manifest(
            {
                "config_patch": dict(patch.get("config_patch") or {}),
                "files": list(patch.get("files") or []),
                "instructions": list(patch.get("instructions") or []),
                "prompt_appends": list(patch.get("prompt_appends") or []),
                "tool_desc_patches": dict(descriptions or {}),
            }
        )
    manifest_path = harness_root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"incumbent {harness} snapshot is missing manifest.json")
    return normalize_native_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
