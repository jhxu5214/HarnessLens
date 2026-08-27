from __future__ import annotations

import os


def _train_rollout_repeats() -> int:
    raw = str(os.environ.get("HAI_TRAIN_ROLLOUT_REPEATS", "2")).strip()
    try:
        repeats = int(raw)
    except ValueError as exc:
        raise ValueError("HAI_TRAIN_ROLLOUT_REPEATS must be an integer") from exc
    if repeats not in {1, 2}:
        raise ValueError("HAI_TRAIN_ROLLOUT_REPEATS must be 1 or 2")
    return repeats


TRAIN_ROLLOUT_REPEATS = _train_rollout_repeats()
TRAIN_BASELINE_CREATIONS = 30 * TRAIN_ROLLOUT_REPEATS


def _max_rollout_concurrency() -> int:
    raw = str(os.environ.get("HAI_MAX_ROLLOUT_CONCURRENCY", "20")).strip()
    try:
        concurrency = int(raw)
    except ValueError as exc:
        raise ValueError("HAI_MAX_ROLLOUT_CONCURRENCY must be an integer") from exc
    if not 1 <= concurrency <= 30:
        raise ValueError("HAI_MAX_ROLLOUT_CONCURRENCY must be between 1 and 30")
    return concurrency


MAX_ROLLOUT_CONCURRENCY = _max_rollout_concurrency()
