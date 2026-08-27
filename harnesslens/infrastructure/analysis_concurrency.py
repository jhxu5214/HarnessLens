from __future__ import annotations

import os


def max_analysis_concurrency() -> int:
    raw = str(os.environ.get("HAI_MAX_ANALYSIS_CONCURRENCY", "20")).strip()
    try:
        concurrency = int(raw)
    except ValueError as exc:
        raise ValueError(
            "HAI_MAX_ANALYSIS_CONCURRENCY must be an integer"
        ) from exc
    if not 1 <= concurrency <= 64:
        raise ValueError(
            "HAI_MAX_ANALYSIS_CONCURRENCY must be between 1 and 64"
        )
    return concurrency


def analysis_workers(job_count: int) -> int:
    count = int(job_count)
    if count < 1:
        raise ValueError("analysis worker pool requires at least one job")
    return min(max_analysis_concurrency(), count)
