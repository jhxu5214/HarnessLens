from __future__ import annotations

import fcntl
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def provider_trial_concurrency() -> int:
    try:
        count = int(
            os.environ.get("HAI_PROVIDER_TRIAL_MAX_CONCURRENCY", "20")
        )
    except ValueError:
        count = 20
    return max(1, min(count, 64))


@contextmanager
def provider_trial_slot(*, timeout_s: float = 21600.0) -> Iterator[dict[str, float | int]]:
    """Limit active model-consuming trials across independent E2E processes."""
    root = Path(
        os.environ.get(
            "HAI_PROVIDER_TRIAL_SLOT_DIR",
            "/tmp/harnesslens-provider-trial-slots",
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    count = provider_trial_concurrency()
    started = time.monotonic()
    deadline = started + float(timeout_s)
    offset = (os.getpid() + threading.get_ident()) % count
    descriptor: int | None = None
    slot = -1
    while time.monotonic() < deadline:
        for step in range(count):
            index = (offset + step) % count
            candidate = os.open(
                root / f"slot-{index:02d}.lock",
                os.O_RDWR | os.O_CREAT,
                0o600,
            )
            try:
                fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(candidate)
                continue
            descriptor = candidate
            slot = index
            break
        if descriptor is not None:
            break
        time.sleep(0.1)
    if descriptor is None:
        raise TimeoutError("timed out waiting for provider trial capacity")
    try:
        yield {
            "slot": slot,
            "waited_s": round(time.monotonic() - started, 3),
            "max_concurrency": count,
        }
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
