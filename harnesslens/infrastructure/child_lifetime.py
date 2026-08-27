"""Make long-lived helper processes exit when their orchestrator disappears.

Every helper this package spawns — the provider proxies, the per-task MCP
servers — is a child of the orchestrator and is normally stopped in a ``finally``
block. That covers a clean exit and an exception, but not ``SIGKILL``: the
handlers never run, and because the children are started with
``start_new_session=True`` they are not in the orchestrator's process group
either, so nothing signals them. They become orphans.

The obvious fix, ``PR_SET_PDEATHSIG``, is wrong here. It is scoped to the
*thread* that forked the child, and these helpers are spawned from
``ThreadPoolExecutor`` workers; a worker retiring while the run continues would
kill a proxy that is still in use. Setting it through ``preexec_fn`` is also
documented as unsafe in a process with threads.

So the child watches instead. Being reparented to init is an unambiguous signal
that the orchestrator is gone, it costs one sleeping thread, and it cannot fire
early.
"""

from __future__ import annotations

import os
import signal
import threading


POLL_INTERVAL_S = 5.0


def exit_when_orphaned(
    *, interval_s: float = POLL_INTERVAL_S, signal_number: int = signal.SIGTERM
) -> threading.Thread | None:
    """Start a daemon thread that terminates this process once orphaned.

    Returns the thread, or None when the platform cannot report a parent (in
    which case the caller simply keeps the previous behaviour).
    """
    try:
        original_parent = os.getppid()
    except (AttributeError, OSError):
        return None
    # Already orphaned before we got started, e.g. a stale relaunch.
    if original_parent <= 1:
        return None

    def watch() -> None:
        while True:
            if _sleep_interruptibly(interval_s):
                return
            try:
                parent = os.getppid()
            except OSError:
                return
            if parent == original_parent:
                continue
            # Reparented: the orchestrator exited. Signal ourselves so any
            # SIGTERM handler this script installed still gets to run.
            try:
                os.kill(os.getpid(), signal_number)
            except OSError:
                os._exit(0)
            return

    thread = threading.Thread(
        target=watch, name="harnesslens-orphan-watchdog", daemon=True
    )
    thread.start()
    return thread


_stop = threading.Event()


def _sleep_interruptibly(interval_s: float) -> bool:
    """Sleep, returning True if the watchdog was asked to stop."""
    return _stop.wait(timeout=max(0.5, float(interval_s)))


def stop_watchdog() -> None:
    """Let a helper opt out again, mainly so tests do not leak threads."""
    _stop.set()


def reset_watchdog() -> None:
    _stop.clear()
