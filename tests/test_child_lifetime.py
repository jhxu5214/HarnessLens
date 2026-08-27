"""The orphan watchdog is verified against real processes and a real SIGKILL.

Mocking os.getppid would prove nothing: the behaviour under test is what the
kernel does to a child when its parent is killed without warning.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from harnesslens.infrastructure.child_lifetime import exit_when_orphaned


REPO_ROOT = Path(__file__).resolve().parents[1]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie answers signal 0; treat it as gone.
    try:
        status = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return False
    return ") Z" not in status[:200]


def test_watchdog_declines_when_already_orphaned(monkeypatch):
    monkeypatch.setattr(os, "getppid", lambda: 1)
    assert exit_when_orphaned() is None


def test_watchdog_starts_a_daemon_thread():
    thread = exit_when_orphaned(interval_s=0.5)
    assert thread is not None
    assert thread.daemon is True


@pytest.mark.skipif(
    not Path("/proc").is_dir(), reason="needs /proc to observe process state"
)
def test_grandchild_exits_after_its_parent_is_sigkilled(tmp_path):
    """Parent spawns a watched grandchild, then dies without running handlers."""
    pid_file = tmp_path / "grandchild.pid"
    grandchild = tmp_path / "grandchild.py"
    grandchild.write_text(
        textwrap.dedent(
            f"""
            import os, sys, time
            sys.path.insert(0, {str(REPO_ROOT)!r})
            from harnesslens.infrastructure.child_lifetime import exit_when_orphaned
            exit_when_orphaned(interval_s=0.5)
            open({str(pid_file)!r}, "w").write(str(os.getpid()))
            time.sleep(120)
            """
        ),
        encoding="utf-8",
    )
    parent = tmp_path / "parent.py"
    parent.write_text(
        textwrap.dedent(
            f"""
            import subprocess, sys, time
            # start_new_session mirrors how the orchestrator spawns helpers, so
            # the grandchild survives a process-group signal.
            subprocess.Popen(
                [sys.executable, {str(grandchild)!r}], start_new_session=True
            )
            time.sleep(120)
            """
        ),
        encoding="utf-8",
    )

    process = subprocess.Popen([sys.executable, str(parent)])
    try:
        deadline = time.time() + 30
        while time.time() < deadline and not pid_file.is_file():
            time.sleep(0.2)
        assert pid_file.is_file(), "grandchild never reported its pid"
        grandchild_pid = int(pid_file.read_text().strip())
        assert _alive(grandchild_pid)

        os.kill(process.pid, signal.SIGKILL)
        process.wait(timeout=10)

        deadline = time.time() + 30
        while time.time() < deadline and _alive(grandchild_pid):
            time.sleep(0.5)
        assert not _alive(grandchild_pid), (
            f"grandchild {grandchild_pid} outlived its SIGKILLed parent"
        )
    finally:
        if process.poll() is None:
            process.kill()
        if pid_file.is_file():
            try:
                os.kill(int(pid_file.read_text().strip()), signal.SIGKILL)
            except (OSError, ValueError):
                pass
