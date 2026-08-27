"""Reclaim task containers that outlived the run which created them.

Terminal-Bench brings a container up per task per trial and tears it down in a
``finally`` block. That covers a clean exit and an exception but not ``SIGKILL``
— and unlike a stray child process, a container is not in the orchestrator's
process tree, so nothing else reaps it either. One killed run left 132
containers behind, all ``Exited (137)``, holding 57 GB.

So each container is stamped with the pid that created it and the boot it was
created in. A container is an orphan when that pid is gone, or when the machine
has rebooted since. Anything a live run owns is left alone, which is why this
can run unattended at the start of every batch.

Images are never touched.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


OWNER_PID_LABEL = "ai.harnesslens.owner-pid"
BOOT_ID_LABEL = "ai.harnesslens.boot-id"
BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")


def _docker() -> str:
    return os.environ.get("TB_DOCKER") or shutil.which("docker") or "docker"


def current_boot_id() -> str:
    try:
        return BOOT_ID_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def ownership_labels() -> dict[str, str]:
    """Labels to stamp onto every container this process brings up."""
    return {OWNER_PID_LABEL: str(os.getpid()), BOOT_ID_LABEL: current_boot_id()}


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, OverflowError):
        return False
    return True


def _run(command: list[str], timeout_s: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, capture_output=True, text=True, timeout=timeout_s, check=False
    )


def orphaned_containers(*, timeout_s: int = 60) -> list[dict[str, str]]:
    """Containers stamped by this project whose creating run is gone."""
    listed = _run(
        [_docker(), "ps", "-a", "--filter", f"label={OWNER_PID_LABEL}", "--format", "{{.ID}}"],
        timeout_s,
    )
    if listed.returncode != 0:
        return []
    ids = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
    if not ids:
        return []

    inspected = _run([_docker(), "inspect", *ids], timeout_s)
    if inspected.returncode != 0:
        return []
    try:
        payload = json.loads(inspected.stdout)
    except json.JSONDecodeError:
        return []

    boot = current_boot_id()
    orphans = []
    for entry in payload:
        labels = ((entry.get("Config") or {}).get("Labels") or {})
        owner = labels.get(OWNER_PID_LABEL)
        stamped_boot = labels.get(BOOT_ID_LABEL) or ""
        if owner is None:
            continue
        name = str(entry.get("Name") or "").lstrip("/")
        identifier = str(entry.get("Id") or "")[:12]
        if boot and stamped_boot and stamped_boot != boot:
            reason = "created before the last reboot"
        else:
            try:
                alive = _process_alive(int(owner))
            except ValueError:
                continue
            if alive:
                continue
            reason = f"owner pid {owner} is gone"
        orphans.append({"id": identifier, "name": name, "reason": reason})
    return orphans


def reap_orphaned_containers(
    *, dry_run: bool = False, timeout_s: int = 120
) -> list[dict[str, str]]:
    """Remove orphaned containers. Returns what was (or would be) removed."""
    orphans = orphaned_containers(timeout_s=timeout_s)
    if dry_run or not orphans:
        return orphans
    removed = []
    for orphan in orphans:
        result = _run([_docker(), "rm", "-f", orphan["id"]], timeout_s)
        if result.returncode == 0:
            removed.append(orphan)
    return removed
