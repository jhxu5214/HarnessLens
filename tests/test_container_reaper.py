"""The reaper must remove only containers whose run is provably gone.

Getting this wrong in the permissive direction kills a live run's container;
getting it wrong in the strict direction leaves the 57 GB of leftovers that
motivated the module. Both directions are asserted here.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

from harnesslens.infrastructure import container_reaper


def _fake_docker(monkeypatch, *, containers):
    """Answer `docker ps` and `docker inspect` from a fixture, record removals."""
    removed = []

    def fake_run(command, timeout_s):
        if "ps" in command:
            ids = "\n".join(c["Id"] for c in containers)
            return SimpleNamespace(returncode=0, stdout=ids, stderr="")
        if "inspect" in command:
            return SimpleNamespace(
                returncode=0, stdout=json.dumps(containers), stderr=""
            )
        if "rm" in command:
            removed.append(command[-1])
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected docker call: {command}")

    monkeypatch.setattr(container_reaper, "_run", fake_run)
    return removed


def _container(name, *, pid, boot):
    return {
        "Id": f"{name}-id-0000000000",
        "Name": f"/{name}",
        "Config": {
            "Labels": {
                container_reaper.OWNER_PID_LABEL: str(pid),
                container_reaper.BOOT_ID_LABEL: boot,
            }
        },
    }


def test_a_live_runs_container_is_left_alone(monkeypatch):
    boot = container_reaper.current_boot_id()
    removed = _fake_docker(
        monkeypatch, containers=[_container("tb_live", pid=os.getpid(), boot=boot)]
    )

    assert container_reaper.orphaned_containers() == []
    assert container_reaper.reap_orphaned_containers() == []
    assert removed == []


def test_a_dead_owners_container_is_reclaimed(monkeypatch):
    boot = container_reaper.current_boot_id()
    monkeypatch.setattr(container_reaper, "_process_alive", lambda pid: False)
    removed = _fake_docker(
        monkeypatch, containers=[_container("tb_dead", pid=999_000, boot=boot)]
    )

    orphans = container_reaper.orphaned_containers()
    assert [o["name"] for o in orphans] == ["tb_dead"]
    assert "gone" in orphans[0]["reason"]

    container_reaper.reap_orphaned_containers()
    assert removed == [orphans[0]["id"]]


def test_a_container_from_before_the_last_reboot_is_reclaimed(monkeypatch):
    monkeypatch.setattr(container_reaper, "current_boot_id", lambda: "boot-now")
    # the pid may well be alive again after a reboot, so it must not be trusted
    monkeypatch.setattr(container_reaper, "_process_alive", lambda pid: True)
    _fake_docker(
        monkeypatch,
        containers=[_container("tb_old", pid=os.getpid(), boot="boot-before")],
    )

    orphans = container_reaper.orphaned_containers()
    assert [o["name"] for o in orphans] == ["tb_old"]
    assert "reboot" in orphans[0]["reason"]


def test_unlabelled_containers_are_never_touched(monkeypatch):
    """Anything this project did not create is none of its business."""
    foreign = {"Id": "foreign-id-00", "Name": "/somebody-elses", "Config": {"Labels": {}}}
    removed = _fake_docker(monkeypatch, containers=[foreign])

    assert container_reaper.orphaned_containers() == []
    assert removed == []


def test_dry_run_reports_without_removing(monkeypatch):
    boot = container_reaper.current_boot_id()
    monkeypatch.setattr(container_reaper, "_process_alive", lambda pid: False)
    removed = _fake_docker(
        monkeypatch, containers=[_container("tb_dead", pid=999_000, boot=boot)]
    )

    assert len(container_reaper.reap_orphaned_containers(dry_run=True)) == 1
    assert removed == []


def test_ownership_labels_identify_this_process():
    labels = container_reaper.ownership_labels()
    assert labels[container_reaper.OWNER_PID_LABEL] == str(os.getpid())
    assert labels[container_reaper.BOOT_ID_LABEL] == container_reaper.current_boot_id()
