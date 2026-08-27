from __future__ import annotations

from types import SimpleNamespace

import pytest

from harnesslens.benchmarks import terminal_images


def test_task_image_name_uses_the_v1_tb2_client_tag_contract():
    assert terminal_images.task_image_name("count-dataset-tokens") == (
        "alexgshaw/count-dataset-tokens:20251031"
    )


def test_preflight_only_inspects_each_requested_task_image(monkeypatch):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(terminal_images.subprocess, "run", fake_run)

    result = terminal_images.require_preloaded_terminal_images(["one", "two"])

    assert result.task_count == 2
    assert commands == [
        ["docker", "image", "inspect", "alexgshaw/one:20251031"],
        ["docker", "image", "inspect", "alexgshaw/two:20251031"],
    ]
    assert result.to_dict()["policy"] == "prewarmed_only_no_pull_no_build"


def test_preflight_refuses_a_missing_image_without_pull_or_build(monkeypatch):
    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=1, stderr="not found")

    monkeypatch.setattr(terminal_images.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="refusing to pull/build"):
        terminal_images.require_preloaded_terminal_images(["missing-task"])


def test_shared_network_is_created_once(monkeypatch):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1 if command[2] == "inspect" else 0, stderr="")

    monkeypatch.setattr(terminal_images.subprocess, "run", fake_run)

    assert terminal_images.ensure_terminal_shared_network() == "harnesslens-terminal-bench"
    assert commands == [
        ["docker", "network", "inspect", "harnesslens-terminal-bench"],
        ["docker", "network", "create", "--driver", "bridge", "harnesslens-terminal-bench"],
    ]
