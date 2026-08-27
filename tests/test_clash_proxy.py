from __future__ import annotations

import os
from pathlib import Path

import pytest

import run_e2e
from harnesslens.infrastructure.clash_proxy import configure_terminal_clash_proxy


def test_terminal_clash_proxy_uses_container_reachable_address(monkeypatch):
    monkeypatch.setenv("http_proxy", "http://192.0.2.10:16627")
    monkeypatch.setenv("TB_CONTAINER_PROXY_URL", "previous")
    monkeypatch.setenv("TB_ENABLE_CONTAINER_CLASH", "0")

    value = configure_terminal_clash_proxy()

    assert value == "http://192.0.2.10:16627"
    assert value == os.environ["TB_CONTAINER_PROXY_URL"]
    assert os.environ["TB_ENABLE_CONTAINER_CLASH"] == "1"


def test_terminal_clash_proxy_rejects_host_loopback(monkeypatch):
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:7890")

    with pytest.raises(RuntimeError, match="allow-lan"):
        configure_terminal_clash_proxy()


def test_e2e_configures_terminal_clash_before_controller(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        run_e2e,
        "configure_terminal_clash_proxy",
        lambda: calls.append("configured"),
    )

    repo_root = Path(__file__).resolve().parents[1]
    run_e2e.configure_cell_runtime(repo_root, "terminal-bench")

    assert calls == ["configured"]
