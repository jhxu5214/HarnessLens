from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from harnesslens.benchmarks.terminal_runtime_patch import (
    _APT_LOCK_HELPERS,
    install_terminal_runtime_hooks,
)


def test_terminal_hooks_own_compose_network_override(tmp_path):
    adapter = SimpleNamespace(
        _container_proxy_env=lambda: {"http_proxy": "http://proxy:7890"},
    )

    install_terminal_runtime_hooks(adapter)
    path = adapter._compose_override(tmp_path)

    payload = yaml.safe_load(path.read_text())
    assert payload["networks"]["default"] == {
        "external": True,
        "name": "harnesslens-terminal-bench",
    }
    assert payload["services"]["client"]["environment"]["http_proxy"] == "http://proxy:7890"


def test_terminal_hooks_use_proc_based_apt_lock_wait():
    assert "/proc/[0-9]*/comm" in _APT_LOCK_HELPERS
    assert "apt_retry" in _APT_LOCK_HELPERS
