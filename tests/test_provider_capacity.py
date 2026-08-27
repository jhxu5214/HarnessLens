from __future__ import annotations

from contextlib import contextmanager

import pytest

from harnesslens.benchmarks.pi_tau2 import _run_with_provider_trial_slot
from harnesslens.infrastructure.provider_capacity import (
    provider_trial_concurrency,
    provider_trial_slot,
)


def test_provider_trial_concurrency_is_bounded(monkeypatch):
    monkeypatch.setenv("HAI_PROVIDER_TRIAL_MAX_CONCURRENCY", "8")
    assert provider_trial_concurrency() == 8
    monkeypatch.setenv("HAI_PROVIDER_TRIAL_MAX_CONCURRENCY", "999")
    assert provider_trial_concurrency() == 64


def test_provider_trial_slot_waits_outside_the_trial(monkeypatch, tmp_path):
    monkeypatch.setenv("HAI_PROVIDER_TRIAL_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("HAI_PROVIDER_TRIAL_SLOT_DIR", str(tmp_path))

    with provider_trial_slot() as first:
        assert first["slot"] == 0
        with pytest.raises(TimeoutError, match="provider trial capacity"):
            with provider_trial_slot(timeout_s=0.05):
                raise AssertionError("second lease must not be acquired")

    with provider_trial_slot(timeout_s=0.05) as second:
        assert second["slot"] == 0


def test_tau2_runner_starts_only_after_trial_capacity_is_acquired(monkeypatch):
    state = {"acquired": False}

    @contextmanager
    def fake_slot():
        state["acquired"] = True
        yield {"slot": 3, "waited_s": 1.25, "max_concurrency": 8}
        state["acquired"] = False

    def runner(**kwargs):
        assert state["acquired"] is True
        return {"task_id": kwargs["task_id"], "reward": 1.0}

    monkeypatch.setattr(
        "harnesslens.benchmarks.pi_tau2.provider_trial_slot", fake_slot
    )

    result = _run_with_provider_trial_slot(runner, task_id="12")

    assert result["provider_capacity"] == {
        "slot": 3,
        "waited_s": 1.25,
        "max_concurrency": 8,
    }
