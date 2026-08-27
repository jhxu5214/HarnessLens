import pytest

from harnesslens.infrastructure.analysis_concurrency import (
    analysis_workers,
    max_analysis_concurrency,
)


def test_analysis_concurrency_defaults_to_twenty(monkeypatch):
    monkeypatch.delenv("HAI_MAX_ANALYSIS_CONCURRENCY", raising=False)

    assert max_analysis_concurrency() == 20
    assert analysis_workers(3) == 3
    assert analysis_workers(25) == 20


def test_analysis_concurrency_honors_explicit_limit(monkeypatch):
    monkeypatch.setenv("HAI_MAX_ANALYSIS_CONCURRENCY", "7")

    assert max_analysis_concurrency() == 7
    assert analysis_workers(20) == 7


@pytest.mark.parametrize("value", ["0", "65", "invalid"])
def test_analysis_concurrency_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("HAI_MAX_ANALYSIS_CONCURRENCY", value)

    with pytest.raises(ValueError, match="HAI_MAX_ANALYSIS_CONCURRENCY"):
        max_analysis_concurrency()


def test_analysis_workers_requires_a_job():
    with pytest.raises(ValueError, match="at least one job"):
        analysis_workers(0)
