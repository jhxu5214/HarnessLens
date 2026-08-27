import json

from harnesslens.infrastructure.chat_completions_proxy import (
    _provider_error_code,
    _provider_slot_count,
    _upstream_exception_retry_delay,
    _upstream_retry_policy,
)


def _error_body(code):
    return json.dumps({"error": {"type": code, "code": code}}).encode()


def test_burst_rate_uses_short_exponential_backoff(monkeypatch):
    monkeypatch.setenv("HAI_PROVIDER_RETRY_ATTEMPTS", "4")
    body = _error_body("limit_burst_rate")

    assert _provider_error_code(body) == "limit_burst_rate"
    assert [_upstream_retry_policy(429, body, attempt) for attempt in range(1, 5)] == [
        5.0,
        10.0,
        20.0,
        None,
    ]


def test_quota_uses_bounded_long_backoff_and_retry_after(monkeypatch):
    monkeypatch.setenv("HAI_PROVIDER_RETRY_ATTEMPTS", "5")
    body = _error_body("insufficient_quota")

    assert [_upstream_retry_policy(429, body, attempt) for attempt in range(1, 6)] == [
        30.0,
        60.0,
        120.0,
        120.0,
        None,
    ]
    assert _upstream_retry_policy(429, body, 1, retry_after="45") == 45.0


def test_transient_server_errors_use_bounded_backoff(monkeypatch):
    monkeypatch.setenv("HAI_PROVIDER_RETRY_ATTEMPTS", "4")

    assert [_upstream_retry_policy(502, _error_body("server_error"), attempt) for attempt in range(1, 5)] == [
        5.0,
        10.0,
        20.0,
        None,
    ]
    assert [_upstream_exception_retry_delay(attempt) for attempt in range(1, 5)] == [
        5.0,
        10.0,
        20.0,
        None,
    ]


def test_non_retryable_errors_are_relayed_immediately(monkeypatch):
    monkeypatch.setenv("HAI_PROVIDER_RETRY_ATTEMPTS", "4")

    assert _upstream_retry_policy(400, _error_body("server_error"), 1) is None
    assert _upstream_retry_policy(429, _error_body("billing_disabled"), 1) is None


def test_provider_concurrency_is_bounded(monkeypatch):
    monkeypatch.setenv("HAI_PROVIDER_MAX_CONCURRENCY", "20")
    assert _provider_slot_count() == 20
    monkeypatch.setenv("HAI_PROVIDER_MAX_CONCURRENCY", "999")
    assert _provider_slot_count() == 64
