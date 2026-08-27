from __future__ import annotations

import json

from harnesslens.evaluation.rollout_outcome import (
    harness_execution_error,
    provider_trace_error,
)
from harnesslens.harnesses.opencode_runtime import validate_api_trace


def _write_jsonl(path, events):
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )


def test_provider_trace_accepts_retried_call_that_eventually_succeeds(tmp_path):
    trace = tmp_path / "api.jsonl"
    _write_jsonl(
        trace,
        [
            {"event": "upstream_retry", "upstream_status": 429},
            {
                "request": {"messages": []},
                "response": {"choices": [{"message": {"content": "ok"}}]},
                "upstream_status": 200,
            },
        ],
    )
    assert provider_trace_error(trace) == ""
    assert validate_api_trace(trace) == ""


def test_provider_trace_rejects_exhausted_quota_as_infrastructure(tmp_path):
    trace = tmp_path / "api.jsonl"
    _write_jsonl(
        trace,
        [
            {
                "request": {"messages": []},
                "response": None,
                "upstream_status": 429,
                "error": '{"error":{"code":"insufficient_quota"}}',
            }
        ],
    )
    error = provider_trace_error(trace)
    assert "429" in error
    assert "insufficient_quota" in error
    assert "429" in validate_api_trace(trace)


def test_provider_trace_rejects_unknown_non_call_event(tmp_path):
    trace = tmp_path / "api.jsonl"
    _write_jsonl(trace, [{"event": "unexpected_diagnostic"}])

    assert "missing its request" in provider_trace_error(trace)


def test_provider_trace_accepts_legacy_completed_call_without_status(tmp_path):
    trace = tmp_path / "api.jsonl"
    _write_jsonl(
        trace,
        [{"request": {"messages": []}, "response": {"choices": []}}],
    )

    assert provider_trace_error(trace) == ""


def test_harness_stop_reason_error_is_infrastructure():
    stdout = json.dumps(
        {
            "type": "message_end",
            "message": {
                "stopReason": "error",
                "errorMessage": "429: insufficient_quota",
            },
        }
    )
    error = harness_execution_error(stdout, "")
    assert "stopReason=error" in error
    assert "insufficient_quota" in error
