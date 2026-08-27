from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping


SUPPORTED_HARNESSES = frozenset({"opencode", "codex", "pi"})
DEFAULT_CONTEXT_LIMIT = 65_536
DEFAULT_OPENCODE_CONTEXT_LIMIT = 1_000_000
DEFAULT_OUTPUT_LIMIT = 24_576


@dataclass(frozen=True)
class HarnessProfile:
    harness: str
    model: str
    reasoning_effort: str
    max_steps: int
    context_limit: int = DEFAULT_CONTEXT_LIMIT
    output_limit: int = DEFAULT_OUTPUT_LIMIT

    def __post_init__(self) -> None:
        if self.harness not in SUPPORTED_HARNESSES:
            raise ValueError(f"unsupported intelligent harness: {self.harness}")
        if self.reasoning_effort != "high":
            raise ValueError("intelligent harnesses require the power/high-reasoning profile")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.context_limit <= 0:
            raise ValueError("context_limit must be positive")
        if self.output_limit <= 0:
            raise ValueError("output_limit must be positive")

    @property
    def provider_options(self) -> Mapping[str, Any]:
        if self.harness == "opencode":
            return {"reasoningEffort": "high"}
        if self.harness == "codex":
            return {"reasoning_effort": "high"}
        if self.harness == "pi":
            return {"thinking": "high"}
        return {"thinking": {"type": "enabled"}}


def power_profile(harness: str, *, max_steps: int) -> HarnessProfile:
    normalized = str(harness).strip().lower().replace("-", "_")
    if normalized == "pi_agent":
        normalized = "pi"
    models = {
        "opencode": "deepseek/deepseek-v4-flash",
        "codex": "deepseek-v4-flash",
        "pi": "deepseek/deepseek-v4-flash",
    }
    if normalized not in models:
        raise ValueError(f"unsupported intelligent harness: {harness}")
    default_context_limit = (
        DEFAULT_OPENCODE_CONTEXT_LIMIT
        if normalized == "opencode"
        else DEFAULT_CONTEXT_LIMIT
    )
    return HarnessProfile(
        harness=normalized,
        model=models[normalized],
        reasoning_effort="high",
        max_steps=int(max_steps),
        context_limit=_positive_env_int(
            "HAI_INTELLIGENT_CONTEXT_LIMIT", default=default_context_limit
        ),
    )


def _positive_env_int(name: str, *, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return int(default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value
