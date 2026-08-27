import pytest

from harnesslens.core.profiles import power_profile


@pytest.mark.parametrize(
    ("harness", "model", "option"),
    [
        ("opencode", "deepseek/deepseek-v4-flash", {"reasoningEffort": "high"}),
        ("codex", "deepseek-v4-flash", {"reasoning_effort": "high"}),
        ("pi", "deepseek/deepseek-v4-flash", {"thinking": "high"}),
    ],
)
def test_all_intelligent_harnesses_use_deepseek_power(harness, model, option):
    profile = power_profile(harness, max_steps=60)

    assert profile.model == model
    assert profile.reasoning_effort == "high"
    assert profile.provider_options == option


def test_power_profile_rejects_unknown_harness():
    with pytest.raises(ValueError, match="unsupported"):
        power_profile("unknown", max_steps=60)


def test_opencode_defaults_to_one_million_token_context(monkeypatch):
    monkeypatch.delenv("HAI_INTELLIGENT_CONTEXT_LIMIT", raising=False)

    assert power_profile("opencode", max_steps=60).context_limit == 1_000_000
    assert power_profile("pi", max_steps=60).context_limit == 65_536


def test_power_profile_accepts_intelligent_context_override(monkeypatch):
    monkeypatch.setenv("HAI_INTELLIGENT_CONTEXT_LIMIT", "1000000")

    profile = power_profile("pi", max_steps=60)

    assert profile.context_limit == 1_000_000


@pytest.mark.parametrize("value", ["0", "-1", "not-an-int"])
def test_power_profile_rejects_invalid_context_override(monkeypatch, value):
    monkeypatch.setenv("HAI_INTELLIGENT_CONTEXT_LIMIT", value)

    with pytest.raises(ValueError, match="positive integer"):
        power_profile("pi", max_steps=60)
