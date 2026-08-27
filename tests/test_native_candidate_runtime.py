import json
from pathlib import Path

import pytest

from harnesslens.harnesses.native_candidate_runtime import (
    CODEX_HOOK_CONTEXT_PATH,
    codex_hook_declared,
    codex_project_trust_config,
    install_codex_hook_dispatcher,
    native_manifest,
    prepare_codex_project_hooks,
)


def test_codex_hook_materializer_owns_executable_configuration(tmp_path):
    manifest = native_manifest(
        "codex",
        {
            "files": [
                {
                    "path": CODEX_HOOK_CONTEXT_PATH,
                    "content": "Keep this bounded behavior visible.",
                }
            ]
        },
    )
    assert codex_hook_declared(manifest)
    assert not codex_hook_declared({"files": []})
    context = tmp_path / CODEX_HOOK_CONTEXT_PATH
    context.parent.mkdir(parents=True)
    context.write_text(manifest["files"][0]["content"], encoding="utf-8")

    assert install_codex_hook_dispatcher(tmp_path, manifest)
    assert (tmp_path / ".git").is_dir()
    assert codex_project_trust_config(tmp_path) == {
        "projects": {str(tmp_path.resolve()): {"trust_level": "trusted"}}
    }

    hooks = json.loads((tmp_path / ".codex/hooks.json").read_text(encoding="utf-8"))
    entry = hooks["hooks"]["SessionStart"][0]
    assert "matcher" not in entry
    assert "codex_session_hook.py" in entry["hooks"][0]["command"]
    assert "Keep this bounded behavior visible." not in entry["hooks"][0]["command"]


@pytest.mark.parametrize(
    "path",
    [
        ".codex/hooks.json",
        ".codex/hooks/candidate-command.sh",
        ".codex/agents/candidate.toml",
    ],
)
def test_codex_manifest_keeps_native_candidate_surfaces_for_isolated_runtime(path):
    manifest = native_manifest(
        "codex", {"files": [{"path": path, "content": "candidate"}]}
    )

    assert manifest["files"] == [{"path": path, "content": "candidate"}]


def test_codex_project_hooks_accept_candidate_owned_configuration(tmp_path):
    hooks = tmp_path / ".codex" / "hooks.json"
    hooks.parent.mkdir(parents=True)
    hooks.write_text('{"hooks": {}}\n', encoding="utf-8")

    assert prepare_codex_project_hooks(tmp_path, {})
    assert hooks.read_text(encoding="utf-8") == '{"hooks": {}}\n'
    assert (tmp_path / ".git").is_dir()
