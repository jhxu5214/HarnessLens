import ast
import os
from pathlib import Path

from harnesslens.core.config import (
    DEFAULT_REPO_ROOT,
    load_repo_env,
    repo_root,
)


def test_repo_env_prefers_local_override_then_shared_env_file(tmp_path, monkeypatch):
    (tmp_path / ".env.local").write_text("PRIMARY_VALUE=local\n", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "PRIMARY_VALUE=shared\nFALLBACK_VALUE=shared\n", encoding="utf-8"
    )
    monkeypatch.delenv("PRIMARY_VALUE", raising=False)
    monkeypatch.delenv("FALLBACK_VALUE", raising=False)

    load_repo_env(tmp_path)

    assert os.environ["PRIMARY_VALUE"] == "local"
    assert os.environ["FALLBACK_VALUE"] == "shared"


def test_exported_environment_wins_over_env_file(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("PRIMARY_VALUE=shared\n", encoding="utf-8")
    monkeypatch.setenv("PRIMARY_VALUE", "exported")

    load_repo_env(tmp_path)

    assert os.environ["PRIMARY_VALUE"] == "exported"


def test_repo_root_resolution_order(tmp_path, monkeypatch):
    monkeypatch.delenv("HARNESSLENS_ROOT", raising=False)
    assert repo_root() == DEFAULT_REPO_ROOT
    assert DEFAULT_REPO_ROOT == Path(__file__).resolve().parents[1]

    monkeypatch.setenv("HARNESSLENS_ROOT", str(tmp_path))
    assert repo_root() == tmp_path.resolve()
    assert repo_root(tmp_path / "explicit") == (tmp_path / "explicit").resolve()


def test_unknown_cell_names_the_valid_ones(tmp_path):
    """A typo used to raise a bare ValueError from deep inside the run."""
    import pytest

    from harnesslens.benchmarks.cell_config import (
        SUPPORTED_CELLS,
        benchmark_cell,
        normalize_benchmark_cell,
    )

    with pytest.raises(ValueError, match="unknown benchmark cell") as excinfo:
        normalize_benchmark_cell("nosuchcell")
    for cell in SUPPORTED_CELLS:
        assert cell in str(excinfo.value)


def test_cell_aliases_survive_the_argparse_validator():
    """The validator must not narrow --cell to the canonical names only."""
    from harnesslens.benchmarks.cell_config import benchmark_cell, normalize_benchmark_cell

    for alias in ("retail", "tau2-retail", "banking", "terminal-bench", "bird-mini-dev"):
        assert benchmark_cell(alias) == alias
        assert normalize_benchmark_cell(alias)


def test_removed_cells_are_rejected():
    """The Claw cells are gone; their aliases must not silently resolve."""
    import argparse

    import pytest

    from harnesslens.benchmarks.cell_config import benchmark_cell

    for alias in ("claw_general", "claw-multi-turn", "multiturn-diague"):
        with pytest.raises(argparse.ArgumentTypeError):
            benchmark_cell(alias)


# Variables the framework sets for its own children, or that come from the OS.
# They are not user knobs, so they are exempt from the documentation rule.
FRAMEWORK_OWNED = {
    "PATH", "HOME", "PYTHONPATH", "TERM", "LANG", "NO_COLOR",
    "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME",
    "XDG_STATE_HOME", "XDG_RUNTIME_DIR", "LOGURU_LEVEL",
    "T_BENCH_TASK_LOGS_PATH", "T_BENCH_TASK_AGENT_LOGS_PATH",
    "HERMES_HOME", "TEST_DIR",
}


def _environment_variables_read_by_the_package() -> set[str]:
    package = DEFAULT_REPO_ROOT / "harnesslens"
    found: set[str] = set()
    for path in sorted(package.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            name = None
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"get", "setdefault"}
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "environ"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                name = node.args[0].value
            elif (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "environ"
                and isinstance(node.slice, ast.Constant)
            ):
                name = node.slice.value
            if isinstance(name, str) and name.isupper():
                found.add(name)
    return found - FRAMEWORK_OWNED


def test_every_environment_variable_is_documented():
    """A knob nobody can find is the same as no knob at all.

    This is why the check exists rather than a note asking people to remember:
    the package reads dozens of variables and the list drifts every time a
    runtime is added.
    """
    documented = (DEFAULT_REPO_ROOT / "docs" / "configuration_zh.md").read_text(
        encoding="utf-8"
    )
    undocumented = sorted(
        name
        for name in _environment_variables_read_by_the_package()
        if name not in documented
    )
    assert undocumented == []


def test_pi_is_resolved_the_same_way_everywhere():
    """One resolver, or the analyst and the rollout disagree about where pi is.

    They used to: the rollout accepted third_party/pi-agent while the analyst
    only looked in .pi-agent, so a run rolled out fine and then failed at
    Discovery with the baseline already paid for.
    """
    from harnesslens.benchmarks import pi_tau2
    from harnesslens.harnesses import native_intelligent_runtime
    from harnesslens.core.config import pi_binary

    source = (DEFAULT_REPO_ROOT / "harnesslens").rglob("*.py")
    duplicates = [
        path.name
        for path in source
        if path.name not in {"config.py"}
        and "node_modules" in path.read_text(encoding="utf-8")
        and '".bin"' in path.read_text(encoding="utf-8")
    ]
    assert duplicates == [], f"pi lookup duplicated in {duplicates}"

    # both call sites must be thin delegations to the shared resolver
    for module in (native_intelligent_runtime, pi_tau2):
        assert hasattr(module, "_pi_binary")
    assert callable(pi_binary)


def test_pi_lookup_accepts_either_install_location(tmp_path, monkeypatch):
    from harnesslens.core.config import pi_binary

    monkeypatch.delenv("PI_AGENT_BIN", raising=False)
    monkeypatch.delenv("PI_BIN", raising=False)
    monkeypatch.setattr("harnesslens.core.config.shutil.which", lambda _name: None)

    for relative in (
        Path("third_party") / "pi-agent" / "node_modules" / ".bin" / "pi",
        Path(".pi-agent") / "node_modules" / ".bin" / "pi",
    ):
        root = tmp_path / relative.parts[0].lstrip(".")
        root.mkdir(exist_ok=True)
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("#!/bin/sh\n", encoding="utf-8")
        assert pi_binary(tmp_path) == target.resolve()
        target.unlink()


def test_pi_install_root_is_the_outermost_node_modules_parent(tmp_path, monkeypatch):
    """npm packages nest node_modules; the innermost match is the wrong answer.

    Picking it produced a bind mount that covered only the pi package, so the
    sandbox failed with `bwrap: Can't find source path`.
    """
    from harnesslens.core.config import pi_install_root

    monkeypatch.delenv("PI_AGENT_BIN", raising=False)
    monkeypatch.delenv("PI_BIN", raising=False)
    monkeypatch.setattr("harnesslens.core.config.shutil.which", lambda _name: None)

    install = tmp_path / "third_party" / "pi-agent"
    nested = install / "node_modules" / "@earendil-works" / "pi-coding-agent"
    (nested / "node_modules").mkdir(parents=True)
    launcher = install / "node_modules" / ".bin" / "pi"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")

    assert pi_install_root(tmp_path) == install
