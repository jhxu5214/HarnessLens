#!/usr/bin/env python3
"""Report whether this machine can run a given HarnessLens cell.

Every check is read-only. The exit code is non-zero when a required item is
missing, so this doubles as a preflight step in a wrapper script:

    python scripts/check_env.py --cell retail --harness opencode
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harnesslens.benchmarks.cell_config import benchmark_config  # noqa: E402
from harnesslens.core.config import load_repo_env, repo_root  # noqa: E402


OK = "ok"
MISSING = "MISSING"
OPTIONAL = "optional"

# Agent runtimes a candidate harness can be rolled out on. Only the one being
# used has to be present.
RUNTIME_BINARIES = {
    "opencode": ("opencode", "OPENCODE_PREFIX / HAI_OPENCODE_BIN"),
    "codex": ("codex", "PATH"),
    "pi": ("pi", "PI_AGENT_BIN or ./.pi-agent/node_modules/.bin/pi"),
}

# Benchmark checkouts, keyed by the cell kind that needs them.
BENCHMARK_CHECKOUTS = {
    "tau2": ("third_party/tau3-bench", "https://github.com/sierra-research/tau2-bench"),
    "bird": ("third_party/bird-mini-dev", "https://github.com/bird-bench/mini_dev"),
    "terminal_bench": (
        "third_party/terminal-bench",
        "https://github.com/harbor-framework/terminal-bench",
    ),
}


class Report:
    def __init__(self) -> None:
        self.failed = False

    def line(self, status: str, label: str, detail: str = "") -> None:
        if status == MISSING:
            self.failed = True
        mark = {OK: "  ok  ", MISSING: " MISS ", OPTIONAL: " skip "}[status]
        suffix = f"  — {detail}" if detail else ""
        print(f"[{mark}] {label}{suffix}")

    def section(self, title: str) -> None:
        print(f"\n{title}")
        print("-" * len(title))


def check_python(report: Report) -> None:
    report.section("Python")
    version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info >= (3, 11):
        report.line(OK, f"python {version}")
    else:
        report.line(MISSING, f"python {version}", "3.11 or newer required")
    for module in ("yaml", "httpx"):
        found = importlib.util.find_spec(module) is not None
        report.line(OK if found else MISSING, f"import {module}",
                    "" if found else "run scripts/setup.sh")
    for module in ("loguru",):
        found = importlib.util.find_spec(module) is not None
        report.line(OK if found else OPTIONAL, f"import {module}",
                    "" if found else "only needed by the tau2 MCP bridge")


def check_credentials(report: Report) -> None:
    report.section("Credentials")
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        report.line(OK, "DEEPSEEK_API_KEY", f"set ({len(key)} chars)")
    else:
        report.line(MISSING, "DEEPSEEK_API_KEY", "set it in .env or export it")
    base = os.environ.get("DEEPSEEK_BASE_URL", "").strip()
    report.line(OK if base else OPTIONAL, "DEEPSEEK_BASE_URL", base or "provider default")


def check_runtime(report: Report, harness: str) -> None:
    report.section(f"Agent runtime ({harness})")
    binary, hint = RUNTIME_BINARIES[harness]
    located = shutil.which(binary)
    if binary == "opencode" and not located:
        prefix = Path(os.environ.get("OPENCODE_PREFIX") or "~/.opencode").expanduser()
        candidate = prefix / "bin" / binary
        located = str(candidate) if candidate.is_file() else None
    if binary == "pi":
        # the same resolver the run uses, so preflight cannot disagree with it
        from harnesslens.core.config import pi_binary

        try:
            located = str(pi_binary())
        except RuntimeError:
            located = None
    report.line(OK if located else MISSING, binary, located or f"not found via {hint}")

    node = shutil.which("node")
    report.line(OK if node else OPTIONAL, "node", node or "needed by the pi runtime")


def check_benchmark(report: Report, root: Path, cell: str) -> None:
    report.section(f"Benchmark cell ({cell})")
    try:
        config = benchmark_config(root, cell)
    except Exception as exc:  # noqa: BLE001 — the message is the diagnosis
        report.line(MISSING, f"cell {cell}", str(exc))
        return
    report.line(OK, f"cell {cell}", f"kind={config.kind}, {len(config.train_task_ids)} TRAIN tasks")

    checkout, url = BENCHMARK_CHECKOUTS[config.kind]
    path = root / checkout
    report.line(OK if path.is_dir() else MISSING, checkout, "" if path.is_dir() else f"clone {url}")

    for relative in config.task_source_files()[:6]:
        target = root / relative
        report.line(OK if target.exists() else MISSING, relative)

    if config.kind == "tau2":
        venv = root / "third_party" / "tau3-bench" / ".venv" / "bin" / "python3"
        report.line(OK if venv.is_file() else MISSING, str(venv.relative_to(root)),
                    "" if venv.is_file() else "create the tau2 virtualenv")
    if config.kind == "terminal_bench":
        docker = shutil.which("docker")
        report.line(OK if docker else MISSING, "docker", docker or "terminal-bench runs one container per task")
        check_docker_daemon(report)


def check_docker_daemon(report: Report) -> None:
    """Report which docker daemon terminal-bench will actually talk to.

    The rootless state directory defaults to ~/dockers. A machine that keeps it
    on another volume silently gets an unreachable socket, so resolve it here
    rather than letting the first rollout discover it.
    """
    from harnesslens.infrastructure.rootless_docker import rootless_docker_host, rootless_docker_root

    # resolve now rather than at import, so a .env loaded since then counts
    root = rootless_docker_root()
    host = rootless_docker_host(root)
    source = (
        "DOCKER_HOST"
        if os.environ.get("DOCKER_HOST")
        else "HAI_DOCKER_HOST"
        if os.environ.get("HAI_DOCKER_HOST")
        else "derived from HAI_DOCKER_ROOT"
    )
    report.line(OK, "docker host", f"{host} (from {source})")

    if host.startswith("unix://"):
        socket_path = Path(host[len("unix://") :])
        if socket_path.exists():
            report.line(OK, "docker socket", str(socket_path))
        else:
            report.line(
                MISSING,
                "docker socket",
                f"{socket_path} does not exist — set HAI_DOCKER_ROOT (currently "
                f"{root}) or DOCKER_HOST",
            )


def check_assets(report: Report, root: Path) -> None:
    report.section("Vendored assets")
    for relative in (
        "assets/docs_cache/opencode/config.md",
        "assets/docs_cache/codex/config_reference.md",
        "assets/canonical_splits/banking_knowledge_split.json",
        "assets/terminal_task_assets/headless-terminal/tests/test.sh",
    ):
        target = root / relative
        report.line(OK if target.is_file() else MISSING, relative)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", default="retail",
                        help="retail, banking, terminal-bench, bird")
    parser.add_argument("--harness", choices=tuple(RUNTIME_BINARIES), default="opencode")
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args()

    root = repo_root(args.repo_root)
    load_repo_env(root)

    print(f"HarnessLens preflight — {root}")
    report = Report()
    check_python(report)
    check_credentials(report)
    check_runtime(report, args.harness)
    check_assets(report, root)
    check_benchmark(report, root, args.cell)

    print()
    if report.failed:
        print("Preflight FAILED — resolve the MISS lines above.")
        print("See docs/configuration.md and docs/benchmarks.md.")
        return 1
    print("Preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
