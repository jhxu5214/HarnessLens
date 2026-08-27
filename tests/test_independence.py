import ast
import json
import re
import subprocess
from pathlib import Path

import pytest

from harnesslens.harnesses.opencode_harness import OpencodeHarnessAdapter
from harnesslens.evaluation.rollout_bridge import (
    TrainRolloutRecord,
    retain_trial_trajectories,
    validate_rollout_interactions,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

# Undeclared local projects must never become release dependencies.
FORBIDDEN_MODULES = ("harness_autoiter", "cell_track")
# Local development paths must not leak into the release.
FORBIDDEN_PATH_FRAGMENTS = (
    "harnesses/opencode",
    "harnesses/_generalize",
)
FORBIDDEN_VERSION_PATH = re.compile(r"versions/v\d+")


def _source_files() -> list[Path]:
    """Every first-party module, minus this file, which spells out the patterns."""
    this_file = Path(__file__).resolve()
    return [
        path
        for path in REPO_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
        and "third_party" not in path.parts
        and ".venv" not in path.parts
        and "venv" not in path.parts
        and path.resolve() != this_file
    ]


def test_repository_is_self_contained():
    """No module may reach outside this checkout for code or data.

    The only permitted external dependencies are the benchmark checkouts under
    ``third_party/`` and the agent runtimes (opencode, codex, pi), both of which
    are resolved through documented environment variables.
    """
    violations = []
    for path in _source_files():
        source = path.read_text(encoding="utf-8")
        relative = path.relative_to(REPO_ROOT)
        if FORBIDDEN_VERSION_PATH.search(source):
            violations.append(f"{relative}: internal version path")
        for fragment in FORBIDDEN_PATH_FRAGMENTS:
            if fragment in source:
                violations.append(f"{relative}: path fragment {fragment!r}")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                names = (node.module,)
            elif isinstance(node, ast.Import):
                names = tuple(alias.name for alias in node.names)
            else:
                continue
            for name in names:
                root = name.split(".")[0]
                if root in FORBIDDEN_MODULES:
                    violations.append(f"{relative}: imports {name}")
    assert violations == []


# Absolute paths under a user's home or a machine's data volume. Matched
# anywhere in the line, not just right after the opening quote: the first
# version of this check anchored on the quote and therefore missed paths
# embedded in a URL, e.g. "unix:///data2/someone/dockers/run/docker.sock".
# The lookbehind keeps this anchored to the start of an absolute path, so a
# relative segment like third_party/.../data/tau2/ is not a match, and `data`
# requires a digit for the same reason.
DEVELOPER_PATH = re.compile(
    r"(?<![A-Za-z0-9_.-])/(?:home|root|Users|data\d+)/[A-Za-z0-9_.-]+/"
)


TEXT_SUFFIXES = {".py", ".md", ".sh", ".toml", ".json", ".yaml", ".yml", ".mjs", ".txt"}
# Upstream documentation snapshots are reproduced verbatim; their example paths
# are the vendor's, not this machine's.
VENDORED = ("assets/docs_cache",)


def _text_files() -> list[Path]:
    """Everything a reader could copy a path out of, not just importable code."""
    this_file = Path(__file__).resolve()
    files = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.resolve() == this_file:
            continue
        # runs/, results/ and the caches hold generated artifacts that record
        # real absolute paths by design, and none of them are tracked by git.
        if {
            "__pycache__",
            "third_party",
            ".git",
            "runs",
            "results",
            ".cache",
            ".venv",
        } & set(path.parts):
            continue
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative.startswith(VENDORED) or path.suffix not in TEXT_SUFFIXES:
            if path.name not in {".env.example", ".gitignore"}:
                continue
        files.append(path)
    return files


def test_no_source_file_hardcodes_a_developer_absolute_path():
    """Host-specific absolute paths must go through an environment variable."""
    violations = []
    for path in _text_files():
        for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1
        ):
            if DEVELOPER_PATH.search(line):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{number}: {line.strip()}")
    assert violations == []


# Credential shapes that must never appear in tracked source.
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[A-Z0-9]{12,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
# Hosts a source file may name. Anything else risks pinning private
# infrastructure — a relay endpoint, an internal mirror — into a public
# repository, where it also silently becomes the documented default.
ALLOWED_HOSTS = re.compile(
    r"""^https?://(?:
        # RFC 2606 / RFC 5737 reserved names and addresses, for fixtures
        (?:[a-z0-9-]+\.)*example(?:\.(?:com|org|net|invalid))?
        | (?:[a-z0-9-]+\.)*(?:test|invalid|localhost)
        | localhost | 127\.0\.0\.1 | 0\.0\.0\.0
        | 192\.0\.2\.\d+ | 198\.51\.100\.\d+ | 203\.0\.113\.\d+
        # container-internal service names, resolved by the sandbox network
        | proxy | host\.docker\.internal
        # public infrastructure this project genuinely talks to
        | api\.deepseek\.com | api\.anthropic\.com | api\.openai\.com
        | developers\.openai\.com | opencode\.ai
        | archive\.ubuntu\.com | (?:[a-z0-9-]+\.)*astral\.sh
        | (?:[a-z0-9-]+\.)*(?:github\.com|huggingface\.co)
        | (?:[a-z0-9-]+\.)*(?:schemastore\.org|json-schema\.org)
    )(?::\d+)?$""",
    re.VERBOSE,
)


def test_no_credentials_are_committed():
    violations = []
    for path in _source_files():
        text = path.read_text(encoding="utf-8")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                violations.append(f"{path.relative_to(REPO_ROOT)}: {pattern.pattern}")
    assert violations == []
    tracked_env = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--error-unmatch", ".env"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    assert tracked_env.returncode != 0, ".env must never be committed"


def test_no_credential_is_passed_on_a_command_line():
    """Argv is world-readable via `ps`; secrets must travel in the child env.

    The proxies still accept ``--key`` so an externally built command keeps
    working, which is why this scans for the flag being *passed*, not parsed.
    """
    offenders = []
    # Scoped to the package: only it builds command lines. Tests legitimately
    # name the flag in order to assert that it is absent.
    for path in sorted((REPO_ROOT / "harnesslens").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        # Exact nodes that belong to an add_argument("--key") declaration.
        declared = {
            id(argument)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            for argument in node.args
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and node.value == "--key"
                and id(node) not in declared
            ):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert offenders == []


def test_no_private_provider_endpoint_is_hardcoded():
    """Fixtures may only name reserved or official hosts.

    A relay endpoint copied out of a developer's .env is not a secret by itself,
    but it identifies private infrastructure and it silently becomes the default
    for anyone who reads the test as documentation.
    """
    url = re.compile(r"https?://[A-Za-z0-9.-]+(?::\d+)?")
    violations = []
    for path in _source_files():
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for match in url.finditer(line):
                if not ALLOWED_HOSTS.match(match.group(0)):
                    violations.append(
                        f"{path.relative_to(REPO_ROOT)}:{number}: {match.group(0)}"
                    )
    assert violations == []


def test_skill_validation_keeps_sop_in_a_real_skill():
    delta = {
        "config_patch": {"tools.skill": True},
        "files": [
            {
                "path": ".opencode/skills/order-change/SKILL.md",
                "content": (
                    "---\nname: order-change\n"
                    "description: Use for pending order changes.\n---\n\n"
                    "1. Authenticate.\n2. Inspect.\n3. Confirm.\n4. Mutate.\n"
                ),
            },
            {
                "path": ".opencode/skills/order-change/reference.md",
                "content": "Exact branch notes.",
            },
        ],
    }

    OpencodeHarnessAdapter().validate_delta(delta)

    invalid = dict(delta)
    invalid["files"] = [dict(delta["files"][0], path=".opencode/skills/order_change/SKILL.md")]
    with pytest.raises(ValueError, match="hyphenated"):
        OpencodeHarnessAdapter().validate_delta(invalid)


def test_rollout_bridge_retains_complete_api_sidecar(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    api_trace = source_root / "trial.jsonl"
    api_trace.write_text(
        json.dumps(
            {
                "request": {"messages": [{"role": "system", "content": "policy"}]},
                "response": {"choices": [{"message": {"reasoning_content": "think"}}]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    trajectory = source_root / "rollout_2.jsonl"
    trajectory.write_text(
        json.dumps({"task_id": "0", "trial": 0, "api_calls_jsonl": str(api_trace)})
        + "\n",
        encoding="utf-8",
    )
    payload = {
        "per_task": {
            "0": {
                "repeats": 1,
                "rewards": [1.0],
                "trajectory_paths": [str(trajectory)],
            }
        },
        "records": [
            {
                "task_id": "0",
                "rewards": [1.0],
                "trajectory_paths": [str(trajectory)],
            }
        ],
    }

    retained = retain_trial_trajectories(payload, target_root=tmp_path / "retained")

    retained_path = Path(retained["records"][0]["trajectory_paths"][0])
    row = json.loads(retained_path.read_text(encoding="utf-8"))
    sidecar = retained_path.parent / row["api_calls_jsonl"]
    assert sidecar.is_file()
    assert "reasoning_content" in sidecar.read_text(encoding="utf-8")
    validate_rollout_interactions(
        [
            TrainRolloutRecord(
                task_id="0",
                rewards=(1.0,),
                harness_version="v0",
                trajectory_paths=(str(retained_path),),
            )
        ]
    )


def test_rollout_bridge_rejects_missing_api_trace(tmp_path):
    trajectory = tmp_path / "trial.jsonl"
    trajectory.write_text(json.dumps({"task_id": "0", "trial": 0}) + "\n")

    with pytest.raises(RuntimeError, match="no complete OpenCode trace"):
        validate_rollout_interactions(
            [
                TrainRolloutRecord(
                    task_id="0",
                    rewards=(0.0,),
                    harness_version="v0",
                    trajectory_paths=(str(trajectory),),
                )
            ]
        )


def test_provider_endpoint_is_never_resolved_from_the_legacy_name_alone():
    """DEEPSEEK_URL may only appear as a fallback behind DEEPSEEK_BASE_URL.

    Three sites once read the legacy name on its own. With only the documented
    DEEPSEEK_BASE_URL set they fell back to the public DeepSeek endpoint while
    still sending a relay's key, and the failure surfaced as an authentication
    error from the tau2 user simulator — nowhere near the configuration.
    """
    offenders = []
    for path in sorted((REPO_ROOT / "harnesslens").rglob("*.py")):
        if path.name == "config.py":  # the shared resolver lives here
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Constant) and node.value == "DEEPSEEK_URL"
            ):
                continue
            # Accept it only when the enclosing expression also names the
            # documented variable, i.e. `... or os.environ.get("DEEPSEEK_URL")`.
            enclosing = next(
                (
                    parent
                    for parent in ast.walk(tree)
                    if isinstance(parent, ast.BoolOp)
                    and any(
                        isinstance(c, ast.Constant) and c.value == "DEEPSEEK_URL"
                        for c in ast.walk(parent)
                    )
                    and any(
                        isinstance(c, ast.Constant) and c.value == "DEEPSEEK_BASE_URL"
                        for c in ast.walk(parent)
                    )
                ),
                None,
            )
            if enclosing is None:
                offenders.append(f"{path.name}:{node.lineno}")
    assert sorted(set(offenders)) == []
