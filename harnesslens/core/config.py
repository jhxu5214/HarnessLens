from __future__ import annotations

import os
import shutil
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ROOT = PACKAGE_ROOT.parent
REPO_ROOT_ENV = "HARNESSLENS_ROOT"


def repo_root(override: str | Path | None = None) -> Path:
    """Resolve the HarnessLens checkout root.

    Precedence: explicit argument, then ``HARNESSLENS_ROOT``, then the directory
    that contains the installed ``harnesslens`` package.
    """
    if override is not None:
        return Path(override).resolve()
    from_env = os.environ.get(REPO_ROOT_ENV)
    if from_env and from_env.strip():
        return Path(from_env.strip()).resolve()
    return DEFAULT_REPO_ROOT


DEFAULT_PROVIDER_BASE_URL = "https://api.deepseek.com/v1"


def provider_base_url() -> str:
    """Resolve the provider endpoint from one place.

    ``DEEPSEEK_BASE_URL`` is the documented variable; ``DEEPSEEK_URL`` is an
    older name kept for compatibility. Three separate call sites once consulted
    only the older one, so with just the documented variable set they silently
    fell back to the public DeepSeek endpoint and presented a relay's key to it
    — which surfaces as "your api key is invalid" from a component nowhere near
    the configuration.
    """
    value = str(
        os.environ.get("DEEPSEEK_BASE_URL")
        or os.environ.get("DEEPSEEK_URL")
        or DEFAULT_PROVIDER_BASE_URL
    ).rstrip("/")
    if value.endswith("/chat/completions"):
        value = value[: -len("/chat/completions")]
    return value



def pi_binary(repo_root_path: str | Path | None = None) -> Path:
    """Locate the pi runtime, the same way for every caller.

    pi is looked up twice: once as the harness under test and once as the
    intelligent analyst. Those lookups used to consult different directories, so
    an install under third_party/pi-agent produced a run that rolled out fine
    and then died at Discovery — after the baseline had already been paid for.
    """
    configured = os.environ.get("PI_AGENT_BIN") or os.environ.get("PI_BIN")
    root = repo_root(repo_root_path)
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
        located = shutil.which(configured)
        if located:
            candidates.append(Path(located))
    candidates += [
        root / "third_party" / "pi-agent" / "node_modules" / ".bin" / "pi",
        root / ".pi-agent" / "node_modules" / ".bin" / "pi",
    ]
    located = shutil.which("pi")
    if located:
        candidates.append(Path(located))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(
        "pi executable is unavailable: set PI_AGENT_BIN, or install it under "
        "third_party/pi-agent or .pi-agent"
    )



def pi_install_root(repo_root_path: str | Path | None = None) -> Path:
    """Directory holding pi's ``node_modules``.

    A sandbox has to bind-mount this, and the Harness Query reads pi's bundled
    documentation from under it. Both used to name a fixed directory, so an
    install anywhere else produced `bwrap: Can't find source path` after the
    binary itself had already resolved fine.
    """
    binary = pi_binary(repo_root_path)
    # npm packages carry nested node_modules, so the innermost match is the pi
    # package itself, not the install. Take the outermost one.
    nested = [parent for parent in binary.parents if parent.name == "node_modules"]
    if nested:
        return nested[-1].parent
    return binary.parent


def load_env_file(path: str | Path) -> None:
    target = Path(path)
    if not target.exists():
        return
    for raw in target.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_repo_env(repo_root_path: str | Path | None = None) -> None:
    """Load ``<repo>/.env`` into the process environment without overriding it.

    Values already present in ``os.environ`` always win, so an exported shell
    variable overrides the file. ``.env.local`` is read first and therefore takes
    precedence over ``.env`` for machine-specific overrides.
    """
    root = repo_root(repo_root_path)
    load_env_file(root / ".env.local")
    load_env_file(root / ".env")
