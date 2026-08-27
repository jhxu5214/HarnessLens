from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Mapping, Sequence


_SAFE_ENV_NAMES = {
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "NO_COLOR",
    "PATH",
    "TERM",
    "TZ",
}
_SECRET_MARKERS = (
    "API_KEY",
    "PASSWORD",
    "SECRET",
    "TOKEN",
    "CREDENTIAL",
    "AUTH",
    "COOKIE",
)


def isolated_child_env(
    base: Mapping[str, str] | None = None,
    *,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    source = dict(base or os.environ)
    env = {
        key: str(value)
        for key, value in source.items()
        if key in _SAFE_ENV_NAMES or key.startswith("LC_")
    }
    env.update({str(key): str(value) for key, value in (overrides or {}).items()})
    leaked = sorted(
        key
        for key in env
        if any(marker in key.upper() for marker in _SECRET_MARKERS)
        and env[key] not in {"", "editor-local-proxy"}
    )
    if leaked:
        raise ValueError(f"isolated child environment contains credentials: {leaked}")
    return env


def bubblewrap_command(
    command: Sequence[str],
    *,
    writable_root: str | Path,
    working_directory: str | Path,
    read_only_roots: Sequence[str | Path] = (),
) -> list[str]:
    executable = shutil.which("bwrap")
    if not executable:
        raise RuntimeError("bubblewrap is required for isolated Harness Editor calls")
    writable = Path(writable_root).resolve()
    cwd = Path(working_directory).resolve()
    try:
        cwd.relative_to(writable)
    except ValueError as exc:
        raise ValueError("isolated working directory must be under writable_root") from exc
    roots = [
        path
        for path in (Path("/usr"), Path("/bin"), Path("/lib"), Path("/lib64"), Path("/etc"))
        if path.exists()
    ]
    roots.extend(Path(path).resolve() for path in read_only_roots)
    args = [
        executable,
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    destinations = [*roots, writable]
    created: set[str] = set()
    for destination in destinations:
        for parent in reversed(destination.parents):
            text = str(parent)
            if text == "/" or text in created:
                continue
            args.extend(["--dir", text])
            created.add(text)
        text = str(destination)
        if destination.is_dir() and text not in created:
            args.extend(["--dir", text])
            created.add(text)
    seen: set[str] = set()
    for root in roots:
        text = str(root)
        if text in seen:
            continue
        args.extend(["--ro-bind", text, text])
        seen.add(text)
    args.extend(
        [
            "--bind",
            str(writable),
            str(writable),
            "--chdir",
            str(cwd),
            "--",
            *[str(item) for item in command],
        ]
    )
    return args


def node_runtime_root() -> Path | None:
    executable = shutil.which("node")
    if not executable:
        return None
    resolved = Path(executable).resolve()
    for parent in resolved.parents:
        if parent.name == "node" and parent.parent.name == "installs":
            return parent
    return resolved.parent.parent
