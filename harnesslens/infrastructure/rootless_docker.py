from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


# Where the rootless dockerd keeps its state. Everything else is derived from
# it, because a host and a state directory that disagree produce a daemon that
# serves the expected socket out of the wrong (often empty) storage — which
# looks exactly like "all my images and containers vanished".
DEFAULT_DOCKER_ROOT_TEMPLATE = "~/dockers"
ROOTLESS_CONTAINER_HOST = "192.168.127.254"


def rootless_docker_root(override: str | Path | None = None) -> Path:
    """State directory of the rootless daemon (``HAI_DOCKER_ROOT``)."""
    if override is not None:
        return Path(override).expanduser().resolve()
    return Path(
        os.environ.get("HAI_DOCKER_ROOT") or DEFAULT_DOCKER_ROOT_TEMPLATE
    ).expanduser().resolve()


def rootless_docker_host(docker_root: str | Path | None = None) -> str:
    """Socket for the rootless daemon, derived from its state directory.

    An explicit ``HAI_DOCKER_HOST`` or ``DOCKER_HOST`` still wins, so a machine
    with an existing daemon needs no other configuration.
    """
    explicit = os.environ.get("HAI_DOCKER_HOST") or os.environ.get("DOCKER_HOST")
    if explicit and explicit.strip():
        return explicit.strip()
    return f"unix://{rootless_docker_root(docker_root) / 'run' / 'docker.sock'}"


def assert_host_matches_root(docker_host: str, docker_root: str | Path) -> None:
    """Refuse a unix socket that lives outside the state directory.

    Starting a daemon with ``--data-root A --host unix://B/run/docker.sock``
    succeeds and then reports an empty engine, because the socket is served out
    of a different directory than the one holding the images. Catching it here
    turns a mystifying outcome into a message.
    """
    if not docker_host.startswith("unix://"):
        return
    socket_path = Path(docker_host[len("unix://") :])
    root = Path(docker_root).expanduser().resolve()
    try:
        socket_path.resolve().relative_to(root)
    except ValueError:
        raise ValueError(
            "rootless docker host and state directory disagree: the socket "
            f"{socket_path} is not inside {root}. Starting a daemon this way "
            "serves the expected socket from the wrong storage, so it reports "
            "no images and no containers. Set HAI_DOCKER_ROOT to the directory "
            "that already holds them, or pass a matching docker_root."
        ) from None


# Import-time snapshots, kept because entrypoints use them as argparse
# defaults. Call the functions above when the environment may have changed.
DEFAULT_DOCKER_ROOT = rootless_docker_root()
DEFAULT_DOCKER_HOST = rootless_docker_host()


def ensure_rootless_docker(
    docker_host: str | None = None,
    *,
    docker_root: str | Path | None = None,
    timeout_s: int = 30,
) -> None:
    root = rootless_docker_root(docker_root)
    host = str(docker_host or rootless_docker_host(root))
    assert_host_matches_root(host, root)

    env = dict(os.environ)
    env["DOCKER_HOST"] = host
    if _docker_ready(env):
        return

    runtime_dir = Path(
        env.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    )
    _remove_stale_pid(runtime_dir / "docker.pid")
    for path in (root / "run", root / "logs", root / "exec"):
        path.mkdir(parents=True, exist_ok=True)

    env["DOCKERD_ROOTLESS_ROOTLESSKIT_DISABLE_HOST_LOOPBACK"] = "false"
    log_handle = (root / "logs" / "dockerd-rootless.log").open(
        "w", encoding="utf-8"
    )
    process = subprocess.Popen(
        [
            "dockerd-rootless.sh",
            "--data-root",
            str(root),
            "--exec-root",
            str(root / "exec"),
            "--host",
            str(host),
        ],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    log_handle.close()
    deadline = time.monotonic() + int(timeout_s)
    while time.monotonic() < deadline:
        if _docker_ready(env):
            return
        if process.poll() is not None:
            break
        time.sleep(1)
    raise RuntimeError(
        f"rootless Docker did not become ready at {host}; "
        f"see {root / 'logs' / 'dockerd-rootless.log'}"
    )


def _docker_ready(env: dict[str, str]) -> bool:
    result = subprocess.run(
        ["docker", "version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5,
        check=False,
        env=env,
    )
    return result.returncode == 0


def _remove_stale_pid(pid_path: Path) -> None:
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pid_path.unlink(missing_ok=True)
    except PermissionError:
        return
