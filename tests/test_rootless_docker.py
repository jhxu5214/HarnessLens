from __future__ import annotations

from types import SimpleNamespace

import pytest

from harnesslens.infrastructure import rootless_docker


def _socket_in(root) -> str:
    """The socket a daemon with this state directory is expected to serve."""
    return f"unix://{root / 'run' / 'docker.sock'}"


def test_existing_rootless_docker_is_reused(monkeypatch, tmp_path):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(rootless_docker.subprocess, "run", fake_run)
    monkeypatch.setattr(
        rootless_docker.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected start")),
    )

    host = _socket_in(tmp_path)
    rootless_docker.ensure_rootless_docker(host, docker_root=tmp_path)

    assert calls[0][1]["env"]["DOCKER_HOST"] == host


def test_started_rootless_docker_allows_host_loopback(monkeypatch, tmp_path):
    readiness = iter([False, True])
    captured = {}

    monkeypatch.setattr(rootless_docker, "_docker_ready", lambda env: next(readiness))

    class FakeProcess:
        def poll(self):
            return None

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(rootless_docker.subprocess, "Popen", fake_popen)

    host = _socket_in(tmp_path)
    rootless_docker.ensure_rootless_docker(host, docker_root=tmp_path)

    assert captured["env"][
        "DOCKERD_ROOTLESS_ROOTLESSKIT_DISABLE_HOST_LOOPBACK"
    ] == "false"
    assert captured["start_new_session"] is True
    assert captured["args"][-1] == host
    # the daemon must store its state where the socket says it does
    assert captured["args"][captured["args"].index("--data-root") + 1] == str(
        tmp_path.resolve()
    )


def test_host_outside_the_state_directory_is_refused(monkeypatch, tmp_path):
    """A daemon serving one directory's socket out of another reports an empty
    engine, which reads as data loss. Refuse the pair instead."""
    monkeypatch.setattr(
        rootless_docker.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not start")),
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    with pytest.raises(ValueError, match="disagree"):
        rootless_docker.ensure_rootless_docker(
            _socket_in(elsewhere), docker_root=tmp_path / "state"
        )


def test_host_is_derived_from_the_state_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("HAI_DOCKER_HOST", raising=False)
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.setenv("HAI_DOCKER_ROOT", str(tmp_path))

    assert rootless_docker.rootless_docker_root() == tmp_path.resolve()
    assert rootless_docker.rootless_docker_host() == _socket_in(tmp_path.resolve())


def test_explicit_docker_host_still_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCKER_HOST", "tcp://127.0.0.1:2375")
    monkeypatch.setenv("HAI_DOCKER_ROOT", str(tmp_path))

    assert rootless_docker.rootless_docker_host() == "tcp://127.0.0.1:2375"
    # a tcp host makes no claim about local storage, so it is not constrained
    rootless_docker.assert_host_matches_root("tcp://127.0.0.1:2375", tmp_path)
