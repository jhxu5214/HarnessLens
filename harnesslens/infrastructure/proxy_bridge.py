from __future__ import annotations

import socket
import threading
from dataclasses import dataclass, field


@dataclass
class TcpProxyBridge:
    upstream_host: str
    upstream_port: int
    listen_host: str = "0.0.0.0"
    listen_port: int = 0
    _listener: socket.socket | None = field(default=None, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)

    @property
    def port(self) -> int:
        if self._listener is None:
            raise RuntimeError("proxy bridge has not started")
        return int(self._listener.getsockname()[1])

    def start(self) -> "TcpProxyBridge":
        with socket.create_connection(
            (self.upstream_host, int(self.upstream_port)), timeout=5
        ):
            pass
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.listen_host, int(self.listen_port)))
        listener.listen(64)
        listener.settimeout(1)
        self._listener = listener
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        if self._listener is not None:
            self._listener.close()
            self._listener = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def __enter__(self) -> "TcpProxyBridge":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _serve(self) -> None:
        assert self._listener is not None
        while not self._stop.is_set():
            try:
                client, _ = self._listener.accept()
            except (OSError, socket.timeout):
                continue
            threading.Thread(target=self._relay, args=(client,), daemon=True).start()

    def _relay(self, client: socket.socket) -> None:
        try:
            upstream = socket.create_connection(
                (self.upstream_host, int(self.upstream_port)), timeout=10
            )
        except OSError:
            client.close()
            return
        client.settimeout(None)
        upstream.settimeout(None)
        threads = (
            threading.Thread(target=_copy, args=(client, upstream), daemon=True),
            threading.Thread(target=_copy, args=(upstream, client), daemon=True),
        )
        for thread in threads:
            thread.start()


def _copy(source: socket.socket, destination: socket.socket) -> None:
    try:
        while data := source.recv(65536):
            destination.sendall(data)
    except OSError:
        pass
    finally:
        try:
            destination.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        source.close()


def host_route_address() -> str:
    """Return the host IPv4 source address containers can reach via normal routing."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("1.1.1.1", 53))
        return str(probe.getsockname()[0])
    finally:
        probe.close()
