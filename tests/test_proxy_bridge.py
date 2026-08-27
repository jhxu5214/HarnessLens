from __future__ import annotations

import socket
import threading

from harnesslens.infrastructure.proxy_bridge import TcpProxyBridge, host_route_address


def test_tcp_proxy_bridge_relays_bytes():
    upstream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    upstream.bind(("127.0.0.1", 0))
    upstream.listen(2)
    upstream_port = upstream.getsockname()[1]

    def echo_after_probe():
        for _ in range(2):
            connection, _ = upstream.accept()
            data = connection.recv(1024)
            if data:
                connection.sendall(data.upper())
            connection.close()
        upstream.close()

    threading.Thread(target=echo_after_probe, daemon=True).start()
    with TcpProxyBridge("127.0.0.1", upstream_port, listen_host="127.0.0.1") as bridge:
        with socket.create_connection(("127.0.0.1", bridge.port), timeout=5) as client:
            client.sendall(b"hello")
            assert client.recv(1024) == b"HELLO"


def test_host_route_address_is_an_ipv4_address():
    parts = host_route_address().split(".")
    assert len(parts) == 4
    assert all(0 <= int(part) <= 255 for part in parts)
