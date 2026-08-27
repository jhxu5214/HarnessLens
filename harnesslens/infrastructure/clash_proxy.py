from __future__ import annotations

import os
from urllib.parse import urlsplit


def configure_terminal_clash_proxy() -> str:
    """Validate Clash and enable a trial-local proxy inside each task container.

    The host endpoint is only a launch preflight. Terminal-Bench copies the
    Clash runtime into each task container and points package traffic at that
    container-local proxy. Model endpoints are added to ``NO_PROXY`` by the
    terminal runtime so inference remains direct.
    """
    value = str(os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY") or "")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or not parsed.port:
        raise RuntimeError(
            "Terminal-Bench requires `source ~/.bashrc && clashctl on` before launch"
        )
    if parsed.hostname in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError(
            "clashctl proxy must use allow-lan and export a container-reachable host address"
        )
    os.environ["TB_CONTAINER_PROXY_URL"] = value
    os.environ["TB_ENABLE_CONTAINER_CLASH"] = "1"
    return value
