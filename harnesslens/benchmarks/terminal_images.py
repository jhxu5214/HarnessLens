from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Iterable


TERMINAL_BENCH_IMAGE_TEMPLATE = "alexgshaw/{task_id}:20251031"
TERMINAL_BENCH_SHARED_NETWORK = "harnesslens-terminal-bench"


@dataclass(frozen=True)
class TerminalImagePreflight:
    image_template: str
    task_count: int
    images: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "image_template": self.image_template,
            "task_count": self.task_count,
            "images": list(self.images),
            "policy": "prewarmed_only_no_pull_no_build",
        }


def task_image_name(task_id: str) -> str:
    return TERMINAL_BENCH_IMAGE_TEMPLATE.format(task_id=str(task_id))


def require_preloaded_terminal_images(task_ids: Iterable[str]) -> TerminalImagePreflight:
    """Verify the exact TB2 client tags in the selected rootless Docker daemon.

    Terminal-Bench 2.0 task Compose files accept their client image name via
    ``T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME``. HarnessLens intentionally treats these
    as prewarmed, immutable task images: it never pulls or builds during TEST.
    """
    ids = tuple(str(task_id) for task_id in task_ids)
    images = tuple(task_image_name(task_id) for task_id in ids)
    env = dict(os.environ)
    missing: list[str] = []
    for task_id, image in zip(ids, images):
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", image],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Terminal-Bench image preflight timed out for {task_id}: {image}"
            ) from exc
        if result.returncode != 0:
            missing.append(f"{task_id} ({image})")
    if missing:
        raise RuntimeError(
            "Terminal-Bench requires prewarmed per-task images in the configured "
            "Docker daemon; refusing to pull/build during TEST. Missing: "
            + ", ".join(missing)
        )
    return TerminalImagePreflight(
        image_template=TERMINAL_BENCH_IMAGE_TEMPLATE,
        task_count=len(ids),
        images=images,
    )


def ensure_terminal_shared_network() -> str:
    """Create once, then reuse a Compose-compatible bridge for all HarnessLens trials."""
    network = os.environ.get("TB_SHARED_NETWORK", TERMINAL_BENCH_SHARED_NETWORK)
    env = dict(os.environ)
    inspect = subprocess.run(
        ["docker", "network", "inspect", network],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
        env=env,
        check=False,
    )
    if inspect.returncode == 0:
        return network
    create = subprocess.run(
        ["docker", "network", "create", "--driver", "bridge", network],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
        check=False,
    )
    if create.returncode != 0:
        raise RuntimeError(
            f"failed to create shared Terminal-Bench network {network}: {create.stderr[-600:]}"
        )
    return network
