from __future__ import annotations

import contextlib
import copy
import fcntl
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping


CACHE_SCHEMA = "terminal-bench.shared-trial.v1"


def workspace_root(start: str | Path) -> Path:
    path = Path(start).resolve()
    candidates = (path, *path.parents)
    for candidate in candidates:
        if (candidate / "harness_autoiter").is_dir() and (candidate / "baseline").is_dir():
            return candidate
    return path


def cache_root(start: str | Path) -> Path:
    configured = str(os.environ.get("HAI_TERMINAL_BENCH_CACHE") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return workspace_root(start) / ".cache" / "terminal_bench" / "shared_trials"


def stable_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def hash_path(path: str | Path) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    if not root.exists():
        digest.update(b"missing")
        return digest.hexdigest()
    if root.is_file():
        digest.update(root.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(root.read_bytes())
        return digest.hexdigest()
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def is_cacheable(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("task_id")
        and "reward" in row
        and str(row.get("status") or "completed") == "completed"
        and not row.get("error")
        and not row.get("infrastructure_error")
        and row.get("verifier_completed", True)
    )


@dataclass
class CacheEntry:
    key: str
    manifest: dict[str, Any]
    path: Path

    def load(self) -> dict[str, Any] | None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            payload.get("schema") != CACHE_SCHEMA
            or payload.get("key") != self.key
            or payload.get("manifest") != self.manifest
        ):
            return None
        row = payload.get("trajectory")
        if not isinstance(row, Mapping) or not is_cacheable(row):
            return None
        result = copy.deepcopy(dict(row))
        result["shared_cache"] = {
            "hit": True,
            "key": self.key,
            "object": str(self.path),
        }
        return result

    def store(self, row: Mapping[str, Any]) -> bool:
        if not is_cacheable(row):
            return False
        trajectory = copy.deepcopy(dict(row))
        trajectory.pop("shared_cache", None)
        payload = {
            "schema": CACHE_SCHEMA,
            "key": self.key,
            "manifest": self.manifest,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "trajectory": trajectory,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw_tmp = tempfile.mkstemp(prefix=f".{self.key}.", suffix=".tmp", dir=self.path.parent)
        tmp = Path(raw_tmp)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        finally:
            tmp.unlink(missing_ok=True)
        return True


@contextlib.contextmanager
def locked_entry(start: str | Path, manifest: Mapping[str, Any]) -> Iterator[CacheEntry]:
    normalized = json.loads(json.dumps(dict(manifest), sort_keys=True, ensure_ascii=True))
    key = stable_hash(normalized)
    root = cache_root(start)
    path = root / "objects" / key[:2] / f"{key}.json"
    lock_path = root / "locks" / key[:2] / f"{key}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield CacheEntry(key=key, manifest=normalized, path=path)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
