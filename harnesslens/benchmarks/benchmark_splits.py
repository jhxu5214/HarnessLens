from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class BenchmarkSplit:
    benchmark: str
    cell: str
    train: tuple[str, ...]
    test: tuple[str, ...]
    local_rootless_rollout: bool

    def fingerprint(self) -> str:
        payload = {"train": list(self.train), "test": list(self.test)}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


_SPECS: Mapping[str, Mapping[str, Any]] = {
    "retail": {
        "cell": "retail",
        "config": "retail_split.json",
        "local_rootless_rollout": True,
    },
    "banking": {
        "cell": "banking_knowledge",
        "config": "banking_knowledge_split.json",
        "local_rootless_rollout": True,
    },
    "terminal-bench": {
        "cell": "terminal_bench",
        "config": "terminal_bench_split.json",
        "local_rootless_rollout": False,
    },
    "bird-mini-dev-challenging": {
        "cell": "bird_mini_dev_challenging",
        "config": "bird_mini_dev_challenging_split.json",
        "local_rootless_rollout": False,
    },
}

_ALIASES = {
    "retail": "retail",
    "tau2-retail": "retail",
    "banking": "banking",
    "banking_knowledge": "banking",
    "terminal": "terminal-bench",
    "terminal_bench": "terminal-bench",
    "terminal-bench": "terminal-bench",
    "bird": "bird-mini-dev-challenging",
    "bird_minidev": "bird-mini-dev-challenging",
    "bird-mini-dev": "bird-mini-dev-challenging",
    "bird_mini_dev_challenging": "bird-mini-dev-challenging",
    "bird-mini-dev-challenging": "bird-mini-dev-challenging",
}


def supported_test_benchmarks() -> tuple[str, ...]:
    return tuple(_SPECS)


def load_benchmark_split(benchmark: str) -> BenchmarkSplit:
    normalized = _ALIASES.get(str(benchmark).strip().lower())
    if normalized is None:
        raise ValueError(f"unsupported HarnessLens test benchmark: {benchmark}")
    spec = _SPECS[normalized]
    path = Path(__file__).resolve().parents[2] / "configs" / str(spec["config"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    train = _validated_ids(payload.get("train"), label=f"{normalized} TRAIN")
    test = _validated_ids(payload.get("test"), label=f"{normalized} TEST")
    overlap = set(train) & set(test)
    if overlap:
        raise ValueError(f"{normalized} split overlaps: {sorted(overlap)}")
    return BenchmarkSplit(
        benchmark=normalized,
        cell=str(spec["cell"]),
        train=train,
        test=test,
        local_rootless_rollout=bool(spec["local_rootless_rollout"]),
    )


def _validated_ids(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} split is empty or malformed")
    normalized = tuple(str(item) for item in value)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} split contains duplicate task IDs")
    return normalized
