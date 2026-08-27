from __future__ import annotations

import json
import sys
from pathlib import Path

from harnesslens.benchmarks.benchmark_splits import load_benchmark_split
from harnesslens.benchmarks.tau2_driver import Tau2Limits
from harnesslens.evaluation.rollout_bridge import RolloutRequest


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: native_tau2_worker.py INPUT.json")
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    response = run_payload(payload)
    print(json.dumps(response.to_dict(), ensure_ascii=False), flush=True)
    return 0


def run_payload(payload: dict) -> object:
    raw_request = payload["request"]
    request = RolloutRequest(
        request_id=str(raw_request["request_id"]),
        run_id=str(raw_request["run_id"]),
        scope=str(raw_request["scope"]),
        harness_version=str(raw_request["harness_version"]),
        task_repeats={
            str(key): int(value)
            for key, value in raw_request["task_repeats"].items()
        },
        max_concurrency=int(raw_request["max_concurrency"]),
        purpose=str(raw_request["purpose"]),
        pairing_offsets={
            str(key): int(value)
            for key, value in raw_request["pairing_offsets"].items()
        },
    )
    limits = Tau2Limits(**payload["limits"])
    kwargs = {
        "repo_root": payload["repo_root"],
        "run_root": payload["run_root"],
        "split": load_benchmark_split(payload["benchmark"]),
        "request": request,
        "retrieval_config": payload.get("retrieval_config"),
        "limits": limits,
        "harness_manifest": payload["harness_manifest"],
    }
    harness = str(payload["harness"])
    if harness == "opencode":
        from harnesslens.benchmarks.opencode_tau2 import (
            run_opencode_tau2_test_baseline,
        )

        response = run_opencode_tau2_test_baseline(**kwargs)
    elif harness == "pi":
        from harnesslens.benchmarks.pi_tau2 import run_pi_tau2_test_baseline

        response = run_pi_tau2_test_baseline(**kwargs)
    elif harness == "codex":
        from harnesslens.benchmarks.codex_tau2 import run_codex_tau2_test_baseline

        response = run_codex_tau2_test_baseline(**kwargs)
    else:
        raise ValueError(f"unsupported native Tau2 harness: {harness}")
    return response


if __name__ == "__main__":
    raise SystemExit(main())
