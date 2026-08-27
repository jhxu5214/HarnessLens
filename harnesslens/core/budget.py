from __future__ import annotations

import fcntl
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from harnesslens.core.artifacts import read_json, write_json


DEFAULT_TOTAL_CREATION_BUDGET = 200


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CreationBudget:
    """Durable budget for new isolated intelligent-harness sessions.

    The method names intentionally cover the small meter protocol used by v3's stable
    OpenCode process adapter. Token reservation arguments are recorded for diagnostics
    but never gate execution.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        total: int = DEFAULT_TOTAL_CREATION_BUDGET,
        baseline_used: int = 60,
    ) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if total <= 0 or not 0 <= baseline_used <= total:
            raise ValueError("creation budget requires 0 <= baseline_used <= total")
        with self._locked():
            existing = read_json(self.path)
            if existing is None:
                write_json(
                    self.path,
                    {
                        "version": 1,
                        "total": int(total),
                        "baseline_used": int(baseline_used),
                        "jobs": {},
                        "created_at": _now(),
                    },
                )
            elif int(existing.get("total", -1)) != int(total) or int(
                existing.get("baseline_used", -1)
            ) != int(baseline_used):
                raise ValueError(
                    "creation budget configuration differs from durable ledger"
                )

    def status(self) -> dict[str, Any]:
        with self._locked():
            return self._status(self._read())

    def next_attempt_id(self, base_id: str) -> str:
        with self._locked():
            state = self._read()
            matching = [
                (job_id, item)
                for job_id, item in state["jobs"].items()
                if job_id == base_id or job_id.startswith(f"{base_id}-retry-")
            ]
            active = [
                job_id
                for job_id, item in matching
                if item.get("status") in {"reserved", "launch_claimed", "launched"}
            ]
            if active:
                raise ValueError(f"creation-budget job is still active: {active[0]}")
            if not matching:
                return str(base_id)
            return f"{base_id}-retry-{len(matching) + 1:02d}"

    def recover_interrupted_jobs(self, *, reason: str) -> list[dict[str, Any]]:
        """Close durable reservations left by a terminated controller process."""

        recovered: list[dict[str, Any]] = []
        with self._locked():
            state = self._read()
            for record in state["jobs"].values():
                if not isinstance(record, dict):
                    continue
                previous_status = str(record.get("status") or "")
                if previous_status == "reserved":
                    record.update(
                        {
                            "status": "refunded_before_launch",
                            "settled_at": _now(),
                            "reason": str(reason),
                        }
                    )
                elif previous_status in {"launch_claimed", "launched"}:
                    record.update(
                        {
                            "status": "settled",
                            "settled_at": _now(),
                            "outcome": "interrupted_controller_restart",
                            "usage": {},
                            "usage_complete": False,
                        }
                    )
                else:
                    continue
                record["recovery"] = {
                    "recovered_at": _now(),
                    "previous_status": previous_status,
                    "reason": str(reason),
                }
                recovered.append(dict(record))
            if recovered:
                write_json(self.path, state)
        return recovered

    def reserve(
        self,
        call_id: str,
        amount: int = 1,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        count = int((metadata or {}).get("creation_count", 1))
        if count <= 0:
            raise ValueError("creation_count must be positive")
        with self._locked():
            state = self._read()
            jobs = state["jobs"]
            if call_id in jobs:
                return dict(jobs[call_id])
            status = self._status(state)
            if count > status["remaining"]:
                raise ValueError(
                    "creation budget exhausted: "
                    f"requested={count} remaining={status['remaining']}"
                )
            record = {
                "job_id": str(call_id),
                "status": "reserved",
                "creation_count": count,
                "reserved_at": _now(),
                "metadata": dict(metadata or {}),
                "ignored_token_reservation": int(amount),
            }
            jobs[str(call_id)] = record
            write_json(self.path, state)
            return dict(record)

    def reserve_job(
        self,
        job_id: str,
        *,
        creation_count: int = 1,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.reserve(
            job_id,
            0,
            metadata={**dict(metadata or {}), "creation_count": int(creation_count)},
        )

    def import_settled_usage(
        self,
        job_id: str,
        *,
        creation_count: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Charge previously created sessions whose artifacts are reused verbatim."""

        count = int(creation_count)
        if count <= 0:
            raise ValueError("imported creation_count must be positive")
        with self._locked():
            state = self._read()
            jobs = state["jobs"]
            if job_id in jobs:
                record = jobs[job_id]
                if (
                    record.get("status") != "settled"
                    or int(record.get("creation_count") or 0) != count
                    or record.get("imported") is not True
                ):
                    raise ValueError(
                        "imported creation usage differs from durable ledger"
                    )
                return dict(record)
            if count > self._status(state)["remaining"]:
                raise ValueError(
                    "creation budget exhausted by reused analysis: "
                    f"requested={count} remaining={self._status(state)['remaining']}"
                )
            record = {
                "job_id": str(job_id),
                "status": "settled",
                "creation_count": count,
                "imported": True,
                "settled_at": _now(),
                "outcome": "reused_artifact",
                "usage": {},
                "usage_complete": False,
                "metadata": dict(metadata or {}),
            }
            jobs[str(job_id)] = record
            write_json(self.path, state)
            return dict(record)

    def claim_launch(self, call_id: str) -> dict[str, Any]:
        return self._transition(call_id, {"reserved"}, "launch_claimed")

    def mark_launched(self, call_id: str) -> dict[str, Any]:
        return self._transition(
            call_id,
            {"reserved", "launch_claimed"},
            "launched",
            extra={"launched_at": _now()},
        )

    def refund_before_launch(self, call_id: str, *, reason: str) -> dict[str, Any]:
        return self._transition(
            call_id,
            {"reserved", "launch_claimed"},
            "refunded_before_launch",
            extra={"settled_at": _now(), "reason": str(reason)},
        )

    def settle(
        self,
        call_id: str,
        *,
        usage: Mapping[str, Any] | None = None,
        outcome: str,
        usage_complete: bool = False,
    ) -> dict[str, Any]:
        return self._transition(
            call_id,
            {"launched"},
            "settled",
            extra={
                "settled_at": _now(),
                "outcome": str(outcome),
                "usage": dict(usage or {}),
                "usage_complete": bool(usage_complete),
            },
        )

    def settle_job(
        self, job_id: str, *, outcome: str, details: Any = None
    ) -> dict[str, Any]:
        return self.settle(
            job_id,
            usage={"details": details} if details is not None else {},
            outcome=outcome,
            usage_complete=False,
        )

    def correct_prelaunch_failure(
        self, job_id: str, *, evidence: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Correct a conservative launch mark when durable evidence proves no session started."""
        with self._locked():
            state = self._read()
            record = state["jobs"].get(str(job_id))
            if not isinstance(record, dict):
                raise ValueError(f"unknown creation-budget job: {job_id}")
            if record.get("status") != "settled" or record.get("outcome") != "failed":
                raise ValueError("prelaunch correction requires a settled failed job")
            if not evidence or evidence.get("intelligent_sessions_created") != 0:
                raise ValueError("prelaunch correction requires zero-session evidence")
            record["status"] = "refunded_before_launch"
            record["prelaunch_correction"] = {
                "corrected_at": _now(),
                "evidence": dict(evidence),
            }
            write_json(self.path, state)
            return dict(record)

    def correct_invalidated_job(
        self, job_id: str, *, evidence: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Refund a launched session whose result was invalidated by the runner itself."""
        with self._locked():
            state = self._read()
            record = state["jobs"].get(str(job_id))
            if not isinstance(record, dict):
                raise ValueError(f"unknown creation-budget job: {job_id}")
            previous_status = str(record.get("status") or "")
            if previous_status not in {"launched", "settled"}:
                raise ValueError(
                    "postlaunch correction requires a launched or settled job"
                )
            if (
                not evidence
                or evidence.get("result_used") is not False
                or not str(evidence.get("reason") or "").strip()
            ):
                raise ValueError(
                    "postlaunch correction requires an unused result and an audit reason"
                )
            record["status"] = "refunded_after_launch"
            record["postlaunch_correction"] = {
                "corrected_at": _now(),
                "previous_status": previous_status,
                "evidence": dict(evidence),
            }
            write_json(self.path, state)
            return dict(record)

    def _transition(
        self,
        call_id: str,
        allowed: set[str],
        target: str,
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._locked():
            state = self._read()
            record = state["jobs"].get(str(call_id))
            if not isinstance(record, dict):
                raise ValueError(f"unknown creation-budget job: {call_id}")
            current = str(record.get("status") or "")
            if current not in allowed:
                if current == target or (target == "settled" and current == "settled"):
                    return dict(record)
                raise ValueError(
                    f"creation-budget job {call_id!r} cannot transition {current} -> {target}"
                )
            record.update(dict(extra or {}))
            record["status"] = target
            write_json(self.path, state)
            return dict(record)

    def _read(self) -> dict[str, Any]:
        state = read_json(self.path)
        if not isinstance(state, dict) or not isinstance(state.get("jobs"), dict):
            raise RuntimeError(f"invalid creation budget ledger: {self.path}")
        return state

    @staticmethod
    def _status(state: Mapping[str, Any]) -> dict[str, Any]:
        jobs = state.get("jobs") or {}
        reserved = sum(
            int(item.get("creation_count") or 0)
            for item in jobs.values()
            if item.get("status") in {"reserved", "launch_claimed"}
        )
        launched = sum(
            int(item.get("creation_count") or 0)
            for item in jobs.values()
            if item.get("status") in {"launched", "settled"}
        )
        baseline = int(state.get("baseline_used") or 0)
        total = int(state.get("total") or 0)
        return {
            "total": total,
            "baseline_used": baseline,
            "created": launched,
            "reserved": reserved,
            "used": baseline + launched,
            "remaining": total - baseline - launched - reserved,
        }

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
