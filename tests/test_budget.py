import pytest

from harnesslens.core.budget import CreationBudget


def test_creation_budget_counts_launched_instances_and_baseline(tmp_path):
    budget = CreationBudget(tmp_path / "budget.json", total=120, baseline_used=60)
    budget.reserve_job("experience", metadata={"role": "experience"})
    assert budget.status() == {
        "total": 120,
        "baseline_used": 60,
        "created": 0,
        "reserved": 1,
        "used": 60,
        "remaining": 59,
    }

    budget.claim_launch("experience")
    budget.mark_launched("experience")
    budget.settle_job("experience", outcome="completed")

    assert budget.status()["used"] == 61
    assert budget.status()["remaining"] == 59


def test_creation_budget_imports_reused_analysis_cost(tmp_path):
    budget = CreationBudget(tmp_path / "budget.json", total=100, baseline_used=60)

    record = budget.import_settled_usage(
        "analysis-reuse",
        creation_count=17,
        metadata={"source_run": "prior-run"},
    )

    assert record["status"] == "settled"
    assert record["imported"] is True
    assert budget.status()["created"] == 17
    assert budget.status()["remaining"] == 23


def test_creation_budget_refunds_only_before_launch(tmp_path):
    budget = CreationBudget(tmp_path / "budget.json", total=3, baseline_used=0)
    budget.reserve_job("before")
    budget.claim_launch("before")
    budget.refund_before_launch("before", reason="binary missing")
    assert budget.status()["remaining"] == 3

    budget.reserve_job("after")
    budget.mark_launched("after")
    with pytest.raises(ValueError, match="cannot transition"):
        budget.refund_before_launch("after", reason="too late")


def test_creation_budget_reserves_rollout_batch_atomically(tmp_path):
    budget = CreationBudget(tmp_path / "budget.json", total=12, baseline_used=0)
    budget.reserve_job("five-tasks", creation_count=10)
    assert budget.status()["remaining"] == 2

    with pytest.raises(ValueError, match="exhausted"):
        budget.reserve_job("second-batch", creation_count=3)
    assert budget.status()["reserved"] == 10


def test_creation_budget_retry_ids_preserve_failed_attempts(tmp_path):
    budget = CreationBudget(tmp_path / "budget.json", total=4, baseline_used=0)
    assert budget.next_attempt_id("job") == "job"
    budget.reserve_job("job")
    budget.mark_launched("job")
    with pytest.raises(ValueError, match="still active"):
        budget.next_attempt_id("job")
    budget.settle_job("job", outcome="network_blocked")

    assert budget.next_attempt_id("job") == "job-retry-02"


def test_creation_budget_recovers_interrupted_controller_jobs(tmp_path):
    budget = CreationBudget(tmp_path / "budget.json", total=5, baseline_used=0)
    budget.reserve_job("not-launched")
    budget.reserve_job("launch-claimed")
    budget.claim_launch("launch-claimed")
    budget.reserve_job("launched")
    budget.mark_launched("launched")

    recovered = budget.recover_interrupted_jobs(reason="controller restart")

    by_id = {item["job_id"]: item for item in recovered}
    assert by_id["not-launched"]["status"] == "refunded_before_launch"
    assert by_id["launch-claimed"]["status"] == "settled"
    assert by_id["launched"]["status"] == "settled"
    assert budget.status()["created"] == 2
    assert budget.status()["reserved"] == 0
    assert budget.status()["remaining"] == 3
    assert budget.next_attempt_id("launched") == "launched-retry-02"


def test_creation_budget_can_audit_correct_proven_prelaunch_failure(tmp_path):
    budget = CreationBudget(tmp_path / "budget.json", total=4, baseline_used=0)
    budget.reserve_job("rollout", creation_count=3)
    budget.mark_launched("rollout")
    budget.settle_job("rollout", outcome="failed", details="resource guard")

    budget.correct_prelaunch_failure(
        "rollout", evidence={"intelligent_sessions_created": 0, "reason": "preflight"}
    )

    assert budget.status()["remaining"] == 4


def test_creation_budget_can_audit_refund_system_invalidated_result(tmp_path):
    budget = CreationBudget(tmp_path / "budget.json", total=4, baseline_used=0)
    budget.reserve_job("review")
    budget.mark_launched("review")
    budget.settle_job("review", outcome="completed", details="unused output")

    corrected = budget.correct_invalidated_job(
        "review",
        evidence={
            "result_used": False,
            "reason": "runner validation defect rejected an otherwise valid output",
        },
    )

    assert corrected["status"] == "refunded_after_launch"
    assert corrected["postlaunch_correction"]["previous_status"] == "settled"
    assert budget.status()["remaining"] == 4
    assert budget.next_attempt_id("review") == "review-retry-02"


def test_creation_budget_does_not_refund_a_used_result(tmp_path):
    budget = CreationBudget(tmp_path / "budget.json", total=4, baseline_used=0)
    budget.reserve_job("review")
    budget.mark_launched("review")

    with pytest.raises(ValueError, match="unused result"):
        budget.correct_invalidated_job(
            "review",
            evidence={"result_used": True, "reason": "result selected"},
        )

    assert budget.status()["remaining"] == 3
