import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import harnesslens.evaluation.rollout_bridge as rollout_bridge
from harnesslens.evolution.baseline import build_baseline_fingerprint
from harnesslens.core.budget import CreationBudget
from harnesslens.evolution.rollout import (
    PairedRolloutResult,
    ROLLOUT_RUNTIME_LIMITS,
    RolloutModule,
    RolloutResult,
    annotate_paired_validity,
    paired_infrastructure_failures,
    _settle_recovered_rollout_job,
    _merge_trial_retries,
    _next_trial_retry_label,
    _trial_is_infrastructure_failure,
    validate_rollout_task_ids,
)
from harnesslens.evaluation.rollout_bridge import (
    IncompleteRolloutTraceError,
    RolloutRequest,
    TrainRolloutRecord,
    TrainRolloutService,
    CellHarnessRepository,
    _cleanup_rollout_group_workspace,
    _summarize,
    validate_native_rollout_interactions,
)


def test_rollout_pass_at_1_uses_all_independent_trials():
    metrics = _summarize(
        [
            TrainRolloutRecord("0", (0.0, 1.0), "candidate-01"),
            TrainRolloutRecord("1", (0.0, 0.0), "candidate-01"),
        ]
    )

    assert metrics["pass_at_1"] == 0.25
    assert metrics["trial_success_rate"] == 0.25
    assert metrics["pass_at_2"] == 0.5


def test_evaluator_error_after_interaction_is_not_an_infrastructure_failure():
    assert not _trial_is_infrastructure_failure(
        {
            "error": "eval ValueError: user tool replay mismatch",
            "n_messages": 12,
            "n_tool_calls": 3,
            "termination": "TerminationReason.USER_STOP",
        }
    )


def test_pre_interaction_worker_error_remains_an_infrastructure_failure():
    assert _trial_is_infrastructure_failure(
        {
            "error": "FileNotFoundError: missing runtime file",
            "n_messages": 0,
            "n_tool_calls": 0,
            "termination": "",
        }
    )


def test_trial_retry_label_uses_a_new_attempt_after_a_failed_retry(tmp_path):
    root = tmp_path / "rollout"
    root.mkdir()
    base = "screen-infra-trial-retry-task_1-s00"
    (root / f"{base}.json").write_text("{}", encoding="utf-8")

    assert _next_trial_retry_label(root, "screen", "task_1", 0) == f"{base}-attempt-02"


def test_single_trial_retry_restores_its_requested_pairing_slot(tmp_path):
    initial_path = tmp_path / "initial.json"
    initial = RolloutResult(
        {
            "per_task": {
                "task_1": {
                    "trial_summaries": [
                        {"pairing_slot": 0, "reward": 1, "termination": "completed", "error": ""},
                        {"pairing_slot": 1, "reward": 0, "termination": "error", "error": "socket gone"},
                    ],
                    "trajectory_paths": ["original-0", "original-1"],
                }
            },
            "metrics": {},
            "budget_spent": 2,
        },
        str(initial_path),
    )
    retry = RolloutResult(
        {
            "per_task": {
                "task_1": {
                    "trial_summaries": [
                        {"pairing_slot": 0, "reward": 0, "termination": "completed", "error": ""}
                    ],
                    "trajectory_paths": ["retry-1"],
                }
            }
        },
        str(tmp_path / "retry.json"),
    )

    merged = _merge_trial_retries(
        initial,
        [({"task_id": "task_1", "pairing_slot": 1, "error": "socket gone"}, retry)],
    )

    summary = merged.output["per_task"]["task_1"]["trial_summaries"][1]
    assert summary["pairing_slot"] == 1
    assert summary["error"] == ""
    assert merged.output["per_task"]["task_1"]["trajectory_paths"] == ["original-0", "retry-1"]
    assert merged.output["budget_spent"] == 3


@pytest.mark.parametrize("harness", ["pi", "codex"])
def test_native_rollout_requires_retained_nonempty_api_trace(tmp_path, harness):
    trajectory = tmp_path / "trial_0001.jsonl"
    row = {
        "harness": harness,
        "messages": [{"role": "assistant", "content": "hello"}],
        "tool_definitions": [],
        "api_calls_jsonl": "trial_0001.api_calls.jsonl",
    }
    trajectory.write_text(json.dumps(row) + "\n", encoding="utf-8")
    record = TrainRolloutRecord(
        task_id="0",
        rewards=(0.0,),
        harness_version="candidate-01",
        trajectory_paths=(str(trajectory),),
    )

    with pytest.raises(
        IncompleteRolloutTraceError,
        match=f"missing {harness.title()} API trace",
    ):
        validate_native_rollout_interactions([record], harness=harness)

    (tmp_path / "trial_0001.api_calls.jsonl").write_text(
        '{"request": {}}\n', encoding="utf-8"
    )
    validate_native_rollout_interactions([record], harness=harness)


def test_rollout_accepts_five_train_tasks():
    validate_rollout_task_ids(("0", "1", "2", "3", "4"), train_task_ids=("0", "1", "2", "3", "4"))


def test_rollout_rejects_less_than_five_or_outside_train():
    with pytest.raises(ValueError, match="at least five"):
        validate_rollout_task_ids(("0", "1", "2", "3"), train_task_ids=("0", "1", "2", "3"))
    with pytest.raises(ValueError, match="outside TRAIN"):
        validate_rollout_task_ids(("0", "1", "2", "3", "x"), train_task_ids=("0", "1", "2", "3", "4"))


def test_rollout_accepts_controller_bounded_residual_probe():
    validate_rollout_task_ids(
        ("0", "1"),
        train_task_ids=("0", "1", "2", "3", "4"),
        minimum_task_count=2,
    )

    with pytest.raises(ValueError, match="at least 2"):
        validate_rollout_task_ids(
            ("0",),
            train_task_ids=("0", "1", "2", "3", "4"),
            minimum_task_count=2,
        )


def test_channel_preflight_uses_one_task_and_one_repeat(tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    (run_root / "experience").mkdir(parents=True)
    (run_root / "experience" / "baseline_source_index.json").write_text(
        json.dumps({"task_ids": ["0", "1", "2", "3", "4"]}), encoding="utf-8"
    )
    decision = run_root / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "harness_version": "candidate-01",
                "rollout_request": {
                    "task_ids": ["2", "3", "4", "0", "1"],
                    "rationale": "screen",
                },
            }
        ),
        encoding="utf-8",
    )
    module = RolloutModule(
        repo_root=tmp_path,
        run_root=run_root,
        budget=CreationBudget(run_root / "budget.json", total=10, baseline_used=0),
    )
    calls = []
    monkeypatch.setattr(
        module,
        "run_version",
        lambda **kwargs: calls.append(kwargs) or RolloutResult({}, str(run_root / "rollout" / "preflight.json")),
    )

    module.run_channel_preflight_from_main(main_decision=decision, label="preflight")

    assert calls[0]["task_ids"] == ("2",)
    assert calls[0]["minimum_task_count"] == 1
    assert calls[0]["repeats"] == 1


def test_pair_rollout_uses_exact_non_v0_parent_with_matching_offset(
    tmp_path, monkeypatch
):
    run_root = tmp_path / "run"
    (run_root / "experience").mkdir(parents=True)
    (run_root / "experience" / "baseline_source_index.json").write_text(
        json.dumps({"task_ids": ["0", "1", "2", "3", "4"]}), encoding="utf-8"
    )
    decision = run_root / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "harness_version": "candidate-01",
                "candidate": {"parent_version": "incumbent-00"},
                "rollout_request": {
                    "task_ids": ["0", "1", "2", "3", "4"],
                    "rationale": "paired check",
                },
            }
        ),
        encoding="utf-8",
    )
    module = RolloutModule(
        repo_root=tmp_path,
        run_root=run_root,
        budget=CreationBudget(run_root / "budget.json", total=100, baseline_used=0),
    )
    calls = []

    def fake_run_version(**kwargs):
        calls.append(kwargs)
        return RolloutResult(
            {
                "harness_version": kwargs["harness_version"],
                "per_task": {
                    "0": {
                        "trial_summaries": [
                            {
                                "pairing_slot": kwargs["pairing_offset"] + trial,
                                "termination": "completed",
                                "error": "",
                            }
                            for trial in range(2)
                        ]
                    }
                },
                "metrics": {"pass_at_1": 1.0, "trial_success_rate": 1.0},
            },
            str(run_root / "rollout" / f"{kwargs['label']}.json"),
        )

    monkeypatch.setattr(module, "run_version", fake_run_version)

    pair = module.run_pair_from_main(
        main_decision=decision, label="screen", pairing_offset=2
    )

    assert pair.parent.output["harness_version"] == "incumbent-00"
    assert pair.candidate.output["harness_version"] == "candidate-01"
    assert {call["pairing_offset"] for call in calls} == {2}
    assert sum(call["max_concurrency"] for call in calls) == 20


def test_fresh_seed_confirmation_runs_an_exact_v0_parent(tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    (run_root / "experience").mkdir(parents=True)
    (run_root / "experience" / "baseline_source_index.json").write_text(
        json.dumps({"task_ids": ["0", "1", "2", "3", "4"]}), encoding="utf-8"
    )
    decision = run_root / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "harness_version": "candidate-01",
                "candidate": {"parent_version": "v0"},
                "rollout_request": {
                    "task_ids": ["0", "1", "2", "3", "4"],
                    "rationale": "fresh confirmation",
                },
            }
        ),
        encoding="utf-8",
    )
    module = RolloutModule(
        repo_root=tmp_path,
        run_root=run_root,
        budget=CreationBudget(run_root / "budget.json", total=100, baseline_used=0),
    )
    calls = []
    monkeypatch.setattr(
        module,
        "run_version",
        lambda **kwargs: calls.append(kwargs)
        or RolloutResult(
            {
                "harness_version": kwargs["harness_version"],
                "per_task": {
                    "0": {
                        "trial_summaries": [
                            {
                                "pairing_slot": kwargs["pairing_offset"] + trial,
                                "termination": "completed",
                                "error": "",
                            }
                            for trial in range(2)
                        ]
                    }
                },
                "metrics": {"pass_at_1": 1.0, "trial_success_rate": 1.0},
            },
            str(run_root / "rollout" / f"{kwargs['label']}.json"),
        ),
    )

    pair = module.run_pair_from_main(
        main_decision=decision, label="confirmation", pairing_offset=2
    )

    assert pair.parent.output["harness_version"] == "v0"
    assert {call["pairing_offset"] for call in calls} == {2}


def test_pair_rollout_accepts_controller_confirmation_task_override(
    tmp_path, monkeypatch
):
    run_root = tmp_path / "run"
    (run_root / "experience").mkdir(parents=True)
    train_ids = [str(index) for index in range(10)]
    (run_root / "experience" / "baseline_source_index.json").write_text(
        json.dumps({"task_ids": train_ids}), encoding="utf-8"
    )
    decision = run_root / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "harness_version": "candidate-01",
                "candidate": {"parent_version": "v0"},
                "rollout_request": {
                    "task_ids": train_ids[:5],
                    "rationale": "screen tasks",
                },
            }
        ),
        encoding="utf-8",
    )
    module = RolloutModule(
        repo_root=tmp_path,
        run_root=run_root,
        budget=CreationBudget(run_root / "budget.json", total=100, baseline_used=0),
    )
    calls = []

    def fake_run_version(**kwargs):
        calls.append(kwargs)
        per_task = {
            task_id: {
                "trial_summaries": [
                    {
                        "pairing_slot": kwargs["pairing_offset"] + trial,
                        "termination": "completed",
                        "error": "",
                    }
                    for trial in range(2)
                ]
            }
            for task_id in kwargs["task_ids"]
        }
        return RolloutResult(
            {
                "harness_version": kwargs["harness_version"],
                "per_task": per_task,
                "metrics": {"pass_at_1": 1.0, "trial_success_rate": 1.0},
            },
            str(run_root / "rollout" / f"{kwargs['label']}.json"),
        )

    monkeypatch.setattr(module, "run_version", fake_run_version)

    module.run_pair_from_main(
        main_decision=decision,
        label="confirmation",
        task_ids_override=train_ids[5:],
    )

    assert calls
    assert all(tuple(call["task_ids"]) == tuple(train_ids[5:]) for call in calls)


def test_exact_pair_retries_only_invalid_trial_with_original_slot(
    tmp_path, monkeypatch
):
    run_root = tmp_path / "run"
    (run_root / "experience").mkdir(parents=True)
    task_ids = [str(index) for index in range(5)]
    (run_root / "experience" / "baseline_source_index.json").write_text(
        json.dumps({"task_ids": task_ids}), encoding="utf-8"
    )
    decision = run_root / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "harness_version": "candidate-01",
                "candidate": {"parent_version": "candidate-00"},
                "rollout_request": {
                    "task_ids": task_ids,
                    "rationale": "paired retry",
                },
            }
        ),
        encoding="utf-8",
    )
    module = RolloutModule(
        repo_root=tmp_path,
        run_root=run_root,
        budget=CreationBudget(run_root / "budget.json", total=100, baseline_used=0),
    )
    calls = []

    def fake_run_version(**kwargs):
        calls.append(kwargs)
        initial_parent = (
            "-parent-" in kwargs["label"]
            and "infra-trial-retry" not in kwargs["label"]
        )
        per_task = {}
        for task_id in task_ids:
            summaries = [
                {
                    "trial": trial,
                    "pairing_slot": kwargs["pairing_offset"] + trial,
                    "reward": 1,
                    "termination": "completed",
                    "error": "",
                }
                for trial in range(2)
            ]
            if initial_parent and task_id == "0":
                summaries[0]["termination"] = "error"
                summaries[0]["error"] = "worker failed"
            per_task[task_id] = {"trial_summaries": summaries, "rewards": [1, 1]}
        output = {
            "harness_version": kwargs["harness_version"],
            "per_task": per_task,
            "metrics": {"trial_success_rate": 1.0},
        }
        return RolloutResult(
            output, str(run_root / "rollout" / f"{kwargs['label']}.json")
        )

    monkeypatch.setattr(module, "run_version", fake_run_version)

    pair = module.run_pair_from_main(main_decision=decision, label="screen")

    assert len(calls) == 3
    retry_calls = [call for call in calls if "infra-trial-retry" in call["label"]]
    assert len(retry_calls) == 1
    assert retry_calls[0]["task_ids"] == ("0",)
    assert retry_calls[0]["pairing_offset"] == 0
    assert pair.candidate.output["metrics"]["paired_infrastructure_valid"] is True
    assert paired_infrastructure_failures(pair) == ()


def test_v0_screen_retries_only_invalid_trials_with_original_slots(
    tmp_path, monkeypatch
):
    run_root = tmp_path / "run"
    (run_root / "experience").mkdir(parents=True)
    task_ids = [str(index) for index in range(5)]
    (run_root / "experience" / "baseline_source_index.json").write_text(
        json.dumps({"task_ids": task_ids}), encoding="utf-8"
    )
    decision = run_root / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "harness_version": "candidate-01",
                "candidate": {"parent_version": "v0"},
                "rollout_request": {
                    "task_ids": task_ids,
                    "rationale": "single-sided screen retry",
                },
            }
        ),
        encoding="utf-8",
    )
    module = RolloutModule(
        repo_root=tmp_path,
        run_root=run_root,
        budget=CreationBudget(run_root / "budget.json", total=100, baseline_used=0),
    )
    calls = []

    def fake_run_version(**kwargs):
        calls.append(kwargs)
        failed = "infra-trial-retry" not in kwargs["label"]
        summaries = [
            {
                "trial": trial,
                "pairing_slot": kwargs["pairing_offset"] + trial,
                "termination": "timeout" if failed and trial == 0 else "completed",
                "error": "TIMEOUT after 180s" if failed and trial == 0 else "",
            }
            for trial in range(2)
        ]
        return RolloutResult(
            {
                "per_task": {
                    task_id: {"trial_summaries": list(summaries)}
                    for task_id in task_ids
                },
                "metrics": {"pass_at_1": 0.0},
            },
            str(run_root / "rollout" / f"{kwargs['label']}.json"),
        )

    monkeypatch.setattr(module, "run_version", fake_run_version)

    pair = module.run_pair_from_main(main_decision=decision, label="screen")

    assert pair.parent is None
    assert len(calls) == 6
    assert calls[0]["pairing_offset"] == 0
    retry_calls = [call for call in calls if "infra-trial-retry" in call["label"]]
    assert {call["task_ids"] for call in retry_calls} == {(task_id,) for task_id in task_ids}
    assert {call["pairing_offset"] for call in retry_calls} == {0}


def test_v0_screen_never_reviews_an_infrastructure_invalid_retry(
    tmp_path, monkeypatch
):
    run_root = tmp_path / "run"
    (run_root / "experience").mkdir(parents=True)
    task_ids = [str(index) for index in range(5)]
    (run_root / "experience" / "baseline_source_index.json").write_text(
        json.dumps({"task_ids": task_ids}), encoding="utf-8"
    )
    decision = run_root / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "harness_version": "candidate-01",
                "candidate": {"parent_version": "v0"},
                "rollout_request": {"task_ids": task_ids, "rationale": "invalid"},
            }
        ),
        encoding="utf-8",
    )
    module = RolloutModule(
        repo_root=tmp_path,
        run_root=run_root,
        budget=CreationBudget(run_root / "budget.json", total=100, baseline_used=0),
    )
    monkeypatch.setattr(
        module,
        "run_version",
        lambda **kwargs: RolloutResult(
            {
                "per_task": {
                    task_id: {
                        "trial_summaries": [
                            {
                                "pairing_slot": kwargs["pairing_offset"],
                                "termination": "timeout",
                                "error": "TIMEOUT after 180s",
                            }
                        ]
                    }
                    for task_id in task_ids
                }
            },
            str(run_root / "rollout" / f"{kwargs['label']}.json"),
        ),
    )

    with pytest.raises(RuntimeError, match="infrastructure-invalid after retry"):
        module.run_pair_from_main(main_decision=decision, label="screen")


def test_paired_rollout_marks_a_missing_side_payload_invalid(tmp_path):
    candidate = RolloutResult(
        {
            "per_task": {
                "0": {
                    "trial_summaries": [
                        {"pairing_slot": 0, "termination": "completed", "error": ""}
                    ]
                }
            },
            "metrics": {"pass_at_1": 1.0, "trial_success_rate": 1.0},
        },
        str(tmp_path / "candidate.json"),
    )
    parent = RolloutResult(
        {"metrics": {"pass_at_1": 1.0, "trial_success_rate": 1.0}},
        str(tmp_path / "parent.json"),
    )
    pair = PairedRolloutResult(candidate=candidate, parent=parent)

    failures = paired_infrastructure_failures(pair)

    assert failures == (
        {
            "task_id": "*",
            "pairing_slot": -1,
            "side": "parent",
            "error": "missing per_task mapping",
        },
    )
    annotated = annotate_paired_validity(pair, failures=failures)
    assert annotated.candidate.output["metrics"]["paired_infrastructure_valid"] is False
    assert annotated.parent.output["metrics"]["paired_infrastructure_valid"] is False
    assert "estimated_pass_at_1" not in annotated.candidate.output["metrics"]


def test_residual_pair_propagates_the_smaller_controller_minimum(
    tmp_path, monkeypatch
):
    run_root = tmp_path / "run"
    (run_root / "experience").mkdir(parents=True)
    (run_root / "experience" / "baseline_source_index.json").write_text(
        json.dumps({"task_ids": ["0", "1", "2", "3", "4"]}), encoding="utf-8"
    )
    decision = run_root / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "harness_version": "candidate-01",
                "evaluation_mode": "residual_probe",
                "candidate": {"parent_version": "incumbent-00"},
                "rollout_request": {
                    "task_ids": ["0", "1", "2"],
                    "rationale": "diagnostic probe",
                },
            }
        ),
        encoding="utf-8",
    )
    module = RolloutModule(
        repo_root=tmp_path,
        run_root=run_root,
        budget=CreationBudget(run_root / "budget.json", total=100, baseline_used=0),
    )
    calls = []
    monkeypatch.setattr(
        module,
        "run_version",
        lambda **kwargs: calls.append(kwargs)
        or RolloutResult({"harness_version": kwargs["harness_version"]}, str(run_root / "rollout" / f"{kwargs['label']}.json")),
    )

    module.run_pair_from_main(main_decision=decision, label="probe")

    assert {call["minimum_task_count"] for call in calls} == {2}


def test_paired_rollout_resumes_launched_jobs_without_a_second_budget_reservation(
    tmp_path, monkeypatch
):
    run_root = tmp_path / "run"
    (run_root / "experience").mkdir(parents=True)
    (run_root / "experience" / "baseline_source_index.json").write_text(
        json.dumps({"task_ids": ["0", "1"]}), encoding="utf-8"
    )
    decision = run_root / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "harness_version": "candidate-02",
                "evaluation_mode": "residual_probe",
                "candidate": {"parent_version": "candidate-01"},
                "rollout_request": {
                    "task_ids": ["0", "1"],
                    "rationale": "resume paired probe",
                },
            }
        ),
        encoding="utf-8",
    )
    budget = CreationBudget(run_root / "budget.json", total=8, baseline_used=0)
    for job_id in ("rollout-probe", "rollout-probe-parent-candidate-01"):
        budget.reserve_job(job_id, creation_count=4)
        budget.claim_launch(job_id)
        budget.mark_launched(job_id)
    module = RolloutModule(repo_root=tmp_path, run_root=run_root, budget=budget)
    calls = []
    monkeypatch.setattr(
        module,
        "run_version",
        lambda **kwargs: calls.append(kwargs)
        or RolloutResult({"harness_version": kwargs["harness_version"]}, str(run_root / "rollout" / f"{kwargs['label']}.json")),
    )

    module.run_pair_from_main(main_decision=decision, label="probe")

    assert budget.status()["remaining"] == 0
    assert {call["label"] for call in calls} == {
        "probe",
        "probe-parent-candidate-01",
    }


def test_rollout_version_reuses_and_settles_its_launched_job(tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    (run_root / "experience").mkdir(parents=True)
    (run_root / "experience" / "baseline_source_index.json").write_text(
        json.dumps({"task_ids": ["0", "1", "2", "3", "4"]}),
        encoding="utf-8",
    )
    budget = CreationBudget(run_root / "budget.json", total=10, baseline_used=0)
    budget.reserve_job("rollout-screen", creation_count=10)
    budget.claim_launch("rollout-screen")
    budget.mark_launched("rollout-screen")
    calls = []

    class FakeService:
        def __init__(self, **_kwargs):
            pass

        def recover_retained(self, _request):
            return None

        def run(self, request):
            calls.append(request)
            return SimpleNamespace(
                metrics={"pass_at_2": 1.0},
                to_dict=lambda: {
                    "harness_version": request.harness_version,
                    "metrics": {"pass_at_2": 1.0},
                },
            )

    monkeypatch.setattr(
        "harnesslens.evolution.rollout.TrainRolloutService", FakeService
    )
    module = RolloutModule(repo_root=tmp_path, run_root=run_root, budget=budget)

    result = module.run_version(
        task_ids=("0", "1", "2", "3", "4"),
        harness_version="candidate-01",
        label="screen",
        purpose="resume",
    )

    state = json.loads((run_root / "budget.json").read_text(encoding="utf-8"))
    assert result.output["harness_version"] == "candidate-01"
    assert len(calls) == 1
    assert set(state["jobs"]) == {"rollout-screen"}
    assert state["jobs"]["rollout-screen"]["status"] == "settled"
    assert state["jobs"]["rollout-screen"]["usage"]["details"][
        "resumed_launched_job"
    ] is True


def test_rollout_runtime_limits_match_the_reused_runtime():
    assert ROLLOUT_RUNTIME_LIMITS == {
        "opencode_steps": 10,
        "max_turns": 40,
        "max_tool_calls": 60,
        "timeout_per_turn_s": 180,
        "trial_timeout_s": 7200,
    }


def test_tau2_timeout_per_turn_can_be_configured(monkeypatch):
    from harnesslens.evaluation.rollout_bridge import tau2_timeout_per_turn_s

    monkeypatch.setenv("HAI_TAU2_TIMEOUT_PER_TURN_S", "360")

    assert tau2_timeout_per_turn_s() == 360


@pytest.mark.parametrize("value", ["0", "invalid"])
def test_tau2_timeout_per_turn_rejects_invalid_values(monkeypatch, value):
    from harnesslens.evaluation.rollout_bridge import tau2_timeout_per_turn_s

    monkeypatch.setenv("HAI_TAU2_TIMEOUT_PER_TURN_S", value)

    with pytest.raises(ValueError, match="must be a positive integer"):
        tau2_timeout_per_turn_s()


def test_rollout_group_workspace_cleanup_removes_empty_scaffolding(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "group" / ".sandbox_home" / ".cache"
    workspace.mkdir(parents=True)
    monkeypatch.delenv("HAI_KEEP_TRAJECTORY_WORKSPACE", raising=False)

    _cleanup_rollout_group_workspace(tmp_path / "group")

    assert not (tmp_path / "group").exists()


@pytest.mark.parametrize("harness", ["opencode", "pi", "codex"])
def test_native_candidate_repository_materializes_cumulative_harness_snapshot(
    tmp_path, harness
):
    repository = CellHarnessRepository(
        cell="retail",
        repo_root=tmp_path,
        run_id="run",
        evidence_root=tmp_path / "evidence",
        harness=harness,
    )

    baseline = repository.read_candidate_snapshot("v0")
    assert {key: baseline[key] for key in (
        "config_patch",
        "files",
        "instructions",
        "prompt_appends",
        "tool_desc_patches",
    )} == {
        "config_patch": {},
        "files": [],
        "instructions": [],
        "prompt_appends": [],
        "tool_desc_patches": {},
    }
    first = repository.materialize_candidate(
        base_version="v0",
        candidate_label="candidate-01",
        delta={
            "files": [{"path": "AGENTS.md", "content": "First rule.\n"}],
            "tool_desc_patches": {"lookup": {"desc": "Use exact lookup."}},
        },
    )
    second = repository.materialize_candidate(
        base_version=first,
        candidate_label="candidate-02",
        delta={
            "prompt_appends": ["Second rule."],
            "config_patch": {"features.shell_tool": False},
        },
    )

    snapshot = repository.read_candidate_snapshot(second)
    assert snapshot["files"] == [
        {"path": "AGENTS.md", "content": "First rule.\n"}
    ]
    assert snapshot["prompt_appends"] == ["Second rule."]
    assert snapshot["config_patch"] == {"features.shell_tool": False}
    assert snapshot["tool_desc_patches"] == {
        "lookup": {"desc": "Use exact lookup."}
    }
    version_root = (
        tmp_path
        / "evidence"
        / "run"
        / "versions_percell"
        / "retail"
        / second
        / "harness"
        / harness
    )
    assert (version_root / "manifest.json").is_file()
    assert (version_root / "workspace.json").is_file()
    assert json.loads((version_root.parent.parent / "lineage.json").read_text())["harness"] == harness


@pytest.mark.parametrize("harness", ["opencode", "pi", "codex"])
def test_candidate_repository_replaces_editor_workspace_and_inherits_manifest(
    tmp_path, harness
):
    repository = CellHarnessRepository(
        cell="retail",
        repo_root=tmp_path,
        run_id="run",
        evidence_root=tmp_path / "evidence",
        harness=harness,
    )
    parent = repository.materialize_candidate(
        base_version="v0",
        candidate_label="candidate-01",
        delta={"prompt_appends": ["legacy behavior"]},
    )

    child = repository.materialize_workspace_candidate(
        base_version=parent,
        candidate_label="candidate-02",
        workspace={
            "files": [
                {
                    "scope": "project",
                    "path": "AGENTS.md",
                    "content": "Editor behavior.\n",
                }
            ]
        },
    )

    assert repository.read_candidate_snapshot(child)["prompt_appends"] == [
        "legacy behavior"
    ]
    assert repository.read_workspace_snapshot(child)["files"] == [
        {
            "scope": "project",
            "path": "AGENTS.md",
            "content": "Editor behavior.\n",
            "executable": False,
        }
    ]


@pytest.mark.parametrize("harness", ["opencode", "pi", "codex"])
def test_workspace_candidate_merges_controller_mcp_manifest_delta(tmp_path, harness):
    repository = CellHarnessRepository(
        cell="retail",
        repo_root=tmp_path,
        run_id="run",
        evidence_root=tmp_path / "evidence",
        harness=harness,
    )

    child = repository.materialize_workspace_candidate(
        base_version="v0",
        candidate_label="candidate-mcp",
        workspace={"files": []},
        manifest_delta={
            "tool_desc_patches": {
                "lookup_record": {"desc": "Use exact identifiers."}
            }
        },
    )

    assert repository.read_candidate_snapshot(child)["tool_desc_patches"] == {
        "lookup_record": {"desc": "Use exact identifiers."}
    }



def test_native_candidate_repositories_do_not_share_versions(tmp_path):
    snapshots = {}
    for harness in ("pi", "codex"):
        repository = CellHarnessRepository(
            cell="retail",
            repo_root=tmp_path,
            run_id="run",
            evidence_root=tmp_path / "evidence",
            harness=harness,
        )
        repository.materialize_candidate(
            base_version="v0",
            candidate_label="candidate-01",
            delta={"prompt_appends": [harness]},
        )
        snapshots[harness] = repository.read_candidate_snapshot("candidate-01")

    assert snapshots["pi"]["prompt_appends"] == ["pi"]
    assert snapshots["codex"]["prompt_appends"] == ["codex"]


@pytest.mark.parametrize("harness", ["opencode", "pi", "codex"])
def test_native_rollout_dispatches_to_same_harness_with_version_manifest(
    tmp_path, monkeypatch, harness
):
    repository = CellHarnessRepository(
        cell="retail",
        repo_root=tmp_path,
        run_id="run",
        evidence_root=tmp_path / "evidence",
        harness=harness,
    )
    repository.materialize_candidate(
        base_version="v0",
        candidate_label="candidate-01",
        delta={"tool_desc_patches": {"lookup": {"desc": f"{harness} patch"}}},
    )
    captured = {}

    def fake_native_rollout_worker(**kwargs):
        captured.update(kwargs)
        return {"per_task": {}, "records": [], "metrics": {}}

    monkeypatch.setattr(
        rollout_bridge, "_run_native_tau2_worker", fake_native_rollout_worker
    )
    service = TrainRolloutService(
        cell="retail",
        repo_root=tmp_path,
        run_id="run",
        artifact_root=tmp_path / "artifacts",
        train_task_ids=["0"],
        initial_budget=1,
        evidence_root=tmp_path / "evidence",
        workspace_root=tmp_path / "workspaces",
        harness=harness,
    )
    request = RolloutRequest(
        request_id="request",
        run_id="run",
        scope="TRAIN",
        harness_version="candidate-01",
        task_repeats={"0": 1},
        max_concurrency=1,
        purpose="same harness dispatch",
        pairing_offsets={"0": 0},
    )

    service._run_group(request, ["0"], 1)

    assert captured["payload"]["harness"] == harness
    assert captured["payload"]["request"]["scope"] == "TRAIN"
    assert captured["payload"]["harness_manifest"]["tool_desc_patches"] == {
        "lookup": {"desc": f"{harness} patch"}
    }


@pytest.mark.parametrize("harness", ["opencode", "pi", "codex"])
def test_bird_rollout_dispatches_every_target_harness(monkeypatch, tmp_path, harness):
    captured = {}
    monkeypatch.setattr(
        rollout_bridge,
        "run_bird_batch",
        lambda **kwargs: captured.update(kwargs)
        or {"trajectory_root": "unused", "per_task": {}, "records": [], "metrics": {}},
    )
    service = TrainRolloutService(
        cell="bird_mini_dev_challenging",
        repo_root=tmp_path,
        run_id="run",
        artifact_root=tmp_path / "artifacts",
        train_task_ids=["0"],
        initial_budget=1,
        evidence_root=tmp_path / "evidence",
        workspace_root=tmp_path / "workspaces",
        harness=harness,
    )
    request = RolloutRequest(
        request_id="request",
        run_id="run",
        scope="TRAIN",
        harness_version="v0",
        task_repeats={"0": 1},
        max_concurrency=1,
        purpose="bird harness dispatch",
        pairing_offsets={"0": 0},
    )

    service._run_group(request, ["0"], 1)

    assert captured["harness"] == harness
    expected = {
        "config_patch": {},
        "files": [],
        "instructions": [],
        "prompt_appends": [],
        "tool_desc_patches": {},
    }
    assert {
        key: captured["harness_manifest"][key] for key in expected
    } == expected
    assert captured["harness_manifest"]["_workspace"] == {
        "schema": 1,
        "files": [],
    }


@pytest.mark.parametrize("harness", ["opencode", "pi", "codex"])
def test_terminal_rollout_dispatches_every_target_harness(
    monkeypatch, tmp_path, harness
):
    captured = {}
    monkeypatch.setattr(
        rollout_bridge,
        "run_terminal_batch",
        lambda **kwargs: captured.update(kwargs)
        or {"trajectory_root": "unused", "per_task": {}, "records": [], "metrics": {}},
    )
    service = TrainRolloutService(
        cell="terminal_bench",
        repo_root=tmp_path,
        run_id="run",
        artifact_root=tmp_path / "artifacts",
        train_task_ids=["task"],
        initial_budget=1,
        evidence_root=tmp_path / "evidence",
        workspace_root=tmp_path / "workspaces",
        harness=harness,
    )
    request = RolloutRequest(
        request_id="request",
        run_id="run",
        scope="TRAIN",
        harness_version="v0",
        task_repeats={"task": 1},
        max_concurrency=1,
        purpose="terminal harness dispatch",
        pairing_offsets={"task": 0},
    )

    service._run_group(request, ["task"], 1)

    assert captured["harness"] == harness
    expected = {
        "config_patch": {},
        "files": [],
        "instructions": [],
        "prompt_appends": [],
        "tool_desc_patches": {},
    }
    assert {
        key: captured["harness_manifest"][key] for key in expected
    } == expected
    assert captured["harness_manifest"]["_workspace"] == {
        "schema": 1,
        "files": [],
    }


@pytest.mark.parametrize("harness", ["pi", "codex"])
def test_native_rollout_rejects_unknown_benchmark_without_fallback(tmp_path, harness):
    service = TrainRolloutService(
        cell="unknown_cell",
        repo_root=tmp_path,
        run_id="run",
        artifact_root=tmp_path / "artifacts",
        train_task_ids=["0"],
        initial_budget=1,
        evidence_root=tmp_path / "evidence",
        workspace_root=tmp_path / "workspaces",
        harness=harness,
    )
    request = RolloutRequest(
        request_id="request",
        run_id="run",
        scope="TRAIN",
        harness_version="v0",
        task_repeats={"0": 1},
        max_concurrency=1,
        purpose="unsupported benchmark",
        pairing_offsets={"0": 0},
    )

    with pytest.raises(ValueError, match="not implemented"):
        service._run_group(request, ["0"], 1)


def test_baseline_fingerprint_records_the_actual_opencode_step_limit():
    repo_root = Path(__file__).resolve().parents[1]
    fingerprint = build_baseline_fingerprint(repo_root)

    assert fingerprint["rollout"]["max_steps"] == ROLLOUT_RUNTIME_LIMITS[
        "opencode_steps"
    ]


def test_baseline_fingerprint_separates_target_harnesses():
    repo_root = Path(__file__).resolve().parents[1]

    fingerprints = {
        harness: build_baseline_fingerprint(repo_root, harness=harness)
        for harness in ("opencode", "pi", "codex")
    }

    assert {item["harness"] for item in fingerprints.values()} == {
        "opencode",
        "pi",
        "codex",
    }
    assert len({item["fingerprint_sha256"] for item in fingerprints.values()}) == 3


def test_rollout_service_reconstructs_completed_retained_batch(tmp_path):
    request = RolloutRequest(
        request_id="candidate-request",
        run_id="run",
        scope="TRAIN",
        harness_version="candidate-01",
        task_repeats={"0": 2},
        max_concurrency=2,
        purpose="test",
        pairing_offsets={"0": 0},
    )
    root = tmp_path / "artifacts" / "run" / "candidate-request" / "trajectories" / "0"
    root.mkdir(parents=True)
    for index, reward in enumerate((1.0, 0.0), start=1):
        sidecar = root / f"trial_{index:04d}.api_calls.jsonl"
        sidecar.write_text(
            json.dumps(
                {
                    "request": {"messages": [{"role": "system", "content": "policy"}]},
                    "response": {"choices": [{"message": {"content": "done"}}]},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (root / f"trial_{index:04d}.jsonl").write_text(
            json.dumps(
                {
                    "task_id": "0",
                    "trial": index - 1,
                    "pairing_slot": index - 1,
                    "reward": reward,
                    "api_calls_jsonl": sidecar.name,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    service = TrainRolloutService(
        cell="retail",
        repo_root=tmp_path,
        run_id="run",
        artifact_root=tmp_path / "artifacts",
        train_task_ids=["0"],
        initial_budget=2,
        evidence_root=tmp_path / "evidence",
        workspace_root=tmp_path / "workspaces",
    )

    response = service.recover_retained(request)

    assert response is not None
    assert response.records[0].rewards == (1.0, 0.0)
    assert response.metrics["trial_count"] == 2
    assert response.metrics["pass_at_2"] == 1.0


def test_recovered_rollout_settles_stale_active_job(tmp_path):
    budget = CreationBudget(tmp_path / "budget.json", total=20, baseline_used=0)
    budget.reserve_job("rollout-iteration-01-candidate-01", creation_count=2)
    budget.claim_launch("rollout-iteration-01-candidate-01")
    budget.mark_launched("rollout-iteration-01-candidate-01")

    _settle_recovered_rollout_job(
        budget,
        base_job_id="rollout-iteration-01-candidate-01",
        output_path=tmp_path / "rollout.json",
        metrics={"pass_at_2": 1.0},
    )

    state = json.loads((tmp_path / "budget.json").read_text(encoding="utf-8"))
    record = state["jobs"]["rollout-iteration-01-candidate-01"]
    assert record["status"] == "settled"
    assert record["outcome"] == "completed"
    assert record["usage"]["details"]["recovered_from_retained_trials"] is True


def test_rollout_service_identifies_the_exact_corrupt_retained_trial(tmp_path):
    request = RolloutRequest(
        request_id="candidate-request",
        run_id="run",
        scope="TRAIN",
        harness_version="candidate-01",
        task_repeats={"0": 1},
        max_concurrency=1,
        purpose="test",
        pairing_offsets={"0": 0},
    )
    root = tmp_path / "artifacts" / "run" / "candidate-request" / "trajectories" / "0"
    root.mkdir(parents=True)
    sidecar = root / "trial_0001.api_calls.jsonl"
    sidecar.write_text("{broken\n", encoding="utf-8")
    trajectory = root / "trial_0001.jsonl"
    trajectory.write_text(
        json.dumps(
            {
                "task_id": "0",
                "trial": 0,
                "reward": 0.0,
                "api_calls_jsonl": sidecar.name,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    service = TrainRolloutService(
        cell="retail",
        repo_root=tmp_path,
        run_id="run",
        artifact_root=tmp_path / "artifacts",
        train_task_ids=["0"],
        initial_budget=1,
        evidence_root=tmp_path / "evidence",
        workspace_root=tmp_path / "workspaces",
    )

    with pytest.raises(IncompleteRolloutTraceError) as error:
        service.recover_retained(request)

    assert error.value.trajectory_path == trajectory.resolve()
