import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import harnesslens.evolution.experience as experience_module

from harnesslens.evolution.experience import (
    canonicalize_comparison,
    canonicalize_experience_merge_plan,
    comparison_reference_paths,
    ExperienceModule,
    _ensure_authoritative_failure_dispositions,
    _draft_catalog,
    _settle_recovered_intelligent_job,
    _visible_trial_packet,
    materialize_experience_merge_plan,
    normalize_experience,
    validate_experience,
    validate_experience_merge_plan,
    validate_comparison,
)
from harnesslens.core.budget import CreationBudget


class _RetryBudget:
    def __init__(self):
        self.calls = 0

    def next_attempt_id(self, base):
        self.calls += 1
        return f"{base}-attempt-{self.calls}"


class _AlwaysInvalidRunner:
    payloads = []

    def __init__(self, **kwargs):
        del kwargs

    def run_json(self, **kwargs):
        self.payloads.append(kwargs["input_payload"])
        attempt = len(self.payloads)
        return SimpleNamespace(
            output=None,
            outcome="malformed_output",
            validation_error=f"exact-error-{attempt}",
        )


class _SplitComparisonRunner:
    calls = []

    def __init__(self, **kwargs):
        del kwargs

    def run_json(self, **kwargs):
        task_ids = tuple(kwargs["input_payload"]["task_ids"])
        self.calls.append(task_ids)
        if len(task_ids) > 1:
            return SimpleNamespace(
                output=None,
                outcome="malformed_output",
                validation_error="output limit reached",
            )
        task_id = task_ids[0]
        output = _comparison_output()
        output["task_comparisons"][0]["task_id"] = task_id
        output["coverage"]["task_ids"] = [task_id]
        for field in ("baseline_refs", "candidate_refs"):
            output["coverage"][field] = []
            output["task_comparisons"][0][field] = []
        return SimpleNamespace(output=output, outcome="completed", validation_error="")


def _experience_module_for_retry(tmp_path):
    module = object.__new__(ExperienceModule)
    module.run_root = tmp_path
    module.root = tmp_path / "experience"
    module.root.mkdir(parents=True)
    module.harness = "opencode"
    module.budget = _RetryBudget()
    return module


def test_comparison_requires_exact_direct_parent_trials():
    parent = {
        "7": {
            "trajectory_paths": ["parent-1.jsonl", "parent-2.jsonl"]
        }
    }
    assert comparison_reference_paths(
        task_id="7",
        baseline_paths=("v0-1.jsonl", "v0-2.jsonl"),
        reference_records=parent,
        reference_version="candidate-01",
    ) == (("parent-1.jsonl", "parent-2.jsonl"), "candidate-01")
    with pytest.raises(ValueError, match="lacks task '8'"):
        comparison_reference_paths(
            task_id="8",
            baseline_paths=("v0-1.jsonl", "v0-2.jsonl"),
            reference_records=parent,
            reference_version="candidate-01",
        )

    assert comparison_reference_paths(
        task_id="8",
        baseline_paths=("v0-1.jsonl", "v0-2.jsonl"),
        reference_records={},
        reference_version="",
    ) == (("v0-1.jsonl", "v0-2.jsonl"), "v0")


def _valid_output():
    return {
        "reusable": [
            {
                "id": "route-a",
                "text": "When the state is visible, inspect it, call the exact lookup tool, preserve the returned identifier, and verify the resulting branch before mutation.",
                "evidence_refs": ["ev_pass"],
            }
        ],
        "needs_adjustment": [
            {
                "id": "failure-a",
                "text": "When confirmation is still absent, the failed trajectory performs the mutation anyway; the successful contrast does not, so limit the candidate to that observable condition.",
                "evidence_refs": ["ev_fail", "ev_pass"],
            }
        ],
        "coverage": {"task_ids": ["1"], "evidence_refs": ["ev_pass", "ev_fail"]},
    }


def test_experience_requires_structural_evidence_coverage():
    validate_experience(
        _valid_output(),
        task_ids=("1",),
        evidence_refs=("ev_pass", "ev_fail"),
        outcomes={"ev_pass": "pass", "ev_fail": "fail"},
    )


def test_experience_allows_reward_zero_trajectory_to_be_semantically_reusable():
    output = _valid_output()
    output["reusable"][0]["evidence_refs"] = ["ev_pass", "ev_fail"]
    output["needs_adjustment"] = []
    validate_experience(
        output,
        task_ids=("1",),
        evidence_refs=("ev_pass", "ev_fail"),
        outcomes={"ev_pass": "pass", "ev_fail": "fail"},
    )


def test_authoritative_outcomes_reject_failure_only_reusable_experience():
    output = _valid_output()
    output["reusable"][0]["evidence_refs"] = ["ev_fail"]
    output["needs_adjustment"] = []

    with pytest.raises(ValueError, match="authoritative passing evidence"):
        validate_experience(
            output,
            task_ids=("1",),
            evidence_refs=("ev_pass", "ev_fail"),
            outcomes={"ev_pass": "pass", "ev_fail": "fail"},
            outcome_authority="authoritative",
        )


def test_authoritative_failure_requires_adjustment_disposition():
    output = _valid_output()
    output["needs_adjustment"] = []

    with pytest.raises(ValueError, match="failed evidence requires needs_adjustment"):
        validate_experience(
            output,
            task_ids=("1",),
            evidence_refs=("ev_pass", "ev_fail"),
            outcomes={"ev_pass": "pass", "ev_fail": "fail"},
            outcome_authority="authoritative",
        )


def test_authoritative_failure_gets_unresolved_disposition():
    output = _valid_output()
    output["needs_adjustment"] = []

    normalized = _ensure_authoritative_failure_dispositions(
        output,
        outcomes={"ev_pass": "pass", "ev_fail": "fail"},
        outcome_authority="authoritative",
    )

    assert normalized["needs_adjustment"][0]["evidence_refs"] == ["ev_fail"]
    assert "unresolved diagnostic evidence" in normalized["needs_adjustment"][0]["text"]
    validate_experience(
        normalized,
        task_ids=("1",),
        evidence_refs=("ev_pass", "ev_fail"),
        outcomes={"ev_pass": "pass", "ev_fail": "fail"},
        outcome_authority="authoritative",
    )


def test_experience_does_not_keyword_gate_failure_only_language():
    output = _valid_output()
    output["needs_adjustment"][0]["evidence_refs"] = ["ev_fail"]
    validate_experience(
        output,
        task_ids=("1",),
        evidence_refs=("ev_pass", "ev_fail"),
        outcomes={"ev_pass": "pass", "ev_fail": "fail"},
    )


def test_visible_trial_packet_keeps_exact_calls_but_bounds_large_results(tmp_path):
    path = tmp_path / "trial.jsonl"
    path.write_text(
        json.dumps(
            {
                "trial": 0,
                "reward": 1,
                "model_context": {
                    "skills_available": [{"name": "route", "description": "bounded"}],
                    "skills_invoked": [{"name": "route", "n_calls": 1}],
                },
                "messages": [
                    {"role": "user", "content": "Please exchange the shirt."},
                    {
                        "role": "assistant",
                        "content": "I will inspect the item first.",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "name": "get_order_details",
                                "arguments": {"order_id": "#123"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "content": json.dumps({"items": ["x" * 1000] * 20}),
                    },
                    {"role": "assistant", "content": "The order is eligible."},
                ],
            }
        )
        + "\n"
    )

    packet = _visible_trial_packet(path, evidence_id="ev_a")

    assert packet["tool_interactions"][0]["arguments"] == {"order_id": "#123"}
    assert '"items"' in packet["tool_interactions"][0]["observed_result"]
    assert len(packet["tool_interactions"][0]["observed_result"]) <= 1800
    assert packet["channel_usage"] == {
        "skills_available": [{"name": "route", "n_calls": 0}],
        "skills_invoked": [{"name": "route", "n_calls": 1}],
    }
    assert "events" not in packet


def test_visible_trial_packet_keeps_terminal_execution_authority(tmp_path):
    path = tmp_path / "terminal.jsonl"
    path.write_text(
        json.dumps(
            {
                "trial": 0,
                "reward": 0.0,
                "status": "completed",
                "termination": "done",
                "n_messages": 3,
                "n_tool_calls": 47,
                "infrastructure_error": False,
                "verifier_completed": True,
                "verifier_timed_out": False,
                "messages": [
                    {"role": "user", "content": "finish all steps"},
                    {"role": "assistant", "content": "bounded event tail"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    packet = _visible_trial_packet(path, evidence_id="ev_terminal")

    assert packet["execution_summary"] == {
        "status": "completed",
        "n_messages": 3,
        "n_tool_calls": 47,
        "worker_error": "",
        "infrastructure_error": False,
        "verifier_completed": True,
        "verifier_timed_out": False,
    }
def test_visible_trial_packet_normalizes_bird_tool_messages(tmp_path):
    path = tmp_path / "bird-trial.jsonl"
    path.write_text(
        json.dumps(
            {
                "trial": 0,
                "reward": 0,
                "messages": [
                    {"role": "user", "content": "Generate the SQLite query."},
                    {
                        "role": "assistant",
                        "tool_name": "execute_sql",
                        "tool_arguments": {"sql": "SELECT COUNT(*) FROM users"},
                    },
                    {
                        "role": "tool",
                        "name": "execute_sql",
                        "content": '{"columns":["COUNT(*)"],"rows":[[3]]}',
                        "is_error": False,
                    },
                    {
                        "role": "assistant",
                        "content": "```sql\nSELECT COUNT(*) FROM users\n```",
                    },
                ],
            }
        )
        + "\n"
    )

    packet = _visible_trial_packet(path, evidence_id="ev_bird")

    assert len(packet["tool_interactions"]) == 1
    interaction = packet["tool_interactions"][0]
    assert interaction["sequence"] == 1
    assert interaction["call_id"] == ""
    assert interaction["tool"] == "execute_sql"
    assert interaction["arguments"] == {"sql": "SELECT COUNT(*) FROM users"}
    assert json.loads(interaction["observed_result"]) == {
        "columns": ["COUNT(*)"],
        "rows": [[3]],
    }


def test_visible_trial_packet_exposes_only_sanitized_grader_diagnostic(tmp_path):
    path = tmp_path / "bird.jsonl"
    path.write_text(
        json.dumps(
            {
                "task_id": "bird_1",
                "reward": 0.0,
                "termination": "completed",
                "messages": [{"role": "assistant", "content": "SELECT 1"}],
                "grader_diagnostic": {
                    "mismatch_type": "row_count_mismatch",
                    "predicted_row_count": 1,
                    "reference_row_count": 2,
                },
                "gold_sql": "SELECT hidden FROM secret",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    packet = _visible_trial_packet(path, evidence_id="ev_bird")

    assert packet["grader_diagnostic"] == {
        "mismatch_type": "row_count_mismatch",
        "predicted_row_count": 1,
        "reference_row_count": 2,
    }
    assert "gold_sql" not in packet


def test_small_selection_stays_in_one_experience_job(tmp_path):
    module = object.__new__(ExperienceModule)
    module.run_root = tmp_path

    assert module._task_groups(("0", "1", "2", "3", "4")) == [
        {"id": "selected-01", "task_ids": ["0", "1", "2", "3", "4"]}
    ]


def test_normalize_experience_drops_free_form_notes():
    output = _valid_output()
    output["synthesis_notes"] = ["unsupported aggregate"]

    assert "synthesis_notes" not in normalize_experience(output)


def test_normalize_experience_adds_evidence_explicitly_named_in_text():
    output = _valid_output()
    output["needs_adjustment"][0]["text"] += " Contrast evidence is ev_pass."
    output["needs_adjustment"][0]["evidence_refs"] = ["ev_fail"]

    normalized = normalize_experience(output, ("ev_pass", "ev_fail"))

    assert normalized["needs_adjustment"][0]["evidence_refs"] == [
        "ev_fail",
        "ev_pass",
    ]


def test_failure_only_experience_can_describe_observed_absence():
    output = _valid_output()
    output["needs_adjustment"][0]["text"] = (
        "This unresolved failed trace transferred after a payment shortfall; neither trial "
        "offered a partial payment or alternative variant, and both ended after transfer."
    )
    output["needs_adjustment"][0]["evidence_refs"] = ["ev_fail"]
    validate_experience(
        output,
        task_ids=("1",),
        evidence_refs=("ev_pass", "ev_fail"),
        outcomes={"ev_pass": "pass", "ev_fail": "fail"},
    )


def test_experience_semantics_are_not_decided_by_keyword_matching():
    output = _valid_output()
    output["needs_adjustment"][0]["text"] = (
        "This failed trace probably disagrees with the grader despite following the visible "
        "steps, so modify the route to satisfy that hidden signal in later executions."
    )
    validate_experience(
        output,
        task_ids=("1",),
        evidence_refs=("ev_pass", "ev_fail"),
        outcomes={"ev_pass": "pass", "ev_fail": "fail"},
    )


def test_merge_plan_preserves_unmentioned_passages_and_unions_evidence():
    drafts = [
        {
            "reusable": [
                {"id": "route", "text": "A" * 100, "evidence_refs": ["ev_p1"]}
            ],
            "needs_adjustment": [
                {
                    "id": "failure",
                    "text": "Unresolved: " + "B" * 100,
                    "evidence_refs": ["ev_f1"],
                }
            ],
        },
        {
            "reusable": [
                {"id": "route", "text": "C" * 100, "evidence_refs": ["ev_p2"]}
            ],
            "needs_adjustment": [
                {
                    "id": "other-failure",
                    "text": "Unresolved: " + "D" * 100,
                    "evidence_refs": ["ev_f2"],
                }
            ],
        },
    ]
    catalog = _draft_catalog(drafts)
    plan = {
        "reusable_merges": [
            {
                "source_keys": ["d01-r01", "d02-r01"],
                "id": "merged-route",
                "text": "Equivalent observed procedure with detailed parameters. " + "E" * 80,
            }
        ],
        "needs_adjustment_merges": [],
    }

    validate_experience_merge_plan(plan, catalog)
    output = materialize_experience_merge_plan(
        plan,
        catalog=catalog,
        task_ids=("1", "2"),
        evidence_refs=("ev_p1", "ev_f1", "ev_p2", "ev_f2"),
        outcomes={"ev_p1": "pass", "ev_f1": "fail", "ev_p2": "pass", "ev_f2": "fail"},
    )

    assert output["reusable"][0]["evidence_refs"] == ["ev_p1", "ev_p2"]
    assert len(output["needs_adjustment"]) == 2
    assert len({item["id"] for item in output["needs_adjustment"]}) == 2


def test_merge_plan_can_discard_a_non_experience_passage():
    drafts = [
        {
            "reusable": [
                {"id": "route", "text": "A" * 100, "evidence_refs": ["ev_p"]}
            ],
            "needs_adjustment": [
                {
                    "id": "no-visible-problem",
                    "text": "B" * 100,
                    "evidence_refs": ["ev_f"],
                }
            ],
        }
    ]
    catalog = _draft_catalog(drafts)
    plan = {
        "reusable_merges": [],
        "needs_adjustment_merges": [],
        "discard_source_keys": ["d01-a01"],
    }

    validate_experience_merge_plan(plan, catalog)
    output = materialize_experience_merge_plan(
        plan,
        catalog=catalog,
        task_ids=("1",),
        evidence_refs=("ev_p", "ev_f"),
        outcomes={"ev_p": "pass", "ev_f": "fail"},
    )

    assert [item["id"] for item in output["reusable"]] == ["route"]
    assert output["needs_adjustment"] == []


def test_merge_plan_rejects_unknown_source():
    catalog = _draft_catalog(
        [{"reusable": [{"id": "a", "text": "A" * 100, "evidence_refs": ["ev"]}]}]
    )
    plan = {
        "reusable_merges": [
            {"source_keys": ["d01-r01", "missing"], "id": "x", "text": "X" * 100}
        ],
        "needs_adjustment_merges": [],
    }

    with pytest.raises(ValueError, match="unknown source"):
        validate_experience_merge_plan(plan, catalog)


def test_merge_plan_keeps_sources_unchanged_when_multiple_merges_claim_them():
    plan = {
        "reusable_merges": [
            {
                "source_keys": ["route-a", "compound", "route-b"],
                "id": "route",
                "text": "A detailed route that remains grounded in the non-ambiguous sources. "
                + "A" * 80,
            },
            {
                "source_keys": ["address-a", "compound", "address-b"],
                "id": "address",
                "text": "A detailed address route grounded in its non-ambiguous sources. "
                + "B" * 80,
            },
        ],
        "needs_adjustment_merges": [],
    }

    canonicalize_experience_merge_plan(plan)

    assert plan["reusable_merges"][0]["source_keys"] == ["route-a", "route-b"]
    assert plan["reusable_merges"][1]["source_keys"] == ["address-a", "address-b"]


def _comparison_output():
    return {
        "task_comparisons": [
            {
                "task_id": "1",
                "status": "recovered",
                "text": "The candidate set executes after confirmation in both trials, while only one baseline trial executes the confirmed action with the same visible parameters.",
                "baseline_refs": ["ev_b1", "ev_b2"],
                "candidate_refs": ["ev_c1", "ev_c2"],
            }
        ],
        "reusable": [
            {
                "id": "confirmed-execution",
                "text": "After explicit confirmation, both candidate trajectories call the previously proposed execution tool with the same order and item parameters, without asking again.",
                "evidence_refs": ["ev_c1", "ev_c2"],
            }
        ],
        "needs_adjustment": [],
        "coverage": {
            "task_ids": ["1"],
            "baseline_refs": ["ev_b1", "ev_b2"],
            "candidate_refs": ["ev_c1", "ev_c2"],
        },
    }


def _comparison_index():
    return {
        "baseline_by_task": {"1": ["ev_b1", "ev_b2"]},
        "candidate_by_task": {"1": ["ev_c1", "ev_c2"]},
        "outcomes": {
            "ev_b1": "fail",
            "ev_b2": "pass",
            "ev_c1": "pass",
            "ev_c2": "pass",
        },
    }


def test_comparison_requires_exact_sets_and_candidate_evidence():
    validate_comparison(
        _comparison_output(), task_ids=("1",), index=_comparison_index()
    )


def test_comparison_status_is_not_overwritten_by_reward_counts():
    output = _comparison_output()
    output["task_comparisons"][0]["status"] = "mixed"
    validate_comparison(output, task_ids=("1",), index=_comparison_index())


def test_authoritative_comparison_status_is_canonicalized_from_outcomes():
    output = _comparison_output()
    output["task_comparisons"][0]["status"] = "stable_success"
    index = _comparison_index()
    index["evaluation_contract"] = {"outcome_authority": "authoritative"}
    index["outcomes"] = {
        "ev_b1": "fail",
        "ev_b2": "fail",
        "ev_c1": "pass",
        "ev_c2": "fail",
    }

    canonicalize_comparison(output, task_ids=("1",), index=index)

    assert output["task_comparisons"][0]["status"] == "recovered"
    validate_comparison(output, task_ids=("1",), index=index)


def test_authoritative_mixed_reference_to_all_pass_is_stable_success():
    output = _comparison_output()
    index = _comparison_index()
    index["evaluation_contract"] = {"outcome_authority": "authoritative"}

    canonicalize_comparison(output, task_ids=("1",), index=index)

    assert output["task_comparisons"][0]["status"] == "stable_success"
    assert output["task_comparisons"][0]["outcome_summary"] == {
        "reference_pass_count": 1,
        "reference_trial_count": 2,
        "candidate_pass_count": 2,
        "candidate_trial_count": 2,
    }
    validate_comparison(output, task_ids=("1",), index=index)


def test_authoritative_comparison_rejects_failure_only_reusable_passage():
    output = _comparison_output()
    output["reusable"][0]["evidence_refs"] = ["ev_b1"]
    index = _comparison_index()
    index["evaluation_contract"] = {"outcome_authority": "authoritative"}
    canonicalize_comparison(output, task_ids=("1",), index=index)

    with pytest.raises(ValueError, match="authoritative passing evidence"):
        validate_comparison(output, task_ids=("1",), index=index)


def test_comparison_treats_split_candidate_outcomes_as_mixed():
    output = _comparison_output()
    output["task_comparisons"][0]["status"] = "mixed"
    index = _comparison_index()
    index["outcomes"] = {
        "ev_b1": "pass",
        "ev_b2": "pass",
        "ev_c1": "pass",
        "ev_c2": "fail",
    }
    output["needs_adjustment"] = [
        {
            "id": "candidate-split-failure",
            "text": "One candidate trajectory transfers before executing the confirmed action, while the other candidate and both baseline trajectories execute it first.",
            "evidence_refs": ["ev_c2", "ev_c1", "ev_b1"],
        }
    ]

    validate_comparison(output, task_ids=("1",), index=index)


def test_comparison_accepts_behavioral_status_for_split_candidate_outcomes():
    output = _comparison_output()
    output["task_comparisons"][0]["status"] = "mixed"
    index = _comparison_index()
    index["outcomes"] = {
        "ev_b1": "pass",
        "ev_b2": "pass",
        "ev_c1": "pass",
        "ev_c2": "fail",
    }
    output["needs_adjustment"] = [
        {
            "id": "candidate-split-failure",
            "text": "One candidate trajectory transfers before executing the confirmed action, while the other candidate and both baseline trajectories execute it first.",
            "evidence_refs": ["ev_c2", "ev_c1", "ev_b1"],
        }
    ]

    validate_comparison(output, task_ids=("1",), index=index)


def test_comparison_allows_one_reusable_ref_for_duplicate_successes():
    output = _comparison_output()
    output["reusable"][0]["evidence_refs"] = ["ev_c1"]
    validate_comparison(output, task_ids=("1",), index=_comparison_index())


def test_comparison_canonicalizes_model_copied_reference_ids():
    output = _comparison_output()
    output["coverage"]["baseline_refs"] = ["typo"]
    output["coverage"]["candidate_refs"] = ["wrong"]
    output["task_comparisons"][0]["baseline_refs"] = ["typo"]
    output["task_comparisons"][0]["candidate_refs"] = ["wrong"]

    canonicalize_comparison(output, task_ids=("1",), index=_comparison_index())

    validate_comparison(output, task_ids=("1",), index=_comparison_index())


def test_comparison_canonicalization_preserves_passage_ref_validation():
    output = _comparison_output()
    output["reusable"][0]["evidence_refs"] = ["hallucinated"]
    canonicalize_comparison(output, task_ids=("1",), index=_comparison_index())

    with pytest.raises(ValueError, match="valid evidence"):
        validate_comparison(output, task_ids=("1",), index=_comparison_index())


def test_comparison_canonicalizes_near_miss_passage_evidence_refs():
    output = _comparison_output()
    index = _comparison_index()
    index["baseline_by_task"]["1"] = [
        "ev_3277d369770d0d5b8b28170efb2e4c4a351a2ffa3f86331cdd1e96e44f4bdca5"
    ]
    index["candidate_by_task"]["1"] = ["ev_c1"]
    output["coverage"]["baseline_refs"] = ["wrong"]
    output["coverage"]["candidate_refs"] = ["wrong"]
    output["task_comparisons"][0]["baseline_refs"] = ["wrong"]
    output["task_comparisons"][0]["candidate_refs"] = ["wrong"]
    output["reusable"][0]["evidence_refs"] = [
        "ev_3277f369770d0d5b8b28170efb2e4c4a351a2ffa3f86331cdd1e96e44f4bdca5",
        "ev_harnesslens_c1",  # candidate ref with an extra rollout prefix
        "hallucinated",
    ]

    canonicalize_comparison(output, task_ids=("1",), index=index)

    assert output["reusable"][0]["evidence_refs"] == [
        "ev_3277d369770d0d5b8b28170efb2e4c4a351a2ffa3f86331cdd1e96e44f4bdca5",
        "ev_c1",
    ]
    validate_comparison(output, task_ids=("1",), index=index)


def test_recovered_comparison_settles_stale_active_job(tmp_path):
    budget = CreationBudget(tmp_path / "budget.json", total=20, baseline_used=0)
    job_id = "experience-compare-iteration-01-comparison-task_001-task_002-task_003"
    budget.reserve_job(job_id, creation_count=1)
    budget.claim_launch(job_id)
    budget.mark_launched(job_id)

    _settle_recovered_intelligent_job(
        budget,
        base_job_id=job_id,
        workspace_name=job_id,
        artifact_path=tmp_path / "opencode.stdout",
    )

    state = json.loads((tmp_path / "budget.json").read_text(encoding="utf-8"))
    record = state["jobs"][job_id]
    assert record["status"] == "settled"
    assert record["outcome"] == "recovered"
    assert record["usage"]["details"]["artifact"].endswith("opencode.stdout")


def test_comparison_review_falls_back_to_validated_draft_after_exact_retries(
    tmp_path, monkeypatch
):
    module = _experience_module_for_retry(tmp_path)
    _AlwaysInvalidRunner.payloads = []
    monkeypatch.setattr(
        experience_module, "IntelligentHarnessRunner", _AlwaysInvalidRunner
    )
    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}\n", encoding="utf-8")
    draft_path = module.root / "draft.json"
    draft_path.write_text(json.dumps(_comparison_output()), encoding="utf-8")
    index = _comparison_index()
    index["bundle_paths"] = {"1": str(bundle)}
    index["evaluation_contract"] = {"outcome_authority": "behavioral"}

    output = module._review_comparison_group(
        label="comparison",
        task_ids=("1",),
        index=index,
        draft_path=draft_path,
    )

    assert output["coverage"] == _comparison_output()["coverage"]
    assert len(_AlwaysInvalidRunner.payloads) == 3
    assert _AlwaysInvalidRunner.payloads[0]["retry_context"] == ""
    assert _AlwaysInvalidRunner.payloads[1]["retry_context"].endswith(
        "exact-error-1"
    )
    assert _AlwaysInvalidRunner.payloads[2]["retry_context"].endswith(
        "exact-error-2"
    )
    fallback = json.loads(
        next(module.root.glob("experience-review-*_fallback.json")).read_text()
    )
    assert fallback["fallback"] == "validated_draft"
    assert fallback["last_validation_error"] == "exact-error-3"


def test_comparison_splits_output_limited_multi_task_job(tmp_path, monkeypatch):
    module = _experience_module_for_retry(tmp_path)
    _SplitComparisonRunner.calls = []
    monkeypatch.setattr(
        experience_module, "IntelligentHarnessRunner", _SplitComparisonRunner
    )
    index = {
        "baseline_by_task": {"1": [], "2": []},
        "candidate_by_task": {"1": [], "2": []},
        "outcomes": {},
        "bundle_paths": {"1": str(tmp_path / "one.json"), "2": str(tmp_path / "two.json")},
        "evaluation_contract": {"outcome_authority": "behavioral"},
    }

    result = module._run_comparison_group("comparison", ("1", "2"), index)

    assert _SplitComparisonRunner.calls[:3] == [("1", "2")] * 3
    assert _SplitComparisonRunner.calls[3:].count(("1",)) == 2
    assert _SplitComparisonRunner.calls[3:].count(("2",)) == 2
    assert [item["task_id"] for item in result["task_comparisons"]] == ["1", "2"]


def test_synthesis_failure_retains_every_validated_draft(tmp_path, monkeypatch):
    module = _experience_module_for_retry(tmp_path)
    _AlwaysInvalidRunner.payloads = []
    monkeypatch.setattr(
        experience_module, "IntelligentHarnessRunner", _AlwaysInvalidRunner
    )
    draft = _valid_output()
    source_index = {
        "task_ids": ["1"],
        "evidence_refs": ["ev_pass", "ev_fail"],
        "outcomes": {"ev_pass": "pass", "ev_fail": "fail"},
        "evaluation_contract": {"outcome_authority": "authoritative"},
    }

    output = module._synthesize(
        label="baseline",
        source_index=source_index,
        drafts=[draft],
    )

    assert [item["id"] for item in output["reusable"]] == ["route-a"]
    assert [item["id"] for item in output["needs_adjustment"]] == ["failure-a"]
    assert len(_AlwaysInvalidRunner.payloads) == 3
    assert _AlwaysInvalidRunner.payloads[1]["retry_context"].endswith(
        "exact-error-1"
    )
    marker = json.loads(
        (module.root / "experience-baseline-synthesis_fallback.json").read_text()
    )
    assert marker["fallback"] == "retain_all_validated_drafts"


def test_initial_experience_group_still_hard_fails_after_three_exact_retries(
    tmp_path, monkeypatch
):
    module = _experience_module_for_retry(tmp_path)
    _AlwaysInvalidRunner.payloads = []
    monkeypatch.setattr(
        experience_module, "IntelligentHarnessRunner", _AlwaysInvalidRunner
    )
    bundle = tmp_path / "bundle.json"
    bundle.write_text("{}\n", encoding="utf-8")
    source_index = {
        "task_ids": ["1"],
        "task_bundle_paths": [str(bundle)],
        "evidence_by_task": {"1": ["ev_pass", "ev_fail"]},
        "outcomes": {"ev_pass": "pass", "ev_fail": "fail"},
        "evaluation_contract": {"outcome_authority": "authoritative"},
    }

    with pytest.raises(RuntimeError, match="exact-error-3"):
        module._run_group(
            label="baseline-group",
            source_index=source_index,
            group={"id": "group", "task_ids": ["1"]},
        )

    assert len(_AlwaysInvalidRunner.payloads) == 3
    assert _AlwaysInvalidRunner.payloads[1]["retry_context"].endswith(
        "exact-error-1"
    )
    assert _AlwaysInvalidRunner.payloads[2]["retry_context"].endswith(
        "exact-error-2"
    )
