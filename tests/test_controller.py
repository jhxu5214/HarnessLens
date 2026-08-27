from types import SimpleNamespace
import importlib
import json

import pytest

from harnesslens.evolution.controller import (
    ANALYSIS_REUSE_WORKFLOW_FILES,
    _evaluation_requires_confirmation,
    _paired_rollout_outputs_exist,
    TOTAL_CREATION_BUDGET,
    _can_finalize_frozen_run,
    advance_cumulative_version,
    assert_candidate_extends_champion,
    assert_confirmed_promotion,
    confirmation_creation_cost,
    defer_residual_candidate,
    initialize_champion,
    minimum_residual_start_creations,
    minimum_screen_start_creations,
    prepare_analysis_reuse,
    reference_metrics_from_comparison_index,
    reject_unconfirmed_candidate,
    replan_candidate_from_review,
    revision_candidate_from_review,
    select_confirmation_task_ids,
)
from harnesslens.evolution.main_agent import MIN_CANDIDATE_ITERATION_CREATIONS


def test_iteration_budget_includes_selection_rollout_comparison_and_review():
    assert MIN_CANDIDATE_ITERATION_CREATIONS == 20
    assert TOTAL_CREATION_BUDGET == 200


def test_total_creation_budget_can_be_overridden(monkeypatch):
    import harnesslens.evolution.controller as controller_module

    monkeypatch.setenv("HAI_TOTAL_CREATION_BUDGET", "400")
    reloaded = importlib.reload(controller_module)
    try:
        assert reloaded.TOTAL_CREATION_BUDGET == 400
    finally:
        monkeypatch.delenv("HAI_TOTAL_CREATION_BUDGET", raising=False)
        importlib.reload(controller_module)


def test_controller_uses_retry_buffer_before_starting_an_iteration():
    from harnesslens.evolution.controller import MIN_ITERATION_START_CREATIONS

    assert MIN_ITERATION_START_CREATIONS == 23
    assert minimum_screen_start_creations("v0") == 23
    assert minimum_screen_start_creations("incumbent-00") == 33
    assert minimum_residual_start_creations("v0") == 15
    assert minimum_residual_start_creations("incumbent-00") == 19


def test_confirmation_budget_covers_fresh_candidate_and_exact_parent():
    assert confirmation_creation_cost(5, parent_version="v0") == 27
    assert confirmation_creation_cost(5, parent_version="incumbent-00") == 27


def test_terminal_screen_requires_separate_confirmation_for_promotion():
    assert _evaluation_requires_confirmation("standard") is True
    assert _evaluation_requires_confirmation("terminal_screen") is True
    assert _evaluation_requires_confirmation("residual_probe") is False


def test_confirmation_can_be_disabled_for_attribution_only_ablation(monkeypatch):
    monkeypatch.setenv("HAI_CONFIRMATION_MODE", "off")

    assert _evaluation_requires_confirmation("standard") is False
    assert _evaluation_requires_confirmation("terminal_screen") is False


def test_reference_metrics_preserve_repeat_order(tmp_path):
    path = tmp_path / "comparison_source_index.json"
    path.write_text(
        json.dumps(
            {
                "outcomes": {
                    "a0": "fail",
                    "a1": "pass",
                    "b0": "pass",
                    "b1": "fail",
                },
                "baseline_by_task": {"a": ["a0", "a1"], "b": ["b0", "b1"]},
            }
        ),
        encoding="utf-8",
    )

    assert reference_metrics_from_comparison_index(path) == {
        "pass_at_1": 0.5,
        "pass_at_2": 1.0,
    }


def test_reference_pass1_uses_all_independent_trials(tmp_path):
    path = tmp_path / "comparison_source_index.json"
    path.write_text(
        json.dumps(
            {
                "outcomes": {
                    "a0": "fail",
                    "a1": "fail",
                    "b0": "pass",
                    "b1": "fail",
                },
                "baseline_by_task": {"a": ["a0", "a1"], "b": ["b0", "b1"]},
            }
        ),
        encoding="utf-8",
    )

    assert reference_metrics_from_comparison_index(path) == {
        "pass_at_1": 0.25,
        "pass_at_2": 0.5,
    }


def test_analysis_reuse_requires_exact_baseline_and_charges_source_jobs(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    trajectories = tmp_path / "trajectories"
    trajectories.mkdir()
    artifacts = []
    for trial in range(2):
        path = trajectories / f"trial-{trial}.jsonl"
        path.write_text(json.dumps({"task_id": "0", "reward": trial}) + "\n")
        artifacts.append({"path": str(path), "evidence_id": f"ev-{trial}"})
    event = tmp_path / "baseline.json"
    event.write_text(
        json.dumps(
            {
                "baseline_fingerprint": {"task_ids": ["0"]},
                "agent_workspace_entry": {"trajectory_artifacts": artifacts},
            }
        ),
        encoding="utf-8",
    )

    for section in ("discovery", "experience", "analyzer"):
        (source / section).mkdir(parents=True)
    (source / "discovery" / "task_explorer.json").write_text("{}")
    (source / "discovery" / "harness_query.json").write_text(
        json.dumps({"harness": "opencode"})
    )
    (source / "discovery" / "task_explorer_input.json").write_text(
        json.dumps({"domain": "retail"})
    )
    source_index = {
        "task_ids": ["0"],
        "evidence_refs": ["ev-0", "ev-1"],
    }
    (source / "experience" / "baseline_source_index.json").write_text(
        json.dumps(source_index)
    )
    for name in ("baseline.json", "current.json"):
        (source / "experience" / name).write_text("{}")
    for name in (
        "baseline_reusable.json",
        "baseline_adjustment.json",
        "baseline_experience_dispositions.json",
    ):
        (source / "analyzer" / name).write_text("{}")
    (source / "creation_budget.json").write_text(
        json.dumps(
            {
                "jobs": {
                    "discovery-task-explorer": {
                        "status": "settled",
                        "creation_count": 1,
                    },
                    "discovery-harness-query": {
                        "status": "settled",
                        "creation_count": 1,
                    },
                    "experience-baseline-a": {
                        "status": "settled",
                        "creation_count": 2,
                    },
                    "main-agent-iteration-01": {
                        "status": "settled",
                        "creation_count": 9,
                    },
                }
            }
        )
    )
    analysis_files = {
        path: f"sha-{index}"
        for index, path in enumerate(ANALYSIS_REUSE_WORKFLOW_FILES)
    }
    workflow = {
        "schema": "harnesslens.workflow-source.v1",
        "sha256": "full-workflow",
        "files": analysis_files,
    }
    (source / "workflow_fingerprint.json").write_text(json.dumps(workflow))

    assert prepare_analysis_reuse(
        source_run=source,
        target_run=target,
        baseline_event=event,
        harness="opencode",
        cell="retail",
        workflow_fingerprint=workflow,
    ) == 4
    assert (target / "discovery" / "harness_query.json").is_file()
    assert (target / "experience" / "baseline.json").is_file()
    assert json.loads((target / "analysis_reuse.json").read_text())["reused_creations"] == 4


def test_analysis_reuse_rejects_a_different_analysis_workflow(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    source_workflow = {
        "schema": "harnesslens.workflow-source.v1",
        "sha256": "source",
        "files": {
            path: f"sha-{index}"
            for index, path in enumerate(ANALYSIS_REUSE_WORKFLOW_FILES)
        },
    }
    source_workflow["files"][ANALYSIS_REUSE_WORKFLOW_FILES[0]] = "old-analysis"
    (source / "workflow_fingerprint.json").write_text(json.dumps(source_workflow))
    current_workflow = {
        **source_workflow,
        "sha256": "current",
        "files": {
            **source_workflow["files"],
            ANALYSIS_REUSE_WORKFLOW_FILES[0]: "current-analysis",
        },
    }

    with pytest.raises(ValueError, match="analysis workflow differs"):
        prepare_analysis_reuse(
            source_run=source,
            target_run=tmp_path / "target",
            baseline_event=tmp_path / "unused.json",
            harness="opencode",
            cell="retail",
            workflow_fingerprint=current_workflow,
        )


def test_frozen_run_only_finalizes_after_budget_and_all_reviews(tmp_path):
    main = tmp_path / "main_agent"
    main.mkdir()
    (main / "iteration-01.json").write_text("{}", encoding="utf-8")

    assert not _can_finalize_frozen_run(tmp_path, {"remaining": 0})
    (main / "iteration-01-review.json").write_text("{}", encoding="utf-8")
    assert not _can_finalize_frozen_run(tmp_path, {"remaining": 1})
    assert _can_finalize_frozen_run(tmp_path, {"remaining": 0})


def test_cumulative_version_advances_only_to_a_direct_child():
    decision = SimpleNamespace(
        output={"candidate": {"parent_version": "v1"}},
        harness_version="v2",
    )

    assert advance_cumulative_version(
        current_version="v1",
        decision=decision,
        review=SimpleNamespace(harness_version="v2"),
    ) == ("v2", True)
    assert advance_cumulative_version(
        current_version="v1",
        decision=decision,
        review=SimpleNamespace(harness_version="v1"),
    ) == ("v1", False)

    with pytest.raises(RuntimeError, match="parent differs"):
        advance_cumulative_version(
            current_version="v0",
            decision=decision,
            review=SimpleNamespace(harness_version="v2"),
        )


def test_candidate_must_extend_the_current_champion_before_rollout():
    decision = SimpleNamespace(
        output={"candidate": {"parent_version": "candidate-01"}},
        harness_version="candidate-02",
    )

    assert_candidate_extends_champion(
        current_version="candidate-01", decision=decision
    )
    with pytest.raises(RuntimeError, match="must extend the current champion"):
        assert_candidate_extends_champion(current_version="v0", decision=decision)


def test_child_cannot_be_promoted_without_independent_confirmation():
    decision = SimpleNamespace(harness_version="candidate-02")
    review = SimpleNamespace(harness_version="candidate-02")

    with pytest.raises(RuntimeError, match="independent confirmation"):
        assert_confirmed_promotion(
            decision=decision,
            review=review,
            confirmation_pair=None,
        )
    assert_confirmed_promotion(
        decision=decision,
        review=review,
        confirmation_pair=SimpleNamespace(),
    )


def test_confirmation_combines_effect_anchors_with_fresh_diverse_canaries(tmp_path):
    (tmp_path / "experience").mkdir()
    (tmp_path / "discovery").mkdir()
    task_ids = ("a", "b", "c", "d", "e", "f", "g", "h")
    evidence_by_task = {
        task_id: [f"{task_id}-0", f"{task_id}-1"] for task_id in task_ids
    }
    outcomes = {
        ref: "pass" for refs in evidence_by_task.values() for ref in refs
    }
    (tmp_path / "experience" / "baseline_source_index.json").write_text(
        json.dumps(
            {
                "task_ids": list(task_ids),
                "evidence_by_task": evidence_by_task,
                "outcomes": outcomes,
            }
        )
    )
    (tmp_path / "experience" / "current.json").write_text(
        json.dumps(
            {
                "reusable": [],
                "needs_adjustment": [
                    {"id": "direct-exp", "evidence_refs": evidence_by_task["b"]}
                ],
            }
        )
    )
    (tmp_path / "discovery" / "task_explorer.json").write_text(
        json.dumps(
            {
                "categories": [
                    {"id": "cat-1", "task_ids": ["a", "f"]},
                    {"id": "cat-2", "task_ids": ["b", "g"]},
                    {"id": "cat-3", "task_ids": ["c", "h"]},
                    {"id": "cat-4", "task_ids": ["d", "e"]},
                ]
            }
        )
    )
    decision = SimpleNamespace(
        output={
            "candidate": {
                "channel_diffs": [
                    {"channel_id": "instructions", "experience_ids": ["direct-exp"]}
                ]
            }
        }
    )
    screen_review = SimpleNamespace(
        output={"evidence": {"recovered_task_ids": ["a"]}}
    )

    selected = select_confirmation_task_ids(
        run_root=tmp_path,
        decision=decision,
        screen_review=screen_review,
        screen_task_ids=("a", "b", "c", "d", "e"),
        task_count=5,
    )

    assert selected[:2] == ("a", "b")
    assert selected[2] == "h"
    assert set(selected[3:]) == {"f", "g"}


def test_confirmation_selection_does_not_overfill_after_fresh_pool_fills_target(
    tmp_path,
):
    (tmp_path / "experience").mkdir()
    (tmp_path / "discovery").mkdir()
    task_ids = ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j")
    evidence_by_task = {
        task_id: [f"{task_id}-0", f"{task_id}-1"] for task_id in task_ids
    }
    (tmp_path / "experience" / "baseline_source_index.json").write_text(
        json.dumps(
            {
                "task_ids": list(task_ids),
                "evidence_by_task": evidence_by_task,
                "outcomes": {
                    ref: "pass"
                    for refs in evidence_by_task.values()
                    for ref in refs
                },
            }
        )
    )
    (tmp_path / "experience" / "current.json").write_text(
        json.dumps({"reusable": [], "needs_adjustment": []})
    )
    (tmp_path / "discovery" / "task_explorer.json").write_text(
        json.dumps({"categories": []})
    )
    decision = SimpleNamespace(output={"candidate": {"channel_diffs": []}})
    screen_review = SimpleNamespace(
        output={"evidence": {"recovered_task_ids": ["a"]}}
    )

    selected = select_confirmation_task_ids(
        run_root=tmp_path,
        decision=decision,
        screen_review=screen_review,
        screen_task_ids=("a", "b", "c", "d", "e", "f"),
        task_count=6,
    )

    assert len(selected) == 6
    assert selected[0] == "a"
    assert set(selected[2:]) == {"g", "h", "i", "j"}


def test_explicit_incumbent_initializes_the_current_champion():
    class Repository:
        def materialize_candidate(self, **kwargs):
            assert kwargs == {
                "base_version": "v0",
                "candidate_label": "incumbent-00",
                "delta": {"instructions": ["Keep prior behavior."]},
            }
            return "incumbent-00"

    version, context, imported_ids = initialize_champion(
        repository=Repository(),
        incumbent_candidates=[
            {
                "id": "incumbent-abc",
                "manifest_delta": {"instructions": ["Keep prior behavior."]},
                "_direct_task_ids": ["7"],
                "_prior_train_evidence": {
                    "recovered_task_ids": ["7"],
                    "preserved_task_ids": ["8"],
                    "attributable_regression_task_ids": [],
                },
            }
        ],
    )

    assert version == "incumbent-00"
    assert imported_ids == {"incumbent-abc"}
    assert context[0]["review_decision"] == "imported_champion"
    assert context[0]["review_evidence"]["recovered_task_ids"] == ["7"]


def test_multiple_incumbents_do_not_silently_choose_a_champion():
    with pytest.raises(ValueError, match="exactly one"):
        initialize_champion(
            repository=object(),
            incumbent_candidates=[{"id": "a"}, {"id": "b"}],
        )


def test_revise_review_queues_candidate_without_accepting_it():
    revision = {"id": "revision-candidate-01", "manifest_delta": {"files": []}}

    assert revision_candidate_from_review(
        {"decision": "revise_delta", "revision_candidate": revision}
    ) == revision
    assert revision_candidate_from_review({"decision": "reject_delta"}) is None

    with pytest.raises(RuntimeError, match="missing"):
        revision_candidate_from_review({"decision": "revise_delta"})


def test_revision_inherits_the_tested_task_evidence():
    revision = revision_candidate_from_review(
        {
            "decision": "revise_delta",
            "revision_candidate": {"id": "revision", "manifest_delta": {"files": []}},
            "evidence": {"recovered_task_ids": ["7"]},
        },
        rollout_task_ids=["8", "7", "8"],
    )

    assert revision["_direct_task_ids"] == ["7", "8"]
    assert revision["_prior_train_evidence"]["recovered_task_ids"] == ["7"]


def test_unconfirmed_screen_keeps_parent(tmp_path):
    screen = SimpleNamespace(
        output={
            "decision": "accept_delta",
            "selected_version": "candidate-01",
            "evidence": {"unresolved_findings": []},
            "revision_candidate": {"id": "must-not-survive"},
        }
    )

    review = reject_unconfirmed_candidate(
        screen_review=screen,
        parent_version="incumbent-00",
        output_path=tmp_path / "review.json",
    )

    assert review.output["decision"] == "reject_delta"
    assert review.harness_version == "incumbent-00"
    assert "revision_candidate" not in review.output
    assert "confirmation" in review.output["evidence"]["unresolved_findings"][0]


def test_cached_confirmation_pair_survives_low_budget_resume(tmp_path):
    rollout = tmp_path / "rollout"
    rollout.mkdir()
    label = "iteration-01-confirm-candidate-01"
    (rollout / f"{label}.json").write_text("{}\n", encoding="utf-8")

    assert not _paired_rollout_outputs_exist(
        run_root=tmp_path,
        label=label,
        parent_version="v0",
    )

    (rollout / f"{label}-parent-v0.json").write_text("{}\n", encoding="utf-8")

    assert _paired_rollout_outputs_exist(
        run_root=tmp_path,
        label=label,
        parent_version="v0",
    )


def test_residual_probe_cannot_promote_a_candidate(tmp_path):
    screen = SimpleNamespace(
        output={
            "decision": "accept_delta",
            "selected_version": "candidate-01",
            "evidence": {"unresolved_findings": []},
            "revision_candidate": {"id": "must-not-survive"},
        }
    )

    review = defer_residual_candidate(
        screen_review=screen,
        parent_version="incumbent-00",
        output_path=tmp_path / "review.json",
    )

    assert review.output["decision"] == "reject_delta"
    assert review.output["evidence_disposition"] == "diagnostic_only"
    assert review.harness_version == "incumbent-00"
    assert "revision_candidate" not in review.output
    assert "Residual probe" in review.output["evidence"]["unresolved_findings"][0]


def test_replan_problem_returns_post_analyzer_candidate_to_portfolio():
    candidate = {
        "id": "replan-count-distinct",
        "objective": "Check entity cardinality before choosing COUNT.",
        "observed_terminal_failure": "The query counts rows instead of entities.",
        "causal_hypothesis": "The agent skips a cardinality check.",
        "intervention_point": "developer_instructions",
        "expected_runtime_event": "The agent compares row and distinct-entity counts.",
        "falsifying_observation": "The check occurs but COUNT remains row-level.",
        "channel_plan": [
            {
                "channel_id": "developer_instructions",
                "operation": "append",
                "experience_ids": ["count-failure"],
                "rationale": "The check must be visible before query construction.",
            }
        ],
        "validation": {"local_behavior_checks": ["Uses entity cardinality."]},
    }
    result = replan_candidate_from_review(
        {"decision": "replan_problem", "evidence": {"recovered_task_ids": []}},
        adjustment={"replan_candidate": candidate},
        rollout_task_ids=("bird_1037",),
    )

    assert result["id"] == "replan-count-distinct"
    assert result["_portfolio_side"] == "adjustment"
    assert result["_direct_task_ids"] == ["bird_1037"]
