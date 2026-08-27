import json

import pytest

from harnesslens.evolution.analyzer import (
    ADJUSTMENT_ANALYZER_SYSTEM,
    POST_ADJUSTMENT_ANALYZER_SYSTEM,
    POST_REUSABLE_ANALYZER_SYSTEM,
    REUSABLE_ANALYZER_SYSTEM,
    _validate_candidate,
    _post_candidate_context,
    _query_channel_contracts,
    _rollout_channel_usage,
    analyzer_retry_context,
    build_experience_dispositions,
    canonicalize_analyzer_output,
    canonicalize_post_analyzer_output,
    combine_adjustment_outputs,
    combine_reusable_outputs,
    conservative_post_analyzer_fallback,
    partition_adjustment_experiences,
    validate_analyzer_output,
    validate_reusable_plan,
    validate_post_analyzer_output,
)


def test_query_channel_contracts_exclude_conditional_surfaces():
    contracts = _query_channel_contracts(
        {
            "modifiable_modules": [
                {"id": "project_instructions", "status": "modifiable"}
            ],
            "mcp_editable_points": [
                {"id": "mcp_tool_description", "status": "conditional"}
            ],
        }
    )

    assert set(contracts) == {"project_instructions"}


def test_adjustment_prompt_delegates_concrete_files_to_harness_editor():
    assert (
        "evidence supports a concrete behavior hypothesis" in ADJUSTMENT_ANALYZER_SYSTEM
    )
    assert "Editor owns concrete native changes" in ADJUSTMENT_ANALYZER_SYSTEM
    assert (
        "do not write harness\nfiles or `manifest_delta`" in ADJUSTMENT_ANALYZER_SYSTEM
    )
    assert (
        '"manifest_delta"'
        not in ADJUSTMENT_ANALYZER_SYSTEM.split("Return exactly:", 1)[1]
    )


def test_analyzer_prompts_require_atomic_behavior_hypotheses():
    assert "one atomic behavior\nhypothesis" in REUSABLE_ANALYZER_SYSTEM
    assert "exactly one problem" in ADJUSTMENT_ANALYZER_SYSTEM
    assert (
        "independent problem fixes must remain separate" in ADJUSTMENT_ANALYZER_SYSTEM
    )


def test_analyzer_prompts_compile_minimal_causal_invariants():
    assert "smallest causal invariant" in REUSABLE_ANALYZER_SYSTEM
    assert "smallest causal invariant" in ADJUSTMENT_ANALYZER_SYSTEM
    assert "incidental syntax" in REUSABLE_ANALYZER_SYSTEM


def test_post_analyzers_use_channel_specific_observability():
    for prompt in (POST_REUSABLE_ANALYZER_SYSTEM, POST_ADJUSTMENT_ANALYZER_SYSTEM):
        assert "startup-visible" in prompt
        assert "not invoked" in prompt
        assert "on-demand" in prompt


def test_post_analyzer_canonicalization_copies_metric_outcomes():
    output = {
        "coverage": {},
        "primary_problem": {
            "task_assessments": [
                {
                    "task_id": "task-a",
                    "status": "stable_success",
                    "relation": "attributed",
                    "evidence_refs": ["ev-a"],
                    "reason": "Visible behavior improved in the candidate trial set.",
                }
            ],
            "channel_attribution": {
                "relation": "attributed",
                "channel_ids": ["skills"],
                "reason": "The skill was invoked on the changed behavior.",
            },
            "summary": "The target behavior improved.",
            "local_recovery": "The target behavior became consistent.",
            "recommendation": "accept",
            "further_rollout_needed": False,
        },
    }
    summary = {
        "task-a": {
            "reference_pass_count": 1,
            "reference_trial_count": 2,
            "candidate_pass_count": 2,
            "candidate_trial_count": 2,
        }
    }

    canonical = canonicalize_post_analyzer_output(
        output,
        side="adjustment",
        baseline_ids=("exp-a",),
        comparison_ids=("cmp-a",),
        task_statuses={"task-a": "stable_success"},
        changed_channels={"skills"},
        task_outcomes=summary,
        available_channel_ids={"skills", "mcp_tool_description"},
    )

    assert (
        canonical["primary_problem"]["task_assessments"][0]["outcome_summary"]
        == summary["task-a"]
    )


def test_analyzer_retry_context_carries_the_exact_validator_error():
    context = analyzer_retry_context("candidate requires a nonempty manifest_delta")

    assert context.startswith(
        "Previous output failed validation: candidate requires a nonempty manifest_delta."
    )
    assert "evidence-backed editor brief" in context
    assert "do not write manifest_delta" in context
    assert "never put multiple channel_id/operation/experience_ids fields" in context


def test_reusable_plan_partitions_every_experience_once():
    plan = {
        "groups": [
            {"id": "orders", "experience_ids": ["exp-a"]},
            {"id": "returns", "experience_ids": ["exp-b"]},
        ]
    }
    validate_reusable_plan(plan, experience_ids=("exp-a", "exp-b"))

    plan["groups"][1]["experience_ids"] = ["exp-a"]
    with pytest.raises(ValueError, match="partition every experience"):
        validate_reusable_plan(plan, experience_ids=("exp-a", "exp-b"))


def test_reusable_group_outputs_keep_one_candidate_per_source_hypothesis():
    first = _candidate()
    second = _candidate()
    second["id"] = "route-b"
    second["channel_plan"][0]["experience_ids"] = ["exp-b"]
    second["manifest_delta"]["files"][0] = {
        "path": ".opencode/skills/route-b/SKILL.md",
        "content": "---\nname: route-b\ndescription: Use for route B.\n---\nRoute B details.",
    }
    combined = combine_reusable_outputs(
        plan={
            "groups": [
                {"id": "one", "experience_ids": ["exp-a"]},
                {"id": "two", "experience_ids": ["exp-b"]},
            ]
        },
        outputs=[
            ("one", {"candidates": [first]}),
            ("two", {"candidates": [second]}),
        ],
    )

    assert [item["id"] for item in combined["candidates"]] == [
        "reusable-one-candidate-a",
        "reusable-two-route-b",
    ]
    first_manifest = combined["candidates"][0]["manifest_delta"]
    second_manifest = combined["candidates"][1]["manifest_delta"]
    assert first_manifest["files"][0]["path"] == (
        ".opencode/skills/one-candidate-a-bounded-route/SKILL.md"
    )
    assert second_manifest["files"][0]["path"] == (
        ".opencode/skills/two-route-b-route-b/SKILL.md"
    )
    assert "Apply the bounded route" in first_manifest["files"][0]["content"]
    assert "Route B details" in second_manifest["files"][0]["content"]
    validate_analyzer_output(
        combined,
        side="reusable",
        experience_ids=("exp-a", "exp-b"),
        channel_ids={"skills"},
    )


def test_adjustment_partitioning_bounds_group_size_and_preserves_every_experience():
    experiences = [{"id": f"exp-{index}"} for index in range(15)]

    groups = partition_adjustment_experiences(experiences)

    assert len(groups) == 4
    assert max(len(items) for _, items in groups) == 4
    assert {
        str(item["id"]) for _, items in groups for item in items
    } == {str(item["id"]) for item in experiences}


def test_adjustment_partition_merge_namespaces_ids_and_global_priorities():
    def output(experience_id):
        candidate = {
            "id": "candidate-a",
            "priority": 1,
            "objective": "Repair one observable failure.",
            "observed_terminal_failure": "The final result shape mismatches.",
            "causal_hypothesis": "The agent skips a shape check.",
            "intervention_point": "instructions_rules",
            "expected_runtime_event": "The agent checks the result shape.",
            "falsifying_observation": "The check occurs but the mismatch remains.",
            "channel_plan": [
                {
                    "channel_id": "instructions_rules",
                    "operation": "append",
                    "experience_ids": [experience_id],
                    "rationale": "Visible before final SQL construction.",
                }
            ],
            "validation": {"local_behavior_checks": ["Checks result shape."]},
        }
        return {
            "coverage": {"experience_ids": [experience_id]},
            "problems": [
                {
                    "id": "problem-a",
                    "priority": 1,
                    "summary": "Wrong result shape.",
                    "experience_ids": [experience_id],
                    "evidence_refs": [f"ev-{experience_id}"],
                    "channel_hypotheses": [
                        {"channel_id": "instructions_rules", "reason": "visible"}
                    ],
                    "modification_direction": "Add a bounded shape check.",
                    "diagnostic_rollout_needed": True,
                    "local_success_criteria": ["The result shape matches."],
                    "candidate_id": "candidate-a",
                }
            ],
            "candidates": [candidate],
        }

    combined = combine_adjustment_outputs(
        experience_ids=("exp-a", "exp-b"),
        outputs=(
            ("adjustment-01", output("exp-a")),
            ("adjustment-02", output("exp-b")),
        ),
    )

    assert [item["id"] for item in combined["candidates"]] == [
        "adjustment-01-candidate-a",
        "adjustment-02-candidate-a",
    ]
    assert [item["priority"] for item in combined["problems"]] == [1, 2]
    validate_analyzer_output(
        combined,
        side="adjustment",
        experience_ids=("exp-a", "exp-b"),
        channel_ids={"instructions_rules"},
    )


def test_reusable_candidates_in_the_same_planner_group_remain_atomic():
    first = _candidate()
    second = _candidate()
    second["id"] = "route-b"
    second["channel_plan"][0]["experience_ids"] = ["exp-b"]
    second["channel_plan"][0]["preserve_experience_ids"] = []
    second["manifest_delta"]["files"][0] = {
        "path": ".opencode/skills/route-b/SKILL.md",
        "content": "---\nname: route-b\ndescription: Use for route B.\n---\nRoute B details.",
    }
    second["validation"]["no_regression_experience_ids"] = []

    combined = combine_reusable_outputs(
        plan={"groups": [{"id": "orders", "experience_ids": ["exp-a", "exp-b"]}]},
        outputs=[("orders", {"candidates": [first, second]})],
    )

    assert [item["id"] for item in combined["candidates"]] == [
        "reusable-orders-candidate-a",
        "reusable-orders-route-b",
    ]
    assert [
        sorted(
            {
                experience_id
                for plan in item["channel_plan"]
                for experience_id in plan["experience_ids"]
            }
        )
        for item in combined["candidates"]
    ] == [["exp-a"], ["exp-b"]]


def test_adjustment_candidates_cannot_bundle_independent_problems():
    candidate = {
        "id": "combined-fix",
        "objective": "Fix two unrelated failures.",
        "observed_terminal_failure": "Two unrelated terminal failures were observed.",
        "causal_hypothesis": "One global rule could affect both failures.",
        "intervention_point": "instructions_rules",
        "expected_runtime_event": "Both target behaviors change.",
        "falsifying_observation": "Only one or neither behavior changes.",
        "channel_plan": [
            {
                "channel_id": "instructions_rules",
                "operation": "append",
                "experience_ids": ["exp-a", "exp-b"],
                "rationale": "Both are startup-visible.",
            }
        ],
        "manifest_delta": {"instructions": ["Apply both unrelated rules."]},
        "validation": {"local_behavior_checks": ["Both behaviors recover."]},
    }
    problems = [
        {
            "id": problem_id,
            "priority": priority,
            "summary": summary,
            "experience_ids": [experience_id],
            "evidence_refs": [evidence_ref],
            "channel_hypotheses": [
                {"channel_id": "instructions_rules", "reason": "startup-visible"}
            ],
            "modification_direction": "Add one bounded rule.",
            "diagnostic_rollout_needed": True,
            "local_success_criteria": ["The bounded behavior recovers."],
            "candidate_id": "combined-fix",
        }
        for priority, problem_id, summary, experience_id, evidence_ref in (
            (1, "confirmation", "Execute after confirmation.", "exp-a", "ev-a"),
            (2, "retrieval", "Use the matching policy source.", "exp-b", "ev-b"),
        )
    ]

    with pytest.raises(ValueError, match="one atomic problem"):
        validate_analyzer_output(
            {
                "coverage": {"experience_ids": ["exp-a", "exp-b"]},
                "problems": problems,
                "candidates": [candidate],
            },
            side="adjustment",
            experience_ids=("exp-a", "exp-b"),
            channel_ids={"instructions_rules"},
        )


def test_diagnostic_only_adjustment_problem_can_have_no_channel_hypothesis():
    validate_analyzer_output(
        {
            "coverage": {"experience_ids": ["exp-a"]},
            "problems": [
                {
                    "id": "unattributed-mismatch",
                    "priority": 1,
                    "summary": "The visible behavior does not explain the mismatch.",
                    "experience_ids": ["exp-a"],
                    "evidence_refs": ["ev-a"],
                    "channel_hypotheses": [],
                    "modification_direction": "Diagnose the hidden mismatch before editing.",
                    "diagnostic_rollout_needed": True,
                    "local_success_criteria": ["The mismatch cause is identified."],
                    "candidate_id": "",
                }
            ],
            "candidates": [],
        },
        side="adjustment",
        experience_ids=("exp-a",),
        channel_ids={"instructions_rules"},
    )


def test_actionable_adjustment_problem_requires_a_channel_hypothesis():
    candidate = _candidate()
    candidate.update(
        {
            "observed_terminal_failure": "The final result shape mismatches.",
            "causal_hypothesis": "The agent skips a shape check.",
            "intervention_point": "instructions_rules",
            "expected_runtime_event": "The agent checks the result shape.",
            "falsifying_observation": "The check occurs but the mismatch remains.",
        }
    )
    with pytest.raises(
        ValueError, match="actionable adjustment problem requires channel hypotheses"
    ):
        validate_analyzer_output(
            {
                "coverage": {"experience_ids": ["exp-a"]},
                "problems": [
                    {
                        "id": "shape-mismatch",
                        "priority": 1,
                        "summary": "The final result shape mismatches.",
                        "experience_ids": ["exp-a"],
                        "evidence_refs": ["ev-a"],
                        "channel_hypotheses": [],
                        "modification_direction": "Add a bounded shape check.",
                        "diagnostic_rollout_needed": True,
                        "local_success_criteria": ["The result shape matches."],
                        "candidate_id": "candidate-a",
                    }
                ],
                "candidates": [candidate],
            },
            side="adjustment",
            experience_ids=("exp-a",),
            channel_ids={"instructions_rules"},
        )


def _candidate():
    return {
        "id": "candidate-a",
        "objective": "Preserve a detailed route.",
        "channel_plan": [
            {
                "channel_id": "skills",
                "operation": "add procedure",
                "experience_ids": ["exp-a"],
                "preserve_experience_ids": ["exp-b"],
                "rationale": "On-demand detail.",
            }
        ],
        "manifest_delta": {
            "config_patch": {"tools.skill": True},
            "files": [
                {
                    "path": ".opencode/skills/bounded-route/SKILL.md",
                    "content": (
                        "---\nname: bounded-route\n"
                        "description: Use when the bounded route applies.\n---\n\n"
                        "Apply the bounded route and preserve its branches.\n"
                    ),
                }
            ],
        },
        "validation": {
            "no_regression_experience_ids": ["exp-b"],
            "local_behavior_checks": ["The route remains available."],
        },
    }


def test_candidate_without_manifest_delta_survives_canonicalization_for_editor():
    candidate = _candidate()
    candidate.pop("manifest_delta")
    output = {
        "coverage": {"experience_ids": ["exp-a", "exp-b"]},
        "candidates": [candidate],
    }

    canonicalize_analyzer_output(
        output,
        harness_query={"harness": "opencode"},
    )
    _validate_candidate(
        output["candidates"][0],
        experience_ids={"exp-a", "exp-b"},
        channel_ids={"skills"},
    )

    assert output["candidates"][0]["channel_plan"][0]["channel_id"] == "skills"
    assert output["candidates"][0]["manifest_delta"] == {}


def test_adjustment_candidate_requires_falsifiable_causal_contract():
    candidate = _candidate()
    candidate.pop("manifest_delta")

    with pytest.raises(ValueError, match="causal contract"):
        _validate_candidate(
            candidate,
            experience_ids={"exp-a", "exp-b"},
            channel_ids={"skills"},
            require_causal_contract=True,
        )

    candidate.update(
        {
            "observed_terminal_failure": "The final query returns the wrong shape.",
            "causal_hypothesis": "The agent does not inspect result shape before submission.",
            "intervention_point": "skills",
            "expected_runtime_event": "The agent executes and compares a shape check.",
            "falsifying_observation": "The shape check occurs but the result still mismatches.",
        }
    )
    _validate_candidate(
        candidate,
        experience_ids={"exp-a", "exp-b"},
        channel_ids={"skills"},
        require_causal_contract=True,
    )


def test_analyzer_canonicalization_restores_unique_truncated_experience_id():
    full_id = "experience-count-threshold-mismatch-between-question-and-evidence"
    truncated = "experience-count-threshold-mismatch"
    candidate = _candidate()
    candidate["channel_plan"][0]["experience_ids"] = [truncated]
    output = {
        "coverage": {"experience_ids": [truncated, "experience-other"]},
        "problems": [
            {
                "id": "problem",
                "summary": "summary",
                "experience_ids": [truncated],
                "evidence_refs": ["ev"],
                "channel_hypotheses": [{"channel_id": "skills"}],
                "modification_direction": "direction",
                "local_success_criteria": ["check"],
            }
        ],
        "candidates": [candidate],
    }

    normalized = canonicalize_analyzer_output(
        output,
        expected_experience_ids=(full_id, "experience-other"),
    )

    assert set(normalized["coverage"]["experience_ids"]) == {
        full_id,
        "experience-other",
    }
    assert normalized["problems"][0]["experience_ids"] == [full_id]
    assert normalized["candidates"][0]["channel_plan"][0]["experience_ids"] == [full_id]


def test_reusable_analyzer_requires_full_coverage_and_materializable_candidate():
    candidate = _candidate()
    candidate["channel_plan"][0]["experience_ids"] = ["exp-a", "exp-b"]
    validate_analyzer_output(
        {
            "coverage": {
                "experience_ids": ["exp-a", "exp-b"],
                "public_environment_reviewed": True,
            },
            "candidates": [candidate],
        },
        side="reusable",
        experience_ids=("exp-a", "exp-b"),
        channel_ids={"skills"},
    )


def test_adjustment_analyzer_cannot_choose_rollout_tasks():
    output = {
        "coverage": {"experience_ids": ["exp-a"]},
        "problems": [
            {
                "id": "problem-a",
                "priority": 1,
                "summary": "Missing execution after confirmation.",
                "experience_ids": ["exp-a"],
                "evidence_refs": ["ev-a"],
                "channel_hypotheses": [
                    {"channel_id": "instructions_rules", "reason": "global"}
                ],
                "modification_direction": "Make confirmed plans executable.",
                "diagnostic_rollout_needed": True,
                "local_success_criteria": ["Tool call follows confirmation."],
                "rollout_task_ids": ["0", "1", "2", "3", "4"],
            }
        ],
        "candidates": [],
    }
    with pytest.raises(ValueError, match="must not select"):
        validate_analyzer_output(
            output,
            side="adjustment",
            experience_ids=("exp-a",),
            channel_ids={"instructions_rules"},
        )


def test_analyzer_allows_multiple_candidates_across_channels():
    second = _candidate()
    second["id"] = "candidate-b"
    second["channel_plan"][0]["channel_id"] = "instructions_rules"
    second["channel_plan"][0]["experience_ids"] = ["exp-b"]
    second["manifest_delta"] = {"instructions": ["Use the bounded route."]}

    validate_analyzer_output(
        {
            "coverage": {
                "experience_ids": ["exp-a", "exp-b"],
                "public_environment_reviewed": True,
            },
            "candidates": [_candidate(), second],
        },
        side="reusable",
        experience_ids=("exp-a", "exp-b"),
        channel_ids={"skills", "instructions_rules"},
    )


def test_reusable_analyzer_requires_every_experience_in_a_compilation_batch():
    output = {
        "coverage": {
            "experience_ids": ["exp-a", "exp-b"],
            "public_environment_reviewed": True,
        },
        "candidates": [_candidate()],
    }

    with pytest.raises(ValueError, match="outside compilation batches"):
        validate_analyzer_output(
            output,
            side="reusable",
            experience_ids=("exp-a", "exp-b"),
            channel_ids={"skills"},
        )


def test_reusable_candidate_can_materialize_complementary_files_for_one_behavior():
    candidate = _candidate()
    candidate["channel_plan"][0]["experience_ids"] = ["exp-a", "exp-b"]
    candidate["manifest_delta"]["files"].append(
        {
            "path": ".opencode/skills/bounded-route-reference/SKILL.md",
            "content": (
                "---\nname: bounded-route-reference\n"
                "description: Reference checks for the bounded route.\n---\n"
                "Apply the reference checks needed by the same bounded route."
            ),
        }
    )
    output = {
        "coverage": {
            "experience_ids": ["exp-a", "exp-b"],
            "public_environment_reviewed": True,
        },
        "candidates": [candidate],
    }

    validate_analyzer_output(
        output,
        side="reusable",
        experience_ids=("exp-a", "exp-b"),
        channel_ids={"skills"},
    )


def test_analyzer_allows_complementary_channels_for_one_behavior():
    candidate = _candidate()
    candidate["channel_plan"][0]["experience_ids"] = ["exp-a", "exp-b"]
    candidate["channel_plan"].append(
        {
            "channel_id": "tool_description",
            "operation": "clarify execution point",
            "experience_ids": ["exp-a"],
            "rationale": "The skill carries the SOP while the tool schema marks the call boundary.",
        }
    )
    candidate["manifest_delta"]["tool_desc_patches"] = {
        "modify_pending_order_items": {
            "desc": "Use after the confirmed item plan is complete."
        }
    }

    validate_analyzer_output(
        {
            "coverage": {
                "experience_ids": ["exp-a", "exp-b"],
                "public_environment_reviewed": True,
            },
            "candidates": [candidate],
        },
        side="reusable",
        experience_ids=("exp-a", "exp-b"),
        channel_ids={"skills", "tool_description"},
    )


def test_analyzer_accepts_mcp_editable_point_as_materialized_base_channel():
    candidate = _candidate()
    candidate["channel_plan"] = [
        {
            "channel_id": "mcp_tool_parameter_description",
            "operation": "clarify argument",
            "experience_ids": ["exp-a"],
            "rationale": "The MCP parameter needs local guidance.",
        }
    ]
    candidate["manifest_delta"] = {
        "tool_desc_patches": {
            "modify_pending_order_items": {
                "params": {"item_ids": "All confirmed item IDs in positional order."}
            }
        }
    }
    candidate["validation"]["no_regression_experience_ids"] = []

    validate_analyzer_output(
        {
            "coverage": {
                "experience_ids": ["exp-a"],
                "public_environment_reviewed": True,
            },
            "candidates": [candidate],
        },
        side="reusable",
        experience_ids=("exp-a",),
        channel_ids={"mcp_tool_parameter_description"},
    )


def test_analyzer_canonicalizes_mcp_channel_shorthand_when_discovered():
    candidate = _candidate()
    candidate["channel_plan"] = [
        {
            "channel_id": "tool_parameter_description",
            "operation": "clarify SQL argument",
            "experience_ids": ["exp-a"],
            "rationale": "The parameter needs point-of-use guidance.",
        }
    ]
    candidate.pop("manifest_delta")
    output = {
        "coverage": {
            "experience_ids": ["exp-a"],
            "public_environment_reviewed": True,
        },
        "candidates": [candidate],
    }

    canonicalize_analyzer_output(
        output,
        harness_query={
            "harness": "opencode",
            "modifiable_modules": [],
            "mcp_editable_points": [
                {
                    "id": "mcp_tool_parameter_description",
                    "status": "modifiable",
                    "operation": {"kind": "tool_schema_patch"},
                }
            ],
        },
    )

    assert output["candidates"][0]["channel_plan"][0]["channel_id"] == (
        "mcp_tool_parameter_description"
    )


def test_analyzer_rejects_channel_label_that_is_not_materialized():
    candidate = _candidate()
    candidate["manifest_delta"] = {"instructions": ["This is not a skill."]}

    with pytest.raises(ValueError, match="does not match"):
        validate_analyzer_output(
            {
                "coverage": {
                    "experience_ids": ["exp-a", "exp-b"],
                    "public_environment_reviewed": True,
                },
                "candidates": [candidate],
            },
            side="reusable",
            experience_ids=("exp-a", "exp-b"),
            channel_ids={"skills"},
        )


def test_analyzer_canonicalizes_materialization_syntax_without_rewriting_content():
    output = {
        "coverage": {
            "experience_ids": ["exp-a", "exp-b"],
            "public_environment_reviewed": True,
        },
        "candidates": [_candidate()],
    }
    candidate = output["candidates"][0]
    candidate["channel_plan"][0]["channel_id"] = "skill"
    candidate["channel_plan"][0]["experience_ids"] = ["exp-a", "exp-b"]
    candidate["manifest_delta"]["config_patch"] = {
        "tools": {"skill": {"enabled": True}},
        "permission": {"skill": True},
    }
    candidate["manifest_delta"]["files"][0]["content"] = candidate["manifest_delta"][
        "files"
    ][0]["content"].replace("name: bounded-route", "name: Bounded Route")

    canonicalize_analyzer_output(output)
    validate_analyzer_output(
        output,
        side="reusable",
        experience_ids=("exp-a", "exp-b"),
        channel_ids={"skills"},
    )

    delta = output["candidates"][0]["manifest_delta"]
    assert delta["config_patch"] == {"tools.skill": True}
    assert output["candidates"][0]["channel_plan"][0]["channel_id"] == "skills"


def test_analyzer_canonicalizes_legacy_instruction_and_parameter_patch_shapes():
    candidate = _candidate()
    candidate["channel_plan"] = [
        {
            "channel_id": "skills",
            "operation": "add procedure",
            "experience_ids": ["exp-a"],
            "rationale": "Keep the detailed procedure.",
        }
    ]
    candidate["manifest_delta"].update(
        {
            "instructions_rules": {"diff": "Execute after explicit confirmation."},
            "tool_param_patches": {
                "modify_pending_order_items": {
                    "params": {"item_ids": "All confirmed item IDs."}
                }
            },
        }
    )
    output = {
        "coverage": {
            "experience_ids": ["exp-a"],
            "public_environment_reviewed": True,
        },
        "candidates": [candidate],
    }

    canonicalize_analyzer_output(output)

    delta = output["candidates"][0]["manifest_delta"]
    assert delta["instructions"] == ["Execute after explicit confirmation."]
    assert delta["tool_desc_patches"] == {
        "modify_pending_order_items": {
            "params": {"item_ids": "All confirmed item IDs."}
        }
    }
    assert {item["channel_id"] for item in output["candidates"][0]["channel_plan"]} == {
        "skills",
        "instructions_rules",
        "tool_parameter_description",
    }
    assert "name: bounded-route" in delta["files"][0]["content"]


def test_adjustment_canonicalizer_requests_diagnosis_and_materializes_instruction():
    output = {
        "coverage": {"experience_ids": ["exp-a"]},
        "problems": [
            {
                "id": "problem-a",
                "priority": 1,
                "summary": "Execution stalls after confirmation.",
                "experience_ids": ["exp-a"],
                "evidence_refs": ["ev-a"],
                "channel_hypotheses": [
                    {"channel_id": "instructions_rules", "reason": "global"}
                ],
                "modification_direction": "Execute after confirmation.",
                "diagnostic_rollout_needed": False,
                "local_success_criteria": ["The action tool is called."],
                "candidate_id": "candidate-a",
            }
        ],
        "candidates": [
                {
                    "id": "candidate-a",
                    "objective": "Execute confirmed plans.",
                    "observed_terminal_failure": "Execution stalls after confirmation.",
                    "causal_hypothesis": "The confirmation boundary is not actionable.",
                    "intervention_point": "instructions_rules",
                    "expected_runtime_event": "The action tool follows confirmation.",
                    "falsifying_observation": "The rule is visible but no action follows.",
                "channel_plan": [
                    {
                        "channel_id": "instructions_rules",
                        "operation": "append",
                        "experience_ids": ["exp-a"],
                        "rationale": "Universal confirmation boundary.",
                    }
                ],
                "manifest_delta": {
                    "instructions_rules": {"append": "Execute the confirmed plan."}
                },
                "validation": {"local_behavior_checks": ["Tool follows confirmation."]},
            }
        ],
    }

    canonicalize_analyzer_output(output)
    validate_analyzer_output(
        output,
        side="adjustment",
        experience_ids=("exp-a",),
        channel_ids={"instructions_rules"},
    )

    assert output["problems"][0]["diagnostic_rollout_needed"] is True
    assert output["candidates"][0]["manifest_delta"] == {
        "instructions": ["Execute the confirmed plan."]
    }


def test_adjustment_canonicalizer_drops_non_actionable_problem_notes():
    output = {
        "coverage": {"experience_ids": ["exp-a", "exp-b"]},
        "problems": [
            {
                "id": "problem-a",
                "priority": 1,
                "summary": "Execution stalls after confirmation.",
                "experience_ids": ["exp-a"],
                "evidence_refs": ["ev-a"],
                "channel_hypotheses": [
                    {"channel_id": "instructions_rules", "reason": "global"}
                ],
                "modification_direction": "Execute after confirmation.",
                "diagnostic_rollout_needed": True,
                "local_success_criteria": ["The action tool is called."],
                "candidate_id": "candidate-a",
            },
            {
                "id": "capability-gap",
                "priority": 2,
                "summary": "No available tool can add a payment method.",
                "experience_ids": ["exp-b"],
                "evidence_refs": ["ev-b"],
                "channel_hypotheses": [
                    {"channel_id": "instructions_rules", "reason": "uncertain"}
                ],
                "modification_direction": "No concrete harness change identified.",
                "diagnostic_rollout_needed": True,
                "local_success_criteria": [],
                "candidate_id": "",
            },
        ],
        "candidates": [
                {
                    "id": "candidate-a",
                    "objective": "Execute confirmed plans.",
                    "observed_terminal_failure": "Execution stalls after confirmation.",
                    "causal_hypothesis": "The confirmation boundary is not actionable.",
                    "intervention_point": "instructions_rules",
                    "expected_runtime_event": "The action tool follows confirmation.",
                    "falsifying_observation": "The rule is visible but no action follows.",
                "channel_plan": [
                    {
                        "channel_id": "instructions_rules",
                        "operation": "append",
                        "experience_ids": ["exp-a"],
                        "rationale": "Universal confirmation boundary.",
                    }
                ],
                "manifest_delta": {"instructions": ["Execute the confirmed plan."]},
                "validation": {"local_behavior_checks": ["Tool follows confirmation."]},
            }
        ],
    }

    canonicalize_analyzer_output(output)
    validate_analyzer_output(
        output,
        side="adjustment",
        experience_ids=("exp-a", "exp-b"),
        channel_ids={"instructions_rules"},
    )

    assert [problem["id"] for problem in output["problems"]] == ["problem-a"]


def test_canonicalizer_clamps_candidate_plan_experience_refs_to_coverage():
    output = {
        "coverage": {
            "experience_ids": ["partial-return"],
            "public_environment_reviewed": True,
        },
        "candidates": [
            {
                "id": "return-batch",
                "objective": "Preserve partial return procedure.",
                "channel_plan": [
                    {
                        "channel_id": "skill",
                        "operation": "add",
                        "experience_ids": [
                            "partial-return",
                            "partial-return-typo",
                        ],
                        "rationale": "Specific return procedure.",
                    }
                ],
                "manifest_delta": {
                    "files": [
                        {
                            "path": ".opencode/skills/partial-return/SKILL.md",
                            "content": (
                                "---\n"
                                "name: partial-return\n"
                                "description: Return a subset of items.\n"
                                "---\n"
                                "Confirm item IDs, then call return_delivered_order_items.\n"
                            ),
                        }
                    ]
                },
                "validation": {"local_behavior_checks": ["Skill remains specific."]},
            }
        ],
    }

    canonicalize_analyzer_output(output)
    validate_analyzer_output(
        output,
        side="reusable",
        experience_ids=("partial-return",),
        channel_ids={"skills"},
    )

    assert output["candidates"][0]["channel_plan"][0]["channel_id"] == "skills"
    assert output["candidates"][0]["channel_plan"][0]["experience_ids"] == [
        "partial-return"
    ]


def test_experience_dispositions_are_derived_from_candidate_coverage():
    dispositions = build_experience_dispositions(
        experience={
            "reusable": [{"id": "exp-a"}, {"id": "exp-b"}],
            "needs_adjustment": [{"id": "exp-c"}],
        },
        reusable={"candidates": [_candidate()]},
        adjustment={"candidates": []},
    )

    by_id = {item["experience_id"]: item for item in dispositions["items"]}
    assert by_id["exp-a"]["status"] == "materialized"
    assert by_id["exp-b"]["status"] == "deferred"
    assert by_id["exp-c"]["status"] == "deferred"


def test_post_reusable_only_counts_attributable_channel_regression():
    output = {
        "coverage": {
            "baseline_experience_ids": ["base"],
            "comparison_experience_ids": ["new"],
            "task_ids": ["0", "1"],
            "channel_usage_reviewed": True,
        },
        "preservation": {
            "attributable_regressions": [
                {
                    "task_id": "1",
                    "evidence_refs": ["ev1"],
                    "channel_ids": ["instructions_rules"],
                    "reason": "The new global rule caused the visible behavior change.",
                }
            ],
            "preserved_task_ids": ["0"],
            "candidate_recommendation": "reject",
            "rationale": "One attributable regression remains.",
        },
    }
    validate_post_analyzer_output(
        output,
        side="reusable",
        baseline_ids=("base",),
        comparison_ids=("new",),
        task_statuses={"0": "recovered", "1": "regressed"},
        changed_channels={"instructions_rules"},
    )


def test_post_reusable_allows_branch_regression_inside_stable_success():
    output = {
        "coverage": {
            "baseline_experience_ids": ["base"],
            "comparison_experience_ids": ["new"],
            "task_ids": ["91"],
            "channel_usage_reviewed": True,
        },
        "preservation": {
            "attributable_regressions": [
                {
                    "task_id": "91",
                    "evidence_refs": ["baseline", "candidate"],
                    "channel_ids": ["skills"],
                    "reason": "A consistently completed baseline branch became partial.",
                }
            ],
            "preserved_task_ids": [],
            "candidate_recommendation": "reject",
            "rationale": "The candidate lost one previously consistent branch.",
        },
    }

    validate_post_analyzer_output(
        output,
        side="reusable",
        baseline_ids={"base"},
        comparison_ids={"new"},
        task_statuses={"91": "stable_success"},
        changed_channels={"skills"},
    )


def test_post_adjustment_requires_every_task_and_exact_status():
    output = {
        "coverage": {
            "baseline_experience_ids": ["base"],
            "comparison_experience_ids": ["new"],
            "task_ids": ["0"],
            "channel_usage_reviewed": True,
        },
        "primary_problem": {
            "summary": "Confirmation behavior improved.",
            "task_assessments": [
                {
                    "task_id": "0",
                    "status": "recovered",
                    "relation": "attributed",
                    "evidence_refs": ["ev0"],
                    "reason": "The changed instruction matches the visible transition.",
                }
            ],
            "channel_attribution": {
                "relation": "attributed",
                "channel_ids": ["instructions_rules"],
                "reason": "Repeated set-level recovery follows the changed instruction.",
            },
            "local_recovery": "The agent now executes after confirmation.",
            "recommendation": "accept",
            "further_rollout_needed": False,
        },
    }
    validate_post_analyzer_output(
        output,
        side="adjustment",
        baseline_ids=("base",),
        comparison_ids=("new",),
        task_statuses={"0": "recovered"},
        changed_channels={"instructions_rules"},
    )


def test_post_adjustment_canonicalizes_task_statuses_from_comparison():
    output = {
        "coverage": {
            "baseline_experience_ids": ["wrong"],
            "comparison_experience_ids": ["wrong"],
            "task_ids": ["0"],
            "channel_usage_reviewed": False,
        },
        "primary_problem": {
            "summary": "Confirmation behavior changed.",
            "task_assessments": [
                {
                    "task_id": "0",
                    "status": "improved",
                    "relation": "possibly_related",
                    "evidence_refs": ["ev0"],
                    "reason": "The candidate changed the observed path.",
                }
            ],
            "channel_attribution": {
                "relation": "partially_attributed",
                "channel_ids": ["instructions_rules", "unrelated"],
                "reason": "Only the changed instruction can explain the shift.",
            },
            "local_recovery": "One branch improved.",
            "recommendation": "refine",
            "further_rollout_needed": True,
        },
    }

    normalized = canonicalize_post_analyzer_output(
        output,
        side="adjustment",
        baseline_ids=("base",),
        comparison_ids=("new",),
        task_statuses={"0": "mixed"},
        changed_channels={"instructions_rules"},
    )

    assert normalized["primary_problem"]["task_assessments"][0]["status"] == "mixed"
    assert normalized["primary_problem"]["channel_attribution"]["channel_ids"] == [
        "instructions_rules"
    ]
    validate_post_analyzer_output(
        normalized,
        side="adjustment",
        baseline_ids=("base",),
        comparison_ids=("new",),
        task_statuses={"0": "mixed"},
        changed_channels={"instructions_rules"},
    )


def test_post_adjustment_fallback_preserves_exact_measured_statuses():
    fallback = conservative_post_analyzer_fallback(
        side="adjustment",
        baseline_ids=("base",),
        comparison_ids=("comparison",),
        task_statuses={"0": "mixed"},
        changed_channels=("instructions_rules",),
        task_outcomes={"0": {"candidate_pass_count": 0}},
        task_comparisons=[
            {"task_id": "0", "baseline_refs": ["base-ref"], "candidate_refs": ["candidate-ref"]}
        ],
        failure="malformed_output",
    )

    validate_post_analyzer_output(
        fallback,
        side="adjustment",
        baseline_ids=("base",),
        comparison_ids=("comparison",),
        task_statuses={"0": "mixed"},
        changed_channels={"instructions_rules"},
        task_outcomes={"0": {"candidate_pass_count": 0}},
    )
    assessment = fallback["primary_problem"]["task_assessments"][0]
    assert assessment["relation"] == "unresolved"
    assert assessment["evidence_refs"] == ["candidate-ref", "base-ref"]


def test_post_candidate_context_preserves_main_validation_objective():
    context = _post_candidate_context(
        {
            "harness_version": "candidate-01",
            "selected_candidate_side": "reusable",
            "candidate": {
                "parent_version": "accepted-v1",
                "source_candidate_ids": ["user-identification-fallback"],
                "channel_diffs": [{"channel_id": "instructions_rules"}],
                "manifest_delta": {
                    "tool_desc_patches": {
                        "lookup": {"desc": "Use the bounded fallback."}
                    }
                },
                "workspace_delta": {
                    "files": [
                        {
                            "scope": "project",
                            "path": "AGENTS.md",
                            "change": "added",
                            "content": "Use the bounded fallback.\n",
                            "executable": False,
                        }
                    ]
                },
                "workspace_diff": [{"scope": "project", "path": "AGENTS.md"}],
                "editor_summary": {"rationale": "Keep the rule visible."},
            },
            "rollout_request": {
                "task_ids": ["1", "2"],
                "local_success_criteria": ["Preserve name/zip lookup."],
                "rationale": "Validate a reusable lookup route without regression.",
            },
        }
    )

    assert context["selected_candidate_side"] == "reusable"
    assert context["parent_version"] == "accepted-v1"
    assert context["source_candidate_ids"] == ["user-identification-fallback"]
    assert context["manifest_delta"]["tool_desc_patches"]["lookup"]["desc"] == (
        "Use the bounded fallback."
    )
    assert context["workspace_delta"]["files"][0]["content"] == (
        "Use the bounded fallback.\n"
    )
    assert context["editor_summary"] == {"rationale": "Keep the rule visible."}
    assert context["rollout_request"] == {
        "task_ids": ["1", "2"],
        "local_success_criteria": ["Preserve name/zip lookup."],
        "rationale": "Validate a reusable lookup route without regression.",
    }


def test_rollout_channel_usage_keeps_only_compact_runtime_signals(tmp_path):
    trajectory = tmp_path / "trial.jsonl"
    trajectory.write_text(
        json.dumps(
            {
                "trial": 0,
                "model_context": {
                    "system_prompt": "large prompt is not forwarded",
                    "skills_available": [{"name": "route", "description": "bounded"}],
                    "skills_invoked": [{"name": "route", "n_calls": 1}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rollout = tmp_path / "rollout.json"
    rollout.write_text(
        json.dumps({"per_task": {"7": {"trajectory_paths": [str(trajectory)]}}}),
        encoding="utf-8",
    )

    decision = {
        "candidate": {
            "manifest_delta": {},
            "workspace_delta": {
                "files": [
                    {
                        "scope": "project",
                        "path": ".opencode/skills/route/SKILL.md",
                        "change": "added",
                        "content": "body",
                        "executable": False,
                    }
                ]
            },
        }
    }
    assert _rollout_channel_usage(rollout, decision) == [
        {
            "task_id": "7",
            "trial": 0,
            "on_demand": [
                {
                    "channel": "skills",
                    "artifact": "route",
                    "available": True,
                    "invocations": 1,
                }
            ],
        }
    ]


def test_candidate_must_satisfy_harness_query_operation_contract():
    candidate = {
        "id": "grounded-instruction",
        "objective": "Compile one bounded rule.",
        "channel_plan": [
            {
                "channel_id": "instructions_rules",
                "operation": "add one bounded instruction",
                "experience_ids": ["exp-a"],
                "rationale": "Directly represents the evidence.",
            }
        ],
        "manifest_delta": {"instructions": ["Use the bounded route."]},
        "validation": {"local_behavior_checks": ["Uses the bounded route."]},
    }
    contracts = {
        "instructions_rules": {
            "id": "instructions_rules",
            "status": "verified",
            "operation": {
                "kind": "prompt_content",
                "manifest_field": "instructions",
            },
        }
    }

    _validate_candidate(
        candidate,
        experience_ids={"exp-a"},
        channel_ids={"instructions_rules"},
        channel_contracts=contracts,
    )

    contracts["instructions_rules"]["operation"] = {
        "kind": "project_file",
        "path_pattern": "AGENTS.md",
    }
    with pytest.raises(ValueError, match="operation contract"):
        _validate_candidate(
            candidate,
            experience_ids={"exp-a"},
            channel_ids={"instructions_rules"},
            channel_contracts=contracts,
        )


@pytest.mark.parametrize(
    ("harness", "channel_id", "delta", "operation"),
    [
        (
            "pi",
            "project_instructions",
            {"files": [{"path": "AGENTS.md", "content": "Use the bounded route."}]},
            {"kind": "project_file", "path_pattern": "AGENTS.md"},
        ),
        (
            "codex",
            "developer_instructions",
            {
                "files": [
                    {
                        "path": ".codex/config.toml",
                        "content": 'developer_instructions = "Use the bounded route."\n',
                    }
                ]
            },
            {
                "kind": "workspace_config",
                "scope": "project",
                "path": ".codex/config.toml",
                "key": "developer_instructions",
            },
        ),
    ],
)
def test_non_opencode_candidate_uses_its_own_query_contract(
    harness, channel_id, delta, operation
):
    candidate = {
        "id": f"{harness}-candidate",
        "objective": "Compile one bounded rule.",
        "channel_plan": [
            {
                "channel_id": channel_id,
                "operation": "materialize the discovered channel",
                "experience_ids": ["exp-a"],
                "rationale": "Uses the harness-native channel.",
            }
        ],
        "manifest_delta": delta,
        "validation": {"local_behavior_checks": ["Uses the bounded route."]},
    }

    _validate_candidate(
        candidate,
        experience_ids={"exp-a"},
        channel_ids={channel_id},
        channel_contracts={
            channel_id: {
                "id": channel_id,
                "status": "verified",
                "operation": operation,
            }
        },
        harness=harness,
    )
