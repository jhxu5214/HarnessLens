from __future__ import annotations

import copy
import json
import re
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from harnesslens.infrastructure.analysis_concurrency import analysis_workers
from harnesslens.core.artifacts import write_json
from harnesslens.core.budget import CreationBudget
from harnesslens.evolution.experience import compact_channel_usage
from harnesslens.harnesses.opencode_harness import OpencodeHarnessAdapter
from harnesslens.core.profiles import power_profile
from harnesslens.harnesses.runner import (
    IntelligentHarnessRunner,
    intelligent_stdout_path,
    parse_json_object,
)


MCP_CHANNEL_BASE = {
    "mcp_tool_description": "tool_description",
    "mcp_tool_parameter_description": "tool_parameter_description",
}
CHANNEL_ID_ALIASES = {
    "instruction": "instructions_rules",
    "instructions": "instructions_rules",
    "skill": "skills",
}
MAX_REUSABLE_COMPILATION_GROUPS = 4
MAX_ADJUSTMENT_ANALYSIS_GROUPS = 4
TARGET_ADJUSTMENT_EXPERIENCES_PER_GROUP = 4
POST_ANALYZER_OUTPUT_LIMIT = 49_152


REUSABLE_PLANNER_SYSTEM = """
You are the reusable compilation planner. Read every reusable experience and the frozen task
categories. Partition the experience IDs into at most four nonempty groups. Group experiences only
when their harness artifacts can be exercised by a shared bounded rollout task set. This is output
planning, not semantic merging: every source passage remains unchanged and every ID appears exactly
once. Keep materially different validation risks in different groups.

Return exactly:
{"groups":[{"id":"short-lowercase-id","experience_ids":["..."]}]}
""".strip()


REUSABLE_ANALYZER_SYSTEM = """
You are the reusable-experience Analyzer. Analyze every supplied reusable experience, the
frozen task categories, public environment policy/tools, and every editable harness/MCP surface. Preserve different successful
branches; task categories organize review but never justify merging behavior.

Return a ranked portfolio of materializable candidates. Each candidate has one atomic behavior
hypothesis with one observable trigger and outcome. It may use complementary channels or files only
when they jointly implement that same hypothesis; keep independent procedures in separate
candidates even when one bounded rollout task set could exercise them together.
Compile the smallest causal invariant shared by the supporting successful branches. Preserve
necessary decisions and checks, but omit incidental syntax, call order, or formatting that another
supporting success does not require; do not turn one complete trajectory into a universal recipe.

Decide channel granularity from the harness-query channel semantics: visibility timing, scope,
constraint strength, trigger mechanism, and regression risk. Route the experience to the narrowest
channel that can actually carry it. Ordered procedures with branches, parameters, checks, or
recovery logic often belong in on-demand skills, but do not default to skills: tool-selection hints
belong near the relevant tool, parameter-choice constraints belong near the parameter, concise
universal policy belongs in instructions, runtime enablement belongs in config, and skill
reachability problems may require trigger wording or a small earlier-visible artifact. In each
channel_plan rationale, state why this channel is the right carrier and why a skill-only artifact
would be insufficient or unnecessarily risky when another channel is used. Every reusable
experience must be incorporated by at least one candidate. A candidate is an evidence-backed
editor brief, not a file patch: identify the behavior, likely harness mechanisms, evidence, and
observable checks. A separate Harness Editor will inspect the actual isolated harness workspace
and choose the concrete native files or configuration. Treat Harness Query as an architectural map,
not a whitelist, and do not write `manifest_delta`.
Use one channel_plan entry per changed channel and explain that channel's responsibility. A
candidate may contain multiple artifacts only when they cooperate on its single behavior
hypothesis. Rank candidates by
expected reusable coverage, frequency, and regression risk. Do not force channel diversity without
evidence, but do not discard an experience merely because it is infrequent.
Every normative claim in candidate content must agree with the public policy. Tool acceptance or
a successful trajectory does not prove a policy restriction. In particular, do not add payment
method restrictions unless the policy states them for that exact operation.

Return exactly:
{"coverage":{"experience_ids":["all supplied IDs"],"public_environment_reviewed":true},"candidates":[{"id":"...","priority":1,"objective":"...","channel_plan":[{"channel_id":"exact discovered channel","operation":"...","experience_ids":["..."],"rationale":"..."}],"validation":{"local_behavior_checks":["..."]}}]}
""".strip()


ADJUSTMENT_ANALYZER_SYSTEM = """
You are the needs-adjustment Analyzer. Analyze every supplied needs_adjustment experience and
all editable harness/MCP surfaces. Return a ranked list of the main modifiable problems,
prioritizing repeated cross-task patterns with observable triggers and local behavior checks. A shared
surface error is only a channel hypothesis until rollout evidence supports attribution.

For each problem include: (1) the problem and supporting experience/evidence, (2) possibly related
channels and why, and (3) a modification direction or the exact uncertainty needing diagnosis.
Every initial problem must set diagnostic_rollout_needed to true because channel attribution has
not yet been tested; a candidate may still express the bounded hypothesis to validate.
Do not select rollout task IDs or categories; the Main Agent owns task selection. Each emitted
candidate must belong to exactly one problem and represent one behavior hypothesis with an
observable trigger and outcome. Multiple channels may cooperate only on that same hypothesis;
independent problem fixes must remain separate candidates. Omit a candidate when diagnosis must
precede editing. Validation must require local behavior recovery, not reward alone. Candidate
content should express the smallest causal invariant supported by the failure/success contrast,
without copying incidental syntax or an entire trajectory as the fix.
priority is global across this side. Consider instructions, skills, system prompt, tool descriptions,
and parameter descriptions according to their visibility, scope, trigger mechanism, and constraint
strength; do not default every problem to instructions or skills. Route the fix to the narrowest
channel that can change the observed behavior. If a skill was available but not invoked, consider
whether trigger wording, skill description/name, a concise earlier-visible instruction, or a
tool/parameter-local artifact is the better next channel; do not merely lengthen the skill body.
In each candidate channel_plan rationale, explain why the chosen channel can carry this fix and why
skill-only handling would be insufficient or unnecessarily risky when another channel is used.
Only emit a candidate when the evidence supports a concrete behavior hypothesis for the Harness
Editor to attempt. Describe the likely mechanism and observable recovery, but do not write harness
files or `manifest_delta`; the Editor owns concrete native changes after Main selects the problem.
Every candidate must state the observed terminal failure, causal hypothesis, intervention point,
expected runtime event, and a falsifying observation. These fields must describe observable behavior,
not merely restate reward or promise that an instruction will be followed.
If diagnosis must precede any edit, return the problem with an empty `candidate_id` and omit its
candidate. Treat Harness Query as an architectural map rather than a whitelist.

Return exactly:
{"coverage":{"experience_ids":["all supplied IDs"]},"problems":[{"id":"...","priority":1,"summary":"...","experience_ids":["..."],"evidence_refs":["..."],"channel_hypotheses":[{"channel_id":"exact discovered channel or MCP point","reason":"..."}],"modification_direction":"...","diagnostic_rollout_needed":true,"local_success_criteria":["..."],"candidate_id":"candidate id or empty"}],"candidates":[{"id":"...","priority":1,"objective":"...","observed_terminal_failure":"...","causal_hypothesis":"...","intervention_point":"...","expected_runtime_event":"...","falsifying_observation":"...","channel_plan":[{"channel_id":"...","operation":"...","experience_ids":["..."],"rationale":"..."}],"validation":{"local_behavior_checks":["..."]}}]}
Never include rollout_task_ids.
""".strip()


POST_REUSABLE_ANALYZER_SYSTEM = """
You are the post-rollout reusable-experience Analyzer. Analyze every baseline reusable
experience, every new reusable comparison passage, and every task comparison. The input also
contains the exact candidate channel diff and compact per-trial channel usage. Determine whether
successful behavior was preserved.
For cumulative iterations, `reference_versions` identifies whether each task comparison uses the
direct parent or frozen v0. Attribute the current delta against the direct parent when available;
when only v0 is available, do not claim the current delta caused behavior already present in an
earlier accepted version.

A lower reward or different action is a regression only when the visible behavior change can be
attributed to one or more changed channels. Randomness, unchanged behavior, or an ungrounded
relationship is not a regression. A task-level stable_success or mixed label may still contain an
attributable regressed branch when the baseline consistently completed and the candidate became
partial, but
failure to obey a new instruction does not by itself mean the instruction caused the failure.
An infrastructure-error trial with no messages or tool interactions contains no behavioral
mechanism: report it as infrastructure uncertainty and never infer an action from it. Use metric
names and values exactly as supplied; do not relabel trial success rate as pass@k.
For an on-demand channel such as a skill, distinguish availability from actual invocation; do not
accept it as validated when its target trials never load it. Unknown observability stays unknown.
For startup-visible instructions/prompts and tool-schema-visible descriptions, an empty invocation
counter does not mean "not invoked"; assess whether the artifact was materialized and exposed, then
use behavior for effect. Reserve invocation gating for genuinely on-demand channels.
When an artifact is available but not invoked, state whether the next candidate should pivot to
trigger wording, skill metadata, a concise earlier-visible instruction, or a tool/parameter-local
artifact instead of only editing the artifact body.
Separate attribution safety from promotion readiness. `preserved_task_ids` means no attributable
changed-channel regression was found; it does not mean the task succeeded or that the candidate is
deployment-ready. Use the supplied rollout_metrics and task comparisons. In the rationale,
explicitly state whether the rollout shows successful behavior, shared baseline/candidate failures,
candidate-only failures, or low success metrics. Do not
recommend accept merely because attributable_regressions is empty; recommend uncertain or reject
when the supplied task comparisons and rollout evidence are too weak to justify promoting this
candidate as a parent version.
Do not propose a new candidate. Return exactly:
{"coverage":{"baseline_experience_ids":["..."],"comparison_experience_ids":["..."],"task_ids":["..."],"channel_usage_reviewed":true},"preservation":{"attributable_regressions":[{"task_id":"...","evidence_refs":["..."],"channel_ids":["..."],"reason":"..."}],"preserved_task_ids":["..."],"candidate_recommendation":"accept|reject|uncertain","rationale":"..."}}
""".strip()


POST_ADJUSTMENT_ANALYZER_SYSTEM = """
You are the post-rollout needs-adjustment Analyzer. Analyze every baseline needs_adjustment
experience, every new needs_adjustment comparison passage, and every task comparison. Compare
the candidate against baseline as controller-sized trial sets and use the exact candidate channel diff to
assess attribution. Use compact per-trial channel usage to distinguish an available on-demand
channel from one that was actually invoked. Do not select task IDs for another rollout.
For startup-visible instructions/prompts and tool-schema-visible descriptions, an empty invocation
counter does not mean "not invoked"; assess materialization/exposure and behavior. Reserve
invocation gating for genuinely on-demand channels such as skill bodies.
Use `reference_versions` and candidate.parent_version to distinguish this delta from behavior
already introduced by an accepted parent. A v0 fallback can diagnose end-to-end behavior but does
not by itself prove which cumulative delta caused the change.

For every rolled-out task, retain its comparison status and state whether the changed channel is
attributed, possibly_related, not_attributed, unresolved, or random. Reward alone never proves
attribution. Summarize whether the original main problem locally recovered, including remaining
failure behavior.
An infrastructure-error trial with no messages or tool interactions contains no behavioral
mechanism: report it as infrastructure uncertainty and never infer an action from it. Use metric
names and values exactly as supplied; do not relabel trial success rate as pass@k.

Do not credit a changed channel merely because a candidate trajectory follows the new instruction.
If baseline and candidate already show the same successful behavior, that behavior is preserved,
not improved or enabled by the change, unless a new visible difference supports attribution.
Use candidate.selected_candidate_side and candidate.rollout_request as the authoritative validation
objective. Promotion requires an attributable positive effect on the target behavior, including
making it consistently succeed when the reference had only a mixed branch. Merely exposing a
reusable procedure that the reference already performed is preservation, not a promotion benefit.
The candidate is not required to repair unrelated baseline needs_adjustment patterns. Unrelated
failures remain useful observations but must not by themselves cause refine or reject. Base
recommendation on the candidate's target behavior, attributable changes, and attributable
regressions. Distinguish preserved behavior from recovered behavior in local_recovery.
Separate local attribution from promotion readiness. A recovered task or absence of attributable
regression does not by itself mean the candidate is deployment-ready. Use the supplied
rollout_metrics and task comparisons. In summary/local_recovery, explicitly report whether the
rolled-out tasks mostly succeeded, mostly failed together with the reference, or introduced
candidate-only failures. If the candidate only repairs a small local issue while the rollout
evidence remains broadly weak, recommend refine or reject unless there is a clear reason that the
weak outcomes are outside the candidate's validation objective.
When the target artifact was available but not invoked, state whether a future revision should
pivot to trigger wording, skill metadata, a concise earlier-visible instruction, or a tool/parameter-
local artifact instead of only editing the artifact body.
If this rollout falsifies the tested hypothesis but exposes a different concrete blocker, emit one
`replan_candidate` grounded in baseline experience IDs. It must use a discovered editable channel
and the same falsifiable causal fields as an initial adjustment candidate. This is a new independent
hypothesis, not a rewrite of the rejected delta. Use null when the evidence only says that the
tested hypothesis failed or remains opaque.

Return exactly:
{"coverage":{"baseline_experience_ids":["..."],"comparison_experience_ids":["..."],"task_ids":["..."],"channel_usage_reviewed":true},"primary_problem":{"summary":"...","task_assessments":[{"task_id":"...","status":"...","relation":"attributed|possibly_related|not_attributed|unresolved|random","evidence_refs":["..."],"reason":"..."}],"channel_attribution":{"relation":"attributed|partially_attributed|not_attributed|unresolved|random","channel_ids":["..."],"reason":"..."},"local_recovery":"...","recommendation":"accept|reject|refine","further_rollout_needed":true},"replan_candidate":null}
""".strip()


@dataclass(frozen=True)
class AnalyzerResult:
    reusable: Mapping[str, Any]
    adjustment: Mapping[str, Any]
    reusable_path: str
    adjustment_path: str


class AnalyzerModule:
    def __init__(
        self,
        *,
        run_root: str | Path,
        budget: CreationBudget,
        harness: str = "opencode",
    ) -> None:
        self.run_root = Path(run_root).resolve()
        self.budget = budget
        self.harness = harness
        self.root = self.run_root / "analyzer"
        self.root.mkdir(parents=True, exist_ok=True)

    def run(self, *, label: str = "baseline") -> AnalyzerResult:
        reusable_path = self.root / f"{label}_reusable.json"
        adjustment_path = self.root / f"{label}_adjustment.json"
        experience = json.loads(
            (self.run_root / "experience" / "current.json").read_text(encoding="utf-8")
        )
        discovery = {
            "task_categories": json.loads(
                (self.run_root / "discovery" / "task_explorer.json").read_text(
                    encoding="utf-8"
                )
            ),
            "harness_query": json.loads(
                (self.run_root / "discovery" / "harness_query.json").read_text(
                    encoding="utf-8"
                )
            ),
            "public_environment": json.loads(
                (self.run_root / "discovery" / "task_explorer_input.json").read_text(
                    encoding="utf-8"
                )
            )["environment"],
        }
        with ThreadPoolExecutor(max_workers=analysis_workers(2)) as executor:
            reusable_future = executor.submit(
                self._run_reusable_partitioned,
                label=label,
                experiences=experience["reusable"],
                discovery=discovery,
            )
            adjustment_future = executor.submit(
                self._run_adjustment_partitioned,
                label=label,
                experiences=experience["needs_adjustment"],
                discovery=discovery,
            )
            reusable = reusable_future.result()
            adjustment = adjustment_future.result()
        dispositions = build_experience_dispositions(
            experience=experience,
            reusable=reusable,
            adjustment=adjustment,
        )
        write_json(self.root / f"{label}_experience_dispositions.json", dispositions)
        return AnalyzerResult(
            reusable=reusable,
            adjustment=adjustment,
            reusable_path=str(reusable_path),
            adjustment_path=str(adjustment_path),
        )

    def _run_reusable_partitioned(
        self,
        *,
        label: str,
        experiences: Sequence[Mapping[str, Any]],
        discovery: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        output_path = self.root / f"{label}_reusable.json"
        experience_ids = tuple(str(item["id"]) for item in experiences)
        channel_contracts = _query_channel_contracts(discovery["harness_query"])
        channel_ids = set(channel_contracts)
        if output_path.exists():
            cached = json.loads(output_path.read_text(encoding="utf-8"))
            try:
                validate_analyzer_output(
                    cached,
                    side="reusable",
                    experience_ids=experience_ids,
                    channel_ids=channel_ids,
                    channel_contracts=channel_contracts,
                    harness=self.harness,
                )
                plan_groups = (cached.get("compilation_plan") or {}).get("groups") or []
                expected_group_ids = {
                    str(group.get("id") or "")
                    for group in plan_groups
                    if isinstance(group, Mapping)
                }
                actual_group_ids = {
                    str(item.get("compilation_group_id") or "")
                    for item in cached.get("candidates") or []
                    if isinstance(item, Mapping)
                }
                if not expected_group_ids or actual_group_ids != expected_group_ids:
                    raise ValueError(
                        "cached reusable output predates atomic portfolio compilation"
                    )
            except ValueError:
                pass
            else:
                return cached
        plan = self._run_reusable_plan(
            label=label,
            experiences=experiences,
            task_categories=discovery["task_categories"],
        )
        by_id = {str(item["id"]): item for item in experiences}
        groups = list(plan["groups"])
        with ThreadPoolExecutor(
            max_workers=analysis_workers(len(groups))
        ) as executor:
            futures = [
                (
                    str(group["id"]),
                    executor.submit(
                        self._run_side,
                        label=f"{label}-{group['id']}",
                        side="reusable",
                        experiences=[
                            by_id[str(item)] for item in group["experience_ids"]
                        ],
                        discovery=discovery,
                    ),
                )
                for group in groups
            ]
            outputs = [(group_id, future.result()) for group_id, future in futures]
        combined = combine_reusable_outputs(plan=plan, outputs=outputs)
        validate_analyzer_output(
            combined,
            side="reusable",
            experience_ids=experience_ids,
            channel_ids=channel_ids,
            channel_contracts=channel_contracts,
            harness=self.harness,
        )
        write_json(output_path, combined)
        return combined

    def _run_adjustment_partitioned(
        self,
        *,
        label: str,
        experiences: Sequence[Mapping[str, Any]],
        discovery: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        output_path = self.root / f"{label}_adjustment.json"
        experience_ids = tuple(str(item["id"]) for item in experiences)
        channel_contracts = _query_channel_contracts(discovery["harness_query"])
        channel_ids = set(channel_contracts)
        if output_path.exists():
            cached = json.loads(output_path.read_text(encoding="utf-8"))
            try:
                validate_analyzer_output(
                    cached,
                    side="adjustment",
                    experience_ids=experience_ids,
                    channel_ids=channel_ids,
                    channel_contracts=channel_contracts,
                    harness=self.harness,
                )
            except ValueError:
                pass
            else:
                return cached
        groups = partition_adjustment_experiences(experiences)
        with ThreadPoolExecutor(
            max_workers=analysis_workers(len(groups))
        ) as executor:
            futures = [
                (
                    group_id,
                    executor.submit(
                        self._run_side,
                        label=f"{label}-{group_id}",
                        side="adjustment",
                        experiences=group_experiences,
                        discovery=discovery,
                    ),
                )
                for group_id, group_experiences in groups
            ]
            outputs = [(group_id, future.result()) for group_id, future in futures]
        combined = combine_adjustment_outputs(
            experience_ids=experience_ids,
            outputs=outputs,
        )
        validate_analyzer_output(
            combined,
            side="adjustment",
            experience_ids=experience_ids,
            channel_ids=channel_ids,
            channel_contracts=channel_contracts,
            harness=self.harness,
        )
        write_json(output_path, combined)
        return combined

    def _run_reusable_plan(
        self,
        *,
        label: str,
        experiences: Sequence[Mapping[str, Any]],
        task_categories: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        output_path = self.root / f"{label}_reusable_plan.json"
        experience_ids = tuple(str(item["id"]) for item in experiences)
        if output_path.exists():
            cached = json.loads(output_path.read_text(encoding="utf-8"))
            validate_reusable_plan(cached, experience_ids=experience_ids)
            return cached
        runner = IntelligentHarnessRunner(
            profile=power_profile(self.harness, max_steps=60),
            budget=self.budget,
            workspace_root=self.run_root / "intelligent_jobs",
            timeout_s=3600,
        )
        result = None
        for attempt in range(3):
            previous_error = result.validation_error if result is not None else ""
            result = runner.run_json(
                job_id=self.budget.next_attempt_id(f"analyzer-reusable-plan-{label}"),
                system_prompt=REUSABLE_PLANNER_SYSTEM,
                input_payload={
                    "experiences": list(experiences),
                    "task_categories": task_categories,
                    "maximum_groups": MAX_REUSABLE_COMPILATION_GROUPS,
                    "retry_context": (
                        analyzer_retry_context(previous_error) if attempt else ""
                    ),
                },
                validator=lambda output: validate_reusable_plan(
                    output, experience_ids=experience_ids
                ),
            )
            if result.output is not None:
                write_json(output_path, result.output)
                return result.output
        assert result is not None
        raise RuntimeError(
            f"Reusable Analyzer plan failed: {result.outcome}: {result.validation_error}"
        )

    def run_post_rollout(
        self,
        *,
        comparison_label: str,
        main_decision: str | Path,
        rollout_output: str | Path,
        label: str,
    ) -> AnalyzerResult:
        reusable_path = self.root / f"{label}_reusable.json"
        adjustment_path = self.root / f"{label}_adjustment.json"
        baseline = json.loads(
            (self.run_root / "experience" / "current.json").read_text(encoding="utf-8")
        )
        comparison = json.loads(
            (self.run_root / "experience" / f"{comparison_label}.json").read_text(
                encoding="utf-8"
            )
        )
        decision = json.loads(Path(main_decision).read_text(encoding="utf-8"))
        changed_channels = tuple(
            str(item["channel_id"]) for item in decision["candidate"]["channel_diffs"]
        )
        rollout_payload = json.loads(Path(rollout_output).read_text(encoding="utf-8"))
        harness_query = json.loads(
            (self.run_root / "discovery" / "harness_query.json").read_text(
                encoding="utf-8"
            )
        )
        common = {
            "task_comparisons": comparison["task_comparisons"],
            "reference_versions": dict(
                (comparison.get("coverage") or {}).get("reference_versions") or {}
            ),
            "candidate": _post_candidate_context(decision),
            "rollout_metrics": dict(rollout_payload.get("metrics") or {}),
            "channel_usage": _rollout_channel_usage(rollout_output, decision),
            "available_channel_ids": sorted(_query_channel_contracts(harness_query)),
        }
        with ThreadPoolExecutor(max_workers=analysis_workers(2)) as executor:
            reusable_future = executor.submit(
                self._run_post_side,
                label=label,
                side="reusable",
                baseline_experiences=baseline["reusable"],
                comparison_experiences=comparison["reusable"],
                common=common,
                changed_channels=changed_channels,
            )
            adjustment_future = executor.submit(
                self._run_post_side,
                label=label,
                side="adjustment",
                baseline_experiences=baseline["needs_adjustment"],
                comparison_experiences=comparison["needs_adjustment"],
                common=common,
                changed_channels=changed_channels,
            )
            reusable = reusable_future.result()
            adjustment = adjustment_future.result()
        return AnalyzerResult(
            reusable=reusable,
            adjustment=adjustment,
            reusable_path=str(reusable_path),
            adjustment_path=str(adjustment_path),
        )

    def _run_post_side(
        self,
        *,
        label: str,
        side: str,
        baseline_experiences: Sequence[Mapping[str, Any]],
        comparison_experiences: Sequence[Mapping[str, Any]],
        common: Mapping[str, Any],
        changed_channels: Sequence[str],
    ) -> Mapping[str, Any]:
        baseline_ids = tuple(str(item["id"]) for item in baseline_experiences)
        comparison_ids = tuple(str(item["id"]) for item in comparison_experiences)
        task_statuses = {
            str(item["task_id"]): str(item["status"])
            for item in common["task_comparisons"]
        }
        task_outcomes = {
            str(item["task_id"]): dict(item.get("outcome_summary") or {})
            for item in common["task_comparisons"]
        }
        expected_coverage = {
            "baseline_experience_ids": list(baseline_ids),
            "comparison_experience_ids": list(comparison_ids),
            "task_ids": list(task_statuses),
            "channel_usage_reviewed": True,
        }
        expected_task_statuses = dict(task_statuses)
        changed_channel_ids = tuple(str(item) for item in changed_channels)
        available_channel_ids = {
            str(item) for item in common.get("available_channel_ids") or []
        }
        output_path = self.root / f"{label}_{side}.json"
        if output_path.exists():
            cached = json.loads(output_path.read_text(encoding="utf-8"))
            try:
                cached = canonicalize_post_analyzer_output(
                    cached,
                    side=side,
                    baseline_ids=baseline_ids,
                    comparison_ids=comparison_ids,
                    task_statuses=task_statuses,
                    changed_channels=set(changed_channels),
                    task_outcomes=task_outcomes,
                    available_channel_ids=available_channel_ids,
                )
                validate_post_analyzer_output(
                    cached,
                    side=side,
                    baseline_ids=baseline_ids,
                    comparison_ids=comparison_ids,
                    task_statuses=task_statuses,
                    changed_channels=set(changed_channels),
                    task_outcomes=task_outcomes,
                    available_channel_ids=available_channel_ids,
                )
            except ValueError:
                pass
            else:
                write_json(output_path, cached)
                return cached
        base_job_id = f"analyzer-post-{side}-{label}"
        for workspace in sorted(
            (self.run_root / "intelligent_jobs").glob(f"{base_job_id}*"),
            reverse=True,
        ):
            stdout_path = intelligent_stdout_path(workspace, self.harness)
            if not stdout_path.is_file():
                continue
            try:
                recovered = dict(
                    parse_json_object(
                        stdout_path.read_text(encoding="utf-8", errors="replace")
                    )
                )
                recovered = canonicalize_post_analyzer_output(
                    recovered,
                    side=side,
                    baseline_ids=baseline_ids,
                    comparison_ids=comparison_ids,
                    task_statuses=task_statuses,
                    changed_channels=set(changed_channels),
                    task_outcomes=task_outcomes,
                    available_channel_ids=available_channel_ids,
                )
                validate_post_analyzer_output(
                    recovered,
                    side=side,
                    baseline_ids=baseline_ids,
                    comparison_ids=comparison_ids,
                    task_statuses=task_statuses,
                    changed_channels=set(changed_channels),
                    task_outcomes=task_outcomes,
                    available_channel_ids=available_channel_ids,
                )
            except (ValueError, KeyError, TypeError):
                continue
            write_json(output_path, recovered)
            return recovered
        # Post-rollout analysis receives all ten task comparisons and can exceed
        # the normal response budget before emitting its required JSON object.
        runner = IntelligentHarnessRunner(
            profile=replace(
                power_profile(self.harness, max_steps=60),
                output_limit=POST_ANALYZER_OUTPUT_LIMIT,
            ),
            budget=self.budget,
            workspace_root=self.run_root / "intelligent_jobs",
            timeout_s=3600,
        )
        system = (
            POST_REUSABLE_ANALYZER_SYSTEM
            if side == "reusable"
            else POST_ADJUSTMENT_ANALYZER_SYSTEM
        )
        result = None
        for attempt in range(3):
            result = runner.run_json(
                job_id=self.budget.next_attempt_id(f"analyzer-post-{side}-{label}"),
                system_prompt=system,
                input_payload={
                    "side": side,
                    "baseline_experiences": list(baseline_experiences),
                    "comparison_experiences": list(comparison_experiences),
                    "expected_coverage": expected_coverage,
                    "expected_task_statuses": expected_task_statuses,
                    "expected_task_outcomes": task_outcomes,
                    "changed_channel_ids": list(changed_channel_ids),
                    **dict(common),
                    "retry_context": (
                        "Previous output failed validation. Copy expected_coverage exactly into "
                        "coverage. For task_assessments, copy each status exactly from "
                        "expected_task_statuses; do not rename statuses to improved, preserved, "
                        "recovered, or failed. channel_attribution.channel_ids may only contain "
                        "IDs from changed_channel_ids and must name at least one changed channel "
                        "unless the relation is random. Review channel usage and distinguish "
                        "availability from invocation."
                        if attempt
                        else ""
                    ),
                },
                validator=lambda output: validate_post_analyzer_output(
                    canonicalize_post_analyzer_output(
                        output,
                        side=side,
                        baseline_ids=baseline_ids,
                        comparison_ids=comparison_ids,
                        task_statuses=task_statuses,
                        changed_channels=set(changed_channels),
                        task_outcomes=task_outcomes,
                        available_channel_ids=available_channel_ids,
                    ),
                    side=side,
                    baseline_ids=baseline_ids,
                    comparison_ids=comparison_ids,
                    task_statuses=task_statuses,
                    changed_channels=set(changed_channels),
                    task_outcomes=task_outcomes,
                ),
            )
            if result.output is not None:
                output = canonicalize_post_analyzer_output(
                    result.output,
                    side=side,
                    baseline_ids=baseline_ids,
                    comparison_ids=comparison_ids,
                    task_statuses=task_statuses,
                    changed_channels=set(changed_channels),
                    task_outcomes=task_outcomes,
                    available_channel_ids=available_channel_ids,
                )
                write_json(output_path, output)
                return output
        assert result is not None
        fallback = conservative_post_analyzer_fallback(
            side=side,
            baseline_ids=baseline_ids,
            comparison_ids=comparison_ids,
            task_statuses=task_statuses,
            changed_channels=changed_channels,
            task_outcomes=task_outcomes,
            task_comparisons=common["task_comparisons"],
            failure=f"{result.outcome}: {result.validation_error}",
        )
        validate_post_analyzer_output(
            fallback,
            side=side,
            baseline_ids=baseline_ids,
            comparison_ids=comparison_ids,
            task_statuses=task_statuses,
            changed_channels=set(changed_channels),
            task_outcomes=task_outcomes,
            available_channel_ids=available_channel_ids,
        )
        write_json(output_path, fallback)
        write_json(
            self.root / f"{label}_{side}_fallback.json",
            {
                "schema": "harnesslens.post-analyzer-fallback.v1",
                "side": side,
                "attempts": 3,
                "last_failure": f"{result.outcome}: {result.validation_error}",
                "reason": "retain exact rollout comparison status without inventing attribution",
            },
        )
        return fallback

    def _run_side(
        self,
        *,
        label: str,
        side: str,
        experiences: Sequence[Mapping[str, Any]],
        discovery: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        experience_ids = tuple(str(item["id"]) for item in experiences)
        harness_query = discovery["harness_query"]
        channel_contracts = _query_channel_contracts(harness_query)
        channel_ids = set(channel_contracts)
        output_path = self.root / f"{label}_{side}.json"
        if output_path.exists():
            cached = json.loads(output_path.read_text(encoding="utf-8"))
            try:
                validate_analyzer_output(
                    cached,
                    side=side,
                    experience_ids=experience_ids,
                    channel_ids=channel_ids,
                    channel_contracts=channel_contracts,
                    harness=self.harness,
                )
            except ValueError:
                pass
            else:
                return cached
        base_job_id = f"analyzer-{side}-{label}"
        for workspace in sorted(
            (self.run_root / "intelligent_jobs").glob(f"{base_job_id}*"),
            reverse=True,
        ):
            stdout_path = intelligent_stdout_path(workspace, self.harness)
            if not stdout_path.is_file():
                continue
            try:
                recovered = canonicalize_analyzer_output(
                    dict(
                        parse_json_object(
                            stdout_path.read_text(encoding="utf-8", errors="replace")
                        )
                    ),
                    harness_query=harness_query,
                    expected_experience_ids=experience_ids,
                )
                validate_analyzer_output(
                    recovered,
                    side=side,
                    experience_ids=experience_ids,
                    channel_ids=channel_ids,
                    channel_contracts=channel_contracts,
                    harness=self.harness,
                )
            except (ValueError, KeyError, TypeError):
                continue
            write_json(output_path, recovered)
            return recovered
        system = (
            REUSABLE_ANALYZER_SYSTEM
            if side == "reusable"
            else ADJUSTMENT_ANALYZER_SYSTEM
        )
        runner = IntelligentHarnessRunner(
            profile=power_profile(self.harness, max_steps=60),
            budget=self.budget,
            workspace_root=self.run_root / "intelligent_jobs",
            timeout_s=3600,
        )
        result = None
        for attempt in range(3):
            previous_error = result.validation_error if result is not None else ""
            result = runner.run_json(
                job_id=self.budget.next_attempt_id(f"analyzer-{side}-{label}"),
                system_prompt=system,
                input_payload={
                    "side": side,
                    "experiences": list(experiences),
                    "task_categories": discovery["task_categories"],
                    "harness_query": harness_query,
                    "public_environment": discovery["public_environment"],
                    "current_harness": {
                        "version": "v0",
                        "candidate_history": [],
                    },
                    "retry_context": (
                        analyzer_retry_context(previous_error)
                        + (
                            " Cover every reusable experience with adapter-valid candidates while "
                            "preserving distinct behavior hypotheses as distinct candidates. "
                            "If the previous response was malformed JSON, reduce the number of "
                            "candidates and keep skill/instruction text concise; valid JSON is "
                            "more important than long prose."
                            if side == "reusable"
                            else " Describe each proposed adjustment as one concrete, evidence-backed "
                            "Editor brief with observable checks; omit hypotheses that are not ready "
                            "for a bounded edit. Do not write manifest_delta or harness files."
                        )
                        if attempt
                        else ""
                    ),
                },
                validator=lambda output: validate_analyzer_output(
                    canonicalize_analyzer_output(
                        output,
                        harness_query=harness_query,
                        expected_experience_ids=experience_ids,
                    ),
                    side=side,
                    experience_ids=experience_ids,
                    channel_ids=channel_ids,
                    channel_contracts=channel_contracts,
                    harness=self.harness,
                ),
            )
            if result.output is not None:
                write_json(output_path, result.output)
                return result.output
        assert result is not None
        raise RuntimeError(
            f"Analyzer {side} failed: {result.outcome}: {result.validation_error}"
        )


def validate_reusable_plan(
    output: Mapping[str, Any], *, experience_ids: Sequence[str]
) -> None:
    groups = output.get("groups")
    if (
        not isinstance(groups, list)
        or not 1 <= len(groups) <= MAX_REUSABLE_COMPILATION_GROUPS
    ):
        raise ValueError("reusable plan requires one to four groups")
    group_ids: set[str] = set()
    seen: list[str] = []
    for group in groups:
        if not isinstance(group, Mapping):
            raise ValueError("reusable plan groups must be objects")
        group_id = str(group.get("id") or "").strip()
        members = [str(item) for item in group.get("experience_ids") or []]
        if (
            not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", group_id)
            or group_id in group_ids
            or not members
        ):
            raise ValueError("reusable plan groups require unique IDs and members")
        group_ids.add(group_id)
        seen.extend(members)
    if len(seen) != len(set(seen)) or set(seen) != set(experience_ids):
        raise ValueError("reusable plan must partition every experience exactly once")


def combine_reusable_outputs(
    *,
    plan: Mapping[str, Any],
    outputs: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    output_by_group = {str(group_id): output for group_id, output in outputs}
    experience_ids: list[str] = []
    compiled_candidates: list[dict[str, Any]] = []
    for group in plan["groups"]:
        group_id = str(group["id"])
        experience_ids.extend(str(item) for item in group["experience_ids"])
        group_output = output_by_group[group_id]
        source_candidates = [
            (group_id, candidate) for candidate in analyzer_candidates(group_output)
        ]
        incorporated = {
            str(experience_id)
            for _, candidate in source_candidates
            for plan_item in candidate.get("channel_plan") or []
            for experience_id in plan_item.get("experience_ids") or []
        }
        if incorporated != {str(item) for item in group["experience_ids"]}:
            raise ValueError(
                "reusable materializer did not cover its complete planner group"
            )
        compiled_candidates.extend(
            compile_reusable_candidate(group_id=group_id, candidate=candidate)
            for _, candidate in source_candidates
        )
    return {
        "coverage": {
            "experience_ids": experience_ids,
            "public_environment_reviewed": True,
        },
        "candidates": compiled_candidates,
        "compilation_plan": dict(plan),
    }


def partition_adjustment_experiences(
    experiences: Sequence[Mapping[str, Any]],
) -> list[tuple[str, list[Mapping[str, Any]]]]:
    if not experiences:
        raise ValueError("adjustment analysis requires at least one experience")
    group_count = min(
        MAX_ADJUSTMENT_ANALYSIS_GROUPS,
        max(
            1,
            (
                len(experiences) + TARGET_ADJUSTMENT_EXPERIENCES_PER_GROUP - 1
            )
            // TARGET_ADJUSTMENT_EXPERIENCES_PER_GROUP,
        ),
    )
    groups: list[list[Mapping[str, Any]]] = [[] for _ in range(group_count)]
    for index, experience in enumerate(experiences):
        groups[index % group_count].append(experience)
    return [
        (f"adjustment-{index:02d}", group)
        for index, group in enumerate(groups, start=1)
        if group
    ]


def combine_adjustment_outputs(
    *,
    experience_ids: Sequence[str],
    outputs: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    problems: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for group_id, output in outputs:
        candidate_id_map = {
            str(candidate.get("id") or ""): (
                f"{group_id}-{str(candidate.get('id') or 'candidate')}"
            )
            for candidate in output.get("candidates") or []
            if isinstance(candidate, Mapping)
        }
        for raw_problem in output.get("problems") or []:
            problem = copy.deepcopy(dict(raw_problem))
            problem["id"] = f"{group_id}-{str(problem.get('id') or 'problem')}"
            source_candidate_id = str(problem.get("candidate_id") or "")
            if source_candidate_id:
                problem["candidate_id"] = candidate_id_map[source_candidate_id]
            problems.append(problem)
        for raw_candidate in output.get("candidates") or []:
            candidate = copy.deepcopy(dict(raw_candidate))
            candidate["id"] = candidate_id_map[str(candidate.get("id") or "")]
            candidates.append(candidate)
    problems.sort(key=lambda item: int(item.get("priority") or 1_000_000))
    candidate_by_id = {str(item["id"]): item for item in candidates}
    ranked_candidates: list[dict[str, Any]] = []
    for priority, problem in enumerate(problems, start=1):
        problem["priority"] = priority
        candidate_id = str(problem.get("candidate_id") or "")
        if candidate_id and candidate_id in candidate_by_id:
            candidate = candidate_by_id.pop(candidate_id)
            candidate["priority"] = priority
            ranked_candidates.append(candidate)
    for candidate in candidates:
        candidate_id = str(candidate["id"])
        if candidate_id in candidate_by_id:
            ranked_candidates.append(candidate_by_id.pop(candidate_id))
    return {
        "coverage": {"experience_ids": [str(item) for item in experience_ids]},
        "problems": problems,
        "candidates": ranked_candidates,
        "analysis_partitions": [str(group_id) for group_id, _ in outputs],
    }


def compile_reusable_candidate(
    *,
    group_id: str,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    compiled = copy.deepcopy(dict(candidate))
    source_id = str(candidate.get("id") or "candidate")
    source_slug = (
        re.sub(r"[^a-z0-9]+", "-", source_id.lower()).strip("-") or "candidate"
    )
    compiled["id"] = f"reusable-{group_id}-{source_slug}"
    compiled["compilation_group_id"] = group_id
    compiled["source_candidate_id"] = source_id
    delta = dict(compiled.get("manifest_delta") or {})
    namespace = f"{group_id}-{source_slug}"
    if delta.get("files"):
        delta["files"] = [
            _namespace_skill_file(namespace, raw_file) for raw_file in delta["files"]
        ]
    compiled["manifest_delta"] = delta
    return compiled


def _namespace_skill_file(group_id: str, raw_file: Mapping[str, Any]) -> dict[str, Any]:
    item = dict(raw_file)
    path = str(item.get("path") or "")
    match = re.fullmatch(
        r"\.opencode/skills/([a-z0-9]+(?:-[a-z0-9]+)*)/SKILL\.md", path
    )
    if not match:
        return item
    slug = f"{group_id}-{match.group(1)}"
    item["path"] = f".opencode/skills/{slug}/SKILL.md"
    item["content"] = re.sub(
        r"(?m)^name:\s*.*$", f"name: {slug}", str(item.get("content") or ""), count=1
    )
    return item


def validate_analyzer_output(
    output: Mapping[str, Any],
    *,
    side: str,
    experience_ids: Sequence[str],
    channel_ids: set[str],
    channel_contracts: Mapping[str, Mapping[str, Any]] | None = None,
    harness: str = "opencode",
) -> None:
    coverage = output.get("coverage")
    if not isinstance(coverage, Mapping) or set(
        coverage.get("experience_ids") or []
    ) != set(experience_ids):
        raise ValueError("Analyzer must cover every experience ID")
    if side == "reusable" and coverage.get("public_environment_reviewed") is not True:
        raise ValueError("Reusable Analyzer must review the public environment")
    if side == "adjustment":
        problems = output.get("problems")
        if not isinstance(problems, list) or not problems:
            raise ValueError("Adjustment Analyzer requires ranked problems")
        problem_ids: set[str] = set()
        for problem in problems:
            problem_id = str(problem.get("id") or "").strip()
            supported = {str(item) for item in problem.get("experience_ids") or []}
            if not problem_id or problem_id in problem_ids:
                raise ValueError("adjustment problems require unique IDs")
            problem_ids.add(problem_id)
            if not supported or not supported.issubset(set(experience_ids)):
                raise ValueError("adjustment problem requires valid experience support")
            if not problem.get("evidence_refs") or not problem.get(
                "local_success_criteria"
            ):
                raise ValueError(
                    "adjustment problem requires evidence and local success criteria"
                )
            hypotheses = problem.get("channel_hypotheses")
            if not isinstance(hypotheses, list):
                raise ValueError("adjustment problem channel hypotheses must be an array")
            if problem.get("candidate_id") and not hypotheses:
                raise ValueError(
                    "actionable adjustment problem requires channel hypotheses"
                )
            if any(
                str(item.get("channel_id") or "") not in channel_ids
                for item in hypotheses
            ):
                raise ValueError("Analyzer referenced an undiscovered channel")
            if "rollout_task_ids" in problem:
                raise ValueError("Adjustment Analyzer must not select rollout task IDs")
            if problem.get("diagnostic_rollout_needed") is not True:
                raise ValueError(
                    "initial adjustment problem must request diagnostic rollout"
                )
    if "rollout_task_ids" in output:
        raise ValueError("Analyzer must not select rollout task IDs")
    candidates = output.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Analyzer candidates must be an array")
    if side == "reusable" and not candidates:
        raise ValueError("Reusable Analyzer requires at least one candidate")
    candidate_ids: set[str] = set()
    for candidate in candidates:
        candidate_id = str(candidate.get("id") or "")
        if candidate_id in candidate_ids:
            raise ValueError("Analyzer candidate IDs must be unique")
        candidate_ids.add(candidate_id)
        _validate_candidate(
            candidate,
            experience_ids=set(experience_ids),
            channel_ids=channel_ids,
            channel_contracts=channel_contracts,
            harness=harness,
            require_causal_contract=(side == "adjustment"),
        )
    if side == "reusable":
        incorporated = {
            str(experience_id)
            for candidate in candidates
            for plan in candidate.get("channel_plan") or []
            for experience_id in plan.get("experience_ids") or []
        }
        missing = set(experience_ids) - incorporated
        if missing:
            raise ValueError(
                "Reusable Analyzer left experiences outside compilation batches: "
                f"{sorted(missing)}"
            )
    if side == "adjustment":
        referenced_by_candidate: dict[str, list[Mapping[str, Any]]] = {}
        for problem in output["problems"]:
            candidate_id = str(problem.get("candidate_id") or "")
            if candidate_id:
                referenced_by_candidate.setdefault(candidate_id, []).append(problem)
        referenced = set(referenced_by_candidate)
        if referenced - candidate_ids:
            raise ValueError("adjustment problem references an unknown candidate")
        if candidate_ids - referenced:
            raise ValueError("adjustment candidate must belong to one atomic problem")
        candidates_by_id = {
            str(candidate.get("id") or ""): candidate for candidate in candidates
        }
        for candidate_id, candidate_problems in referenced_by_candidate.items():
            if len(candidate_problems) != 1:
                raise ValueError(
                    "adjustment candidate must belong to one atomic problem"
                )
            problem_experiences = {
                str(item) for item in candidate_problems[0].get("experience_ids") or []
            }
            candidate_experiences = {
                str(experience_id)
                for plan in candidates_by_id[candidate_id].get("channel_plan") or []
                for experience_id in plan.get("experience_ids") or []
            }
            if not candidate_experiences.issubset(problem_experiences):
                raise ValueError(
                    "adjustment candidate includes evidence outside its one atomic problem"
                )


def analyzer_retry_context(validation_error: str) -> str:
    return (
        f"Previous output failed validation: {str(validation_error).strip()}. "
        "Return an evidence-backed editor brief; do not write manifest_delta or harness files. "
        "Each channel_plan array item must be one JSON object with exactly one "
        "channel_id, one operation, one nonempty experience_ids array, and one rationale; "
        "never put multiple channel_id/operation/experience_ids fields in the same object."
    )


def canonicalize_analyzer_output(
    output: Mapping[str, Any],
    *,
    harness_query: Mapping[str, Any] | None = None,
    expected_experience_ids: Sequence[str] | None = None,
) -> Mapping[str, Any]:
    normalized = _restore_truncated_experience_ids(
        output, expected_experience_ids or ()
    )
    target_harness = str((harness_query or {}).get("harness") or "opencode")
    discovered_channel_ids = set(_query_channel_contracts(harness_query or {}))
    valid_experience_ids = {
        str(item)
        for item in (normalized.get("coverage") or {}).get("experience_ids") or []
    }
    if "problems" in normalized:
        normalized["problems"] = [
            (
                {**dict(problem), "diagnostic_rollout_needed": True}
                if isinstance(problem, Mapping)
                else problem
            )
            for problem in normalized.get("problems") or []
            if _actionable_adjustment_problem(problem)
        ]
    candidates = []
    for raw_candidate in normalized.get("candidates") or []:
        if not isinstance(raw_candidate, Mapping):
            candidates.append(raw_candidate)
            continue
        candidate = dict(raw_candidate)
        channel_plan = []
        for raw_plan in candidate.get("channel_plan") or []:
            if not isinstance(raw_plan, Mapping):
                channel_plan.append(raw_plan)
                continue
            channel_id = CHANNEL_ID_ALIASES.get(
                str(raw_plan.get("channel_id") or ""),
                str(raw_plan.get("channel_id") or ""),
            )
            mcp_channel_id = f"mcp_{channel_id}"
            if (
                channel_id not in discovered_channel_ids
                and mcp_channel_id in discovered_channel_ids
            ):
                channel_id = mcp_channel_id
            plan = {
                **dict(raw_plan),
                "channel_id": channel_id,
            }
            refs = [str(item) for item in plan.get("experience_ids") or []]
            if valid_experience_ids:
                refs = [item for item in refs if item in valid_experience_ids]
            if not refs:
                continue
            plan["experience_ids"] = refs
            channel_plan.append(plan)
        candidate["channel_plan"] = channel_plan
        raw_delta = candidate.get("manifest_delta")
        delta = dict(raw_delta) if isinstance(raw_delta, Mapping) else {}
        legacy_instruction = (
            delta.pop("instructions_rules", None)
            if target_harness == "opencode"
            else None
        )
        if legacy_instruction is not None and "instructions" not in delta:
            if isinstance(legacy_instruction, Mapping):
                legacy_instruction = (
                    legacy_instruction.get("append")
                    or legacy_instruction.get("diff")
                    or legacy_instruction.get("content")
                )
            if str(legacy_instruction or "").strip():
                delta["instructions"] = [str(legacy_instruction)]
        legacy_params = delta.pop("tool_param_patches", None) or delta.pop(
            "tool_parameter_patches", None
        )
        if isinstance(legacy_params, Mapping):
            tool_patches = dict(delta.get("tool_desc_patches") or {})
            for tool, raw_patch in legacy_params.items():
                existing = dict(tool_patches.get(str(tool)) or {})
                incoming = (
                    dict(raw_patch)
                    if isinstance(raw_patch, Mapping)
                    else {"params": raw_patch}
                )
                existing_params = dict(existing.get("params") or {})
                existing_params.update(dict(incoming.get("params") or {}))
                existing["params"] = existing_params
                tool_patches[str(tool)] = existing
            delta["tool_desc_patches"] = tool_patches
        if isinstance(delta.get("instructions"), str):
            delta["instructions"] = [str(delta["instructions"])]
        if isinstance(delta.get("prompt_appends"), str):
            delta["prompt_appends"] = [str(delta["prompt_appends"])]
        delta["config_patch"] = _canonical_config_patch(delta.get("config_patch"))
        files = []
        has_skill = False
        for raw_file in delta.get("files") or []:
            if not isinstance(raw_file, Mapping):
                files.append(raw_file)
                continue
            item = dict(raw_file)
            path = str(item.get("path") or "")
            match = (
                re.fullmatch(
                    r"\.opencode/skills/([a-z0-9]+(?:-[a-z0-9]+)*)/SKILL\.md",
                    path,
                )
                if target_harness == "opencode"
                else None
            )
            if match:
                has_skill = True
                content = str(item.get("content") or "")
                if re.search(r"(?m)^name:\s*.*$", content):
                    content = re.sub(
                        r"(?m)^name:\s*.*$",
                        f"name: {match.group(1)}",
                        content,
                        count=1,
                    )
                item["content"] = content
            files.append(item)
        if files or "files" in delta:
            delta["files"] = files
        if has_skill:
            delta["config_patch"]["tools.skill"] = True
        if not delta["config_patch"]:
            delta.pop("config_patch", None)
        if not delta:
            candidate["manifest_delta"] = {}
            candidates.append(candidate)
            continue
        if target_harness != "opencode":
            candidate["channel_plan"] = channel_plan
            candidate["manifest_delta"] = delta
            candidates.append(candidate)
            continue
        materialized = OpencodeHarnessAdapter().materialized_channel_ids(delta)
        plans = [
            plan
            for plan in candidate.get("channel_plan") or []
            if not isinstance(plan, Mapping)
            or MCP_CHANNEL_BASE.get(
                str(plan.get("channel_id") or ""), str(plan.get("channel_id") or "")
            )
            in materialized
        ]
        declared = {
            MCP_CHANNEL_BASE.get(
                str(plan.get("channel_id") or ""), str(plan.get("channel_id") or "")
            )
            for plan in plans
            if isinstance(plan, Mapping)
        }
        refs = sorted(
            {
                str(experience_id)
                for plan in plans
                if isinstance(plan, Mapping)
                for experience_id in plan.get("experience_ids") or []
            }
        )
        for channel in sorted(materialized - declared):
            plans.append(
                {
                    "channel_id": channel,
                    "operation": "materialize manifest content",
                    "experience_ids": refs,
                    "rationale": (
                        "The manifest materializes this channel for the same evidence-bounded "
                        "behavior hypothesis."
                    ),
                }
            )
        candidate["channel_plan"] = plans
        candidate["manifest_delta"] = delta
        candidates.append(candidate)
    normalized["candidates"] = candidates
    if isinstance(output, dict):
        output.clear()
        output.update(normalized)
        return output
    return normalized


def _restore_truncated_experience_ids(
    output: Mapping[str, Any], expected_experience_ids: Sequence[str]
) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(output))
    expected = {str(item) for item in expected_experience_ids}
    if not expected:
        return normalized
    reported = {
        str(item)
        for item in (normalized.get("coverage") or {}).get("experience_ids") or []
    }
    missing = expected - reported
    unknown = reported - expected
    aliases: dict[str, str] = {}
    for source in unknown:
        matches = [
            target
            for target in missing
            if min(len(source), len(target)) >= 16
            and (source.startswith(target) or target.startswith(source))
        ]
        if len(matches) == 1:
            aliases[source] = matches[0]
    if len(set(aliases.values())) != len(aliases):
        return normalized

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "experience_ids" and isinstance(child, list):
                    value[key] = [aliases.get(str(item), item) for item in child]
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(normalized)
    return normalized


def _actionable_adjustment_problem(problem: Any) -> bool:
    if not isinstance(problem, Mapping):
        return True
    return bool(problem.get("local_success_criteria"))


def _canonical_config_patch(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    flattened: dict[str, Any] = {}

    def visit(prefix: str, item: Any) -> None:
        if isinstance(item, Mapping):
            if set(item) == {"enabled"} and isinstance(item.get("enabled"), bool):
                flattened[prefix] = bool(item["enabled"])
                return
            for key, child in item.items():
                visit(f"{prefix}.{key}" if prefix else str(key), child)
            return
        flattened[prefix] = item

    visit("", value)
    return {
        key: item
        for key, item in flattened.items()
        if not (key == "permission.skill" and isinstance(item, bool))
    }


def _validate_candidate(
    candidate: Mapping[str, Any],
    *,
    experience_ids: set[str],
    channel_ids: set[str],
    channel_contracts: Mapping[str, Mapping[str, Any]] | None = None,
    harness: str = "opencode",
    require_causal_contract: bool = False,
) -> None:
    if (
        not str(candidate.get("id") or "").strip()
        or not str(candidate.get("objective") or "").strip()
    ):
        raise ValueError("Analyzer candidate requires ID and objective")
    if require_causal_contract:
        causal_fields = (
            "observed_terminal_failure",
            "causal_hypothesis",
            "intervention_point",
            "expected_runtime_event",
            "falsifying_observation",
        )
        if any(not str(candidate.get(field) or "").strip() for field in causal_fields):
            raise ValueError(
                "adjustment candidate requires a complete falsifiable causal contract"
            )
    plans = candidate.get("channel_plan")
    if not isinstance(plans, list) or not plans:
        raise ValueError("Analyzer candidate requires at least one channel plan")
    selected: set[str] = set()
    for plan in plans:
        if str(plan.get("channel_id") or "") not in channel_ids:
            raise ValueError("candidate channel was not discovered")
        refs = {str(item) for item in plan.get("experience_ids") or []}
        if not refs or not refs.issubset(experience_ids):
            raise ValueError("candidate requires valid experience references")
        selected.update(refs)
    delta = candidate.get("manifest_delta")
    if delta is None:
        delta = {}
    if not isinstance(delta, Mapping):
        raise ValueError("candidate manifest_delta must be an object when supplied")
    allowed = {
        "instructions",
        "files",
        "config_patch",
        "prompt_appends",
        "tool_desc_patches",
    }
    if set(delta) - allowed:
        raise ValueError("candidate manifest_delta contains unsupported keys")
    declared_channels = {
        MCP_CHANNEL_BASE.get(
            str(plan.get("channel_id") or ""), str(plan.get("channel_id") or "")
        )
        for plan in plans
    }
    normalized_harness = str(harness).strip().lower().replace("-", "_")
    if delta and normalized_harness == "opencode":
        adapter = OpencodeHarnessAdapter()
        adapter.validate_delta(delta)
        materialized_channels = adapter.materialized_channel_ids(delta)
        if declared_channels != materialized_channels:
            raise ValueError(
                "candidate channel plan does not match its materialized manifest: "
                f"declared={sorted(declared_channels)} materialized={sorted(materialized_channels)}"
            )
    elif delta:
        _validate_generic_delta(delta)
    if delta and channel_contracts is not None:
        for plan in plans:
            channel_id = str(plan.get("channel_id") or "")
            contract = channel_contracts.get(channel_id)
            if not isinstance(contract, Mapping):
                raise ValueError("candidate channel has no Harness Query contract")
            if str(contract.get("status") or "") not in {"modifiable", "verified"}:
                raise ValueError(
                    "candidate channel is not modifiable according to Harness Query"
                )
            operation = contract.get("operation")
            if not isinstance(operation, Mapping) or not _operation_materialized(
                delta, operation
            ):
                raise ValueError(
                    "candidate manifest does not satisfy its Harness Query operation contract"
                )
        if normalized_harness != "opencode":
            _validate_no_unclaimed_delta(delta, plans, channel_contracts)
    validation = candidate.get("validation")
    if not isinstance(validation, Mapping) or not validation.get(
        "local_behavior_checks"
    ):
        raise ValueError("candidate requires local behavior checks")
    no_regression = {
        str(item) for item in validation.get("no_regression_experience_ids") or []
    }
    if no_regression - experience_ids:
        raise ValueError("candidate references an unknown no-regression experience")


def _query_channel_contracts(
    harness_query: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(item["id"]): item
        for section in ("modifiable_modules", "mcp_editable_points")
        for item in harness_query.get(section) or []
        if isinstance(item, Mapping)
        and item.get("id")
        and str(item.get("status") or "") in {"modifiable", "verified"}
    }


def _operation_materialized(
    delta: Mapping[str, Any], operation: Mapping[str, Any]
) -> bool:
    kind = str(operation.get("kind") or "")
    field = str(operation.get("manifest_field") or "")
    if kind == "project_file":
        pattern = str(operation.get("path_pattern") or "")
        return any(
            isinstance(item, Mapping)
            and _path_matches_pattern(str(item.get("path") or ""), pattern)
            for item in delta.get("files") or []
        )
    if kind == "workspace_config":
        key = str(operation.get("key") or "")
        if key and delta.get(key):
            return True
        path = str(operation.get("path") or "")
        for item in delta.get("files") or []:
            if not isinstance(item, Mapping) or str(item.get("path") or "") != path:
                continue
            try:
                content = str(item.get("content") or "")
                config = (
                    tomllib.loads(content)
                    if path.endswith(".toml")
                    else json.loads(content)
                )
            except (json.JSONDecodeError, tomllib.TOMLDecodeError):
                continue
            if isinstance(config, Mapping) and key in config:
                return True
        return False
    if kind in {"prompt_content", "tool_schema_patch"}:
        return bool(field and delta.get(field))
    if kind == "harness_config_patch":
        patch = delta.get(field or "config_patch")
        if not isinstance(patch, Mapping) or not patch:
            return False
        exact = str(operation.get("key") or "")
        prefix = str(operation.get("key_prefix") or "")
        keys = {str(key) for key in patch}
        if exact:
            return exact in keys
        if prefix:
            return any(key.startswith(prefix) for key in keys)
        return True
    return False


def _validate_generic_delta(delta: Mapping[str, Any]) -> None:
    files = delta.get("files") or []
    if not isinstance(files, list) or any(
        not isinstance(item, Mapping) for item in files
    ):
        raise ValueError("candidate files must be an array of objects")
    for item in files:
        raw = str(item.get("path") or "")
        path = PurePosixPath(raw)
        if not raw or path.is_absolute() or ".." in path.parts or "\\" in raw:
            raise ValueError(f"unsafe candidate file path: {raw}")
        if not str(item.get("content") or "").strip():
            raise ValueError("candidate file content must be nonempty")
    config = delta.get("config_patch") or {}
    tools = delta.get("tool_desc_patches") or {}
    if not isinstance(config, Mapping) or not isinstance(tools, Mapping):
        raise ValueError("config_patch and tool_desc_patches must be objects")
    for field in ("instructions", "prompt_appends"):
        value = delta.get(field) or []
        if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value
        ):
            raise ValueError(f"{field} must be an array of strings")


def _validate_no_unclaimed_delta(
    delta: Mapping[str, Any],
    plans: Sequence[Mapping[str, Any]],
    contracts: Mapping[str, Mapping[str, Any]],
) -> None:
    operations = [
        contracts.get(str(plan.get("channel_id") or ""), {}).get("operation")
        for plan in plans
    ]
    operations = [item for item in operations if isinstance(item, Mapping)]
    for field in ("instructions", "prompt_appends", "tool_desc_patches"):
        if delta.get(field) and not any(
            str(operation.get("manifest_field") or "") == field
            for operation in operations
        ):
            raise ValueError(
                f"candidate manifest field is not claimed by Harness Query: {field}"
            )
    for item in delta.get("files") or []:
        path = str(item.get("path") or "")
        if not any(
            (
                str(operation.get("kind") or "") == "project_file"
                and _path_matches_pattern(
                    path, str(operation.get("path_pattern") or "")
                )
            )
            or (
                str(operation.get("kind") or "") == "workspace_config"
                and path == str(operation.get("path") or "")
            )
            for operation in operations
        ):
            raise ValueError(f"candidate file is not claimed by Harness Query: {path}")
    for key in delta.get("config_patch") or {}:
        if not any(
            str(operation.get("kind") or "") == "harness_config_patch"
            and (
                str(operation.get("key") or "") == str(key)
                or str(key).startswith(str(operation.get("key_prefix") or "\0"))
            )
            for operation in operations
        ):
            raise ValueError(
                f"candidate config key is not claimed by Harness Query: {key}"
            )


def _path_matches_pattern(path: str, pattern: str) -> bool:
    if not path or not pattern:
        return False
    parts = re.split(r"(<[^/>]+>)", pattern)
    expression = "".join(
        r"[^/]+" if re.fullmatch(r"<[^/>]+>", part) else re.escape(part)
        for part in parts
    )
    return re.fullmatch(expression, path) is not None


def analyzer_candidates(output: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = output.get("candidates")
    return [dict(item) for item in candidates or [] if isinstance(item, Mapping)]


def build_experience_dispositions(
    *,
    experience: Mapping[str, Any],
    reusable: Mapping[str, Any],
    adjustment: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_refs: dict[str, list[str]] = {}
    for candidate in [
        *analyzer_candidates(reusable),
        *analyzer_candidates(adjustment),
    ]:
        candidate_id = str(candidate["id"])
        for plan in candidate.get("channel_plan") or []:
            for experience_id in plan.get("experience_ids") or []:
                candidate_refs.setdefault(str(experience_id), []).append(candidate_id)
    items = []
    for section in ("reusable", "needs_adjustment"):
        for passage in experience.get(section) or []:
            experience_id = str(passage["id"])
            candidates = sorted(set(candidate_refs.get(experience_id) or []))
            items.append(
                {
                    "experience_id": experience_id,
                    "section": section,
                    "status": "materialized" if candidates else "deferred",
                    "candidate_ids": candidates,
                    "reason": (
                        "Included in an Analyzer candidate manifest."
                        if candidates
                        else "No evidence-bounded candidate selected this experience yet."
                    ),
                }
            )
    return {"items": items}


def _post_candidate_context(decision: Mapping[str, Any]) -> dict[str, Any]:
    candidate = decision["candidate"]
    rollout_request = decision.get("rollout_request") or {}
    return {
        "harness_version": decision["harness_version"],
        "parent_version": candidate["parent_version"],
        "selected_candidate_side": str(
            decision.get("selected_candidate_side") or "unknown"
        ),
        "source_candidate_ids": list(candidate.get("source_candidate_ids") or []),
        "channel_diffs": candidate["channel_diffs"],
        "manifest_delta": candidate["manifest_delta"],
        "workspace_delta": dict(candidate.get("workspace_delta") or {"files": []}),
        "workspace_diff": list(candidate.get("workspace_diff") or []),
        "editor_summary": candidate.get("editor_summary"),
        "causal_contract": {
            field: str(candidate.get(field) or "")
            for field in (
                "observed_terminal_failure",
                "causal_hypothesis",
                "intervention_point",
                "expected_runtime_event",
                "falsifying_observation",
            )
            if candidate.get(field)
        },
        "rollout_request": {
            "task_ids": list(rollout_request.get("task_ids") or []),
            "local_success_criteria": list(
                rollout_request.get("local_success_criteria") or []
            ),
            "rationale": str(rollout_request.get("rationale") or ""),
            **(
                {"task_roles": dict(rollout_request["task_roles"])}
                if isinstance(rollout_request.get("task_roles"), Mapping)
                and rollout_request.get("task_roles")
                else {}
            ),
        },
    }


def _rollout_channel_usage(
    rollout_output: str | Path, decision: Mapping[str, Any]
) -> list[dict[str, Any]]:
    candidate = decision["candidate"]
    legacy_files = (candidate.get("manifest_delta") or {}).get("files") or []
    workspace_files = (candidate.get("workspace_delta") or {}).get("files") or []
    skill_names = {
        Path(str(item.get("path") or "")).parent.name
        for item in [*legacy_files, *workspace_files]
        if isinstance(item, Mapping)
        and str(item.get("change") or "") != "deleted"
        and str(item.get("path") or "").endswith("/SKILL.md")
    }
    if not skill_names:
        return []
    rollout = json.loads(Path(rollout_output).read_text(encoding="utf-8"))
    result: list[dict[str, Any]] = []
    for task_id, task in (rollout.get("per_task") or {}).items():
        if not isinstance(task, Mapping):
            continue
        for trajectory in task.get("trajectory_paths") or []:
            path = Path(str(trajectory))
            rows = [
                json.loads(line)
                for line in path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                if line.strip()
            ]
            if len(rows) != 1 or not isinstance(rows[0], Mapping):
                raise ValueError(f"channel usage requires one retained trial: {path}")
            usage = compact_channel_usage(rows[0])
            available = {item["name"] for item in usage["skills_available"]}
            invoked = {
                item["name"]: item["n_calls"] for item in usage["skills_invoked"]
            }
            result.append(
                {
                    "task_id": str(task_id),
                    "trial": rows[0].get("trial"),
                    "on_demand": [
                        {
                            "channel": "skills",
                            "artifact": name,
                            "available": name in available,
                            "invocations": invoked.get(name, 0),
                        }
                        for name in sorted(skill_names)
                    ],
                }
            )
    return result


def validate_post_analyzer_output(
    output: Mapping[str, Any],
    *,
    side: str,
    baseline_ids: Sequence[str],
    comparison_ids: Sequence[str],
    task_statuses: Mapping[str, str],
    changed_channels: set[str],
    task_outcomes: Mapping[str, Mapping[str, Any]] | None = None,
    available_channel_ids: set[str] | None = None,
) -> None:
    coverage = output.get("coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("post Analyzer requires coverage")
    expected = {
        "baseline_experience_ids": set(str(item) for item in baseline_ids),
        "comparison_experience_ids": set(str(item) for item in comparison_ids),
        "task_ids": set(task_statuses),
    }
    for field, values in expected.items():
        if set(str(item) for item in coverage.get(field) or []) != values:
            raise ValueError(f"post Analyzer must cover every {field}")
    if coverage.get("channel_usage_reviewed") is not True:
        raise ValueError("post Analyzer must review channel usage")
    if side == "reusable":
        preservation = output.get("preservation")
        if not isinstance(preservation, Mapping):
            raise ValueError("reusable post Analyzer requires preservation")
        regressions = preservation.get("attributable_regressions")
        if not isinstance(regressions, list):
            raise ValueError("attributable_regressions must be an array")
        regression_tasks: set[str] = set()
        for item in regressions:
            task_id = str(item.get("task_id") or "")
            channels = {str(value) for value in item.get("channel_ids") or []}
            if (
                task_id not in task_statuses
                or task_statuses[task_id]
                not in {
                    "regressed",
                    "mixed",
                    "stable_success",
                }
                or not item.get("evidence_refs")
                or not channels
                or channels - changed_channels
                or not str(item.get("reason") or "").strip()
            ):
                raise ValueError(
                    "attributable regression lacks changed-channel evidence"
                )
            regression_tasks.add(task_id)
        preserved = set(
            str(item) for item in preservation.get("preserved_task_ids") or []
        )
        if preserved != set(task_statuses) - regression_tasks:
            raise ValueError("preservation must account for every non-regressed task")
        if (
            preservation.get("candidate_recommendation")
            not in {
                "accept",
                "reject",
                "uncertain",
            }
            or not str(preservation.get("rationale") or "").strip()
        ):
            raise ValueError("preservation requires a reasoned recommendation")
        return
    problem = output.get("primary_problem")
    if not isinstance(problem, Mapping):
        raise ValueError("adjustment post Analyzer requires primary_problem")
    assessments = problem.get("task_assessments")
    if not isinstance(assessments, list) or len(assessments) != len(task_statuses):
        raise ValueError("adjustment post Analyzer requires one assessment per task")
    relations = {
        "attributed",
        "possibly_related",
        "not_attributed",
        "unresolved",
        "random",
    }
    seen: set[str] = set()
    for item in assessments:
        task_id = str(item.get("task_id") or "")
        if task_id not in task_statuses or task_id in seen:
            raise ValueError("post task assessments must be exact and unique")
        if str(item.get("status") or "") != task_statuses[task_id]:
            raise ValueError("post task assessment changed comparison status")
        if task_outcomes is not None and dict(
            item.get("outcome_summary") or {}
        ) != dict(task_outcomes.get(task_id) or {}):
            raise ValueError("post task assessment changed metric outcomes")
        if item.get("relation") not in relations or not item.get("evidence_refs"):
            raise ValueError("post task assessment requires relation and evidence")
        if not str(item.get("reason") or "").strip():
            raise ValueError("post task assessment requires a reason")
        seen.add(task_id)
    attribution = problem.get("channel_attribution")
    if not isinstance(attribution, Mapping):
        raise ValueError("post Analyzer requires channel attribution")
    if attribution.get("relation") not in {
        "attributed",
        "partially_attributed",
        "not_attributed",
        "unresolved",
        "random",
    }:
        raise ValueError("post Analyzer has invalid channel attribution")
    channels = {str(item) for item in attribution.get("channel_ids") or []}
    if not channels or channels - changed_channels or not attribution.get("reason"):
        raise ValueError("post Analyzer attribution must name changed channels")
    if problem.get("recommendation") not in {"accept", "reject", "refine"}:
        raise ValueError("post Analyzer requires an actionable recommendation")
    if not str(problem.get("local_recovery") or "").strip():
        raise ValueError("post Analyzer requires local recovery detail")
    if not isinstance(problem.get("further_rollout_needed"), bool):
        raise ValueError("post Analyzer must state whether more rollout is needed")
    replan = output.get("replan_candidate")
    if replan is not None:
        _validate_candidate(
            replan,
            experience_ids={str(item) for item in baseline_ids},
            channel_ids=set(available_channel_ids or changed_channels),
            require_causal_contract=True,
        )


def conservative_post_analyzer_fallback(
    *,
    side: str,
    baseline_ids: Sequence[str],
    comparison_ids: Sequence[str],
    task_statuses: Mapping[str, str],
    changed_channels: Sequence[str],
    task_outcomes: Mapping[str, Mapping[str, Any]],
    task_comparisons: Sequence[Mapping[str, Any]],
    failure: str,
) -> dict[str, Any]:
    """Retain measured evidence when repeated analysis attempts are invalid."""
    channel_ids = sorted({str(item) for item in changed_channels if str(item)})
    if not channel_ids:
        raise ValueError("post Analyzer fallback requires a changed channel")
    refs_by_task = {
        str(item.get("task_id") or ""): [
            *[str(ref) for ref in item.get("candidate_refs") or []],
            *[str(ref) for ref in item.get("baseline_refs") or []],
        ]
        for item in task_comparisons
        if isinstance(item, Mapping)
    }
    coverage = {
        "baseline_experience_ids": [str(item) for item in baseline_ids],
        "comparison_experience_ids": [str(item) for item in comparison_ids],
        "task_ids": [str(item) for item in task_statuses],
        "channel_usage_reviewed": True,
    }
    if side == "reusable":
        return {
            "coverage": coverage,
            "preservation": {
                "attributable_regressions": [],
                "preserved_task_ids": [str(item) for item in task_statuses],
                "candidate_recommendation": "uncertain",
                "rationale": (
                    "Post-rollout reusable analysis was invalid after bounded retries; "
                    "no channel-attributable regression is asserted. "
                    f"Last failure: {failure}"
                ),
            },
        }
    assessments = []
    for task_id, status in task_statuses.items():
        refs = refs_by_task.get(str(task_id)) or [f"comparison:{task_id}"]
        assessments.append(
            {
                "task_id": str(task_id),
                "status": str(status),
                "relation": "unresolved",
                "evidence_refs": refs,
                "reason": (
                    "The recorded comparison is retained, but repeated invalid Analyzer "
                    "outputs prevent a causal attribution."
                ),
                "outcome_summary": dict(task_outcomes.get(str(task_id)) or {}),
            }
        )
    return {
        "coverage": coverage,
        "primary_problem": {
            "summary": "Post-rollout adjustment attribution is unavailable after bounded retries.",
            "task_assessments": assessments,
            "channel_attribution": {
                "relation": "unresolved",
                "channel_ids": channel_ids,
                "reason": "No valid Analyzer output established a causal channel effect.",
            },
            "local_recovery": "No attribution-based recovery claim is made.",
            "recommendation": "reject",
            "further_rollout_needed": False,
        },
        "replan_candidate": None,
    }


def canonicalize_post_analyzer_output(
    output: Mapping[str, Any],
    *,
    side: str,
    baseline_ids: Sequence[str],
    comparison_ids: Sequence[str],
    task_statuses: Mapping[str, str],
    changed_channels: set[str],
    task_outcomes: Mapping[str, Mapping[str, Any]] | None = None,
    available_channel_ids: set[str] | None = None,
) -> dict[str, Any]:
    normalized = dict(output)
    normalized["coverage"] = {
        "baseline_experience_ids": [str(item) for item in baseline_ids],
        "comparison_experience_ids": [str(item) for item in comparison_ids],
        "task_ids": [str(item) for item in task_statuses],
        "channel_usage_reviewed": True,
    }
    if side == "adjustment":
        problem = dict(normalized.get("primary_problem") or {})
        assessments_by_task = {
            str(item.get("task_id") or ""): dict(item)
            for item in problem.get("task_assessments") or []
            if isinstance(item, Mapping)
        }
        problem["task_assessments"] = [
            {
                **assessments_by_task.get(str(task_id), {"task_id": str(task_id)}),
                "task_id": str(task_id),
                "status": str(status),
                **(
                    {"outcome_summary": dict(task_outcomes[str(task_id)])}
                    if task_outcomes and str(task_id) in task_outcomes
                    else {}
                ),
            }
            for task_id, status in task_statuses.items()
        ]
        attribution = dict(problem.get("channel_attribution") or {})
        channels = [
            str(item)
            for item in attribution.get("channel_ids") or []
            if str(item) in changed_channels
        ]
        if channels:
            attribution["channel_ids"] = channels
        problem["channel_attribution"] = attribution
        normalized["primary_problem"] = problem
    return normalized
