from __future__ import annotations

import json
import hashlib
import math
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from harnesslens.evolution.analyzer import (
    _validate_candidate,
    analyzer_candidates,
    canonicalize_analyzer_output,
)
from harnesslens.core.artifacts import content_digest, write_json, write_text
from harnesslens.core.budget import CreationBudget
from harnesslens.benchmarks.cell_config import benchmark_config
from harnesslens.harnesses.channel_preflight import PREFLIGHT_CREATION_COST
from harnesslens.evolution.harness_editor import HarnessEditor
from harnesslens.harnesses.harness_workspace import (
    diff_workspace,
    extract_mcp_tool_patches,
    normalize_workspace_snapshot,
    workspace_digest,
)
from harnesslens.core.profiles import power_profile
from harnesslens.evaluation.rollout_bridge import CellHarnessRepository
from harnesslens.harnesses.runner import (
    IntelligentHarnessRunner,
    intelligent_stdout_path,
    parse_json_object,
)
from harnesslens.core.train_protocol import TRAIN_ROLLOUT_REPEATS


MIN_ROLLOUT_CREATIONS = 5 * TRAIN_ROLLOUT_REPEATS
MIN_STANDARD_ROLLOUT_TASKS = 5
MIN_RESIDUAL_ROLLOUT_TASKS = 2
DEFAULT_EXPLORATION_ROLLOUT_TASKS = 6
ITERATION_RETRY_BUFFER_CREATIONS = 3
PROMOTION_METRICS = frozenset({"pass_at_1", "pass_at_2"})
GLOBAL_INSTRUCTION_CHANNELS = frozenset(
    {
        "developer_instructions",
        "hooks",
        "instructions_rules",
        "project_instructions",
        "system_prompt",
    }
)


def iteration_creation_cost(task_count: int) -> int:
    count = int(task_count)
    if count < 5:
        raise ValueError("an iteration requires at least five tasks")
    return (
        2
        + PREFLIGHT_CREATION_COST
        + count * TRAIN_ROLLOUT_REPEATS
        + math.ceil(count / 3) * 2
        + 2
        + 1
    )


MIN_CANDIDATE_ITERATION_CREATIONS = iteration_creation_cost(5)


def normalize_promotion_metric(value: str) -> str:
    metric = str(value).strip()
    if metric not in PROMOTION_METRICS:
        raise ValueError(
            "promotion metric must be one of " + ", ".join(sorted(PROMOTION_METRICS))
        )
    return metric


def paired_screen_creation_cost(task_count: int, *, parent_version: str) -> int:
    count = int(task_count)
    if count < MIN_RESIDUAL_ROLLOUT_TASKS:
        raise ValueError("a paired screen requires at least two tasks")
    candidate_rollout = count * TRAIN_ROLLOUT_REPEATS
    parent_rollout = count * TRAIN_ROLLOUT_REPEATS if str(parent_version) != "v0" else 0
    comparison = math.ceil(count / 3) * 2
    return (
        2
        + PREFLIGHT_CREATION_COST
        + candidate_rollout
        + parent_rollout
        + comparison
        + 2
        + 1
    )


def paired_confirmation_creation_cost(task_count: int) -> int:
    count = int(task_count)
    if count < MIN_STANDARD_ROLLOUT_TASKS:
        raise ValueError("confirmation requires at least five tasks")
    rollouts = count * TRAIN_ROLLOUT_REPEATS * 2
    comparison = math.ceil(count / 3) * 2
    return rollouts + comparison + 2 + 1


def confirmation_mode() -> str:
    mode = os.environ.get("HAI_CONFIRMATION_MODE", "always").strip().lower()
    if mode not in {"always", "off"}:
        raise ValueError("HAI_CONFIRMATION_MODE must be 'always' or 'off'")
    return mode


def build_rollout_budget_plan(
    *,
    remaining_creations: int,
    parent_version: str,
    unattempted_batch_count: int,
    max_task_count: int,
) -> dict[str, Any]:
    """Pack one evidence screen against its exact promotion cost.

    When confirmation is enabled, a standard screen reserves enough budget to
    independently confirm a pass.
    A terminal screen evaluates one full-size hypothesis when a separate confirmation
    cycle is no longer affordable, but its evidence remains provisional.
    """

    remaining = max(0, int(remaining_creations))
    task_cap = max(0, int(max_task_count))

    def max_affordable(minimum: int, cost) -> int:
        if task_cap < minimum:
            return 0
        affordable = 0
        for count in range(minimum, task_cap + 1):
            if cost(count) + ITERATION_RETRY_BUFFER_CREATIONS > remaining:
                break
            affordable = count
        return affordable

    screen_cost = lambda count: paired_screen_creation_cost(
        count, parent_version=parent_version
    )
    confirmation_required = confirmation_mode() == "always"
    full_cycle_cost = lambda count: screen_cost(count) + (
        paired_confirmation_creation_cost(count) if confirmation_required else 0
    )
    confirmable_limit = max_affordable(MIN_STANDARD_ROLLOUT_TASKS, full_cycle_cost)
    minimum_next_screen = (
        screen_cost(MIN_STANDARD_ROLLOUT_TASKS) + ITERATION_RETRY_BUFFER_CREATIONS
    )

    if confirmable_limit:
        rollout_limit = min(DEFAULT_EXPLORATION_ROLLOUT_TASKS, confirmable_limit)
        reserved_future = 0
        compact_limit = min(DEFAULT_EXPLORATION_ROLLOUT_TASKS, confirmable_limit)
        compact_cycle_cost = (
            full_cycle_cost(compact_limit) + ITERATION_RETRY_BUFFER_CREATIONS
        )
        if (
            int(unattempted_batch_count) > 1
            and remaining - compact_cycle_cost >= minimum_next_screen
        ):
            rollout_limit = compact_limit
            reserved_future = minimum_next_screen
        return {
            "mode": "standard",
            "promotion_eligible": True,
            "minimum_rollout_tasks": MIN_STANDARD_ROLLOUT_TASKS,
            "rollout_task_limit": rollout_limit,
            "recommended_rollout_tasks": rollout_limit,
            "reserved_future_iteration_creations": reserved_future,
            "screen_creation_cost_at_limit": screen_cost(rollout_limit),
            "confirmation_creation_cost_at_limit": (
                paired_confirmation_creation_cost(rollout_limit)
                if confirmation_required
                else None
            ),
            "remaining_creations": remaining,
        }

    screen_limit = min(
        DEFAULT_EXPLORATION_ROLLOUT_TASKS,
        max_affordable(MIN_STANDARD_ROLLOUT_TASKS, screen_cost),
    )
    if screen_limit:
        return {
            "mode": "terminal_screen",
            "promotion_eligible": True,
            "minimum_rollout_tasks": MIN_STANDARD_ROLLOUT_TASKS,
            "rollout_task_limit": screen_limit,
            "recommended_rollout_tasks": screen_limit,
            "reserved_future_iteration_creations": 0,
            "screen_creation_cost_at_limit": screen_cost(screen_limit),
            "confirmation_creation_cost_at_limit": None,
            "remaining_creations": remaining,
        }

    return {
        "mode": "stop",
        "promotion_eligible": False,
        "minimum_rollout_tasks": 0,
        "rollout_task_limit": 0,
        "recommended_rollout_tasks": 0,
        "reserved_future_iteration_creations": 0,
        "screen_creation_cost_at_limit": None,
        "confirmation_creation_cost_at_limit": None,
        "remaining_creations": remaining,
    }


def normalize_budget_disposition(
    output: Mapping[str, Any], *, remaining_after_decision: int
) -> dict[str, Any]:
    """Return a decision with controller-owned budget fields normalized.

    The Main Agent chooses accept/reject/revise and candidate content; the
    creation ledger is owned by the controller.  Keep the budget_disposition
    contract visible in prompts, but do not let an LLM arithmetic mismatch turn
    an otherwise valid review decision into malformed output.
    """

    normalized = dict(output)
    disposition = dict(normalized.get("budget_disposition") or {})
    remaining = int(remaining_after_decision)
    disposition.update(
        {
            "remaining_creations_after_this_decision": remaining,
            "minimum_next_rollout_creations": MIN_ROLLOUT_CREATIONS,
            "rollout_possible": remaining >= MIN_ROLLOUT_CREATIONS,
            "minimum_next_iteration_creations": MIN_CANDIDATE_ITERATION_CREATIONS,
            "iteration_possible": remaining >= MIN_CANDIDATE_ITERATION_CREATIONS,
        }
    )
    normalized["budget_disposition"] = disposition
    return normalized


MAIN_AGENT_SYSTEM = """
You are the Main Agent for one cumulative harness iteration. You allocate the remaining creation
budget and select one evidence-grounded behavior problem for a separate Harness Editor; you do not
write harness files or re-analyze trajectories. Review the ranked Analyzer portfolio, public policy,
frozen task categories, evidence-to-task mapping, prior attempts, accepted history, exploration
status, and the current parent state.

Use the remaining budget to maximize expected final harness quality. At the start, identify the
small set of open behavior problems worth exploring. First test several distinct, high-potential
behavior hypotheses with the smallest valid experiments. Continue, scale up, or revise a hypothesis
only after it shows attributable positive evidence; otherwise stop that line and move to the next
promising problem. Do not spend iterations merely to balance candidate types, and reserve enough
budget to evaluate alternatives before committing to refinement.
The supplied `promotion_metric` is the experiment objective; prioritize candidates whose direct
evidence can improve that metric rather than a different repeat-derived score.

Select exactly one unattempted candidate with the best marginal evidence, pass-conversion potential,
and regression-risk tradeoff. A pending revision competes with the other open problems and is preferred
only when its parent showed attributable benefit and a localized repair has better expected value than
switching problems. A tested revision must then be accepted or rejected, never revised again. Keep
each change focused on one behavior question. A passed candidate becomes the next parent, so preserve
and regression-check previously accepted behavior.
Failure density alone is not pass-conversion potential: prefer an observed passing contrast, and do
not edit a failure-only pattern whose evidence establishes no feasible recovery action.
Treat revisions sharing an origin as repeated tests of one hypothesis. If a bounded revision does not improve attributable recoveries or reduce attributable regressions, turn to another open
problem instead of continuing cosmetic rewrites.
The controller admits only candidates with explicit baseline failure headroom. Every rollout must
include at least one supplied conversion task for the selected candidate; passing evidence and
preservation canaries alone cannot justify spending a promotion iteration.
An explicitly supplied incumbent is the current TRAIN-accepted champion and appears as the parent
manifest and accepted history. Improve it through cumulative deltas; do not reconstruct or replace
its independent artifacts unless the tested delta specifically changes them.
Use channel appropriateness as part of the decision: detailed ordered SOPs with branches and
parameters should remain skill candidates; concise global behavior can use instructions; guidance
local to one tool can use its description. Do not select an instruction merely because it is
always visible, and do not select a skill merely to create channel diversity.

Select distinct TRAIN task IDs directly, never category labels, within the range in
`rollout_budget_plan`. Use compact screens while the plan reserves another problem, and use the
recommended capacity when this is the last affordable screen and additional cases are informative.
    A terminal screen has no affordable confirmation, so use it only for an informative
    provisional result; the controller will keep the champion even if that screen is positive.
    Cover direct evidence for the chosen problem's distinct artifacts and
retain success-preservation cases for accepted behavior. The controller owns the fixed training
repeat count supplied in the input; do not specify repeats. The local success criteria must test
behavior, not reward alone.
Before choosing rollout task IDs, use the initial task categories and harness-query channel
descriptions to reflect on the changed channel's blast radius. Treat system instructions, tool
descriptions, tool parameter descriptions, MCP tool descriptions, and skill names/descriptions as
startup-visible/global enough to affect unrelated behavior. For these channels, do not validate only
on the narrow evidence tasks: broaden the rollout across target tasks, nearby categories, and a few
preservation canaries from supplied accepted/baseline evidence or unrelated task categories, or
choose/revise toward a more triggered/tool-scoped artifact. If the changed channel is truly
triggered, tool-scoped, or MCP tool-specific, focus rollout on tasks where that trigger/tool should
appear while keeping a small preservation check for nearby non-trigger tasks.
The controller will classify the selected IDs into conversion_tasks, positive_controls,
preservation_canaries, and diagnostic_tasks from frozen baseline evidence. Do not invent those
roles in your response.

Return exactly:
{"decision":"materialize_and_rollout","selected_candidate_id":"...","exploration_plan":{"open_problem_ids":["..."],"chosen_problem_id":"...","budget_strategy":"...","turning_point":"..."},"rollout_request":{"task_ids":["..."],"rationale":"...","local_success_criteria":["..."]}}
""".strip()


FINAL_MAIN_AGENT_SYSTEM = """
You are the cumulative-delta review Main Agent. You do not inspect or re-analyze trajectories. Read
the two post-rollout Analyzer decisions, rollout metrics, exact tested delta, its direct parent,
and creation budget. Accept the tested delta, keep its parent, or create one bounded revision for
the next iteration when the Analyzers have localized a failure to changed channels.

Judge the tested delta by causal evidence, not by aggregate rollout score alone. Prefer changes
whose intended local behavior is visibly recovered and whose failures are not attributable to the
changed channels. Low overall pass rates or unrelated rollout failures should reduce confidence or
motivate a smaller revision, but they are not by themselves a reason to reject a locally effective
delta. When a candidate mixes helpful, unresolved, and harmful changes, prefer the smallest
isolatable revision that keeps the helpful attributed change and drops the rest. Budget exhaustion
does not lower the acceptance standard, but unrelated unresolved failures do not raise it. Judge the
completed paired evidence for the changed behavior: accept an attributable net pass gain with no
attributable regression even when an Analyzer asks for more confidence or reports failures it
classifies as not attributed. Keep the parent when follow-up is still required for a failure
attributed to the changed channel and that follow-up is unaffordable. A possibly-related failure
does not justify spending the next iteration on a revision without stronger causal evidence.
`further_rollout_needed` is advisory rather than a veto when the completed evidence already satisfies
the attribution and preservation gates. A
controller-marked residual probe is diagnostic-only and cannot be accepted. An actual changed
channel named `workspace_file:*` was not explained by Harness Query; remove it in a revision and
retest before promotion.
The supplied `promotion_metric` is authoritative when `require_primary_metric_improvement` is true:
the candidate must strictly improve it over the paired reference. When that flag is false, apply the
attribution and preservation gates; the controller decides whether acceptance promotes immediately
or requests independent confirmation. Improvements only in another
pass@k metric are diagnostic evidence. When two or more independent trials are available, pass@1 is
estimated from all of them rather than privileging the first trial's arbitrary ordering.

Acceptance requires attributable positive evidence. An attributed `stable_success` counts when the
comparison shows that the target behavior became more consistent even though the reference already
had one successful branch, but only when the candidate also increases the recorded pass count.
Behavioral completion without a recorded-pass gain remains diagnostic evidence, not promotion
evidence. Mere preservation of behavior already present in the reference is not a
benefit and cannot justify accepting a new parent. Once an atomic candidate has attributable positive
evidence and no attributable regression, unrelated low aggregate scores are not a reason to reject it.
For a prior TRAIN-accepted incumbent, do not discard independently validated artifacts merely because
this small rollout did not exercise them. Preserve the full control when it reproduces a pass gain
without attributable regression; revise it only for a localized harmful changed channel.

An attributable regression is a reason to reject. A remaining local failure is not evidence of
regression, but must remain visible in unresolved_findings. A revision is never accepted directly:
it is a new complete candidate relative to the same parent and must receive its own rollout. Prefer
revision over terminal rejection when the evidence identifies removable or narrowable changed
channels while leaving useful, unimplicated artifacts. The revision must not expand the evidence
scope. It may add a discovered channel only when post-rollout channel usage shows that an existing
artifact was unavailable or not invoked and the added channel only makes that same artifact
reachable; it must not introduce unrelated domain logic. Distinguish whether ten rollout
trials alone are affordable from whether a
complete next iteration (selection, rollout, comparison+review, post-analysis, review) is affordable.
List a recovered task only when its post-Analyzer relation is attributed. A possibly-related recorded
pass gain is uncertainty, not promotion evidence; choose `confirm_delta` when an independent paired
rerun can resolve it and there is no attributable regression. If the post-adjustment Analyzer exposes
a distinct evidence-grounded `replan_candidate`, choose `replan_problem` to return that hypothesis to
the portfolio instead of silently moving through the stale baseline list. Do not use replan merely
because the tested hypothesis failed. Evidence task-ID sets are copied from the post-Analyzers;
do not reinterpret their preservation semantics.
The reusable preservation Analyzer owns the regression gate: if its
`preservation.attributable_regressions` list is nonempty, `accept_delta` is forbidden even when the
needs-adjustment Analyzer disagrees. Analyzer disagreement is uncertainty to resolve through a
bounded revision or rejection, never permission to erase a preservation regression. A revision may
remove, narrow, or rewrite implicated channels to address the observed failure while retaining other
useful evidence-grounded content. Choosing `revise_delta` adds one bounded revision to the open
portfolio; it does not reserve the next iteration. A tested revision must be accepted or rejected,
never revised again. Choose `reject_delta` when the evidence instead warrants switching to another behavior problem. Do
not revise a changed channel to chase failures that both Analyzers classify as non-attributed or
unresolved.
For `revise_delta`, include an Editor brief with id, objective, channel_plan, and
validation.local_behavior_checks. Describe the bounded repair; do not write harness files or a
manifest_delta. The next iteration's Harness Editor owns concrete native changes. Omit
revision_candidate for accept/reject.

Return exactly:
{"decision":"accept_delta|reject_delta|revise_delta|confirm_delta|replan_problem","selected_version":"tested child or direct parent","rationale":"...","evidence":{"recovered_task_ids":["..."],"uncertain_recovery_task_ids":["..."],"preserved_task_ids":["..."],"attributable_regression_task_ids":["..."],"unresolved_findings":["..."]},"revision_candidate":{"id":"...","objective":"...","channel_plan":[{"channel_id":"...","operation":"...","experience_ids":["..."],"rationale":"..."}],"validation":{"local_behavior_checks":["..."]}},"budget_disposition":{"remaining_creations_after_this_decision":0,"minimum_next_rollout_creations":10,"rollout_possible":false,"minimum_next_iteration_creations":19,"iteration_possible":false}}
""".strip()


@dataclass(frozen=True)
class MainDecision:
    output: Mapping[str, Any]
    output_path: str
    harness_version: str


class CandidateMaterializationError(RuntimeError):
    def __init__(self, *, candidate_id: str, reason: str):
        self.candidate_id = str(candidate_id)
        self.reason = str(reason)
        super().__init__(
            f"candidate {self.candidate_id!r} could not be materialized: {self.reason}"
        )


def materialize_selected_problem(
    *,
    repository: Any,
    editor: Any,
    parent_version: str,
    candidate_label: str,
    editor_job_id: str,
    harness_query: Mapping[str, Any],
    selected_candidate: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    base_problem = {
        key: selected_candidate[key]
        for key in ("id", "objective", "channel_plan", "validation")
        if key in selected_candidate
    }
    base_workspace = repository.read_workspace_snapshot(parent_version)
    parent_manifest = repository.read_candidate_snapshot(parent_version)
    result = None
    result_changes: Sequence[Mapping[str, Any]] = ()
    workspace: dict[str, Any] | None = None
    mcp_tool_patches: dict[str, Any] | None = None
    retry_context = ""
    channel_diffs: list[dict[str, Any]] | None = None
    for attempt in range(2):
        problem = dict(base_problem)
        if retry_context:
            problem["previous_materialization_error"] = retry_context
        try:
            result = editor.edit(
                job_id=(
                    editor_job_id
                    if attempt == 0
                    else f"{editor_job_id}-repair-{attempt + 1:02d}"
                ),
                base_workspace=base_workspace,
                harness_query=harness_query,
                problem=problem,
                evidence=[dict(item) for item in evidence],
                current_manifest=parent_manifest,
            )
            compiled_snapshot = _compile_declared_config_channels(
                snapshot=result.snapshot,
                base_workspace=base_workspace,
                harness_query=harness_query,
                selected_candidate=selected_candidate,
            )
            _validate_opencode_instruction_workspace(compiled_snapshot)
            result_changes = [
                change
                for change in diff_workspace(base_workspace, compiled_snapshot)
                if not (
                    str(change.get("scope") or "") == "project"
                    and str(change.get("path") or "")
                    == ".harness-autoiter/mcp-tool-patches.json"
                )
            ]
            workspace, mcp_tool_patches = extract_mcp_tool_patches(compiled_snapshot)
            _validate_mcp_tool_patches_against_query(mcp_tool_patches, harness_query)
            channel_diffs = actual_editor_channel_diffs(
                harness_query=harness_query,
                selected_candidate=selected_candidate,
                workspace_changes=result_changes,
                mcp_tool_patches=mcp_tool_patches,
                base_mcp_tool_patches=(
                    parent_manifest.get("tool_desc_patches") or {}
                    if isinstance(parent_manifest, Mapping)
                    else {}
                ),
                base_workspace=base_workspace,
                candidate_workspace=compiled_snapshot,
            )
            declared_channel_ids = {
                str(item.get("channel_id") or "")
                for item in selected_candidate.get("channel_plan") or []
                if isinstance(item, Mapping) and item.get("channel_id")
            }
            actual_channel_ids = {
                str(item.get("channel_id") or "") for item in channel_diffs
            }
            validate_candidate_channel_combination(actual_channel_ids)
            if declared_channel_ids and declared_channel_ids.isdisjoint(
                actual_channel_ids
            ):
                raise ValueError(
                    "Harness Editor did not implement any declared channel; actual changes="
                    + ",".join(sorted(actual_channel_ids))
                )
            unclassified_channel_ids = {
                channel_id
                for channel_id in actual_channel_ids
                if channel_id.startswith("workspace_file:")
            }
            if declared_channel_ids and unclassified_channel_ids:
                raise ValueError(
                    "Harness Editor must remove changes outside discovered channels: "
                    + ",".join(sorted(unclassified_channel_ids))
                )
            break
        except (RuntimeError, ValueError) as exc:
            if attempt:
                raise
            retry_context = str(exc)
    assert (
        result is not None
        and workspace is not None
        and mcp_tool_patches is not None
        and channel_diffs is not None
    )
    version = repository.materialize_workspace_candidate(
        base_version=parent_version,
        candidate_label=candidate_label,
        workspace=workspace,
        manifest_delta=(
            {"tool_desc_patches": mcp_tool_patches} if mcp_tool_patches else None
        ),
    )
    manifest_delta = {"tool_desc_patches": mcp_tool_patches} if mcp_tool_patches else {}
    return version, {
        "workspace_diff": [dict(item) for item in result_changes],
        "workspace_delta": _editor_workspace_delta(workspace, result_changes),
        "manifest_delta": manifest_delta,
        "channel_diffs": channel_diffs,
        "workspace_sha256": workspace_digest(workspace),
        "mcp_tool_patch_count": len(mcp_tool_patches),
        "editor_summary": dict(result.summary) if result.summary is not None else None,
        "editor_root": str(result.root),
        "editor_stdout_path": str(result.stdout_path),
        "editor_stderr_path": str(result.stderr_path),
        "editor_api_trace_path": str(result.api_trace_path),
    }


def _compile_declared_config_channels(
    *,
    snapshot: Mapping[str, Any],
    base_workspace: Mapping[str, Any] | None,
    harness_query: Mapping[str, Any],
    selected_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile Editor config output into the native channel's typed artifact form."""

    normalized = normalize_workspace_snapshot(snapshot)
    files = {
        (str(item["scope"]), str(item["path"])): dict(item)
        for item in normalized["files"]
    }
    declared = {
        str(plan.get("channel_id") or "")
        for plan in selected_candidate.get("channel_plan") or []
        if isinstance(plan, Mapping)
    }
    modules = {
        str(module.get("id") or ""): module
        for module in harness_query.get("modifiable_modules") or []
        if isinstance(module, Mapping)
        and isinstance(module.get("edit_contract"), Mapping)
    }
    for channel_id in sorted(declared):
        module = modules.get(channel_id)
        if module is None:
            continue
        contract = module["edit_contract"]
        if str(contract.get("mechanism") or "") != "config":
            continue
        selector = str(contract.get("key") or "").strip()
        if not selector or "." in selector:
            continue
        key = (str(contract.get("scope") or ""), str(contract.get("path") or ""))
        item = files.get(key)
        if item is None:
            continue
        parsed = _parse_candidate_config(key[1], str(item["content"]))
        if (
            channel_id == "instructions_rules"
            and key == ("project", "opencode.json")
            and selector == "instructions"
            and isinstance(parsed, Mapping)
        ):
            instruction_entries = parsed.get("instructions")
            if isinstance(instruction_entries, list):
                compiled_entries: list[Any] = []
                changed = False
                for entry in instruction_entries:
                    if not isinstance(entry, str) or _looks_like_instruction_path(entry):
                        compiled_entries.append(entry)
                        continue
                    content = entry.strip() + "\n"
                    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
                    instruction_path = (
                        f".opencode/instructions/autoiter-{digest}.md"
                    )
                    files[("project", instruction_path)] = {
                        "scope": "project",
                        "path": instruction_path,
                        "content": content,
                        "executable": False,
                    }
                    compiled_entries.append(instruction_path)
                    changed = True
                if changed:
                    compiled_config = dict(parsed)
                    compiled_config["instructions"] = compiled_entries
                    item["content"] = (
                        json.dumps(
                            compiled_config,
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    files[key] = item
            continue
        if not key[1].endswith(".toml"):
            continue
        if parsed is None or _config_selector_value(parsed, contract) is not None:
            continue
        misplaced = _nested_config_values(parsed, selector)
        if len(misplaced) != 1 or not isinstance(misplaced[0], str):
            continue
        base_content = _workspace_file_content(
            base_workspace, scope=key[0], path=key[1]
        )
        if base_content and _parse_candidate_config(key[1], base_content):
            continue
        item["content"] = (
            f"{selector} = {json.dumps(misplaced[0], ensure_ascii=False)}\n"
        )
        files[key] = item
    return normalize_workspace_snapshot(
        {"schema": 1, "files": list(files.values())}
    )


def _looks_like_instruction_path(value: str) -> bool:
    text = str(value).strip()
    return bool(
        text
        and not re.search(r"\s", text)
        and (
            text.startswith((".", "/"))
            or "/" in text
            or "*" in text
            or text.endswith((".md", ".txt"))
        )
        and "\n" not in text
    )


def _nested_config_values(config: Mapping[str, Any], key: str) -> list[Any]:
    values: list[Any] = []
    for name, value in config.items():
        if str(name) == key:
            values.append(value)
        if isinstance(value, Mapping):
            values.extend(_nested_config_values(value, key))
    return values


def validate_candidate_channel_combination(channel_ids: set[str]) -> None:
    global_channels = set(channel_ids) & GLOBAL_INSTRUCTION_CHANNELS
    if len(global_channels) > 1:
        raise ValueError(
            "candidate may change at most one global instruction channel: "
            + ",".join(sorted(global_channels))
        )


def _editor_workspace_delta(
    workspace: Mapping[str, Any], changes: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    files = {
        (str(item["scope"]), str(item["path"])): item
        for item in normalize_workspace_snapshot(workspace)["files"]
    }
    delta: list[dict[str, Any]] = []
    for raw_change in changes:
        scope = str(raw_change.get("scope") or "")
        path = str(raw_change.get("path") or "")
        if scope == "project" and path == ".harness-autoiter/mcp-tool-patches.json":
            continue
        change = str(raw_change.get("change") or "")
        item: dict[str, Any] = {
            "scope": scope,
            "path": path,
            "change": change,
        }
        if change != "deleted":
            current = files.get((scope, path))
            if current is None:
                raise ValueError(
                    f"Editor workspace delta is missing changed file: {scope}/{path}"
                )
            item.update(
                {
                    "content": str(current["content"]),
                    "executable": bool(current.get("executable", False)),
                }
            )
        delta.append(item)
    return {"files": delta}


def actual_editor_channel_diffs(
    *,
    harness_query: Mapping[str, Any],
    selected_candidate: Mapping[str, Any],
    workspace_changes: Sequence[Mapping[str, Any]],
    mcp_tool_patches: Mapping[str, Any],
    base_mcp_tool_patches: Mapping[str, Any] | None = None,
    base_workspace: Mapping[str, Any] | None = None,
    candidate_workspace: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Describe the channels implemented by the Editor's captured delta."""

    declared_refs: dict[str, set[str]] = {}
    all_refs: set[str] = set()
    for plan in selected_candidate.get("channel_plan") or []:
        if not isinstance(plan, Mapping):
            continue
        channel_id = str(plan.get("channel_id") or "")
        refs = {str(item) for item in plan.get("experience_ids") or []}
        declared_refs.setdefault(channel_id, set()).update(refs)
        all_refs.update(refs)

    actual: dict[str, set[str]] = {}

    def add(channel_id: str) -> None:
        refs = declared_refs.get(channel_id) or all_refs
        actual.setdefault(channel_id, set()).update(refs)

    base_tool_patches = base_mcp_tool_patches or {}
    if any(
        isinstance(patch, Mapping)
        and "desc" in patch
        and patch.get("desc")
        != (
            base_tool_patches.get(tool_name, {}).get("desc")
            if isinstance(base_tool_patches.get(tool_name), Mapping)
            else None
        )
        for tool_name, patch in mcp_tool_patches.items()
    ):
        add("mcp_tool_description")
    if any(
        isinstance(patch, Mapping)
        and isinstance(patch.get("params"), Mapping)
        and any(
            value
            != (
                (base_tool_patches.get(tool_name, {}).get("params") or {}).get(name)
                if isinstance(base_tool_patches.get(tool_name), Mapping)
                and isinstance(
                    base_tool_patches.get(tool_name, {}).get("params"), Mapping
                )
                else None
            )
            for name, value in patch["params"].items()
        )
        for tool_name, patch in mcp_tool_patches.items()
    ):
        add("mcp_tool_parameter_description")

    modules = [
        item
        for item in harness_query.get("modifiable_modules") or []
        if isinstance(item, Mapping) and isinstance(item.get("edit_contract"), Mapping)
    ]
    for change in workspace_changes:
        scope = str(change.get("scope") or "")
        path = str(change.get("path") or "")
        if scope == "project" and path == ".harness-autoiter/mcp-tool-patches.json":
            continue
        if (
            "instructions_rules" in declared_refs
            and scope == "project"
            and _workspace_is_referenced_opencode_instruction(
                candidate_workspace, path=path
            )
        ):
            add("instructions_rules")
            continue
        matching_modules = []
        for module in modules:
            contract = module["edit_contract"]
            if str(contract.get("scope") or "") != scope:
                continue
            if _workspace_path_matches(
                str(contract.get("path") or ""), path, scope=scope
            ):
                matching_modules.append(module)
        matched_modules = _modules_matching_actual_config_change(
            matching_modules,
            scope=scope,
            path=path,
            base_workspace=base_workspace,
            candidate_workspace=candidate_workspace,
        )
        for module in matched_modules:
            add(str(module.get("id") or ""))
        if not matched_modules:
            add(f"workspace_file:{scope}:{path}")

    return [
        {"channel_id": channel_id, "experience_ids": sorted(refs)}
        for channel_id, refs in sorted(actual.items())
        if channel_id
    ]


def _workspace_is_referenced_opencode_instruction(
    workspace: Mapping[str, Any] | None,
    *,
    path: str,
) -> bool:
    if not path or path == "opencode.json":
        return False
    content = _workspace_file_content(
        workspace, scope="project", path="opencode.json"
    )
    config = _parse_candidate_config("opencode.json", content)
    return bool(
        isinstance(config, Mapping)
        and path in {
            str(item)
            for item in config.get("instructions") or []
            if isinstance(item, str)
        }
    )


def _validate_opencode_instruction_workspace(
    workspace: Mapping[str, Any],
) -> None:
    content = _workspace_file_content(
        workspace, scope="project", path="opencode.json"
    )
    if not content:
        return
    config = _parse_candidate_config("opencode.json", content)
    if not isinstance(config, Mapping) or "instructions" not in config:
        return
    instructions = config.get("instructions")
    if not isinstance(instructions, list):
        raise ValueError(
            "OpenCode instructions must be a list of captured project file paths"
        )
    project_paths = {
        str(item.get("path") or "")
        for item in normalize_workspace_snapshot(workspace)["files"]
        if str(item.get("scope") or "") == "project"
    }
    for raw_path in instructions:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(
                "OpenCode instructions must contain non-empty project file paths"
            )
        path = raw_path.strip()
        candidate_path = Path(path)
        if (
            candidate_path.is_absolute()
            or "://" in path
            or ".." in candidate_path.parts
        ):
            raise ValueError(
                "OpenCode instruction paths must stay inside the project workspace"
            )
        if path not in project_paths:
            raise ValueError(
                f"OpenCode instruction path is not captured in the candidate workspace: {path}"
            )


def _modules_matching_actual_config_change(
    modules: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    path: str,
    base_workspace: Mapping[str, Any] | None,
    candidate_workspace: Mapping[str, Any] | None,
) -> list[Mapping[str, Any]]:
    if not modules:
        return []
    selected = [
        module
        for module in modules
        if str((module.get("edit_contract") or {}).get("key") or "").strip()
        or str((module.get("edit_contract") or {}).get("key_prefix") or "").strip()
    ]
    if not selected or candidate_workspace is None:
        return list(modules)
    before = _workspace_file_content(base_workspace, scope=scope, path=path)
    after = _workspace_file_content(candidate_workspace, scope=scope, path=path)
    before_config = _parse_candidate_config(path, before)
    after_config = _parse_candidate_config(path, after)
    if before_config is None or after_config is None:
        return []
    return [
        module
        for module in selected
        if _config_selector_value(before_config, module["edit_contract"])
        != _config_selector_value(after_config, module["edit_contract"])
    ]


def _workspace_file_content(
    snapshot: Mapping[str, Any] | None, *, scope: str, path: str
) -> str | None:
    for item in normalize_workspace_snapshot(snapshot)["files"]:
        if item["scope"] == scope and item["path"] == path:
            return str(item["content"])
    return None


def _parse_candidate_config(path: str, content: str | None) -> Mapping[str, Any] | None:
    if content is None:
        return {}
    try:
        if path.endswith(".toml"):
            parsed = tomllib.loads(content)
        elif path.endswith(".json"):
            parsed = json.loads(content)
        else:
            return None
    except (ValueError, tomllib.TOMLDecodeError):
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _config_selector_value(
    config: Mapping[str, Any], contract: Mapping[str, Any]
) -> Any:
    selector = str(contract.get("key") or contract.get("key_prefix") or "").strip()
    current: Any = config
    for part in selector.rstrip(".").split(".") if selector else ():
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _workspace_path_matches(pattern: str, path: str, *, scope: str) -> bool:
    candidate = str(pattern).strip().removeprefix(f"{scope}/")
    if not candidate:
        return False
    pieces: list[str] = []
    cursor = 0
    for match in re.finditer(r"<[^/<>]+>", candidate):
        pieces.append(re.escape(candidate[cursor : match.start()]))
        pieces.append(r"[^/]+")
        cursor = match.end()
    pieces.append(re.escape(candidate[cursor:]))
    return re.fullmatch("".join(pieces), path) is not None


def _validate_mcp_tool_patches_against_query(
    patches: Mapping[str, Any], harness_query: Mapping[str, Any]
) -> None:
    if not patches:
        return
    points = {
        str(item.get("id") or ""): item
        for item in harness_query.get("mcp_editable_points") or []
        if isinstance(item, Mapping)
    }
    description_point = points.get("mcp_tool_description")
    parameter_point = points.get("mcp_tool_parameter_description")
    if description_point is None and parameter_point is None:
        return
    description_tools = {
        str(item) for item in (description_point or {}).get("targets") or []
    }
    parameter_targets = {
        str(item.get("tool") or ""): {
            str(name) for name in item.get("parameters") or []
        }
        for item in (parameter_point or {}).get("targets") or []
        if isinstance(item, Mapping)
    }
    for tool, raw_patch in patches.items():
        patch = dict(raw_patch) if isinstance(raw_patch, Mapping) else {}
        if patch.get("desc") and str(tool) not in description_tools:
            raise ValueError(f"MCP description patch targets undiscovered tool: {tool}")
        for parameter in patch.get("params") or {}:
            if str(parameter) not in parameter_targets.get(str(tool), set()):
                raise ValueError(
                    f"MCP parameter patch targets undiscovered parameter: {tool}.{parameter}"
                )


class MainAgentModule:
    def __init__(
        self,
        *,
        repo_root: str | Path,
        run_root: str | Path,
        budget: CreationBudget,
        harness: str = "opencode",
        cell: str = "retail",
        promotion_metric: str = "pass_at_1",
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.run_root = Path(run_root).resolve()
        self.budget = budget
        self.harness = harness
        self.promotion_metric = normalize_promotion_metric(promotion_metric)
        self.config = benchmark_config(self.repo_root, cell)
        self.root = self.run_root / "main_agent"
        self.root.mkdir(parents=True, exist_ok=True)

    def decide_and_materialize(
        self,
        *,
        label: str = "iteration-01",
        candidate_label: str = "v1",
        analyzer_label: str = "baseline",
        parent_version: str = "v0",
        attempted_candidate_ids: Sequence[str] = (),
        accepted_candidates: Sequence[Mapping[str, Any]] = (),
        additional_candidates: Sequence[Mapping[str, Any]] = (),
        iteration_history: Sequence[Mapping[str, Any]] = (),
    ) -> MainDecision:
        output_path = self.root / f"{label}.json"
        cached_payload = (
            json.loads(output_path.read_text(encoding="utf-8"))
            if output_path.exists()
            else None
        )
        experience_index = json.loads(
            (self.run_root / "experience" / "baseline_source_index.json").read_text(
                encoding="utf-8"
            )
        )
        evidence_to_task = {
            str(ref): str(task_id)
            for task_id, refs in experience_index["evidence_by_task"].items()
            for ref in refs
        }
        train_task_ids = tuple(str(item) for item in experience_index["task_ids"])
        analyzer = {
            "reusable": json.loads(
                (
                    self.run_root / "analyzer" / f"{analyzer_label}_reusable.json"
                ).read_text(encoding="utf-8")
            ),
            "adjustment": json.loads(
                (
                    self.run_root / "analyzer" / f"{analyzer_label}_adjustment.json"
                ).read_text(encoding="utf-8")
            ),
        }
        experience = json.loads(
            (self.run_root / "experience" / "current.json").read_text(encoding="utf-8")
        )
        passages = [*experience["reusable"], *experience["needs_adjustment"]]
        passage_by_id = {str(item["id"]): item for item in passages}
        portfolio = []
        for side in ("reusable", "adjustment"):
            for candidate in analyzer_candidates(analyzer[side]):
                refs = sorted(
                    {
                        str(experience_id)
                        for plan in candidate.get("channel_plan") or []
                        for experience_id in plan.get("experience_ids") or []
                    }
                )
                evidence_refs = sorted(
                    {
                        str(ref)
                        for experience_id in refs
                        for ref in passage_by_id.get(experience_id, {}).get(
                            "evidence_refs", []
                        )
                    }
                )
                portfolio.append(
                    {
                        "side": side,
                        "candidate": candidate,
                        "experience_ids": refs,
                        "evidence_refs": evidence_refs,
                    }
                )
        for raw_candidate in additional_candidates:
            candidate = dict(raw_candidate)
            side = str(candidate.pop("_portfolio_side", "revision"))
            direct_task_ids = sorted(
                {str(item) for item in candidate.pop("_direct_task_ids", ())}
            )
            declared_conversion_task_ids = sorted(
                {str(item) for item in candidate.pop("_conversion_task_ids", ())}
            )
            prior_train_evidence = dict(
                candidate.pop("_prior_train_evidence", {}) or {}
            )
            refs = sorted(
                {
                    str(experience_id)
                    for plan in candidate.get("channel_plan") or []
                    for experience_id in plan.get("experience_ids") or []
                }
            )
            portfolio.append(
                {
                    "side": side,
                    "candidate": candidate,
                    "experience_ids": refs,
                    "evidence_refs": sorted(
                        {
                            str(ref)
                            for experience_id in refs
                            for ref in passage_by_id.get(experience_id, {}).get(
                                "evidence_refs", []
                            )
                        }
                    ),
                    "direct_task_ids": direct_task_ids,
                    "declared_conversion_task_ids": declared_conversion_task_ids,
                    "prior_train_evidence": prior_train_evidence,
                }
            )
        candidate_ids = [str(item["candidate"]["id"]) for item in portfolio]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Analyzer portfolio candidate IDs must be globally unique")
        portfolio = prioritize_incumbent_controls(portfolio)
        evidence_outcomes = experience_index.get("outcomes") or {}
        baseline_task_outcomes = {
            str(task_id): (
                "pass"
                if refs
                and all(
                    str(evidence_outcomes.get(str(ref)) or "") == "pass"
                    for ref in refs
                )
                else "fail"
            )
            for task_id, refs in experience_index["evidence_by_task"].items()
        }
        portfolio = [
            {
                **dict(item),
                "evidence_profile": _baseline_evidence_summary(
                    evidence_refs=item.get("evidence_refs") or [],
                    evidence_to_task=evidence_to_task,
                    evidence_outcomes=evidence_outcomes,
                ),
            }
            for item in portfolio
        ]
        portfolio = promotion_headroom_portfolio(
            portfolio,
            baseline_task_outcomes=baseline_task_outcomes,
        )
        attempted = {str(item) for item in attempted_candidate_ids}
        all_unattempted = selectable_portfolio(
            portfolio, attempted_candidate_ids=attempted
        )
        available = selectable_portfolio(
            portfolio,
            attempted_candidate_ids=attempted,
            pending_revision_candidate_ids=[
                str(item.get("id") or "")
                for item in additional_candidates
                if str(item.get("_portfolio_side", "revision")) == "revision"
            ],
        )
        available = defer_failure_only_portfolio(available)
        if not available:
            raise RuntimeError("Analyzer portfolio has no unattempted candidate")
        if cached_payload is not None:
            validate_cached_main_decision(
                cached_payload,
                parent_version=parent_version,
                candidate_label=candidate_label,
                available=available,
            )
            return MainDecision(
                cached_payload,
                str(output_path),
                str(cached_payload["harness_version"]),
            )
        relevant_ids = {
            experience_id
            for item in available
            for experience_id in item["experience_ids"]
        }
        relevant_experiences = [
            item for item in passages if str(item["id"]) in relevant_ids
        ]
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
        repository = CellHarnessRepository(
            cell=self.config.cell,
            repo_root=self.repo_root,
            run_id=self.run_root.name,
            evidence_root=self.run_root / "rollout_evidence",
            harness=self.harness,
        )
        parent_manifest = repository.read_candidate_snapshot(parent_version)
        budget_remaining = self.budget.status()["remaining"]
        rollout_budget_plan = build_rollout_budget_plan(
            remaining_creations=budget_remaining,
            parent_version=parent_version,
            unattempted_batch_count=len(all_unattempted),
            max_task_count=len(train_task_ids),
        )
        rollout_task_limit = int(rollout_budget_plan["rollout_task_limit"])
        minimum_rollout_tasks = int(rollout_budget_plan["minimum_rollout_tasks"])
        if str(rollout_budget_plan["mode"]) == "stop":
            raise RuntimeError(
                "creation budget cannot fund a complete cumulative iteration"
            )
        exploration_status = build_exploration_status(
            available=available,
            iteration_history=iteration_history,
            evidence_to_task=evidence_to_task,
            evidence_outcomes=evidence_outcomes,
            budget_status=self.budget.status(),
            rollout_task_limit=rollout_task_limit,
            rollout_budget_plan=rollout_budget_plan,
        )
        candidate_evidence = {
            str(item["candidate"]["id"]): item["evidence_refs"] for item in available
        }
        candidate_sides = {
            str(item["candidate"]["id"]): str(item["side"]) for item in available
        }
        candidate_direct_tasks = {
            str(item["candidate"]["id"]): item.get("direct_task_ids") or []
            for item in available
        }
        candidate_conversion_tasks = {
            str(item["candidate"]["id"]): item.get("conversion_task_ids") or []
            for item in available
        }

        def validated_decision(raw: Mapping[str, Any]) -> Mapping[str, Any]:
            normalized = canonicalize_main_decision(raw)
            selected_id = str(normalized.get("selected_candidate_id") or "")
            request = normalized.get("rollout_request")
            if isinstance(normalized, dict) and isinstance(request, Mapping):
                direct_tasks = {
                    str(task_id)
                    for task_id in candidate_direct_tasks.get(selected_id, ())
                }
                direct_tasks.update(
                    str(evidence_to_task[ref])
                    for ref in candidate_evidence.get(selected_id, ())
                    if ref in evidence_to_task
                )
                normalized["rollout_request"] = {
                    **dict(request),
                    "task_roles": assign_rollout_roles(
                        task_ids=request.get("task_ids") or [],
                        conversion_task_ids=candidate_conversion_tasks.get(
                            selected_id, ()
                        ),
                        direct_task_ids=direct_tasks,
                        baseline_task_outcomes=baseline_task_outcomes,
                    ),
                }
            validate_main_decision(
                normalized,
                train_task_ids=train_task_ids,
                candidate_evidence=candidate_evidence,
                candidate_sides=candidate_sides,
                candidate_direct_tasks=candidate_direct_tasks,
                candidate_conversion_tasks=candidate_conversion_tasks,
                evidence_to_task=evidence_to_task,
                baseline_task_outcomes=baseline_task_outcomes,
                min_rollout_tasks=minimum_rollout_tasks,
                max_rollout_tasks=rollout_task_limit,
            )
            return normalized

        decision_output: Mapping[str, Any] | None = None
        base_job_id = f"main-agent-{label}"
        for workspace in sorted(
            (self.run_root / "intelligent_jobs").glob(f"{base_job_id}*")
        ):
            stdout_path = intelligent_stdout_path(workspace, self.harness)
            if not stdout_path.is_file():
                continue
            try:
                decision_output = validated_decision(
                    dict(
                        parse_json_object(
                            stdout_path.read_text(encoding="utf-8", errors="replace")
                        )
                    )
                )
            except (ValueError, KeyError, TypeError):
                continue
            break

        runner = IntelligentHarnessRunner(
            profile=power_profile(self.harness, max_steps=60),
            budget=self.budget,
            workspace_root=self.run_root / "intelligent_jobs",
            timeout_s=3600,
        )
        result = None
        for attempt in range(2) if decision_output is None else ():
            result = runner.run_json(
                job_id=self.budget.next_attempt_id(f"main-agent-{label}"),
                system_prompt=MAIN_AGENT_SYSTEM,
                input_payload={
                    "candidate_portfolio": available,
                    "discovery": discovery,
                    "relevant_experiences": relevant_experiences,
                    "evidence_to_task": evidence_to_task,
                    "train_task_ids": list(train_task_ids),
                    "promotion_metric": self.promotion_metric,
                    "budget": self.budget.status(),
                    "attempted_candidate_ids": sorted(attempted),
                    "accepted_candidates": list(accepted_candidates),
                    "pending_revision_candidate_ids": [
                        str(item.get("id"))
                        for item in additional_candidates
                        if str(item.get("_portfolio_side", "revision")) == "revision"
                    ],
                    "incumbent_candidate_ids": [
                        str(item.get("id"))
                        for item in additional_candidates
                        if str(item.get("_portfolio_side", "revision")) == "incumbent"
                    ],
                    "exploration_status": exploration_status,
                    "parent_version": parent_version,
                    "parent_manifest": parent_manifest,
                    "rollout_task_limit": rollout_task_limit,
                    "minimum_rollout_tasks": minimum_rollout_tasks,
                    "rollout_budget_plan": rollout_budget_plan,
                    "training_rollout_repeats": TRAIN_ROLLOUT_REPEATS,
                    "reserved_future_iteration_creations": rollout_budget_plan[
                        "reserved_future_iteration_creations"
                    ],
                    "retry_context": (
                        "Previous output was invalid. Select one listed candidate ID and "
                        "a budget-bounded set of evidence-grounded task IDs."
                        if attempt
                        else ""
                    ),
                },
                validator=validated_decision,
            )
            if result.output is not None:
                decision_output = canonicalize_main_decision(result.output)
                break
        if decision_output is None:
            raise RuntimeError(
                f"Main Agent failed: {result.outcome if result else 'not_launched'}: "
                f"{result.validation_error if result else 'no result'}"
            )
        decision = dict(decision_output)
        selected_id = str(decision.pop("selected_candidate_id"))
        selected_record = next(
            item for item in available if str(item["candidate"]["id"]) == selected_id
        )
        selected_candidate = dict(selected_record["candidate"])
        editor = HarnessEditor(
            harness=self.harness,
            budget=self.budget,
            run_root=self.run_root / "harness_editor",
            max_steps=40,
            timeout_s=3600,
        )
        selected_evidence = [
            dict(passage_by_id[experience_id])
            for experience_id in selected_record["experience_ids"]
            if experience_id in passage_by_id
        ]
        try:
            version, edit_metadata = materialize_selected_problem(
                repository=repository,
                editor=editor,
                parent_version=parent_version,
                candidate_label=candidate_label,
                editor_job_id=self.budget.next_attempt_id(f"harness-editor-{label}"),
                harness_query=discovery["harness_query"],
                selected_candidate=selected_candidate,
                evidence=selected_evidence,
            )
        except (RuntimeError, ValueError) as exc:
            raise CandidateMaterializationError(
                candidate_id=selected_id, reason=str(exc)
            ) from exc
        decision["candidate"] = {
            "label": candidate_label,
            "parent_version": parent_version,
            "objective": str(selected_candidate.get("objective") or ""),
            "validation": dict(selected_candidate.get("validation") or {}),
            **{
                field: str(selected_candidate.get(field) or "")
                for field in (
                    "observed_terminal_failure",
                    "causal_hypothesis",
                    "intervention_point",
                    "expected_runtime_event",
                    "falsifying_observation",
                )
                if selected_candidate.get(field)
            },
            "source_candidate_ids": [selected_id],
            "origin_candidate_id": selected_candidate.get(
                "origin_candidate_id", selected_id
            ),
            "channel_diffs": [
                {
                    "channel_id": str(plan["channel_id"]),
                    "experience_ids": [
                        str(item) for item in plan.get("experience_ids") or []
                    ],
                }
                for plan in selected_candidate["channel_plan"]
            ],
            "micro_adjustments": [],
            "manifest_delta": {},
            **edit_metadata,
            "portfolio_side": selected_record["side"],
            "prior_train_evidence": dict(
                selected_record.get("prior_train_evidence") or {}
            ),
            "conversion_task_ids": list(
                selected_record.get("conversion_task_ids") or []
            ),
        }
        decision["selected_candidate_side"] = selected_record["side"]
        decision["selected_candidate_priority"] = selected_candidate.get("priority")
        decision["evaluation_mode"] = str(rollout_budget_plan["mode"])
        decision["promotion_eligible"] = bool(rollout_budget_plan["promotion_eligible"])
        decision["rollout_budget_plan"] = rollout_budget_plan
        decision["promotion_metric"] = self.promotion_metric
        decision["harness_version"] = version
        write_json(output_path, decision)
        return MainDecision(decision, str(output_path), version)

    def finalize(
        self,
        *,
        tested_main_decision: str | Path,
        post_label: str,
        rollout_output: str | Path,
        reference_rollout_metrics: Mapping[str, Any] | None = None,
        require_primary_metric_improvement: bool = True,
        label: str = "final-submission",
        base_version: str = "v0",
        publish: bool = True,
    ) -> MainDecision:
        output_path = self.root / f"{label}.json"
        dependencies = [
            Path(tested_main_decision),
            self.run_root / "analyzer" / f"{post_label}_reusable.json",
            self.run_root / "analyzer" / f"{post_label}_adjustment.json",
            Path(rollout_output),
        ]
        cache_current = output_path.exists() and all(
            path.exists() for path in dependencies
        )
        if cache_current:
            cache_current = output_path.stat().st_mtime_ns >= max(
                path.stat().st_mtime_ns for path in dependencies
            )
        if cache_current:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            disposition = dict(payload.get("budget_disposition") or {})
            disposition.pop("remaining_creations", None)
            remaining = self.budget.status()["remaining"]
            disposition.update(
                {
                    "remaining_creations_after_this_decision": remaining,
                    "minimum_next_rollout_creations": MIN_ROLLOUT_CREATIONS,
                    "rollout_possible": remaining >= MIN_ROLLOUT_CREATIONS,
                    "minimum_next_iteration_creations": (
                        MIN_CANDIDATE_ITERATION_CREATIONS
                    ),
                    "iteration_possible": (
                        remaining >= MIN_CANDIDATE_ITERATION_CREATIONS
                    ),
                }
            )
            payload["budget_disposition"] = disposition
            write_json(output_path, payload)
            if publish:
                write_json(self.run_root / "submission" / "final.json", payload)
            return MainDecision(
                payload, str(output_path), str(payload["selected_version"])
            )
        tested = json.loads(Path(tested_main_decision).read_text(encoding="utf-8"))
        reusable = json.loads(
            (self.run_root / "analyzer" / f"{post_label}_reusable.json").read_text(
                encoding="utf-8"
            )
        )
        adjustment = json.loads(
            (self.run_root / "analyzer" / f"{post_label}_adjustment.json").read_text(
                encoding="utf-8"
            )
        )
        rollout = json.loads(Path(rollout_output).read_text(encoding="utf-8"))
        harness_query = json.loads(
            (self.run_root / "discovery" / "harness_query.json").read_text(
                encoding="utf-8"
            )
        )
        revision_channel_ids = {
            str(item["id"])
            for section in ("modifiable_modules", "mcp_editable_points")
            for item in harness_query.get(section) or []
            if isinstance(item, Mapping)
            and item.get("id")
            and str(item.get("status") or "") in {"modifiable", "verified"}
        }
        candidate_version = str(tested["harness_version"])
        task_ids = tuple(str(item) for item in rollout["requested_task_ids"])
        base_job_id = f"main-agent-{label}"
        decision_output: Mapping[str, Any] | None = None
        recovered_validation_error = ""
        recovery_budget = self.budget.status()
        for workspace in sorted(
            (self.run_root / "intelligent_jobs").glob(f"{base_job_id}*")
        ):
            stdout_path = intelligent_stdout_path(workspace, self.harness)
            if not stdout_path.exists():
                continue
            try:
                parsed = parse_json_object(
                    stdout_path.read_text(encoding="utf-8", errors="replace")
                )
            except (ValueError, KeyError, TypeError):
                continue
            try:
                recovered = canonicalize_final_decision(
                    parsed,
                    candidate_version=candidate_version,
                    base_version=base_version,
                    reusable=reusable,
                    adjustment=adjustment,
                    tested_candidate=tested["candidate"],
                )
                recovered = normalize_budget_disposition(
                    recovered,
                    remaining_after_decision=recovery_budget["remaining"],
                )
                validate_final_decision(
                    recovered,
                    candidate_version=candidate_version,
                    task_ids=task_ids,
                    reusable=reusable,
                    adjustment=adjustment,
                    budget_status=recovery_budget,
                    base_version=base_version,
                    remaining_after_decision=recovery_budget["remaining"],
                    tested_candidate=tested["candidate"],
                    revision_channel_ids=revision_channel_ids,
                    rollout_metrics=rollout["metrics"],
                    reference_rollout_metrics=reference_rollout_metrics,
                    promotion_metric=self.promotion_metric,
                    require_primary_metric_improvement=(
                        require_primary_metric_improvement
                    ),
                    promotion_eligible=bool(tested.get("promotion_eligible", True)),
                )
            except (ValueError, KeyError, TypeError) as exc:
                recovered_validation_error = str(exc)
                continue
            decision_output = recovered
            break
        runner = IntelligentHarnessRunner(
            profile=power_profile(self.harness, max_steps=60),
            budget=self.budget,
            workspace_root=self.run_root / "intelligent_jobs",
            timeout_s=3600,
        )
        result = None
        for attempt in _final_review_attempts(decision_output):
            budget_status = self.budget.status()
            remaining_after_review = budget_status["remaining"] - 1
            next_iteration_budget_plans = {
                "if_parent_kept": build_rollout_budget_plan(
                    remaining_creations=remaining_after_review,
                    parent_version=base_version,
                    unattempted_batch_count=2,
                    max_task_count=len(self.config.train_task_ids),
                ),
                "if_candidate_promoted": build_rollout_budget_plan(
                    remaining_creations=remaining_after_review,
                    parent_version=candidate_version,
                    unattempted_batch_count=2,
                    max_task_count=len(self.config.train_task_ids),
                ),
            }
            previous_error = (
                result.validation_error
                if result is not None
                else recovered_validation_error
            )
            result = runner.run_json(
                job_id=self.budget.next_attempt_id(base_job_id),
                system_prompt=FINAL_MAIN_AGENT_SYSTEM,
                input_payload={
                    "post_analyzer": {
                        "reusable": reusable,
                        "adjustment": adjustment,
                    },
                    "tested_candidate": {
                        "version": candidate_version,
                        "parent_version": base_version,
                        "channel_diffs": tested["candidate"]["channel_diffs"],
                        "manifest_delta": tested["candidate"]["manifest_delta"],
                        "workspace_diff": tested["candidate"].get("workspace_diff", []),
                        "workspace_delta": tested["candidate"].get(
                            "workspace_delta", {"files": []}
                        ),
                        "workspace_sha256": tested["candidate"].get(
                            "workspace_sha256", ""
                        ),
                        "editor_summary": tested["candidate"].get("editor_summary"),
                        "portfolio_side": tested["candidate"].get(
                            "portfolio_side", tested.get("selected_candidate_side")
                        ),
                        "prior_train_evidence": dict(
                            tested["candidate"].get("prior_train_evidence") or {}
                        ),
                        "evaluation_mode": str(
                            tested.get("evaluation_mode") or "standard"
                        ),
                        "promotion_eligible": bool(
                            tested.get("promotion_eligible", True)
                        ),
                    },
                    "rollout_metrics": rollout["metrics"],
                    "reference_rollout_metrics": dict(reference_rollout_metrics or {}),
                    "promotion_metric": self.promotion_metric,
                    "promotion_metric_values": {
                        "candidate": _required_metric(
                            rollout["metrics"],
                            self.promotion_metric,
                            side="candidate",
                        ),
                        "reference": _required_metric(
                            reference_rollout_metrics,
                            self.promotion_metric,
                            side="reference",
                        ),
                    },
                    "require_primary_metric_improvement": (
                        require_primary_metric_improvement
                    ),
                    "rolled_out_task_ids": list(task_ids),
                    "budget": budget_status,
                    "remaining_after_this_main_creation": remaining_after_review,
                    "next_iteration_budget_plans": next_iteration_budget_plans,
                    "discovered_revision_channel_ids": sorted(revision_channel_ids),
                    "retry_context": (
                        "Previous output failed validation: "
                        + previous_error
                        + " Follow the preservation regression gate and do not invent an "
                        "untested accepted version. selected_version must be exactly "
                        f"{candidate_version!r} for accept_delta or {base_version!r} for "
                        "reject_delta/revise_delta. If you return revise_delta, every "
                        "revision_candidate.channel_plan[].experience_ids item must be copied "
                        "from the tested candidate's channel_diffs only: "
                        + ", ".join(
                            sorted(
                                {
                                    str(item)
                                    for plan in tested["candidate"].get("channel_diffs")
                                    or []
                                    for item in plan.get("experience_ids") or []
                                }
                            )
                        )
                        + ". Do not introduce new experience IDs."
                        if previous_error
                        else ""
                    ),
                },
                validator=lambda output: validate_final_decision(
                    normalize_budget_disposition(
                        canonicalize_final_decision(
                            output,
                            candidate_version=candidate_version,
                            base_version=base_version,
                            reusable=reusable,
                            adjustment=adjustment,
                            tested_candidate=tested["candidate"],
                        ),
                        remaining_after_decision=budget_status["remaining"] - 1,
                    ),
                    candidate_version=candidate_version,
                    task_ids=task_ids,
                    reusable=reusable,
                    adjustment=adjustment,
                    budget_status=budget_status,
                    base_version=base_version,
                    tested_candidate=tested["candidate"],
                    revision_channel_ids=revision_channel_ids,
                    rollout_metrics=rollout["metrics"],
                    reference_rollout_metrics=reference_rollout_metrics,
                    promotion_metric=self.promotion_metric,
                    require_primary_metric_improvement=(
                        require_primary_metric_improvement
                    ),
                    promotion_eligible=bool(tested.get("promotion_eligible", True)),
                ),
            )
            if result.output is not None:
                decision_output = normalize_budget_disposition(
                    canonicalize_final_decision(
                        result.output,
                        candidate_version=candidate_version,
                        base_version=base_version,
                        reusable=reusable,
                        adjustment=adjustment,
                        tested_candidate=tested["candidate"],
                    ),
                    remaining_after_decision=self.budget.status()["remaining"],
                )
                break
        if decision_output is None:
            failure_reason = (
                recovered_validation_error
                or (result.validation_error if result is not None else "")
                or (result.outcome if result is not None else "malformed output")
            )
            decision_output = deterministic_reject_decision(
                base_version=base_version,
                reusable=reusable,
                adjustment=adjustment,
                remaining=self.budget.status()["remaining"],
                failure_reason=failure_reason,
            )
        decision = dict(decision_output)
        decision["tested_candidate_version"] = candidate_version
        decision["rollout_output"] = str(Path(rollout_output).resolve())
        selected = str(decision["selected_version"])
        submission_root = self.run_root / "submission"
        track_root = (
            self.run_root
            / "rollout_evidence"
            / self.run_root.name
            / "versions_percell"
            / self.config.cell
        )
        snapshot = track_root / selected
        patch_path = _snapshot_manifest_path(snapshot, self.harness)
        patch = (
            json.loads(patch_path.read_text(encoding="utf-8"))
            if patch_path.exists()
            else {}
        )
        workspace_path = _snapshot_workspace_path(snapshot, self.harness)
        workspace = (
            json.loads(workspace_path.read_text(encoding="utf-8"))
            if workspace_path.exists()
            else {"schema": 1, "files": []}
        )
        decision["snapshot_path"] = str(snapshot.resolve())
        decision["patch_digest"] = content_digest(patch)
        decision["workspace_digest"] = workspace_digest(workspace)
        write_json(output_path, decision)
        if publish:
            write_json(submission_root / "final.json", decision)
        if selected == candidate_version:
            meta_path = snapshot / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["status"] = "kept" if publish else "accepted"
            meta["temporary_candidate"] = not publish
            meta["validation"] = {
                "mode": "harnesslens_post_rollout_analyzer",
                "rollout_output": str(Path(rollout_output).resolve()),
                "post_reusable": str(
                    (
                        self.run_root / "analyzer" / f"{post_label}_reusable.json"
                    ).resolve()
                ),
                "post_adjustment": str(
                    (
                        self.run_root / "analyzer" / f"{post_label}_adjustment.json"
                    ).resolve()
                ),
                "metrics": rollout["metrics"],
            }
            write_json(meta_path, meta)
        else:
            meta_path = track_root / candidate_version / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["status"] = "rejected"
            meta["temporary_candidate"] = True
            write_json(meta_path, meta)
        if publish:
            write_text(track_root / "CURRENT_HEAD.txt", selected + "\n")
        return MainDecision(decision, str(output_path), selected)

    def publish_selected(
        self,
        *,
        selected_version: str,
        iteration_history: Sequence[Mapping[str, Any]],
        dispositions_path: str | Path,
        label: str = "final-submission",
    ) -> MainDecision:
        output_path = self.root / f"{label}.json"
        track_root = (
            self.run_root
            / "rollout_evidence"
            / self.run_root.name
            / "versions_percell"
            / self.config.cell
        )
        snapshot = track_root / str(selected_version)
        patch_path = _snapshot_manifest_path(snapshot, self.harness)
        patch = (
            json.loads(patch_path.read_text(encoding="utf-8"))
            if patch_path.exists()
            else {}
        )
        workspace_path = _snapshot_workspace_path(snapshot, self.harness)
        workspace = (
            json.loads(workspace_path.read_text(encoding="utf-8"))
            if workspace_path.exists()
            else {"schema": 1, "files": []}
        )
        payload = {
            "decision": (
                "submit_candidate" if selected_version != "v0" else "submit_baseline"
            ),
            "selected_version": str(selected_version),
            "snapshot_path": str(snapshot.resolve()),
            "patch_digest": content_digest(patch),
            "workspace_digest": workspace_digest(workspace),
            "iteration_history": list(iteration_history),
            "experience_dispositions": str(Path(dispositions_path).resolve()),
            "budget": self.budget.status(),
        }
        write_json(output_path, payload)
        write_json(self.run_root / "submission" / "final.json", payload)
        write_text(track_root / "CURRENT_HEAD.txt", str(selected_version) + "\n")
        if selected_version != "v0":
            meta_path = snapshot / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["status"] = "kept"
            meta["temporary_candidate"] = False
            write_json(meta_path, meta)
        return MainDecision(payload, str(output_path), str(selected_version))


def _final_review_attempts(decision_output: Mapping[str, Any] | None) -> range:
    return range(2) if decision_output is None else range(0)


def _snapshot_manifest_path(snapshot: Path, harness: str) -> Path:
    normalized = str(harness).strip().lower().replace("-", "_")
    if normalized == "pi_agent":
        normalized = "pi"
    return snapshot / "harness" / normalized / "manifest.json"


def _snapshot_workspace_path(snapshot: Path, harness: str) -> Path:
    normalized = str(harness).strip().lower().replace("-", "_")
    if normalized == "pi_agent":
        normalized = "pi"
    return snapshot / "harness" / normalized / "workspace.json"


def selectable_portfolio(
    portfolio: Sequence[Mapping[str, Any]],
    *,
    attempted_candidate_ids: Sequence[str],
    pending_revision_candidate_ids: Sequence[str] = (),
) -> list[Mapping[str, Any]]:
    attempted = {str(item) for item in attempted_candidate_ids}
    return [
        item
        for item in portfolio
        if str((item.get("candidate") or {}).get("id") or "") not in attempted
    ]


def defer_failure_only_portfolio(
    portfolio: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Keep failure-only hypotheses available after supported options are exhausted."""

    candidates = list(portfolio)
    preferred = [
        item
        for item in candidates
        if str(item.get("side") or "") in {"revision", "incumbent"}
        or str((item.get("evidence_profile") or {}).get("support_tier") or "")
        != "failure_only"
    ]
    return preferred or candidates


def promotion_headroom_portfolio(
    portfolio: Sequence[Mapping[str, Any]],
    *,
    baseline_task_outcomes: Mapping[str, str] | None = None,
) -> list[Mapping[str, Any]]:
    """Admit only hypotheses that can convert a recorded TRAIN failure."""

    task_outcomes = baseline_task_outcomes or {}
    admitted: list[Mapping[str, Any]] = []
    for raw_item in portfolio:
        item = dict(raw_item)
        profile = item.get("evidence_profile") or {}
        prior = item.get("prior_train_evidence") or {}
        conversion_tasks = sorted(
            {
                str(task_id)
                for task_id in profile.get("all_failed_task_ids") or []
                if str(task_id)
            }
            | {
                str(task_id)
                for task_id in prior.get("recovered_task_ids") or []
                if str(task_id)
            }
            | {
                str(task_id)
                for task_id in item.get("declared_conversion_task_ids") or []
                if str(task_id)
            }
        )
        if str(item.get("side") or "") == "reusable":
            conversion_tasks = sorted(
                set(conversion_tasks)
                | {
                    str(task_id)
                    for task_id in profile.get("all_passed_task_ids") or []
                    if str(task_outcomes.get(str(task_id)) or "") == "fail"
                }
            )
        if not conversion_tasks:
            continue
        item["conversion_task_ids"] = conversion_tasks
        admitted.append(item)
    return admitted


def assign_rollout_roles(
    *,
    task_ids: Sequence[str],
    conversion_task_ids: Sequence[str],
    direct_task_ids: Sequence[str],
    baseline_task_outcomes: Mapping[str, str],
) -> dict[str, list[str]]:
    """Assign every requested TRAIN task one controller-owned validation role."""

    requested = [str(item) for item in task_ids]
    conversions = {str(item) for item in conversion_task_ids}
    direct = {str(item) for item in direct_task_ids}
    roles = {
        "conversion_tasks": [],
        "positive_controls": [],
        "preservation_canaries": [],
        "diagnostic_tasks": [],
    }
    for task_id in requested:
        outcome = str(baseline_task_outcomes.get(task_id) or "fail")
        if task_id in conversions:
            role = "conversion_tasks"
        elif outcome == "pass" and task_id in direct:
            role = "positive_controls"
        elif outcome == "pass":
            role = "preservation_canaries"
        else:
            role = "diagnostic_tasks"
        roles[role].append(task_id)
    return roles


def validate_cached_main_decision(
    output: Mapping[str, Any],
    *,
    parent_version: str,
    candidate_label: str,
    available: Sequence[Mapping[str, Any]],
) -> None:
    candidate = output.get("candidate")
    if not isinstance(candidate, Mapping):
        raise RuntimeError("cached Main Agent decision has no candidate")
    source_ids = [str(item) for item in candidate.get("source_candidate_ids") or []]
    available_ids = {
        str((item.get("candidate") or {}).get("id") or "") for item in available
    }
    if len(source_ids) != 1 or source_ids[0] not in available_ids:
        raise RuntimeError(
            "cached Main Agent decision no longer follows the current review chain"
        )
    if str(candidate.get("parent_version") or "") != str(parent_version):
        raise RuntimeError("cached Main Agent decision has a stale parent version")
    if str(candidate.get("label") or "") != str(candidate_label):
        raise RuntimeError("cached Main Agent decision has a stale candidate label")
    if str(output.get("harness_version") or "") != str(candidate_label):
        raise RuntimeError("cached Main Agent decision has a stale harness version")


def prioritize_incumbent_controls(
    portfolio: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return sorted(
        portfolio,
        key=lambda item: 0 if str(item.get("side") or "") == "incumbent" else 1,
    )


def canonicalize_main_decision(output: Mapping[str, Any]) -> Mapping[str, Any]:
    normalized = dict(output)
    raw_plan = normalized.get("exploration_plan")
    if isinstance(raw_plan, Mapping):
        plan = dict(raw_plan)
        chosen = str(plan.get("chosen_problem_id") or "")
        open_ids = [str(item) for item in plan.get("open_problem_ids") or []]
        if chosen and chosen not in open_ids:
            plan["open_problem_ids"] = [chosen, *open_ids]
        normalized["exploration_plan"] = plan
    if isinstance(output, dict):
        output.clear()
        output.update(normalized)
        return output
    return normalized


def validate_main_decision(
    output: Mapping[str, Any],
    *,
    train_task_ids: Sequence[str],
    candidate_evidence: Mapping[str, Sequence[str]],
    candidate_direct_tasks: Mapping[str, Sequence[str]] | None = None,
    candidate_conversion_tasks: Mapping[str, Sequence[str]] | None = None,
    candidate_sides: Mapping[str, str] | None = None,
    evidence_to_task: Mapping[str, str],
    baseline_task_outcomes: Mapping[str, str] | None = None,
    min_rollout_tasks: int = MIN_STANDARD_ROLLOUT_TASKS,
    max_rollout_tasks: int | None = None,
) -> None:
    if output.get("decision") != "materialize_and_rollout":
        raise ValueError("Main Agent must materialize and request rollout")
    candidate_id = str(output.get("selected_candidate_id") or "")
    if candidate_id not in candidate_evidence:
        raise ValueError("Main Agent selected an unavailable candidate")
    plan = output.get("exploration_plan")
    if not isinstance(plan, Mapping):
        raise ValueError("Main Agent requires an exploration_plan")
    open_problem_ids = [str(item) for item in plan.get("open_problem_ids") or []]
    chosen_problem_id = str(plan.get("chosen_problem_id") or "")
    if not open_problem_ids or not chosen_problem_id:
        raise ValueError("exploration_plan requires open and chosen problem IDs")
    if chosen_problem_id not in open_problem_ids:
        raise ValueError("chosen problem must be listed as open")
    if not str(plan.get("budget_strategy") or "").strip():
        raise ValueError("exploration_plan requires a budget strategy")
    if not str(plan.get("turning_point") or "").strip():
        raise ValueError("exploration_plan requires a turning point")
    request = output.get("rollout_request")
    if not isinstance(request, Mapping):
        raise ValueError("Main Agent requires rollout_request")
    task_ids = [str(item) for item in request.get("task_ids") or []]
    minimum = int(min_rollout_tasks)
    if len(task_ids) < minimum or len(task_ids) != len(set(task_ids)):
        raise ValueError(f"rollout requires at least {minimum} distinct task IDs")
    if set(task_ids) - set(train_task_ids):
        raise ValueError("rollout selected a task outside TRAIN")
    if max_rollout_tasks is not None and len(task_ids) > int(max_rollout_tasks):
        raise ValueError("rollout request exceeds the complete-iteration task limit")
    evidence_tasks = {
        str(evidence_to_task[ref])
        for ref in candidate_evidence[candidate_id]
        if ref in evidence_to_task
    }
    evidence_tasks.update(
        str(task_id)
        for task_id in (candidate_direct_tasks or {}).get(candidate_id, ())
        if str(task_id) in set(train_task_ids)
    )
    if evidence_tasks and not (set(task_ids) & evidence_tasks):
        raise ValueError("rollout tasks lack primary-problem evidence support")
    if candidate_conversion_tasks is not None:
        conversion_tasks = {
            str(task_id)
            for task_id in candidate_conversion_tasks.get(candidate_id, ())
        }
        if not conversion_tasks or not (set(task_ids) & conversion_tasks):
            raise ValueError(
                "rollout tasks must include a declared baseline conversion task"
            )
    if baseline_task_outcomes is not None:
        direct_tasks = set(
            str(task_id)
            for task_id in (candidate_direct_tasks or {}).get(candidate_id, ())
        ) | evidence_tasks
        expected_roles = assign_rollout_roles(
            task_ids=task_ids,
            conversion_task_ids=(candidate_conversion_tasks or {}).get(
                candidate_id, ()
            ),
            direct_task_ids=direct_tasks,
            baseline_task_outcomes=baseline_task_outcomes,
        )
        roles = request.get("task_roles")
        if not isinstance(roles, Mapping) or {
            str(key): [str(item) for item in value]
            for key, value in roles.items()
            if isinstance(value, list)
        } != expected_roles:
            raise ValueError("rollout task roles differ from controller assignment")
    if not request.get("rationale") or not request.get("local_success_criteria"):
        raise ValueError(
            "rollout request requires rationale and local success criteria"
        )
    if "repeats" in request:
        raise ValueError("rollout repeats are fixed by the controller")


def build_exploration_status(
    *,
    available: Sequence[Mapping[str, Any]],
    iteration_history: Sequence[Mapping[str, Any]],
    evidence_to_task: Mapping[str, str],
    evidence_outcomes: Mapping[str, str] | None = None,
    budget_status: Mapping[str, Any],
    rollout_task_limit: int,
    rollout_budget_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compact controller state for budget-aware issue exploration."""

    outcomes = {
        str(ref): str(outcome) for ref, outcome in (evidence_outcomes or {}).items()
    }
    revision_signals = {
        str(record.get("revision_candidate_id")): {
            "attributed_recovery_count": len(
                (record.get("review_evidence") or {}).get("recovered_task_ids") or []
            ),
            "attributable_regression_count": len(
                (record.get("review_evidence") or {}).get(
                    "attributable_regression_task_ids"
                )
                or []
            ),
        }
        for record in iteration_history
        if str(record.get("revision_candidate_id") or "")
    }
    open_problems = []
    for item in available:
        candidate = item["candidate"]
        candidate_id = str(candidate["id"])
        evidence_tasks = sorted(
            {
                str(evidence_to_task[ref])
                for ref in item.get("evidence_refs") or []
                if ref in evidence_to_task
            }
            | {str(task_id) for task_id in item.get("direct_task_ids") or []}
        )
        plans = candidate.get("channel_plan") or []
        manifest = candidate.get("manifest_delta") or {}
        file_count = len(manifest.get("files") or [])
        channel_ids = sorted({str(plan.get("channel_id") or "") for plan in plans})
        side = str(item.get("side") or "")
        baseline_evidence = dict(
            item.get("evidence_profile")
            or _baseline_evidence_summary(
                evidence_refs=item.get("evidence_refs") or [],
                evidence_to_task=evidence_to_task,
                evidence_outcomes=outcomes,
            )
        )
        risk_flags = []
        if len(channel_ids) > 1:
            risk_flags.append("multi_channel_delta")
        if len(item.get("experience_ids") or []) > 4:
            risk_flags.append("large_experience_batch")
        if file_count > 1:
            risk_flags.append("multi_file_delta")
        if not evidence_tasks:
            risk_flags.append("no_direct_evidence_task")
        if baseline_evidence["support_tier"] == "failure_only":
            risk_flags.append("failure_only_without_passing_contrast")
        if (
            side == "reusable"
            and baseline_evidence["trial_count"]
            and not baseline_evidence["passed_trial_count"]
        ):
            risk_flags.append("reusable_without_passing_evidence_support")
        if (
            side == "reusable"
            and baseline_evidence["trial_count"]
            and not baseline_evidence["failed_trial_count"]
        ):
            risk_flags.append("no_observed_failure_headroom")
        revision_signal = (
            revision_signals.get(candidate_id) if side == "revision" else None
        )
        if revision_signal and not revision_signal["attributed_recovery_count"]:
            risk_flags.append("revision_without_positive_parent_evidence")
        problem = {
            "problem_id": candidate_id,
            "side": side,
            "objective": str(candidate.get("objective") or ""),
            "experience_count": len(item.get("experience_ids") or []),
            "evidence_task_ids": evidence_tasks,
            "baseline_evidence": baseline_evidence,
            "changed_channel_count": len(channel_ids),
            "changed_file_count": file_count,
            "risk_flags": risk_flags,
        }
        if item.get("prior_train_evidence"):
            problem["prior_train_evidence"] = dict(item["prior_train_evidence"])
        if revision_signal is not None:
            problem["revision_parent_signal"] = revision_signal
        open_problems.append(problem)

    recent_results = []
    for record in iteration_history[-4:]:
        evidence = record.get("review_evidence") or {}
        metrics = record.get("rollout_metrics") or {}
        recent_results.append(
            {
                "problem_id": str(
                    record.get("candidate_id")
                    or record.get("revision_candidate_id")
                    or ""
                ),
                "candidate_version": str(record.get("candidate_version") or ""),
                "review_decision": str(record.get("review_decision") or ""),
                "selected_version": str(record.get("selected_version") or ""),
                "rollout_task_count": len(record.get("rollout_task_ids") or []),
                "pass_at_1": metrics.get("pass_at_1"),
                "pass_at_2": metrics.get("pass_at_2"),
                "recovered_count": len(evidence.get("recovered_task_ids") or []),
                "attributable_regression_count": len(
                    evidence.get("attributable_regression_task_ids") or []
                ),
                "revision_candidate_id": record.get("revision_candidate_id"),
                "evaluation_mode": str(record.get("evaluation_mode") or "standard"),
                "promotion_eligible": bool(record.get("promotion_eligible", True)),
            }
        )
    hypothesis_attempts: dict[str, list[dict[str, int]]] = {}
    for record in iteration_history:
        origin = str(
            record.get("origin_candidate_id") or record.get("candidate_id") or ""
        )
        if not origin:
            continue
        evidence = record.get("review_evidence") or {}
        hypothesis_attempts.setdefault(origin, []).append(
            {
                "recovered_count": len(evidence.get("recovered_task_ids") or []),
                "regression_count": len(
                    evidence.get("attributable_regression_task_ids") or []
                ),
            }
        )
    hypothesis_history = []
    for origin, attempts in hypothesis_attempts.items():
        latest = attempts[-1]
        previous = attempts[-2] if len(attempts) > 1 else None
        hypothesis_history.append(
            {
                "origin_candidate_id": origin,
                "attempt_count": len(attempts),
                "latest_recovered_count": latest["recovered_count"],
                "latest_attributable_regression_count": latest["regression_count"],
                "latest_revision_improved": (
                    None
                    if previous is None
                    else (
                        latest["recovered_count"] > previous["recovered_count"]
                        or latest["regression_count"] < previous["regression_count"]
                    )
                ),
            }
        )
    open_side_counts = {
        side: sum(1 for item in open_problems if item["side"] == side)
        for side in ("reusable", "adjustment", "revision", "incumbent")
    }
    remaining = int(budget_status.get("remaining") or 0)
    legacy_budget_summary = {
        "remaining": remaining,
        "rollout_task_limit": int(rollout_task_limit),
        "minimum_complete_iteration_creations": MIN_CANDIDATE_ITERATION_CREATIONS,
        "compact_iteration_creations": iteration_creation_cost(
            DEFAULT_EXPLORATION_ROLLOUT_TASKS
        ),
        "affordable_minimum_iterations": max(
            0, remaining - ITERATION_RETRY_BUFFER_CREATIONS
        )
        // MIN_CANDIDATE_ITERATION_CREATIONS,
        "affordable_compact_iterations": (
            max(0, remaining - ITERATION_RETRY_BUFFER_CREATIONS)
            // iteration_creation_cost(DEFAULT_EXPLORATION_ROLLOUT_TASKS)
        ),
        "small_batch_default_task_limit": DEFAULT_EXPLORATION_ROLLOUT_TASKS,
    }
    budget_summary = (
        {"remaining": remaining, **dict(rollout_budget_plan)}
        if rollout_budget_plan is not None
        else legacy_budget_summary
    )
    return {
        "budget": budget_summary,
        "policy": {
            "prefer_small_batches": len(open_problems) > 1,
            "switch_problem_when": [
                "the candidate has attributable preservation regressions",
                "the rollout shows no attributed recovery for the chosen problem",
                "a narrow revision is unaffordable or would expand scope",
                "the latest comparable revision does not improve recovery or regression evidence",
            ],
            "scale_up_when": [
                "a small rollout shows attributed recovery",
                "preservation checks remain stable",
                "remaining budget still covers final review or another complete iteration",
            ],
        },
        "candidate_type_counts": open_side_counts,
        "open_problems": open_problems,
        "recent_results": recent_results,
        "hypothesis_history": hypothesis_history,
    }


def _baseline_evidence_summary(
    *,
    evidence_refs: Sequence[str],
    evidence_to_task: Mapping[str, str],
    evidence_outcomes: Mapping[str, str],
) -> dict[str, Any]:
    by_task: dict[str, list[str]] = {}
    for raw_ref in evidence_refs:
        ref = str(raw_ref)
        task_id = evidence_to_task.get(ref)
        outcome = evidence_outcomes.get(ref)
        if task_id is None or outcome not in {"pass", "fail"}:
            continue
        by_task.setdefault(str(task_id), []).append(str(outcome))
    flat = [outcome for values in by_task.values() for outcome in values]
    failed = sum(outcome == "fail" for outcome in flat)
    mixed_task_ids = sorted(
        task_id
        for task_id, values in by_task.items()
        if "pass" in values and "fail" in values
    )
    passed = len(flat) - failed
    support_tier = (
        "within_task_contrast"
        if mixed_task_ids
        else (
            "mixed_evidence"
            if passed and failed
            else "success_only" if passed else "failure_only" if failed else "unknown"
        )
    )
    return {
        "trial_count": len(flat),
        "failed_trial_count": failed,
        "passed_trial_count": passed,
        "task_count": len(by_task),
        "support_tier": support_tier,
        "failure_density": (failed / len(flat) if flat else None),
        "all_failed_task_ids": sorted(
            task_id
            for task_id, values in by_task.items()
            if values and all(value == "fail" for value in values)
        ),
        "mixed_task_ids": mixed_task_ids,
        "all_passed_task_ids": sorted(
            task_id
            for task_id, values in by_task.items()
            if values and all(value == "pass" for value in values)
        ),
    }


def validate_final_decision(
    output: Mapping[str, Any],
    *,
    candidate_version: str,
    task_ids: Sequence[str],
    reusable: Mapping[str, Any],
    adjustment: Mapping[str, Any],
    budget_status: Mapping[str, Any],
    base_version: str = "v0",
    remaining_after_decision: int | None = None,
    tested_candidate: Mapping[str, Any] | None = None,
    revision_channel_ids: set[str] | None = None,
    rollout_metrics: Mapping[str, Any] | None = None,
    reference_rollout_metrics: Mapping[str, Any] | None = None,
    promotion_metric: str | None = None,
    require_primary_metric_improvement: bool = True,
    promotion_eligible: bool = True,
) -> None:
    decision = str(output.get("decision") or "")
    selected = str(output.get("selected_version") or "")
    expected = candidate_version if decision == "accept_delta" else base_version
    if (
        decision
        not in {
            "accept_delta",
            "reject_delta",
            "revise_delta",
            "confirm_delta",
            "replan_problem",
        }
        or selected != expected
    ):
        raise ValueError(
            "review Main Agent must choose the tested candidate or its parent"
        )
    if decision == "accept_delta" and not promotion_eligible:
        raise ValueError("a residual probe is not promotion eligible")
    unclassified_channels = {
        str(item.get("channel_id") or "")
        for item in (tested_candidate or {}).get("channel_diffs") or []
        if str(item.get("channel_id") or "").startswith("workspace_file:")
    }
    if decision == "accept_delta" and unclassified_channels:
        raise ValueError(
            "candidate acceptance requires removing unclassified workspace changes"
        )
    if not str(output.get("rationale") or "").strip():
        raise ValueError("final Main Agent requires a rationale")
    evidence = output.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("final Main Agent requires evidence")
    recovered = {str(item) for item in evidence.get("recovered_task_ids") or []}
    preserved = {str(item) for item in evidence.get("preserved_task_ids") or []}
    regressions = {
        str(item) for item in evidence.get("attributable_regression_task_ids") or []
    }
    if (recovered | preserved | regressions) - set(str(item) for item in task_ids):
        raise ValueError("final evidence references an unrolled task")
    expected_recovered = _attributable_positive_task_ids(adjustment)
    expected_uncertain = uncertain_recovery_task_ids(adjustment)
    expected_preserved = {
        str(item) for item in reusable["preservation"]["preserved_task_ids"]
    }
    expected_regressions = {
        str(item["task_id"])
        for item in reusable["preservation"]["attributable_regressions"]
    }
    if (
        recovered != expected_recovered
        or preserved != expected_preserved
        or regressions != expected_regressions
    ):
        raise ValueError("final evidence differs from reusable Analyzer")
    if decision == "accept_delta" and expected_regressions:
        raise ValueError("candidate with attributable regressions cannot be submitted")
    if decision == "accept_delta" and not expected_recovered:
        raise ValueError("candidate acceptance requires attributable positive evidence")
    uncertain = {
        str(item) for item in evidence.get("uncertain_recovery_task_ids") or []
    }
    if uncertain != expected_uncertain:
        raise ValueError("final uncertain recovery evidence differs from Analyzer")
    if (
        decision == "accept_delta"
        and promotion_metric is not None
        and require_primary_metric_improvement
    ):
        if (rollout_metrics or {}).get("paired_infrastructure_valid") is False:
            raise ValueError(
                "candidate acceptance requires infrastructure-valid paired evidence"
            )
        metric = normalize_promotion_metric(promotion_metric)
        candidate_value = _required_metric(rollout_metrics, metric, side="candidate")
        reference_value = _required_metric(
            reference_rollout_metrics, metric, side="reference"
        )
        if candidate_value <= reference_value:
            raise ValueError(f"candidate acceptance requires positive {metric} delta")
    unresolved = evidence.get("unresolved_findings")
    if not isinstance(unresolved, list):
        raise ValueError("final decision must preserve unresolved findings")
    has_revision = decision == "revise_delta" and isinstance(
        output.get("revision_candidate"), Mapping
    )
    has_replan = decision == "replan_problem" and isinstance(
        adjustment.get("replan_candidate"), Mapping
    )
    if (
        adjustment["primary_problem"]["recommendation"] == "refine"
        and not unresolved
        and not (has_revision or has_replan)
    ):
        raise ValueError("final decision dropped a requested refinement")
    disposition = output.get("budget_disposition")
    if not isinstance(disposition, Mapping):
        raise ValueError("final Main Agent requires budget disposition")
    remaining_after = (
        int(remaining_after_decision)
        if remaining_after_decision is not None
        else int(budget_status["remaining"]) - 1
    )
    if (
        int(disposition.get("remaining_creations_after_this_decision", -1))
        != remaining_after
    ):
        raise ValueError("final Main Agent changed the creation budget")
    if (
        int(disposition.get("minimum_next_rollout_creations", -1))
        != MIN_ROLLOUT_CREATIONS
    ):
        raise ValueError(
            "next rollout minimum must remain five tasks times the training repeat count"
        )
    if bool(disposition.get("rollout_possible")) != (
        remaining_after >= MIN_ROLLOUT_CREATIONS
    ):
        raise ValueError("final Main Agent misreported rollout feasibility")
    if int(disposition.get("minimum_next_iteration_creations", -1)) != (
        MIN_CANDIDATE_ITERATION_CREATIONS
    ):
        raise ValueError("next iteration minimum does not cover every required module")
    if bool(disposition.get("iteration_possible")) != (
        remaining_after >= MIN_CANDIDATE_ITERATION_CREATIONS
    ):
        raise ValueError("final Main Agent misreported iteration feasibility")
    primary = adjustment["primary_problem"]
    if decision == "confirm_delta":
        if require_primary_metric_improvement:
            raise ValueError("confirmation evidence cannot request another confirmation")
        if expected_recovered or expected_regressions or not expected_uncertain:
            raise ValueError(
                "confirm_delta requires only an uncertain recorded pass gain"
            )
    if decision == "replan_problem":
        if not isinstance(adjustment.get("replan_candidate"), Mapping):
            raise ValueError("replan_problem requires a post-Analyzer candidate")
        if remaining_after < MIN_CANDIDATE_ITERATION_CREATIONS:
            raise ValueError("replan_problem requires budget for another iteration")
    if (
        decision == "accept_delta"
        and remaining_after < MIN_CANDIDATE_ITERATION_CREATIONS
        and _localized_followup_task_ids(adjustment)
    ):
        raise ValueError(
            "budget-exhausted review cannot accept a candidate with unresolved "
            "failures attributed to the changed channel"
        )
    revision = output.get("revision_candidate")
    if decision != "revise_delta":
        if revision is not None:
            raise ValueError(
                "non-revision decision must not include a revision candidate"
            )
        return
    if remaining_after < MIN_CANDIDATE_ITERATION_CREATIONS:
        raise ValueError("revision requires budget for another complete iteration")
    if not isinstance(tested_candidate, Mapping):
        raise ValueError("revision validation requires the tested candidate")
    if str(tested_candidate.get("portfolio_side") or "") == "revision":
        raise ValueError("a tested revision must be accepted or rejected")
    if not isinstance(revision, Mapping):
        raise ValueError("revise decision requires a complete revision candidate")
    localized_revision_failure = bool(expected_regressions) or bool(
        _localized_followup_task_ids(adjustment)
    )
    if not localized_revision_failure:
        raise ValueError("revision requires a failure localized to the changed channel")
    original_plans = tested_candidate.get("channel_diffs") or []
    experience_ids = {
        str(item)
        for plan in original_plans
        for item in plan.get("experience_ids") or []
    }
    channel_ids = (
        set(revision_channel_ids)
        if revision_channel_ids is not None
        else {str(plan.get("channel_id") or "") for plan in original_plans}
    )
    _validate_candidate(
        revision,
        experience_ids=experience_ids,
        channel_ids=channel_ids,
    )
    revision_delta = revision.get("manifest_delta")
    tested_delta = tested_candidate.get("manifest_delta")
    if (
        isinstance(revision_delta, Mapping)
        and revision_delta
        and isinstance(tested_delta, Mapping)
        and content_digest(revision_delta) == content_digest(tested_delta)
    ):
        raise ValueError("revision candidate must change the tested manifest")


def _localized_followup_task_ids(
    adjustment: Mapping[str, Any],
) -> set[str]:
    primary = adjustment.get("primary_problem") or {}
    return {
        str(item.get("task_id") or "")
        for item in primary.get("task_assessments") or []
        if isinstance(item, Mapping)
        and str(item.get("status") or "") in {"regressed", "mixed", "still_failing"}
        and str(item.get("relation") or "") == "attributed"
        and str(item.get("task_id") or "")
    }


def canonicalize_final_decision(
    output: Mapping[str, Any],
    *,
    candidate_version: str,
    base_version: str,
    reusable: Mapping[str, Any],
    adjustment: Mapping[str, Any],
    tested_candidate: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if isinstance(output, dict):
        decision = str(output.get("decision") or "")
        if decision == "accept_delta":
            output["selected_version"] = str(candidate_version)
        elif decision in {
            "reject_delta",
            "revise_delta",
            "confirm_delta",
            "replan_problem",
        }:
            output["selected_version"] = str(base_version)
        if (
            decision == "revise_delta"
            and isinstance(output.get("revision_candidate"), Mapping)
            and isinstance(output["revision_candidate"].get("manifest_delta"), Mapping)
        ):
            _canonicalize_revision_manifest_delta(output["revision_candidate"])
            allowed_experience_ids = sorted(
                {
                    str(item)
                    for plan in (
                        tested_candidate.get("channel_diffs") or []
                        if isinstance(tested_candidate, Mapping)
                        else []
                    )
                    for item in plan.get("experience_ids") or []
                }
            )
            holder = canonicalize_analyzer_output(
                {
                    "coverage": {"experience_ids": allowed_experience_ids},
                    "candidates": [dict(output["revision_candidate"])],
                }
            )
            revision = dict(holder["candidates"][0])
            revision["id"] = f"revision-{candidate_version}"
            if isinstance(tested_candidate, Mapping):
                source_ids = tested_candidate.get("source_candidate_ids") or []
                if source_ids:
                    revision["origin_candidate_id"] = str(source_ids[0])
            output["revision_candidate"] = revision
        elif decision != "revise_delta":
            output.pop("revision_candidate", None)
    evidence = output.get("evidence")
    if not isinstance(evidence, dict):
        return output
    evidence["recovered_task_ids"] = sorted(_attributable_positive_task_ids(adjustment))
    evidence["uncertain_recovery_task_ids"] = sorted(
        uncertain_recovery_task_ids(adjustment)
    )
    evidence["preserved_task_ids"] = sorted(
        str(item) for item in reusable["preservation"]["preserved_task_ids"]
    )
    evidence["attributable_regression_task_ids"] = sorted(
        str(item["task_id"])
        for item in reusable["preservation"]["attributable_regressions"]
    )
    return output


def _attributable_positive_task_ids(
    adjustment: Mapping[str, Any],
) -> set[str]:
    result = set()
    for item in adjustment["primary_problem"]["task_assessments"]:
        if str(item.get("status") or "") not in {"recovered", "stable_success"}:
            continue
        if str(item.get("relation") or "") != "attributed":
            continue
        summary = item.get("outcome_summary")
        if isinstance(summary, Mapping) and int(
            summary.get("candidate_pass_count") or 0
        ) <= int(summary.get("reference_pass_count") or 0):
            continue
        result.add(str(item["task_id"]))
    return result


def uncertain_recovery_task_ids(
    adjustment: Mapping[str, Any],
) -> set[str]:
    """Return recorded pass gains whose causal relation still needs confirmation."""

    result: set[str] = set()
    primary = adjustment.get("primary_problem") or {}
    for item in primary.get("task_assessments") or []:
        if str(item.get("status") or "") not in {"recovered", "stable_success"}:
            continue
        if str(item.get("relation") or "") != "possibly_related":
            continue
        summary = item.get("outcome_summary") or {}
        if int(summary.get("candidate_pass_count") or 0) <= int(
            summary.get("reference_pass_count") or 0
        ):
            continue
        result.add(str(item.get("task_id") or ""))
    return {item for item in result if item}


def _required_metric(
    metrics: Mapping[str, Any] | None, metric: str, *, side: str
) -> float:
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{side} rollout metrics are missing {metric}")
    if metric == "pass_at_1" and "estimated_pass_at_1" in metrics:
        value = float(metrics["estimated_pass_at_1"])
    elif metric == "pass_at_1" and "trial_success_rate" in metrics:
        value = float(metrics["trial_success_rate"])
    elif metric in metrics:
        value = float(metrics[metric])
    else:
        raise ValueError(f"{side} rollout metrics are missing {metric}")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{side} rollout metric {metric} is outside [0, 1]")
    return value


def deterministic_reject_decision(
    *,
    base_version: str,
    reusable: Mapping[str, Any],
    adjustment: Mapping[str, Any],
    remaining: int,
    failure_reason: str,
) -> dict[str, Any]:
    regressions = [
        str(item["task_id"])
        for item in reusable["preservation"]["attributable_regressions"]
    ]
    preserved = [str(item) for item in reusable["preservation"]["preserved_task_ids"]]
    payload = {
        "decision": "reject_delta",
        "selected_version": str(base_version),
        "rationale": (
            "Keep the current champion because the proposed promotion did not satisfy "
            f"the deterministic admission gate: {str(failure_reason)}"
        ),
        "evidence": {
            "recovered_task_ids": sorted(_attributable_positive_task_ids(adjustment)),
            "preserved_task_ids": sorted(preserved),
            "attributable_regression_task_ids": sorted(regressions),
            "unresolved_findings": [
                "The Main Agent did not produce a valid admissible decision; the "
                "controller retained the direct parent."
            ],
        },
    }
    return normalize_budget_disposition(
        payload, remaining_after_decision=int(remaining)
    )


def _canonicalize_revision_manifest_delta(candidate: Mapping[str, Any]) -> None:
    if not isinstance(candidate, dict):
        return
    delta = candidate.get("manifest_delta")
    if not isinstance(delta, dict):
        return
    instructions = _normalize_instruction_entries(delta.get("instructions") or [])
    tool_desc_patches = dict(delta.get("tool_desc_patches") or {})
    files = []
    for raw_file in delta.get("files") or []:
        if not isinstance(raw_file, Mapping):
            files.append(raw_file)
            continue
        path = str(raw_file.get("path") or "").strip()
        content = raw_file.get("content")
        if path in {
            "instructions",
            "instructions_rules",
            "instructions_rules.md",
            ".opencode/instructions_rules.md",
        } or path.startswith("instructions/"):
            if isinstance(content, list):
                instructions.extend(_normalize_instruction_entries(content))
            elif str(content or "").strip():
                instructions.extend(_normalize_instruction_entries([content]))
            continue
        if path.rsplit("/", 1)[-1].lower() in {
            "tool_desc_patches",
            "tool_desc_patches.json",
            "tool_description_patches",
            "tool_description_patches.json",
        }:
            parsed = content
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    parsed = None
            if isinstance(parsed, Mapping):
                tool_desc_patches = _merge_tool_description_patches(
                    tool_desc_patches,
                    parsed,
                )
                continue
        files.append(raw_file)
    if instructions:
        delta["instructions"] = instructions
    if tool_desc_patches:
        delta["tool_desc_patches"] = tool_desc_patches
    if files:
        delta["files"] = files
    else:
        delta.pop("files", None)


def _merge_tool_description_patches(
    existing: Mapping[str, Any], incoming: Mapping[str, Any]
) -> dict[str, Any]:
    merged = dict(existing)
    for raw_tool, raw_patch in incoming.items():
        tool = str(raw_tool)
        if not isinstance(raw_patch, Mapping):
            merged[tool] = raw_patch
            continue
        existing_patch = merged.get(tool)
        patch = dict(existing_patch) if isinstance(existing_patch, Mapping) else {}
        patch.update(dict(raw_patch))
        if (
            isinstance(existing_patch, Mapping)
            and isinstance(existing_patch.get("params"), Mapping)
            and isinstance(raw_patch.get("params"), Mapping)
        ):
            patch["params"] = {
                **dict(existing_patch["params"]),
                **dict(raw_patch["params"]),
            }
        merged[tool] = patch
    return merged


def _normalize_instruction_entries(raw_entries: Sequence[Any]) -> list[str]:
    normalized: list[str] = []
    for raw_entry in raw_entries:
        text = str(raw_entry or "").strip()
        if not text:
            continue
        parsed = None
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
        if isinstance(parsed, list):
            normalized.extend(str(item).strip() for item in parsed if str(item).strip())
        else:
            normalized.append(text)
    return normalized


def _max_rollout_tasks(remaining_creations: int) -> int:
    remaining = int(remaining_creations)
    usable = remaining - ITERATION_RETRY_BUFFER_CREATIONS
    if iteration_creation_cost(5) > usable:
        return 0
    count = 5
    while iteration_creation_cost(count + 1) <= usable:
        count += 1
    return count


def _rollout_task_limit(
    remaining_creations: int, *, unattempted_batch_count: int
) -> int:
    remaining = int(remaining_creations)
    batch_count = int(unattempted_batch_count)
    exploring_multiple_problems = batch_count > 1
    if (
        exploring_multiple_problems
        and remaining
        >= 2 * MIN_CANDIDATE_ITERATION_CREATIONS + ITERATION_RETRY_BUFFER_CREATIONS
    ):
        remaining -= MIN_CANDIDATE_ITERATION_CREATIONS
    limit = _max_rollout_tasks(remaining)
    if exploring_multiple_problems:
        return min(limit, DEFAULT_EXPLORATION_ROLLOUT_TASKS)
    return limit
