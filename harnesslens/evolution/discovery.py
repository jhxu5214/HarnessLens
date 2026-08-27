from __future__ import annotations

import json
import re
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from harnesslens.infrastructure.analysis_concurrency import analysis_workers
from harnesslens.core.artifacts import write_json
from harnesslens.core.budget import CreationBudget
from harnesslens.harnesses.harness_query_adapters import harness_query_adapter
from harnesslens.harnesses.harness_workspace import MCP_PATCH_WORKSPACE_PATH
from harnesslens.core.profiles import power_profile
from harnesslens.harnesses.runner import (
    IntelligentHarnessRunner,
    IntelligentRunResult,
    parse_json_object,
)
from harnesslens.benchmarks.task_data import BaselineDataset, benchmark_task_explorer_input


TASK_EXPLORER_SYSTEM = """
You are the task-definition explorer for a harness optimization run.
Classify the supplied TRAIN tasks mainly by user goal. Use at most ten primary categories.
You may use a few auxiliary angles only when they clarify important behavioral differences.
Do not infer expected actions, grader logic, hidden answers, or harness modifications.
Every task must appear exactly once in a primary category.
If a task contains multiple goals, choose the dominant user-facing outcome as
the primary category. Do not list the same task in any second category; record
secondary interpretations only in auxiliary_angles or notes.

Return:
{"categories":[{"id":"short-id","name":"name","purpose":"one precise sentence","task_ids":["..."],"auxiliary_angles":["..."]}],"notes":["..."]}
""".strip()


HARNESS_QUERY_SYSTEM = """
You are the harness-query explorer for a harness optimization run.
Query the supplied architecture probe, documentation evidence, runtime observations, and candidate
workspace contract to determine what this exact harness can actually customize in this experiment.
The supplied surface hypotheses are leads, not conclusions: account for each one, and add a newly
discovered native harness surface when the evidence supports it. Do not analyze task trajectories
or propose domain-specific behavior. Connected MCP tools and parameters are environment facts that
the controller attaches after your response; do not discover, assess, or return MCP editable points.

A surface is currently modifiable only when the candidate can implement the exact mechanism using
captured files under `home/` or `project/`, the fixed runtime will load it, and cited evidence
supports the claim. A CLI-only switch, legacy manifest field, disabled feature, overwritten setting,
or unselected integration is conditional or unavailable, not modifiable. Distinguish documented,
materialized, and request-observed evidence; do not upgrade one into another. The controller owns
only harness identity, experimental safety constraints, and exact MCP server/tool/parameter bindings.
When an exact request-observed workspace edit contract agrees with the candidate runtime loader,
treat that surface as established modifiable unless a fixed override explicitly makes it ineffective;
disabled automatic discovery does not negate an explicit candidate-loading path.

For a modifiable surface, provide its visibility and use, a concrete workspace `edit_contract`, the
runtime conditions required for it to work, evidence references, and regression risk. Put every
supplied hypothesis that is not currently writable and effective in `unavailable_modules` with a
reason. Each edit contract must choose one preferred scope
and one path relative to that scope; do not combine alternatives or repeat `home/` or `project/` in
the path. Before classifying, compare every hypothesis with every fixed override and disabled
behavior in the candidate workspace contract. Produce a compact base profile for downstream roles.

Return:
{"harness":"...","harness_version":"...","summary":"...","modifiable_modules":[{"id":"stable-id","status":"modifiable","evidence_level":"runtime_observed|materialized|documented","visibility":"...","use":"...","edit_contract":{"scope":"project","path":"relative/path-or-pattern","mechanism":"file|config|extension"},"runtime_constraints":["..."],"risks":"...","evidence_refs":["exact-ref"]}],"unavailable_modules":[{"id":"stable-id","status":"conditional|unsupported|forbidden","reason":"...","runtime_constraints":["..."],"evidence_refs":["exact-ref"]}],"base_profile":{"shared_rules":["..."],"role_isolation":"..."}}
""".strip()


@dataclass(frozen=True)
class DiscoveryResult:
    task_explorer: Mapping[str, Any]
    harness_query: Mapping[str, Any]
    task_output_path: str
    harness_output_path: str


_HARNESS_HOME_MAPPINGS = {
    "opencode": "XDG_CONFIG_HOME/opencode",
    "pi": "PI_CODING_AGENT_DIR",
    "codex": "CODEX_HOME",
}


_CANDIDATE_RUNTIME_LOADING = {
    "opencode": {
        "config_paths": [
            "home/config.json",
            "home/opencode.json",
            "project/opencode.json",
        ],
        "preserved_behavior": [
            "agent.build.prompt from candidate config is appended to the task system prompt",
            "candidate opencode.json instructions entries load referenced project files",
            "candidate agent definitions are discovered from .opencode/agents",
            "candidate project files are materialized before OpenCode starts",
        ],
        "fixed_overrides": [
            "model and provider",
            "task tools and permissions",
            "the task MCP server binding",
        ],
        "fixed_surface_ids": ["builtin_tools", "permissions", "mcp_servers"],
        "conditional_surface_ids": ["compaction_config"],
        "disabled_behavior": [
            (
                "automatic native project-config discovery beyond the captured "
                "candidate config explicitly merged by the controller"
            ),
            "automatic compaction and pruning",
        ],
    },
    "pi": {
        "config_paths": [
            "home/settings.json",
            "home/.pi/settings.json",
            "project/.pi/settings.json",
        ],
        "preserved_behavior": [
            "candidate project files are materialized before Pi starts",
            "candidate .pi/APPEND_SYSTEM.md is appended by Pi after project trust",
            "candidate settings are merged before fixed runtime settings",
        ],
        "fixed_overrides": [
            "model and provider",
            "task tool authority and permissions",
            "process invocation arguments",
        ],
        "conditional_behavior": [
            "skills require the fixed task runtime to expose a way to read their bodies",
            "extensions cannot expand the fixed benchmark authority boundary",
        ],
    },
    "codex": {
        "config_paths": [
            "home/config.toml",
            "project/.codex/config.toml",
        ],
        "preserved_behavior": [
            "developer_instructions from candidate config are appended to the task developer context",
            "candidate agent definitions are discovered from .codex/agents",
            "candidate hook context is dispatched by a harness-owned SessionStart hook",
            "candidate project files are materialized before Codex starts",
        ],
        "fixed_overrides": [
            "model, provider, reasoning, sandbox, and approval policy",
            "default tool enablement and task tool authority",
            "the task MCP server binding and disabled feature flags",
        ],
        "fixed_surface_ids": ["feature_flags", "mcp_servers"],
        "conditional_behavior": [
            "agent definitions are inert unless the fixed runtime enables delegation",
            "hook handlers must remain inside the isolated candidate workspace",
        ],
    },
}


class DiscoveryModule:
    def __init__(
        self,
        *,
        repo_root: str | Path,
        run_root: str | Path,
        budget: CreationBudget,
        harness: str = "opencode",
        cell: str = "retail",
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.run_root = Path(run_root).resolve()
        self.budget = budget
        self.harness = str(harness)
        self.cell = str(cell)
        self.output_root = self.run_root / "discovery"
        self.output_root.mkdir(parents=True, exist_ok=True)

    def run(self, *, baseline_event: str | Path) -> DiscoveryResult:
        cached_task = self.output_root / "task_explorer.json"
        cached_harness = self.output_root / "harness_query.json"
        baseline = BaselineDataset.from_ingest_event(baseline_event)
        task_input = benchmark_task_explorer_input(
            repo_root=self.repo_root,
            baseline=baseline,
            cell=self.cell,
        )
        harness_input = self._harness_query_input(task_input)
        write_json(self.output_root / "task_explorer_input.json", task_input)
        write_json(self.output_root / "harness_query_input.json", harness_input)

        cached_task_output = (
            json.loads(cached_task.read_text(encoding="utf-8"))
            if cached_task.exists()
            else None
        )
        cached_harness_output = (
            json.loads(cached_harness.read_text(encoding="utf-8"))
            if cached_harness.exists()
            else None
        )
        if cached_task_output is not None:
            validate_task_explorer(cached_task_output, baseline.task_ids)
        if cached_harness_output is not None:
            try:
                cached_harness_output = canonicalize_harness_query_output(
                    cached_harness_output, harness_input
                )
                self._validate_harness_output(cached_harness_output, harness_input)
            except ValueError:
                cached_harness_output = None
            else:
                write_json(cached_harness, cached_harness_output)
        if cached_task_output is not None and cached_harness_output is not None:
            return DiscoveryResult(
                task_explorer=cached_task_output,
                harness_query=cached_harness_output,
                task_output_path=str(cached_task),
                harness_output_path=str(cached_harness),
            )

        if cached_task_output is not None:
            harness_result = self._run_harness_query(harness_input)
            if harness_result.output is None:
                raise RuntimeError("harness query did not produce a required output")
            write_json(cached_harness, harness_result.output)
            return DiscoveryResult(
                task_explorer=cached_task_output,
                harness_query=harness_result.output,
                task_output_path=str(cached_task),
                harness_output_path=str(cached_harness),
            )

        if cached_harness_output is not None:
            task_result = self._run_task_explorer(task_input, baseline.task_ids)
            if task_result.output is None:
                raise RuntimeError("task explorer did not produce a required output")
            write_json(cached_task, task_result.output)
            return DiscoveryResult(
                task_explorer=task_result.output,
                harness_query=cached_harness_output,
                task_output_path=str(cached_task),
                harness_output_path=str(cached_harness),
            )

        with ThreadPoolExecutor(max_workers=analysis_workers(2)) as executor:
            task_future = executor.submit(
                self._run_task_explorer,
                task_input,
                baseline.task_ids,
            )
            harness_future = executor.submit(self._run_harness_query, harness_input)
            task_result = task_future.result()
            harness_result = harness_future.result()
        if task_result.output is None or harness_result.output is None:
            raise RuntimeError("discovery did not produce both required outputs")
        write_json(cached_task, task_result.output)
        write_json(cached_harness, harness_result.output)
        return DiscoveryResult(
            task_explorer=task_result.output,
            harness_query=harness_result.output,
            task_output_path=str(cached_task),
            harness_output_path=str(cached_harness),
        )

    def _run_task_explorer(
        self,
        payload: Mapping[str, Any],
        task_ids: tuple[str, ...],
    ) -> IntelligentRunResult:
        runner = IntelligentHarnessRunner(
            profile=power_profile(self.harness, max_steps=30),
            budget=self.budget,
            workspace_root=self.run_root / "intelligent_jobs",
        )
        result = None
        for attempt in range(2):
            current_payload = _harness_query_model_payload(payload)
            current_payload["retry_context"] = (
                "Previous task classification failed validation. Return a partition: "
                "each TRAIN task ID must appear exactly once across all primary "
                "category task_ids. For multi-intent tasks, choose one dominant "
                "primary category and mention secondary aspects only in notes."
                if attempt
                else ""
            )
            result = runner.run_json(
                job_id=self.budget.next_attempt_id("discovery-task-explorer"),
                system_prompt=TASK_EXPLORER_SYSTEM,
                input_payload=current_payload,
                validator=lambda output: validate_task_explorer(output, task_ids),
            )
            if result.output is not None:
                write_json(self.output_root / "task_explorer.json", result.output)
                return result
        assert result is not None
        return result

    def _run_harness_query(self, payload: Mapping[str, Any]) -> IntelligentRunResult:
        runner = IntelligentHarnessRunner(
            profile=power_profile(self.harness, max_steps=30),
            budget=self.budget,
            workspace_root=self.run_root / "intelligent_jobs",
        )
        result = None
        previous_error = ""
        previous_output: Mapping[str, Any] | None = None
        for attempt in range(2):
            current_payload = _harness_query_model_payload(payload)
            current_payload["retry_context"] = (
                f"Previous output failed validation: {previous_error}. "
                "Revise the supplied previous_output only where needed to fix that error. "
                "Account for every surface hypothesis, cite supplied evidence, and provide a safe home/project edit "
                "contract only for surfaces that the fixed runtime can actually load."
                if attempt
                else ""
            )
            if previous_output is not None:
                current_payload["previous_output"] = previous_output
            result = runner.run_json(
                job_id=self.budget.next_attempt_id("discovery-harness-query"),
                system_prompt=HARNESS_QUERY_SYSTEM,
                input_payload=current_payload,
                validator=lambda output: self._validate_harness_output(
                    canonicalize_harness_query_output(output, payload), payload
                ),
            )
            if result.output is not None:
                canonical = canonicalize_harness_query_output(result.output, payload)
                write_json(self.output_root / "harness_query.json", canonical)
                return replace(result, output=canonical)
            previous_error = str(result.validation_error or "unknown validation error")
            previous_output = _recover_harness_query_output(result)
        assert result is not None
        if previous_output is not None:
            fallback = _conservative_harness_query_output(previous_output, payload)
            self._validate_harness_output(fallback, payload)
            write_json(self.output_root / "harness_query.json", fallback)
            return replace(result, output=fallback, validation_error="")
        return result

    def _validate_harness_output(
        self, output: Mapping[str, Any], payload: Mapping[str, Any]
    ) -> None:
        model_payload = _harness_query_model_payload(payload)
        required = {
            str(item["id"])
            for item in model_payload.get("surface_hypotheses") or ()
            if isinstance(item, Mapping) and item.get("id")
        }
        probe = payload.get("architecture_probe") or {}
        channel_contracts = {
            str(item["id"]): item
            for item in model_payload.get("surface_hypotheses") or ()
            if isinstance(item, Mapping) and item.get("id")
        }
        evidence_catalog = _query_evidence_with_input_paths(payload)
        validate_harness_query(
            output,
            required,
            payload.get("predefined_mcp_editable_points") or [],
            expected_harness=str(probe.get("harness_id") or self.harness),
            expected_harness_version=str(probe.get("harness_version") or ""),
            evidence_ids={
                str(item.get("id"))
                for item in evidence_catalog
                if isinstance(item, Mapping) and item.get("id")
            },
            evidence_catalog=evidence_catalog,
            channel_contracts=channel_contracts,
            workspace_contract=payload.get("candidate_workspace_contract") or {},
        )

    def _harness_query_input(self, task_input: Mapping[str, Any]) -> dict[str, Any]:
        adapter = harness_query_adapter(self.harness, repo_root=self.repo_root)
        probe = adapter.architecture_probe()
        evidence_catalog = adapter.query_evidence_catalog(probe)
        runtime_loading = _CANDIDATE_RUNTIME_LOADING[str(probe["harness_id"])]
        evidence_catalog.append(
            {
                "id": "runtime:candidate-workspace-loader",
                "kind": "runtime_loader_contract",
                "source": "HarnessLens candidate workspace materialization and fixed-runtime merge",
                "value": runtime_loading,
            }
        )
        inventory = adapter.query_channel_inventory(probe)
        inventory_ids = {str(item.get("id") or "") for item in inventory}
        mcp_points = (
            _mcp_editable_points(task_input, harness_id=str(probe["harness_id"]))
            if {"tool_description", "tool_parameter_description"}.issubset(
                inventory_ids
            )
            else []
        )
        if mcp_points:
            evidence_catalog.append(
                {
                    "id": "runtime:mcp-workspace-bridge",
                    "kind": "materializer_contract",
                    "source": "HarnessLens controller-managed MCP workspace bridge",
                    "value": {
                        "kind": "tool_schema_patch",
                        "workspace_bridge_path": MCP_PATCH_WORKSPACE_PATH,
                    },
                }
            )
        fixed_surface_ids = {
            str(item) for item in runtime_loading.get("fixed_surface_ids") or []
        }
        conditional_surface_ids = {
            str(item) for item in runtime_loading.get("conditional_surface_ids") or []
        }
        surface_hypotheses = [_surface_hypothesis(item) for item in inventory]
        for hypothesis in surface_hypotheses:
            channel_id = str(hypothesis.get("id") or "")
            if channel_id in fixed_surface_ids:
                hypothesis["policy_status"] = "forbidden"
            elif channel_id in conditional_surface_ids:
                hypothesis["policy_status"] = "conditional"
        return {
            "architecture_probe": probe,
            "evidence_catalog": evidence_catalog,
            "surface_hypotheses": surface_hypotheses,
            "candidate_workspace_contract": {
                "captured_scopes": [
                    {
                        "scope": "home",
                        "runtime_mapping": _HARNESS_HOME_MAPPINGS[
                            str(probe["harness_id"])
                        ],
                    },
                    {
                        "scope": "project",
                        "runtime_mapping": "the isolated task working directory",
                    },
                ],
                "submission_rule": (
                    "Only UTF-8 files captured under home/ and project/ are submitted."
                ),
                "not_writable": [
                    "process invocation arguments",
                    "process environment and credentials",
                    "legacy manifest fields without a corresponding captured workspace file",
                    "host or global harness configuration",
                ],
                "fixed_dimensions": [
                    "model and provider",
                    "reasoning level and inference budget",
                    "task tool authority and permissions",
                    "evaluation and hidden task state",
                ],
                "forbidden_surface_ids": [
                    str(item["id"])
                    for item in surface_hypotheses
                    if str(item.get("policy_status") or "") == "forbidden"
                ],
                "conditional_surface_ids": [
                    str(item["id"])
                    for item in surface_hypotheses
                    if str(item.get("policy_status") or "") == "conditional"
                ],
                "runtime_loading": runtime_loading,
                "runtime_loading_evidence_ref": "runtime:candidate-workspace-loader",
            },
            "public_runtime": {
                "benchmark_kind": str(task_input.get("benchmark_kind") or ""),
                "task_tool_names": [
                    str(item.get("name"))
                    for item in (
                        (task_input.get("environment") or {}).get("tools") or []
                    )
                    if isinstance(item, Mapping) and item.get("name")
                ],
                "tool_transport": dict(
                    ((task_input.get("environment") or {}).get("tool_transport") or {})
                ),
            },
            "predefined_mcp_editable_points": deepcopy(mcp_points),
        }


def _surface_hypothesis(raw: Mapping[str, Any]) -> dict[str, Any]:
    hypothesis = {
        key: raw[key]
        for key in (
            "id",
            "visibility",
            "use",
            "artifact_contract",
            "risks",
            "evidence_refs",
            "verification",
        )
        if key in raw
    }
    status = str(raw.get("status") or "")
    if status == "forbidden":
        hypothesis["policy_status"] = "forbidden"
    elif status == "exists_but_unsupported":
        hypothesis["policy_status"] = "unsupported"
    elif status == "proposal_only":
        hypothesis["policy_status"] = "conditional"
    operation = raw.get("operation")
    if isinstance(operation, Mapping):
        kind = str(operation.get("kind") or "")
        if kind == "workspace_config":
            hypothesis["trusted_edit_contract"] = {
                key: str(operation[key])
                for key in ("scope", "path", "mechanism", "key")
                if str(operation.get(key) or "").strip()
            }
        elif (
            kind == "project_file" and str(operation.get("path_pattern") or "").strip()
        ):
            hypothesis["trusted_edit_contract"] = {
                "scope": "project",
                "path": str(operation["path_pattern"]),
                "mechanism": "file",
            }
        if str(operation.get("key") or "").strip():
            hypothesis["edit_selector"] = {"key": str(operation["key"])}
        elif str(operation.get("key_prefix") or "").strip():
            hypothesis["edit_selector"] = {"key_prefix": str(operation["key_prefix"])}
    return hypothesis


def _harness_query_model_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    model_payload = deepcopy(dict(payload))
    points = model_payload.pop("predefined_mcp_editable_points", [])
    managed_base_channels = {
        str(point.get("base_channel_id") or "")
        for point in points
        if isinstance(point, Mapping) and point.get("base_channel_id")
    }
    model_payload["surface_hypotheses"] = [
        item
        for item in model_payload.get("surface_hypotheses") or []
        if not (
            isinstance(item, Mapping)
            and str(item.get("id") or "") in managed_base_channels
        )
    ]
    model_payload["evidence_catalog"] = [
        item
        for item in model_payload.get("evidence_catalog") or []
        if not (
            isinstance(item, Mapping)
            and str(item.get("id") or "") == "runtime:mcp-workspace-bridge"
        )
    ]
    public_runtime = model_payload.get("public_runtime")
    if isinstance(public_runtime, Mapping):
        model_payload["public_runtime"] = {
            key: deepcopy(value)
            for key, value in public_runtime.items()
            if key not in {"task_tool_names", "tool_transport"}
        }
    model_payload["controller_managed_mcp"] = {
        "connected": bool(points),
        "note": (
            "Connected MCP editable points are attached automatically after Harness Query; "
            "do not discover or return them."
        ),
    }
    return model_payload


def _recover_harness_query_output(
    result: IntelligentRunResult,
) -> Mapping[str, Any] | None:
    path_text = str(getattr(result, "stdout_path", "") or "")
    if not path_text:
        return None
    try:
        return dict(
            parse_json_object(
                Path(path_text).read_text(encoding="utf-8", errors="replace")
            )
        )
    except (OSError, ValueError):
        return None


def _query_evidence_with_input_paths(
    payload: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    catalog = [
        item
        for item in payload.get("evidence_catalog") or []
        if isinstance(item, Mapping)
    ]
    workspace = payload.get("candidate_workspace_contract") or {}
    probe = payload.get("architecture_probe") or {}
    catalog.extend(
        (
            {
                "id": "candidate_workspace_contract.fixed_overrides",
                "kind": "runtime_loader_contract",
                "value": (workspace.get("runtime_loading") or {}).get(
                    "fixed_overrides", []
                ),
            },
            {
                "id": "candidate_workspace_contract.disabled_behavior",
                "kind": "runtime_loader_contract",
                "value": (workspace.get("runtime_loading") or {}).get(
                    "disabled_behavior", []
                ),
            },
            {
                "id": "architecture_probe.materialization_points",
                "kind": "architecture_probe",
                "value": probe.get("materialization_points") or [],
            },
        )
    )
    for root_key in (
        "architecture_probe",
        "candidate_workspace_contract",
        "public_runtime",
    ):
        catalog.extend(_input_reference_evidence(root_key, payload.get(root_key) or {}))
    for point in payload.get("predefined_mcp_editable_points") or []:
        if not isinstance(point, Mapping) or not point.get("id"):
            continue
        catalog.append(
            {
                "id": f"predefined_mcp_editable_points:{point['id']}",
                "kind": "input_contract",
                "value": point,
            }
        )
    return list(
        {
            str(item["id"]): item
            for item in catalog
            if isinstance(item, Mapping) and item.get("id")
        }.values()
    )


def _input_reference_evidence(root: str, value: Any) -> list[Mapping[str, Any]]:
    evidence: list[Mapping[str, Any]] = []

    def visit(dot_path: str, colon_path: str, current: Any) -> None:
        evidence.append({"id": dot_path, "kind": "input_contract", "value": current})
        if colon_path != dot_path:
            evidence.append(
                {"id": colon_path, "kind": "input_contract", "value": current}
            )
        if isinstance(current, Mapping):
            for key, child in current.items():
                visit(
                    f"{dot_path}.{key}",
                    f"{colon_path}:{key}",
                    child,
                )

    visit(root, root, value)
    return evidence


def validate_task_explorer(
    output: Mapping[str, Any], task_ids: tuple[str, ...]
) -> None:
    categories = output.get("categories")
    if not isinstance(categories, list) or not 1 <= len(categories) <= 10:
        raise ValueError("task explorer must return between one and ten categories")
    seen: list[str] = []
    category_ids: set[str] = set()
    for category in categories:
        if not isinstance(category, Mapping):
            raise ValueError("task category must be an object")
        category_id = str(category.get("id") or "").strip()
        if not category_id or category_id in category_ids:
            raise ValueError("task category IDs must be unique and nonempty")
        if (
            not str(category.get("name") or "").strip()
            or not str(category.get("purpose") or "").strip()
        ):
            raise ValueError("task category requires name and purpose")
        members = category.get("task_ids")
        if not isinstance(members, list) or not members:
            raise ValueError("task category must contain task IDs")
        category_ids.add(category_id)
        seen.extend(str(item) for item in members)
    if len(seen) != len(set(seen)):
        raise ValueError("each TRAIN task must have exactly one primary category")
    if set(seen) != set(task_ids):
        raise ValueError(
            f"task explorer coverage must exactly equal the {len(task_ids)} TRAIN tasks"
        )


def validate_harness_query(
    output: Mapping[str, Any],
    required_channels: set[str],
    required_mcp_points: Sequence[Mapping[str, Any]] = (),
    *,
    expected_harness: str | None = None,
    expected_harness_version: str | None = None,
    evidence_ids: set[str] | None = None,
    evidence_catalog: Sequence[Mapping[str, Any]] = (),
    channel_contracts: Mapping[str, Mapping[str, Any]] | None = None,
    workspace_contract: Mapping[str, Any] | None = None,
) -> None:
    modules = output.get("modifiable_modules")
    if not isinstance(modules, list):
        raise ValueError("harness query must return modifiable_modules")
    unavailable = output.get("unavailable_modules")
    if not isinstance(unavailable, list):
        raise ValueError("harness query must return unavailable_modules")
    module_ids = _surface_ids(modules, section="modifiable_modules")
    unavailable_ids = _surface_ids(unavailable, section="unavailable_modules")
    overlap = set(module_ids) & set(unavailable_ids)
    if overlap:
        raise ValueError(
            f"harness query surfaces cannot be both modifiable and unavailable: {sorted(overlap)}"
        )
    missing = set(required_channels) - set(module_ids) - set(unavailable_ids)
    if missing:
        raise ValueError(f"harness query surface coverage missing={sorted(missing)}")
    if expected_harness is not None and str(output.get("harness") or "") != str(
        expected_harness
    ):
        raise ValueError("harness query harness identity mismatch")
    if expected_harness_version is not None and str(
        output.get("harness_version") or ""
    ) != str(expected_harness_version):
        raise ValueError("harness query harness version mismatch")
    known_evidence = set(evidence_ids or ())
    evidence_by_id = {
        str(item.get("id") or ""): item
        for item in evidence_catalog
        if isinstance(item, Mapping) and item.get("id")
    }
    contracts = dict(channel_contracts or {})
    surface_errors: list[str] = []
    for item in modules:
        channel_id = str(item.get("id") or "")
        policy_status = str(
            (contracts.get(channel_id) or {}).get("policy_status") or ""
        )
        if policy_status in {"forbidden", "conditional", "unsupported"}:
            surface_errors.append(
                f"{channel_id}: policy-{policy_status} harness surface cannot be modifiable"
            )
            continue
        try:
            _validate_modifiable_surface(
                item,
                known_evidence=known_evidence,
                evidence_by_id=evidence_by_id,
                evidence_subject=channel_id,
                workspace_contract=workspace_contract or {},
            )
        except ValueError as exc:
            surface_errors.append(f"{channel_id}: {exc}")
    for item in unavailable:
        channel_id = str(item.get("id") or "")
        try:
            _validate_unavailable_surface(item, known_evidence=known_evidence)
            if (
                str((contracts.get(channel_id) or {}).get("policy_status") or "")
                == "forbidden"
                and str(item.get("status") or "") != "forbidden"
            ):
                raise ValueError(
                    "policy-forbidden harness surface must remain forbidden"
                )
        except ValueError as exc:
            surface_errors.append(f"{channel_id}: {exc}")
    if not isinstance(output.get("base_profile"), Mapping):
        raise ValueError("harness query must return a base_profile")
    points = output.get("mcp_editable_points")
    if not isinstance(points, list):
        raise ValueError("harness query must return mcp_editable_points")
    required_by_id = {str(item["id"]): item for item in required_mcp_points}
    reported_by_id = {
        str(item.get("id") or ""): item for item in points if isinstance(item, Mapping)
    }
    if set(reported_by_id) != set(required_by_id):
        raise ValueError("harness query MCP editable-point coverage mismatch")
    for point_id, expected in required_by_id.items():
        reported = reported_by_id[point_id]
        if str(reported.get("base_channel_id") or "") != str(
            expected["base_channel_id"]
        ) or str(reported.get("server_id") or "") != str(expected["server_id"]):
            raise ValueError("harness query MCP editable-point binding mismatch")
        if reported.get("targets") != expected.get("targets"):
            raise ValueError("harness query MCP editable-point targets mismatch")
        try:
            if str(reported.get("status") or "") == "modifiable":
                _validate_modifiable_surface(
                    reported,
                    known_evidence=known_evidence,
                    evidence_by_id=evidence_by_id,
                    evidence_subject=str(reported.get("base_channel_id") or point_id),
                    workspace_contract=workspace_contract or {},
                )
            else:
                _validate_unavailable_surface(reported, known_evidence=known_evidence)
        except ValueError as exc:
            surface_errors.append(f"{point_id}: {exc}")
    if surface_errors:
        raise ValueError("; ".join(surface_errors))


def _surface_ids(items: Sequence[Any], *, section: str) -> list[str]:
    ids: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError(f"harness query {section} entries must be objects")
        channel_id = str(item.get("id") or "").strip()
        if not re.fullmatch(r"[a-z][a-z0-9_.-]*", channel_id):
            raise ValueError(
                "harness query surface IDs must be stable lowercase identifiers"
            )
        ids.append(channel_id)
    if len(ids) != len(set(ids)):
        raise ValueError(f"harness query {section} IDs must be unique")
    return ids


def _validate_modifiable_surface(
    item: Mapping[str, Any],
    *,
    known_evidence: set[str],
    evidence_by_id: Mapping[str, Mapping[str, Any]],
    evidence_subject: str,
    workspace_contract: Mapping[str, Any],
) -> None:
    if str(item.get("status") or "") != "modifiable":
        raise ValueError("modifiable harness surface must use status=modifiable")
    for field in ("visibility", "use", "risks"):
        if not str(item.get(field) or "").strip():
            raise ValueError(f"modifiable harness surface requires {field}")
    constraints = item.get("runtime_constraints")
    if (
        not isinstance(constraints, list)
        or not constraints
        or any(not str(value).strip() for value in constraints)
    ):
        raise ValueError("modifiable harness surface requires runtime_constraints")
    edit = item.get("edit_contract")
    if not isinstance(edit, Mapping):
        raise ValueError("modifiable harness surface requires edit_contract")
    _validate_edit_contract(edit)
    refs = _validate_evidence_refs(item, known_evidence=known_evidence)
    level = str(item.get("evidence_level") or "")
    if level not in {"runtime_observed", "materialized", "documented"}:
        raise ValueError("modifiable harness surface has invalid evidence_level")
    if level == "runtime_observed" and not any(
        _runtime_evidence_supports(evidence_by_id.get(ref), evidence_subject)
        for ref in refs
    ):
        raise ValueError(
            "runtime-observed evidence does not support this harness surface"
        )
    if level == "materialized" and not any(
        str((evidence_by_id.get(ref) or {}).get("kind") or "")
        == "materializer_contract"
        for ref in refs
    ):
        raise ValueError("materialized evidence_level requires materializer evidence")
    if not _edit_contract_is_grounded(
        edit,
        refs=refs,
        evidence_by_id=evidence_by_id,
        evidence_subject=evidence_subject,
        workspace_contract=workspace_contract,
    ):
        raise ValueError(
            f"{evidence_subject} edit_contract is not supported by the cited runtime evidence"
        )


def _validate_edit_contract(edit: Mapping[str, Any]) -> None:
    if str(edit.get("scope") or "") not in {"home", "project"}:
        raise ValueError("edit_contract scope must be home or project")
    path_text = str(edit.get("path") or "").strip()
    path = PurePosixPath(path_text)
    if (
        not path_text
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or "\\" in path_text
        or path.parts[0] in {"home", "project"}
        or re.search(r"[\s|()]", path_text)
    ):
        raise ValueError("edit_contract path must stay inside its workspace scope")
    if not str(edit.get("mechanism") or "").strip():
        raise ValueError("edit_contract requires a mechanism")


def _validate_unavailable_surface(
    item: Mapping[str, Any], *, known_evidence: set[str]
) -> None:
    if str(item.get("status") or "") not in {"conditional", "unsupported", "forbidden"}:
        raise ValueError("unavailable harness surface has invalid status")
    if not str(item.get("reason") or "").strip():
        raise ValueError("unavailable harness surface requires a reason")
    constraints = item.get("runtime_constraints")
    if constraints is not None and (
        not isinstance(constraints, list)
        or any(not str(value).strip() for value in constraints)
    ):
        raise ValueError("unavailable harness surface has invalid runtime_constraints")
    _validate_evidence_refs(item, known_evidence=known_evidence)
    edit = item.get("edit_contract")
    if edit is not None:
        if not isinstance(edit, Mapping):
            raise ValueError("unavailable harness surface has invalid edit_contract")
        _validate_edit_contract(edit)


def _validate_evidence_refs(
    item: Mapping[str, Any], *, known_evidence: set[str]
) -> set[str]:
    refs = item.get("evidence_refs")
    if not isinstance(refs, list) or not refs:
        raise ValueError("harness query surface requires evidence references")
    reported = {str(ref) for ref in refs if str(ref).strip()}
    if len(reported) != len(refs) or (known_evidence and reported - known_evidence):
        raise ValueError("harness query surface evidence mismatch")
    return reported


def _runtime_evidence_supports(
    evidence: Mapping[str, Any] | None, channel_id: str
) -> bool:
    if (
        not isinstance(evidence, Mapping)
        or str(evidence.get("kind") or "") != "request_sentinel_probe"
    ):
        return False
    value = evidence.get("value") or {}
    if not isinstance(value, Mapping):
        return False
    observed = value.get("observed_channels") or []
    observations = value.get("observations") or {}
    return (
        channel_id in {str(item) for item in observed}
        or (isinstance(observations, Mapping) and channel_id in observations)
        or channel_id in value
    )


def _edit_contract_is_grounded(
    edit: Mapping[str, Any],
    *,
    refs: set[str],
    evidence_by_id: Mapping[str, Mapping[str, Any]],
    evidence_subject: str,
    workspace_contract: Mapping[str, Any],
) -> bool:
    materializers = [
        evidence_by_id[ref].get("value") or {}
        for ref in refs
        if ref in evidence_by_id
        and str(evidence_by_id[ref].get("kind") or "") == "materializer_contract"
        and isinstance(evidence_by_id[ref].get("value") or {}, Mapping)
    ]
    if not materializers:
        return True
    scope = str(edit.get("scope") or "")
    path = str(edit.get("path") or "")
    mechanism = str(edit.get("mechanism") or "")
    config_paths = {
        str(item)
        for item in (
            (workspace_contract.get("runtime_loading") or {}).get("config_paths") or []
        )
    }
    for ref in refs:
        evidence = evidence_by_id.get(ref) or {}
        if str(evidence.get("kind") or "") != "request_sentinel_probe":
            continue
        value = evidence.get("value") or {}
        if not isinstance(value, Mapping):
            continue
        contracts = value.get("workspace_edit_contracts") or {}
        observed = (
            contracts.get(evidence_subject) if isinstance(contracts, Mapping) else None
        )
        compared_fields = ["scope", "path", "mechanism"]
        if isinstance(observed, Mapping) and "key" in observed:
            compared_fields.append("key")
        if isinstance(observed, Mapping) and all(
            str(edit.get(field) or "") == str(observed.get(field) or "")
            for field in compared_fields
        ):
            return True
    for operation in materializers:
        kind = str(operation.get("kind") or "")
        if kind == "workspace_config" and all(
            str(edit.get(field) or "") == str(operation.get(field) or "")
            for field in ("scope", "path", "mechanism", "key")
        ):
            return True
        if (
            kind == "project_file"
            and scope == "project"
            and _query_path_matches(path, str(operation.get("path_pattern") or ""))
            and mechanism == "file"
        ):
            return True
        if (
            kind == "harness_config_patch"
            and mechanism == "config"
            and f"{scope}/{path}" in config_paths
        ):
            return True
        if (
            kind == "tool_schema_patch"
            and scope == "project"
            and path == str(operation.get("workspace_bridge_path") or "")
        ):
            return True
    return False


def _query_path_matches(path: str, pattern: str) -> bool:
    if not path or not pattern:
        return False
    expression = re.escape(pattern)
    for token in re.findall(r"<[^>]+>", pattern):
        replacement = ".+" if token == "<relative-path>" else "[^/]+"
        expression = expression.replace(re.escape(token), replacement)
    expression = expression.replace(r"\*", "[^/]+")
    return re.fullmatch(expression, path) is not None


def canonicalize_harness_query_output(
    output: Mapping[str, Any], payload: Mapping[str, Any]
) -> Mapping[str, Any]:
    if not isinstance(output, dict):
        return output
    probe = payload.get("architecture_probe") or {}
    output["harness"] = str(probe.get("harness_id") or output.get("harness") or "")
    output["harness_version"] = str(
        probe.get("harness_version") or output.get("harness_version") or ""
    )
    controller_mcp_ids = {
        str(item.get("id") or "")
        for item in payload.get("predefined_mcp_editable_points") or []
        if isinstance(item, Mapping) and item.get("id")
    }
    controller_mcp_ids.update(
        str(item.get("base_channel_id") or "")
        for item in payload.get("predefined_mcp_editable_points") or []
        if isinstance(item, Mapping) and item.get("base_channel_id")
    )
    for section in ("modifiable_modules", "unavailable_modules"):
        output[section] = [
            item
            for item in output.get(section) or []
            if not (
                isinstance(item, Mapping)
                and str(item.get("id") or "") in controller_mcp_ids
            )
        ]
    hypotheses = {
        str(item.get("id") or ""): item
        for item in payload.get("surface_hypotheses") or []
        if isinstance(item, Mapping) and item.get("id")
    }
    _enforce_native_channel_admission(
        output,
        hypotheses=hypotheses,
        harness=str(output["harness"]),
        controller_mcp_ids=controller_mcp_ids,
    )
    for module in output.get("modifiable_modules") or []:
        if not isinstance(module, dict):
            continue
        hypothesis = hypotheses.get(str(module.get("id") or "")) or {}
        trusted = hypothesis.get("trusted_edit_contract")
        if isinstance(trusted, Mapping):
            module["edit_contract"] = dict(trusted)
        selector = hypothesis.get("edit_selector")
        if isinstance(selector, Mapping) and isinstance(
            module.get("edit_contract"), Mapping
        ):
            edit_contract = dict(module["edit_contract"])
            edit_contract.update(
                {
                    str(key): str(value)
                    for key, value in selector.items()
                    if str(value).strip()
                }
            )
            module["edit_contract"] = edit_contract
    for section in ("unavailable_modules",):
        for item in output.get(section) or []:
            if (
                not isinstance(item, dict)
                or str(item.get("status") or "") == "modifiable"
            ):
                continue
            edit = item.get("edit_contract")
            if isinstance(edit, Mapping) and not str(edit.get("path") or "").strip():
                item.pop("edit_contract", None)
    output["mcp_editable_points"] = deepcopy(
        list(payload.get("predefined_mcp_editable_points") or [])
    )
    model_summary = str(output.get("summary") or "").strip()
    if model_summary:
        output["native_summary"] = model_summary
    output["summary"] = (
        f"{output['harness']} {output['harness_version']}: "
        f"{len(output.get('modifiable_modules') or [])} native modifiable surfaces, "
        f"{len(output.get('unavailable_modules') or [])} native unavailable surfaces; "
        f"controller attached {len(output['mcp_editable_points'])} MCP editable points."
    )
    return output


_ADMITTED_NATIVE_CHANNELS = {
    "opencode": {
        "instructions_rules": {
            "scope": "project",
            "path": "opencode.json",
            "mechanism": "config",
            "key": "instructions",
        },
        "skills": {
            "scope": "project",
            "path": ".opencode/skills/<slug>/SKILL.md",
            "mechanism": "file",
        },
        "system_prompt": {
            "scope": "project",
            "path": "opencode.json",
            "mechanism": "config",
            "key": "agent.build.prompt",
        },
        "agent_definitions": {
            "scope": "project",
            "path": ".opencode/agents/<name>.md",
            "mechanism": "file",
        },
    },
    "pi": {
        "project_instructions": {
            "scope": "project",
            "path": "AGENTS.md",
            "mechanism": "file",
        },
        "skills": {
            "scope": "project",
            "path": ".pi/skills/<slug>/SKILL.md",
            "mechanism": "file",
        },
        "system_prompt": {
            "scope": "project",
            "path": ".pi/APPEND_SYSTEM.md",
            "mechanism": "file",
        },
        "compaction_config": {
            "scope": "project",
            "path": ".pi/settings.json",
            "mechanism": "config",
            "key_prefix": "compaction.",
        },
    },
    "codex": {
        "developer_instructions": {
            "scope": "project",
            "path": ".codex/config.toml",
            "mechanism": "config",
            "key": "developer_instructions",
        },
        "project_instructions": {
            "scope": "project",
            "path": "AGENTS.md",
            "mechanism": "file",
        },
        "skills": {
            "scope": "project",
            "path": ".agents/skills/<slug>/SKILL.md",
            "mechanism": "file",
        },
        "hooks": {
            "scope": "project",
            "path": ".codex/harness-hook-context.md",
            "mechanism": "file",
        },
        "compaction": {
            "scope": "project",
            "path": ".codex/config.toml",
            "mechanism": "config",
            "key": "compact_prompt",
        },
    },
}

_NATIVE_CHANNEL_LANES = {
    "opencode": {
        "agent_definitions": "isolated_declarative",
    },
    "pi": {},
    "codex": {
        "hooks": "isolated_declarative",
    },
}

_SANDBOX_PENDING_CHANNELS = frozenset(
    {"plugin_tool_system", "custom_tools", "extensions", "hook_event_handlers"}
)

_NATIVE_CHANNEL_POLICY = {
    "opencode": {
        "agent_tool_config": "forbidden",
        "commands": "unsupported",
        "reference_sources": "conditional",
        "plugin_tool_system": "conditional",
        "custom_tools": "conditional",
        "experimental_policies": "forbidden",
    },
    "pi": {
        "extensions": "conditional",
    },
    "codex": {
        "hook_event_handlers": "conditional",
    },
}


def _enforce_native_channel_admission(
    output: dict[str, Any],
    *,
    hypotheses: Mapping[str, Mapping[str, Any]],
    harness: str,
    controller_mcp_ids: set[str],
) -> None:
    """Make channel admission controller-owned instead of model-selected."""

    admitted = _ADMITTED_NATIVE_CHANNELS.get(str(harness), {})
    lanes = _NATIVE_CHANNEL_LANES.get(str(harness), {})
    policy = _NATIVE_CHANNEL_POLICY.get(str(harness), {})
    model_items = {
        str(item.get("id") or ""): dict(item)
        for section in ("modifiable_modules", "unavailable_modules")
        for item in output.get(section) or []
        if isinstance(item, Mapping) and item.get("id")
    }
    modifiable: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    for channel_id, hypothesis in hypotheses.items():
        if channel_id in controller_mcp_ids:
            continue
        source = model_items.get(channel_id, {})
        refs = [
            str(item)
            for item in (
                hypothesis.get("evidence_refs") or source.get("evidence_refs") or []
            )
        ]
        if channel_id in admitted:
            edit_contract = dict(admitted[channel_id])
            verification = hypothesis.get("verification") or {}
            evidence_level = (
                "runtime_observed"
                if isinstance(verification, Mapping)
                and verification.get("runtime_observed")
                else "materialized"
            )
            modifiable.append(
                {
                    "id": channel_id,
                    "status": "modifiable",
                    "evidence_level": evidence_level,
                    "visibility": str(
                        hypothesis.get("visibility") or "runtime context"
                    ),
                    "use": str(hypothesis.get("use") or "Candidate guidance."),
                    "edit_contract": edit_contract,
                    "execution_lane": str(lanes.get(channel_id) or "direct"),
                    "verification_required": (
                        "runtime_request_or_event"
                        if channel_id in {"system_prompt", "agent_definitions", "hooks"}
                        else (
                            "effective_runtime_config"
                            if channel_id in {"compaction", "compaction_config"}
                            else "runtime_load"
                        )
                    ),
                    "runtime_constraints": list(
                        source.get("runtime_constraints")
                        or ["Use the controller-owned candidate workspace contract."]
                    ),
                    "risks": str(hypothesis.get("risks") or "Broad behavioral impact."),
                    "evidence_refs": refs,
                }
            )
            continue
        status = str(
            policy.get(channel_id) or hypothesis.get("policy_status") or "conditional"
        )
        if status not in {"conditional", "unsupported", "forbidden"}:
            status = "conditional"
        unavailable_item = {
            "id": channel_id,
            "status": status,
            "reason": (
                "The controller does not admit this channel without a dedicated "
                "runtime verifier and preflight checker."
            ),
            "runtime_constraints": [
                "Documented or materialized presence alone is insufficient for admission."
            ],
            "evidence_refs": refs,
        }
        if channel_id in _SANDBOX_PENDING_CHANNELS and status == "conditional":
            unavailable_item.update(
                {
                    "execution_lane": "sandboxed_executable",
                    "verification_required": "isolated_load_and_trigger_trace",
                    "runtime_constraints": [
                        "The target process must receive no host credentials or evaluator paths.",
                        "The candidate executable must run inside a trial-scoped filesystem and network boundary.",
                        "A preflight trace must prove both native loading and event/tool invocation.",
                    ],
                }
            )
        unavailable.append(unavailable_item)
    output["modifiable_modules"] = modifiable
    output["unavailable_modules"] = unavailable


def _conservative_harness_query_output(
    output: Mapping[str, Any], payload: Mapping[str, Any]
) -> Mapping[str, Any]:
    normalized = dict(
        canonicalize_harness_query_output(deepcopy(dict(output)), payload)
    )
    model_payload = _harness_query_model_payload(payload)
    hypotheses = {
        str(item["id"]): item
        for item in model_payload.get("surface_hypotheses") or []
        if isinstance(item, Mapping) and item.get("id")
    }
    evidence_catalog = _query_evidence_with_input_paths(model_payload)
    evidence_by_id = {
        str(item["id"]): item
        for item in evidence_catalog
        if isinstance(item, Mapping) and item.get("id")
    }
    known_evidence = set(evidence_by_id)
    workspace_contract = model_payload.get("candidate_workspace_contract") or {}
    valid_modifiable: list[dict[str, Any]] = []
    valid_unavailable: list[dict[str, Any]] = []
    classified: set[str] = set()

    def fallback(channel_id: str, error: str) -> dict[str, Any]:
        hypothesis = hypotheses[channel_id]
        refs = [
            str(ref)
            for ref in hypothesis.get("evidence_refs") or []
            if str(ref) in known_evidence
        ]
        if not refs:
            refs = [sorted(known_evidence)[0]]
        policy_status = str(hypothesis.get("policy_status") or "")
        forbidden = policy_status == "forbidden"
        return {
            "id": channel_id,
            "status": (
                "forbidden"
                if forbidden
                else (
                    policy_status
                    if policy_status in {"conditional", "unsupported"}
                    else "conditional"
                )
            ),
            "reason": (
                "The fixed experiment forbids this harness surface."
                if forbidden
                else "Harness Query did not establish a safe, effective edit contract."
            ),
            "runtime_constraints": [
                str(error).strip() or "No supported edit contract was established."
            ],
            "evidence_refs": refs,
        }

    for item in normalized.get("modifiable_modules") or []:
        if not isinstance(item, Mapping):
            continue
        candidate = dict(item)
        channel_id = str(candidate.get("id") or "")
        if not channel_id or channel_id in classified:
            continue
        contract = hypotheses.get(channel_id) or {}
        try:
            policy_status = str(contract.get("policy_status") or "")
            if policy_status in {"forbidden", "conditional", "unsupported"}:
                raise ValueError(
                    f"policy-{policy_status} harness surface cannot be modifiable"
                )
            trusted_edit_contract = contract.get("trusted_edit_contract")
            if isinstance(trusted_edit_contract, Mapping):
                candidate["edit_contract"] = dict(trusted_edit_contract)
            _validate_modifiable_surface(
                candidate,
                known_evidence=known_evidence,
                evidence_by_id=evidence_by_id,
                evidence_subject=channel_id,
                workspace_contract=workspace_contract,
            )
        except ValueError as exc:
            if channel_id in hypotheses:
                valid_unavailable.append(fallback(channel_id, str(exc)))
                classified.add(channel_id)
        else:
            selector = contract.get("edit_selector")
            if isinstance(selector, Mapping):
                edit_contract = dict(candidate["edit_contract"])
                edit_contract.update(
                    {
                        str(key): str(value)
                        for key, value in selector.items()
                        if str(value).strip()
                    }
                )
                candidate["edit_contract"] = edit_contract
            valid_modifiable.append(candidate)
            classified.add(channel_id)

    for item in normalized.get("unavailable_modules") or []:
        if not isinstance(item, Mapping):
            continue
        candidate = dict(item)
        channel_id = str(candidate.get("id") or "")
        if not channel_id or channel_id in classified:
            continue
        contract = hypotheses.get(channel_id) or {}
        try:
            _validate_unavailable_surface(candidate, known_evidence=known_evidence)
            if (
                str(contract.get("policy_status") or "") == "forbidden"
                and str(candidate.get("status") or "") != "forbidden"
            ):
                raise ValueError(
                    "policy-forbidden harness surface must remain forbidden"
                )
        except ValueError as exc:
            if channel_id in hypotheses:
                valid_unavailable.append(fallback(channel_id, str(exc)))
                classified.add(channel_id)
        else:
            valid_unavailable.append(candidate)
            classified.add(channel_id)

    for channel_id in hypotheses:
        if channel_id not in classified:
            valid_unavailable.append(
                fallback(
                    channel_id, "The model did not classify this supplied hypothesis."
                )
            )
    normalized["modifiable_modules"] = valid_modifiable
    normalized["unavailable_modules"] = valid_unavailable
    if not isinstance(normalized.get("base_profile"), Mapping):
        normalized["base_profile"] = {
            "shared_rules": [],
            "role_isolation": "The controller retains fixed experiment authority.",
        }
    return canonicalize_harness_query_output(normalized, payload)


def _mcp_editable_points(
    task_input: Mapping[str, Any], *, harness_id: str = "opencode"
) -> list[dict[str, Any]]:
    environment = task_input.get("environment")
    if not isinstance(environment, Mapping):
        return []
    transport = environment.get("tool_transport")
    if not isinstance(transport, Mapping) or transport.get("kind") != "mcp":
        return []
    server_id = str(transport.get("server_id") or "").strip()
    tools = [
        item for item in environment.get("tools") or [] if isinstance(item, Mapping)
    ]
    tool_names = [str(item.get("name") or "") for item in tools if item.get("name")]
    parameter_targets = [
        {
            "tool": str(item.get("name")),
            "parameters": _public_tool_parameter_names(item.get("parameters")),
        }
        for item in tools
        if item.get("name")
    ]
    return [
        {
            "id": "mcp_tool_description",
            "base_channel_id": "tool_description",
            "server_id": server_id,
            "targets": tool_names,
            "status": "modifiable",
            "evidence_level": "materialized",
            "visibility": "whenever the connected MCP tool is available",
            "use": "Compact call-selection guidance for one exact MCP tool.",
            "edit_contract": {
                "scope": "project",
                "path": MCP_PATCH_WORKSPACE_PATH,
                "mechanism": "mcp",
            },
            "runtime_constraints": [
                "Use exact raw tool names from targets; the controller applies the patch before tools/list."
            ],
            "risks": "Long descriptions bloat every decision involving the tool.",
            "evidence_refs": ["runtime:mcp-workspace-bridge"],
            "operation": {
                "kind": "tool_schema_patch",
                "manifest_field": "tool_desc_patches",
                "workspace_bridge_path": MCP_PATCH_WORKSPACE_PATH,
                "target": "description",
                "harness": str(harness_id),
            },
            "artifact_contract": (
                f"{MCP_PATCH_WORKSPACE_PATH}: " "{<raw_tool_name>:{desc:<description>}}"
            ),
        },
        {
            "id": "mcp_tool_parameter_description",
            "base_channel_id": "tool_parameter_description",
            "server_id": server_id,
            "targets": parameter_targets,
            "status": "modifiable",
            "evidence_level": "materialized",
            "visibility": "whenever the connected MCP parameter schema is available",
            "use": "Compact meaning or constraint for one exact MCP parameter.",
            "edit_contract": {
                "scope": "project",
                "path": MCP_PATCH_WORKSPACE_PATH,
                "mechanism": "mcp",
            },
            "runtime_constraints": [
                "Use exact raw tool and parameter names from targets; descriptions cannot enforce values."
            ],
            "risks": "Incorrect parameter guidance can systematically corrupt tool calls.",
            "evidence_refs": ["runtime:mcp-workspace-bridge"],
            "operation": {
                "kind": "tool_schema_patch",
                "manifest_field": "tool_desc_patches",
                "workspace_bridge_path": MCP_PATCH_WORKSPACE_PATH,
                "target": "parameter_description",
                "harness": str(harness_id),
            },
            "artifact_contract": (
                f"{MCP_PATCH_WORKSPACE_PATH}: "
                "{<raw_tool_name>:{params:{<parameter>:<description>}}}"
            ),
        },
    ]


def _public_tool_parameter_names(raw: Any) -> list[str]:
    """Normalize public tool parameters from JSON Schema or compact catalogs."""

    if not isinstance(raw, Mapping):
        return []
    properties = raw.get("properties")
    source = properties if isinstance(properties, Mapping) else raw
    return sorted(
        str(name)
        for name in source
        if str(name).strip() and str(name) not in {"type", "required", "$schema"}
    )
