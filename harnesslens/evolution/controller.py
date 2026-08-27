from __future__ import annotations

import json
import os
import shutil
import fcntl
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from harnesslens.evolution.analyzer import AnalyzerModule
from harnesslens.core.artifacts import write_json
from harnesslens.evolution.baseline import ensure_baseline_event
from harnesslens.core.budget import (
    DEFAULT_TOTAL_CREATION_BUDGET,
    CreationBudget,
)
from harnesslens.benchmarks.cell_config import benchmark_config
from harnesslens.harnesses.channel_preflight import (
    ChannelPreflightError,
    validate_channel_preflight,
)
from harnesslens.evolution.discovery import DiscoveryModule
from harnesslens.evolution.experience import ExperienceModule
from harnesslens.evolution.incumbent import load_incumbent_candidate
from harnesslens.evolution.main_agent import (
    CandidateMaterializationError,
    ITERATION_RETRY_BUFFER_CREATIONS,
    MIN_CANDIDATE_ITERATION_CREATIONS,
    MIN_RESIDUAL_ROLLOUT_TASKS,
    MIN_STANDARD_ROLLOUT_TASKS,
    MainDecision,
    MainAgentModule,
    confirmation_mode,
    normalize_promotion_metric,
    paired_confirmation_creation_cost,
    paired_screen_creation_cost,
)
from harnesslens.evolution.rollout import PairedRolloutResult, RolloutModule
from harnesslens.evaluation.rollout_bridge import CellHarnessRepository
from harnesslens.benchmarks.task_data import BaselineDataset
from harnesslens.core.train_protocol import TRAIN_ROLLOUT_REPEATS
from harnesslens.core.workflow_fingerprint import (
    assert_workflow_fingerprint,
    establish_workflow_fingerprint,
)


TOTAL_CREATION_BUDGET = int(
    os.environ.get("HAI_TOTAL_CREATION_BUDGET", DEFAULT_TOTAL_CREATION_BUDGET)
)
MIN_ITERATION_START_CREATIONS = (
    MIN_CANDIDATE_ITERATION_CREATIONS + ITERATION_RETRY_BUFFER_CREATIONS
)
ANALYSIS_REUSE_WORKFLOW_FILES = (
    "harnesslens/evolution/analyzer.py",
    "harnesslens/core/artifacts.py",
    "harnesslens/benchmarks/cell_config.py",
    "harnesslens/evolution/discovery.py",
    "harnesslens/evolution/experience.py",
    "harnesslens/harnesses/harness_query_adapters.py",
    "harnesslens/harnesses/harness_workspace.py",
    "harnesslens/benchmarks/task_data.py",
)


@dataclass(frozen=True)
class ControllerResult:
    selected_version: str
    submission_path: str
    budget: dict


class IterationController:
    def __init__(
        self,
        *,
        repo_root: str | Path,
        run_id: str,
        harness: str = "opencode",
        cell: str = "retail",
        incumbent_submissions: tuple[str | Path, ...] = (),
        promotion_metric: str = "pass_at_1",
        analysis_source_run: str | Path | None = None,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.run_root = self.repo_root / "runs" / "train" / str(run_id)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.harness = str(harness)
        self.promotion_metric = normalize_promotion_metric(promotion_metric)
        self.config = benchmark_config(self.repo_root, cell)
        self.analysis_source_run = (
            _resolve_analysis_source_run(self.repo_root, analysis_source_run)
            if analysis_source_run is not None
            else None
        )
        self.incumbent_submissions = tuple(
            Path(item).resolve() for item in incumbent_submissions
        )
        self.budget = CreationBudget(
            self.run_root / "creation_budget.json",
            total=TOTAL_CREATION_BUDGET,
            baseline_used=len(self.config.train_task_ids) * TRAIN_ROLLOUT_REPEATS,
        )

    def run(self, *, baseline_event: str | Path | None = None) -> ControllerResult:
        with self._controller_lock():
            recovered = self.budget.recover_interrupted_jobs(
                reason="controller process restarted"
            )
            if recovered:
                write_json(
                    self.run_root / "controller_recovery.json",
                    {"recovered_jobs": recovered},
                )
            return self._run_locked(baseline_event=baseline_event)

    def _run_locked(
        self, *, baseline_event: str | Path | None = None
    ) -> ControllerResult:
        workflow = establish_workflow_fingerprint(
            repo_root=self.repo_root,
            run_root=self.run_root,
        )
        workflow_sha256 = str(workflow["sha256"])
        try:
            baseline = ensure_baseline_event(
                repo_root=self.repo_root,
                run_root=self.run_root,
                baseline_event=baseline_event,
                cell=self.config.cell,
                harness=self.harness,
            )
        except ValueError as exc:
            if (
                str(exc) != "baseline event runtime fingerprint differs from this run"
                or not _can_finalize_frozen_run(self.run_root, self.budget.status())
                or baseline_event is None
            ):
                raise
            baseline = Path(baseline_event).resolve()
            BaselineDataset.from_ingest_event(baseline)
        self._checkpoint("baseline_ready", {"baseline_event": str(baseline)})
        self._assert_workflow_source(workflow_sha256)
        reusing_analysis = self.analysis_source_run is not None
        if self.analysis_source_run is not None:
            reused_creations = prepare_analysis_reuse(
                source_run=self.analysis_source_run,
                target_run=self.run_root,
                baseline_event=baseline,
                harness=self.harness,
                cell=self.config.cell,
                workflow_fingerprint=workflow,
            )
            self.budget.import_settled_usage(
                "analysis-reuse",
                creation_count=reused_creations,
                metadata={"source_run": str(self.analysis_source_run)},
            )
        DiscoveryModule(
            repo_root=self.repo_root,
            run_root=self.run_root,
            budget=self.budget,
            harness=self.harness,
            cell=self.config.cell,
        ).run(baseline_event=baseline)
        analysis_budget_before = (
            self.budget.status()["created"] if reusing_analysis else None
        )
        self._checkpoint("discovery_complete", {})
        self._assert_workflow_source(workflow_sha256)
        ExperienceModule(
            repo_root=self.repo_root,
            run_root=self.run_root,
            budget=self.budget,
            harness=self.harness,
            cell=self.config.cell,
        ).run_baseline(baseline_event=baseline, label="baseline")
        self._checkpoint("experience_complete", {})
        self._assert_workflow_source(workflow_sha256)
        AnalyzerModule(
            run_root=self.run_root, budget=self.budget, harness=self.harness
        ).run(label="baseline")
        if (
            analysis_budget_before is not None
            and self.budget.status()["created"] != analysis_budget_before
        ):
            raise RuntimeError(
                "reused analysis is incompatible with the current workflow; "
                "refusing to mix cached and regenerated analysis"
            )

        self._checkpoint("analyzer_complete", {})
        self._assert_workflow_source(workflow_sha256)
        main = MainAgentModule(
            repo_root=self.repo_root,
            run_root=self.run_root,
            budget=self.budget,
            harness=self.harness,
            cell=self.config.cell,
            promotion_metric=self.promotion_metric,
        )
        attempted: list[str] = []
        accepted: list[dict[str, Any]] = []
        revision_candidates: list[dict[str, Any]] = []
        incumbent_candidates = [
            load_incumbent_candidate(
                path,
                cell=self.config.cell,
                harness=self.harness,
                train_task_ids=self.config.train_task_ids,
            )
            for path in self.incumbent_submissions
        ]
        repository = CellHarnessRepository(
            cell=self.config.cell,
            repo_root=self.repo_root,
            run_id=self.run_root.name,
            evidence_root=self.run_root / "rollout_evidence",
            harness=self.harness,
        )
        current_version, champion_context, imported_champion_ids = initialize_champion(
            repository=repository,
            incumbent_candidates=incumbent_candidates,
        )
        history: list[dict[str, Any]] = []
        iteration = 1
        while True:
            self._assert_workflow_source(workflow_sha256)
            label = f"iteration-{iteration:02d}"
            resuming = (self.run_root / "main_agent" / f"{label}.json").exists()
            if not resuming and self.budget.status()[
                "remaining"
            ] < minimum_screen_start_creations(current_version):
                break
            try:
                decision = main.decide_and_materialize(
                    label=label,
                    candidate_label=f"candidate-{iteration:02d}",
                    analyzer_label="baseline",
                    parent_version=current_version,
                    attempted_candidate_ids=attempted,
                    accepted_candidates=[*champion_context, *accepted],
                    additional_candidates=[
                        *revision_candidates,
                    ],
                    iteration_history=history,
                )
            except CandidateMaterializationError as exc:
                attempted.append(exc.candidate_id)
                record = {
                    "iteration": iteration,
                    "candidate_id": exc.candidate_id,
                    "origin_candidate_id": exc.candidate_id,
                    "parent_version": current_version,
                    "candidate_version": "",
                    "evaluation_mode": "materialization",
                    "promotion_eligible": False,
                    "promotion_metric": self.promotion_metric,
                    "rollout_task_ids": [],
                    "rollout_metrics": {},
                    "review_decision": "materialization_failed",
                    "selected_version": current_version,
                    "review_rationale": exc.reason,
                    "review_evidence": {
                        "recovered_task_ids": [],
                        "preserved_task_ids": [],
                        "attributable_regression_task_ids": [],
                        "unresolved_findings": [exc.reason],
                    },
                    "revision_candidate_id": None,
                }
                history.append(record)
                self._checkpoint(
                    f"{label}_materialization_failed",
                    {"record": record, "accepted_count": len(accepted)},
                )
                iteration += 1
                continue
            except RuntimeError as exc:
                if "no unattempted candidate" in str(exc) or (
                    "cannot fund a complete cumulative iteration" in str(exc)
                ):
                    break
                raise
            assert_candidate_extends_champion(
                current_version=current_version,
                decision=decision,
            )
            source_id = str(decision.output["candidate"]["source_candidate_ids"][0])
            attempted.append(source_id)
            rollout_module = RolloutModule(
                repo_root=self.repo_root,
                run_root=self.run_root,
                budget=self.budget,
                cell=self.config.cell,
                harness=self.harness,
            )
            preflight = rollout_module.run_channel_preflight_from_main(
                main_decision=decision.output_path,
                label=f"{label}-{decision.harness_version}-channel-preflight",
            )
            try:
                preflight_report = validate_channel_preflight(
                    decision=decision.output,
                    rollout=preflight.output,
                    output_path=(
                        self.run_root
                        / "channel_preflight"
                        / f"{label}-{decision.harness_version}.json"
                    ),
                )
            except ChannelPreflightError as exc:
                record = {
                    "iteration": iteration,
                    "candidate_id": source_id,
                    "origin_candidate_id": source_id,
                    "parent_version": current_version,
                    "candidate_version": decision.harness_version,
                    "evaluation_mode": "channel_preflight",
                    "promotion_eligible": False,
                    "promotion_metric": self.promotion_metric,
                    "rollout_task_ids": list(
                        preflight.output.get("requested_task_ids") or []
                    ),
                    "rollout_metrics": dict(preflight.output.get("metrics") or {}),
                    "review_decision": "channel_preflight_failed",
                    "selected_version": current_version,
                    "review_rationale": str(exc),
                    "review_evidence": {
                        "recovered_task_ids": [],
                        "preserved_task_ids": [],
                        "attributable_regression_task_ids": [],
                        "unresolved_findings": [str(exc)],
                        "channel_preflight": exc.report,
                    },
                    "revision_candidate_id": None,
                }
                history.append(record)
                self._checkpoint(
                    f"{label}_channel_preflight_failed",
                    {"record": record, "accepted_count": len(accepted)},
                )
                iteration += 1
                continue
            self._checkpoint(
                f"{label}_channel_preflight_complete",
                {
                    "rollout": preflight.output_path,
                    "report": preflight_report,
                },
            )
            rollout_pair = rollout_module.run_pair_from_main(
                main_decision=decision.output_path,
                label=f"{label}-{decision.harness_version}",
            )
            rollout = rollout_pair.candidate
            self._assert_workflow_source(workflow_sha256)
            self._checkpoint(
                f"{label}_rollout_complete",
                {
                    "rollout": rollout.output_path,
                    "parent_rollout": (
                        rollout_pair.parent.output_path if rollout_pair.parent else None
                    ),
                },
            )
            evaluation_mode = str(decision.output.get("evaluation_mode") or "standard")
            post_label = f"{label}-post"
            screen_review = self._review_pair(
                baseline_event=baseline,
                main=main,
                decision=decision,
                rollout_pair=rollout_pair,
                comparison_label=f"{label}-comparison",
                label=post_label,
                review_label=f"{label}-review",
                base_version=current_version,
                require_primary_metric_improvement=(
                    evaluation_mode == "terminal_screen"
                ),
            )
            review = screen_review
            confirmation_pair: PairedRolloutResult | None = None
            if (
                evaluation_mode == "residual_probe"
                and str(screen_review.output.get("decision") or "") == "accept_delta"
            ):
                review = defer_residual_candidate(
                    screen_review=screen_review,
                    parent_version=current_version,
                    output_path=(
                        self.run_root / "main_agent" / f"{label}-residual-deferred.json"
                    ),
                )
            elif str(screen_review.output.get("decision") or "") in {
                "accept_delta",
                "confirm_delta",
            } and _evaluation_requires_confirmation(evaluation_mode):
                task_count = len(rollout.output["requested_task_ids"])
                required = (
                    confirmation_creation_cost(
                        task_count, parent_version=current_version
                    )
                    + ITERATION_RETRY_BUFFER_CREATIONS
                )
                confirmation_label = f"{label}-confirm-{decision.harness_version}"
                confirmation_cached = _paired_rollout_outputs_exist(
                    run_root=self.run_root,
                    label=confirmation_label,
                    parent_version=current_version,
                )
                if confirmation_cached or self.budget.status()["remaining"] >= required:
                    confirmation_task_ids = select_confirmation_task_ids(
                        run_root=self.run_root,
                        decision=decision,
                        screen_review=screen_review,
                        screen_task_ids=rollout.output["requested_task_ids"],
                        task_count=task_count,
                    )
                    confirmation_pair = rollout_module.run_pair_from_main(
                        main_decision=decision.output_path,
                        label=confirmation_label,
                        pairing_offset=TRAIN_ROLLOUT_REPEATS,
                        task_ids_override=confirmation_task_ids,
                    )
                    self._checkpoint(
                        f"{label}_confirmation_rollout_complete",
                        {
                            "rollout": confirmation_pair.candidate.output_path,
                            "parent_rollout": (
                                confirmation_pair.parent.output_path
                                if confirmation_pair.parent
                                else None
                            ),
                            "task_ids": list(confirmation_task_ids),
                        },
                    )
                    post_label = f"{label}-confirmation-post"
                    review = self._review_pair(
                        baseline_event=baseline,
                        main=main,
                        decision=decision,
                        rollout_pair=confirmation_pair,
                        comparison_label=f"{label}-confirmation-comparison",
                        label=post_label,
                        review_label=f"{label}-confirmation-review",
                        base_version=current_version,
                        require_primary_metric_improvement=True,
                    )
                else:
                    review = reject_unconfirmed_candidate(
                        screen_review=screen_review,
                        parent_version=current_version,
                        output_path=(
                            self.run_root
                            / "main_agent"
                            / f"{label}-confirmation-unavailable.json"
                        ),
                    )
            if _evaluation_requires_confirmation(evaluation_mode):
                assert_confirmed_promotion(
                    decision=decision,
                    review=review,
                    confirmation_pair=confirmation_pair,
                )
            effective_rollout = (
                confirmation_pair.candidate if confirmation_pair else rollout
            )
            record = {
                "iteration": iteration,
                "candidate_id": source_id,
                "origin_candidate_id": decision.output["candidate"].get(
                    "origin_candidate_id", source_id
                ),
                "parent_version": current_version,
                "candidate_version": decision.harness_version,
                "candidate_side": decision.output.get("selected_candidate_side"),
                "evaluation_mode": evaluation_mode,
                "promotion_eligible": bool(
                    decision.output.get("promotion_eligible", True)
                ),
                "promotion_metric": self.promotion_metric,
                "channel_diffs": decision.output["candidate"]["channel_diffs"],
                "workspace_diff": decision.output["candidate"].get(
                    "workspace_diff", []
                ),
                "workspace_delta": decision.output["candidate"].get(
                    "workspace_delta", {"files": []}
                ),
                "manifest_delta": decision.output["candidate"].get(
                    "manifest_delta", {}
                ),
                "workspace_sha256": decision.output["candidate"].get(
                    "workspace_sha256", ""
                ),
                "editor_summary": decision.output["candidate"].get("editor_summary"),
                "rollout_task_ids": effective_rollout.output["requested_task_ids"],
                "rollout_task_roles": dict(
                    (decision.output.get("rollout_request") or {}).get("task_roles")
                    or {}
                ),
                "review_decision": review.output["decision"],
                "selected_version": review.harness_version,
                "main_decision": decision.output_path,
                "rollout_output": effective_rollout.output_path,
                "post_label": post_label,
                "rollout_metrics": dict(effective_rollout.output["metrics"]),
                "review_rationale": review.output.get("rationale"),
                "review_evidence": review.output.get("evidence"),
                "screen_rollout_output": rollout.output_path,
                "screen_rollout_task_ids": rollout.output["requested_task_ids"],
                "screen_rollout_metrics": dict(rollout.output["metrics"]),
                "screen_review_decision": screen_review.output.get("decision"),
                "confirmation_rollout_output": (
                    confirmation_pair.candidate.output_path
                    if confirmation_pair
                    else None
                ),
                "confirmation_rollout_task_ids": (
                    confirmation_pair.candidate.output["requested_task_ids"]
                    if confirmation_pair
                    else None
                ),
                "confirmation_review_decision": (
                    review.output.get("decision") if confirmation_pair else None
                ),
                "revision_candidate_id": (
                    review.output.get("revision_candidate") or {}
                ).get("id"),
                "replan_candidate_id": None,
            }
            history.append(record)
            current_version, advanced = advance_cumulative_version(
                current_version=current_version,
                decision=decision,
                review=review,
            )
            if advanced:
                accepted.append(
                    {
                        **record,
                        "manifest_delta": decision.output["candidate"][
                            "manifest_delta"
                        ],
                        "origin_candidate_id": decision.output["candidate"].get(
                            "origin_candidate_id", source_id
                        ),
                    }
                )
            revision = revision_candidate_from_review(
                review.output,
                rollout_task_ids=effective_rollout.output["requested_task_ids"],
            )
            if revision is not None:
                revision_candidates.append(revision)
            post_adjustment = json.loads(
                (
                    self.run_root / "analyzer" / f"{post_label}_adjustment.json"
                ).read_text(encoding="utf-8")
            )
            replan = replan_candidate_from_review(
                review.output,
                adjustment=post_adjustment,
                rollout_task_ids=effective_rollout.output["requested_task_ids"],
            )
            if replan is not None:
                revision_candidates.append(replan)
                record["replan_candidate_id"] = replan["id"]
            self._checkpoint(
                f"{label}_reviewed",
                {"record": record, "accepted_count": len(accepted)},
            )
            iteration += 1

        self._assert_workflow_source(workflow_sha256)
        selected_version = current_version
        selected_candidate_ids = {
            str(candidate_id)
            for item in accepted
            for candidate_id in (
                item["candidate_id"],
                item.get("origin_candidate_id", item["candidate_id"]),
            )
        } | imported_champion_ids
        dispositions_path = self._finalize_dispositions(
            selected_candidate_ids=selected_candidate_ids
        )
        final = main.publish_selected(
            selected_version=selected_version,
            iteration_history=history,
            dispositions_path=dispositions_path,
        )
        self._checkpoint(
            "finalized",
            {
                "selected_version": final.harness_version,
                "submission": final.output_path,
            },
        )
        return ControllerResult(
            selected_version=final.harness_version,
            submission_path=final.output_path,
            budget=self.budget.status(),
        )

    @contextmanager
    def _controller_lock(self) -> Iterator[None]:
        path = self.run_root / "controller.lock"
        with path.open("a+", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    f"another controller is already active for run {self.run_root.name}"
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _assert_workflow_source(self, expected_sha256: str) -> None:
        assert_workflow_fingerprint(
            repo_root=self.repo_root,
            run_root=self.run_root,
            expected_sha256=expected_sha256,
        )

    def _review_pair(
        self,
        *,
        baseline_event: str | Path,
        main: MainAgentModule,
        decision: MainDecision,
        rollout_pair: PairedRolloutResult,
        comparison_label: str,
        label: str,
        review_label: str,
        base_version: str,
        require_primary_metric_improvement: bool,
    ) -> MainDecision:
        rollout = rollout_pair.candidate
        comparison = ExperienceModule(
            repo_root=self.repo_root,
            run_root=self.run_root,
            budget=self.budget,
            harness=self.harness,
            cell=self.config.cell,
        ).run_comparison(
            baseline_event=baseline_event,
            rollout_output=rollout.output_path,
            label=comparison_label,
            reference_rollout_output=(
                rollout_pair.parent.output_path if rollout_pair.parent else None
            ),
        )
        AnalyzerModule(
            run_root=self.run_root,
            budget=self.budget,
            harness=self.harness,
        ).run_post_rollout(
            comparison_label=comparison_label,
            main_decision=decision.output_path,
            rollout_output=rollout.output_path,
            label=label,
        )
        return main.finalize(
            tested_main_decision=decision.output_path,
            post_label=label,
            rollout_output=rollout.output_path,
            reference_rollout_metrics=reference_metrics_from_comparison_index(
                comparison.source_index_path
            ),
            require_primary_metric_improvement=require_primary_metric_improvement,
            label=review_label,
            base_version=base_version,
            publish=False,
        )

    def _finalize_dispositions(self, *, selected_candidate_ids: set[str]) -> str:
        source = self.run_root / "analyzer" / "baseline_experience_dispositions.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        for item in payload["items"]:
            candidates = {str(value) for value in item.get("candidate_ids") or []}
            selected = sorted(candidates & selected_candidate_ids)
            if selected:
                item["status"] = "materialized"
                item["candidate_ids"] = selected
                item["reason"] = "Included in the accepted cumulative version chain."
            else:
                item["status"] = "deferred"
                item["reason"] = (
                    "Candidate was not accepted or was not reached within the creation budget."
                    if candidates
                    else "No evidence-bounded candidate selected this experience yet."
                )
        output = write_json(
            self.run_root / "submission" / "experience_dispositions.json",
            payload,
        )
        return str(output)

    def _checkpoint(self, stage: str, details: dict) -> None:
        write_json(
            self.run_root / "controller_state.json",
            {
                "stage": str(stage),
                "details": dict(details),
                "budget": self.budget.status(),
            },
        )


def _resolve_analysis_source_run(repo_root: Path, source: str | Path) -> Path:
    path = Path(source)
    if not path.is_absolute():
        path = repo_root / "runs" / "train" / path
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError(f"analysis source run does not exist: {resolved}")
    return resolved


def prepare_analysis_reuse(
    *,
    source_run: str | Path,
    target_run: str | Path,
    baseline_event: str | Path,
    harness: str,
    cell: str,
    workflow_fingerprint: Mapping[str, Any],
) -> int:
    source = Path(source_run).resolve()
    target = Path(target_run).resolve()
    if source == target:
        raise ValueError("analysis source run must differ from the target run")
    source_workflow_path = source / "workflow_fingerprint.json"
    if not source_workflow_path.is_file():
        raise ValueError("analysis source is missing workflow provenance")
    source_workflow = json.loads(source_workflow_path.read_text(encoding="utf-8"))
    source_analysis_files = _analysis_workflow_files(source_workflow, side="source")
    current_analysis_files = _analysis_workflow_files(
        workflow_fingerprint, side="current"
    )
    if source_analysis_files != current_analysis_files:
        raise ValueError("analysis workflow differs between source and current runs")
    baseline = BaselineDataset.from_ingest_event(baseline_event)
    source_index_path = source / "experience" / "baseline_source_index.json"
    source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
    if set(str(item) for item in source_index.get("task_ids") or []) != set(
        baseline.task_ids
    ):
        raise ValueError("analysis source uses a different TRAIN task set")
    if set(str(item) for item in source_index.get("evidence_refs") or []) != set(
        baseline.evidence_by_path.values()
    ):
        raise ValueError("analysis source does not use the exact baseline trajectories")

    harness_query = json.loads(
        (source / "discovery" / "harness_query.json").read_text(encoding="utf-8")
    )
    if str(harness_query.get("harness") or "") != str(harness):
        raise ValueError("analysis source uses a different harness")
    task_input = json.loads(
        (source / "discovery" / "task_explorer_input.json").read_text(encoding="utf-8")
    )
    if str(task_input.get("domain") or "") != str(cell):
        raise ValueError("analysis source uses a different benchmark cell")

    required = [
        source / "discovery" / "task_explorer.json",
        source / "experience" / "baseline.json",
        source_index_path,
        source / "experience" / "current.json",
        source / "analyzer" / "baseline_reusable.json",
        source / "analyzer" / "baseline_adjustment.json",
        source / "analyzer" / "baseline_experience_dispositions.json",
        source / "creation_budget.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("analysis source is incomplete: " + ", ".join(missing))

    copy_groups = {
        "discovery": [
            source / "discovery" / "task_explorer.json",
            source / "discovery" / "harness_query.json",
        ],
        "experience": [
            *sorted((source / "experience").glob("baseline*.json")),
            source / "experience" / "current.json",
        ],
        "analyzer": sorted((source / "analyzer").glob("baseline*.json")),
    }
    for section, paths in copy_groups.items():
        destination = target / section
        destination.mkdir(parents=True, exist_ok=True)
        for path in paths:
            shutil.copy2(path, destination / path.name)

    source_budget = json.loads(
        (source / "creation_budget.json").read_text(encoding="utf-8")
    )
    prefixes = (
        "discovery-task-explorer",
        "discovery-harness-query",
        "experience-baseline",
        "analyzer-adjustment-baseline",
        "analyzer-reusable-baseline",
        "analyzer-reusable-plan-baseline",
    )
    reused_creations = sum(
        int(item.get("creation_count") or 0)
        for job_id, item in (source_budget.get("jobs") or {}).items()
        if str(job_id).startswith(prefixes)
        and str(item.get("status") or "") in {"launched", "settled"}
    )
    if reused_creations <= 0:
        raise ValueError("analysis source has no chargeable initial analysis usage")
    write_json(
        target / "analysis_reuse.json",
        {
            "source_run": str(source),
            "baseline_evidence_refs": sorted(baseline.evidence_by_path.values()),
            "harness": str(harness),
            "cell": str(cell),
            "reused_creations": reused_creations,
            "analysis_workflow_files": current_analysis_files,
        },
    )
    return reused_creations


def _analysis_workflow_files(
    workflow: Mapping[str, Any], *, side: str
) -> dict[str, str]:
    files = workflow.get("files")
    if not isinstance(files, Mapping):
        raise ValueError(f"{side} workflow fingerprint has no file inventory")
    missing = [path for path in ANALYSIS_REUSE_WORKFLOW_FILES if path not in files]
    if missing:
        raise ValueError(
            f"{side} workflow fingerprint is missing analysis files: "
            + ", ".join(missing)
        )
    return {path: str(files[path]) for path in ANALYSIS_REUSE_WORKFLOW_FILES}


def reference_metrics_from_comparison_index(path: str | Path) -> dict[str, float]:
    index = json.loads(Path(path).read_text(encoding="utf-8"))
    outcomes = index.get("outcomes") or {}
    by_task = index.get("baseline_by_task") or {}
    if not isinstance(by_task, Mapping) or not by_task:
        raise ValueError("comparison index has no paired reference tasks")
    pass_at_1_successes = 0
    pass_at_1_trials = 0
    pass_at_2 = 0
    for refs in by_task.values():
        ordered = [str(outcomes.get(str(ref)) or "") for ref in refs]
        if not ordered or set(ordered) - {"pass", "fail"}:
            raise ValueError("comparison reference outcomes are incomplete")
        pass_at_1_successes += sum(outcome == "pass" for outcome in ordered)
        pass_at_1_trials += len(ordered)
        pass_at_2 += int("pass" in ordered[:2])
    task_count = len(by_task)
    return {
        "pass_at_1": pass_at_1_successes / pass_at_1_trials,
        "pass_at_2": pass_at_2 / task_count,
    }


def select_confirmation_task_ids(
    *,
    run_root: str | Path,
    decision: Any,
    screen_review: Any,
    screen_task_ids: Sequence[str],
    task_count: int,
) -> tuple[str, ...]:
    """Mix direct effect anchors with fresh, diverse preservation canaries."""

    root = Path(run_root)
    screen = tuple(dict.fromkeys(str(item) for item in screen_task_ids))
    count = int(task_count)
    if count < MIN_STANDARD_ROLLOUT_TASKS or len(screen) < count:
        raise ValueError("confirmation selection requires a complete standard screen")

    source = json.loads(
        (root / "experience" / "baseline_source_index.json").read_text(encoding="utf-8")
    )
    experiences = json.loads(
        (root / "experience" / "current.json").read_text(encoding="utf-8")
    )
    explorer = json.loads(
        (root / "discovery" / "task_explorer.json").read_text(encoding="utf-8")
    )
    evidence_by_task = {
        str(task_id): tuple(str(ref) for ref in refs)
        for task_id, refs in (source.get("evidence_by_task") or {}).items()
    }
    outcomes = {
        str(ref): str(outcome)
        for ref, outcome in (source.get("outcomes") or {}).items()
    }
    evidence_to_task = {
        ref: task_id for task_id, refs in evidence_by_task.items() for ref in refs
    }
    passage_by_id = {
        str(item.get("id") or ""): item
        for section in ("reusable", "needs_adjustment")
        for item in experiences.get(section) or []
        if isinstance(item, Mapping) and item.get("id")
    }
    experience_ids = {
        str(experience_id)
        for channel in decision.output["candidate"].get("channel_diffs") or []
        for experience_id in channel.get("experience_ids") or []
    }
    direct_tasks = {
        evidence_to_task[str(ref)]
        for experience_id in experience_ids
        for ref in passage_by_id.get(experience_id, {}).get("evidence_refs") or []
        if str(ref) in evidence_to_task
    }
    recovered = tuple(
        str(item)
        for item in (
            (screen_review.output.get("evidence") or {}).get("recovered_task_ids") or []
        )
    )
    uncertain = tuple(
        str(item)
        for item in (
            (screen_review.output.get("evidence") or {}).get(
                "uncertain_recovery_task_ids"
            )
            or []
        )
    )

    anchor_limit = min(2, count - 1)
    selected: list[str] = []
    for task_id in (
        *recovered,
        *uncertain,
        *(task for task in screen if task in direct_tasks),
        *screen,
    ):
        if task_id in screen and task_id not in selected:
            selected.append(task_id)
        if len(selected) >= anchor_limit:
            break

    fresh = [
        task_id
        for task_id in source.get("task_ids") or []
        if str(task_id) not in screen
    ]

    def preservation_rank(task_id: str) -> tuple[float, int, str]:
        refs = evidence_by_task.get(str(task_id), ())
        passes = sum(outcomes.get(ref) == "pass" for ref in refs)
        rate = passes / len(refs) if refs else 0.0
        return (-rate, -passes, str(task_id))

    fresh_set = {str(item) for item in fresh}
    stable_set = {
        task_id
        for task_id in fresh_set
        if evidence_by_task.get(task_id)
        and all(outcomes.get(ref) == "pass" for ref in evidence_by_task[task_id])
    }
    categories = [
        (
            str(category.get("id") or f"category-{index}"),
            {str(task_id) for task_id in category.get("task_ids") or []},
        )
        for index, category in enumerate(explorer.get("categories") or [])
        if isinstance(category, Mapping)
    ]
    categories_by_task: dict[str, set[str]] = {}
    for category_id, category_tasks in categories:
        for task_id in category_tasks:
            categories_by_task.setdefault(task_id, set()).add(category_id)
    represented_categories = {
        category_id
        for task_id in selected
        for category_id in categories_by_task.get(task_id, ())
    }
    for pool in (stable_set, fresh_set):
        while len(selected) < count:
            candidates = [task_id for task_id in pool if task_id not in selected]
            if not candidates:
                break
            task_id = min(
                candidates,
                key=lambda item: (
                    -len(categories_by_task.get(item, set()) - represented_categories),
                    *preservation_rank(item),
                ),
            )
            selected.append(task_id)
            represented_categories.update(categories_by_task.get(task_id, ()))
        if len(selected) >= count:
            break
    for task_id in sorted(fresh_set, key=preservation_rank):
        if len(selected) >= count:
            break
        if task_id not in selected:
            selected.append(task_id)
    for task_id in screen:
        if len(selected) >= count:
            break
        if task_id not in selected:
            selected.append(task_id)
    if len(selected) != count:
        raise ValueError("could not construct a complete confirmation task set")
    return tuple(selected)


def _can_finalize_frozen_run(run_root: Path, budget_status: Mapping[str, Any]) -> bool:
    if int(budget_status.get("remaining") or 0) != 0:
        return False
    decisions = sorted((run_root / "main_agent").glob("iteration-??.json"))
    return bool(decisions) and all(
        decision.with_name(f"{decision.stem}-review.json").is_file()
        for decision in decisions
    )


def _evaluation_requires_confirmation(mode: str) -> bool:
    return (
        confirmation_mode() == "always" and str(mode or "standard") != "residual_probe"
    )


def _paired_rollout_outputs_exist(
    *,
    run_root: Path,
    label: str,
    parent_version: str,
) -> bool:
    root = run_root / "rollout"
    return (root / f"{label}.json").is_file() and (
        root / f"{label}-parent-{parent_version}.json"
    ).is_file()


def advance_cumulative_version(
    *, current_version: str, decision: Any, review: Any
) -> tuple[str, bool]:
    parent = str(decision.output["candidate"]["parent_version"])
    child = str(decision.harness_version)
    selected = str(review.harness_version)
    if parent != str(current_version):
        raise RuntimeError(
            "cumulative iteration parent differs from the last accepted version"
        )
    if selected == child:
        return child, True
    if selected == parent:
        return parent, False
    raise RuntimeError(
        "cumulative review selected neither the child nor its direct parent"
    )


def assert_candidate_extends_champion(*, current_version: str, decision: Any) -> None:
    parent = str(decision.output["candidate"]["parent_version"])
    child = str(decision.harness_version)
    if parent != str(current_version) or child in {"", parent, "v0"}:
        raise RuntimeError(
            "candidate must extend the current champion before rollout: "
            f"champion={current_version} parent={parent} child={child}"
        )


def assert_confirmed_promotion(
    *, decision: Any, review: Any, confirmation_pair: Any | None
) -> None:
    if (
        str(review.harness_version) == str(decision.harness_version)
        and confirmation_pair is None
    ):
        raise RuntimeError(
            "candidate promotion requires independent confirmation against its direct parent"
        )


def initialize_champion(
    *,
    repository: CellHarnessRepository,
    incumbent_candidates: list[Mapping[str, Any]],
) -> tuple[str, list[dict[str, Any]], set[str]]:
    """Install one explicit prior winner as the current run's initial champion."""

    if not incumbent_candidates:
        return "v0", [], set()
    if len(incumbent_candidates) != 1:
        raise ValueError("exactly one incumbent submission may initialize a champion")
    candidate = dict(incumbent_candidates[0])
    candidate_id = str(candidate["id"])
    version = repository.materialize_candidate(
        base_version="v0",
        candidate_label="incumbent-00",
        delta=candidate["manifest_delta"],
    )
    prior = dict(candidate.get("_prior_train_evidence") or {})
    context = {
        "candidate_id": candidate_id,
        "origin_candidate_id": candidate_id,
        "candidate_side": "incumbent",
        "candidate_version": version,
        "selected_version": version,
        "review_decision": "imported_champion",
        "rollout_task_ids": list(candidate.get("_direct_task_ids") or []),
        "review_evidence": {
            "recovered_task_ids": list(prior.get("recovered_task_ids") or []),
            "preserved_task_ids": list(prior.get("preserved_task_ids") or []),
            "attributable_regression_task_ids": list(
                prior.get("attributable_regression_task_ids") or []
            ),
        },
        "manifest_delta": dict(candidate["manifest_delta"]),
        "prior_train_evidence": prior,
    }
    return version, [context], {candidate_id}


def minimum_screen_start_creations(parent_version: str) -> int:
    return (
        paired_screen_creation_cost(
            MIN_STANDARD_ROLLOUT_TASKS, parent_version=parent_version
        )
        + ITERATION_RETRY_BUFFER_CREATIONS
    )


def minimum_residual_start_creations(parent_version: str) -> int:
    return (
        paired_screen_creation_cost(
            MIN_RESIDUAL_ROLLOUT_TASKS, parent_version=parent_version
        )
        + ITERATION_RETRY_BUFFER_CREATIONS
    )


def confirmation_creation_cost(task_count: int, *, parent_version: str) -> int:
    del parent_version
    return paired_confirmation_creation_cost(task_count)


def reject_unconfirmed_candidate(
    *, screen_review: MainDecision, parent_version: str, output_path: Path
) -> MainDecision:
    payload = dict(screen_review.output)
    payload["decision"] = "reject_delta"
    payload["selected_version"] = str(parent_version)
    payload["rationale"] = (
        "The screen passed, but independent confirmation was not affordable. "
        "Keep the current champion rather than promote provisional evidence."
    )
    evidence = dict(payload.get("evidence") or {})
    unresolved = list(evidence.get("unresolved_findings") or [])
    unresolved.append(
        "Fresh-seed confirmation was not run within the remaining budget."
    )
    evidence["unresolved_findings"] = unresolved
    payload["evidence"] = evidence
    payload.pop("revision_candidate", None)
    written = write_json(output_path, payload)
    return MainDecision(payload, str(written), str(parent_version))


def defer_residual_candidate(
    *, screen_review: MainDecision, parent_version: str, output_path: Path
) -> MainDecision:
    payload = dict(screen_review.output)
    payload["decision"] = "reject_delta"
    payload["selected_version"] = str(parent_version)
    payload["rationale"] = (
        "The residual probe is diagnostic-only. Keep the current champion until "
        "the candidate receives a standard screen and independent confirmation."
    )
    payload["evidence_disposition"] = "diagnostic_only"
    evidence = dict(payload.get("evidence") or {})
    unresolved = list(evidence.get("unresolved_findings") or [])
    unresolved.append("Residual probe evidence cannot satisfy the promotion contract.")
    evidence["unresolved_findings"] = unresolved
    payload["evidence"] = evidence
    payload.pop("revision_candidate", None)
    written = write_json(output_path, payload)
    return MainDecision(payload, str(written), str(parent_version))


def revision_candidate_from_review(
    review_output: Mapping[str, Any],
    *,
    rollout_task_ids: list[str] | tuple[str, ...] = (),
) -> dict[str, Any] | None:
    if str(review_output.get("decision") or "") != "revise_delta":
        return None
    revision = review_output.get("revision_candidate")
    if not isinstance(revision, Mapping):
        raise RuntimeError("revise_delta review is missing its revision candidate")
    result = dict(revision)
    if rollout_task_ids:
        result["_direct_task_ids"] = sorted({str(item) for item in rollout_task_ids})
        result["_prior_train_evidence"] = dict(review_output.get("evidence") or {})
    return result


def replan_candidate_from_review(
    review_output: Mapping[str, Any],
    *,
    adjustment: Mapping[str, Any],
    rollout_task_ids: Sequence[str] = (),
) -> dict[str, Any] | None:
    if str(review_output.get("decision") or "") != "replan_problem":
        return None
    candidate = adjustment.get("replan_candidate")
    if not isinstance(candidate, Mapping):
        raise RuntimeError("replan_problem review is missing the Analyzer candidate")
    result = dict(candidate)
    result["_portfolio_side"] = "adjustment"
    result["_direct_task_ids"] = sorted(
        {str(item) for item in rollout_task_ids if str(item)}
    )
    result["_conversion_task_ids"] = sorted(
        {
            str(item.get("task_id") or "")
            for item in (adjustment.get("primary_problem") or {}).get(
                "task_assessments"
            )
            or []
            if isinstance(item, Mapping)
            and str(item.get("status") or "")
            in {"still_failing", "mixed", "regressed"}
            and int((item.get("outcome_summary") or {}).get("candidate_pass_count") or 0)
            == 0
            and str(item.get("task_id") or "")
        }
    )
    result["_prior_train_evidence"] = dict(review_output.get("evidence") or {})
    return result
