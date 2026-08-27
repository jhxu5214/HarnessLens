from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from harnesslens.core.artifacts import write_json, write_text
from harnesslens.core.budget import CreationBudget
from harnesslens.harnesses.harness_manifest import normalize_harness
from harnesslens.harnesses.harness_workspace import (
    capture_workspace,
    diff_workspace,
    normalize_workspace_snapshot,
    seed_workspace,
    workspace_digest,
)
from harnesslens.harnesses.native_intelligent_runtime import NativeIntelligentAdapter
from harnesslens.harnesses.opencode_runtime import OpenCodeIntelligentAdapter
from harnesslens.core.profiles import power_profile
from harnesslens.harnesses.runner import parse_json_object


EDITOR_SYSTEM_PROMPT = """You are the Harness Editor for the target agent harness.
Use the Harness Query report to understand the harness, then directly edit the isolated `home/`
and `project/` trees to address the selected evidence-backed problem. The query is a map, not a
permission list: you may use any genuine user-customizable mechanism of this harness, including
local MCP or extension code, when it is justified by the evidence.

Make one coherent, reviewable change. Preserve behavior that the evidence does not challenge.
Express learned behavior as a reusable principle: candidate artifacts must not contain supplied
experience IDs, task IDs, benchmark-specific entities, literal answers, or per-example mappings.
This is an editing constraint, not a rule to copy into the target agent's runtime instructions.
Validation checks describe tests, not examples to copy into runtime content.
The experiment fixes the model, provider, reasoning level, task tools, permissions, budget, and
evaluation; do not attempt to change them or access benchmark tests, rewards, hidden labels,
credentials, the host repository, or the network. Work only inside the supplied candidate and
scratch trees. Use the scratch tree for temporary validation files; only runtime artifacts belong
in the candidate tree. Use local checks when available. End with one JSON object summarizing changed paths, rationale,
and checks; the controller will independently capture the actual files and diff.
Every SKILL.md must start with YAML frontmatter containing a `name` that exactly matches its
parent directory and a non-empty `description`, followed by the closing `---` delimiter."""


EDITOR_TOOLS = {
    "opencode": ("read", "glob", "grep", "list", "write", "edit", "apply_patch", "patch"),
    "pi": ("read", "edit", "write", "grep", "find", "ls"),
    "codex": (),
}

HOME_SCOPE = {
    "opencode": "XDG_CONFIG_HOME/opencode",
    "pi": "PI_CODING_AGENT_DIR",
    "codex": "CODEX_HOME",
}


@dataclass(frozen=True)
class HarnessEditResult:
    harness: str
    job_id: str
    snapshot: Mapping[str, Any]
    changes: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any] | None
    root: str
    workspace: str
    stdout_path: str
    stderr_path: str
    api_trace_path: str


class HarnessEditor:
    def __init__(
        self,
        *,
        harness: str,
        budget: CreationBudget,
        run_root: str | Path,
        max_steps: int = 40,
        timeout_s: int = 3600,
    ) -> None:
        self.harness = normalize_harness(harness)
        self.budget = budget
        self.run_root = Path(run_root).resolve()
        self.max_steps = int(max_steps)
        self.timeout_s = int(timeout_s)
        if self.max_steps <= 0 or self.timeout_s <= 0:
            raise ValueError("Harness Editor limits must be positive")
        self.run_root.mkdir(parents=True, exist_ok=True)

    def edit(
        self,
        *,
        job_id: str,
        base_workspace: Mapping[str, Any] | None,
        harness_query: Mapping[str, Any],
        problem: Mapping[str, Any],
        evidence: list[Mapping[str, Any]],
        current_manifest: Mapping[str, Any] | None = None,
    ) -> HarnessEditResult:
        base = normalize_workspace_snapshot(base_workspace)
        job_root = self.run_root / str(job_id)
        editable = seed_workspace(job_root / "candidate", base)
        scratch = job_root / "scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        payload = {
            "target_harness": self.harness,
            "candidate_tree": {
                "home": str(editable / "home"),
                "project": str(editable / "project"),
                "home_maps_to": HOME_SCOPE[self.harness],
            },
            "scratch_tree": str(scratch),
            "harness_query": dict(harness_query),
            "selected_problem": dict(problem),
            "evidence": [dict(item) for item in evidence],
            "current_legacy_manifest": dict(current_manifest or {}),
            "current_workspace": base,
        }
        write_json(job_root / "input.json", payload)
        prompt = (
            EDITOR_SYSTEM_PROMPT
            + "\n\nINPUT:\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        )
        write_text(job_root / "submitted_prompt.txt", prompt)
        adapter = self._adapter()
        result = adapter.run(
            prompt=prompt,
            workspace=job_root,
            working_directory=editable,
            call_id=str(job_id),
            budget=self.budget,
            max_steps=self.max_steps,
            output_validator=None,
        )
        if result.outcome != "completed":
            raise RuntimeError(
                f"Harness Editor failed: {result.outcome}: {result.validation_error}"
            )
        snapshot = capture_workspace(editable)
        changes = diff_workspace(base, snapshot)
        if not changes:
            raise ValueError("Harness Editor completed without changing the candidate workspace")
        validate_editor_candidate_artifacts(
            snapshot=snapshot,
            changes=changes,
            evidence=evidence,
        )
        stdout = Path(result.stdout_path).read_text(
            encoding="utf-8", errors="replace"
        )
        try:
            summary: Mapping[str, Any] | None = dict(parse_json_object(stdout))
        except ValueError:
            summary = None
        write_json(job_root / "workspace.json", snapshot)
        write_json(job_root / "diff.json", changes)
        write_json(
            job_root / "result.json",
            {
                "harness": self.harness,
                "job_id": str(job_id),
                "workspace_sha256": workspace_digest(snapshot),
                "changes": changes,
                "summary": summary,
                "stdout_path": str(result.stdout_path),
                "stderr_path": str(result.stderr_path),
                "api_trace_path": str(result.api_trace_path),
            },
        )
        return HarnessEditResult(
            harness=self.harness,
            job_id=str(job_id),
            snapshot=snapshot,
            changes=tuple(changes),
            summary=summary,
            root=str(job_root),
            workspace=str(editable),
            stdout_path=str(result.stdout_path),
            stderr_path=str(result.stderr_path),
            api_trace_path=str(result.api_trace_path),
        )

    def _adapter(self) -> OpenCodeIntelligentAdapter | NativeIntelligentAdapter:
        profile = power_profile(self.harness, max_steps=self.max_steps)
        common = {
            "model": profile.model,
            "context_limit": profile.context_limit,
            "output_limit": profile.output_limit,
            "max_steps": profile.max_steps,
            "timeout_s": self.timeout_s,
            "workspace_root": self.run_root,
            "model_options": profile.provider_options,
            "allowed_builtin_tools": EDITOR_TOOLS[self.harness],
        }
        if self.harness == "opencode":
            return OpenCodeIntelligentAdapter(**common)
        return NativeIntelligentAdapter(harness=self.harness, **common)


def validate_editor_candidate_artifacts(
    *,
    snapshot: Mapping[str, Any],
    changes: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    evidence: list[Mapping[str, Any]],
) -> None:
    normalized = normalize_workspace_snapshot(snapshot)
    changed_paths = {
        (str(item.get("scope") or ""), str(item.get("path") or ""))
        for item in changes
        if str(item.get("change") or "") != "deleted"
    }
    for item in normalized["files"]:
        key = (str(item["scope"]), str(item["path"]))
        if key in changed_paths:
            _validate_skill_frontmatter(key[1], str(item["content"]))

    forbidden_identifiers = {
        str(value)
        for item in evidence
        for value in [item.get("id"), *(item.get("evidence_refs") or [])]
        if str(value or "").strip()
    }
    evidence_text = "\n".join(str(item.get("text") or "") for item in evidence)
    forbidden_identifiers.update(
        token
        for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", evidence_text)
        if any(char.islower() for char in token)
        and any(char.isupper() for char in token)
        and any(char.isdigit() for char in token)
    )
    if not forbidden_identifiers:
        return
    violations: list[str] = []
    for item in normalized["files"]:
        key = (str(item["scope"]), str(item["path"]))
        if key not in changed_paths:
            continue
        content = str(item["content"])
        leaked = sorted(
            identifier for identifier in forbidden_identifiers if identifier in content
        )
        if leaked:
            violations.append(f"{key[0]}/{key[1]}: {', '.join(leaked)}")
    if violations:
        raise ValueError(
            "candidate artifacts must generalize evidence instead of naming it: "
            + "; ".join(violations)
        )


_SKILL_PATH = re.compile(
    r"(?:\.pi|\.opencode|\.agents)/skills/"
    r"([a-z0-9]+(?:-[a-z0-9]+)*)/SKILL\.md"
)


def _validate_skill_frontmatter(path: str, content: str) -> None:
    match = _SKILL_PATH.fullmatch(path)
    if match is None:
        return
    if not content.startswith("---\n") or "\n---\n" not in content[4:]:
        raise ValueError(f"{path} requires YAML frontmatter")
    frontmatter = content.split("\n---\n", 1)[0]
    name = re.search(r"(?m)^name:\s*([^\s]+)\s*$", frontmatter)
    description = re.search(r"(?m)^description:\s*(.+?)\s*$", frontmatter)
    if name is None or name.group(1) != match.group(1):
        raise ValueError(f"{path} frontmatter name must match its parent directory")
    if description is None or not description.group(1).strip().strip("'\""):
        raise ValueError(f"{path} frontmatter requires a non-empty description")
