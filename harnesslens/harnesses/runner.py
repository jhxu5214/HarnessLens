from __future__ import annotations

import hashlib
import json
import re
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from harnesslens.core.budget import CreationBudget
from harnesslens.harnesses.native_intelligent_runtime import NativeIntelligentAdapter
from harnesslens.harnesses.opencode_runtime import OpenCodeIntelligentAdapter
from harnesslens.core.profiles import HarnessProfile


@dataclass(frozen=True)
class IntelligentRunResult:
    job_id: str
    harness: str
    outcome: str
    output: Mapping[str, Any] | None
    stdout_path: str
    stderr_path: str
    validation_error: str = ""
    api_trace_path: str = ""
    interaction_manifest_path: str = ""


def parse_json_object(stdout: str) -> Mapping[str, Any]:
    text_fragments: list[str] = []
    for line in str(stdout).splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping):
            continue
        part = event.get("part") if isinstance(event.get("part"), Mapping) else {}
        if event.get("type") in {"text", "assistant"}:
            value = part.get("text") or event.get("content") or event.get("text")
            if value:
                text_fragments.append(str(value))
        if event.get("type") == "message_end" and isinstance(
            event.get("message"), Mapping
        ):
            message = event["message"]
            if message.get("role") == "assistant":
                content = message.get("content")
                if isinstance(content, list):
                    value = "".join(
                        str(item.get("text") or "")
                        for item in content
                        if isinstance(item, Mapping) and item.get("type") == "text"
                    )
                    if value:
                        text_fragments.append(value)
        if event.get("type") == "item.completed" and isinstance(
            event.get("item"), Mapping
        ):
            item = event["item"]
            if item.get("type") == "agent_message" and item.get("text"):
                text_fragments.append(str(item["text"]))
    # OpenCode may emit analysis between read-tool steps. Its newest text message
    # is the final assistant response, so inspect messages newest-first.
    candidates = [*reversed(text_fragments), "\n".join(text_fragments), str(stdout)]
    for candidate in candidates:
        stripped = candidate.strip()
        variants = [stripped]
        fenced = re.findall(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
        if len(fenced) == 1:
            variants.append(fenced[0].strip())
        for variant in variants:
            try:
                value = json.loads(variant)
            except json.JSONDecodeError:
                value = _decode_one_embedded_object(variant)
                if value is None:
                    value = _close_truncated_json_object(variant)
            if isinstance(value, Mapping):
                if value.get("type") in {"text", "assistant"} and isinstance(
                    value.get("part"), Mapping
                ):
                    continue
                return dict(value)
    raise ValueError("intelligent harness did not return one JSON object")


def _decode_one_embedded_object(text: str) -> Mapping[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, end = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if not isinstance(value, Mapping):
            continue
        remainder = text[end:]
        for later in re.finditer(r"\{", remainder):
            try:
                second, _ = decoder.raw_decode(remainder, later.start())
            except json.JSONDecodeError:
                continue
            if isinstance(second, Mapping):
                return None
        return dict(value)
    return None


def _close_truncated_json_object(text: str) -> Mapping[str, Any] | None:
    stripped = str(text).strip()
    if not stripped.startswith("{"):
        return None
    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"}": "{", "]": "["}
    for char in stripped:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack or stack.pop() != pairs[char]:
                return None
    if in_string or not stack or len(stack) > 8:
        return None
    suffix = "".join("}" if opener == "{" else "]" for opener in reversed(stack))
    try:
        value = json.loads(stripped + suffix)
    except json.JSONDecodeError:
        return None
    return dict(value) if isinstance(value, Mapping) else None


class IntelligentHarnessRunner:
    def __init__(
        self,
        *,
        profile: HarnessProfile,
        budget: CreationBudget,
        workspace_root: str | Path,
        timeout_s: int = 1800,
        allowed_builtin_tools: tuple[str, ...] = (),
    ) -> None:
        self.profile = profile
        self.budget = budget
        self.workspace_root = Path(workspace_root)
        self.timeout_s = int(timeout_s)
        self.allowed_builtin_tools = tuple(allowed_builtin_tools)
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def run_json(
        self,
        *,
        job_id: str,
        system_prompt: str,
        input_payload: Mapping[str, Any],
        validator: Callable[[Mapping[str, Any]], Any] | None = None,
    ) -> IntelligentRunResult:
        job_root = self.workspace_root / str(job_id)
        job_root.mkdir(parents=True, exist_ok=True)
        effective_payload = deepcopy(dict(input_payload))
        adapter_tools = self.allowed_builtin_tools
        if self.profile.harness == "codex" and set(adapter_tools) == {"read"}:
            effective_payload = _inline_controller_read_files(
                effective_payload,
                allowed_root=self.workspace_root.resolve().parent,
            )
            adapter_tools = ()
        input_path = job_root / "input.json"
        input_path.write_text(
            json.dumps(effective_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        prompt = (
            str(system_prompt).strip()
            + "\n\nReturn exactly one JSON object and no markdown.\n"
            + "INPUT:\n"
            + json.dumps(effective_payload, ensure_ascii=False, sort_keys=True)
        )
        system_prompt_path = job_root / "system_prompt.txt"
        submitted_prompt_path = job_root / "submitted_prompt.txt"
        system_prompt_path.write_text(str(system_prompt), encoding="utf-8")
        submitted_prompt_path.write_text(prompt, encoding="utf-8")

        def validate_stdout(stdout: str) -> Mapping[str, Any]:
            output = parse_json_object(stdout)
            if validator is not None:
                validator(output)
            return output

        adapter = (
            OpenCodeIntelligentAdapter(
                model=self.profile.model,
                context_limit=self.profile.context_limit,
                output_limit=self.profile.output_limit,
                max_steps=self.profile.max_steps,
                timeout_s=self.timeout_s,
                workspace_root=self.workspace_root,
                model_options=self.profile.provider_options,
                allowed_builtin_tools=adapter_tools,
            )
            if self.profile.harness == "opencode"
            else NativeIntelligentAdapter(
                harness=self.profile.harness,
                model=self.profile.model,
                context_limit=self.profile.context_limit,
                output_limit=self.profile.output_limit,
                max_steps=self.profile.max_steps,
                timeout_s=self.timeout_s,
                workspace_root=self.workspace_root,
                model_options=self.profile.provider_options,
                allowed_builtin_tools=adapter_tools,
            )
        )
        result = adapter.run(
            prompt=prompt,
            workspace=job_root,
            call_id=str(job_id),
            budget=self.budget,
            max_steps=self.profile.max_steps,
            output_validator=None,
        )
        stdout = Path(result.stdout_path).read_text(encoding="utf-8", errors="replace")
        output = None
        outcome = result.outcome
        validation_error = result.validation_error
        if outcome == "completed":
            try:
                output = dict(validate_stdout(stdout))
            except Exception as exc:
                outcome = "malformed_output"
                validation_error = str(exc)
            else:
                (job_root / "output.json").write_text(
                    json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        interaction_manifest_path = job_root / "interaction_manifest.json"
        native_config = (
            job_root / ".pi_home" / "models.json"
            if self.profile.harness == "pi"
            else job_root / ".codex_home" / "config.toml"
        )
        artifact_paths = {
            "input": input_path,
            "system_prompt": system_prompt_path,
            "submitted_prompt": submitted_prompt_path,
            "opencode_config": job_root / "opencode.json",
            "native_config": native_config,
            "invocation": job_root / "invocation.json",
            "stdout": Path(result.stdout_path),
            "stderr": Path(result.stderr_path),
            "api_calls": Path(result.api_trace_path),
        }
        interaction_manifest_path.write_text(
            json.dumps(
                {
                    "schema": f"harnesslens.{self.profile.harness}-interaction.v1",
                    "job_id": str(job_id),
                    "harness": self.profile.harness,
                    "model": self.profile.model,
                    "outcome": outcome,
                    "validation_error": validation_error,
                    "artifacts": {
                        name: {
                            "path": str(path),
                            "sha256": _sha256(path),
                            "bytes": path.stat().st_size,
                        }
                        for name, path in artifact_paths.items()
                        if path.is_file()
                    },
                    "session_root": str(
                        job_root / (
                            ".oc_data"
                            if self.profile.harness == "opencode"
                            else ".pi_home"
                            if self.profile.harness == "pi"
                            else ".codex_home"
                        )
                    ),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if self.profile.harness == "opencode":
            _cleanup_opencode_dependency_cache(job_root)
        return IntelligentRunResult(
            job_id=str(job_id),
            harness=self.profile.harness,
            outcome=outcome,
            output=output,
            stdout_path=result.stdout_path,
            stderr_path=result.stderr_path,
            validation_error=validation_error,
            api_trace_path=result.api_trace_path,
            interaction_manifest_path=str(interaction_manifest_path),
        )


def _inline_controller_read_files(
    payload: Mapping[str, Any], *, allowed_root: str | Path
) -> dict[str, Any]:
    root = Path(allowed_root).resolve()
    paths: list[Path] = []

    def visit(value: Any, key: str = "") -> None:
        if key.endswith("_path") and isinstance(value, str):
            paths.append(Path(value))
            return
        if key.endswith("_paths") and isinstance(value, list):
            paths.extend(Path(item) for item in value if isinstance(item, str))
            return
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child)

    effective = deepcopy(dict(payload))
    visit(effective)
    inlined: list[dict[str, str]] = []
    total_bytes = 0
    for raw_path in dict.fromkeys(paths):
        path = raw_path.resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"controller read path is outside the run root: {path}") from exc
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"controller read path is not a regular file: {path}")
        size = path.stat().st_size
        if size > 2 * 1024 * 1024:
            raise ValueError(f"controller read file is too large: {path}")
        total_bytes += size
        if total_bytes > 8 * 1024 * 1024:
            raise ValueError("controller read files exceed the inline size limit")
        inlined.append(
            {
                "path": str(path),
                "content": path.read_text(encoding="utf-8"),
            }
        )
    effective["controller_file_transport"] = {
        "mode": "inline",
        "instruction": (
            "Every required run-owned file is present in controller_inlined_files; "
            "inspect those contents directly and do not call file tools."
        ),
    }
    effective["controller_inlined_files"] = inlined
    instruction = str(effective.get("instruction") or "").strip()
    effective["instruction"] = (
        instruction + " " if instruction else ""
    ) + "The controller inlined every required file; do not call file tools."
    return effective


def intelligent_stdout_path(workspace: str | Path, harness: str) -> Path:
    normalized = str(harness).strip().lower().replace("-", "_")
    if normalized == "pi_agent":
        normalized = "pi"
    root = Path(workspace)
    if normalized == "codex":
        last_message = root / "last_message.txt"
        if last_message.is_file() and last_message.stat().st_size > 0:
            return last_message
    name = "opencode" if normalized == "opencode" else normalized
    return root / f"{name}.stdout"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cleanup_opencode_dependency_cache(job_root: Path) -> None:
    dependency_cache = job_root / ".oc_config" / "opencode" / "node_modules"
    if dependency_cache.is_dir():
        shutil.rmtree(dependency_cache)
