from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Protocol

from harnesslens.core.config import pi_binary, pi_install_root
from harnesslens.harnesses.opencode_harness import OpencodeHarnessAdapter


class HarnessQueryAdapter(Protocol):
    def architecture_probe(self) -> dict[str, Any]: ...

    def query_evidence_catalog(
        self, probe: Mapping[str, Any]
    ) -> list[dict[str, Any]]: ...

    def query_channel_inventory(
        self, probe: Mapping[str, Any] | None = None
    ) -> list[dict[str, Any]]: ...


def harness_query_adapter(
    harness: str, *, repo_root: str | Path
) -> HarnessQueryAdapter:
    normalized = str(harness).strip().lower().replace("-", "_")
    if normalized == "opencode":
        return OpencodeHarnessAdapter(repo_root=repo_root)
    if normalized in {"pi", "pi_agent"}:
        return PiHarnessQueryAdapter(repo_root=repo_root)
    if normalized == "codex":
        return CodexHarnessQueryAdapter(repo_root=repo_root)
    raise ValueError(f"unsupported Harness Query target: {harness}")


def _channel(
    channel_id: str,
    *,
    visibility: str,
    use: str,
    artifact_contract: str,
    risks: str,
    operation: Mapping[str, Any] | None,
    evidence_refs: tuple[str, ...],
    status: str = "verified",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": channel_id,
        "visibility": visibility,
        "use": use,
        "artifact_contract": artifact_contract,
        "risks": risks,
        "status": status,
        "evidence_refs": list(evidence_refs),
        "verification": {
            "level": "materializer_contract" if operation else "native_only",
            "runtime_observed": False,
            "observations": [],
        },
        "materialization": "main_candidate" if operation else "proposal_only",
    }
    if operation:
        payload["operation"] = dict(operation)
    return payload


class _StaticNativeAdapter:
    harness_id = ""
    executable_name = ""
    channels: tuple[Mapping[str, Any], ...] = ()
    request_observations: Mapping[str, tuple[str, ...]] = {}

    def __init__(self, *, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root).resolve()

    def executable(self) -> Path:
        found = shutil.which(self.executable_name)
        if found:
            return Path(found)
        raise RuntimeError(f"{self.executable_name} executable is unavailable")

    def architecture_probe(self) -> dict[str, Any]:
        executable = self.executable()
        with tempfile.TemporaryDirectory(prefix=f"harnesslens-{self.harness_id}-probe-") as raw:
            root = Path(raw)
            env = self._probe_env(root)
            version = _run((str(executable), "--version"), cwd=root, env=env).strip()
            help_text = _run((str(executable), "--help"), cwd=root, env=env)
            extra = self._extra_probe(executable=executable, root=root, env=env)
        facts = {
            "harness_id": self.harness_id,
            "harness_version": version,
            "native_options": sorted(
                set(re.findall(r"(?m)^\s{2,}(--[a-z0-9-]+)", help_text))
            ),
            "native_commands": sorted(
                set(re.findall(r"(?m)^\s{2}([a-z][a-z0-9-]+)\s{2,}", help_text))
            ),
            **extra,
        }
        facts["probe_digest"] = hashlib.sha256(
            json.dumps(facts, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return facts

    def _probe_env(self, root: Path) -> dict[str, str]:
        return {
            **os.environ,
            "HOME": str(root / "home"),
            "XDG_CONFIG_HOME": str(root / "config"),
            "XDG_DATA_HOME": str(root / "data"),
            "XDG_CACHE_HOME": str(root / "cache"),
            "NO_COLOR": "1",
        }

    def _extra_probe(
        self, *, executable: Path, root: Path, env: Mapping[str, str]
    ) -> dict[str, Any]:
        return {}

    def query_evidence_catalog(self, probe: Mapping[str, Any]) -> list[dict[str, Any]]:
        catalog = [
            {
                "id": "probe:version",
                "kind": "command_output",
                "source": f"{self.executable_name} --version",
                "value": str(probe.get("harness_version") or ""),
            },
            {
                "id": "probe:help",
                "kind": "command_output",
                "source": f"{self.executable_name} --help",
                "value": {
                    "options": list(probe.get("native_options") or []),
                    "commands": list(probe.get("native_commands") or []),
                },
            },
        ]
        catalog.extend(self._documentation_evidence())
        if self.request_observations:
            catalog.append(
                {
                    "id": f"runtime:probe:{self.harness_id}-startup",
                    "kind": "request_sentinel_probe",
                    "source": "tests/test_harness_query_runtime.py",
                    "value": {
                        key: list(value)
                        for key, value in self.request_observations.items()
                    },
                }
            )
        for channel in self.channels:
            if str(channel.get("status") or "") != "verified":
                continue
            catalog.append(
                {
                    "id": f"runtime:channel:{channel['id']}",
                    "kind": "materializer_contract",
                    "source": f"HarnessLens {self.harness_id} candidate runtime contract",
                    "value": dict(channel.get("operation") or {}),
                }
            )
        return catalog

    def _documentation_evidence(self) -> list[dict[str, Any]]:
        return []

    def query_channel_inventory(
        self, probe: Mapping[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        inventory: list[dict[str, Any]] = []
        for raw in self.channels:
            channel = dict(raw)
            observations = self.request_observations.get(str(channel["id"]), ())
            if observations:
                channel["evidence_refs"] = [
                    *channel["evidence_refs"],
                    f"runtime:probe:{self.harness_id}-startup",
                ]
                channel["verification"] = {
                    "level": "request_trace",
                    "runtime_observed": True,
                    "observations": list(observations),
                }
            inventory.append(channel)
        return inventory


class PiHarnessQueryAdapter(_StaticNativeAdapter):
    harness_id = "pi"
    executable_name = "pi"
    request_observations = {
        "system_prompt": ("appended_content_present_in_system_message",),
        "project_instructions": ("AGENTS.md_content_present_in_system_message",),
        "skills": (
            "skill_description_present_when_read_enabled",
            "skill_body_absent_from_startup_request",
        ),
    }

    channels = (
        _channel(
            "system_prompt",
            visibility="startup system context",
            use="Append concise global agent framing while preserving Pi's native prompt.",
            artifact_contract="project .pi/APPEND_SYSTEM.md",
            risks="Broad content affects every task; full replacement can remove native guidance and skill discovery.",
            operation={"kind": "project_file", "path_pattern": ".pi/APPEND_SYSTEM.md"},
            evidence_refs=(
                "probe:help",
                "docs:pi-settings",
                "runtime:channel:system_prompt",
            ),
        ),
        _channel(
            "project_instructions",
            visibility="startup project context unless --no-context-files is active",
            use="Project-scoped standing instructions.",
            artifact_contract="AGENTS.md in the candidate workspace",
            risks="The channel is inert when the runtime uses --no-context-files.",
            operation={"kind": "project_file", "path_pattern": "AGENTS.md"},
            evidence_refs=(
                "probe:help",
                "docs:pi-settings",
                "runtime:channel:project_instructions",
            ),
        ),
        _channel(
            "skills",
            visibility="names and descriptions at startup when read is enabled; body read on demand",
            use="Triggered procedures and detailed conditional guidance.",
            artifact_contract=".pi/skills/<slug>/SKILL.md or --skill <path>",
            risks="Requires project trust and the read tool; --no-tools or --no-skills makes it inert.",
            operation={
                "kind": "project_file",
                "path_pattern": ".pi/skills/<slug>/SKILL.md",
            },
            evidence_refs=("probe:help", "docs:pi-skills", "runtime:channel:skills"),
        ),
        _channel(
            "tool_description",
            visibility="whenever the registered extension tool is available",
            use="Compact call-selection guidance for one exact tool.",
            artifact_contract="tool_desc_patches.<tool>.desc before extension registration",
            risks="Long guidance bloats every decision involving the tool.",
            operation={
                "kind": "tool_schema_patch",
                "manifest_field": "tool_desc_patches",
                "target": "description",
            },
            evidence_refs=("runtime:channel:tool_description",),
        ),
        _channel(
            "tool_parameter_description",
            visibility="whenever the registered extension tool schema is available",
            use="Compact meaning or constraint for one exact parameter.",
            artifact_contract="tool_desc_patches.<tool>.params.<parameter>",
            risks="Description guidance cannot enforce parameter values.",
            operation={
                "kind": "tool_schema_patch",
                "manifest_field": "tool_desc_patches",
                "target": "parameter_description",
            },
            evidence_refs=("runtime:channel:tool_parameter_description",),
        ),
        _channel(
            "tool_enablement",
            visibility="startup tool catalog",
            use="Enable or disable existing built-in or extension tools.",
            artifact_contract="--tools/--exclude-tools or .pi/settings.json",
            risks="Capability changes can alter task feasibility and comparability.",
            operation=None,
            evidence_refs=("probe:help", "docs:pi-settings"),
            status="proposal_only",
        ),
        _channel(
            "extensions",
            visibility="startup extension registration and tool-call time",
            use="Add a separately reviewed integration or custom tool.",
            artifact_contract=".pi/extensions/<name>.ts or --extension <path>",
            risks="Loads executable code and can change the agent capability boundary.",
            operation={
                "kind": "project_file",
                "path_pattern": ".pi/extensions/<name>.ts",
            },
            evidence_refs=(
                "probe:help",
                "docs:pi-settings",
                "runtime:channel:extensions",
            ),
        ),
        _channel(
            "compaction_config",
            visibility="session context lifecycle",
            use="Configure retention of long-running session context.",
            artifact_contract=".pi/settings.json compaction fields",
            risks="Changes which prior evidence remains visible.",
            operation={
                "kind": "harness_config_patch",
                "manifest_field": "config_patch",
                "key_prefix": "compaction.",
            },
            evidence_refs=("docs:pi-settings", "runtime:channel:compaction_config"),
        ),
        _channel(
            "mcp_servers",
            visibility="not natively available",
            use="Only through a separately implemented extension.",
            artifact_contract="separate extension-backed integration",
            risks="Pi has no built-in MCP configuration surface.",
            operation=None,
            evidence_refs=("docs:pi-extensions",),
            status="proposal_only",
        ),
    )

    def executable(self) -> Path:
        return pi_binary(self.repo_root)

    def _probe_env(self, root: Path) -> dict[str, str]:
        env = super()._probe_env(root)
        env.update(
            {
                "PI_CODING_AGENT_DIR": str(root / "pi-agent"),
                "PI_OFFLINE": "1",
                "PI_TELEMETRY": "0",
            }
        )
        return env

    def _documentation_evidence(self) -> list[dict[str, Any]]:
        root = (
            pi_install_root(self.repo_root)
            / "node_modules"
            / "@earendil-works"
            / "pi-coding-agent"
            / "docs"
        )
        return [
            _doc_evidence("docs:pi-settings", root / "settings.md"),
            _doc_evidence("docs:pi-skills", root / "skills.md"),
            _doc_evidence("docs:pi-extensions", root / "extensions.md"),
        ]


class CodexHarnessQueryAdapter(_StaticNativeAdapter):
    harness_id = "codex"
    executable_name = "codex"
    request_observations = {
        "developer_instructions": ("content_present_in_developer_instructions",),
        "project_instructions": ("AGENTS.md_content_present_in_request",),
        "skills": (
            "skill_description_present_in_request",
            "skill_body_absent_from_startup_request",
        ),
        "hooks": ("session_start_context_present_in_request",),
    }

    def _documentation_evidence(self) -> list[dict[str, Any]]:
        return [
            _doc_evidence(
                "docs:codex-config-reference",
                self.repo_root
                / "assets"
                / "docs_cache"
                / "codex"
                / "config_reference.md",
            )
        ]

    channels = (
        _channel(
            "developer_instructions",
            visibility="startup developer context",
            use="Concise global instructions for the candidate agent.",
            artifact_contract="config.toml developer_instructions",
            risks="Broad content affects every task.",
            operation={
                "kind": "workspace_config",
                "scope": "project",
                "path": ".codex/config.toml",
                "mechanism": "config",
                "key": "developer_instructions",
            },
            evidence_refs=("probe:help", "runtime:channel:developer_instructions"),
        ),
        _channel(
            "user_instructions",
            visibility="configured instruction context before the user prompt",
            use="Provide standing user-level preferences separate from developer policy.",
            artifact_contract="config.toml instructions",
            risks=(
                "The current codex exec request probe does not observe this config field as a "
                "distinct model-visible layer, so routing behavior to it would be speculative."
            ),
            operation=None,
            evidence_refs=("probe:app-server-schema",),
            status="exists_but_unsupported",
        ),
        _channel(
            "project_instructions",
            visibility="startup developer context scoped by workspace ancestry",
            use="Project-scoped standing instructions.",
            artifact_contract="AGENTS.md in the candidate workspace",
            risks="Nested files can override or extend the effective instruction chain.",
            operation={"kind": "project_file", "path_pattern": "AGENTS.md"},
            evidence_refs=("probe:help", "runtime:channel:project_instructions"),
        ),
        _channel(
            "skills",
            visibility="skill metadata at startup; body on demand",
            use="Triggered procedures with supporting files.",
            artifact_contract=".agents/skills/<slug>/SKILL.md",
            risks="Poor descriptions prevent selection or trigger unrelated tasks.",
            operation={
                "kind": "project_file",
                "path_pattern": ".agents/skills/<slug>/SKILL.md",
            },
            evidence_refs=("runtime:channel:skills",),
        ),
        _channel(
            "skill_configuration",
            visibility="skill discovery, startup metadata, and invocation approval",
            use="Enable, disable, approve, or add discovery roots for skills.",
            artifact_contract="skills configuration and skill approval policy",
            risks=(
                "Changing discovery roots or approval policy can expose unrelated host skills or "
                "alter available capability beyond candidate-owned skill files."
            ),
            operation=None,
            evidence_refs=("probe:app-server-schema",),
            status="forbidden",
        ),
        _channel(
            "hooks",
            visibility="session start, resume, and post-compaction context",
            use=(
                "Reinforce concise experience that must remain visible across a multi-turn "
                "session. The candidate controls context text only; the harness owns the hook "
                "event, matcher, and executable."
            ),
            artifact_contract=(
                ".codex/harness-hook-context.md consumed by the harness-owned SessionStart "
                "dispatcher"
            ),
            risks=(
                "Repeated global injection can bloat context or over-constrain unrelated tasks; "
                "it cannot run candidate-supplied commands."
            ),
            operation={
                "kind": "project_file",
                "path_pattern": ".codex/harness-hook-context.md",
            },
            evidence_refs=(
                "probe:help",
                "probe:features",
                "probe:app-server-schema",
                "runtime:channel:hooks",
            ),
        ),
        _channel(
            "hook_event_handlers",
            visibility="before or after tools, compaction, prompts, subagents, and stop events",
            use="Observe, gate, or inject context at a specific native hook event.",
            artifact_contract=".codex/hooks.json event entries with command, prompt, or agent handlers",
            risks=(
                "Handlers execute candidate code. They must remain inside the per-trial project "
                "and must not read evaluator artifacts, host configuration, or credentials."
            ),
            operation={
                "kind": "project_file",
                "path_pattern": ".codex/hooks.json",
            },
            evidence_refs=("probe:app-server-schema",),
        ),
        _channel(
            "tool_description",
            visibility="whenever the exact MCP tool is available",
            use="Compact call-selection guidance for one exact tool.",
            artifact_contract="tool_desc_patches.<tool>.desc before MCP registration",
            risks="Long guidance bloats every decision involving the tool.",
            operation={
                "kind": "tool_schema_patch",
                "manifest_field": "tool_desc_patches",
                "target": "description",
            },
            evidence_refs=("runtime:channel:tool_description",),
        ),
        _channel(
            "tool_parameter_description",
            visibility="whenever the exact MCP parameter schema is available",
            use="Compact meaning or constraint for one exact parameter.",
            artifact_contract="tool_desc_patches.<tool>.params.<parameter>",
            risks="Description guidance cannot enforce parameter values.",
            operation={
                "kind": "tool_schema_patch",
                "manifest_field": "tool_desc_patches",
                "target": "parameter_description",
            },
            evidence_refs=("runtime:channel:tool_parameter_description",),
        ),
        _channel(
            "mcp_servers",
            visibility="startup tool catalog and tool-call time",
            use="Register an external tool server.",
            artifact_contract="config.toml [mcp_servers.<name>]",
            risks=(
                "Adds executable or network-backed capability; use only candidate-owned local "
                "servers inside the per-trial runtime."
            ),
            operation={
                "kind": "harness_config_patch",
                "manifest_field": "config_patch",
                "key_prefix": "mcp_servers.",
            },
            evidence_refs=("probe:help",),
        ),
        _channel(
            "mcp_tool_policy",
            visibility="tool catalog construction and MCP tool-call approval",
            use="Enable, disable, or require approval for selected tools on an existing server.",
            artifact_contract="mcp_servers.<name>.enabled_tools, disabled_tools, and approval settings",
            risks="Changes the fixed capability and approval surface used for benchmark comparison.",
            operation=None,
            evidence_refs=("probe:app-server-schema",),
            status="forbidden",
        ),
        _channel(
            "tool_enablement",
            visibility="startup tool catalog",
            use="Enable or disable existing Codex tools and features.",
            artifact_contract="config.toml [tools]",
            risks="Capability changes can alter task feasibility and comparability.",
            operation=None,
            evidence_refs=("probe:help", "probe:features"),
            status="forbidden",
        ),
        _channel(
            "feature_flags",
            visibility="startup runtime and tool configuration",
            use="Enable or disable one documented Codex feature.",
            artifact_contract="config.toml [features]",
            risks="Experimental features can alter behavior beyond the intended task.",
            operation=None,
            evidence_refs=("probe:help", "probe:features"),
            status="proposal_only",
        ),
        _channel(
            "agent_definitions",
            visibility="documented project agent directory, but not the active v2 collaboration schema",
            use="Would provide scoped instructions for a named delegated role.",
            artifact_contract=".codex/agents/<name>.toml",
            risks="Codex 0.144.4 v2 collaboration exposes generic spawn_agent without a role selector; files are inert in this eval path.",
            operation=None,
            evidence_refs=(
                "probe:features",
                "probe:app-server-schema",
            ),
            status="exists_but_unsupported",
        ),
        _channel(
            "commands",
            visibility="interactive slash-command selection before a user turn",
            use="Store reusable user-invoked prompt templates.",
            artifact_contract="Codex custom command import surface",
            risks="Non-interactive codex exec does not invoke slash-command selection.",
            operation=None,
            evidence_refs=("probe:app-server-schema",),
            status="exists_but_unsupported",
        ),
        _channel(
            "plugins",
            visibility="startup plugin discovery and plugin-provided runtime surfaces",
            use="Package reusable executable extensions and their tools or hooks.",
            artifact_contract="Codex plugin installation and manifest",
            risks=(
                "Plugins introduce executable code and external capability, violating the fixed "
                "benchmark authority boundary."
            ),
            operation=None,
            evidence_refs=("probe:help", "probe:features", "probe:app-server-schema"),
            status="forbidden",
        ),
        _channel(
            "apps_connectors",
            visibility="startup tool and external-data catalog",
            use="Expose configured apps or connectors as additional data and action surfaces.",
            artifact_contract="config.toml [apps] and plugin/app-server connector state",
            risks="Adds external data access and tools beyond the fixed benchmark environment.",
            operation=None,
            evidence_refs=("probe:features", "probe:app-server-schema"),
            status="forbidden",
        ),
        _channel(
            "base_instructions",
            visibility="base Responses instructions at thread creation",
            use="Replace the model-specific base instruction layer for a thread.",
            artifact_contract="app-server thread/start baseInstructions",
            risks=(
                "codex exec does not expose a bounded project artifact for this field, and "
                "replacing native base instructions can remove harness invariants."
            ),
            operation=None,
            evidence_refs=("probe:app-server-schema",),
            status="exists_but_unsupported",
        ),
        _channel(
            "personality",
            visibility="startup model configuration",
            use="Select a native response-style preset when the target model supports it.",
            artifact_contract="config.toml personality",
            risks=(
                "The current DeepSeek-backed runtime has no request-level evidence that Codex "
                "personality presets change effective instructions."
            ),
            operation=None,
            evidence_refs=("probe:features", "probe:app-server-schema"),
            status="exists_but_unsupported",
        ),
        _channel(
            "collaboration_mode",
            visibility="thread or turn developer context and delegation policy",
            use="Select a built-in or custom collaboration policy for subsequent turns.",
            artifact_contract="thread/turn collaborationMode settings",
            risks=(
                "Can override model, reasoning effort, and developer instructions, and the "
                "non-interactive benchmark runtime does not expose a stable bounded contract."
            ),
            operation=None,
            evidence_refs=("probe:app-server-schema",),
            status="exists_but_unsupported",
        ),
        _channel(
            "memory",
            visibility="startup and model context assembled from persisted prior sessions",
            use="Reuse consolidated experience across sessions.",
            artifact_contract="Codex memories feature and persistent memory store",
            risks="Leaks evidence across tasks or splits and invalidates independent trial attribution.",
            operation=None,
            evidence_refs=("probe:features",),
            status="forbidden",
        ),
        _channel(
            "compaction",
            visibility="automatic or explicit context compaction during a long session",
            use="Control compaction threshold or the prompt used to preserve context.",
            artifact_contract="project .codex/config.toml compact_prompt",
            risks=(
                "The prompt changes what history survives compaction; the fixed runtime still "
                "owns the model context limit and compaction trigger."
            ),
            operation={
                "kind": "workspace_config",
                "scope": "project",
                "path": ".codex/config.toml",
                "mechanism": "config",
                "key": "compact_prompt",
            },
            evidence_refs=("probe:app-server-schema", "runtime:channel:compaction"),
        ),
        _channel(
            "model_runtime",
            visibility="thread creation and every model request",
            use="Select model, provider, reasoning effort, verbosity, summary, or service tier.",
            artifact_contract="model and model runtime configuration",
            risks="Changes the fixed executor model or inference budget instead of improving its harness.",
            operation=None,
            evidence_refs=("probe:app-server-schema",),
            status="forbidden",
        ),
        _channel(
            "exec_policy",
            visibility="before shell or command execution",
            use="Allow, prompt for, or deny matching commands.",
            artifact_contract="project or user execpolicy .rules files",
            risks=(
                "Changes execution authority, and benchmark codex exec intentionally uses "
                "--ignore-rules."
            ),
            operation=None,
            evidence_refs=("probe:help", "probe:app-server-schema"),
            status="forbidden",
        ),
        _channel(
            "environment_authority",
            visibility="runtime filesystem, process, and network enforcement",
            use="Configure writable roots, environment inheritance, sandbox, network, or approvals.",
            artifact_contract="sandbox, permissions, shell environment, and network policy settings",
            risks="Changes task feasibility and benchmark authority boundaries.",
            operation=None,
            evidence_refs=("probe:help", "probe:app-server-schema"),
            status="forbidden",
        ),
        _channel(
            "session_history",
            visibility="thread resume, fork, rollback, and persistent-history replay",
            use="Seed or rewrite the model-visible conversation history.",
            artifact_contract="thread history and historyMode app-server fields",
            risks="Can inject answers or cross-task evidence and bypass normal rollout provenance.",
            operation=None,
            evidence_refs=("probe:app-server-schema",),
            status="forbidden",
        ),
        _channel(
            "profiles",
            visibility="startup configuration selection",
            use="Select a bounded reusable configuration layer.",
            artifact_contract="$CODEX_HOME/<name>.config.toml selected with --profile",
            risks="Profiles can silently alter multiple unrelated settings.",
            operation=None,
            evidence_refs=("probe:help",),
            status="proposal_only",
        ),
        _channel(
            "sandbox_approval",
            visibility="runtime enforcement",
            use="Infrastructure and authority experiments only.",
            artifact_contract="sandbox_mode and approval_policy",
            risks="Changes authority and invalidates behavioral attribution.",
            operation=None,
            evidence_refs=("probe:help",),
            status="proposal_only",
        ),
    )

    def _extra_probe(
        self, *, executable: Path, root: Path, env: Mapping[str, str]
    ) -> dict[str, Any]:
        exec_help = _run((str(executable), "exec", "--help"), cwd=root, env=env)
        features = _run(
            (str(executable), "features", "list"),
            cwd=root,
            env=env,
            allow_failure=True,
        )
        schema_root = root / "app-server-schema"
        _run(
            (
                str(executable),
                "app-server",
                "generate-json-schema",
                "--experimental",
                "--out",
                str(schema_root),
            ),
            cwd=root,
            env=env,
            allow_failure=True,
        )
        return {
            "exec_options": sorted(
                set(re.findall(r"(?m)^\s{2,}(--[a-z0-9-]+)", exec_help))
            ),
            "feature_lines": [
                line.strip() for line in features.splitlines() if line.strip()
            ],
            "app_server_schema": _codex_app_server_surface_probe(schema_root),
        }

    def query_evidence_catalog(self, probe: Mapping[str, Any]) -> list[dict[str, Any]]:
        catalog = super().query_evidence_catalog(probe)
        catalog.append(
            {
                "id": "probe:features",
                "kind": "command_output",
                "source": "codex features list",
                "value": list(probe.get("feature_lines") or []),
            }
        )
        catalog.append(
            {
                "id": "probe:app-server-schema",
                "kind": "generated_native_schema",
                "source": "codex app-server generate-json-schema --experimental",
                "value": dict(probe.get("app_server_schema") or {}),
            }
        )
        return catalog

    def query_channel_inventory(
        self, probe: Mapping[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        current_probe = probe or self.architecture_probe()
        inventory = super().query_channel_inventory(current_probe)
        feature_names = {
            str(line).split()[0]
            for line in current_probe.get("feature_lines") or []
            if str(line).split()
        }
        commands = {str(item) for item in current_probe.get("native_commands") or []}
        options = {
            *[str(item) for item in current_probe.get("native_options") or []],
            *[str(item) for item in current_probe.get("exec_options") or []],
        }
        requirements = {
            "hooks": "hooks" in feature_names
            and "--dangerously-bypass-hook-trust" in options
            and "hooks" in _schema_surface_names(current_probe),
            "user_instructions": "instructions" in _schema_surface_names(current_probe),
            "skill_configuration": "skill_approval"
            in _schema_surface_names(current_probe),
            "plugins": "plugins" in feature_names
            and "plugin" in commands
            and "plugins" in _schema_surface_names(current_probe),
            "personality": "personality" in feature_names
            and "personality" in _schema_surface_names(current_probe),
            "hook_event_handlers": bool(
                (current_probe.get("app_server_schema") or {}).get("hook_events")
            ),
            "mcp_tool_policy": "mcpServers" in _schema_surface_names(current_probe),
            "commands": "commands" in _schema_surface_names(current_probe),
            "apps_connectors": "apps" in feature_names
            and "apps" in _schema_surface_names(current_probe),
            "base_instructions": "baseInstructions"
            in _schema_surface_names(current_probe),
            "collaboration_mode": "collaborationMode"
            in _schema_surface_names(current_probe),
            "memory": "memories" in feature_names,
            "compaction": "compact_prompt" in _schema_surface_names(current_probe),
            "model_runtime": "model" in _schema_surface_names(current_probe),
            "exec_policy": "rules" in _schema_surface_names(current_probe),
            "environment_authority": "sandbox_mode"
            in _schema_surface_names(current_probe),
            "session_history": "historyMode" in _schema_surface_names(current_probe),
        }
        return [
            channel
            for channel in inventory
            if channel["id"] not in requirements or requirements[str(channel["id"])]
        ]


def _codex_app_server_surface_probe(schema_root: Path) -> dict[str, Any]:
    v2 = schema_root / "v2"
    return {
        "external_agent_import_fields": _schema_property_names(
            v2 / "ExternalAgentConfigImportParams.json"
        ),
        "config_fields": _schema_property_names(v2 / "ConfigReadResponse.json"),
        "thread_start_fields": _top_level_schema_properties(
            v2 / "ThreadStartParams.json"
        ),
        "turn_start_fields": _top_level_schema_properties(v2 / "TurnStartParams.json"),
        "hook_events": _schema_enum(v2 / "HooksListResponse.json", "HookEventName"),
        "hook_handler_types": _schema_enum(
            v2 / "HooksListResponse.json", "HookHandlerType"
        ),
    }


def _load_schema(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _schema_property_names(path: Path) -> list[str]:
    names: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            properties = value.get("properties")
            if isinstance(properties, Mapping):
                names.update(str(key) for key in properties)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(_load_schema(path))
    return sorted(names)


def _top_level_schema_properties(path: Path) -> list[str]:
    properties = _load_schema(path).get("properties") or {}
    return (
        sorted(str(key) for key in properties)
        if isinstance(properties, Mapping)
        else []
    )


def _schema_enum(path: Path, definition: str) -> list[str]:
    definitions = _load_schema(path).get("definitions") or {}
    payload = definitions.get(definition) if isinstance(definitions, Mapping) else None
    values = payload.get("enum") if isinstance(payload, Mapping) else None
    return [str(item) for item in values or []]


def _schema_surface_names(probe: Mapping[str, Any]) -> set[str]:
    schema = probe.get("app_server_schema") or {}
    return {
        str(item)
        for field in (
            "external_agent_import_fields",
            "config_fields",
            "thread_start_fields",
            "turn_start_fields",
        )
        for item in schema.get(field) or []
    }


def _doc_evidence(evidence_id: str, path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "id": evidence_id,
            "kind": "local_documentation",
            "source": str(path),
            "sha256": "missing",
        }
    content = path.read_text(encoding="utf-8", errors="replace")
    max_chars = 48_000
    return {
        "id": evidence_id,
        "kind": "local_documentation",
        "source": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "content": content[:max_chars],
        "truncated": len(content) > max_chars,
    }


def _run(
    command: tuple[str, ...],
    *,
    cwd: Path,
    env: Mapping[str, str],
    allow_failure: bool = False,
) -> str:
    process = subprocess.run(
        list(command),
        cwd=str(cwd),
        env=dict(env),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    output = (process.stdout or process.stderr or "").strip()
    if process.returncode != 0 and not allow_failure:
        raise RuntimeError(
            f"probe command failed: {' '.join(command)}: {output[-1000:]}"
        )
    return output
