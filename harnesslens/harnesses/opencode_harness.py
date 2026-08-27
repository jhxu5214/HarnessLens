from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


DEFAULT_OPENCODE_PREFIX = "~/.opencode"


def _channel(
    channel_id: str,
    *,
    visibility: str,
    use: str,
    artifact_contract: str,
    risks: str,
    materialization: str,
) -> dict[str, str]:
    return {
        "id": channel_id,
        "visibility": visibility,
        "use": use,
        "artifact_contract": artifact_contract,
        "risks": risks,
        "materialization": materialization,
    }


OPENCODE_CHANNELS = (
    _channel(
        "instructions_rules",
        visibility="startup/always-visible",
        use="Concise universal or narrowly guarded behavior.",
        artifact_contract=(
            "project/opencode.json instructions entries referencing captured "
            "project instruction files"
        ),
        risks="Broad wording affects unrelated tasks.",
        materialization="main_candidate",
    ),
    _channel(
        "skills",
        visibility="skill tool visible at startup; selected skill metadata and body load on demand",
        use="Detailed conditional SOPs with branches and parameters.",
        artifact_contract=(
            "Write .opencode/skills/<slug>/SKILL.md with matching YAML name and "
            "description; enable tools.skill and permission.skill when explicit."
        ),
        risks="A vague trigger is not selected; a broad trigger primes unrelated tasks.",
        materialization="main_candidate",
    ),
    _channel(
        "tool_description",
        visibility="whenever the exact tool is available",
        use="Local call-selection guidance for one exact tool.",
        artifact_contract="tool_desc_patches.<tool>.desc",
        risks="Long workflows bloat every call decision.",
        materialization="main_candidate",
    ),
    _channel(
        "tool_parameter_description",
        visibility="whenever the exact parameter schema is available",
        use="Local argument meaning or constraint.",
        artifact_contract="tool_desc_patches.<tool>.params.<parameter>",
        risks="Schema guidance cannot enforce values.",
        materialization="main_candidate",
    ),
    _channel(
        "agent_definitions",
        visibility="agent catalog and selected-agent prompt",
        use="A specialized prompt-only worker with a clear trigger.",
        artifact_contract=".opencode/agents/<name>.md or config agent.<name>.*",
        risks="The role may not be selected or may lose context.",
        materialization="main_candidate",
    ),
    _channel(
        "system_prompt",
        visibility="highest-salience startup frame",
        use="Universal framing that cannot be safely additive.",
        artifact_contract="config_patch agent.build.prompt as a complete replacement",
        risks="Replacement can delete native behavior.",
        materialization="isolated_candidate",
    ),
    _channel(
        "agent_tool_config",
        visibility="per-agent tool catalog and policy",
        use="Controlled changes to an existing agent tool surface.",
        artifact_contract="config agent.<name>.tools or agent.<name>.permission",
        risks="Capability changes can make scores incomparable.",
        materialization="isolated_candidate",
    ),
    _channel(
        "builtin_tools",
        visibility="startup tool catalog",
        use="Enable or disable an existing OpenCode built-in.",
        artifact_contract="config tools or permission",
        risks="Changes agent authority and task feasibility.",
        materialization="isolated_candidate",
    ),
    _channel(
        "permissions",
        visibility="tool-time enforcement",
        use="Restrict or explicitly allow existing operations.",
        artifact_contract="config permission including permission.skill",
        risks="Over-restriction causes omissions; over-permission expands blast radius.",
        materialization="isolated_candidate",
    ),
    _channel(
        "commands",
        visibility="command catalog and explicit invocation",
        use="Reusable bounded harness operations.",
        artifact_contract=".opencode/commands/<name>.md or config command.<name>",
        risks="Commands can execute side effects.",
        materialization="isolated_candidate",
    ),
    _channel(
        "reference_sources",
        visibility="on demand or instruction-linked",
        use="Large stable knowledge that should not occupy every prompt.",
        artifact_contract="Manifest file plus a real startup/read retrieval path.",
        risks="The model may not retrieve it; stale sources can mislead.",
        materialization="main_candidate",
    ),
    _channel(
        "compaction_config",
        visibility="session context lifecycle",
        use="Controlled context-retention hypotheses.",
        artifact_contract="documented config compaction fields",
        risks="Changes which evidence remains visible.",
        materialization="isolated_candidate",
    ),
    _channel(
        "plugin_tool_system",
        visibility="startup hooks and added tools",
        use="Separately budgeted executable integrations.",
        artifact_contract=".opencode/plugins/*.ts or config plugin",
        risks="Arbitrary code execution and side effects.",
        materialization="isolated_candidate",
    ),
    _channel(
        "custom_tools",
        visibility="startup and tool invocation",
        use="Separately budgeted capability expansion.",
        artifact_contract=".opencode/tools/*.ts plus permissions",
        risks="Adds executable behavior.",
        materialization="isolated_candidate",
    ),
    _channel(
        "mcp_servers",
        visibility="startup and external tool invocation",
        use="Add a separately budgeted external service.",
        artifact_contract="config mcp with complete command/URL and auth source",
        risks="Network, credentials, nondeterminism, and capability expansion.",
        materialization="isolated_candidate",
    ),
    _channel(
        "experimental_policies",
        visibility="runtime-dependent startup behavior",
        use="Pinned experiments on documented unstable fields.",
        artifact_contract="config experimental with pinned OpenCode version",
        risks="Version drift and unsupported semantics.",
        materialization="isolated_candidate",
    ),
    _channel(
        "model_provider",
        visibility="model runtime",
        use="Compare full agent systems, not learned harness behavior.",
        artifact_contract="separate experiment track",
        risks="Destroys attribution to the harness.",
        materialization="proposal_only",
    ),
    _channel(
        "runtime_env",
        visibility="process runtime",
        use="Diagnose infrastructure in an isolated track.",
        artifact_contract="redacted effective environment",
        risks="Connectivity failures, secret exposure, and non-portability.",
        materialization="proposal_only",
    ),
    _channel(
        "file_attachments",
        visibility="per-message context",
        use="Test a runtime that natively supplies attachments.",
        artifact_contract="content-addressed attachment through a runtime adapter",
        risks="The current rollout bridge cannot materialize this channel.",
        materialization="proposal_only",
    ),
    _channel(
        "tool_return",
        visibility="post-call context",
        use="Separate experiments on observation formatting.",
        artifact_contract="change the owning tool runtime and record schemas",
        risks="Changes environment semantics.",
        materialization="proposal_only",
    ),
    _channel(
        "diagnostic",
        visibility="process and session telemetry",
        use="Infrastructure diagnosis only.",
        artifact_contract="isolated redacted runtime configuration",
        risks="Can expose secrets, alter timing, or leak evaluator data.",
        materialization="proposal_only",
    ),
)


_OPENCODE_OPERATIONS: Mapping[str, Mapping[str, Any]] = {
    "instructions_rules": {
        "kind": "workspace_config",
        "scope": "project",
        "path": "opencode.json",
        "mechanism": "config",
        "key": "instructions",
    },
    "skills": {
        "kind": "project_file",
        "path_pattern": ".opencode/skills/<slug>/SKILL.md",
    },
    "tool_description": {
        "kind": "tool_schema_patch",
        "manifest_field": "tool_desc_patches",
        "target": "description",
    },
    "tool_parameter_description": {
        "kind": "tool_schema_patch",
        "manifest_field": "tool_desc_patches",
        "target": "parameter_description",
    },
    "agent_definitions": {
        "kind": "project_file",
        "path_pattern": ".opencode/agents/<name>.md",
    },
    "system_prompt": {
        "kind": "harness_config_patch",
        "manifest_field": "config_patch",
        "key": "agent.build.prompt",
    },
    "agent_tool_config": {
        "kind": "harness_config_patch",
        "manifest_field": "config_patch",
        "key_prefix": "agent.",
    },
    "builtin_tools": {
        "kind": "harness_config_patch",
        "manifest_field": "config_patch",
        "key_prefix": "tools.",
    },
    "permissions": {
        "kind": "harness_config_patch",
        "manifest_field": "config_patch",
        "key_prefix": "permission.",
    },
    "commands": {
        "kind": "project_file",
        "path_pattern": ".opencode/commands/<name>.md",
    },
    "reference_sources": {
        "kind": "project_file",
        "path_pattern": "<relative-path>",
    },
    "compaction_config": {
        "kind": "harness_config_patch",
        "manifest_field": "config_patch",
        "key_prefix": "compaction.",
    },
    "plugin_tool_system": {
        "kind": "project_file",
        "path_pattern": ".opencode/plugins/<name>.ts",
    },
    "custom_tools": {
        "kind": "project_file",
        "path_pattern": ".opencode/tools/<name>.ts",
    },
    "mcp_servers": {
        "kind": "harness_config_patch",
        "manifest_field": "config_patch",
        "key_prefix": "mcp.",
    },
    "experimental_policies": {
        "kind": "harness_config_patch",
        "manifest_field": "config_patch",
        "key_prefix": "experimental.",
    },
}


_OPENCODE_REQUEST_OBSERVATIONS: Mapping[str, tuple[str, ...]] = {
    "instructions_rules": ("content_present_in_agent_request",),
    "system_prompt": ("content_present_in_agent_request",),
    "skills": (
        "skill_tool_present_in_agent_request",
        "skill_metadata_and_body_absent_from_startup_request",
    ),
}

_OPENCODE_REQUEST_EDIT_CONTRACTS: Mapping[str, Mapping[str, str]] = {
    "instructions_rules": {
        "scope": "project",
        "path": "opencode.json",
        "mechanism": "config",
        "key": "instructions",
    },
}


@dataclass(frozen=True)
class HarnessCapabilities:
    channels: tuple[Mapping[str, Any], ...]


class OpencodeHarnessAdapter:
    def __init__(self, *, repo_root: str | Path | None = None) -> None:
        self.repo_root = (
            Path(repo_root).resolve()
            if repo_root is not None
            else Path(__file__).resolve().parents[2]
        )
        self.capabilities = HarnessCapabilities(OPENCODE_CHANNELS)

    def architecture_probe(self) -> dict[str, Any]:
        prefix = Path(
            os.environ.get("OPENCODE_PREFIX") or DEFAULT_OPENCODE_PREFIX
        ).expanduser()
        executable = shutil.which("opencode") or str(prefix / "bin" / "opencode")
        if not Path(executable).is_file():
            raise RuntimeError(f"opencode executable is unavailable: {executable}")
        with tempfile.TemporaryDirectory(prefix="harnesslens-opencode-probe-") as raw_root:
            root = Path(raw_root)
            env = {
                **os.environ,
                "NO_COLOR": "1",
                "XDG_DATA_HOME": str(root / "data"),
                "XDG_CONFIG_HOME": str(root / "config"),
                "XDG_STATE_HOME": str(root / "state"),
                "XDG_CACHE_HOME": str(root / "cache"),
                "TMPDIR": str(root / "tmp"),
            }
            version = _run_probe(executable, ("--version",), root, env).strip()
            help_text = _run_probe(executable, ("--help",), root, env)
            debug_help = _run_probe(executable, ("debug", "--help"), root, env)
            config_text = _run_probe(executable, ("debug", "config"), root, env)
            agent_text = _run_probe(executable, ("agent", "list"), root, env)
        try:
            resolved_config = json.loads(config_text)
        except json.JSONDecodeError:
            resolved_config = {}
        facts = {
            "harness_id": "opencode",
            "harness_version": version,
            "native_commands": _commands(help_text, r"^\s{2}opencode\s+([^\n]+?)\s{2,}"),
            "native_debug_commands": _commands(
                debug_help, r"^\s{2}opencode\s+debug\s+([^\n]+?)\s{2,}"
            ),
            "resolved_config_keys": sorted(resolved_config) if isinstance(resolved_config, dict) else [],
            "native_agents": [
                {"name": match.group(1), "mode": match.group(2)}
                for match in re.finditer(
                    r"^([A-Za-z0-9._-]+)\s+\((primary|subagent|all)\)$",
                    _plain(agent_text),
                    flags=re.M,
                )
            ],
            "materialization_points": [dict(item) for item in OPENCODE_CHANNELS],
        }
        facts["probe_digest"] = hashlib.sha256(
            json.dumps(facts, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return facts

    def query_evidence_catalog(self, probe: Mapping[str, Any]) -> list[dict[str, Any]]:
        catalog = [
            {
                "id": "probe:version",
                "kind": "command_output",
                "source": "opencode --version",
                "value": str(probe.get("harness_version") or ""),
            },
            {
                "id": "probe:help",
                "kind": "command_output",
                "source": "opencode --help",
                "value": list(probe.get("native_commands") or []),
            },
            {
                "id": "probe:debug-config",
                "kind": "resolved_state",
                "source": "opencode debug config in an isolated HOME",
                "value": list(probe.get("resolved_config_keys") or []),
            },
            {
                "id": "probe:agents",
                "kind": "command_output",
                "source": "opencode agent list",
                "value": list(probe.get("native_agents") or []),
            },
            {
                "id": "runtime:probe:opencode-startup",
                "kind": "request_sentinel_probe",
                "source": "tests/test_harness_query_runtime.py",
                "value": {
                    "observed_channels": sorted(_OPENCODE_REQUEST_OBSERVATIONS),
                    "observations": {
                        key: list(value)
                        for key, value in _OPENCODE_REQUEST_OBSERVATIONS.items()
                    },
                    "workspace_edit_contracts": {
                        key: dict(value)
                        for key, value in _OPENCODE_REQUEST_EDIT_CONTRACTS.items()
                    },
                },
            },
        ]
        for channel in OPENCODE_CHANNELS:
            channel_id = str(channel["id"])
            if channel_id not in _OPENCODE_OPERATIONS:
                continue
            catalog.append(
                {
                    "id": f"runtime:channel:{channel_id}",
                    "kind": "materializer_contract",
                    "source": "OpencodeHarnessAdapter manifest validation and channel routing",
                    "value": dict(_OPENCODE_OPERATIONS[channel_id]),
                }
            )
        docs_root = self.repo_root / "assets" / "docs_cache" / "opencode"
        for name in (
            "config.md",
            "config.schema.json",
            "rules.md",
            "skills.md",
            "agents.md",
            "tools.md",
            "custom-tools.md",
            "plugins.md",
            "commands.md",
            "mcp-servers.md",
            "permissions.md",
        ):
            catalog.append(
                _doc_probe_evidence(
                    f"docs:opencode:{Path(name).stem}", docs_root / name
                )
            )
        return catalog

    def query_channel_inventory(
        self, probe: Mapping[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        inventory: list[dict[str, Any]] = []
        for raw in OPENCODE_CHANNELS:
            channel = dict(raw)
            channel_id = str(channel["id"])
            operation = _OPENCODE_OPERATIONS.get(channel_id)
            if operation is None:
                channel.update(
                    {
                        "status": "proposal_only",
                        "evidence_refs": ["probe:version"],
                    }
                )
            else:
                observations = _OPENCODE_REQUEST_OBSERVATIONS.get(channel_id, ())
                evidence_refs = [
                    "probe:version",
                    f"runtime:channel:{channel_id}",
                ]
                if observations:
                    evidence_refs.append("runtime:probe:opencode-startup")
                channel.update(
                    {
                        "status": "verified",
                        "evidence_refs": evidence_refs,
                        "verification": {
                            "level": (
                                "request_trace"
                                if observations
                                else "materializer_contract"
                            ),
                            "runtime_observed": bool(observations),
                            "observations": list(observations),
                        },
                        "operation": dict(operation),
                    }
                )
            inventory.append(channel)
        return inventory

    def validate_delta(self, delta: Mapping[str, Any]) -> None:
        manifest = normalize_opencode_manifest(delta)
        config = manifest["config_patch"]
        _validate_config_patch(config)
        skill_paths = {
            str(item.get("path") or "")
            for item in manifest["files"]
            if _skill_slug(str(item.get("path") or ""))
        }
        for item in manifest["files"]:
            path = str(item.get("path") or "")
            _validate_path(path)
            slug = _skill_slug(path)
            if path.startswith(".opencode/skills/") and path.endswith("/SKILL.md") and not slug:
                raise ValueError("skill names must be lowercase hyphenated slugs")
            if slug:
                _validate_skill(slug, str(item.get("content") or ""), config)
            elif path.startswith(".opencode/skills/"):
                owner = "/".join(path.split("/")[:3] + ["SKILL.md"])
                if owner not in skill_paths:
                    raise ValueError(f"skill companion file requires its SKILL.md: {path}")
        _validate_tool_patches(manifest["tool_desc_patches"])

    def materialized_channel_ids(self, delta: Mapping[str, Any]) -> set[str]:
        manifest = normalize_opencode_manifest(delta)
        channels: set[str] = set()
        if manifest["instructions"] or manifest["prompt_appends"]:
            channels.add("instructions_rules")
        for item in manifest["files"]:
            path = str(item.get("path") or "")
            if path.startswith(".opencode/skills/"):
                channels.add("skills")
            elif path.startswith(".opencode/agents/"):
                channels.add("agent_definitions")
            elif path.startswith(".opencode/commands/"):
                channels.add("commands")
            elif path.startswith(".opencode/plugins/"):
                channels.add("plugin_tool_system")
            elif path.startswith(".opencode/tools/"):
                channels.add("custom_tools")
            else:
                channels.add("reference_sources")
        for key in manifest["config_patch"]:
            text = str(key)
            if text == "tools.skill" or text.startswith("permission.skill."):
                continue
            if text == "agent.build.prompt":
                channels.add("system_prompt")
            elif text.startswith("agent."):
                channels.add("agent_tool_config")
            elif text.startswith("tools."):
                channels.add("builtin_tools")
            elif text.startswith("permission."):
                channels.add("permissions")
            elif text.startswith("command."):
                channels.add("commands")
            elif text.startswith("compaction."):
                channels.add("compaction_config")
            elif text == "plugin" or text.startswith("plugin."):
                channels.add("plugin_tool_system")
            elif text == "mcp" or text.startswith("mcp."):
                channels.add("mcp_servers")
            elif text == "experimental" or text.startswith("experimental."):
                channels.add("experimental_policies")
        for patch in manifest["tool_desc_patches"].values():
            if isinstance(patch, Mapping) and str(patch.get("desc") or "").strip():
                channels.add("tool_description")
            if isinstance(patch, Mapping) and patch.get("params"):
                channels.add("tool_parameter_description")
        return channels


def normalize_opencode_manifest(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("OpenCode candidate delta must be an object")
    allowed = {
        "config_patch",
        "files",
        "prompt_appends",
        "instructions",
        "removals",
        "tool_desc_patches",
        "replace_channels",
        "replace_instructions",
        "replace_prompt_appends",
        "replace_tool_desc_patches",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unsupported OpenCode delta fields: {unknown}")
    files = raw.get("files") or []
    if not isinstance(files, list) or any(not isinstance(item, Mapping) for item in files):
        raise ValueError("OpenCode delta files must be an array of objects")
    config = raw.get("config_patch") or {}
    tools = raw.get("tool_desc_patches") or {}
    if not isinstance(config, Mapping) or not isinstance(tools, Mapping):
        raise ValueError("config_patch and tool_desc_patches must be objects")
    replace = set(_strings(raw.get("replace_channels"), "replace_channels"))
    if raw.get("replace_instructions"):
        replace.add("instructions")
    if raw.get("replace_prompt_appends"):
        replace.add("prompt_appends")
    if raw.get("replace_tool_desc_patches"):
        replace.add("tool_desc_patches")
    if replace - {"instructions", "prompt_appends", "tool_desc_patches"}:
        raise ValueError("unsupported replacement channel")
    return {
        "config_patch": dict(config),
        "files": [dict(item) for item in files],
        "prompt_appends": _strings(raw.get("prompt_appends"), "prompt_appends"),
        "instructions": _strings(raw.get("instructions"), "instructions"),
        "removals": _strings(raw.get("removals"), "removals"),
        "tool_desc_patches": dict(tools),
        "replace_channels": sorted(replace),
    }


def _strings(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple, set)) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValueError(f"{name} must be an array of strings")
    return [item for item in value if item.strip()]


def _validate_path(raw: str) -> None:
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts or "\\" in raw:
        raise ValueError(f"unsafe candidate file path: {raw}")


def _skill_slug(path: str) -> str | None:
    match = re.fullmatch(r"\.opencode/skills/([a-z0-9]+(?:-[a-z0-9]+)*)/SKILL\.md", path)
    return match.group(1) if match else None


def _validate_skill(slug: str, content: str, config: Mapping[str, Any]) -> None:
    if not content.startswith("---\n") or "\n---\n" not in content[4:]:
        raise ValueError("SKILL.md requires YAML frontmatter")
    frontmatter = content.split("\n---\n", 1)[0]
    name = re.search(r"(?m)^name:\s*([^\s]+)\s*$", frontmatter)
    description = re.search(r"(?m)^description:\s*(.+?)\s*$", frontmatter)
    if not name or name.group(1) != slug or not description or not description.group(1).strip():
        raise ValueError("SKILL.md name must match its path and include a description")
    if config.get("tools.skill") is not True:
        raise ValueError("skill candidates must enable config_patch tools.skill")


def _validate_tool_patches(patches: Mapping[str, Any]) -> None:
    for tool_name, raw in patches.items():
        if not str(tool_name).strip() or not isinstance(raw, Mapping):
            raise ValueError("tool description patches require exact tool names and objects")
        unknown = set(raw) - {"desc", "params"}
        if unknown:
            raise ValueError(f"unsupported tool patch fields: {sorted(unknown)}")
        if "desc" in raw and not str(raw.get("desc") or "").strip():
            raise ValueError("tool description patch cannot be empty")
        params = raw.get("params") or {}
        if not isinstance(params, Mapping) or any(
            not str(name).strip() or not str(value).strip() for name, value in params.items()
        ):
            raise ValueError("tool parameter patches require nonempty names and descriptions")


def _validate_config_patch(config: Mapping[str, Any]) -> None:
    allowed_prefixes = (
        "agent.",
        "tools.",
        "permission.",
        "command.",
        "compaction.",
        "plugin.",
        "mcp.",
        "experimental.",
    )
    allowed_exact = {"plugin", "mcp", "experimental"}
    for key in config:
        text = str(key)
        if text in {"model", "provider"} or text.startswith("provider."):
            raise ValueError("model/provider changes are proposal-only in HarnessLens")
        if text not in allowed_exact and not text.startswith(allowed_prefixes):
            raise ValueError(f"unsupported OpenCode config patch key: {text}")


def _doc_probe_evidence(evidence_id: str, path: Path) -> dict[str, Any]:
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


def _run_probe(executable: str, args: tuple[str, ...], cwd: Path, env: Mapping[str, str]) -> str:
    completed = subprocess.run(
        [executable, *args],
        cwd=cwd,
        env=dict(env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-1000:]
        raise RuntimeError(f"OpenCode architecture probe failed for {args}: {detail}")
    return completed.stdout or completed.stderr


def _plain(value: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value or "")


def _commands(value: str, pattern: str) -> list[str]:
    return sorted({match.group(1).strip() for match in re.finditer(pattern, _plain(value), re.M)})
