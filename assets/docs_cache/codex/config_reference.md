# Codex CLI — Configuration Reference (official, fetched 2026-06-09 from developers.openai.com/codex/config-reference)

Config in `~/.codex/config.toml` (user) + `.codex/config.toml` (project, trusted only). `-c key=value` overrides; `--profile` overlays.

## Instructions / project docs (model-visible context)
- `model_instructions_file` (path): REPLACES built-in agent instructions (instead of AGENTS.md).
- `developer_instructions` (string): additional developer instructions injected into the session.
- AGENTS.md: project instructions, merged repo-root → cwd; `AGENTS.override.md` takes priority; project `.codex/AGENTS.md` overrides global.
- `project_doc_fallback_filenames` (array): extra filenames to try when AGENTS.md missing.
- `project_doc_max_bytes` (default 32 KiB): max bytes read from AGENTS.md.
- `project_root_markers` (array, default [".git"]): how project root is discovered.

## MCP servers (external tools)
- `[mcp_servers.<id>]`: command/args/cwd/env/url (stdio or HTTP), bearer_token_env_var, http_headers.
- `mcp_servers.<id>.enabled`, `.required`, `.startup_timeout_sec`, `.tool_timeout_sec`.
- `mcp_servers.<id>.enabled_tools` / `.disabled_tools`: per-server tool allow/deny.
- `mcp_servers.<id>.default_tools_approval_mode` (auto|prompt|approve); `.tools.<tool>.approval_mode`: per-tool approval.
- MCP allowlist (server name + identity must match) disables non-listed servers.
- `mcp_oauth_callback_port` / `mcp_oauth_callback_url` / `mcp_oauth_credentials_store`.

## Tool enablement (features / tools toggles)
- `[features]` flags: `shell_tool` (default true), `unified_exec` (default true), `web_search`(deprecated), `multi_agent` (default true: spawn_agent/send_input/resume_agent/wait_agent/close_agent), `apps` (ChatGPT Apps/connectors), `hooks`, `memories` (default false), `undo`, `personality` (default true), `fast_mode`, `shell_snapshot`, `skill_mcp_dependency_install`, `prevent_idle_sleep`, `enable_request_compression`, `network_proxy`.
- `tools.view_image` (bool): local-image attachment tool.
- `tools.web_search` (bool|object: context_size/allowed_domains/location); top-level `web_search` (disabled|cached|live).

## Approval / sandbox / permission policy
- `approval_policy` (untrusted|on-request|never|{granular={sandbox_approval,rules,mcp_elicitations,request_permissions,skill_approval}}).
- `approvals_reviewer` (user|auto_review).
- `sandbox_mode` (read-only|workspace-write|danger-full-access); `[sandbox_workspace_write]` writable_roots/network_access.
- `default_permissions` (:read-only|:workspace|:danger-full-access|name).
- `[permissions.<name>]` named profiles: extends, description, filesystem (per-path read/write/deny), network (enabled/mode/domains/unix_sockets/...), workspace_roots.

## Hooks (lifecycle, can inject model-visible context)
- `features.hooks` (bool) enables hooks from `hooks.json` or inline `[hooks]`.
- `[hooks.<Event>]` events: PreToolUse, PermissionRequest, PostToolUse, PreCompact, PostCompact, SessionStart, SubagentStart, SubagentStop, UserPromptSubmit, Stop. Each = matcher groups → command handlers.

## Agents / subagents (multi-agent)
- `[agents.<name>]`: description (role guidance for spawn), config_file (TOML layer for the role), nickname_candidates.
- `agents.max_threads` (6), `agents.max_depth` (1), `agents.job_max_runtime_seconds` (1800).

## Skills
- `[[skills.config]]`: path (folder with SKILL.md) + enabled — per-skill enable/disable override. Skills live in `.agents/skills/` (project) or `~/.codex/skills/` (user).

## Memories (cross-session)
- `features.memories` (default false). `memories.use_memories` (inject existing memories into future sessions), `.generate_memories`, `.extract_model`, `.consolidation_model`, `.disable_on_external_context`, age/idle/rate caps.

## Compaction prompt (summary presentation)
- `compact_prompt` (string): inline override for the history-compaction prompt.
- `experimental_compact_prompt_file` (path): load compaction prompt from file.
- `model_auto_compact_token_limit`: token threshold triggering auto-compaction.

## Personality (response style)
- `personality` (none|friendly|pragmatic): default communication style for models advertising supportsPersonality; `/personality` command.

## Plugins (bundle MCP servers)
- `[plugins.<plugin>.mcp_servers.<server>]`: enabled / enabled_tools / disabled_tools / default_tools_approval_mode / tools.<tool>.approval_mode.

## Apps / connectors
- `[apps._default]` enabled/destructive_enabled/open_world_enabled; `[apps.<id>]` enabled/default_tools_enabled/default_tools_approval_mode/tools.<tool>.{enabled,approval_mode}.

## Image / file attachments
- `-i/--image <FILE>...`: attach image(s) to the initial prompt.

## Tool output
- `tool_output_token_limit` (number): token budget for storing a tool/function output in history (knob).

## Profiles / model (NOT channels: model+config layering)
- `model`, `model_provider`, `[model_providers.<id>]`, `profile`, `profiles`, `model_reasoning_effort`, `model_verbosity`, `service_tier`.

## EXCLUDED — pure execution/UI/observability knobs (NOT channels)
- `[tui.*]` (keymap/theme/animations/status_line/terminal_title/notifications), `disable_paste_burst`, `history.*`, `sqlite_home`, `log_dir`, `[otel.*]`, auth (`cli_auth_credentials_store`/`forced_login_method`/`chatgpt_base_url`), `notify`, `feedback.enabled`, `check_for_update_on_startup`, `[notice.*]`, `[windows.*]`, `file_opener`, timeouts, reasoning effort, model choice.
