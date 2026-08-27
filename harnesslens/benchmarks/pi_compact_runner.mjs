#!/usr/bin/env node

import fs from "node:fs";
import process from "node:process";
import {
  createAgentSession,
  DefaultResourceLoader,
  ModelRuntime,
  SessionManager,
  SettingsManager,
} from "/opt/harness/pi-node-modules/@earendil-works/pi-coding-agent/dist/index.js";
import { configureHttpDispatcher } from "/opt/harness/pi-node-modules/@earendil-works/pi-coding-agent/dist/core/http-dispatcher.js";

configureHttpDispatcher();

const MAX_STRING_CHARS = 8192;
const MAX_ARRAY_ITEMS = 24;
const MAX_OBJECT_KEYS = 32;
const MAX_EVENT_CHARS = 65536;

function parseArgs(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith("--") || value === undefined) {
      throw new Error(`Invalid argument near ${key ?? "<end>"}`);
    }
    result[key.slice(2)] = value;
  }
  return result;
}

function compactValue(value, depth = 0) {
  if (typeof value === "string") {
    return value.length <= MAX_STRING_CHARS
      ? value
      : `${value.slice(0, MAX_STRING_CHARS)}\n...[truncated]...`;
  }
  if (value === null || typeof value !== "object") {
    return value;
  }
  if (depth >= 5) {
    return "[nested value truncated]";
  }
  if (Array.isArray(value)) {
    const compacted = value
      .slice(0, MAX_ARRAY_ITEMS)
      .map((item) => compactValue(item, depth + 1));
    if (value.length > MAX_ARRAY_ITEMS) {
      compacted.push(`[${value.length - MAX_ARRAY_ITEMS} more items]`);
    }
    return compacted;
  }
  const entries = Object.entries(value).slice(0, MAX_OBJECT_KEYS);
  const compacted = Object.fromEntries(
    entries.map(([key, item]) => [key, compactValue(item, depth + 1)]),
  );
  if (Object.keys(value).length > MAX_OBJECT_KEYS) {
    compacted.__truncated_keys__ = true;
  }
  return compacted;
}

function compactMessage(message) {
  if (!message || typeof message !== "object") {
    return undefined;
  }
  return {
    role: message.role,
    content: compactValue(message.content ?? []),
    stopReason: message.stopReason,
    errorMessage: compactValue(message.errorMessage ?? ""),
    usage: compactValue(message.usage ?? {}),
  };
}

function compactEvent(event) {
  switch (event.type) {
    case "agent_start":
    case "turn_start":
      return { type: event.type };
    case "agent_end":
      return { type: event.type };
    case "turn_end":
      return { type: event.type };
    case "message_start":
      return { type: event.type, role: event.message?.role };
    case "message_end":
      return { type: event.type, message: compactMessage(event.message) };
    case "tool_execution_start":
      return {
        type: event.type,
        toolCallId: event.toolCallId,
        toolName: event.toolName,
        args: compactValue(event.args),
      };
    case "tool_execution_end":
      return {
        type: event.type,
        toolCallId: event.toolCallId,
        toolName: event.toolName,
        isError: Boolean(event.isError),
        result: compactValue(event.result),
      };
    case "auto_retry_start":
    case "auto_retry_end":
    case "compaction_start":
    case "compaction_end":
      return compactValue(event);
    default:
      return undefined;
  }
}

function emit(event) {
  let line = JSON.stringify(event);
  if (line.length > MAX_EVENT_CHARS) {
    line = JSON.stringify({
      type: event.type,
      truncated: true,
      originalChars: line.length,
    });
  }
  process.stdout.write(`${line}\n`);
}

function loadAppendSystemPrompts(cwd) {
  const path = `${cwd}/.pi/APPEND_SYSTEM.md`;
  if (!fs.existsSync(path)) {
    return [];
  }
  const content = fs.readFileSync(path, "utf8").trim();
  return content ? [content] : [];
}

const args = parseArgs(process.argv.slice(2));
if (args["self-test"] === "1") {
  const cumulativeUpdate = {
    type: "message_update",
    message: { role: "assistant", content: [{ type: "text", text: "x".repeat(100000) }] },
  };
  const compactedTool = compactEvent({
    type: "tool_execution_start",
    toolCallId: "self-test-call",
    toolName: "bash",
    args: { command: "pwd" },
  });
  if (compactEvent(cumulativeUpdate) !== undefined || !compactedTool) {
    throw new Error("Pi compact event filtering self-test failed");
  }
  emit({
    type: "runner_self_test",
    passed: true,
    compactedTool,
  });
  process.exit(0);
}
const promptFile = args["prompt-file"];
const systemPromptFile = args["system-prompt-file"];
const maxSteps = Number.parseInt(args["max-steps"] ?? "50", 10);
if (!promptFile || !systemPromptFile || !Number.isFinite(maxSteps) || maxSteps < 1) {
  throw new Error("--prompt-file, --system-prompt-file, and a positive --max-steps are required");
}

const prompt = fs.readFileSync(promptFile, "utf8");
const systemPrompt = fs.readFileSync(systemPromptFile, "utf8");
const cwd = process.cwd();
const agentDir = process.env.PI_CODING_AGENT_DIR || "/tmp/harness-home/.pi/agent";
const appendSystemPrompts = loadAppendSystemPrompts(cwd);
let session;
let limitReached = false;
let shuttingDown = false;

async function shutdown(exitCode) {
  if (shuttingDown) return;
  shuttingDown = true;
  try {
    await session?.abort();
  } catch {
    // Best-effort shutdown after an external timeout.
  }
  try {
    session?.dispose();
  } catch {
    // The process is exiting regardless.
  }
  process.exit(exitCode);
}

process.on("SIGTERM", () => void shutdown(143));
process.on("SIGHUP", () => void shutdown(129));

emit({ type: "runner_start", runtime: "pi-sdk-compact-v1" });

try {
  const settingsManager = SettingsManager.create(cwd, agentDir);
  const modelRuntime = await ModelRuntime.create({
    authPath: `${agentDir}/auth.json`,
    modelsPath: `${agentDir}/models.json`,
  });
  if (process.env.DEEPSEEK_API_KEY) {
    modelRuntime.setRuntimeApiKey("deepseek", process.env.DEEPSEEK_API_KEY);
  }
  const model = modelRuntime.getModel("deepseek", "deepseek-v4-flash");
  if (!model) {
    throw new Error("DeepSeek model deepseek-v4-flash is unavailable");
  }
  const resourceLoader = new DefaultResourceLoader({
    cwd,
    agentDir,
    settingsManager,
    systemPromptOverride: () => systemPrompt,
    appendSystemPromptOverride: () => appendSystemPrompts,
  });
  await resourceLoader.reload();
  ({ session } = await createAgentSession({
    cwd,
    agentDir,
    model,
    thinkingLevel: "high",
    modelRuntime,
    resourceLoader,
    sessionManager: SessionManager.inMemory(cwd),
    settingsManager,
  }));

  const toolCallIds = new Set();
  let anonymousToolCalls = 0;
  session.subscribe((event) => {
    if (event.type === "message_update" || event.type === "tool_execution_update") {
      return;
    }
    const compacted = compactEvent(event);
    if (compacted) emit(compacted);
    if (event.type !== "tool_execution_start" || limitReached) {
      return;
    }
    const callId = String(event.toolCallId ?? "");
    if (callId) {
      toolCallIds.add(callId);
    } else {
      anonymousToolCalls += 1;
    }
    const count = toolCallIds.size + anonymousToolCalls;
    if (count >= maxSteps) {
      limitReached = true;
      emit({ type: "harness_limit", kind: "max_steps", count });
      queueMicrotask(() => void session.abort());
    }
  });

  await session.prompt(prompt);
  const lastMessage = session.messages.at(-1);
  if (
    !limitReached &&
    lastMessage?.role === "assistant" &&
    (lastMessage.stopReason === "error" || lastMessage.stopReason === "aborted")
  ) {
    throw new Error(lastMessage.errorMessage || `Request ${lastMessage.stopReason}`);
  }
} catch (error) {
  console.error(error instanceof Error ? error.stack || error.message : String(error));
  process.exitCode = 1;
} finally {
  try {
    session?.dispose();
  } catch {
    // Nothing remains to recover during process teardown.
  }
}
