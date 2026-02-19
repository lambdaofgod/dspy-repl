"use strict";

const readline = require("node:readline");
const vm = require("node:vm");

const COMMAND_END_MARKER = "__DSPY_CMD_END__";
const TOOL_CALL_PREFIX = "__DSPY_TOOL_CALL__";
const SUBMIT_PREFIX = "__DSPY_SUBMIT__";
const ERROR_PREFIX = "__DSPY_ERROR__";
const SUBMIT_SIGNAL = "__DSPY_SUBMIT_SIGNAL__";
const VM_TIMEOUT_MS = 60_000;

const state = {
  toolNames: new Set(),
  submitFields: [],
  nextToolCallId: 1,
  pendingToolCalls: new Map(),
};

const sandbox = {
  console,
  globalThis: null,
};
sandbox.globalThis = sandbox;
const context = vm.createContext(sandbox);

function emit(line) {
  process.stdout.write(String(line) + "\n");
}

function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function ensureObjectWithOutputFields(output) {
  if (state.submitFields.length === 0) {
    if (isPlainObject(output)) {
      return output;
    }
    return { output };
  }

  if (isPlainObject(output)) {
    return output;
  }

  if (state.submitFields.length === 1) {
    return { [state.submitFields[0]]: output };
  }

  throw new Error("submit() expects an object with all output fields.");
}

function installRuntimeFunctions() {
  context.submit = (output) => {
    const payload = ensureObjectWithOutputFields(output);
    emit(`${SUBMIT_PREFIX}${JSON.stringify(payload)}`);
    const err = new Error(SUBMIT_SIGNAL);
    err.name = "SubmitSignal";
    throw err;
  };
  context.SUBMIT = context.submit;

  context.llmQuery = async (prompt) => callTool("llmQuery", [prompt]);
  context.llmQueryBatched = async (prompts) => {
    if (!Array.isArray(prompts)) {
      throw new Error("llmQueryBatched expects an array of prompts.");
    }
    const out = [];
    for (const p of prompts) {
      out.push(await callTool("llmQuery", [p]));
    }
    return out;
  };
}

function installToolAdapters() {
  for (const toolName of state.toolNames) {
    if (toolName === "submit" || toolName === "SUBMIT") {
      continue;
    }
    context[toolName] = async (...args) => callTool(toolName, args);
  }
}

function callTool(name, args) {
  const id = state.nextToolCallId++;
  return new Promise((resolve, reject) => {
    state.pendingToolCalls.set(id, { resolve, reject });
    emit(`${TOOL_CALL_PREFIX}${JSON.stringify({ id, name, args })}`);
  });
}

function rejectAllPending(reason) {
  for (const [id, pending] of state.pendingToolCalls.entries()) {
    state.pendingToolCalls.delete(id);
    pending.reject(new Error(reason));
  }
}

function formatError(err) {
  if (!err) {
    return "Unknown JavaScript execution error";
  }
  const message = err && err.message ? err.message : String(err);
  if (err && err.stack) {
    return `${message}\n${err.stack}`;
  }
  return message;
}

function setVariables(variables) {
  if (!variables || typeof variables !== "object") {
    return;
  }
  for (const [name, value] of Object.entries(variables)) {
    context[name] = value;
  }
}

async function executeCode(code, variables) {
  installRuntimeFunctions();
  installToolAdapters();
  setVariables(variables);

  const wrapped = `(async () => {\n${String(code || "")}\n})()`;
  try {
    await vm.runInContext(wrapped, context, {
      displayErrors: true,
      timeout: VM_TIMEOUT_MS,
    });
  } catch (err) {
    rejectAllPending(formatError(err));
    if (!(err && err.message === SUBMIT_SIGNAL)) {
      emit(`${ERROR_PREFIX}${formatError(err)}`);
    }
  } finally {
    emit(COMMAND_END_MARKER);
  }
}

function handleToolResult(message) {
  const pending = state.pendingToolCalls.get(message.id);
  if (!pending) {
    return;
  }
  state.pendingToolCalls.delete(message.id);
  if (message.ok) {
    pending.resolve(message.result);
    return;
  }
  pending.reject(new Error(message.error || "Tool call failed"));
}

const rl = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

let queue = Promise.resolve();

rl.on("line", (line) => {
  const trimmed = line.trim();
  if (!trimmed) {
    return;
  }

  let message;
  try {
    message = JSON.parse(trimmed);
  } catch (err) {
    emit(`${ERROR_PREFIX}Invalid host message JSON: ${formatError(err)}`);
    emit(COMMAND_END_MARKER);
    return;
  }

  if (message.type === "set_tools") {
    state.toolNames = new Set(Array.isArray(message.tools) ? message.tools : []);
    state.submitFields = Array.isArray(message.submit_fields)
      ? message.submit_fields.map((x) => String(x))
      : [];
    return;
  }

  if (message.type === "tool_result") {
    handleToolResult(message);
    return;
  }

  if (message.type === "exec") {
    queue = queue.then(() => executeCode(message.code, message.variables));
    return;
  }

  emit(`${ERROR_PREFIX}Unknown message type: ${String(message.type)}`);
  emit(COMMAND_END_MARKER);
});
