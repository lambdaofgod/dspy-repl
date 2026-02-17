const COMMAND_END_MARKER = "__DSPY_CMD_END__";
const TOOL_CALL_PREFIX = "__DSPY_TOOL_CALL__";
const SUBMIT_PREFIX = "__DSPY_SUBMIT__";
const ERROR_PREFIX = "__DSPY_ERROR__";
const SUBMIT_SIGNAL = "__DSPY_SUBMIT_SIGNAL__";

const state = {
  toolNames: new Set<string>(),
  submitFields: [] as string[],
  nextToolCallId: 1,
  pendingToolCalls: new Map<
    number,
    { resolve: (v: unknown) => void; reject: (e: Error) => void }
  >(),
};

const encoder = new TextEncoder();

function emit(line: unknown): void {
  const text = String(line).replaceAll("\n", "\\n");
  const bytes = encoder.encode(text + "\n");
  Deno.stdout.writeSync(bytes);
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function ensureObjectWithOutputFields(
  output: unknown,
): Record<string, unknown> {
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

// deno-lint-ignore no-explicit-any
const globals: Record<string, any> = {};

function installRuntimeFunctions(): void {
  globals.submit = (output: unknown) => {
    const payload = ensureObjectWithOutputFields(output);
    emit(`${SUBMIT_PREFIX}${JSON.stringify(payload)}`);
    const err = new Error(SUBMIT_SIGNAL);
    err.name = "SubmitSignal";
    throw err;
  };
  globals.SUBMIT = globals.submit;

  globals.llmQuery = async (prompt: string) => callTool("llmQuery", [prompt]);
  globals.llmQueryBatched = async (prompts: string[]) => {
    if (!Array.isArray(prompts)) {
      throw new Error("llmQueryBatched expects an array of prompts.");
    }
    const out: unknown[] = [];
    for (const p of prompts) {
      out.push(await callTool("llmQuery", [p]));
    }
    return out;
  };

  // Install onto globalThis
  for (const [key, value] of Object.entries(globals)) {
    (globalThis as Record<string, unknown>)[key] = value;
  }
}

function installToolAdapters(): void {
  for (const toolName of state.toolNames) {
    if (toolName === "submit" || toolName === "SUBMIT") {
      continue;
    }
    const fn = async (...args: unknown[]) => callTool(toolName, args);
    globals[toolName] = fn;
    (globalThis as Record<string, unknown>)[toolName] = fn;
  }
}

function callTool(name: string, args: unknown[]): Promise<unknown> {
  const id = state.nextToolCallId++;
  return new Promise((resolve, reject) => {
    state.pendingToolCalls.set(id, { resolve, reject });
    emit(`${TOOL_CALL_PREFIX}${JSON.stringify({ id, name, args })}`);
  });
}

function formatError(err: unknown): string {
  if (!err) {
    return "Unknown TypeScript execution error";
  }
  if (err instanceof Error) {
    return err.message || String(err);
  }
  return String(err);
}

function setVariables(variables: Record<string, unknown> | null): void {
  if (!variables || typeof variables !== "object") {
    return;
  }
  for (const [name, value] of Object.entries(variables)) {
    globals[name] = value;
    (globalThis as Record<string, unknown>)[name] = value;
  }
}

async function executeCode(
  code: string,
  variables: Record<string, unknown> | null,
): Promise<void> {
  installRuntimeFunctions();
  installToolAdapters();
  setVariables(variables);

  const wrapped = `(async () => {\n${String(code || "")}\n})()`;
  try {
    // Indirect eval to execute in global scope
    const indirectEval = eval;
    const result = indirectEval(wrapped);
    await result;
  } catch (err: unknown) {
    if (!(err instanceof Error && err.message === SUBMIT_SIGNAL)) {
      emit(`${ERROR_PREFIX}${formatError(err)}`);
    }
  } finally {
    emit(COMMAND_END_MARKER);
  }
}

interface ToolResultMessage {
  id: number;
  ok: boolean;
  result?: unknown;
  error?: string;
}

function handleToolResult(message: ToolResultMessage): void {
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

// Stdin line reader using Deno.stdin.readable
async function* readLines(): AsyncGenerator<string> {
  const reader = Deno.stdin.readable.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const newlineIdx = buffer.indexOf("\n");
      if (newlineIdx !== -1) {
        const line = buffer.slice(0, newlineIdx);
        buffer = buffer.slice(newlineIdx + 1);
        yield line;
        continue;
      }
      const { value, done } = await reader.read();
      if (done) {
        if (buffer.length > 0) {
          yield buffer;
        }
        return;
      }
      buffer += decoder.decode(value, { stream: true });
    }
  } finally {
    reader.releaseLock();
  }
}

// Main message loop
let queue: Promise<void> = Promise.resolve();

for await (const line of readLines()) {
  const trimmed = line.trim();
  if (!trimmed) {
    continue;
  }

  let message;
  try {
    message = JSON.parse(trimmed);
  } catch (err: unknown) {
    emit(`${ERROR_PREFIX}Invalid host message JSON: ${formatError(err)}`);
    emit(COMMAND_END_MARKER);
    continue;
  }

  if (message.type === "set_tools") {
    state.toolNames = new Set(
      Array.isArray(message.tools) ? message.tools : [],
    );
    state.submitFields = Array.isArray(message.submit_fields)
      ? message.submit_fields.map((x: unknown) => String(x))
      : [];
    continue;
  }

  if (message.type === "tool_result") {
    handleToolResult(message);
    continue;
  }

  if (message.type === "exec") {
    queue = queue.then(() => executeCode(message.code, message.variables));
    continue;
  }

  emit(`${ERROR_PREFIX}Unknown message type: ${String(message.type)}`);
  emit(COMMAND_END_MARKER);
}
