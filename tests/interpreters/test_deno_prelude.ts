/**
 * Interactive test for deno_prelude.ts -- simulates the Python interpreter's
 * send-then-read pattern by spawning the prelude as a subprocess.
 */

const PRELUDE = new URL(
  "../../src/dspy_repl/interpreters/deno_prelude.ts",
  import.meta.url,
).pathname;

const MARKER = "__DSPY_CMD_END__";
const ERROR_PREFIX = "__DSPY_ERROR__";
const SUBMIT_PREFIX = "__DSPY_SUBMIT__";
const TOOL_CALL_PREFIX = "__DSPY_TOOL_CALL__";

interface ExecResult {
  lines: string[];
  errors: string[];
  submit: Record<string, unknown> | null;
}

class PreludeProcess {
  #proc: Deno.ChildProcess;
  #writer: WritableStreamDefaultWriter<Uint8Array>;
  #reader: ReadableStreamDefaultReader<Uint8Array>;
  #buffer = "";
  #encoder = new TextEncoder();
  #decoder = new TextDecoder();

  constructor() {
    const cmd = new Deno.Command("deno", {
      args: ["run", "--allow-all", PRELUDE],
      stdin: "piped",
      stdout: "piped",
      stderr: "piped",
    });
    this.#proc = cmd.spawn();
    this.#writer = this.#proc.stdin.getWriter();
    this.#reader = this.#proc.stdout.getReader();
  }

  async sendLine(json: Record<string, unknown>): Promise<void> {
    const bytes = this.#encoder.encode(JSON.stringify(json) + "\n");
    await this.#writer.write(bytes);
  }

  async readLine(timeoutMs = 5000): Promise<string> {
    const deadline = Date.now() + timeoutMs;
    while (true) {
      const idx = this.#buffer.indexOf("\n");
      if (idx !== -1) {
        const line = this.#buffer.slice(0, idx);
        this.#buffer = this.#buffer.slice(idx + 1);
        return line;
      }
      if (Date.now() > deadline) {
        throw new Error(`Timeout reading line. Buffer so far: ${JSON.stringify(this.#buffer)}`);
      }
      const { value, done } = await this.#reader.read();
      if (done) throw new Error("Process stdout closed");
      this.#buffer += this.#decoder.decode(value, { stream: true });
    }
  }

  async exec(
    code: string,
    variables: Record<string, unknown> = {},
  ): Promise<ExecResult> {
    await this.sendLine({ type: "exec", code, variables });
    const lines: string[] = [];
    const errors: string[] = [];
    let submit: Record<string, unknown> | null = null;
    while (true) {
      const line = await this.readLine();
      if (line === MARKER) break;
      if (line.startsWith(ERROR_PREFIX)) {
        errors.push(line.slice(ERROR_PREFIX.length));
      } else if (line.startsWith(SUBMIT_PREFIX)) {
        submit = JSON.parse(line.slice(SUBMIT_PREFIX.length));
      } else if (line.startsWith(TOOL_CALL_PREFIX)) {
        // Auto-respond to tool calls with a fixed value
        const req = JSON.parse(line.slice(TOOL_CALL_PREFIX.length));
        await this.sendLine({
          type: "tool_result",
          id: req.id,
          ok: true,
          result: "tool-ok",
        });
      } else {
        lines.push(line);
      }
    }
    return { lines, errors, submit };
  }

  async setTools(
    tools: string[],
    submitFields: string[] = [],
  ): Promise<void> {
    await this.sendLine({
      type: "set_tools",
      tools,
      submit_fields: submitFields,
    });
  }

  async close(): Promise<void> {
    try { this.#writer.close(); } catch { /* ignore */ }
    this.#proc.kill("SIGTERM");
    await this.#proc.status;
  }
}

// -- helpers --
function assert(cond: boolean, msg: string): void {
  if (!cond) throw new Error(`ASSERTION FAILED: ${msg}`);
}
let passed = 0;
let failed = 0;

async function runTest(
  name: string,
  fn: (p: PreludeProcess) => Promise<void>,
): Promise<void> {
  const p = new PreludeProcess();
  try {
    await fn(p);
    console.log(`  PASS  ${name}`);
    passed++;
  } catch (err) {
    console.log(`  FAIL  ${name}: ${(err as Error).message}`);
    failed++;
  } finally {
    await p.close();
  }
}

// -- tests --

console.log("Running deno_prelude tests...\n");

await runTest("basic execution", async (p) => {
  const r = await p.exec("console.log(1 + 1);");
  assert(r.lines.includes("2"), `expected '2' in output, got ${JSON.stringify(r.lines)}`);
  assert(r.errors.length === 0, `unexpected errors: ${r.errors}`);
});

await runTest("second execution on same process", async (p) => {
  const r1 = await p.exec("console.log('first');");
  assert(r1.lines.includes("first"), "first exec failed");
  const r2 = await p.exec("console.log('second');");
  assert(r2.lines.includes("second"), `second exec failed, got ${JSON.stringify(r2.lines)}`);
});

await runTest("variable injection", async (p) => {
  const r = await p.exec("console.log(count + 3);", { count: 4 });
  assert(r.lines.includes("7"), `expected '7', got ${JSON.stringify(r.lines)}`);
});

await runTest("submit protocol", async (p) => {
  await p.setTools([], ["answer"]);
  const r = await p.exec('submit("ok");');
  assert(r.submit !== null, "expected submit output");
  assert(r.submit!.answer === "ok", `expected answer='ok', got ${JSON.stringify(r.submit)}`);
});

await runTest("tool call plumbing", async (p) => {
  await p.setTools(["llmQuery"]);
  const r = await p.exec('console.log(await llmQuery("ping"));');
  assert(r.lines.includes("tool-ok"), `expected 'tool-ok', got ${JSON.stringify(r.lines)}`);
});

await runTest("syntax error", async (p) => {
  const r = await p.exec("if (");
  assert(r.errors.length > 0, "expected errors for syntax error");
  assert(r.errors.some((e) => e.toLowerCase().includes("unexpected")), `expected syntax error, got ${r.errors}`);
});

await runTest("sequential execs with variables", async (p) => {
  const r1 = await p.exec("console.log(1 + 1);");
  assert(r1.lines.includes("2"), "first exec failed");
  const r2 = await p.exec("console.log(x * 2);", { x: 5 });
  assert(r2.lines.includes("10"), `expected '10', got ${JSON.stringify(r2.lines)}`);
});

console.log(`\n${passed} passed, ${failed} failed`);
Deno.exit(failed > 0 ? 1 : 0);
