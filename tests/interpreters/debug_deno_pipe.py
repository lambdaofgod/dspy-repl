"""Minimal repro of Python <-> Deno subprocess communication."""
import json
import os
import select
import subprocess
import sys
import threading
import time

prelude = os.path.join(
    os.path.dirname(__file__),
    "../../src/dspy_repl/interpreters/deno_prelude.ts",
)

proc = subprocess.Popen(
    ["deno", "run", "--allow-all", prelude],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    encoding="UTF-8",
    bufsize=1,
)

# Background stderr reader
def stderr_reader():
    while True:
        line = proc.stderr.readline()
        if line == "":
            return
        print(f"  [STDERR] {line.rstrip()}", file=sys.stderr)

t = threading.Thread(target=stderr_reader, daemon=True)
t.start()
time.sleep(0.5)  # let Deno start up

def send(msg):
    line = json.dumps(msg, ensure_ascii=False)
    print(f"  -> SEND: {line}", file=sys.stderr)
    proc.stdin.write(line + "\n")
    proc.stdin.flush()

def readline(timeout=5.0):
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print("  !! TIMEOUT", file=sys.stderr)
            return None
        ready, _, _ = select.select([proc.stdout], [], [], min(remaining, 0.25))
        if ready:
            line = proc.stdout.readline().rstrip("\n")
            print(f"  <- RECV: {line!r}", file=sys.stderr)
            return line

def read_until_marker(timeout=5.0):
    lines = []
    started = time.monotonic()
    while True:
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            print("  !! TIMEOUT waiting for marker", file=sys.stderr)
            return lines
        line = readline(remaining)
        if line is None:
            return lines
        if line == "__DSPY_CMD_END__":
            return lines
        lines.append(line)

# Test 1: basic execution
print("\n=== Test 1: basic exec ===", file=sys.stderr)
send({"type": "set_tools", "tools": [], "submit_fields": []})
send({"type": "exec", "code": "console.log(1 + 1);", "variables": {}})
result = read_until_marker()
print(f"  Result: {result}", file=sys.stderr)
assert "2" in result, f"Expected '2' in {result}"

# Test 2: second exec
print("\n=== Test 2: second exec ===", file=sys.stderr)
send({"type": "exec", "code": "console.log(3 + 4);", "variables": {}})
result = read_until_marker()
print(f"  Result: {result}", file=sys.stderr)
assert "7" in result, f"Expected '7' in {result}"

# Test 3: variable injection
print("\n=== Test 3: variable injection ===", file=sys.stderr)
send({"type": "exec", "code": "console.log(count + 3);", "variables": {"count": 4}})
result = read_until_marker()
print(f"  Result: {result}", file=sys.stderr)
assert "7" in result, f"Expected '7' in {result}"

# Test 4: syntax error
print("\n=== Test 4: syntax error ===", file=sys.stderr)
send({"type": "exec", "code": "if (", "variables": {}})
result = read_until_marker()
print(f"  Result: {result}", file=sys.stderr)
assert any("__DSPY_ERROR__" in l or "unexpected" in l.lower() for l in result), f"Expected error in {result}"

proc.terminate()
proc.wait()
print("\nAll tests passed!", file=sys.stderr)
