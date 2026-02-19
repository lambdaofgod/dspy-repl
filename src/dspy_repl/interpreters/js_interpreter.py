"""
Local interpreter for JavaScript code execution using Node.js.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import logging
import math
import os
import queue
import re
import select
import signal
import subprocess
import threading
import time
from typing import Any, Callable

from dspy_repl.core.code_interpreter import CodeInterpreterError, FinalOutput

__all__ = ["JavaScriptInterpreter", "FinalOutput", "CodeInterpreterError"]

logger = logging.getLogger(__name__)

_COMMAND_END_MARKER = "__DSPY_CMD_END__"
_TOOL_CALL_PREFIX = "__DSPY_TOOL_CALL__"
_SUBMIT_PREFIX = "__DSPY_SUBMIT__"
_ERROR_PREFIX = "__DSPY_ERROR__"
_SUBMIT_CALL_PATTERN = re.compile(r"\bSUBMIT\b")
_TOOL_TIMEOUT_PREFIX = "[TIMEOUT]"
_MAX_TOOL_RESPONSE_CHARS = 5000
_VAR_NAME_PATTERN = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
_JS_RESERVED = frozenset(
    {
        "await",
        "break",
        "case",
        "catch",
        "class",
        "const",
        "continue",
        "debugger",
        "default",
        "delete",
        "do",
        "else",
        "enum",
        "export",
        "extends",
        "false",
        "finally",
        "for",
        "function",
        "if",
        "implements",
        "import",
        "in",
        "instanceof",
        "interface",
        "let",
        "new",
        "null",
        "package",
        "private",
        "protected",
        "public",
        "return",
        "static",
        "super",
        "switch",
        "this",
        "throw",
        "true",
        "try",
        "typeof",
        "var",
        "void",
        "while",
        "with",
        "yield",
    }
)


class JavaScriptInterpreter:
    """Stateful JavaScript interpreter backed by a Node.js subprocess."""

    def __init__(
        self,
        node_command: list[str] | None = None,
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict] | None = None,
        execute_timeout_seconds: float = 90.0,
        tool_call_timeout_s: float = 25.0,
        log_slow_tool_calls_s: float = 5.0,
    ) -> None:
        if isinstance(node_command, dict):
            raise TypeError("node_command must be a list of strings, not a dict")

        self.tools = dict(tools) if tools else {}
        self.output_fields = output_fields
        self.execute_timeout_seconds = float(execute_timeout_seconds)
        self.tool_call_timeout_s = float(tool_call_timeout_s)
        self.log_slow_tool_calls_s = float(log_slow_tool_calls_s)
        self._owner_thread: int | None = None
        self._tools_registered = False
        self._registered_tool_names: tuple[str, ...] = tuple()
        self._registered_submit_fields: tuple[str, ...] = tuple()

        prelude_path = self._get_prelude_path()
        self.node_command = (
            list(node_command)
            if node_command
            else [
                "node",
                "--no-warnings",
                prelude_path,
            ]
        )

        self._process: subprocess.Popen[str] | None = None
        self._stderr_lock = threading.Lock()
        self._stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None

    def _get_prelude_path(self) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "js_prelude.js")

    def _check_thread_ownership(self) -> None:
        current_thread = threading.current_thread().ident
        if self._owner_thread is None:
            self._owner_thread = current_thread
        elif self._owner_thread != current_thread:
            raise RuntimeError(
                "JavaScriptInterpreter is not thread-safe and cannot be shared across threads. "
                "Create a separate interpreter instance for each thread."
            )

    def _ensure_process(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return

        try:
            self._process = subprocess.Popen(
                self.node_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="UTF-8",
                env=os.environ.copy(),
                bufsize=1,
                start_new_session=True,
            )
        except FileNotFoundError as e:
            install_instructions = (
                "Node.js executable not found. Please install Node.js to proceed.\n"
                "Installation instructions:\n"
                "> brew install node  (macOS)\n"
                "> apt install nodejs npm  (Ubuntu/Debian)\n"
                "> pacman -S nodejs npm  (Arch)\n"
                "For more info: https://nodejs.org/"
            )
            raise CodeInterpreterError(install_instructions) from e

        self._start_stderr_reader()
        self._drain_stderr()
        self._tools_registered = False
        self._registered_tool_names = tuple()
        self._registered_submit_fields = tuple()
        self._health_check()

    def _start_stderr_reader(self) -> None:
        if self._stderr_thread and self._stderr_thread.is_alive():
            return

        def _reader() -> None:
            assert self._process is not None and self._process.stderr is not None
            while True:
                line = self._process.stderr.readline()
                if line == "":
                    return
                text = line.rstrip("\n")
                if text:
                    with self._stderr_lock:
                        self._stderr_lines.append(text)

        self._stderr_thread = threading.Thread(target=_reader, name="javascript-interpreter-stderr", daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self) -> str:
        with self._stderr_lock:
            if not self._stderr_lines:
                return ""
            text = "\n".join(self._stderr_lines)
            self._stderr_lines = []
            return text

    def _write_line(self, line: str) -> None:
        assert self._process is not None and self._process.stdin is not None
        try:
            self._process.stdin.write(line + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            self._terminate_process()
            raise CodeInterpreterError(f"Node.js process is unavailable while sending input: {e}") from e

    def _terminate_process(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is not None:
            return
        try:
            os.killpg(self._process.pid, signal.SIGTERM)
            self._process.wait(timeout=2)
        except Exception:
            try:
                os.killpg(self._process.pid, signal.SIGKILL)
            except Exception:
                self._process.kill()
            with contextlib.suppress(Exception):
                self._process.wait(timeout=2)

    def _call_tool_with_timeout(self, fn: Callable[..., Any], args: list[Any]) -> Any:
        result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)
        cancel_event = threading.Event()

        def worker() -> None:
            try:
                sig = inspect.signature(fn)
                params = sig.parameters
                if "cancel_event" in params:
                    result = fn(*args, cancel_event=cancel_event)
                elif "_cancel_event" in params:
                    result = fn(*args, _cancel_event=cancel_event)
                else:
                    result = fn(*args)
                result_queue.put((True, result))
            except Exception as exc:
                result_queue.put((False, exc))

        thread = threading.Thread(target=worker, name="javascript-tool-call", daemon=True)
        thread.start()
        thread.join(self.tool_call_timeout_s)
        if thread.is_alive():
            cancel_event.set()
            raise TimeoutError(f"Tool call timed out after {self.tool_call_timeout_s:.1f}s.")
        ok, payload = result_queue.get_nowait()
        if ok:
            return payload
        raise payload

    def _send_json(self, payload: dict[str, Any]) -> None:
        self._write_line(json.dumps(payload, ensure_ascii=False))

    def _readline_with_timeout(self, timeout_seconds: float, context: str) -> str:
        assert self._process is not None and self._process.stdout is not None
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CodeInterpreterError(
                    f"Timed out after {timeout_seconds:.1f}s while waiting for Node.js output {context}."
                )
            ready, _, _ = select.select([self._process.stdout], [], [], min(remaining, 0.25))
            if ready:
                line = self._process.stdout.readline()
                if line == "":
                    code = self._process.poll()
                    stderr_text = self._drain_stderr()
                    raise CodeInterpreterError(f"Node.js process exited unexpectedly (code {code}). {stderr_text}")
                return line.rstrip("\n")

    def _looks_like_syntax_error(self, text: str) -> bool:
        lowered = text.lower()
        return "syntaxerror" in lowered or "unexpected token" in lowered or "unterminated" in lowered

    def _classify_and_raise(self, error_text: str) -> None:
        if self._looks_like_syntax_error(error_text):
            raise SyntaxError(f"Invalid JavaScript syntax: {error_text}")
        raise CodeInterpreterError(error_text)

    def _normalize_submit_calls(self, code: str) -> str:
        return _SUBMIT_CALL_PATTERN.sub("submit", code)

    def _validate_variable_name(self, name: str) -> None:
        if not _VAR_NAME_PATTERN.match(name):
            raise CodeInterpreterError(f"Invalid JavaScript variable name: '{name}'")
        if name in _JS_RESERVED:
            raise CodeInterpreterError(f"Variable name '{name}' is a JavaScript reserved word")

    def _json_safe(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return value
        if isinstance(value, list):
            return [self._json_safe(v) for v in value]
        if isinstance(value, tuple):
            return [self._json_safe(v) for v in value]
        if isinstance(value, set):
            try:
                ordered = sorted(value)
            except TypeError:
                ordered = list(value)
            return [self._json_safe(v) for v in ordered]
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        return str(value)

    def _prepare_variables(self, variables: dict[str, Any]) -> dict[str, Any]:
        prepared: dict[str, Any] = {}
        for key, value in variables.items():
            self._validate_variable_name(key)
            prepared[key] = self._json_safe(value)
        return prepared

    def _register_host_state(self) -> None:
        tool_names = tuple(sorted(self.tools.keys()))
        submit_fields = tuple(str(field["name"]) for field in (self.output_fields or []) if "name" in field)
        if (
            self._tools_registered
            and tool_names == self._registered_tool_names
            and submit_fields == self._registered_submit_fields
        ):
            return
        self._send_json(
            {
                "type": "set_tools",
                "tools": list(tool_names),
                "submit_fields": list(submit_fields),
            }
        )
        self._registered_tool_names = tool_names
        self._registered_submit_fields = submit_fields
        self._tools_registered = True

    def _handle_tool_call(self, payload: str) -> None:
        try:
            req = json.loads(payload)
            call_id = req.get("id")
            tool_name = req.get("name")
            args = req.get("args", [])
            if not isinstance(args, list):
                args = [args]
        except json.JSONDecodeError as e:
            raise CodeInterpreterError(f"Invalid tool call payload from JavaScript: {payload}") from e

        if tool_name not in self.tools:
            logger.warning("JavaScriptInterpreter tool call failed: unknown tool '%s'", tool_name)
            self._send_json({"type": "tool_result", "id": call_id, "ok": False, "error": f"Unknown tool: {tool_name}"})
            return

        arg_chars = sum(len(str(arg)) for arg in args)
        logger.debug(
            "JavaScriptInterpreter tool call start name=%s args=%d chars=%d timeout=%.1fs",
            tool_name,
            len(args),
            arg_chars,
            self.tool_call_timeout_s,
        )
        start = time.monotonic()
        response_text = ""
        ok = True
        result_payload: Any = ""
        try:
            result = self._call_tool_with_timeout(self.tools[tool_name], args)
        except TimeoutError:
            duration_s = time.monotonic() - start
            ok = False
            result_payload = f"{_TOOL_TIMEOUT_PREFIX} Tool '{tool_name}' exceeded {self.tool_call_timeout_s:.1f}s"
            response_text = str(result_payload)
            logger.warning(
                "JavaScriptInterpreter tool call timeout name=%s elapsed=%.3fs timeout=%.1fs",
                tool_name,
                duration_s,
                self.tool_call_timeout_s,
            )
        except Exception as e:
            ok = False
            result_payload = f"[ERROR] {e}"
            response_text = str(result_payload)
            logger.warning("JavaScriptInterpreter tool call error name=%s err=%s", tool_name, e)
        else:
            result_payload = self._json_safe(result)
            response_text = str(result_payload if result_payload is not None else "")
            duration_s = time.monotonic() - start
            response_len = len(response_text)
            if duration_s >= self.log_slow_tool_calls_s:
                logger.warning(
                    "JavaScriptInterpreter slow tool call name=%s elapsed=%.3fs threshold=%.1fs response_chars=%d",
                    tool_name,
                    duration_s,
                    self.log_slow_tool_calls_s,
                    response_len,
                )
            else:
                logger.debug(
                    "JavaScriptInterpreter tool call end name=%s elapsed=%.3fs response_chars=%d",
                    tool_name,
                    duration_s,
                    response_len,
                )

        if isinstance(result_payload, str) and len(result_payload) > _MAX_TOOL_RESPONSE_CHARS:
            result_payload = result_payload[:_MAX_TOOL_RESPONSE_CHARS] + "\n... (truncated tool response) ..."

        self._send_json(
            {
                "type": "tool_result",
                "id": call_id,
                "ok": ok,
                "result": result_payload,
                "error": None if ok else result_payload,
            }
        )

    def _read_until_marker(self, *, timeout_seconds: float) -> tuple[list[str], dict[str, Any] | None, list[str]]:
        lines: list[str] = []
        final_output: dict[str, Any] | None = None
        errors: list[str] = []
        started = time.monotonic()
        while True:
            elapsed = time.monotonic() - started
            remaining = timeout_seconds - elapsed
            if remaining <= 0:
                raise CodeInterpreterError(f"JavaScript execution timed out after {timeout_seconds:.1f}s.")
            text = self._readline_with_timeout(remaining, "during execute")
            if text == _COMMAND_END_MARKER:
                return lines, final_output, errors
            if text.startswith(_TOOL_CALL_PREFIX):
                self._handle_tool_call(text[len(_TOOL_CALL_PREFIX) :])
                continue
            if text.startswith(_SUBMIT_PREFIX):
                payload = text[len(_SUBMIT_PREFIX) :]
                try:
                    parsed = json.loads(payload)
                except json.JSONDecodeError as e:
                    raise CodeInterpreterError(f"Invalid submit payload from JavaScript: {payload}") from e
                if not isinstance(parsed, dict):
                    raise CodeInterpreterError("submit payload must be a JSON object")
                final_output = parsed
                continue
            if text.startswith(_ERROR_PREFIX):
                errors.append(text[len(_ERROR_PREFIX) :])
                continue
            lines.append(text)

    def _health_check(self) -> None:
        output = self.execute("console.log(1 + 1);")
        output_text = str(output or "")
        if "2" not in output_text:
            raise CodeInterpreterError(f"Unexpected JavaScript health check response: {output_text!r}")

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        self._check_thread_ownership()
        self._ensure_process()
        self._drain_stderr()
        self._register_host_state()

        payload = {
            "type": "exec",
            "code": self._normalize_submit_calls(code),
            "variables": self._prepare_variables(variables or {}),
        }
        self._send_json(payload)

        output_lines, final_output, errors = self._read_until_marker(timeout_seconds=self.execute_timeout_seconds)
        stderr_text = self._drain_stderr()

        if errors:
            self._classify_and_raise("\n".join(errors))
        if stderr_text.strip():
            self._classify_and_raise(stderr_text)
        if final_output is not None:
            return FinalOutput(final_output)
        output = "\n".join(line for line in output_lines if line.strip())
        return output or None

    def start(self) -> None:
        self._ensure_process()

    def shutdown(self) -> None:
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                self._terminate_process()
        self._process = None
        self._owner_thread = None
        self._stderr_thread = None
        self._drain_stderr()
        self._tools_registered = False
        self._registered_tool_names = tuple()
        self._registered_submit_fields = tuple()

    def __enter__(self) -> JavaScriptInterpreter:
        return self

    def __exit__(self, *_: Any) -> None:
        self.shutdown()

    def __call__(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        return self.execute(code, variables)
