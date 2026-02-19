"""
Local interpreter for TypeScript code execution using Deno.
"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Callable

from pydantic import BaseModel

from dspy_repl.core.code_interpreter import CodeInterpreterError, FinalOutput

__all__ = ["DenoInterpreter", "DenoPermissions", "FinalOutput", "CodeInterpreterError"]

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


class DenoPermissions(BaseModel):
    """Fine-grained Deno subprocess permissions.

    By default all permissions are denied. The prelude only needs stdio
    (always available), so deny-all is a safe baseline.

    Examples::

        # No permissions (safe default)
        DenoPermissions()

        # Unrestricted network, read/write in one directory
        DenoPermissions(allow_net=True, allow_read=["/tmp/sandbox"], allow_write=["/tmp/sandbox"])

        # Network limited to one host
        DenoPermissions(allow_net=["api.example.com:443"])
    """

    allow_net: bool | list[str] = True
    allow_read: list[str] = []
    allow_write: list[str] = []

    @staticmethod
    def _validate_paths(paths: list[str], label: str) -> None:
        for p in paths:
            resolved = os.path.abspath(p)
            if not os.path.exists(resolved):
                raise ValueError(f"{label} path does not exist: {p!r} (resolved to {resolved!r})")

    @staticmethod
    def _validate_hosts(hosts: list[str]) -> None:
        for h in hosts:
            parts = h.rsplit(":", 1)
            hostname = parts[0]
            if not hostname or hostname != hostname.strip():
                raise ValueError(f"Invalid network host: {h!r}")
            if len(parts) == 2:
                try:
                    port = int(parts[1])
                except ValueError:
                    raise ValueError(f"Invalid port in network host: {h!r}")
                if not (1 <= port <= 65535):
                    raise ValueError(f"Port out of range in network host: {h!r}")

    def model_post_init(self, __context: Any) -> None:
        self._validate_paths(self.allow_read, "allow_read")
        self._validate_paths(self.allow_write, "allow_write")
        if isinstance(self.allow_net, list):
            self._validate_hosts(self.allow_net)

    def to_args(self) -> list[str]:
        args: list[str] = []
        if self.allow_net is True:
            args.append("--allow-net")
        elif isinstance(self.allow_net, list) and self.allow_net:
            args.append(f"--allow-net={','.join(self.allow_net)}")
        if self.allow_read:
            args.append(f"--allow-read={','.join(self.allow_read)}")
        if self.allow_write:
            args.append(f"--allow-write={','.join(self.allow_write)}")
        return args


class DenoInterpreter:
    """Stateful TypeScript interpreter backed by a Deno subprocess."""

    def __init__(
        self,
        deno_command: list[str] | None = None,
        deno_permissions: DenoPermissions = DenoPermissions(),
        tools: dict[str, Callable[..., Any]] | None = None,
        output_fields: list[dict] | None = None,
        execute_timeout_seconds: float = 90.0,
        tool_call_timeout_s: float = 25.0,
        log_slow_tool_calls_s: float = 5.0,
    ) -> None:
        if isinstance(deno_command, dict):
            raise TypeError("deno_command must be a list of strings, not a dict")

        self.deno_permissions = deno_permissions
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
        if deno_command:
            self.deno_command = list(deno_command)
        else:
            self.deno_command = ["deno", "run"] + deno_permissions.to_args() + [prelude_path]

        self._process: subprocess.Popen[str] | None = None
        self._stdout_queue: queue.Queue[str | None] = queue.Queue()
        self._stderr_lock = threading.Lock()
        self._stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None
        self._stdout_thread: threading.Thread | None = None

    def _get_prelude_path(self) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "deno_prelude.ts")

    def _check_thread_ownership(self) -> None:
        current_thread = threading.current_thread().ident
        if self._owner_thread is None:
            self._owner_thread = current_thread
        elif self._owner_thread != current_thread:
            raise RuntimeError(
                "DenoInterpreter is not thread-safe and cannot be shared across threads. "
                "Create a separate interpreter instance for each thread."
            )

    def _ensure_process(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return

        try:
            self._process = subprocess.Popen(
                self.deno_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="UTF-8",
                env=os.environ.copy(),
                bufsize=1,
            )
        except FileNotFoundError as e:
            install_instructions = (
                "Deno executable not found. Please install Deno to proceed.\n"
                "Installation instructions:\n"
                "> brew install deno  (macOS)\n"
                "> curl -fsSL https://deno.land/install.sh | sh  (Linux/macOS)\n"
                "> irm https://deno.land/install.ps1 | iex  (Windows)\n"
                "For more info: https://deno.land/"
            )
            raise CodeInterpreterError(install_instructions) from e

        self._start_io_readers()
        self._drain_stderr()
        self._tools_registered = False
        self._registered_tool_names = tuple()
        self._registered_submit_fields = tuple()
        self._health_check()

    def _start_io_readers(self) -> None:
        if not (self._stdout_thread and self._stdout_thread.is_alive()):
            self._stdout_queue = queue.Queue()

            def _stdout_reader() -> None:
                assert self._process is not None and self._process.stdout is not None
                while True:
                    line = self._process.stdout.readline()
                    if line == "":
                        self._stdout_queue.put(None)
                        return
                    self._stdout_queue.put(line.rstrip("\n"))

            self._stdout_thread = threading.Thread(target=_stdout_reader, name="deno-interpreter-stdout", daemon=True)
            self._stdout_thread.start()

        if not (self._stderr_thread and self._stderr_thread.is_alive()):

            def _stderr_reader() -> None:
                assert self._process is not None and self._process.stderr is not None
                while True:
                    line = self._process.stderr.readline()
                    if line == "":
                        return
                    text = line.rstrip("\n")
                    if text:
                        with self._stderr_lock:
                            self._stderr_lines.append(text)

            self._stderr_thread = threading.Thread(target=_stderr_reader, name="deno-interpreter-stderr", daemon=True)
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
        self._process.stdin.write(line + "\n")
        self._process.stdin.flush()

    def _send_json(self, payload: dict[str, Any]) -> None:
        self._write_line(json.dumps(payload, ensure_ascii=False))

    def _readline_with_timeout(self, timeout_seconds: float, context: str) -> str:
        try:
            line = self._stdout_queue.get(timeout=timeout_seconds)
        except queue.Empty:
            raise CodeInterpreterError(
                f"Timed out after {timeout_seconds:.1f}s while waiting for Deno output {context}."
            )
        if line is None:
            code = self._process.poll() if self._process else None
            stderr_text = self._drain_stderr()
            raise CodeInterpreterError(f"Deno process exited unexpectedly (code {code}). {stderr_text}")
        return line

    def _looks_like_syntax_error(self, text: str) -> bool:
        lowered = text.lower()
        return "syntaxerror" in lowered or "unexpected token" in lowered or "unterminated" in lowered

    def _classify_and_raise(self, error_text: str) -> None:
        if self._looks_like_syntax_error(error_text):
            raise SyntaxError(f"Invalid TypeScript syntax: {error_text}")
        raise CodeInterpreterError(error_text)

    def _normalize_submit_calls(self, code: str) -> str:
        return _SUBMIT_CALL_PATTERN.sub("submit", code)

    def _validate_variable_name(self, name: str) -> None:
        if not _VAR_NAME_PATTERN.match(name):
            raise CodeInterpreterError(f"Invalid TypeScript variable name: '{name}'")
        if name in _JS_RESERVED:
            raise CodeInterpreterError(f"Variable name '{name}' is a TypeScript reserved word")

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
            raise CodeInterpreterError(f"Invalid tool call payload from Deno: {payload}") from e

        if tool_name not in self.tools:
            logger.warning("DenoInterpreter tool call failed: unknown tool '%s'", tool_name)
            self._send_json({"type": "tool_result", "id": call_id, "ok": False, "error": f"Unknown tool: {tool_name}"})
            return

        arg_chars = sum(len(str(arg)) for arg in args)
        logger.debug(
            "DenoInterpreter tool call start name=%s args=%d chars=%d timeout=%.1fs",
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
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="deno-tool-call")
            try:
                future = executor.submit(self.tools[tool_name], *args)
                try:
                    result = future.result(timeout=self.tool_call_timeout_s)
                except FutureTimeoutError:
                    future.cancel()
                    duration_s = time.monotonic() - start
                    ok = False
                    result_payload = (
                        f"{_TOOL_TIMEOUT_PREFIX} Tool '{tool_name}' exceeded {self.tool_call_timeout_s:.1f}s"
                    )
                    response_text = str(result_payload)
                    logger.warning(
                        "DenoInterpreter tool call timeout name=%s elapsed=%.3fs timeout=%.1fs",
                        tool_name,
                        duration_s,
                        self.tool_call_timeout_s,
                    )
                else:
                    result_payload = self._json_safe(result)
                    response_text = str(result_payload if result_payload is not None else "")
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        except Exception as e:
            ok = False
            result_payload = f"[ERROR] {e}"
            response_text = str(result_payload)
            logger.warning("DenoInterpreter tool call error name=%s err=%s", tool_name, e)
        else:
            duration_s = time.monotonic() - start
            response_len = len(response_text)
            if duration_s >= self.log_slow_tool_calls_s:
                logger.warning(
                    "DenoInterpreter slow tool call name=%s elapsed=%.3fs threshold=%.1fs response_chars=%d",
                    tool_name,
                    duration_s,
                    self.log_slow_tool_calls_s,
                    response_len,
                )
            else:
                logger.debug(
                    "DenoInterpreter tool call end name=%s elapsed=%.3fs response_chars=%d",
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
                raise CodeInterpreterError(f"Deno execution timed out after {timeout_seconds:.1f}s.")
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
                    raise CodeInterpreterError(f"Invalid submit payload from Deno: {payload}") from e
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
            raise CodeInterpreterError(f"Unexpected Deno health check response: {output_text!r}")

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
                self._process.kill()
                self._process.wait()
        self._process = None
        self._owner_thread = None
        self._stdout_thread = None
        self._stderr_thread = None
        self._drain_stderr()
        self._tools_registered = False
        self._registered_tool_names = tuple()
        self._registered_submit_fields = tuple()

    def __enter__(self) -> DenoInterpreter:
        return self

    def __exit__(self, *_: Any) -> None:
        self.shutdown()

    def __call__(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        return self.execute(code, variables)
