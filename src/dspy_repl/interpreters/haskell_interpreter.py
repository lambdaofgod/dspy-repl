"""
Local interpreter for Haskell code execution using GHCi.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
import select
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Callable

from dspy_repl.core.code_interpreter import SIMPLE_TYPES, CodeInterpreterError, FinalOutput

__all__ = ["HaskellInterpreter", "FinalOutput", "CodeInterpreterError"]

logger = logging.getLogger(__name__)

_COMMAND_END_MARKER = "__DSPY_CMD_END__"
_TOOL_CALL_PREFIX = "__DSPY_TOOL_CALL__"
_SUBMIT_PREFIX = "__DSPY_SUBMIT__"
_SUBMIT_SIGNAL = "__DSPY_SUBMIT_SIGNAL__"
_SUBMIT_CALL_PATTERN = re.compile(r"\bSUBMIT\b")
_TOOL_TIMEOUT_PREFIX = "[TIMEOUT]"
_MAX_TOOL_RESPONSE_CHARS = 5000
_VAR_NAME_PATTERN = re.compile(r"^[a-z_][a-zA-Z0-9_']*$")
_HASKELL_RESERVED = frozenset(
    {
        "case",
        "class",
        "data",
        "default",
        "deriving",
        "do",
        "else",
        "foreign",
        "if",
        "import",
        "in",
        "infix",
        "infixl",
        "infixr",
        "instance",
        "let",
        "module",
        "newtype",
        "of",
        "then",
        "type",
        "where",
    }
)


class HaskellInterpreter:
    """Local interpreter for Haskell execution using GHCi."""

    def __init__(
        self,
        ghci_command: list[str] | None = None,
        tools: dict[str, Callable[..., str]] | None = None,
        output_fields: list[dict] | None = None,
        execute_timeout_seconds: float = 90.0,
        tool_call_timeout_s: float = 25.0,
        log_slow_tool_calls_s: float = 5.0,
    ) -> None:
        if isinstance(ghci_command, dict):
            raise TypeError("ghci_command must be a list of strings, not a dict")

        self.tools = dict(tools) if tools else {}
        self.output_fields = output_fields
        self.execute_timeout_seconds = float(execute_timeout_seconds)
        self.tool_call_timeout_s = float(tool_call_timeout_s)
        self.log_slow_tool_calls_s = float(log_slow_tool_calls_s)
        self._owner_thread: int | None = None

        self.ghci_command = (
            list(ghci_command)
            if ghci_command
            else [
                "ghci",
                "-v0",
                "-ignore-dot-ghci",
                "-fno-diagnostics-show-caret",
            ]
        )

        self._process: subprocess.Popen[str] | None = None
        self._stderr_lock = threading.Lock()
        self._stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None

    def _check_thread_ownership(self) -> None:
        current_thread = threading.current_thread().ident
        if self._owner_thread is None:
            self._owner_thread = current_thread
        elif self._owner_thread != current_thread:
            raise RuntimeError(
                "HaskellInterpreter is not thread-safe and cannot be shared across threads. "
                "Create a separate interpreter instance for each thread."
            )

    def _get_prelude_path(self) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "haskell_prelude.ghci")

    def _ensure_process(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return

        try:
            self._process = subprocess.Popen(
                self.ghci_command,
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
                "GHCi executable not found. Please install GHC to proceed.\n"
                "Installation instructions:\n"
                "> brew install ghc  (macOS)\n"
                "> apt install ghc  (Ubuntu/Debian)\n"
                "> pacman -S ghc  (Arch)\n"
                "For more info: https://www.haskell.org/ghc/"
            )
            raise CodeInterpreterError(install_instructions) from e

        self._start_stderr_reader()
        self._drain_stderr()
        self._load_prelude()
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

        self._stderr_thread = threading.Thread(target=_reader, name="haskell-interpreter-stderr", daemon=True)
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

    def _readline_with_timeout(self, timeout_seconds: float, context: str) -> str:
        assert self._process is not None and self._process.stdout is not None
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CodeInterpreterError(
                    f"Timed out after {timeout_seconds:.1f}s while waiting for GHCi output {context}."
                )
            ready, _, _ = select.select([self._process.stdout], [], [], min(remaining, 0.25))
            if ready:
                line = self._process.stdout.readline()
                if line == "":
                    code = self._process.poll()
                    stderr_text = self._drain_stderr()
                    raise CodeInterpreterError(f"GHCi exited unexpectedly (code {code}). {stderr_text}")
                return line.rstrip("\n")

    def _read_until_marker(
        self, *, allow_tool_calls: bool, timeout_seconds: float | None = None
    ) -> tuple[list[str], dict[str, Any] | None]:
        assert self._process is not None and self._process.stdout is not None
        lines: list[str] = []
        final_output: dict[str, Any] | None = None
        started = time.monotonic()
        while True:
            if timeout_seconds is not None:
                elapsed = time.monotonic() - started
                remaining = timeout_seconds - elapsed
                if remaining <= 0:
                    raise CodeInterpreterError(f"Haskell execution timed out after {timeout_seconds:.1f}s.")
                text = self._readline_with_timeout(remaining, "during execute")
            else:
                text = self._readline_with_timeout(30.0, "during command")
            if text == _COMMAND_END_MARKER:
                return lines, final_output
            if allow_tool_calls and text.startswith(_TOOL_CALL_PREFIX):
                self._handle_tool_call(text[len(_TOOL_CALL_PREFIX) :])
                continue
            if text.startswith(_SUBMIT_PREFIX):
                payload = text[len(_SUBMIT_PREFIX) :]
                try:
                    parsed = json.loads(payload)
                except json.JSONDecodeError as e:
                    raise CodeInterpreterError(f"Invalid SUBMIT payload from Haskell: {payload}") from e
                if not isinstance(parsed, dict):
                    raise CodeInterpreterError("SUBMIT payload must be a JSON object")
                final_output = parsed
                continue
            lines.append(text)

    def _run_ghci_command(self, command: str) -> list[str]:
        self._write_line(command)
        self._write_line(f'putStrLn "{_COMMAND_END_MARKER}"')
        lines, _ = self._read_until_marker(allow_tool_calls=False)
        stderr_text = self._drain_stderr()
        if stderr_text and self._looks_like_error(stderr_text):
            raise CodeInterpreterError(stderr_text)
        return lines

    def _load_prelude(self) -> None:
        prelude_path = self._get_prelude_path()
        if not os.path.exists(prelude_path):
            raise CodeInterpreterError(f"Haskell prelude not found: {prelude_path}")
        with open(prelude_path, encoding="utf-8") as f:
            lines = f.read().splitlines()

        import_lines: list[str] = []
        decl_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("--"):
                continue
            if stripped.startswith("import "):
                import_lines.append(stripped)
            else:
                decl_lines.append(line)

        for line in import_lines:
            self._run_ghci_command(line)
        if decl_lines:
            declarations = ":{\n" + "\n".join(decl_lines) + "\n:}"
            self._run_ghci_command(declarations)

    def _health_check(self) -> None:
        output_lines = self._run_ghci_command("print (1 + 1 :: Int)")
        output = "\n".join(line for line in output_lines if line.strip())
        if "2" not in output:
            raise CodeInterpreterError(f"Unexpected health check response: {output!r}")

    def _extract_parameters(self, fn: Callable) -> list[dict]:
        sig = inspect.signature(fn)
        params = []
        for name, param in sig.parameters.items():
            item: dict[str, Any] = {"name": name}
            if param.annotation != inspect.Parameter.empty and param.annotation in SIMPLE_TYPES:
                item["type"] = param.annotation.__name__
            if param.default != inspect.Parameter.empty:
                item["default"] = param.default
            params.append(item)
        return params

    def _register_submit_signature(self) -> None:
        outputs = self.output_fields or []
        if not outputs:
            return
        field_names = [str(field["name"]) for field in outputs]
        args = [f"arg{i}" for i in range(len(field_names))]
        pairs = ", ".join(f'("{name}", dspyRenderValue {arg})' for name, arg in zip(field_names, args, strict=False))
        command = f"let submit {' '.join(args)} = dspySubmit [{pairs}]"
        self._run_ghci_command(command)

    def _serialize_to_haskell(self, value: Any) -> str:
        if value is None:
            return "()"
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            if value != value:
                return "(0/0)"
            if value == float("inf"):
                return "(1/0)"
            if value == float("-inf"):
                return "(-1/0)"
            return repr(value)
        if isinstance(value, str):
            escaped = (
                value.replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t")
            )
            return f'"{escaped}"'
        if isinstance(value, (list, tuple)):
            items = ", ".join(self._serialize_to_haskell(item) for item in value)
            return f"[{items}]"
        if isinstance(value, dict):
            pairs = ", ".join(
                f"({self._serialize_to_haskell(k)}, {self._serialize_to_haskell(v)})" for k, v in value.items()
            )
            return f"[{pairs}]"
        if isinstance(value, set):
            try:
                ordered = sorted(value)
            except TypeError:
                ordered = list(value)
            return self._serialize_to_haskell(ordered)
        raise CodeInterpreterError(f"Unsupported value type for Haskell: {type(value).__name__}")

    def _inject_variables(self, code: str, variables: dict[str, Any]) -> str:
        if not variables:
            return code
        lines: list[str] = []
        for key, value in variables.items():
            if not _VAR_NAME_PATTERN.match(key):
                raise CodeInterpreterError(f"Invalid Haskell variable name: '{key}'")
            if key in _HASKELL_RESERVED:
                raise CodeInterpreterError(f"Variable name '{key}' is a Haskell reserved word")
            lines.append(f"let {key} = {self._serialize_to_haskell(value)}")
        return "\n".join(lines) + "\n" + code

    def _normalize_submit_calls(self, code: str) -> str:
        return _SUBMIT_CALL_PATTERN.sub("submit", code)

    def _looks_like_error(self, stderr_text: str) -> bool:
        lowered = stderr_text.lower()
        return (
            " error:" in lowered
            or lowered.startswith("error:")
            or "parse error" in lowered
            or "*** exception" in lowered
        )

    def _classify_and_raise(self, stderr_text: str) -> None:
        lowered = stderr_text.lower()
        if "parse error" in lowered:
            raise SyntaxError(f"Invalid Haskell syntax: {stderr_text}")
        if "error:" in lowered:
            raise CodeInterpreterError(stderr_text)

    def _handle_tool_call(self, payload: str) -> None:
        try:
            req = json.loads(payload)
            tool_name = req.get("name")
            args = req.get("args", [])
        except json.JSONDecodeError as e:
            raise CodeInterpreterError(f"Invalid tool call payload from Haskell: {payload}") from e

        if tool_name not in self.tools:
            logger.warning("HaskellInterpreter tool call failed: unknown tool '%s'", tool_name)
            response = f"[ERROR] Unknown tool: {tool_name}"
        else:
            arg_chars = sum(len(str(arg)) for arg in args)
            logger.debug(
                "HaskellInterpreter tool call start name=%s args=%d chars=%d timeout=%.1fs",
                tool_name,
                len(args),
                arg_chars,
                self.tool_call_timeout_s,
            )
            start = time.monotonic()
            try:
                executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="haskell-tool-call")
                try:
                    future = executor.submit(self.tools[tool_name], *args)
                    try:
                        result = future.result(timeout=self.tool_call_timeout_s)
                    except FutureTimeoutError:
                        future.cancel()
                        duration_s = time.monotonic() - start
                        response = f"{_TOOL_TIMEOUT_PREFIX} Tool '{tool_name}' exceeded {self.tool_call_timeout_s:.1f}s"
                        logger.warning(
                            "HaskellInterpreter tool call timeout name=%s elapsed=%.3fs timeout=%.1fs",
                            tool_name,
                            duration_s,
                            self.tool_call_timeout_s,
                        )
                    else:
                        response = (
                            json.dumps(result, ensure_ascii=False)
                            if isinstance(result, (list, dict))
                            else str(result if result is not None else "")
                        )
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                response = f"[ERROR] {e}"
                logger.warning("HaskellInterpreter tool call error name=%s err=%s", tool_name, e)
            else:
                duration_s = time.monotonic() - start
                response_len = len(response)
                if duration_s >= self.log_slow_tool_calls_s:
                    logger.warning(
                        "HaskellInterpreter slow tool call name=%s elapsed=%.3fs threshold=%.1fs response_chars=%d",
                        tool_name,
                        duration_s,
                        self.log_slow_tool_calls_s,
                        response_len,
                    )
                else:
                    logger.debug(
                        "HaskellInterpreter tool call end name=%s elapsed=%.3fs response_chars=%d",
                        tool_name,
                        duration_s,
                        response_len,
                    )

        if len(response) > _MAX_TOOL_RESPONSE_CHARS:
            response = response[:_MAX_TOOL_RESPONSE_CHARS] + "\n... (truncated tool response) ..."
        response = response.replace("\r\n", "\n").replace("\n", "\\n")
        self._write_line(response)

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        self._check_thread_ownership()
        self._ensure_process()
        self._register_submit_signature()
        self._drain_stderr()

        full_code = self._normalize_submit_calls(self._inject_variables(code, variables or {}))
        for line in full_code.splitlines():
            self._write_line(line)
        self._write_line(f'putStrLn "{_COMMAND_END_MARKER}"')

        output_lines, final_output = self._read_until_marker(
            allow_tool_calls=True, timeout_seconds=self.execute_timeout_seconds
        )
        stderr_text = self._drain_stderr()

        if final_output is not None:
            cleaned = stderr_text.replace(_SUBMIT_SIGNAL, "")
            if cleaned.strip() and self._looks_like_error(cleaned):
                self._classify_and_raise(cleaned)
            return FinalOutput(final_output)

        if stderr_text and self._looks_like_error(stderr_text):
            self._classify_and_raise(stderr_text)

        output = "\n".join(line for line in output_lines if line.strip())
        return output or None

    def start(self) -> None:
        self._ensure_process()

    def shutdown(self) -> None:
        if self._process and self._process.poll() is None:
            try:
                self._write_line(":quit")
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
                self._process.wait()
        self._process = None
        self._owner_thread = None
        self._stderr_thread = None
        self._drain_stderr()

    def __enter__(self) -> HaskellInterpreter:
        return self

    def __exit__(self, *_: Any) -> None:
        self.shutdown()

    def __call__(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        return self.execute(code, variables)
