"""
Local interpreter for Scheme (Guile) code execution.
"""

import inspect
import json
import logging
import os
import queue
import select
import subprocess
import threading
import time
from typing import Any, Callable

from dspy_repl.core.code_interpreter import SIMPLE_TYPES, CodeInterpreterError, FinalOutput

__all__ = ["SchemeInterpreter", "FinalOutput", "CodeInterpreterError"]

logger = logging.getLogger(__name__)

JSONRPC_APP_ERRORS = {
    "SyntaxError": -32000,
    "RuntimeError": -32007,
    "CodeInterpreterError": -32008,
    "Unknown": -32099,
}


def _jsonrpc_request(method: str, params: dict, id: int | str) -> str:
    return json.dumps({"jsonrpc": "2.0", "method": method, "params": params, "id": id}, ensure_ascii=False)


def _jsonrpc_notification(method: str, params: dict | None = None) -> str:
    msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params:
        msg["params"] = params
    return json.dumps(msg, ensure_ascii=False)


def _jsonrpc_result(result: Any, id: int | str) -> str:
    return json.dumps({"jsonrpc": "2.0", "result": result, "id": id}, ensure_ascii=False)


def _jsonrpc_error(code: int, message: str, id: int | str, data: dict | None = None) -> str:
    err: dict[str, Any] = {"code": code, "message": message}
    if data:
        err["data"] = data
    return json.dumps({"jsonrpc": "2.0", "error": err, "id": id}, ensure_ascii=False)


_SCHEME_RESERVED = frozenset(
    {
        "define",
        "lambda",
        "if",
        "cond",
        "case",
        "and",
        "or",
        "not",
        "let",
        "let*",
        "letrec",
        "begin",
        "do",
        "set!",
        "quote",
        "quasiquote",
        "unquote",
        "syntax-rules",
        "define-syntax",
        "import",
        "export",
        "include",
        "else",
        "=>",
        "when",
        "unless",
    }
)


class SchemeInterpreter:
    """Local interpreter for Scheme execution using Guile."""

    def __init__(
        self,
        scheme_command: list[str] | None = None,
        tools: dict[str, Callable[..., str]] | None = None,
        output_fields: list[dict] | None = None,
        execute_timeout_seconds: float = 90.0,
        request_timeout_seconds: float = 30.0,
        tool_timeout_seconds: float = 45.0,
    ) -> None:
        if isinstance(scheme_command, dict):
            raise TypeError("scheme_command must be a list of strings, not a dict")

        self.tools = dict(tools) if tools else {}
        self.output_fields = output_fields
        self._tools_registered = False

        self.scheme_command = (
            list(scheme_command) if scheme_command else ["guile", "--no-auto-compile", "-s", self._get_runner_path()]
        )
        self._process: subprocess.Popen | None = None
        self._request_id = 0
        self._owner_thread: int | None = None
        self.execute_timeout_seconds = execute_timeout_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.tool_timeout_seconds = tool_timeout_seconds

    def _check_thread_ownership(self) -> None:
        current_thread = threading.current_thread().ident
        if self._owner_thread is None:
            self._owner_thread = current_thread
        elif self._owner_thread != current_thread:
            raise RuntimeError(
                "SchemeInterpreter is not thread-safe and cannot be shared across threads. "
                "Create a separate interpreter instance for each thread."
            )

    def _get_runner_path(self) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "scheme_runner.scm")

    def _ensure_process(self) -> None:
        if self._process is None or self._process.poll() is not None:
            try:
                self._process = subprocess.Popen(
                    self.scheme_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="UTF-8",
                    env=os.environ.copy(),
                )
            except FileNotFoundError as e:
                install_instructions = (
                    "Guile executable not found. Please install Guile to proceed.\n"
                    "Installation instructions:\n"
                    "> brew install guile  (macOS)\n"
                    "> apt install guile-3.0  (Ubuntu/Debian)\n"
                    "> pacman -S guile  (Arch)\n"
                    "For more info: https://www.gnu.org/software/guile/"
                )
                raise CodeInterpreterError(install_instructions) from e
            self._health_check()

    def _health_check(self) -> None:
        response = self._send_request("execute", {"code": "(display (+ 1 1)) (newline)"}, "during health check")
        output = response.get("result", {}).get("output", "").strip()
        if output != "2":
            raise CodeInterpreterError(f"Unexpected health check response: {response}")

    def _readline_with_timeout(self, timeout_seconds: float, context: str) -> str:
        assert self._process is not None and self._process.stdout is not None
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CodeInterpreterError(
                    f"Timed out after {timeout_seconds:.1f}s while waiting for Guile response {context}."
                )
            ready, _, _ = select.select([self._process.stdout], [], [], min(remaining, 0.25))
            if ready:
                line = self._process.stdout.readline()
                if line == "":
                    exit_code = self._process.poll()
                    stderr = self._process.stderr.read() if self._process and self._process.stderr else ""
                    raise CodeInterpreterError(
                        f"Guile subprocess closed output unexpectedly (exit={exit_code}) {context}: {stderr}"
                    )
                return line.rstrip("\n")

    def _send_request(self, method: str, params: dict, context: str) -> dict:
        self._request_id += 1
        request_id = self._request_id
        msg = _jsonrpc_request(method, params, request_id)
        assert self._process is not None and self._process.stdin is not None
        self._process.stdin.write(msg + "\n")
        self._process.stdin.flush()

        response_line = self._readline_with_timeout(self.request_timeout_seconds, context).strip()
        if not response_line:
            exit_code = self._process.poll()
            if exit_code is not None:
                stderr = self._process.stderr.read() if self._process.stderr else ""
                raise CodeInterpreterError(f"Guile exited (code {exit_code}) {context}: {stderr}")
            raise CodeInterpreterError(f"No response {context}")

        response = json.loads(response_line)
        if response.get("id") != request_id:
            raise CodeInterpreterError(
                f"Response ID mismatch {context}: expected {request_id}, got {response.get('id')}"
            )
        if "error" in response:
            raise CodeInterpreterError(f"Error {context}: {response['error'].get('message', 'Unknown error')}")
        return response

    def _call_tool_with_timeout(self, tool_name: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
        result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def worker() -> None:
            try:
                assert tool_name in self.tools
                result_queue.put((True, self.tools[tool_name](*args, **kwargs)))
            except Exception as exc:
                result_queue.put((False, exc))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(self.tool_timeout_seconds)
        if thread.is_alive():
            raise TimeoutError(f"Tool '{tool_name}' timed out after {self.tool_timeout_seconds:.1f}s.")
        ok, payload = result_queue.get_nowait()
        if ok:
            return payload
        raise payload

    def _handle_tool_call(self, request: dict) -> None:
        request_id = request["id"]
        params = request.get("params", {})
        tool_name = params.get("name")
        args = params.get("args", [])
        kwargs = params.get("kwargs") or {}
        if isinstance(kwargs, list):
            kwargs = {}

        try:
            if tool_name not in self.tools:
                raise CodeInterpreterError(f"Unknown tool: {tool_name}")
            result = self._call_tool_with_timeout(tool_name, args, kwargs)
            is_json = isinstance(result, (list, dict))
            response = _jsonrpc_result(
                {
                    "value": json.dumps(result) if is_json else str(result or ""),
                    "type": "json" if is_json else "string",
                },
                request_id,
            )
        except Exception as e:
            error_type = type(e).__name__
            error_code = JSONRPC_APP_ERRORS.get(error_type, JSONRPC_APP_ERRORS["Unknown"])
            response = _jsonrpc_error(error_code, str(e), request_id, {"type": error_type})

        assert self._process is not None and self._process.stdin is not None
        self._process.stdin.write(response + "\n")
        self._process.stdin.flush()

    def _extract_parameters(self, fn: Callable) -> list[dict]:
        sig = inspect.signature(fn)
        params = []
        for name, param in sig.parameters.items():
            p: dict[str, Any] = {"name": name}
            if param.annotation != inspect.Parameter.empty and param.annotation in SIMPLE_TYPES:
                p["type"] = param.annotation.__name__
            if param.default != inspect.Parameter.empty:
                p["default"] = param.default
            params.append(p)
        return params

    def _register_tools(self) -> None:
        if self._tools_registered:
            return

        params: dict[str, Any] = {}
        if self.tools:
            tools_info = [{"name": name, "parameters": self._extract_parameters(fn)} for name, fn in self.tools.items()]
            params["tools"] = tools_info
        if self.output_fields:
            params["outputs"] = self.output_fields
        if not params:
            self._tools_registered = True
            return

        self._send_request("register", params, "registering tools/outputs")
        self._tools_registered = True

    def _serialize_to_scheme(self, value: Any) -> str:
        if value is None:
            return "'()"
        if isinstance(value, bool):
            return "#t" if value else "#f"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return str(value)
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
            if not value:
                return "'()"
            items = " ".join(self._serialize_to_scheme(item) for item in value)
            return f"(list {items})"
        if isinstance(value, dict):
            if not value:
                return "'()"
            pairs = " ".join(
                f"(cons {self._serialize_to_scheme(k)} {self._serialize_to_scheme(v)})" for k, v in value.items()
            )
            return f"(list {pairs})"
        if isinstance(value, set):
            try:
                sorted_items = sorted(value)
            except TypeError:
                sorted_items = list(value)
            return self._serialize_to_scheme(sorted_items)
        raise CodeInterpreterError(f"Unsupported value type for Scheme: {type(value).__name__}")

    def _inject_variables(self, code: str, variables: dict[str, Any]) -> str:
        if not variables:
            return code
        for key in variables:
            if not key.isidentifier():
                raise CodeInterpreterError(f"Invalid variable name: '{key}'")
            if key in _SCHEME_RESERVED:
                raise CodeInterpreterError(f"Variable name '{key}' is a Scheme reserved word")
        definitions = [f"(define {k} {self._serialize_to_scheme(v)})" for k, v in variables.items()]
        return "\n".join(definitions) + "\n" + code

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        self._check_thread_ownership()
        variables = variables or {}
        code = self._inject_variables(code, variables)
        self._ensure_process()
        self._register_tools()

        self._request_id += 1
        execute_request_id = self._request_id
        input_data = _jsonrpc_request("execute", {"code": code}, execute_request_id)

        assert self._process is not None and self._process.stdin is not None
        try:
            self._process.stdin.write(input_data + "\n")
            self._process.stdin.flush()
        except BrokenPipeError:
            self._tools_registered = False
            self._ensure_process()
            self._register_tools()
            assert self._process is not None and self._process.stdin is not None
            self._process.stdin.write(input_data + "\n")
            self._process.stdin.flush()

        assert self._process.stdout is not None
        execute_deadline = time.monotonic() + self.execute_timeout_seconds
        while True:
            remaining = execute_deadline - time.monotonic()
            if remaining <= 0:
                raise CodeInterpreterError(
                    f"Execution timed out after {self.execute_timeout_seconds:.1f}s. "
                    "Try reducing sub-LLM calls, batching better, or simplifying the step."
                )
            output_line = self._readline_with_timeout(remaining, "during execute").strip()
            if not output_line:
                err_output = self._process.stderr.read() if self._process.stderr else ""
                raise CodeInterpreterError(f"No output from Guile subprocess. Stderr: {err_output}")

            if not output_line.startswith("{"):
                logger.debug("Skipping non-JSON output: %s", output_line)
                continue
            try:
                msg = json.loads(output_line)
            except json.JSONDecodeError:
                logger.info("Skipping malformed JSON: %s", output_line[:100])
                continue

            if "method" in msg and msg["method"] == "tool_call":
                self._handle_tool_call(msg)
                continue

            if "result" in msg:
                if msg.get("id") != execute_request_id:
                    raise CodeInterpreterError(
                        f"Response ID mismatch: expected {execute_request_id}, got {msg.get('id')}"
                    )
                result = msg["result"]
                if "final" in result:
                    return FinalOutput(result["final"])
                return result.get("output", None)

            if "error" in msg:
                if msg.get("id") is not None and msg.get("id") != execute_request_id:
                    raise CodeInterpreterError(
                        f"Response ID mismatch: expected {execute_request_id}, got {msg.get('id')}"
                    )
                error = msg["error"]
                error_code = error.get("code", JSONRPC_APP_ERRORS["Unknown"])
                error_message = error.get("message", "Unknown error")
                error_data = error.get("data", {})
                error_type = error_data.get("type", "Error")
                if error_code == JSONRPC_APP_ERRORS["SyntaxError"]:
                    raise SyntaxError(f"Invalid Scheme syntax: {error_message}")
                raise CodeInterpreterError(f"{error_type}: {error_data.get('args') or error_message}")

            raise CodeInterpreterError(f"Unexpected message format from sandbox: {msg}")

    def start(self) -> None:
        self._ensure_process()

    def shutdown(self) -> None:
        if self._process and self._process.poll() is None:
            try:
                assert self._process.stdin is not None
                self._process.stdin.write(_jsonrpc_notification("shutdown") + "\n")
                self._process.stdin.flush()
                self._process.stdin.close()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
                self._process.wait()
        self._process = None
        self._owner_thread = None

    def __enter__(self) -> "SchemeInterpreter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.shutdown()

    def __call__(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        return self.execute(code, variables)
