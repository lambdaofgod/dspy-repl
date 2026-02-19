from __future__ import annotations

import os
import shutil
import time

import pytest

from dspy_repl.core.code_interpreter import FinalOutput
from dspy_repl.interpreters.deno_interpreter import DenoInterpreter, DenoPermissions

_needs_deno = pytest.mark.skipif(
    shutil.which("deno") is None or os.getenv("RUN_REPL_RUNTIME_TESTS") != "1",
    reason="Deno runtime tests require deno and RUN_REPL_RUNTIME_TESTS=1",
)


# -- Unit tests (no Deno required) --


def test_deno_permissions_validates_paths() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        DenoPermissions(allow_read=["/nonexistent/path/abc123"])


def test_deno_permissions_validates_hosts() -> None:
    with pytest.raises(ValueError, match="Invalid port"):
        DenoPermissions(allow_net=["example.com:notaport"])


def test_deno_permissions_to_args() -> None:
    perms = DenoPermissions(allow_net=["example.com:443", "api.test.com"], allow_read=["/tmp"], allow_write=["/tmp"])
    args = perms.to_args()
    assert "--allow-net=example.com:443,api.test.com" in args
    assert "--allow-read=/tmp" in args
    assert "--allow-write=/tmp" in args


def test_deno_command_rejects_dict() -> None:
    with pytest.raises(TypeError, match="must be a list of strings, not a dict"):
        DenoInterpreter(deno_command={"bad": "value"})


# -- Runtime tests (require Deno + RUN_REPL_RUNTIME_TESTS=1) --


@_needs_deno
def test_prelude_loads_and_basic_execution() -> None:
    with DenoInterpreter(execute_timeout_seconds=5) as interp:
        out = interp.execute("console.log(1 + 1);")
        assert "2" in out


@_needs_deno
def test_variable_injection() -> None:
    with DenoInterpreter(execute_timeout_seconds=5) as interp:
        out = interp.execute("console.log(count + 3);", variables={"count": 4})
        assert "7" in out


@_needs_deno
def test_submit_protocol_returns_final_output() -> None:
    with DenoInterpreter(output_fields=[{"name": "answer"}], execute_timeout_seconds=5) as interp:
        result = interp.execute('submit("ok");')
        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": "ok"}


@_needs_deno
def test_tool_call_plumbing() -> None:
    def llm_query(prompt: str) -> str:
        return "tool-ok"

    with DenoInterpreter(tools={"llmQuery": llm_query}, execute_timeout_seconds=5) as interp:
        out = interp.execute('console.log(await llmQuery("ping"));')
        assert "tool-ok" in out


@_needs_deno
def test_tool_timeout_surface_in_error() -> None:
    def slow_tool(prompt: str) -> str:
        time.sleep(0.05)
        return "late"

    with DenoInterpreter(tools={"llmQuery": slow_tool}, tool_call_timeout_s=0.01, execute_timeout_seconds=5) as interp:
        with pytest.raises(Exception) as exc_info:
            interp.execute('console.log(await llmQuery("slow"));')
        assert "[TIMEOUT]" in str(exc_info.value)


@_needs_deno
def test_syntax_error_classification() -> None:
    with DenoInterpreter(execute_timeout_seconds=5) as interp:
        with pytest.raises(SyntaxError, match="Invalid TypeScript syntax"):
            interp.execute("if (")
