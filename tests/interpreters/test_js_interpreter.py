from __future__ import annotations

import os
import shutil
import time

import pytest

from dspy_repl.core.code_interpreter import FinalOutput
from dspy_repl.interpreters.js_interpreter import JavaScriptInterpreter

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or os.getenv("RUN_REPL_RUNTIME_TESTS") != "1",
    reason="JavaScript runtime tests require node and RUN_REPL_RUNTIME_TESTS=1",
)


def test_prelude_loads_and_basic_execution() -> None:
    with JavaScriptInterpreter() as interp:
        out = interp.execute("console.log(1 + 1);")
        assert "2" in out


def test_variable_injection() -> None:
    with JavaScriptInterpreter() as interp:
        out = interp.execute("console.log(count + 3);", variables={"count": 4})
        assert "7" in out


def test_submit_protocol_returns_final_output() -> None:
    with JavaScriptInterpreter(output_fields=[{"name": "answer"}]) as interp:
        result = interp.execute('submit("ok");')
        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": "ok"}


def test_tool_call_plumbing() -> None:
    def llm_query(prompt: str) -> str:
        return "tool-ok"

    with JavaScriptInterpreter(tools={"llmQuery": llm_query}) as interp:
        out = interp.execute('console.log(await llmQuery("ping"));')
        assert "tool-ok" in out


def test_tool_timeout_surface_in_error() -> None:
    def slow_tool(prompt: str) -> str:
        time.sleep(0.05)
        return "late"

    with JavaScriptInterpreter(tools={"llmQuery": slow_tool}, tool_call_timeout_s=0.01) as interp:
        with pytest.raises(Exception) as exc_info:
            interp.execute('console.log(await llmQuery("slow"));')
        assert "[TIMEOUT]" in str(exc_info.value)
