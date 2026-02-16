from __future__ import annotations

import asyncio

from dspy_repl.core.code_interpreter import CodeInterpreterError, FinalOutput
from dspy_repl.languages.haskell_rlm import HaskellRLM
from tests.helpers.mock_interpreter import MockInterpreter
from tests.helpers.predictor_factory import make_mock_predictor


def test_haskell_code_fence_stripping(lm_context) -> None:
    with lm_context([{"output": "x"}]):
        rlm = HaskellRLM("context -> output", interpreter=MockInterpreter())
        code = "```haskell\nprint 42\n```"
        assert rlm._strip_code_fences(code) == "print 42"


def test_haskell_single_iteration_submit(lm_context) -> None:
    mock = MockInterpreter(responses=[FinalOutput({"answer": "42"})])
    with lm_context([{"answer": "42"}]):
        rlm = HaskellRLM("context -> answer", interpreter=mock)
        rlm.generate_action = make_mock_predictor([{"reasoning": "done", "code": 'submit "42"'}])
        result = rlm(context="test")
        assert result.answer == "42"
        assert len(result.trajectory) == 1


def test_haskell_error_recovery(lm_context) -> None:
    mock = MockInterpreter(responses=[CodeInterpreterError("bad code"), FinalOutput({"answer": "ok"})])
    with lm_context([{"answer": "ok"}]):
        rlm = HaskellRLM("context -> answer", interpreter=mock, max_iterations=3)
        rlm.generate_action = make_mock_predictor(
            [
                {"reasoning": "try", "code": "badCall"},
                {"reasoning": "recover", "code": 'submit "ok"'},
            ]
        )
        result = rlm(context="test")
        assert result.answer == "ok"
        assert len(result.trajectory) == 2


def test_haskell_fallback_extract_after_max_iterations(lm_context) -> None:
    mock = MockInterpreter(responses=["out", "out"])
    with lm_context([{"answer": "fallback"}]):
        rlm = HaskellRLM("context -> answer", interpreter=mock, max_iterations=2)
        rlm.generate_action = make_mock_predictor([{"reasoning": "explore", "code": "putStrLn context"}])
        rlm.extract = make_mock_predictor([{"answer": "fallback"}])
        result = rlm(context="test")
        assert result.answer == "fallback"
        assert result.final_reasoning == "Extract forced final output"


def test_haskell_async_forward(lm_context) -> None:
    mock = MockInterpreter(responses=[FinalOutput({"answer": "async"})])
    with lm_context([{"answer": "async"}]):
        rlm = HaskellRLM("context -> answer", interpreter=mock)
        rlm.generate_action = make_mock_predictor([{"reasoning": "done", "code": 'submit "async"'}])
        result = asyncio.run(rlm.aforward(context="test"))
        assert result.answer == "async"
