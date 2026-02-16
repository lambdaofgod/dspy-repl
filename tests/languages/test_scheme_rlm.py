from __future__ import annotations

import asyncio

from dspy_repl.core.code_interpreter import CodeInterpreterError, FinalOutput
from dspy_repl.languages.scheme_rlm import SchemeRLM
from tests.helpers.mock_interpreter import MockInterpreter
from tests.helpers.predictor_factory import make_mock_predictor


def test_scheme_code_fence_stripping(lm_context) -> None:
    with lm_context([{"output": "x"}]):
        rlm = SchemeRLM("context -> output", interpreter=MockInterpreter())
        code = "```scheme\n(display 42)\n```"
        assert rlm._strip_code_fences(code) == "(display 42)"


def test_scheme_single_iteration_submit(lm_context) -> None:
    mock = MockInterpreter(responses=[FinalOutput({"answer": "42"})])
    with lm_context([{"answer": "42"}]):
        rlm = SchemeRLM("context -> answer", interpreter=mock)
        rlm.generate_action = make_mock_predictor([{"reasoning": "done", "code": '(SUBMIT "42")'}])
        result = rlm(context="test")
        assert result.answer == "42"
        assert len(result.trajectory) == 1


def test_scheme_error_recovery(lm_context) -> None:
    mock = MockInterpreter(responses=[CodeInterpreterError("bad call"), FinalOutput({"answer": "ok"})])
    with lm_context([{"answer": "ok"}]):
        rlm = SchemeRLM("context -> answer", interpreter=mock, max_iterations=3)
        rlm.generate_action = make_mock_predictor(
            [
                {"reasoning": "try", "code": "(bad-call)"},
                {"reasoning": "recover", "code": '(SUBMIT "ok")'},
            ]
        )
        result = rlm(context="test")
        assert result.answer == "ok"
        assert len(result.trajectory) == 2


def test_scheme_fallback_extract_after_max_iterations(lm_context) -> None:
    mock = MockInterpreter(responses=["step output", "step output"])
    with lm_context([{"answer": "fallback"}]):
        rlm = SchemeRLM("context -> answer", interpreter=mock, max_iterations=2)
        rlm.generate_action = make_mock_predictor([{"reasoning": "explore", "code": "(display context)"}])
        rlm.extract = make_mock_predictor([{"answer": "fallback"}])
        result = rlm(context="test")
        assert result.answer == "fallback"
        assert result.final_reasoning == "Extract forced final output"


def test_scheme_async_forward(lm_context) -> None:
    mock = MockInterpreter(responses=[FinalOutput({"answer": "async"})])
    with lm_context([{"answer": "async"}]):
        rlm = SchemeRLM("context -> answer", interpreter=mock)
        rlm.generate_action = make_mock_predictor([{"reasoning": "done", "code": '(SUBMIT "async")'}])
        result = asyncio.run(rlm.aforward(context="test"))
        assert result.answer == "async"
