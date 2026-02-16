from __future__ import annotations

import os
import shutil

import dspy
import pytest

from dspy_repl import SQLRLM, HaskellRLM, SchemeRLM


def _live_tests_enabled() -> bool:
    return os.getenv("RUN_LIVE_LM_TESTS") == "1"


def _has_google_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


def _answer_looks_correct(answer: str) -> bool:
    normalized = answer.strip().lower()
    return "4" in normalized or "four" in normalized


@pytest.mark.parametrize(
    ("engine_name", "rlm_cls", "required_runtime"),
    [
        ("scheme", SchemeRLM, "guile"),
        ("sql", SQLRLM, None),
        ("haskell", HaskellRLM, "ghci"),
    ],
)
def test_live_gemini_flash_all_repls(engine_name: str, rlm_cls: type, required_runtime: str | None) -> None:
    """Manual live integration test for all non-Python REPL engines.

    Run with:
      RUN_LIVE_LM_TESTS=1 GEMINI_API_KEY=... pytest tests/integration/test_live_gemini_repls.py -q
    """
    if not _live_tests_enabled():
        pytest.skip("Set RUN_LIVE_LM_TESTS=1 to run live LM integration tests")
    if not _has_google_key():
        pytest.skip("GEMINI_API_KEY is required for Gemini integration tests")
    if required_runtime and shutil.which(required_runtime) is None:
        pytest.skip(f"{required_runtime} is required for {engine_name} integration test")

    lm = dspy.LM("gemini/gemini-3-flash-preview")
    prompt = "What is 2 + 2? Return only the final answer."

    with dspy.context(lm=lm):
        rlm = rlm_cls(
            "question -> answer",
            max_iterations=4,
            max_llm_calls=6,
            verbose=False,
        )
        result = rlm(question=prompt)

    answer = str(result.answer)
    assert _answer_looks_correct(answer), f"{engine_name} produced unexpected answer: {answer!r}"
