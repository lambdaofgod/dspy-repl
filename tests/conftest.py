from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import dspy
import pytest
from dspy.utils.dummies import DummyLM


@contextmanager
def dummy_lm_context(responses: list[dict[str, Any]]) -> Iterator[DummyLM]:
    lm = DummyLM(responses)
    with dspy.context(lm=lm):
        yield lm


@pytest.fixture
def lm_context():
    return dummy_lm_context
