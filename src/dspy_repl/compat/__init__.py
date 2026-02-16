"""Thin shims around DSPy APIs used by dspy-repl."""

from typing import Any

import dspy
from dspy.adapters.types.tool import Tool
from dspy.adapters.utils import parse_value, translate_field_type
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.signatures.signature import Signature, ensure_signature

__all__ = [
    "Module",
    "Prediction",
    "Signature",
    "Tool",
    "dspy",
    "ensure_signature",
    "parse_value",
    "translate_field_type",
]


def get_active_lm(sub_lm: dspy.LM | None) -> Any:
    """Return explicitly passed LM, otherwise the active DSPy LM from global settings."""
    return sub_lm if sub_lm is not None else dspy.settings.lm
