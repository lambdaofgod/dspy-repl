import dspy_repl
from dspy_repl import SQLRLM, HaskellRLM, JavaScriptRLM, SchemeRLM


def test_public_exports() -> None:
    assert SchemeRLM is not None
    assert SQLRLM is not None
    assert HaskellRLM is not None
    assert JavaScriptRLM is not None
    assert hasattr(dspy_repl, "__version__")
