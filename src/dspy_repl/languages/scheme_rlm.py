from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Callable

from pydantic import Field

from dspy_repl.compat import dspy, ensure_signature, get_active_lm, translate_field_type
from dspy_repl.core.base_rlm import BaseReplRLM
from dspy_repl.core.code_interpreter import CodeInterpreter
from dspy_repl.core.repl_types import REPLEntry, REPLHistory
from dspy_repl.interpreters import load_scheme_interpreter

if TYPE_CHECKING:
    from dspy_repl.compat import Signature


class SchemeREPLEntry(REPLEntry):
    def format(self, index: int, max_output_chars: int = 10_000) -> str:
        reasoning_line = f"Reasoning: {self.reasoning}\n" if self.reasoning else ""
        return (
            f"=== Step {index + 1} ===\n"
            f"{reasoning_line}"
            f"Code:\n```scheme\n{self.code}\n```\n"
            f"{self.format_output(self.output, max_output_chars)}"
        )


class SchemeREPLHistory(REPLHistory):
    entries: list[SchemeREPLEntry] = Field(default_factory=list)

    def format(self) -> str:
        if not self.entries:
            return "You have not interacted with the REPL environment yet."
        return "\n".join(
            entry.format(index=i, max_output_chars=self.max_output_chars) for i, entry in enumerate(self.entries)
        )

    def append(self, *, reasoning: str = "", code: str, output: str) -> SchemeREPLHistory:
        new_entry = SchemeREPLEntry(reasoning=reasoning, code=code, output=output)
        return SchemeREPLHistory(entries=list(self.entries) + [new_entry], max_output_chars=self.max_output_chars)


SCHEME_ACTION_INSTRUCTIONS_TEMPLATE = """You are tasked with producing the following outputs given the inputs {inputs}:
{output_fields}

You have access to a Scheme (Guile) REPL environment. Write Scheme code and it will be evaluated. You will see the output, then write more code based on what you learned. This is an iterative process.

Available:
- Variables: {inputs} (your input data, accessible by name)
- (llm-query prompt) - query a sub-LLM for semantic analysis
- (llm-query-batched prompt1 prompt2 ...) - query multiple prompts concurrently (much faster)
- (display expr) and (newline) - ALWAYS use display to see results
- (SUBMIT {final_output_names}) - submit final output when done
- Standard libraries: srfi-1 (lists), srfi-13 (strings), ice-9 regex, etc.

IMPORTANT: This is ITERATIVE. Each code block you write will execute, you'll see the output, then you decide what to do next. Do NOT try to solve everything in one step.

1. EXPLORE FIRST - Look at your data before processing it. Use display to print samples, check types/lengths, understand the structure.
2. ITERATE - Write small code snippets, observe outputs, then decide next steps. State persists between iterations.
3. VERIFY BEFORE SUBMITTING - If results seem wrong (empty, unexpected), reconsider your approach.
4. USE llm-query FOR SEMANTICS - String matching finds WHERE things are; llm-query understands WHAT things mean.
5. USE SCHEME IDIOMS - Prefer map/filter/fold over manual loops. Use let/let* for local bindings. Use association lists for key-value data.
6. SUBMIT ONLY AFTER SEEING OUTPUTS - SUBMIT ends the current run immediately. If you need to inspect displayed output, run it in one step, review the result, then call SUBMIT in a later step.

You have max {max_llm_calls} sub-LLM calls. When done, call SUBMIT with your output."""


class SchemeRLM(BaseReplRLM):
    _RESERVED_TOOL_NAMES = frozenset({"llm-query", "llm-query-batched", "SUBMIT", "display", "tool-call"})
    _CODE_FENCE_PATTERN = re.compile(r"^```(?:scheme|lisp|scm|racket)?\s*\n(.*)\n```\s*$", re.DOTALL)

    def __init__(
        self,
        signature: type[Signature] | str,
        max_iterations: int = 20,
        max_llm_calls: int = 50,
        max_output_chars: int = 10_000,
        verbose: bool = False,
        tools: list[Callable] | None = None,
        sub_lm: dspy.LM | None = None,
        interpreter: CodeInterpreter | None = None,
    ) -> None:
        self.signature = ensure_signature(signature)
        self.sub_lm = sub_lm
        super().__init__(
            max_iterations=max_iterations,
            max_llm_calls=max_llm_calls,
            max_output_chars=max_output_chars,
            verbose=verbose,
            tools=tools,
            interpreter=interpreter,
        )
        action_sig, extract_sig = self._build_signatures()
        self.generate_action = dspy.Predict(action_sig)
        self.extract = dspy.Predict(extract_sig)

    def _new_history(self) -> SchemeREPLHistory:
        return SchemeREPLHistory(max_output_chars=self.max_output_chars)

    def _format_output(self, output: str) -> str:
        return output or "(no output - did you forget to use display?)"

    def _missing_outputs_hint(self, output_field_names: list[str], missing: set[str]) -> str:
        return f"[Error] Missing output fields: {sorted(missing)}. Use (SUBMIT {' '.join(output_field_names)})"

    def _create_interpreter(self, execution_tools: dict[str, Callable]) -> CodeInterpreter:
        interpreter_cls = load_scheme_interpreter()
        return interpreter_cls(tools=execution_tools, output_fields=self._get_output_fields_info())

    def _make_llm_tools(self, max_workers: int = 8) -> dict[str, Callable]:
        state = {"call_count": 0}
        lock = threading.Lock()

        def _check_and_increment(n: int = 1) -> None:
            with lock:
                if state["call_count"] + n > self.max_llm_calls:
                    raise RuntimeError(
                        f"LLM call limit exceeded: {state['call_count']} + {n} > {self.max_llm_calls}. "
                        "Use Scheme code for aggregation instead of making more LLM calls."
                    )
                state["call_count"] += n

        def _query_lm(prompt: str) -> str:
            target_lm = get_active_lm(self.sub_lm)
            if target_lm is None:
                raise RuntimeError("No LM configured. Use dspy.configure(lm=...) or pass sub_lm to SchemeRLM.")
            response = target_lm(prompt)
            if isinstance(response, list) and response:
                item = response[0]
                if isinstance(item, dict) and "text" in item:
                    return item["text"]
                return str(item)
            return str(response)

        def llm_query(prompt: str) -> str:
            if not prompt:
                raise ValueError("prompt cannot be empty")
            _check_and_increment(1)
            return _query_lm(prompt)

        def llm_query_batched(*prompts: str) -> list[str]:
            if not prompts:
                return []
            _check_and_increment(len(prompts))
            results: dict[int, str] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {executor.submit(_query_lm, p): i for i, p in enumerate(prompts)}
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        results[idx] = f"[ERROR] {e}"
            return [results[i] for i in range(len(prompts))]

        return {"llm-query": llm_query, "llm-query-batched": llm_query_batched}

    def _build_signatures(self) -> tuple[Signature, Signature]:
        inputs_str = ", ".join(f"`{n}`" for n in self.signature.input_fields)
        final_output_names = " ".join(self.signature.output_fields.keys())
        output_fields = "\n".join(f"- {translate_field_type(n, f)}" for n, f in self.signature.output_fields.items())
        task_instructions = f"{self.signature.instructions}\n\n" if self.signature.instructions else ""
        tool_docs = self._format_tool_docs(self._user_tools)

        action_sig = (
            dspy.Signature(
                {},
                task_instructions
                + SCHEME_ACTION_INSTRUCTIONS_TEMPLATE.format(
                    inputs=inputs_str,
                    final_output_names=final_output_names,
                    output_fields=output_fields,
                    max_llm_calls=self.max_llm_calls,
                )
                + tool_docs,
            )
            .append(
                "variables_info", dspy.InputField(desc="Metadata about the variables available in the REPL"), type_=str
            )
            .append(
                "repl_history",
                dspy.InputField(desc="Previous REPL code executions and their outputs"),
                type_=SchemeREPLHistory,
            )
            .append(
                "iteration",
                dspy.InputField(desc="Current iteration number (1-indexed) out of max_iterations"),
                type_=str,
            )
            .append(
                "reasoning",
                dspy.OutputField(desc="Think step-by-step: what do you know? What remains? Plan your next action."),
                type_=str,
            )
            .append("code", dspy.OutputField(desc="Scheme code to execute. Use markdown code block format."), type_=str)
        )

        extract_instructions = """Based on the REPL trajectory, extract the final outputs now.

Review your trajectory to see what information you gathered and what values you computed, then provide the final outputs."""
        extended_task_instructions = (
            "The trajectory was generated with the following objective: \n" + task_instructions + "\n"
            if task_instructions
            else ""
        )
        extract_sig = dspy.Signature(
            {**self.signature.output_fields}, extended_task_instructions + extract_instructions
        )
        extract_sig = extract_sig.prepend(
            "repl_history", dspy.InputField(desc="Your REPL interactions so far"), type_=SchemeREPLHistory
        )
        extract_sig = extract_sig.prepend(
            "variables_info",
            dspy.InputField(desc="Metadata about the variables available in the REPL"),
            type_=str,
        )
        return action_sig, extract_sig
