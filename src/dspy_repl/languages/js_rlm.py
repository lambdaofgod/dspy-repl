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
from dspy_repl.interpreters.js_interpreter import JavaScriptInterpreter

if TYPE_CHECKING:
    from dspy_repl.compat import Signature


class JavaScriptREPLEntry(REPLEntry):
    def format(self, index: int, max_output_chars: int = 10_000) -> str:
        reasoning_line = f"Reasoning: {self.reasoning}\n" if self.reasoning else ""
        return (
            f"=== Step {index + 1} ===\n"
            f"{reasoning_line}"
            f"Code:\n```javascript\n{self.code}\n```\n"
            f"{self.format_output(self.output, max_output_chars)}"
        )


class JavaScriptREPLHistory(REPLHistory):
    entries: list[JavaScriptREPLEntry] = Field(default_factory=list)

    def format(self) -> str:
        if not self.entries:
            return "You have not interacted with the JavaScript REPL environment yet."
        return "\n".join(
            entry.format(index=i, max_output_chars=self.max_output_chars) for i, entry in enumerate(self.entries)
        )

    def append(self, *, reasoning: str = "", code: str, output: str) -> JavaScriptREPLHistory:
        new_entry = JavaScriptREPLEntry(reasoning=reasoning, code=code, output=output)
        return JavaScriptREPLHistory(entries=list(self.entries) + [new_entry], max_output_chars=self.max_output_chars)


JS_ACTION_INSTRUCTIONS_TEMPLATE = """You are tasked with producing the following outputs given the inputs {inputs}:
{output_fields}

You have access to a JavaScript (Node.js) REPL environment. Write JavaScript code and it will be evaluated. You will see the output, then write more code based on what you learned. This is an iterative process.

Available:
- Variables: {inputs} (your input data, accessible by name)
- await llmQuery("prompt") - query a sub-LLM for semantic analysis
- await llmQueryBatched(["prompt1", "prompt2", ...]) - query multiple prompts
- console.log(value) - ALWAYS print to inspect intermediate results
- submit({final_output_pairs}) - submit final output when done

IMPORTANT: This is ITERATIVE.
1. EXPLORE FIRST - inspect samples and structure.
2. ITERATE - run small snippets and inspect outputs.
3. VERIFY BEFORE SUBMITTING - if results look wrong, revise.
4. USE llmQuery FOR SEMANTICS where needed.
5. SUBMIT ONLY AFTER SEEING OUTPUTS.

You have max {max_llm_calls} sub-LLM calls. When done, call submit with your output."""


class JavaScriptRLM(BaseReplRLM):
    _RESERVED_TOOL_NAMES = frozenset({"llmQuery", "llmQueryBatched", "submit", "SUBMIT", "console"})
    _CODE_FENCE_PATTERN = re.compile(r"^```(?:javascript|js)?\s*\n(.*)\n```\s*$", re.DOTALL)

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

    def _new_history(self) -> JavaScriptREPLHistory:
        return JavaScriptREPLHistory(max_output_chars=self.max_output_chars)

    def _format_output(self, output: str) -> str:
        return output or "(no output - did you forget console.log?)"

    def _missing_outputs_hint(self, output_field_names: list[str], missing: set[str]) -> str:
        fields = ", ".join(f"{name}: <value>" for name in output_field_names)
        return f"[Error] Missing output fields: {sorted(missing)}. Use submit({{{fields}}})"

    def _create_interpreter(self, execution_tools: dict[str, Callable]) -> CodeInterpreter:
        return JavaScriptInterpreter(tools=execution_tools, output_fields=self._get_output_fields_info())

    def _make_llm_tools(self, max_workers: int = 8) -> dict[str, Callable]:
        state = {"call_count": 0}
        lock = threading.Lock()

        def _check_and_increment(n: int = 1) -> None:
            with lock:
                if state["call_count"] + n > self.max_llm_calls:
                    raise RuntimeError(
                        f"LLM call limit exceeded: {state['call_count']} + {n} > {self.max_llm_calls}. "
                        "Use JavaScript for aggregation instead of making more LLM calls."
                    )
                state["call_count"] += n

        def _query_lm(prompt: str) -> str:
            target_lm = get_active_lm(self.sub_lm)
            if target_lm is None:
                raise RuntimeError("No LM configured. Use dspy.configure(lm=...) or pass sub_lm to JavaScriptRLM.")
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

        def llm_query_batched(prompts: list[str]) -> list[str]:
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

        return {"llmQuery": llm_query, "llmQueryBatched": llm_query_batched}

    def _build_signatures(self) -> tuple[Signature, Signature]:
        inputs_str = ", ".join(f"`{n}`" for n in self.signature.input_fields)
        final_output_pairs = ", ".join(f"{name}: ..." for name in self.signature.output_fields.keys())
        output_fields = "\n".join(f"- {translate_field_type(n, f)}" for n, f in self.signature.output_fields.items())
        task_instructions = f"{self.signature.instructions}\n\n" if self.signature.instructions else ""
        tool_docs = self._format_tool_docs(self._user_tools)

        action_sig = (
            dspy.Signature(
                {},
                task_instructions
                + JS_ACTION_INSTRUCTIONS_TEMPLATE.format(
                    inputs=inputs_str,
                    final_output_pairs=final_output_pairs,
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
                type_=JavaScriptREPLHistory,
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
            .append(
                "code", dspy.OutputField(desc="JavaScript code to execute. Use markdown code block format."), type_=str
            )
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
            "repl_history", dspy.InputField(desc="Your REPL interactions so far"), type_=JavaScriptREPLHistory
        )
        extract_sig = extract_sig.prepend(
            "variables_info",
            dspy.InputField(desc="Metadata about the variables available in the REPL"),
            type_=str,
        )
        return action_sig, extract_sig
