import argparse

import dspy
from dspy_repl import DenoRLM
from dspy_repl.interpreters.deno_interpreter import DenoInterpreter, DenoPermissions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Deno RLM query")
    parser.add_argument("question", help="Question to ask the model")
    parser.add_argument(
        "--model",
        default="openrouter/anthropic/claude-sonnet-4-6",
        help="LiteLLM model name",
    )
    parser.add_argument(
        "--no-net",
        action="store_true",
        default=False,
        help="Disable network access (enabled by default)",
    )
    parser.add_argument(
        "--workdir",
        metavar="PATH",
        default=None,
        help="Directory to allow for both read and write access",
    )
    args = parser.parse_args()

    dspy.configure(lm=dspy.LM(args.model))

    permissions = DenoPermissions(
        allow_net=not args.no_net,
        allow_read=[args.workdir] if args.workdir else [],
        allow_write=[args.workdir] if args.workdir else [],
    )
    interp = DenoInterpreter(deno_permissions=permissions)

    rlm = DenoRLM("context, query -> answer", verbose=True, interpreter=interp)
    result = rlm(context="", query=args.question)
    print(result.answer)
    print(result.trajectory)


if __name__ == "__main__":
    main()
