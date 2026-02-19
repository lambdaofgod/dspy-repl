import argparse
import logging
import os
from datetime import datetime

import dspy
from dspy_repl import DenoRLM
from dspy_repl.interpreters.deno_interpreter import DenoInterpreter, DenoPermissions

logging.basicConfig(level=logging.INFO)


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
    parser.add_argument(
        "--logdir",
        metavar="PATH",
        default=None,
        help="Directory to write log files (e.g. rlm_deno_2026-02-04_17-44-14.log)",
    )
    args = parser.parse_args()

    if args.logdir:
        os.makedirs(args.logdir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path = os.path.join(args.logdir, f"rlm_deno_{timestamp}.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(file_handler)

    dspy.configure(lm=dspy.LM(args.model))

    permissions = DenoPermissions(
        allow_net=not args.no_net,
        allow_read=[args.workdir] if args.workdir else [],
        allow_write=[args.workdir] if args.workdir else [],
    )
    interp = DenoInterpreter(deno_permissions=permissions)

    rlm = DenoRLM("context, query -> answer", verbose=True, interpreter=interp)
    result = rlm(context="", query=args.question)

    print("#" * 50)
    print("TRAJECTORY:")
    print("#" * 50)
    print(result.trajectory)
    print("#" * 50)
    print("ANSWER:")
    print("#" * 50)
    print(result.answer)


if __name__ == "__main__":
    main()
