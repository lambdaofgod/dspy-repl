import argparse
import atexit
import logging
import os
from datetime import datetime

import dspy
from dspy_repl import DenoRLM
from dspy_repl.interpreters.deno_interpreter import DenoInterpreter, DenoPermissions

logging.basicConfig(level=logging.INFO)


def setup_otel_tracing(endpoint: str, otel_project: str) -> None:
    """Set up OpenTelemetry tracing with DSPy instrumentation, sending spans via OTLP."""
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from openinference.instrumentation.dspy import DSPyInstrumentor
    except ImportError as e:
        raise ImportError(
            f"OpenTelemetry tracing requires extra packages: {e}\n"
            "Install with: uv add opentelemetry-sdk opentelemetry-exporter-otlp openinference-instrumentation-dspy"
        ) from e

    provider = TracerProvider(resource=Resource({"service.name": otel_project}))
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
    )
    atexit.register(provider.shutdown)

    DSPyInstrumentor().instrument(tracer_provider=provider)
    logging.info("OTel tracing enabled, sending to %s", endpoint)


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
        "--otel-endpoint",
        metavar="URL",
        default=None,
        help="OTLP gRPC endpoint for OTel traces (e.g. http://localhost:4317)",
    )
    parser.add_argument(
        "--otel-project",
        default="dspy-repl-deno",
        help="OpenTelemetry project name for exporting spans (default: dspy-repl-deno)",
    )
    args = parser.parse_args()

    if args.otel_endpoint:
        setup_otel_tracing(args.otel_endpoint, args.otel_project)

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
