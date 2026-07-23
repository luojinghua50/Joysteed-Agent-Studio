"""Observability setup: Langfuse + OpenTelemetry integration."""
import os


def setup_langfuse():
    """Create Langfuse callback handler for LangChain tracing."""
    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")

    if not public_key or not secret_key:
        return None

    os.environ.setdefault("LANGFUSE_HOST", host)
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", secret_key)

    try:
        from langfuse.langchain import CallbackHandler
        return CallbackHandler()
    except (ImportError, TypeError, Exception) as e:
        import structlog
        structlog.get_logger().warning("langfuse_setup_failed", error=str(e))

    return None


def setup_opentelemetry(app=None):
    """Initialize OpenTelemetry with OTLP exporter."""
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if not endpoint:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        resource = Resource.create({"service.name": "agent-core", "service.version": "0.1.0"})
        provider = TracerProvider(resource=resource)

        exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
        provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)

        if app:
            FastAPIInstrumentor.instrument_app(app, excluded_urls="metrics,health")
    except ImportError:
        pass
