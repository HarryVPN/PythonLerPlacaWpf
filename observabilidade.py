"""Telemetria local do leitor ONNX; sem captura de imagens ou texto de placas."""
import logging
import os
import time
import uuid
from contextlib import contextmanager
from urllib.parse import urlsplit

logger = logging.getLogger("bol.python")
_tracer = None
_duracao = None
_erros = None


def configurar_observabilidade(app):
    global _tracer, _duracao, _erros
    if os.getenv("BOL_ENVIRONMENT", "Development").lower() != "development":
        return
    if os.getenv("BOL_OBSERVABILITY_ENABLED", "true").lower() != "true":
        return
    if "bol_observability" in app.extensions:
        return

    endpoint = os.getenv("BOL_OTLP_ENDPOINT", "http://localhost:4318").rstrip("/")
    url = urlsplit(endpoint)
    if (url.scheme != "http" or url.hostname not in ("localhost", "127.0.0.1", "::1")
            or url.path or url.query or url.fragment or url.username or url.password):
        raise ValueError("BOL_OTLP_ENDPOINT deve ser HTTP local, como http://localhost:4318")

    os.environ.setdefault("OTEL_SEMCONV_STABILITY_OPT_IN", "http")
    from flask import g, request
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    from opentelemetry.instrumentation.flask import FlaskInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.threading import ThreadingInstrumentor
    from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor
    from opentelemetry.instrumentation.logging.handler import LoggingHandler
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import ALWAYS_ON
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.view import View, ExplicitBucketHistogramAggregation
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

    resource = Resource.create({"service.name": "BOL-Python",
                                "service.instance.id": str(uuid.uuid4()),
                                "deployment.environment.name": "Development"})
    traces = TracerProvider(resource=resource, sampler=ALWAYS_ON)
    traces.add_span_processor(BatchSpanProcessor(
        OTLPSpanExporter(endpoint=endpoint + "/v1/traces", timeout=2)))
    reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=endpoint + "/v1/metrics", timeout=2),
        export_interval_millis=5000)
    metrics = MeterProvider(resource=resource, metric_readers=[reader], views=[View(
        instrument_name="bol.python.stage.duration",
        aggregation=ExplicitBucketHistogramAggregation(
            boundaries=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)))])
    logs = LoggerProvider(resource=resource)
    logs.add_log_record_processor(BatchLogRecordProcessor(
        OTLPLogExporter(endpoint=endpoint + "/v1/logs", timeout=2)))
    handler = LoggingHandler(level=logging.INFO, logger_provider=logs)
    # Do not export stdout/print, frame contents or Werkzeug URLs with query strings.
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    _tracer = traces.get_tracer("bol.python")
    meter = metrics.get_meter("bol.python")
    _duracao = meter.create_histogram("bol.python.stage.duration", unit="s")
    _erros = meter.create_counter("bol.python.stage.errors", unit="{error}")
    FlaskInstrumentor().instrument_app(app, tracer_provider=traces, meter_provider=metrics)
    RequestsInstrumentor().instrument(tracer_provider=traces, meter_provider=metrics)
    ThreadingInstrumentor().instrument()
    SystemMetricsInstrumentor(config={
        "process.cpu.time": ["user", "system"],
        "process.cpu.utilization": ["user", "system"],
        "process.memory.usage": None,
        "process.thread.count": None,
        "cpython.gc.collections": None,
    }).instrument(meter_provider=metrics)

    @app.before_request
    def iniciar_tempo():
        g.bol_inicio = time.perf_counter()

    @app.after_request
    def registrar_requisicao(response):
        inicio = getattr(g, "bol_inicio", None)
        if inicio is not None:
            logger.info("HTTP %s %s status=%s duracao_ms=%.2f", request.method,
                        request.url_rule.rule if request.url_rule else "unmatched",
                        response.status_code, (time.perf_counter() - inicio) * 1000)
        return response

    app.extensions["bol_observability"] = {"traces": traces, "metrics": metrics, "logs": logs}


def registrar_falha(etapa, erro):
    if _tracer is None:
        return
    from opentelemetry.trace import get_current_span, StatusCode
    tipo = type(erro).__name__
    span = get_current_span()
    span.set_status(StatusCode.ERROR)
    span.set_attribute("error.type", tipo)
    _erros.add(1, {"stage": etapa, "error.type": tipo})
    logger.error("Falha na etapa %s: %s", etapa, tipo)


@contextmanager
def medir_etapa(etapa):
    if _tracer is None:
        yield
        return
    from opentelemetry.trace import StatusCode
    inicio = time.perf_counter()
    with _tracer.start_as_current_span(etapa, record_exception=False,
                                       set_status_on_exception=False) as span:
        try:
            yield
        except Exception as erro:
            registrar_falha(etapa, erro)
            raise
        finally:
            resultado = "error" if span.status.status_code == StatusCode.ERROR else "ok"
            _duracao.record(time.perf_counter() - inicio, {"stage": etapa, "outcome": resultado})
