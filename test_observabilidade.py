"""Smoke test sem ONNX, câmera, imagens ou OCR externo."""
import logging
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests
from flask import Flask
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest

import observabilidade as telemetry

os.environ.pop("BOL_ENVIRONMENT", None)
os.environ.pop("BOL_OBSERVABILITY_ENABLED", None)
logging.getLogger("opentelemetry.exporter").setLevel(logging.CRITICAL)

for setting in ({"BOL_ENVIRONMENT": "Production"}, {"BOL_OBSERVABILITY_ENABLED": "false"}):
    environment = {**os.environ, **setting, "BOL_OTLP_ENDPOINT": "invalid"}
    subprocess.run([sys.executable, "-c", "from flask import Flask; from observabilidade import configurar_observabilidade; a=Flask('off'); configurar_observabilidade(a); assert 'bol_observability' not in a.extensions"], env=environment, cwd=Path(__file__).resolve().parent, check=True)
os.environ["BOL_OTLP_ENDPOINT"] = "https://example.com"
try:
    telemetry.configurar_observabilidade(Flask("invalid"))
    raise AssertionError("Remote endpoint accepted")
except ValueError:
    pass

received = []


class Receiver(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        received.append((self.path, body))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-protobuf")
        self.end_headers()


receiver = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
threading.Thread(target=receiver.serve_forever, daemon=True).start()
os.environ["BOL_OTLP_ENDPOINT"] = f"http://127.0.0.1:{receiver.server_port}"
app = Flask("smoke")
telemetry.configurar_observabilidade(app)
telemetry.configurar_observabilidade(app)  # Idempotence.


@app.get("/smoke/<int:item>")
@telemetry.medir_etapa("teste.etapa")
def success(item):
    def work():
        with telemetry.medir_etapa("teste.thread"):
            pass
    worker = threading.Thread(target=work)
    worker.start()
    worker.join()
    requests.get(os.environ["BOL_OTLP_ENDPOINT"] + "/dependency", timeout=2)
    return {"ok": item}


@app.get("/failure")
@telemetry.medir_etapa("teste.erro")
def failure():
    raise ValueError("private-value")


@app.get("/caught")
@telemetry.medir_etapa("teste.tratado")
def caught():
    telemetry.registrar_falha("teste.tratado", ValueError("private-value"))
    return {"ok": False}


client = app.test_client()
trace_id = "12345678901234567890123456789012"
assert client.get("/smoke/7", headers={"traceparent": f"00-{trace_id}-1234567890123456-01"}).json == {"ok": 7}
assert client.get("/failure").status_code == 500
assert client.get("/caught").status_code == 200
providers = app.extensions["bol_observability"]
for provider in providers.values():
    assert provider.force_flush()

spans, metrics, logs = [], [], []
for path, body in received:
    if path == "/v1/traces":
        data = ExportTraceServiceRequest.FromString(body)
        spans.extend(s for resource in data.resource_spans for scope in resource.scope_spans for s in scope.spans)
    elif path == "/v1/metrics":
        data = ExportMetricsServiceRequest.FromString(body)
        metrics.extend(m for resource in data.resource_metrics for scope in resource.scope_metrics for m in scope.metrics)
    elif path == "/v1/logs":
        data = ExportLogsServiceRequest.FromString(body)
        logs.extend(log for resource in data.resource_logs for scope in resource.scope_logs for log in scope.log_records)

assert {"/v1/traces", "/v1/metrics", "/v1/logs"} <= {path for path, _ in received}
assert any(s.name == "teste.thread" and s.trace_id.hex() == trace_id for s in spans)
assert any(s.kind == 3 and s.trace_id.hex() == trace_id for s in spans)  # Outgoing HTTP.
assert any(log.trace_id.hex() == trace_id for log in logs)
assert all("private-value" not in str(span) for span in spans if span.name.startswith("teste."))
assert all("private-value" not in str(log) for log in logs)
names = {metric.name for metric in metrics}
assert {"http.server.request.duration", "bol.python.stage.duration", "bol.python.stage.errors",
        "process.memory.usage", "process.cpu.time", "process.cpu.utilization", "process.thread.count", "cpython.gc.collections"} <= names, names
duration = next(m for m in metrics if m.name == "bol.python.stage.duration")
assert any({a.key: a.value.string_value for a in point.attributes}.get("outcome") == "error"
           for point in duration.histogram.data_points)

receiver.shutdown()
receiver.server_close()
assert client.get("/caught").status_code == 200
for provider in providers.values():
    provider.shutdown()
print("PASS: three OTLP signals, HTTP/stage/process metrics, errors, traceparent, threads, HTTP client, privacy, disabled mode and collector outage.")
