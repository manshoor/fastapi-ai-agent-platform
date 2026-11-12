"""Privacy, propagation, bounded export, and real OTLP/protobuf HTTP regressions."""

import asyncio
import importlib.util
import json
import logging
import socket
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
import uvicorn
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics.export import InMemoryMetricReader, PeriodicExportingMetricReader
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import app.telemetry as module
from app.telemetry import Telemetry, _BoundedProcessor, _RedactExportLogs, configure_telemetry


@pytest.fixture
async def receiver():
    path = Path(__file__).parent / "support/otlp_receiver.py"
    spec = importlib.util.spec_from_file_location("otlp_receiver_regression", path)
    imported = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = imported
    spec.loader.exec_module(imported)
    app = imported.create_app()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    server = uvicorn.Server(
        uvicorn.Config(app, access_log=False, log_level="critical", lifespan="off")
    )
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(3):
            while not server.started:
                if task.done():
                    await task
                await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 3)
        sock.close()


def metric_points(reader):
    data = reader.get_metrics_data()
    return {
        metric.name: list(metric.data.data_points)
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
    }


def test_span_allowlist_exception_redaction_and_no_global_provider():
    global_provider = trace.get_tracer_provider()
    exporter = InMemorySpanExporter()
    telemetry = Telemetry("privacy-test", trace_exporter=exporter)
    secret = "PRIVATE_PROMPT api-key JWT source description"
    job_id = uuid4()
    try:
        with pytest.raises(RuntimeError):
            with telemetry.span(
                "ai.worker.process",
                job_id=job_id,
                attempt=1,
                model="synthetic-model",
                source_title=secret,
                authorization=secret,
                **{"gen_ai.prompt": secret, "error_code": secret},
            ):
                raise RuntimeError(secret)
        with telemetry.span(secret, source_title=secret):
            pass
        assert telemetry.provider.force_flush(timeout_millis=1000)
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.attributes == {"job_id": str(job_id), "attempt": 1, "model": "synthetic-model"}
        assert span.events == ()
        assert span.status.status_code.name == "ERROR"
        assert span.status.description is None
        assert secret not in span.to_json()
        assert trace.get_tracer_provider() is global_provider
    finally:
        telemetry.shutdown()


def test_context_links_request_to_worker_without_baggage():
    exporter = InMemorySpanExporter()
    telemetry = Telemetry("propagation-test", trace_exporter=exporter)
    try:
        with telemetry.span("ai.request"):
            headers = telemetry.capture_context()
        assert set(headers) == {"traceparent"}
        headers.update(baggage="token=private", tracestate="private=secret")
        context = telemetry.extract_context(headers)
        with telemetry.span("ai.worker.process", context=context):
            assert set(telemetry.capture_context()) == {"traceparent"}
        assert telemetry.provider.force_flush(1000)
        request, worker = exporter.get_finished_spans()
        assert request.context.trace_id == worker.context.trace_id
        assert worker.parent.span_id == request.context.span_id
        assert not worker.context.trace_state
        invalid = telemetry.extract_context({"traceparent": "not-a-valid-header"})
        with telemetry.span("ai.request", context=invalid):
            pass
    finally:
        telemetry.shutdown()


def test_metrics_are_allowlisted_low_cardinality_and_nonnegative():
    reader = InMemoryMetricReader()
    telemetry = Telemetry("metrics-test", metric_readers=[reader])
    try:
        telemetry.increment(
            "ai.jobs", status="pending", job_id=uuid4(), offset=1000, prompt="secret"
        )
        telemetry.increment("ai.outcomes", outcome="succeeded")
        telemetry.increment("ai.provider.calls", model="synthetic-model")
        telemetry.increment("ai.export.drops", 2)
        telemetry.observe("ai.provider.duration", 0.125, model="synthetic-model")
        telemetry.observe("ai.queue.depth", 3, status="pending")
        telemetry.observe("ai.queue.wait", 0.5)
        telemetry.observe(
            "ai.consumer.lag", 7, topic="synthetic-topic", partition=0, model="discard"
        )
        telemetry.observe("ai.worker.utilization", 0.5, topic="synthetic-topic", partition=0)
        telemetry.observe("ai.worker.utilization", 1.1, topic="synthetic-topic", partition=0)
        telemetry.observe("ai.consumer.lag", float("inf"), topic="synthetic-topic", partition=0)
        telemetry.increment("unknown_secret_metric")
        telemetry.increment("ai.jobs", -1)
        telemetry.observe("ai.queue.wait", float("nan"))
        points = metric_points(reader)
        assert points["ai.jobs"][0].value == 1
        assert points["ai.jobs"][0].attributes == {"status": "pending"}
        assert points["ai.export.drops"][0].value == 2
        assert points["ai.provider.duration"][0].sum == 0.125
        assert points["ai.queue.depth"][0].value == 3
        assert points["ai.consumer.lag"][0].value == 7
        assert points["ai.consumer.lag"][0].attributes == {
            "topic": "synthetic-topic",
            "partition": 0,
        }
        assert points["ai.worker.utilization"][0].value == 0.5
        assert "unknown_secret_metric" not in points
    finally:
        telemetry.shutdown()


def test_bounded_queue_reports_drops_without_blocking():
    entered = threading.Event()
    release = threading.Event()
    losses = []

    class BlockingExporter:
        def export(self, spans):
            entered.set()
            release.wait(2)
            return SpanExportResult.FAILURE

        def shutdown(self):
            pass

    processor = _BoundedProcessor(BlockingExporter(), losses.append, capacity=1)
    processor.on_end(object())
    assert entered.wait(1)
    started = time.monotonic()
    processor.on_end(object())
    processor.on_end(object())
    assert time.monotonic() - started < 0.1
    assert losses == [1]
    release.set()
    assert processor.force_flush(1000)
    processor.shutdown()
    assert sum(losses) == 3  # one queue loss, two failed exported spans


def test_exporter_diagnostics_remove_body_exception_and_stack():
    record = logging.LogRecord(
        "exporter",
        logging.ERROR,
        "",
        1,
        "Provider response: %s",
        ("PRIVATE_SECRET",),
        (RuntimeError, RuntimeError("PRIVATE_SECRET"), None),
    )
    record.exc_text = "PRIVATE_SECRET traceback"
    record.stack_info = "PRIVATE_SECRET stack"
    assert _RedactExportLogs().filter(record)
    assert record.getMessage() == "Telemetry export diagnostic"
    assert record.exc_info is None and record.exc_text is None and record.stack_info is None


def test_configure_is_idempotent_and_disabled_without_explicit_endpoint(monkeypatch):
    for key in (
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(module, "_INSTANCE", None)
    telemetry = configure_telemetry("first-service")
    try:
        assert configure_telemetry("second-service") is telemetry
        assert module._endpoint("traces") is None and module._endpoint("metrics") is None
        assert telemetry.provider.resource.attributes["service.name"] == "first-service"
        assert telemetry.provider._active_span_processor._span_processors == ()
    finally:
        telemetry.shutdown()
        telemetry.shutdown()
    replacement = configure_telemetry("replacement-service")
    try:
        assert replacement is not telemetry
        assert not replacement._closed
        assert replacement.provider.resource.attributes["service.name"] == "replacement-service"
    finally:
        replacement.shutdown()


@pytest.mark.parametrize(
    "endpoint",
    [
        "file:///private",
        "http://user:secret@localhost:4318",
        "http://localhost:4318?token=secret",
        "not-a-url",
    ],
)
def test_otlp_endpoint_rejects_credential_and_unsupported_urls(monkeypatch, endpoint):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", endpoint)
    with pytest.raises(ValueError):
        module._endpoint("traces")


async def test_actual_otlp_protobuf_spans_and_metrics_over_http(receiver):
    exporter = OTLPSpanExporter(endpoint=f"{receiver}/v1/traces", timeout=1, headers={})
    reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=f"{receiver}/v1/metrics", timeout=1, headers={}),
        export_interval_millis=60000,
        export_timeout_millis=1000,
    )
    telemetry = Telemetry("wire-test", trace_exporter=exporter, metric_readers=[reader])
    secret = "PRIVATE_WIRE_SECRET"
    try:
        with telemetry.span("ai.request", job_id=uuid4(), prompt=secret):
            context = telemetry.capture_context()
        with pytest.raises(RuntimeError):
            with telemetry.span(
                "ai.generate",
                context=telemetry.extract_context(context),
                model="synthetic-model",
                response=secret,
            ):
                raise RuntimeError(secret)
        telemetry.increment("ai.jobs", status="pending")
        telemetry.increment("ai.outcomes", outcome="succeeded")
        telemetry.observe("ai.queue.wait", 0.25)
        assert await asyncio.to_thread(telemetry.provider.force_flush, 1500)
        assert await asyncio.to_thread(telemetry.meter_provider.force_flush, 1500)
        async with httpx.AsyncClient(base_url=receiver, trust_env=False) as client:
            state = (await client.get("/__state")).json()
        assert state["received_spans"] == 2
        request, generation = state["spans"]
        assert generation["trace_id"] == request["trace_id"]
        assert generation["parent_span_id"] == request["span_id"]
        assert generation["status"] == 2
        assert generation["events"] == [] and generation["status_description"] == ""
        assert generation["attributes"] == {"model": "synthetic-model"}
        assert secret not in json.dumps(state)
        assert {metric["name"] for metric in state["metrics"]} >= {
            "ai.jobs",
            "ai.outcomes",
            "ai.queue.wait",
        }
    finally:
        await asyncio.to_thread(telemetry.shutdown)


async def test_shutdown_is_bounded_with_unreachable_local_receiver(monkeypatch):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    endpoint = f"http://127.0.0.1:{sock.getsockname()[1]}/v1/traces"
    monkeypatch.setattr(module, "_INSTANCE", None)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", endpoint)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", raising=False)
    telemetry = configure_telemetry("unreachable-test")
    with telemetry.span("ai.generate"):
        pass
    started = time.monotonic()
    try:
        await asyncio.to_thread(telemetry.shutdown)
        assert time.monotonic() - started < 3.5
    finally:
        sock.close()
