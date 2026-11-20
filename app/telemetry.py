"""Application-owned, content-free OTLP telemetry; exporting is opt-in.

Only this facade creates spans/metrics. Never pass prompts, output, headers, exception
objects, or account identifiers. Unknown names/attributes are discarded. No global
OpenTelemetry provider is replaced, and no SDK/network auto-instrumentation is enabled.
"""

import atexit
import logging
import math
import os
import queue
import re
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from urllib.parse import urlsplit
from uuid import UUID

from opentelemetry import context as otel_context
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagators.textmap import CarrierT
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricExportResult, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

_SPANS = frozenset(
    {
        "ai.request",
        "ai.enqueue",
        "ai.claim",
        "ai.generate",
        "ai.embed",
        "ai.persist",
        "ai.publish",
        "ai.consume",
        "ai.recover",
        "ai.purge",
        "ai.job",
        "ai.dispatch",
        "ai.worker.process",
        "ai.dispatcher.publish",
        "ai.submit",
        "ai.poll",
        "rag.upload",
        "rag.update",
        "rag.consume",
        "rag.dispatcher.publish",
        "rag.ingest",
        "rag.parse",
        "rag.chunk",
        "rag.store",
        "rag.query",
        "rag.retrieve",
        "assistant.turn",
        "assistant.tool",
        "assistant.confirm",
    }
)
_COUNTERS = frozenset(
    {
        "ai.jobs",
        "ai.requests",
        "ai.provider.calls",
        "ai.retries",
        "ai.outcomes",
        "ai.publish",
        "ai.consume",
        "ai.export.drops",
        "rag.requests",
        "rag.documents",
        "rag.queries",
        "assistant.turns",
        "assistant.tool_calls",
    }
)
_HISTOGRAMS = frozenset(
    {
        "ai.provider.duration",
        "ai.job.duration",
        "ai.queue.wait",
        "ai.queue.depth",
        "ai.publish.duration",
        "rag.ingest.duration",
        "rag.query.duration",
        "assistant.turn.duration",
    }
)
_NUMBERS = frozenset(
    {
        "attempt",
        "lease_generation",
        "document_version",
        "partition",
        "offset",
        "queue_depth",
        "input_tokens",
        "output_tokens",
        "duration_ms",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "chunk_count",
        "page_count",
        "ocr_pages",
        "ocr_truncated",
        "retrieved_count",
        "top_k",
        "model_calls",
        "tool_calls",
    }
)
_IDS = frozenset(
    {"job_id", "event_id", "item_id", "document_id", "query_id", "turn_id", "action_id"}
)
# Queue-depth gauges keep their metric labels; other gauges keep topic/partition only.
_DEPTH_GAUGES = frozenset({"ai.queue.depth", "rag.queue.depth"})
_TAGS = frozenset(
    {
        "model",
        "prompt_version",
        "schema_version",
        "status",
        "outcome",
        "error_code",
        "provider",
        "operation",
        "event_type",
        "topic",
        "sink",
        "queue",
        "gen_ai.system",
        "gen_ai.operation.name",
        "gen_ai.request.model",
        "gen_ai.response.model",
        "messaging.system",
        "messaging.destination.name",
        "messaging.operation.name",
    }
)
_SAFE_TAG = re.compile(r"[A-Za-z0-9_.:/-]{1,128}\Z")
_PROPAGATOR = TraceContextTextMapPropagator()
_LOCK = threading.Lock()
_INSTANCE = None


def _attributes(attrs, *, metric=False):
    result = {}
    for key, value in attrs.items():
        if metric and key not in {
            "model",
            "prompt_version",
            "schema_version",
            "status",
            "outcome",
            "error_code",
            "provider",
            "operation",
            "topic",
            "sink",
            "partition",
        }:
            continue
        if key in _NUMBERS and isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(value) and value >= 0:
                result[key] = value
        elif key in _IDS and not metric:
            try:
                result[key] = str(UUID(str(value)))
            except (ValueError, TypeError, AttributeError):
                pass
        elif key in _TAGS and isinstance(value, str) and _SAFE_TAG.fullmatch(value):
            result[key] = value
    return result


def _endpoint(signal):
    # No exporter is constructed merely because the OTEL library has a localhost default.
    endpoint = os.environ.get(f"OTEL_EXPORTER_OTLP_{signal.upper()}_ENDPOINT")
    if not endpoint:
        base = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        endpoint = f"{base.rstrip('/')}/v1/{signal}" if base else None
    if endpoint:
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Invalid OTLP endpoint")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("OTLP endpoint must not contain credentials, query, or fragment")
    return endpoint


class _RedactExportLogs(logging.Filter):
    def filter(self, record):
        # The OTLP SDK otherwise logs receiver response bodies and traceback details.
        record.msg = "Telemetry export diagnostic"
        record.args = ()
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return True


_EXPORT_LOG_FILTER = _RedactExportLogs()


class _MetricsExporter(OTLPMetricExporter):
    def __init__(self, endpoint, dropped):
        super().__init__(endpoint=endpoint, timeout=1, headers={})
        self.dropped = dropped

    def export(self, metrics_data, timeout_millis=1000, **kwargs):
        try:
            result = super().export(metrics_data, timeout_millis=timeout_millis, **kwargs)
        except Exception:
            result = MetricExportResult.FAILURE
        if result != MetricExportResult.SUCCESS:
            self.dropped(1)
        return result


class _BoundedProcessor(SpanProcessor):
    """Bounded background export with explicit, observable queue/export losses."""

    def __init__(self, exporter, dropped, capacity=2048):
        self.exporter = exporter
        self.dropped = dropped
        self.pending = queue.Queue(maxsize=capacity)
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, name="ai-otlp", daemon=True)
        self.thread.start()

    def on_start(self, span, parent_context=None):
        pass

    def on_end(self, span):
        if self.stopping.is_set():
            self.dropped(1)
            return
        try:
            self.pending.put_nowait(span)
        except queue.Full:
            self.dropped(1)

    def _run(self):
        try:
            while not self.stopping.is_set() or not self.pending.empty():
                try:
                    first = self.pending.get(timeout=0.05)
                except queue.Empty:
                    continue
                batch = [first]
                while len(batch) < 128:
                    try:
                        batch.append(self.pending.get_nowait())
                    except queue.Empty:
                        break
                try:
                    if self.exporter.export(batch) != SpanExportResult.SUCCESS:
                        self.dropped(len(batch))
                except Exception:
                    # Exporter failures must never expose response/exception contents.
                    self.dropped(len(batch))
                finally:
                    for _ in batch:
                        self.pending.task_done()
        finally:
            self.exporter.shutdown()

    def force_flush(self, timeout_millis=2000):
        deadline = time.monotonic() + timeout_millis / 1000
        while self.pending.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)
        return not self.pending.unfinished_tasks

    def shutdown(self):
        self.stopping.set()
        self.thread.join(timeout=2)
        # A daemon exporter cannot delay process shutdown, even on an unhealthy destination.
        remaining = self.pending.qsize()
        if remaining:
            self.dropped(remaining)


class Telemetry:
    def __init__(self, service, *, trace_exporter=None, metric_readers=()):
        resource = Resource({"service.name": service, "service.namespace": "item-rewrite"})
        self.provider = TracerProvider(resource=resource, shutdown_on_exit=False)
        self.tracer = self.provider.get_tracer("app.ai", "1")
        self.meter_provider = MeterProvider(
            resource=resource, metric_readers=metric_readers, shutdown_on_exit=False
        )
        meter = self.meter_provider.get_meter("app.ai", "1")
        self.counters = {name: meter.create_counter(name) for name in _COUNTERS}
        self.histograms = {
            name: meter.create_histogram(name, unit="s")
            for name in _HISTOGRAMS
            if name != "ai.queue.depth"
        }
        self.gauges = {
            name: meter.create_gauge(name, unit="1")
            for name in (*_DEPTH_GAUGES, "ai.consumer.lag", "ai.worker.utilization")
        }
        self._closed = False
        self._shutdown_lock = threading.Lock()
        if trace_exporter is not None:
            self.provider.add_span_processor(
                _BoundedProcessor(
                    trace_exporter, lambda count: self.increment("ai.export.drops", count)
                )
            )

    @contextmanager
    def span(self, name, *, context=None, **attrs):
        if name not in _SPANS or self._closed:
            yield None
            return
        with self.tracer.start_as_current_span(
            name,
            context=context,
            attributes=_attributes(attrs),
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                yield None
            except BaseException:
                # Status only: neither events nor status descriptions can carry exception text.
                span.set_status(Status(StatusCode.ERROR))
                raise

    def annotate(self, **attrs):
        """Attach only approved metadata to the current span, without exposing a raw span."""
        if not self._closed:
            from opentelemetry.trace import get_current_span

            get_current_span().set_attributes(_attributes(attrs))

    def capture_context(self) -> dict[str, str]:
        carrier = {}
        _PROPAGATOR.inject(carrier)
        # Deliberately omit tracestate and baggage, even if an upstream client supplied them.
        return {key: value for key, value in carrier.items() if key == "traceparent"}

    def extract_context(self, headers: Mapping[str, str]) -> otel_context.Context:
        carrier: CarrierT = {"traceparent": headers.get("traceparent", "")}
        return _PROPAGATOR.extract(carrier)

    def increment(self, name, value=1, **attrs):
        """Add a nonnegative counter; metric dimensions exclude per-job identifiers."""
        if name in self.counters and math.isfinite(value) and value >= 0:
            self.counters[name].add(value, _attributes(attrs, metric=True))

    def observe(self, name, value, **attrs):
        """Observe latency seconds, depth/lag counts, or worker utilization in [0, 1]."""
        if math.isfinite(value) and value >= 0:
            if name in self.gauges:
                if name == "ai.worker.utilization" and value > 1:
                    return
                labels = _attributes(attrs, metric=True)
                if name not in _DEPTH_GAUGES:
                    labels = {
                        key: val for key, val in labels.items() if key in {"topic", "partition"}
                    }
                self.gauges[name].set(value, labels)
            elif name in self.histograms:
                self.histograms[name].record(value, _attributes(attrs, metric=True))

    def shutdown(self):
        global _INSTANCE
        with _LOCK:
            with self._shutdown_lock:
                if self._closed:
                    return
                self._closed = True
            try:
                self.provider.shutdown()
                self.meter_provider.shutdown(timeout_millis=1000)
            finally:
                if _INSTANCE is self:
                    _INSTANCE = None


def configure_telemetry(service: str) -> Telemetry:
    """Idempotent per process; only explicit OTEL endpoints activate external I/O."""
    global _INSTANCE
    with _LOCK:
        if _INSTANCE is None:
            traces = _endpoint("traces")
            metrics = _endpoint("metrics")
            for logger_name in (
                "opentelemetry.exporter.otlp.proto.http.trace_exporter",
                "opentelemetry.exporter.otlp.proto.http.metric_exporter",
                "opentelemetry.sdk.metrics._internal.export",
            ):
                logging.getLogger(logger_name).addFilter(_EXPORT_LOG_FILTER)
            exporter = OTLPSpanExporter(endpoint=traces, timeout=1, headers={}) if traces else None
            readers = []
            owner = []
            if metrics:
                readers.append(
                    PeriodicExportingMetricReader(
                        _MetricsExporter(
                            metrics, lambda count: owner[0].increment("ai.export.drops", count)
                        ),
                        export_interval_millis=10000,
                        export_timeout_millis=1000,
                    )
                )
            _INSTANCE = Telemetry(service, trace_exporter=exporter, metric_readers=readers)
            owner.append(_INSTANCE)
            atexit.register(_INSTANCE.shutdown)
        return _INSTANCE
