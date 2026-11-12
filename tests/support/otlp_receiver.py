"""Private, bounded OTLP HTTP/protobuf capture for local QA (not a hosted UI)."""

import argparse
import asyncio
import zlib
from collections import deque
from contextlib import suppress

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from google.protobuf.message import DecodeError
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
    ExportMetricsServiceResponse,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)

MAX_BODY = 2 * 1024 * 1024
MAX_SPANS = 2000
MAX_METRICS = 1000


def value_of(value):
    kind = value.WhichOneof("value")
    if kind == "array_value":
        return [value_of(entry) for entry in value.array_value.values[:32]]
    if kind == "kvlist_value":
        return attributes_of(value.kvlist_value.values)
    if kind == "bytes_value":
        return value.bytes_value[:128].hex()
    if kind is None:
        return None
    result = getattr(value, kind)
    return result[:4096] if isinstance(result, str) else result


def attributes_of(attributes) -> dict:
    return {entry.key[:256]: value_of(entry.value) for entry in attributes[:128]}


async def read_body(request: Request) -> bytes | None:
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_BODY:
            return None
    encoding = request.headers.get("content-encoding", "identity")
    if encoding == "gzip":
        try:
            decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
            decoded = decoder.decompress(body, MAX_BODY + 1)
            if len(decoded) > MAX_BODY or not decoder.eof or decoder.unused_data:
                return None
            return decoded
        except zlib.error:
            return None
    return bytes(body) if encoding == "identity" else None


def create_app() -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    spans: deque[dict] = deque(maxlen=MAX_SPANS)
    metrics: deque[dict] = deque(maxlen=MAX_METRICS)
    lock = asyncio.Lock()
    counts = {"spans": 0, "metrics": 0, "dropped": 0}

    @app.get("/health/live")
    async def live() -> dict:
        return {"status": "ok"}

    @app.post("/__reset")
    async def reset() -> dict:
        async with lock:
            spans.clear()
            metrics.clear()
            counts.update(spans=0, metrics=0, dropped=0)
        return {"status": "reset"}

    @app.get("/__state")
    async def inspect(request: Request) -> Response:
        try:
            limit = int(request.query_params.get("limit", "2000"))
        except ValueError:
            return JSONResponse({"error": "invalid_limit"}, status_code=400)
        if not 1 <= limit <= MAX_SPANS:
            return JSONResponse({"error": "invalid_limit"}, status_code=400)
        channel = request.query_params.get("channel")
        service = request.query_params.get("service")
        trace_id = request.query_params.get("trace_id")
        async with lock:
            selected = [
                span
                for span in spans
                if (channel is None or span["channel"] == channel)
                and (service is None or span["resource"].get("service.name") == service)
                and (trace_id is None or span["trace_id"] == trace_id)
            ]
            return JSONResponse(
                {
                    "spans": selected[-limit:],
                    "metrics": list(metrics)[-limit:],
                    "received_spans": counts["spans"],
                    "received_metrics": counts["metrics"],
                    "dropped": counts["dropped"],
                }
            )

    @app.post("/v1/traces")
    @app.post("/{channel}/v1/traces")
    async def traces(request: Request, channel: str = "direct") -> Response:
        if channel not in {"direct", "langfuse", "langsmith"}:
            return JSONResponse({"error": "invalid_channel"}, status_code=404)
        body = await read_body(request)
        if body is None:
            return JSONResponse({"error": "invalid_body"}, status_code=413)
        message = ExportTraceServiceRequest()
        try:
            message.ParseFromString(body)
        except DecodeError:
            return JSONResponse({"error": "invalid_protobuf"}, status_code=400)
        async with lock:
            for resource in message.resource_spans:
                resource_attributes = attributes_of(resource.resource.attributes)
                for scope in resource.scope_spans:
                    for span in scope.spans:
                        counts["spans"] += 1
                        if len(spans) == MAX_SPANS:
                            counts["dropped"] += 1
                        spans.append(
                            {
                                "channel": channel,
                                "resource": resource_attributes,
                                "scope": scope.scope.name[:256],
                                "name": span.name[:256],
                                "trace_id": span.trace_id.hex(),
                                "span_id": span.span_id.hex(),
                                "parent_span_id": span.parent_span_id.hex(),
                                "attributes": attributes_of(span.attributes),
                                "status": span.status.code,
                                "status_description": span.status.message[:4096],
                                "events": [
                                    {
                                        "name": event.name[:256],
                                        "attributes": attributes_of(event.attributes),
                                    }
                                    for event in span.events[:32]
                                ],
                                "links": [
                                    {"trace_id": link.trace_id.hex(), "span_id": link.span_id.hex()}
                                    for link in span.links[:32]
                                ],
                            }
                        )
        return Response(
            ExportTraceServiceResponse().SerializeToString(), media_type="application/x-protobuf"
        )

    @app.post("/v1/metrics")
    @app.post("/{channel}/v1/metrics")
    async def receive_metrics(request: Request, channel: str = "direct") -> Response:
        if channel not in {"direct", "langfuse", "langsmith"}:
            return JSONResponse({"error": "invalid_channel"}, status_code=404)
        body = await read_body(request)
        if body is None:
            return JSONResponse({"error": "invalid_body"}, status_code=413)
        message = ExportMetricsServiceRequest()
        try:
            message.ParseFromString(body)
        except DecodeError:
            return JSONResponse({"error": "invalid_protobuf"}, status_code=400)
        async with lock:
            for resource in message.resource_metrics:
                for scope in resource.scope_metrics:
                    for metric in scope.metrics:
                        counts["metrics"] += 1
                        if len(metrics) == MAX_METRICS:
                            counts["dropped"] += 1
                        metrics.append(
                            {
                                "channel": channel,
                                "resource": attributes_of(resource.resource.attributes),
                                "name": metric.name[:256],
                                "type": metric.WhichOneof("data"),
                            }
                        )
        return Response(
            ExportMetricsServiceResponse().SerializeToString(), media_type="application/x-protobuf"
        )

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4318)
    args = parser.parse_args()
    with suppress(KeyboardInterrupt):
        uvicorn.run(app, host=args.host, port=args.port, access_log=False, log_level="critical")


if __name__ == "__main__":
    main()
