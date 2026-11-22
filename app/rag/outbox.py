"""RAG document-event outbox and Kafka record handling. Mirrors app/ai/kafka.py.

created, updated and deleted rows go to the events topic; dead letters to its DLQ.
Each sink (Kafka here, SQS in app/rag/sinks.py) claims and acknowledges its own
rag_outbox_deliveries row, so one sink down never holds back the other.
"""

import asyncio
import hashlib
import json
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from aiokafka.errors import KafkaError
from pydantic import ValidationError
from sqlalchemy import func, or_, select, update

from app.ai.kafka import (
    KAFKA_CLOSE_TIMEOUT,
    KAFKA_IO_TIMEOUT,
    MAX_EVENT_BYTES,
    Producer,
    extract_trace_headers,
)
from app.ai.processing import HandleResult, SessionFactory
from app.models import RAGDocument, RAGOutbox, RAGOutboxDelivery
from app.rag.config import RAGSettings
from app.rag.contracts import DocumentEvent, DocumentEventType, EventSink, ingest_kind
from app.rag.ingestion import Embedder, GraphExtractor, process_ingest
from app.telemetry import Telemetry


@dataclass(frozen=True)
class OutboxLease:
    sink: EventSink
    event_id: UUID
    document_id: UUID
    version: int
    attempt: int
    kind: str
    source_kind: str
    token: UUID
    created_at: datetime
    status: str
    error_code: str | None
    trace_context: dict[str, str]
    # This sink's publish count for the event, the current claim included.
    publish_attempts: int = 1


async def _send(
    producer: Producer, topic: str, *, value: bytes, key: bytes, telemetry: Telemetry | None
) -> Any:
    # Only a validated W3C traceparent travels with an event; never baggage or tracestate.
    headers = []
    if telemetry is not None:
        carrier = telemetry.capture_context()
        validated = extract_trace_headers(
            [(name, val.encode("ascii")) for name, val in carrier.items()]
        )
        headers = [(name, val.encode("ascii")) for name, val in validated.items()]
    return await producer.send_and_wait(topic, value=value, key=key, headers=headers or None)


async def claim_outbox(
    *,
    session_factory: SessionFactory,
    settings: RAGSettings,
    sink: EventSink = EventSink.KAFKA,
    lease_seconds: float = KAFKA_IO_TIMEOUT + KAFKA_CLOSE_TIMEOUT,
) -> OutboxLease | None:
    """Lease this sink's next due delivery. Only this sink's row is locked."""
    now_ = func.clock_timestamp()
    async with session_factory() as session:
        async with session.begin():
            row = (
                await session.execute(
                    select(RAGOutboxDelivery, RAGOutbox, RAGDocument)
                    .join(RAGOutbox, RAGOutbox.id == RAGOutboxDelivery.outbox_id)
                    .join(RAGDocument, RAGDocument.id == RAGOutbox.document_id)
                    .where(
                        RAGOutboxDelivery.sink == str(sink),
                        RAGOutboxDelivery.available_at <= now_,
                        or_(
                            RAGOutboxDelivery.lease_token.is_(None),
                            RAGOutboxDelivery.lease_expires_at <= now_,
                        ),
                        or_(
                            RAGOutboxDelivery.published_at.is_(None),
                            # Redispatch an ingest event whose consumer may have lost it.
                            (RAGOutbox.event_kind.in_(("created", "updated")))
                            & (RAGDocument.status == "pending")
                            & (RAGDocument.version == RAGOutbox.version)
                            & (RAGDocument.attempt == RAGOutbox.attempt)
                            & (RAGDocument.next_attempt_at <= now_)
                            & (RAGDocument.deadline_at > now_)
                            & (
                                RAGOutboxDelivery.last_published_at
                                <= now_ - timedelta(seconds=settings.ai_redispatch_seconds)
                            ),
                        ),
                    )
                    .order_by(RAGOutboxDelivery.available_at, RAGOutboxDelivery.outbox_id)
                    .limit(1)
                    .with_for_update(of=RAGOutboxDelivery, skip_locked=True)
                )
            ).one_or_none()
            if row is None:
                return None
            delivery, outbox, document = row
            now = await session.scalar(select(func.clock_timestamp()))
            delivery.lease_token = uuid4()
            delivery.lease_expires_at = now + timedelta(
                seconds=max(settings.ai_lease_seconds, lease_seconds)
            )
            delivery.publish_attempts += 1
            delivery.published_at = None
            lease = OutboxLease(
                sink=sink,
                event_id=outbox.id,
                document_id=document.id,
                version=outbox.version,
                attempt=outbox.attempt,
                kind=outbox.event_kind,
                source_kind=document.source_kind,
                token=delivery.lease_token,
                created_at=outbox.created_at,
                # A dead letter is a failed version: with an older version still
                # served, its code is in update_error_code.
                status="failed" if outbox.event_kind == "dead_letter" else document.status,
                error_code=document.error_code or document.update_error_code,
                trace_context=dict(document.trace_context),
                publish_attempts=delivery.publish_attempts,
            )
        return lease


def _outbox_fence(lease: OutboxLease):
    return (
        RAGOutboxDelivery.outbox_id == lease.event_id,
        RAGOutboxDelivery.sink == str(lease.sink),
        RAGOutboxDelivery.lease_token == lease.token,
        RAGOutboxDelivery.lease_expires_at > func.clock_timestamp(),
        RAGOutboxDelivery.published_at.is_(None),
    )


async def acknowledge_outbox(*, session_factory: SessionFactory, lease: OutboxLease) -> bool:
    async with session_factory() as session:
        result = await session.execute(
            update(RAGOutboxDelivery)
            .where(*_outbox_fence(lease))
            .values(
                published_at=func.clock_timestamp(),
                last_published_at=func.clock_timestamp(),
                lease_token=None,
                lease_expires_at=None,
            )
        )
        await session.commit()
        return result.rowcount == 1


async def defer_publication(*, session_factory: SessionFactory, lease: OutboxLease) -> None:
    async with session_factory() as session:
        await session.execute(
            update(RAGOutboxDelivery)
            .where(*_outbox_fence(lease))
            .values(
                lease_token=None,
                lease_expires_at=None,
                available_at=func.clock_timestamp() + timedelta(seconds=2),
            )
        )
        await session.commit()


def event_payload(lease: OutboxLease) -> bytes:
    # A dead letter reports the version's ingest event type plus its outcome.
    kind = ingest_kind(lease.version) if lease.kind == "dead_letter" else lease.kind
    event = DocumentEvent(
        version=1,
        event_id=lease.event_id,
        type=DocumentEventType(f"doc.{kind}"),
        document_id=lease.document_id,
        document_version=lease.version,
        attempt=lease.attempt,
        source_kind=lease.source_kind,
        created_at=lease.created_at,
    )
    data = event.model_dump(mode="json")
    if lease.kind == "dead_letter":
        data.update(status=lease.status, error_code=lease.error_code)
    return json.dumps(data, separators=(",", ":")).encode()


async def dispatch_once(
    *,
    session_factory: SessionFactory,
    settings: RAGSettings,
    producer: Producer,
    telemetry: Telemetry | None = None,
    batch_size: int = 10,
    stop: asyncio.Event | None = None,
) -> int:
    """Claim one row at a time, send with a deadline, then fence the broker ACK."""
    published = 0
    for _ in range(batch_size):
        if stop is not None and stop.is_set():
            break
        lease = await claim_outbox(session_factory=session_factory, settings=settings)
        if lease is None:
            break
        topic = (
            settings.kafka_rag_dead_letter_topic
            if lease.kind == "dead_letter"
            else settings.kafka_rag_event_topic
        )
        context = telemetry.extract_context(lease.trace_context) if telemetry else None
        span = (
            telemetry.span(
                "rag.dispatcher.publish",
                context=context,
                document_id=str(lease.document_id),
                event_id=str(lease.event_id),
                attempt=lease.attempt,
                topic=topic,
            )
            if telemetry
            else nullcontext()
        )
        started = time.monotonic()
        try:
            with span:
                async with asyncio.timeout(KAFKA_IO_TIMEOUT):
                    await _send(
                        producer,
                        topic,
                        value=event_payload(lease),
                        key=str(lease.document_id).encode(),
                        telemetry=telemetry,
                    )
        except (KafkaError, TimeoutError):
            if telemetry is not None:
                telemetry.increment("ai.publish", outcome="failed", topic=topic)
            await defer_publication(session_factory=session_factory, lease=lease)
            # Fail the dispatcher closed so it disposes the producer (see app/ai/kafka.py).
            raise
        if await acknowledge_outbox(session_factory=session_factory, lease=lease):
            published += 1
            if telemetry is not None:
                telemetry.increment("ai.publish", outcome="succeeded", topic=topic)
                telemetry.observe(
                    "ai.publish.duration",
                    time.monotonic() - started,
                    outcome="succeeded",
                    topic=topic,
                )
    return published


async def handle_record(
    *,
    session_factory: SessionFactory,
    settings: RAGSettings,
    record: Any,
    producer: Producer,
    embedder: Embedder,
    extractor: GraphExtractor | None = None,
    telemetry: Telemetry | None = None,
) -> HandleResult:
    """Malformed deliveries are acknowledged only after a privacy-safe dead-letter ACK."""
    raw = record.value
    carrier = extract_trace_headers(getattr(record, "headers", None)) if telemetry else {}
    try:
        if not isinstance(raw, bytes) or len(raw) > MAX_EVENT_BYTES:
            raise ValueError("event size")
        event = DocumentEvent.model_validate_json(raw)
    except (ValueError, ValidationError):
        payload = json.dumps(
            {
                "version": 1,
                "error_code": "malformed_event",
                "source_topic": record.topic,
                "source_partition": record.partition,
                "source_offset": record.offset,
                "payload_sha256": hashlib.sha256(
                    raw if isinstance(raw, bytes) else b""
                ).hexdigest(),
            },
            separators=(",", ":"),
        ).encode()
        async with asyncio.timeout(KAFKA_IO_TIMEOUT):
            await _send(
                producer,
                settings.kafka_rag_dead_letter_topic,
                value=payload,
                key=f"{record.topic}:{record.partition}:{record.offset}".encode(),
                telemetry=telemetry,
            )
        if telemetry is not None:
            telemetry.increment("ai.consume", outcome="malformed", topic=record.topic)
        return HandleResult.HANDLED
    context = telemetry.extract_context(carrier) if telemetry and carrier else None
    span = (
        telemetry.span(
            "rag.consume",
            context=context,
            document_id=str(event.document_id),
            event_id=str(event.event_id),
            event_type=str(event.type),
            document_version=event.document_version,
            attempt=event.attempt,
            topic=record.topic,
            partition=record.partition,
            offset=record.offset,
        )
        if telemetry
        else nullcontext()
    )
    with span:
        result = await process_ingest(
            session_factory=session_factory,
            settings=settings,
            event=event,
            embedder=embedder,
            extractor=extractor,
            telemetry=telemetry,
        )
    if result == HandleResult.HANDLED and telemetry is not None:
        telemetry.increment("ai.consume", outcome="handled", topic=record.topic)
    return result
