"""Bounded Kafka I/O and an at-least-once, identity-stable transactional outbox."""

import asyncio
import hashlib
import json
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol
from uuid import UUID, uuid4

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError, ProducerClosed
from pydantic import ValidationError
from sqlalchemy import func, or_, select, update

from app.ai.config import AISettings
from app.ai.contracts import RewriteEvent
from app.ai.processing import HandleResult, ProviderFactory, SessionFactory, process_event
from app.models import AIJob, AIOutbox
from app.telemetry import Telemetry

KAFKA_IO_TIMEOUT = 10
KAFKA_CLOSE_TIMEOUT = 5
MAX_EVENT_BYTES = 4096


class Producer(Protocol):
    async def send_and_wait(
        self,
        topic: str,
        value: bytes,
        *,
        key: bytes,
        headers: list[tuple[str, bytes]] | None = None,
    ) -> Any: ...


@dataclass(frozen=True)
class OutboxLease:
    event_id: UUID
    job_id: UUID
    attempt: int
    kind: str
    token: UUID
    created_at: datetime
    status: str
    error_code: str | None
    trace_context: dict[str, str]


def extract_trace_headers(headers: list[tuple[str, bytes | None]] | None) -> dict[str, str]:
    """Accept one bounded W3C v00 traceparent; never propagate baggage or tracestate."""
    if not headers or len(headers) > 8:
        return {}
    traceparent = None
    total = 0
    for key, value in headers:
        if not isinstance(key, str) or len(key) > 128:
            return {}
        if value is not None and (not isinstance(value, bytes) or len(value) > 512):
            return {}
        total += len(key) + (len(value) if value is not None else 0)
        if total > 1024:
            return {}
        if key != "traceparent":
            continue
        if traceparent is not None or value is None or len(value) != 55:
            return {}
        try:
            traceparent = value.decode("ascii")
        except UnicodeDecodeError:
            return {}
        match = re.fullmatch(r"00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})", traceparent)
        if match is None or int(match[1], 16) == 0 or int(match[2], 16) == 0:
            return {}
    return {"traceparent": traceparent} if traceparent is not None else {}


async def _send(
    producer: Producer,
    topic: str,
    *,
    value: bytes,
    key: bytes,
    telemetry: Telemetry | None,
) -> Any:
    if telemetry is None:
        return await producer.send_and_wait(topic, value=value, key=key)
    context = telemetry.capture_context()
    headers = [
        (name, val.encode("ascii")) for name, val in context.items() if name == "traceparent"
    ]
    validated = extract_trace_headers(headers)
    return await producer.send_and_wait(
        topic,
        value=value,
        key=key,
        headers=[(name, val.encode("ascii")) for name, val in validated.items()],
    )


async def create_producer(settings: AISettings) -> AIOKafkaProducer:
    producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        enable_idempotence=True,
        acks="all",
        request_timeout_ms=5000,
        max_request_size=16384,
        max_batch_size=16384,
        linger_ms=0,
        retry_backoff_ms=250,
        metadata_max_age_ms=30000,
    )
    try:
        async with asyncio.timeout(KAFKA_IO_TIMEOUT):
            await producer.start()
    except BaseException:
        try:
            await close_producer(producer)
        except Exception:
            pass  # Preserve the startup failure; cleanup cannot authorize any publication.
        raise
    return producer


async def close_producer(producer: AIOKafkaProducer) -> None:
    """Idempotent Kafka delivery retries must never turn shutdown into an infinite flush."""
    try:
        async with asyncio.timeout(KAFKA_CLOSE_TIMEOUT):
            await producer.flush()
    except (TimeoutError, KafkaError):
        # aiokafka has no public abort API. With the pinned implementation explicitly fail
        # all queued/in-flight batches before cancelling its sender; cancellation alone does
        # not fail batch futures and stop() would otherwise leave an accumulator task alive.
        producer._message_accumulator.fail_all(ProducerClosed())
        sender = producer._sender.sender_task
        if sender is not None:
            sender.cancel()
        async with asyncio.timeout(KAFKA_CLOSE_TIMEOUT):
            await producer.client.close()
    finally:
        async with asyncio.timeout(KAFKA_CLOSE_TIMEOUT):
            await producer.stop()


async def claim_outbox(
    *, session_factory: SessionFactory, settings: AISettings
) -> OutboxLease | None:
    async with session_factory() as session:
        async with session.begin():
            row = (
                await session.execute(
                    select(AIOutbox, AIJob)
                    .join(AIJob, AIJob.id == AIOutbox.job_id)
                    .where(
                        AIOutbox.available_at <= func.clock_timestamp(),
                        or_(
                            AIOutbox.lease_token.is_(None),
                            AIOutbox.lease_expires_at <= func.clock_timestamp(),
                        ),
                        or_(
                            AIOutbox.published_at.is_(None),
                            (AIOutbox.event_kind == "requested")
                            & (AIJob.status == "pending")
                            & (AIJob.attempt == AIOutbox.attempt)
                            & (AIJob.next_attempt_at <= func.clock_timestamp())
                            & (AIJob.deadline_at > func.clock_timestamp())
                            & (
                                AIOutbox.last_published_at
                                <= func.clock_timestamp()
                                - timedelta(seconds=settings.ai_redispatch_seconds)
                            ),
                        ),
                    )
                    .order_by(AIOutbox.available_at, AIOutbox.id)
                    .limit(1)
                    .with_for_update(skip_locked=True)
                )
            ).one_or_none()
            if row is None:
                return None
            outbox, job = row
            now = await session.scalar(select(func.clock_timestamp()))
            outbox.lease_token = uuid4()
            outbox.lease_expires_at = now + timedelta(
                seconds=max(settings.ai_lease_seconds, KAFKA_IO_TIMEOUT + KAFKA_CLOSE_TIMEOUT),
            )
            outbox.publish_attempts += 1
            # A republish is unsettled until its own broker ACK is fenced into the DB.
            outbox.published_at = None
            lease = OutboxLease(
                outbox.id,
                job.id,
                outbox.attempt,
                outbox.event_kind,
                outbox.lease_token,
                outbox.created_at,
                job.status,
                job.error_code,
                dict(job.trace_context),
            )
        return lease


def _outbox_fence(lease: OutboxLease):
    return (
        AIOutbox.id == lease.event_id,
        AIOutbox.job_id == lease.job_id,
        AIOutbox.attempt == lease.attempt,
        AIOutbox.lease_token == lease.token,
        AIOutbox.lease_expires_at > func.clock_timestamp(),
        AIOutbox.published_at.is_(None),
    )


async def acknowledge_outbox(*, session_factory: SessionFactory, lease: OutboxLease) -> bool:
    async with session_factory() as session:
        result = await session.execute(
            update(AIOutbox)
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


async def _defer_publication(*, session_factory: SessionFactory, lease: OutboxLease) -> None:
    async with session_factory() as session:
        await session.execute(
            update(AIOutbox)
            .where(*_outbox_fence(lease))
            .values(
                lease_token=None,
                lease_expires_at=None,
                available_at=func.clock_timestamp() + timedelta(seconds=2),
            )
        )
        await session.commit()


def _payload(lease: OutboxLease) -> bytes:
    event = RewriteEvent(
        version=1,
        event_id=lease.event_id,
        job_id=lease.job_id,
        attempt=lease.attempt,
        created_at=lease.created_at,
    )
    data = event.model_dump(mode="json")
    if lease.kind == "dead_letter":
        data.update(status=lease.status, error_code=lease.error_code)
    return json.dumps(data, separators=(",", ":")).encode()


async def dispatch_once(
    *,
    session_factory: SessionFactory,
    settings: AISettings,
    producer: Producer,
    telemetry: Telemetry | None = None,
    batch_size: int = 100,
    stop: asyncio.Event | None = None,
) -> int:
    """Claim one row at a time, commit, send with a deadline, then fence the broker ACK."""
    published = 0
    for _ in range(batch_size):
        if stop is not None and stop.is_set():
            break
        lease = await claim_outbox(session_factory=session_factory, settings=settings)
        if lease is None:
            break
        topic = (
            settings.kafka_request_topic
            if lease.kind == "requested"
            else settings.kafka_dead_letter_topic
        )
        context = telemetry.extract_context(lease.trace_context) if telemetry else None
        span = (
            telemetry.span(
                "ai.dispatcher.publish",
                context=context,
                job_id=str(lease.job_id),
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
                        value=_payload(lease),
                        key=str(lease.job_id).encode(),
                        telemetry=telemetry,
                    )
        except (KafkaError, TimeoutError):
            if telemetry is not None:
                telemetry.increment("ai.publish", outcome="failed", topic=topic)
                telemetry.observe(
                    "ai.publish.duration",
                    time.monotonic() - started,
                    outcome="failed",
                    topic=topic,
                )
            await _defer_publication(session_factory=session_factory, lease=lease)
            # Fail the owning runtime closed so it disposes this producer. Continuing with
            # an idempotent producer whose broker ACK is unknown can accumulate batches
            # indefinitely: aiokafka intentionally disables delivery expiry in this mode.
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
    settings: AISettings,
    record: Any,
    producer: Producer,
    provider_factory: ProviderFactory,
    telemetry: Telemetry | None = None,
) -> HandleResult:
    """Malformed deliveries are acknowledged only AFTER a privacy-safe DLQ broker ACK."""
    raw = record.value
    carrier = extract_trace_headers(getattr(record, "headers", None)) if telemetry else {}
    try:
        if not isinstance(raw, bytes) or len(raw) > MAX_EVENT_BYTES:
            raise ValueError("event size")
        event = RewriteEvent.model_validate_json(raw)
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
        context = telemetry.extract_context(carrier) if telemetry else None
        span = (
            telemetry.span(
                "ai.consume",
                context=context,
                topic=record.topic,
                partition=record.partition,
                offset=record.offset,
                error_code="malformed_event",
            )
            if telemetry
            else nullcontext()
        )
        with span:
            async with asyncio.timeout(KAFKA_IO_TIMEOUT):
                await _send(
                    producer,
                    settings.kafka_dead_letter_topic,
                    value=payload,
                    key=f"{record.topic}:{record.partition}:{record.offset}".encode(),
                    telemetry=telemetry,
                )
        if telemetry is not None:
            telemetry.increment("ai.consume", outcome="malformed", topic=record.topic)
        return HandleResult.HANDLED
    if telemetry is not None and not carrier:
        async with session_factory() as session:
            stored = await session.scalar(
                select(AIJob.trace_context).where(
                    AIJob.id == event.job_id,
                )
            )
        carrier = stored or {}
    context = telemetry.extract_context(carrier) if telemetry else None
    span = (
        telemetry.span(
            "ai.consume",
            context=context,
            job_id=str(event.job_id),
            event_id=str(event.event_id),
            attempt=event.attempt,
            topic=record.topic,
            partition=record.partition,
            offset=record.offset,
        )
        if telemetry
        else nullcontext()
    )
    with span:
        result = await process_event(
            session_factory=session_factory,
            settings=settings,
            event=event,
            provider_factory=provider_factory,
            telemetry=telemetry,
            parent_context=telemetry.capture_context() if telemetry else None,
        )
    if result == HandleResult.HANDLED and telemetry is not None:
        telemetry.increment("ai.consume", outcome="handled", topic=record.topic)
    return result
