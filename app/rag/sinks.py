"""SQS sink: client, publisher and dispatcher. Run with ``python -m app.rag.sinks``.

    rag_outbox_deliveries (sink sqs) --claim--> send_message to a FIFO queue
        MessageGroupId          document id: per-document order
        MessageDeduplicationId  event id + publish attempt: a redispatch is a new send,
                                a duplicate is settled by the consumer's claim fence
    dead letters -> RAG_SQS_DEAD_LETTER_QUEUE; created, updated, deleted -> RAG_SQS_QUEUE

The client talks only to RAG_SQS_ENDPOINT with fixed local credentials and no AWS
config or credential files, so nothing here can reach real AWS by default. A failed
send defers the delivery and ends the process (Compose restarts it), as the Kafka
dispatcher does; Kafka delivery runs in its own process and is not held back.
"""

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

from aiobotocore.session import AioSession
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

from app.ai.dispatcher import install_stop_signals, wait_for_tick
from app.ai.kafka import extract_trace_headers
from app.ai.processing import SessionFactory
from app.db import create_engine, create_session_factory
from app.models import RAGEventSink
from app.rag import ingestion
from app.rag.config import SQS_SINK, RAGSettings
from app.rag.contracts import EventSink
from app.rag.outbox import (
    OutboxLease,
    acknowledge_outbox,
    claim_outbox,
    defer_publication,
    event_payload,
)
from app.telemetry import Telemetry, configure_telemetry

SQS_IO_TIMEOUT = 5
SQS_CONNECT_TIMEOUT = 2
# Not a secret: ElasticMQ accepts any key, and real AWS rejects this one.
LOCAL_CREDENTIAL = "local-elasticmq-only"
SQS_ERRORS = (BotoCoreError, ClientError, OSError, TimeoutError)


@dataclass(frozen=True)
class SQSQueues:
    client: Any
    events_url: str
    dead_letter_url: str


@asynccontextmanager
async def open_sqs(settings: RAGSettings) -> AsyncIterator[SQSQueues]:
    if not settings.rag_sqs_endpoint:
        raise RuntimeError("The sqs sink requires RAG_SQS_ENDPOINT")
    session = AioSession()
    # No profile, config or credential files: endpoint, region and keys are explicit.
    session.set_config_variable("config_file", os.devnull)
    session.set_config_variable("credentials_file", os.devnull)
    config = Config(
        connect_timeout=SQS_CONNECT_TIMEOUT,
        read_timeout=settings.rag_sqs_wait_seconds + SQS_IO_TIMEOUT,
        retries={"total_max_attempts": 1},
        # No HTTP_PROXY/HTTPS_PROXY from the environment: the endpoint only.
        proxies={},
    )
    async with session.create_client(
        "sqs",
        endpoint_url=settings.rag_sqs_endpoint,
        region_name=settings.rag_sqs_region,
        aws_access_key_id=LOCAL_CREDENTIAL,
        aws_secret_access_key=LOCAL_CREDENTIAL,
        config=config,
    ) as client:
        async with asyncio.timeout(SQS_IO_TIMEOUT):
            events = await client.get_queue_url(QueueName=settings.rag_sqs_queue)
            dead_letter = await client.get_queue_url(QueueName=settings.rag_sqs_dead_letter_queue)
        yield SQSQueues(client, events["QueueUrl"], dead_letter["QueueUrl"])


def trace_attributes(telemetry: Telemetry | None) -> dict[str, dict[str, str]]:
    """Only a validated W3C traceparent travels with a message; never baggage or tracestate."""
    if telemetry is None:
        return {}
    carrier = telemetry.capture_context()
    validated = extract_trace_headers(
        [(name, value.encode("ascii")) for name, value in carrier.items()]
    )
    return {name: {"DataType": "String", "StringValue": value} for name, value in validated.items()}


async def send_fifo(
    queues: SQSQueues,
    queue_url: str,
    *,
    body: bytes,
    group: str,
    dedup: str,
    telemetry: Telemetry | None,
) -> None:
    attributes = trace_attributes(telemetry)
    request = {
        "QueueUrl": queue_url,
        "MessageBody": body.decode(),
        "MessageGroupId": group,
        "MessageDeduplicationId": dedup,
    }
    if attributes:
        request["MessageAttributes"] = attributes
    async with asyncio.timeout(SQS_IO_TIMEOUT):
        await queues.client.send_message(**request)


async def register_sink(session_factory: SessionFactory, sink: EventSink) -> None:
    """From now on every new outbox row gets a delivery row for this sink.

    SHARE mode waits for transactions that already inserted outbox rows, and holds off
    new inserts until this commits, so no event straddles the registration.
    """
    async with session_factory() as session:
        if await session.get(RAGEventSink, str(sink)) is not None:
            # Already registered: no lock, so a dispatcher restart never blocks uploads.
            await session.rollback()
            return
        await session.execute(text("LOCK TABLE rag_outbox IN SHARE MODE"))
        await session.execute(insert(RAGEventSink).values(sink=str(sink)).on_conflict_do_nothing())
        await session.commit()


def _queue(queues: SQSQueues, lease: OutboxLease) -> str:
    return queues.dead_letter_url if lease.kind == "dead_letter" else queues.events_url


async def dispatch_sqs_once(
    *,
    session_factory: SessionFactory,
    settings: RAGSettings,
    queues: SQSQueues,
    telemetry: Telemetry | None = None,
    batch_size: int = 10,
    stop: asyncio.Event | None = None,
) -> int:
    """Claim this sink's deliveries one at a time, send with a deadline, fence the ACK."""
    published = 0
    for _ in range(batch_size):
        if stop is not None and stop.is_set():
            break
        lease = await claim_outbox(
            session_factory=session_factory,
            settings=settings,
            sink=EventSink.SQS,
            lease_seconds=SQS_IO_TIMEOUT * 2,
        )
        if lease is None:
            break
        queue_name = (
            settings.rag_sqs_dead_letter_queue
            if lease.kind == "dead_letter"
            else settings.rag_sqs_queue
        )
        context = telemetry.extract_context(lease.trace_context) if telemetry else None
        span = (
            telemetry.span(
                "rag.dispatcher.publish",
                context=context,
                document_id=str(lease.document_id),
                event_id=str(lease.event_id),
                attempt=lease.attempt,
                sink=str(EventSink.SQS),
                queue=queue_name,
            )
            if telemetry
            else nullcontext()
        )
        started = time.monotonic()
        try:
            with span:
                await send_fifo(
                    queues,
                    _queue(queues, lease),
                    body=event_payload(lease),
                    group=str(lease.document_id),
                    dedup=f"{lease.event_id}-{lease.publish_attempts}",
                    telemetry=telemetry,
                )
        except SQS_ERRORS:
            if telemetry is not None:
                telemetry.increment("ai.publish", outcome="failed", sink=str(EventSink.SQS))
            await defer_publication(session_factory=session_factory, lease=lease)
            raise
        if await acknowledge_outbox(session_factory=session_factory, lease=lease):
            published += 1
            if telemetry is not None:
                telemetry.increment("ai.publish", outcome="succeeded", sink=str(EventSink.SQS))
                telemetry.observe(
                    "ai.publish.duration",
                    time.monotonic() - started,
                    outcome="succeeded",
                    sink=str(EventSink.SQS),
                )
    return published


async def run_sqs_dispatcher(settings: RAGSettings | None = None) -> None:
    settings = settings or RAGSettings()
    if not settings.rag_enabled or SQS_SINK not in settings.event_sinks:
        raise RuntimeError("SQS dispatcher requires RAG_ENABLED and sqs in RAG_EVENT_SINKS")
    stop = asyncio.Event()
    install_stop_signals(stop)
    telemetry = configure_telemetry("rag-sqs-dispatcher")
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)
    try:
        async with open_sqs(settings) as queues:
            await register_sink(session_factory, EventSink.SQS)
            while not stop.is_set():
                # Recovery is database-only and lock-skipping: it keeps running here when
                # Kafka, and with it the main dispatcher, is down.
                await ingestion.recover_documents(
                    session_factory=session_factory, settings=settings, telemetry=telemetry
                )
                await dispatch_sqs_once(
                    session_factory=session_factory,
                    settings=settings,
                    queues=queues,
                    telemetry=telemetry,
                    stop=stop,
                )
                await wait_for_tick(stop, settings.ai_poll_seconds)
    finally:
        await engine.dispose()
        await asyncio.to_thread(telemetry.shutdown)


def main() -> None:
    # Dependency exceptions may embed URLs or payloads; never print their messages.
    logging.basicConfig(level=logging.CRITICAL)
    try:
        asyncio.run(run_sqs_dispatcher())
    except KeyboardInterrupt:
        pass
    except Exception:
        raise SystemExit("SQS dispatcher stopped after a dependency failure") from None


if __name__ == "__main__":
    main()
