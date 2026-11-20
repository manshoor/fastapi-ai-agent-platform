"""SQS document-event consumer: ``python -m app.rag.sqs_worker``.

    receive (long poll, visibility RAG_SQS_VISIBILITY_SECONDS)
      messages grouped by MessageGroupId (one document); groups run side by side,
      each group in order
        parse (size, schema)    malformed -> privacy-safe DLQ message, then delete
        process_ingest          the claim fence the Kafka worker uses: the first
                                consumer wins, a duplicate settles with no writes
          HANDLED  -> delete    (only after the outcome is durable)
          DEFERRED -> visible again in RETRY_DELAY_SECONDS; its group waits
      while a message is handled its visibility is extended every third of it
      at most AI_CONCURRENCY messages are handled at once; a failure in one group
      cancels the others, which hand their leases back at the same attempt
      on stop: nothing new starts; in-flight work gets DRAIN_SECONDS, then is cancelled

No redrive policy moves messages: a retrying event is not a dead letter.
"""

import asyncio
import hashlib
import json
import logging
from collections import defaultdict
from contextlib import nullcontext
from typing import Any

from pydantic import ValidationError

from app.ai.dispatcher import install_stop_signals
from app.ai.kafka import MAX_EVENT_BYTES, extract_trace_headers
from app.ai.processing import HandleResult, SessionFactory
from app.ai.provider import OpenAIEmbedder
from app.db import create_engine, create_session_factory
from app.rag.config import SQS_SINK, RAGSettings
from app.rag.contracts import DocumentEvent
from app.rag.ingestion import Embedder, process_ingest
from app.rag.sinks import SQS_ERRORS, SQS_IO_TIMEOUT, SQSQueues, open_sqs, send_fifo
from app.telemetry import Telemetry, configure_telemetry

RETRY_DELAY_SECONDS = 5
MAX_MESSAGES = 10
MALFORMED_GROUP = "malformed"
# On stop, in-flight work gets this long before it is cancelled (as the Kafka worker).
DRAIN_SECONDS = 5


def _carrier(message: dict) -> dict[str, str]:
    attributes = message.get("MessageAttributes") or {}
    headers = [
        (name, value["StringValue"].encode("ascii", "replace"))
        for name, value in attributes.items()
        if isinstance(value, dict) and isinstance(value.get("StringValue"), str)
    ]
    return extract_trace_headers(headers)


async def _dead_letter(
    queues: SQSQueues, settings: RAGSettings, message: dict, raw: bytes, telemetry
) -> None:
    """Fixed fields only: never the body, which may hold anything."""
    message_id = str(message.get("MessageId", ""))
    payload = json.dumps(
        {
            "version": 1,
            "error_code": "malformed_event",
            "source_queue": settings.rag_sqs_queue,
            "source_message_id_sha256": hashlib.sha256(message_id.encode()).hexdigest(),
            "payload_sha256": hashlib.sha256(raw).hexdigest(),
        },
        separators=(",", ":"),
    ).encode()
    await send_fifo(
        queues,
        queues.dead_letter_url,
        body=payload,
        group=MALFORMED_GROUP,
        dedup=hashlib.sha256(message_id.encode()).hexdigest(),
        telemetry=telemetry,
    )


async def handle_message(
    *,
    session_factory: SessionFactory,
    settings: RAGSettings,
    queues: SQSQueues,
    message: dict,
    embedder: Embedder,
    telemetry: Telemetry | None = None,
) -> HandleResult:
    body = message.get("Body")
    raw = body.encode("utf-8", "replace") if isinstance(body, str) else b""
    try:
        if not raw or len(raw) > MAX_EVENT_BYTES:
            raise ValueError("event size")
        event = DocumentEvent.model_validate_json(raw)
    except (ValueError, ValidationError):
        await _dead_letter(queues, settings, message, raw, telemetry)
        if telemetry is not None:
            telemetry.increment("ai.consume", outcome="malformed", sink=SQS_SINK)
        return HandleResult.HANDLED
    carrier = _carrier(message) if telemetry else {}
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
            sink=SQS_SINK,
            queue=settings.rag_sqs_queue,
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
            telemetry=telemetry,
        )
    if result == HandleResult.HANDLED and telemetry is not None:
        telemetry.increment("ai.consume", outcome="handled", sink=SQS_SINK)
    return result


async def _set_visibility(queues: SQSQueues, receipt: str, seconds: int) -> None:
    async with asyncio.timeout(SQS_IO_TIMEOUT):
        await queues.client.change_message_visibility(
            QueueUrl=queues.events_url, ReceiptHandle=receipt, VisibilityTimeout=seconds
        )


async def _keep_invisible(queues: SQSQueues, receipt: str, seconds: int) -> None:
    """Extend visibility while the message is handled, so no second consumer gets it."""
    while True:
        await asyncio.sleep(seconds / 3)
        try:
            await _set_visibility(queues, receipt, seconds)
        except SQS_ERRORS:
            # The claim fence still settles a redelivery; keep trying.
            continue


async def process_message(
    *,
    session_factory: SessionFactory,
    settings: RAGSettings,
    queues: SQSQueues,
    message: dict,
    embedder: Embedder,
    telemetry: Telemetry | None = None,
) -> HandleResult:
    receipt = message["ReceiptHandle"]
    keeper = asyncio.create_task(
        _keep_invisible(queues, receipt, settings.rag_sqs_visibility_seconds)
    )
    try:
        result = await handle_message(
            session_factory=session_factory,
            settings=settings,
            queues=queues,
            message=message,
            embedder=embedder,
            telemetry=telemetry,
        )
    except BaseException:
        await _stop_keeper(keeper)
        # Best effort: let it come back soon; otherwise it returns when visibility ends.
        await _release(queues, [message], RETRY_DELAY_SECONDS)
        raise
    # Awaited, so no late extension lands after the release below.
    await _stop_keeper(keeper)
    if result == HandleResult.HANDLED:
        # The outcome is committed: only now may the message go.
        async with asyncio.timeout(SQS_IO_TIMEOUT):
            await queues.client.delete_message(QueueUrl=queues.events_url, ReceiptHandle=receipt)
    else:
        await _set_visibility(queues, receipt, RETRY_DELAY_SECONDS)
    return result


async def _stop_keeper(keeper: asyncio.Task) -> None:
    keeper.cancel()
    await asyncio.gather(keeper, return_exceptions=True)


async def _release(queues: SQSQueues, messages: list[dict], seconds: int) -> None:
    """Best effort: a message not released returns when its visibility ends."""
    for message in messages:
        try:
            await _set_visibility(queues, message["ReceiptHandle"], seconds)
        except SQS_ERRORS:
            continue


async def _run_group(messages: list[dict], semaphore: asyncio.Semaphore, **kwargs: Any) -> None:
    """One document's messages in order; at most ai_concurrency handled at once."""
    for index, message in enumerate(messages):
        try:
            async with semaphore:
                result = await process_message(message=message, **kwargs)
        except BaseException:
            await _release(kwargs["queues"], messages[index + 1 :], 0)
            raise
        if result != HandleResult.HANDLED:
            # Later messages of this document wait: release them to keep their order.
            await _release(kwargs["queues"], messages[index + 1 :], RETRY_DELAY_SECONDS)
            return


async def poll_once(
    *,
    session_factory: SessionFactory,
    settings: RAGSettings,
    queues: SQSQueues,
    embedder: Embedder,
    telemetry: Telemetry | None = None,
    wait_seconds: int | None = None,
    stop: asyncio.Event | None = None,
) -> int:
    """One long poll; returns how many messages were received.

    A failure in one group cancels the others (their leases are handed back at the same
    attempt) and is raised only after every group has finished.
    """
    wait = settings.rag_sqs_wait_seconds if wait_seconds is None else wait_seconds
    async with asyncio.timeout(wait + SQS_IO_TIMEOUT):
        response = await queues.client.receive_message(
            QueueUrl=queues.events_url,
            MaxNumberOfMessages=MAX_MESSAGES,
            WaitTimeSeconds=wait,
            VisibilityTimeout=settings.rag_sqs_visibility_seconds,
            MessageAttributeNames=["traceparent"],
            MessageSystemAttributeNames=["MessageGroupId"],
        )
    messages = response.get("Messages") or []
    if stop is not None and stop.is_set():
        # Stopping: start nothing new, hand the batch straight back.
        await _release(queues, messages, 0)
        return len(messages)
    groups: dict[str, list[dict]] = defaultdict(list)
    for message in messages:
        group = (message.get("Attributes") or {}).get("MessageGroupId", "")
        groups[group].append(message)
    semaphore = asyncio.Semaphore(settings.ai_concurrency)
    async with asyncio.TaskGroup() as tasks:
        for group in groups.values():
            tasks.create_task(
                _run_group(
                    group,
                    semaphore,
                    session_factory=session_factory,
                    settings=settings,
                    queues=queues,
                    embedder=embedder,
                    telemetry=telemetry,
                )
            )
    return len(messages)


async def _drain(batch: asyncio.Task, stop: asyncio.Event) -> None:
    """Wait for the batch; on stop give it DRAIN_SECONDS, then cancel (leases handed back)."""
    stopping = asyncio.create_task(stop.wait())
    try:
        await asyncio.wait({batch, stopping}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        stopping.cancel()
    if not batch.done():
        try:
            await asyncio.wait_for(asyncio.shield(batch), DRAIN_SECONDS)
        except TimeoutError:
            batch.cancel()
            await asyncio.gather(batch, return_exceptions=True)
            return
    batch.result()


async def run_worker(settings: RAGSettings | None = None) -> None:
    settings = settings or RAGSettings()
    if not settings.rag_enabled or SQS_SINK not in settings.event_sinks:
        raise RuntimeError("SQS worker requires RAG_ENABLED and sqs in RAG_EVENT_SINKS")
    if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value():
        raise RuntimeError("SQS worker requires OPENAI_API_KEY")
    stop = asyncio.Event()
    install_stop_signals(stop)
    telemetry = configure_telemetry("rag-sqs-worker")
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)
    embedder = None
    try:
        embedder = OpenAIEmbedder(settings, telemetry)
        async with open_sqs(settings) as queues:
            while not stop.is_set():
                batch = asyncio.create_task(
                    poll_once(
                        session_factory=session_factory,
                        settings=settings,
                        queues=queues,
                        embedder=embedder,
                        telemetry=telemetry,
                        stop=stop,
                    )
                )
                await _drain(batch, stop)
    finally:
        try:
            if embedder is not None:
                await embedder.aclose()
        finally:
            await engine.dispose()
            await asyncio.to_thread(telemetry.shutdown)


def main() -> None:
    # Dependency exceptions may embed URLs or payloads; never print their messages.
    logging.basicConfig(level=logging.CRITICAL)
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        pass
    except Exception:
        raise SystemExit("SQS worker stopped after a dependency failure") from None


if __name__ == "__main__":
    main()
