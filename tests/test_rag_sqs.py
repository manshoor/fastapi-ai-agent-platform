"""SQS sink and consumer against a real ElasticMQ (FIFO), PostgreSQL and the claim fence."""

import asyncio
import json
import os
from datetime import timedelta
from uuid import uuid4

import pytest
from aiobotocore.session import AioSession
from botocore.config import Config
from botocore.exceptions import BotoCoreError, EndpointConnectionError
from sqlalchemy import delete, func, select, text, update

from app.ai.processing import HandleResult
from app.models import RAGDocument, RAGEventSink, RAGOutbox, RAGOutboxDelivery
from app.rag.config import RAGSettings
from app.rag.contracts import DocumentEventType, EventSink
from app.rag.documents import delete_document
from app.rag.ingestion import claim_document
from app.rag.outbox import dispatch_once
from app.rag.sinks import (
    LOCAL_CREDENTIAL,
    SQSQueues,
    dispatch_sqs_once,
    open_sqs,
    register_sink,
    send_fifo,
)
from app.rag.sqs_worker import poll_once
from tests.test_rag_ingestion import ControlledEmbedder, rag_runtime  # noqa: F401
from tests.test_rag_outbox import RecordingProducer

ENDPOINT = os.environ.get("AI_SQS_ENDPOINT")
pytestmark = pytest.mark.skipif(not ENDPOINT, reason="needs the AI QA stack's ElasticMQ")
DOC = b"The boiler is serviced every October by the building manager."


@pytest.fixture
def sqs_settings(runtime) -> RAGSettings:
    return RAGSettings.model_validate(
        {
            **runtime.settings.model_dump(),
            "rag_event_sinks": "kafka,sqs",
            "rag_sqs_endpoint": ENDPOINT,
        }
    )


@pytest.fixture
async def queues(runtime, sqs_settings):
    async with open_sqs(sqs_settings) as opened:
        for url in (opened.events_url, opened.dead_letter_url):
            await opened.client.purge_queue(QueueUrl=url)
        await register_sink(runtime.factory, EventSink.SQS)
        yield opened


async def receive(queues: SQSQueues, url: str, wait: int = 1) -> list[dict]:
    response = await queues.client.receive_message(
        QueueUrl=url,
        MaxNumberOfMessages=10,
        WaitTimeSeconds=wait,
        VisibilityTimeout=30,
        MessageSystemAttributeNames=["MessageGroupId", "MessageDeduplicationId"],
    )
    return response.get("Messages") or []


async def depth(queues: SQSQueues, url: str) -> tuple[int, int]:
    """(visible, in flight) messages."""
    attributes = (
        await queues.client.get_queue_attributes(
            QueueUrl=url,
            AttributeNames=[
                "ApproximateNumberOfMessages",
                "ApproximateNumberOfMessagesNotVisible",
            ],
        )
    )["Attributes"]
    return (
        int(attributes["ApproximateNumberOfMessages"]),
        int(attributes["ApproximateNumberOfMessagesNotVisible"]),
    )


async def publish(runtime, settings, queues) -> int:
    return await dispatch_sqs_once(
        session_factory=runtime.factory, settings=settings, queues=queues
    )


async def poll(runtime, settings, queues, embedder=None) -> int:
    return await poll_once(
        session_factory=runtime.factory,
        settings=settings,
        queues=queues,
        embedder=embedder or ControlledEmbedder(),
        wait_seconds=1,
    )


async def sqs_delivery(runtime, document_id) -> RAGOutboxDelivery:
    async with runtime.factory() as session:
        return await session.scalar(
            select(RAGOutboxDelivery)
            .join(RAGOutbox, RAGOutbox.id == RAGOutboxDelivery.outbox_id)
            .where(RAGOutbox.document_id == document_id, RAGOutboxDelivery.sink == "sqs")
            .order_by(RAGOutbox.created_at.desc())
            .limit(1)
        )


def test_sqs_sink_requires_an_endpoint(runtime):
    with pytest.raises(ValueError, match="RAG_SQS_ENDPOINT"):
        RAGSettings.model_validate(
            {**runtime.settings.model_dump(), "rag_event_sinks": "kafka,sqs"}
        )


async def test_created_event_is_published_in_the_document_group(runtime, sqs_settings, queues):
    event = await runtime.submit(DOC)
    assert await publish(runtime, sqs_settings, queues) == 1
    [message] = await receive(queues, queues.events_url)
    attributes = message["Attributes"]
    assert attributes["MessageGroupId"] == str(event.document_id)
    assert attributes["MessageDeduplicationId"] == f"{event.event_id}-1"
    body = json.loads(message["Body"])
    assert (body["type"], body["event_id"]) == ("doc.created", str(event.event_id))
    assert DOC.decode() not in message["Body"]
    delivery = await sqs_delivery(runtime, event.document_id)
    assert delivery.published_at is not None and delivery.lease_token is None
    # Published and not due for redispatch: nothing more to send.
    assert await publish(runtime, sqs_settings, queues) == 0


async def test_fifo_drops_a_repeated_dedup_id(queues):
    # The broker behaviour the dedup id design rests on.
    for _ in range(2):
        await send_fifo(
            queues, queues.events_url, body=b"{}", group="g", dedup="same", telemetry=None
        )
    assert len(await receive(queues, queues.events_url)) == 1


async def test_redispatch_is_a_new_message(runtime, sqs_settings, queues):
    event = await runtime.submit(DOC)
    await publish(runtime, sqs_settings, queues)
    async with runtime.factory() as session:
        await session.execute(
            update(RAGOutboxDelivery)
            .where(RAGOutboxDelivery.outbox_id == event.event_id)
            .values(last_published_at=func.clock_timestamp() - timedelta(hours=1))
        )
        await session.commit()
    assert await publish(runtime, sqs_settings, queues) == 1
    ids = [
        m["Attributes"]["MessageDeduplicationId"] for m in await receive(queues, queues.events_url)
    ]
    # One group: FIFO hands out its messages in order.
    assert ids == [f"{event.event_id}-1", f"{event.event_id}-2"]


async def test_sqs_down_never_blocks_kafka(runtime, sqs_settings):
    await register_sink(runtime.factory, EventSink.SQS)
    event = await runtime.submit(DOC)
    session = AioSession()
    async with session.create_client(
        "sqs",
        endpoint_url="http://127.0.0.1:9",
        region_name="us-east-1",
        aws_access_key_id=LOCAL_CREDENTIAL,
        aws_secret_access_key=LOCAL_CREDENTIAL,
        config=Config(connect_timeout=1, retries={"total_max_attempts": 1}),
    ) as client:
        down = SQSQueues(client, "http://127.0.0.1:9/q.fifo", "http://127.0.0.1:9/d.fifo")
        with pytest.raises(BotoCoreError):
            await publish(runtime, sqs_settings, down)
    # The SQS delivery is deferred; Kafka still publishes its own.
    delivery = await sqs_delivery(runtime, event.document_id)
    assert delivery.published_at is None and delivery.lease_token is None
    producer = RecordingProducer()
    assert (
        await dispatch_once(
            session_factory=runtime.factory, settings=sqs_settings, producer=producer
        )
        == 1
    )


async def test_sqs_consumer_ingests_then_deletes(runtime, sqs_settings, queues):
    event = await runtime.submit(DOC)
    await publish(runtime, sqs_settings, queues)
    assert await poll(runtime, sqs_settings, queues) == 1
    document = await runtime.read(event.document_id)
    assert (document.status, document.active_version) == ("ready", 1)
    assert await depth(queues, queues.events_url) == (0, 0)


async def test_second_consumer_settles_the_duplicate(runtime, sqs_settings, queues):
    event = await runtime.submit(DOC)
    await publish(runtime, sqs_settings, queues)
    # Kafka's consumer wins the claim first.
    assert await runtime.ingest(event, ControlledEmbedder()) is HandleResult.HANDLED
    chunks = [(c.version, c.ordinal, c.content) for c in await runtime.chunks(event.document_id)]
    embedder = ControlledEmbedder()
    assert await poll(runtime, sqs_settings, queues, embedder) == 1
    assert embedder.calls == []
    assert [
        (c.version, c.ordinal, c.content) for c in await runtime.chunks(event.document_id)
    ] == chunks
    assert await depth(queues, queues.events_url) == (0, 0)


async def test_live_lease_defers_and_keeps_the_message(runtime, sqs_settings, queues):
    event = await runtime.submit(DOC)
    await publish(runtime, sqs_settings, queues)
    # Another consumer holds the document.
    assert await claim_document(
        session_factory=runtime.factory, settings=runtime.settings, event=event
    )
    embedder = ControlledEmbedder()
    assert await poll(runtime, sqs_settings, queues, embedder) == 1
    assert embedder.calls == []
    # Not deleted: back after the retry delay.
    assert sum(await depth(queues, queues.events_url)) == 1


async def test_malformed_message_goes_to_the_dlq_without_content(runtime, sqs_settings, queues):
    secret = "synthetic-message-text-must-not-leak"
    await send_fifo(
        queues,
        queues.events_url,
        body=json.dumps({"text": secret}).encode(),
        group="x",
        dedup=uuid4().hex,
        telemetry=None,
    )
    assert await poll(runtime, sqs_settings, queues) == 1
    assert await depth(queues, queues.events_url) == (0, 0)
    [dead] = await receive(queues, queues.dead_letter_url)
    payload = json.loads(dead["Body"])
    assert payload["error_code"] == "malformed_event" and secret not in dead["Body"]
    assert set(payload) == {
        "version",
        "error_code",
        "source_queue",
        "source_message_id_sha256",
        "payload_sha256",
    }


async def test_delete_purged_by_kafka_first_settles_on_sqs(runtime, sqs_settings, queues):
    event = await runtime.submit(DOC)
    assert await runtime.ingest(event, ControlledEmbedder()) is HandleResult.HANDLED
    async with runtime.factory() as session:
        await session.execute(update(RAGOutboxDelivery).values(published_at=func.clock_timestamp()))
        await session.commit()
        await delete_document(
            session=session, owner_id=runtime.owner_id, document_id=event.document_id
        )
    deleted = await runtime.event_for(event.document_id)
    assert deleted.type is DocumentEventType.DELETED
    # SQS publishes the delete, then Kafka's consumer purges the tombstone first.
    assert await publish(runtime, sqs_settings, queues) == 1
    assert await runtime.ingest(deleted, ControlledEmbedder()) is HandleResult.HANDLED
    async with runtime.factory() as session:
        assert await session.get(RAGDocument, event.document_id) is None
    # The SQS copy finds nothing to purge, settles, and is deleted.
    assert await poll(runtime, sqs_settings, queues) == 1
    assert await depth(queues, queues.events_url) == (0, 0)


async def test_purged_delivery_is_never_published(runtime, sqs_settings, queues):
    event = await runtime.submit(DOC)
    await publish(runtime, sqs_settings, queues)
    async with runtime.factory() as session:
        await delete_document(
            session=session, owner_id=runtime.owner_id, document_id=event.document_id
        )
    deleted = await runtime.event_for(event.document_id)
    # Kafka's consumer purges before SQS publishes: the cascade removed the delivery.
    assert await runtime.ingest(deleted, ControlledEmbedder()) is HandleResult.HANDLED
    assert await publish(runtime, sqs_settings, queues) == 0


# --- Worker lifecycle (deep review fixes) -----------------------------------------------


class FailingDeletes:
    """The real client, except delete_message fails: an SQS error in one group."""

    def __init__(self, client):
        self._client = client

    def __getattr__(self, name):
        return getattr(self._client, name)

    async def delete_message(self, **_):
        raise EndpointConnectionError(endpoint_url="http://sqs:9324")


async def test_failed_group_cancels_siblings_without_spending_attempts(
    runtime, sqs_settings, queues
):
    first = await runtime.submit(b"Alpha notice about the boiler.")
    second = await runtime.submit(b"Beta notice about the lift.")
    await publish(runtime, sqs_settings, queues)
    await publish(runtime, sqs_settings, queues)
    beta_started = asyncio.Event()

    class Embedder(ControlledEmbedder):
        async def embed(self, texts, *, model=None):
            if any("Beta" in text for text in texts):
                beta_started.set()
                await asyncio.Event().wait()
            await beta_started.wait()
            return await super().embed(texts, model=model)

    broken = SQSQueues(FailingDeletes(queues.client), queues.events_url, queues.dead_letter_url)
    with pytest.raises(ExceptionGroup) as caught:
        async with asyncio.timeout(30):
            await poll(runtime, sqs_settings, broken, Embedder())
    assert caught.value.subgroup(BotoCoreError) is not None
    # The healthy sibling was cancelled and handed back at the same attempt.
    beta = await runtime.read(second.document_id)
    assert (beta.status, beta.attempt, beta.lease_token) == ("pending", 1, None)
    assert (await runtime.read(first.document_id)).status == "ready"


async def test_worker_handles_at_most_ai_concurrency(runtime, sqs_settings, queues):
    for index in range(3):
        await runtime.submit(f"Notice number {index} about the garden.".encode())
        await publish(runtime, sqs_settings, queues)
    running, peak = 0, 0

    class Embedder(ControlledEmbedder):
        async def embed(self, texts, *, model=None):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.05)
            running -= 1
            return await super().embed(texts, model=model)

    settings = sqs_settings.model_copy(update={"ai_concurrency": 1})
    handled = 0
    async with asyncio.timeout(30):
        while handled < 3:
            handled += await poll(runtime, settings, queues, Embedder())
    assert peak == 1


async def test_stop_after_receive_hands_messages_back(runtime, sqs_settings, queues):
    event = await runtime.submit(DOC)
    await publish(runtime, sqs_settings, queues)
    stop = asyncio.Event()
    stop.set()
    embedder = ControlledEmbedder()
    await poll_once(
        session_factory=runtime.factory,
        settings=sqs_settings,
        queues=queues,
        embedder=embedder,
        wait_seconds=1,
        stop=stop,
    )
    assert embedder.calls == []
    assert (await runtime.read(event.document_id)).status == "pending"
    # Visible again at once for the next worker.
    assert await depth(queues, queues.events_url) == (1, 0)


async def test_registration_waits_for_inserting_transactions(app, runtime, queues):
    async with runtime.factory() as session:
        await session.execute(delete(RAGEventSink).where(RAGEventSink.sink == "sqs"))
        await session.commit()
    event = await runtime.submit(DOC)
    async with runtime.factory() as writer:
        # An event inserted, not yet committed, before the sink registers.
        writer.add(
            RAGOutbox(document_id=event.document_id, event_kind="deleted", version=1, attempt=1)
        )
        await writer.flush()
        registering = asyncio.create_task(register_sink(runtime.factory, EventSink.SQS))
        async with asyncio.timeout(5):
            while not await _waiting_on_lock(app):
                await asyncio.sleep(0.01)
        assert not registering.done()
        await writer.commit()
    await registering
    # The registration came after that event; every later one gets an sqs delivery.
    later = await runtime.submit(b"A later notice.")
    assert await sqs_delivery(runtime, later.document_id) is not None


async def _waiting_on_lock(app) -> bool:
    async with app.state.engine.connect() as connection:
        return bool(
            await connection.scalar(
                text(
                    "SELECT count(*) FROM pg_stat_activity "
                    "WHERE wait_event_type = 'Lock' AND datname = current_database()"
                )
            )
        )


@pytest.mark.parametrize("sinks", ["sqs", "kafka,kafka", "sqs,kafka,sqs"])
def test_event_sinks_always_include_kafka_once(runtime, sinks):
    with pytest.raises(ValueError):
        RAGSettings.model_validate(
            {
                **runtime.settings.model_dump(),
                "rag_event_sinks": sinks,
                "rag_sqs_endpoint": ENDPOINT,
            }
        )
