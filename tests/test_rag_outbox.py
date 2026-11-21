"""RAG outbox publication, broker ACK fencing, and malformed-record dead letters."""

import hashlib
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from aiokafka.errors import KafkaError
from sqlalchemy import func, select, update

from app.ai.processing import HandleResult
from app.models import RAGDocument, RAGEventSink, RAGOutbox, RAGOutboxDelivery, User
from app.rag.config import RAGSettings
from app.rag.contracts import ContentType, EventSink
from app.rag.documents import delete_document, submit_document
from app.rag.outbox import claim_outbox, dispatch_once, handle_record
from app.rag.uploads import Upload

SECRET = "synthetic-document-text-must-not-leak"


class RecordingProducer:
    def __init__(self, *, failure: Exception | None = None):
        self.failure = failure
        self.sent: list[tuple[str, bytes, bytes]] = []

    async def send_and_wait(self, topic, value, *, key, headers=None):
        if self.failure is not None:
            raise self.failure
        self.sent.append((topic, value, key))


class UnusedEmbedder:
    async def embed(self, texts, *, model=None):
        raise AssertionError("A malformed record must never reach the provider")


@pytest.fixture
async def rag(app, settings):
    rag_settings = RAGSettings.model_validate(
        {**settings.model_dump(), "ai_enabled": True, "rag_enabled": True}
    )
    factory = app.state.session_factory
    async with factory() as session:
        user = User(email=f"rag-{uuid4().hex}@example.com", password_hash="unused-test-hash")
        session.add(user)
        await session.commit()
        owner_id = user.id

    async def submit() -> str:
        async with factory() as session:
            response, _ = await submit_document(
                session=session,
                settings=rag_settings,
                owner_id=owner_id,
                upload=Upload(title="Doc", content_type=ContentType.TEXT, data=SECRET.encode()),
                idempotency_key=uuid4().hex,
                trace_context={},
            )
        return response.id

    return SimpleNamespace(settings=rag_settings, factory=factory, submit=submit, owner_id=owner_id)


async def test_created_event_is_published_then_acknowledged(rag):
    document_id = await rag.submit()
    producer = RecordingProducer()
    assert (
        await dispatch_once(session_factory=rag.factory, settings=rag.settings, producer=producer)
        == 1
    )
    topic, value, key = producer.sent[0]
    assert topic == rag.settings.kafka_rag_event_topic
    assert key == str(document_id).encode()
    payload = json.loads(value)
    assert set(payload) == {
        "version",
        "event_id",
        "type",
        "document_id",
        "document_version",
        "attempt",
        "source_kind",
        "created_at",
    }
    assert payload["document_id"] == str(document_id) and payload["attempt"] == 1
    assert (payload["type"], payload["document_version"]) == ("doc.created", 1)
    assert payload["source_kind"] == "upload"
    assert SECRET not in value.decode()
    async with rag.factory() as session:
        delivery = await session.scalar(select(RAGOutboxDelivery))
        assert delivery.sink == "kafka" and delivery.publish_attempts == 1
        assert delivery.published_at is not None and delivery.lease_token is None
    # Published and not yet due for redispatch: nothing more to send.
    assert (
        await dispatch_once(session_factory=rag.factory, settings=rag.settings, producer=producer)
        == 0
    )


async def test_failed_publish_is_deferred_and_raised(rag):
    await rag.submit()
    with pytest.raises(KafkaError):
        await dispatch_once(
            session_factory=rag.factory,
            settings=rag.settings,
            producer=RecordingProducer(failure=KafkaError()),
        )
    async with rag.factory() as session:
        delivery = await session.scalar(select(RAGOutboxDelivery))
        assert delivery.published_at is None and delivery.lease_token is None
        assert delivery.available_at > await session.scalar(select(func.clock_timestamp()))


async def test_dead_letter_payload_carries_status_and_code_only(rag):
    document_id = await rag.submit()
    async with rag.factory() as session:
        await session.execute(
            update(RAGDocument)
            .where(RAGDocument.id == document_id)
            .values(status="failed", source=None, error_code="no_text", finished_at=func.now())
        )
        await session.execute(update(RAGOutboxDelivery).values(published_at=func.now()))
        session.add(
            RAGOutbox(document_id=document_id, event_kind="dead_letter", version=1, attempt=1)
        )
        await session.commit()
    producer = RecordingProducer()
    assert (
        await dispatch_once(session_factory=rag.factory, settings=rag.settings, producer=producer)
        == 1
    )
    topic, value, _ = producer.sent[0]
    assert topic == rag.settings.kafka_rag_dead_letter_topic
    payload = json.loads(value)
    assert (payload["status"], payload["error_code"]) == ("failed", "no_text")
    assert SECRET not in value.decode()


async def test_malformed_record_goes_to_dead_letter_without_content(rag):
    raw = json.dumps({"document_id": "not-a-uuid", "text": SECRET}).encode()
    record = SimpleNamespace(
        value=raw, topic=rag.settings.kafka_rag_event_topic, partition=3, offset=42, headers=[]
    )
    producer = RecordingProducer()
    result = await handle_record(
        session_factory=rag.factory,
        settings=rag.settings,
        record=record,
        producer=producer,
        embedder=UnusedEmbedder(),
    )
    assert result is HandleResult.HANDLED
    topic, value, key = producer.sent[0]
    assert topic == rag.settings.kafka_rag_dead_letter_topic
    assert key == f"{record.topic}:3:42".encode()
    payload = json.loads(value)
    assert payload == {
        "version": 1,
        "error_code": "malformed_event",
        "source_topic": record.topic,
        "source_partition": 3,
        "source_offset": 42,
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
    }
    assert SECRET not in value.decode()


async def test_deleted_event_goes_to_the_events_topic(rag):
    document_id = await rag.submit()
    async with rag.factory() as session:
        await session.execute(update(RAGOutboxDelivery).values(published_at=func.now()))
        await session.commit()
        await delete_document(session=session, owner_id=rag.owner_id, document_id=document_id)
    producer = RecordingProducer()
    assert (
        await dispatch_once(session_factory=rag.factory, settings=rag.settings, producer=producer)
        == 1
    )
    topic, value, key = producer.sent[0]
    assert topic == rag.settings.kafka_rag_event_topic and key == str(document_id).encode()
    payload = json.loads(value)
    assert (payload["type"], payload["document_version"]) == ("doc.deleted", 1)
    assert SECRET not in value.decode()


async def test_failed_update_dead_letter_reports_the_version_failure(rag):
    document_id = await rag.submit()
    async with rag.factory() as session:
        # Version 2 failed while version 1 stays served.
        await session.execute(
            update(RAGDocument)
            .where(RAGDocument.id == document_id)
            .values(
                status="ready",
                source=None,
                version=2,
                active_version=1,
                chunk_count=1,
                update_error_code="embedding_rejected",
                finished_at=func.now(),
            )
        )
        await session.execute(update(RAGOutboxDelivery).values(published_at=func.now()))
        session.add(
            RAGOutbox(document_id=document_id, event_kind="dead_letter", version=2, attempt=1)
        )
        await session.commit()
    producer = RecordingProducer()
    await dispatch_once(session_factory=rag.factory, settings=rag.settings, producer=producer)
    topic, value, _ = producer.sent[0]
    payload = json.loads(value)
    assert topic == rag.settings.kafka_rag_dead_letter_topic
    assert (payload["status"], payload["error_code"]) == ("failed", "embedding_rejected")
    assert (payload["type"], payload["document_version"]) == ("doc.updated", 2)


async def deliveries(rag, document_id) -> list[tuple[str, str]]:
    async with rag.factory() as session:
        rows = await session.execute(
            select(RAGOutbox.event_kind, RAGOutboxDelivery.sink)
            .join(RAGOutboxDelivery, RAGOutboxDelivery.outbox_id == RAGOutbox.id)
            .where(RAGOutbox.document_id == document_id)
            .order_by(RAGOutboxDelivery.sink)
        )
        return [tuple(row) for row in rows]


async def register_sqs(rag) -> None:
    async with rag.factory() as session:
        session.add(RAGEventSink(sink="sqs"))
        await session.commit()


async def test_each_registered_sink_gets_a_delivery(rag):
    before = await rag.submit()
    await register_sqs(rag)
    after = await rag.submit()
    # Events written before a sink registers are never replayed to it.
    assert await deliveries(rag, before) == [("created", "kafka")]
    assert await deliveries(rag, after) == [("created", "kafka"), ("created", "sqs")]


async def test_one_sink_down_never_blocks_the_other(rag):
    await register_sqs(rag)
    document_id = await rag.submit()
    with pytest.raises(KafkaError):
        await dispatch_once(
            session_factory=rag.factory,
            settings=rag.settings,
            producer=RecordingProducer(failure=KafkaError()),
        )
    lease = await claim_outbox(
        session_factory=rag.factory, settings=rag.settings, sink=EventSink.SQS
    )
    assert (lease.sink, lease.document_id, lease.publish_attempts) == (
        EventSink.SQS,
        document_id,
        1,
    )
    # Kafka's delivery stays deferred and leased by nobody; SQS holds its own lease.
    async with rag.factory() as session:
        rows = {row.sink: row for row in await session.scalars(select(RAGOutboxDelivery))}
    assert rows["kafka"].lease_token is None and rows["kafka"].published_at is None
    assert rows["sqs"].lease_token == lease.token
