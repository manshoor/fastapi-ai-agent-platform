"""RAG outbox publication, broker ACK fencing, and malformed-record dead letters."""

import hashlib
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from aiokafka.errors import KafkaError
from sqlalchemy import func, select, update

from app.ai.processing import HandleResult
from app.models import RAGDocument, RAGOutbox, User
from app.rag.config import RAGSettings
from app.rag.contracts import ContentType
from app.rag.documents import submit_document
from app.rag.outbox import dispatch_once, handle_record
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

    return SimpleNamespace(settings=rag_settings, factory=factory, submit=submit)


async def test_requested_event_is_published_then_acknowledged(rag):
    document_id = await rag.submit()
    producer = RecordingProducer()
    assert (
        await dispatch_once(session_factory=rag.factory, settings=rag.settings, producer=producer)
        == 1
    )
    topic, value, key = producer.sent[0]
    assert topic == rag.settings.kafka_rag_request_topic
    assert key == str(document_id).encode()
    payload = json.loads(value)
    assert set(payload) == {"version", "event_id", "document_id", "attempt", "created_at"}
    assert payload["document_id"] == str(document_id) and payload["attempt"] == 1
    assert SECRET not in value.decode()
    async with rag.factory() as session:
        outbox = await session.scalar(select(RAGOutbox))
        assert outbox.published_at is not None and outbox.lease_token is None
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
        outbox = await session.scalar(select(RAGOutbox))
        assert outbox.published_at is None and outbox.lease_token is None
        assert outbox.available_at > await session.scalar(select(func.clock_timestamp()))


async def test_dead_letter_payload_carries_status_and_code_only(rag):
    document_id = await rag.submit()
    async with rag.factory() as session:
        await session.execute(
            update(RAGDocument)
            .where(RAGDocument.id == document_id)
            .values(status="failed", source=None, error_code="no_text", finished_at=func.now())
        )
        await session.execute(update(RAGOutbox).values(published_at=func.now()))
        session.add(RAGOutbox(document_id=document_id, event_kind="dead_letter", attempt=1))
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
        value=raw, topic=rag.settings.kafka_rag_request_topic, partition=3, offset=42, headers=[]
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
