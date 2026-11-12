"""Fenced ingestion transitions at real PostgreSQL commit and lock boundaries."""

import asyncio
import hashlib
from datetime import timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import undefer

from app.ai.processing import HandleResult
from app.ai.provider import Embeddings, ProviderFailure
from app.models import RAG_EMBEDDING_DIMENSIONS, RAGChunk, RAGDocument, RAGOutbox, User
from app.rag.config import RAGSettings
from app.rag.contracts import ContentType, IngestError, IngestEvent
from app.rag.documents import submit_document
from app.rag.ingestion import claim_document, process_ingest, recover_documents, store_batch
from app.rag.uploads import Upload
from tests.support.pdfs import text_pdf


def unit_vector(text: str) -> list[float]:
    values = [0.0] * RAG_EMBEDDING_DIMENSIONS
    values[int(hashlib.sha256(text.encode()).hexdigest(), 16) % RAG_EMBEDDING_DIMENSIONS] = 1.0
    return values


class ControlledEmbedder:
    """Embedder protocol fake: fails on chosen call numbers, can run a hook per call."""

    def __init__(self, *, failures=None, on_call=None):
        self.failures = failures or {}
        self.on_call = on_call
        self.calls: list[list[str]] = []

    async def embed(self, texts, *, model=None):
        self.calls.append(list(texts))
        if self.on_call:
            await self.on_call(len(self.calls))
        failure = self.failures.get(len(self.calls))
        if failure:
            raise failure
        return Embeddings([unit_vector(text) for text in texts], "simulated-embedding", len(texts))


@pytest.fixture
async def runtime(app, settings):
    rag = RAGSettings.model_validate(
        {
            **settings.model_dump(),
            "ai_enabled": True,
            "rag_enabled": True,
            "rag_chunk_size": 100,
            "rag_chunk_overlap": 10,
            "rag_embedding_batch_size": 2,
        }
    )
    factory = app.state.session_factory
    async with factory() as session:
        user = User(email=f"rag-{uuid4().hex}@example.com", password_hash="unused-test-hash")
        session.add(user)
        await session.commit()
        owner_id = user.id

    async def submit(data: bytes, content_type=ContentType.TEXT) -> IngestEvent:
        async with factory() as session:
            response, _ = await submit_document(
                session=session,
                settings=rag,
                owner_id=owner_id,
                upload=Upload(title="Doc", content_type=content_type, data=data),
                idempotency_key=uuid4().hex,
                trace_context={},
            )
        return await event_for(response.id)

    async def event_for(document_id: UUID, attempt: int = 1) -> IngestEvent:
        async with factory() as session:
            outbox = await session.scalar(
                select(RAGOutbox).where(
                    RAGOutbox.document_id == document_id,
                    RAGOutbox.event_kind == "requested",
                    RAGOutbox.attempt == attempt,
                )
            )
            return IngestEvent(
                version=1,
                event_id=outbox.id,
                document_id=document_id,
                attempt=attempt,
                created_at=outbox.created_at,
            )

    async def read(document_id: UUID) -> RAGDocument:
        async with factory() as session:
            # Load the deferred source too: tests read it after the session closes.
            return await session.get(
                RAGDocument, document_id, options=[undefer(RAGDocument.source)]
            )

    async def chunks(document_id: UUID) -> list[RAGChunk]:
        async with factory() as session:
            return list(
                await session.scalars(
                    select(RAGChunk)
                    .where(RAGChunk.document_id == document_id)
                    .order_by(RAGChunk.ordinal)
                )
            )

    async def outbox(document_id: UUID) -> list[tuple[str, int]]:
        async with factory() as session:
            rows = await session.execute(
                select(RAGOutbox.event_kind, RAGOutbox.attempt)
                .where(RAGOutbox.document_id == document_id)
                .order_by(RAGOutbox.created_at)
            )
            return [tuple(row) for row in rows]

    async def expire(document_id: UUID) -> None:
        async with factory() as session:
            await session.execute(
                update(RAGDocument)
                .where(RAGDocument.id == document_id)
                .values(lease_expires_at=func.clock_timestamp() - timedelta(seconds=1))
            )
            await session.commit()

    async def ingest(event: IngestEvent, embedder) -> HandleResult:
        return await process_ingest(
            session_factory=factory, settings=rag, event=event, embedder=embedder
        )

    return type(
        "Runtime",
        (),
        {
            "settings": rag,
            "factory": factory,
            "owner_id": owner_id,
            "submit": staticmethod(submit),
            "event_for": staticmethod(event_for),
            "read": staticmethod(read),
            "chunks": staticmethod(chunks),
            "outbox": staticmethod(outbox),
            "expire": staticmethod(expire),
            "ingest": staticmethod(ingest),
        },
    )


TEXT = " ".join(f"sentence{index} about pumps and filters." for index in range(30)).encode()


async def test_text_document_becomes_ready_in_batches(runtime):
    event = await runtime.submit(TEXT)
    embedder = ControlledEmbedder()
    assert await runtime.ingest(event, embedder) is HandleResult.HANDLED
    document = await runtime.read(event.document_id)
    stored = await runtime.chunks(event.document_id)
    assert document.status == "ready" and document.source is None
    assert document.chunk_count == len(stored) > 2 and document.error_code is None
    assert document.lease_token is None and document.finished_at is not None
    assert document.embedding_tokens == len(stored)
    # Batches of two, in ordinal order, all owned by the document's owner.
    assert [len(batch) for batch in embedder.calls] == [
        min(2, len(stored) - start) for start in range(0, len(stored), 2)
    ]
    assert [chunk.ordinal for chunk in stored] == list(range(len(stored)))
    assert {chunk.owner_id for chunk in stored} == {runtime.owner_id}
    assert stored[0].embedding == unit_vector(stored[0].content)


async def test_pdf_chunks_keep_page_numbers(runtime):
    event = await runtime.submit(
        text_pdf(["Page one mentions pumps.", "Page two mentions filters."]), ContentType.PDF
    )
    assert await runtime.ingest(event, ControlledEmbedder()) is HandleResult.HANDLED
    document = await runtime.read(event.document_id)
    stored = await runtime.chunks(event.document_id)
    assert (document.status, document.page_count) == ("ready", 2)
    assert [chunk.page for chunk in stored] == [1, 2]


async def test_parse_failure_is_terminal_with_dead_letter(runtime):
    event = await runtime.submit(b"   \n  ")
    embedder = ControlledEmbedder()
    assert await runtime.ingest(event, embedder) is HandleResult.HANDLED
    document = await runtime.read(event.document_id)
    assert (document.status, document.error_code) == ("failed", IngestError.NO_TEXT)
    assert document.source is None and embedder.calls == []
    assert await runtime.outbox(event.document_id) == [("requested", 1), ("dead_letter", 1)]


async def test_retryable_embedding_failure_starts_next_attempt(runtime):
    event = await runtime.submit(TEXT)
    embedder = ControlledEmbedder(failures={2: ProviderFailure("provider_timeout", ambiguous=True)})
    assert await runtime.ingest(event, embedder) is HandleResult.HANDLED
    document = await runtime.read(event.document_id)
    assert (document.status, document.attempt, document.error_code) == (
        "pending",
        2,
        "provider_timeout",
    )
    assert document.source is not None and document.lease_token is None
    # The first batch was stored, then removed with the failed attempt.
    assert await runtime.chunks(event.document_id) == []
    assert await runtime.outbox(event.document_id) == [("requested", 1), ("requested", 2)]


async def test_rejected_embedding_fails_without_retry(runtime):
    event = await runtime.submit(TEXT)
    embedder = ControlledEmbedder(failures={1: ProviderFailure("provider_rejected")})
    assert await runtime.ingest(event, embedder) is HandleResult.HANDLED
    document = await runtime.read(event.document_id)
    assert (document.status, document.error_code) == ("failed", IngestError.EMBEDDING_REJECTED)
    assert await runtime.outbox(event.document_id) == [("requested", 1), ("dead_letter", 1)]


async def test_last_attempt_failure_exhausts_attempts(runtime):
    event = await runtime.submit(TEXT)
    async with runtime.factory() as session:
        await session.execute(
            update(RAGOutbox).where(RAGOutbox.id == event.event_id).values(attempt=3)
        )
        await session.execute(
            update(RAGDocument).where(RAGDocument.id == event.document_id).values(attempt=3)
        )
        await session.commit()
    event = event.model_copy(update={"attempt": 3})
    embedder = ControlledEmbedder(failures={1: ProviderFailure("provider_unavailable")})
    assert await runtime.ingest(event, embedder) is HandleResult.HANDLED
    document = await runtime.read(event.document_id)
    assert (document.status, document.error_code) == ("failed", IngestError.ATTEMPTS_EXHAUSTED)


async def test_stale_worker_cannot_write_after_recovery(runtime):
    event = await runtime.submit(TEXT)
    lease, _ = await claim_document(
        session_factory=runtime.factory, settings=runtime.settings, event=event
    )
    await runtime.expire(event.document_id)
    assert await recover_documents(session_factory=runtime.factory, settings=runtime.settings)
    from app.rag.chunking import Chunk

    stored = await store_batch(
        session_factory=runtime.factory,
        settings=runtime.settings,
        lease=lease,
        chunks=[Chunk(ordinal=0, page=None, content="stale")],
        vectors=[unit_vector("stale")],
    )
    assert stored is False
    assert await runtime.chunks(event.document_id) == []
    document = await runtime.read(event.document_id)
    assert (document.status, document.attempt) == ("pending", 2)
    assert await runtime.outbox(event.document_id) == [("requested", 1), ("requested", 2)]


async def test_delete_during_ingestion_writes_nothing(runtime):
    event = await runtime.submit(TEXT)

    async def delete_document(call: int) -> None:
        if call == 2:
            async with runtime.factory() as session:
                await session.execute(
                    delete(RAGDocument).where(RAGDocument.id == event.document_id)
                )
                await session.commit()

    embedder = ControlledEmbedder(on_call=delete_document)
    assert await runtime.ingest(event, embedder) is HandleResult.HANDLED
    assert len(embedder.calls) == 2
    assert await runtime.read(event.document_id) is None
    assert await runtime.chunks(event.document_id) == []


async def test_duplicate_and_forged_events_do_nothing(runtime):
    event = await runtime.submit(TEXT)
    assert await runtime.ingest(event, ControlledEmbedder()) is HandleResult.HANDLED
    embedder = ControlledEmbedder()
    # Redelivery after ready.
    assert await runtime.ingest(event, embedder) is HandleResult.HANDLED
    # An event id that was never persisted.
    pending = await runtime.submit(b"Another document about valves.")
    forged = pending.model_copy(update={"event_id": uuid4()})
    assert await runtime.ingest(forged, embedder) is HandleResult.HANDLED
    assert embedder.calls == []
    assert (await runtime.read(pending.document_id)).status == "pending"


async def test_recovery_moves_dead_leases_and_expires_queue(runtime):
    retried = await runtime.submit(TEXT)
    await claim_document(session_factory=runtime.factory, settings=runtime.settings, event=retried)
    await runtime.expire(retried.document_id)

    exhausted = await runtime.submit(b"Third attempt document.")
    async with runtime.factory() as session:
        await session.execute(
            update(RAGDocument)
            .where(RAGDocument.id == exhausted.document_id)
            .values(
                status="processing",
                attempt=3,
                lease_token=uuid4(),
                lease_expires_at=func.clock_timestamp() - timedelta(seconds=1),
            )
        )
        await session.commit()

    expired = await runtime.submit(b"Never claimed document.")
    async with runtime.factory() as session:
        await session.execute(
            update(RAGDocument)
            .where(RAGDocument.id == expired.document_id)
            .values(deadline_at=func.clock_timestamp() - timedelta(seconds=1))
        )
        await session.commit()

    assert await recover_documents(session_factory=runtime.factory, settings=runtime.settings) == 3
    first = await runtime.read(retried.document_id)
    assert (first.status, first.attempt, first.lease_token) == ("pending", 2, None)
    second = await runtime.read(exhausted.document_id)
    assert (second.status, second.error_code) == ("failed", IngestError.ATTEMPTS_EXHAUSTED)
    assert second.source is None
    third = await runtime.read(expired.document_id)
    assert (third.status, third.error_code) == ("failed", IngestError.QUEUE_EXPIRED)
    # The retried document's next attempt is consumable.
    event = await runtime.event_for(retried.document_id, attempt=2)
    assert await runtime.ingest(event, ControlledEmbedder()) is HandleResult.HANDLED
    assert (await runtime.read(retried.document_id)).status == "ready"


async def test_cancelled_ingestion_releases_without_using_an_attempt(runtime):
    event = await runtime.submit(TEXT)
    second_call = asyncio.Event()
    never = asyncio.Event()

    async def block_second_batch(call: int) -> None:
        # The first batch is stored; the second hangs until the task is cancelled.
        if call == 2:
            second_call.set()
            await never.wait()

    task = asyncio.create_task(
        runtime.ingest(event, ControlledEmbedder(on_call=block_second_batch))
    )
    async with asyncio.timeout(10):
        await second_call.wait()
    assert len(await runtime.chunks(event.document_id)) == 2
    task.cancel()
    async with asyncio.timeout(10):
        with pytest.raises(asyncio.CancelledError):
            await task

    document = await runtime.read(event.document_id)
    assert (document.status, document.attempt) == ("pending", 1)
    assert document.lease_token is None and document.lease_expires_at is None
    assert document.source is not None
    assert await runtime.chunks(event.document_id) == []
    # The same event (redelivered because its offset was never committed) finishes it.
    assert await runtime.ingest(event, ControlledEmbedder()) is HandleResult.HANDLED
    assert (await runtime.read(event.document_id)).status == "ready"
