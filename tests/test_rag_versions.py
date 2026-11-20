"""Document versions, the swap, tombstone deletes and stale events on real PostgreSQL."""

from datetime import timedelta
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select, update

from app.ai.processing import HandleResult
from app.ai.provider import ProviderFailure
from app.models import RAGChunk, RAGDocument, RAGOutbox
from app.rag.contracts import ContentType, DocumentEventType, IngestError
from app.rag.documents import delete_document, get_document, update_document
from app.rag.ingestion import (
    TOMBSTONE_PURGE_SECONDS,
    claim_document,
    recover_documents,
)
from app.rag.retrieval import OwnerScopedRetriever
from app.rag.uploads import Upload
from tests.test_rag_ingestion import ControlledEmbedder, rag_runtime  # noqa: F401

OLD = b"The pump runs at 40 litres per minute."
NEW = b"The pump runs at 55 litres per minute after the upgrade."


async def ready_document(runtime, data: bytes = OLD):
    event = await runtime.submit(data)
    assert await runtime.ingest(event, ControlledEmbedder()) is HandleResult.HANDLED
    return event.document_id


async def put_content(runtime, document_id, data: bytes = NEW, key: str | None = None):
    async with runtime.factory() as session:
        return await update_document(
            session=session,
            settings=runtime.settings,
            owner_id=runtime.owner_id,
            document_id=document_id,
            upload=Upload(title=None, content_type=ContentType.TEXT, data=data),
            idempotency_key=key or uuid4().hex,
            trace_context={},
        )


async def remove(runtime, document_id) -> None:
    async with runtime.factory() as session:
        await delete_document(session=session, owner_id=runtime.owner_id, document_id=document_id)


async def search(runtime) -> list[str]:
    retriever = OwnerScopedRetriever(
        session_factory=runtime.factory,
        embedder=ControlledEmbedder(),
        embedding_model=runtime.settings.ai_embedding_model,
        top_k=8,
        owner_id=runtime.owner_id,
    )
    return [document.page_content for document in await retriever.ainvoke("pump")]


async def chunk_versions(runtime, document_id) -> list[int]:
    async with runtime.factory() as session:
        return list(
            await session.scalars(
                select(RAGChunk.version)
                .where(RAGChunk.document_id == document_id)
                .order_by(RAGChunk.version, RAGChunk.ordinal)
            )
        )


async def test_update_serves_old_version_until_swap(runtime):
    document_id = await ready_document(runtime)
    response, status = await put_content(runtime, document_id)
    assert status == 202
    assert (response.status, response.version, response.active_version) == ("pending", 2, 1)
    # Counts still describe the served version.
    assert response.chunk_count == 1
    assert await search(runtime) == [OLD.decode()]

    seen_during_ingest: list[list[str]] = []

    async def look(call: int) -> None:
        seen_during_ingest.append(await search(runtime))

    event = await runtime.event_for(document_id, version=2)
    assert event.type is DocumentEventType.UPDATED
    assert await runtime.ingest(event, ControlledEmbedder(on_call=look)) is HandleResult.HANDLED
    assert seen_during_ingest == [[OLD.decode()]]

    document = await runtime.read(document_id)
    assert (document.status, document.version, document.active_version) == ("ready", 2, 2)
    assert document.update_error_code is None and document.source is None
    assert await search(runtime) == [NEW.decode()]
    # The swap removed the old version's chunks in the same transaction.
    assert await chunk_versions(runtime, document_id) == [2]
    assert await runtime.outbox(document_id) == [("created", 1), ("updated", 1)]


async def test_failed_update_keeps_active_version(runtime):
    document_id = await ready_document(runtime)
    await put_content(runtime, document_id)
    event = await runtime.event_for(document_id, version=2)
    embedder = ControlledEmbedder(failures={1: ProviderFailure("provider_rejected")})
    assert await runtime.ingest(event, embedder) is HandleResult.HANDLED

    document = await runtime.read(document_id)
    assert (document.status, document.version, document.active_version) == ("ready", 2, 1)
    assert document.update_error_code == IngestError.EMBEDDING_REJECTED
    assert document.error_code is None and document.source is None
    assert await search(runtime) == [OLD.decode()]
    assert await chunk_versions(runtime, document_id) == [1]
    assert await runtime.outbox(document_id) == [
        ("created", 1),
        ("updated", 1),
        ("dead_letter", 1),
    ]
    # A later update clears the failure and can succeed.
    await put_content(runtime, document_id, data=b"Third content about the pump.")
    assert (await runtime.read(document_id)).update_error_code is None


async def test_failed_first_version_stays_failed(runtime):
    event = await runtime.submit(b"   \n  ")
    assert await runtime.ingest(event, ControlledEmbedder()) is HandleResult.HANDLED
    document = await runtime.read(event.document_id)
    assert (document.status, document.active_version) == ("failed", None)
    # An update of a failed document ingests a new version.
    response, status = await put_content(runtime, event.document_id)
    assert (status, response.status, response.version, response.error_code) == (
        202,
        "pending",
        2,
        None,
    )
    retry = await runtime.event_for(event.document_id, version=2)
    assert await runtime.ingest(retry, ControlledEmbedder()) is HandleResult.HANDLED
    assert (await runtime.read(event.document_id)).active_version == 2


async def test_update_in_progress_conflicts_and_replay_matches(runtime):
    document_id = await ready_document(runtime)
    key = uuid4().hex
    first, status = await put_content(runtime, document_id, key=key)
    assert status == 202
    # The same key and content replays; other content with that key conflicts.
    again, status = await put_content(runtime, document_id, key=key)
    assert (status, again.version) == (202, 2)
    with pytest.raises(HTTPException) as conflict:
        await put_content(runtime, document_id, data=b"Other content.", key=key)
    assert conflict.value.status_code == 409
    # A different key while version 2 is in progress is also a conflict.
    with pytest.raises(HTTPException) as busy:
        await put_content(runtime, document_id, data=b"Other content.")
    assert busy.value.status_code == 409
    assert (await runtime.read(document_id)).version == 2


async def test_same_content_update_is_a_no_op(runtime):
    document_id = await ready_document(runtime)
    response, status = await put_content(runtime, document_id, data=OLD)
    assert (status, response.version, response.status) == (200, 1, "ready")
    assert await runtime.outbox(document_id) == [("created", 1)]


async def test_update_of_foreign_or_missing_document_is_404(runtime):
    document_id = await ready_document(runtime)
    async with runtime.factory() as session:
        with pytest.raises(HTTPException) as foreign:
            await update_document(
                session=session,
                settings=runtime.settings,
                owner_id=uuid4(),
                document_id=document_id,
                upload=Upload(title=None, content_type=ContentType.TEXT, data=NEW),
                idempotency_key=uuid4().hex,
                trace_context={},
            )
    assert foreign.value.status_code == 404
    with pytest.raises(HTTPException) as missing:
        await put_content(runtime, uuid4())
    assert missing.value.status_code == 404


async def test_stale_version_event_settles_without_writes(runtime):
    document_id = await ready_document(runtime)
    stale = await runtime.event_for(document_id, version=1)
    await put_content(runtime, document_id)
    # Version 1, attempt 1 redelivered while version 2, attempt 1 is pending.
    embedder = ControlledEmbedder()
    assert await runtime.ingest(stale, embedder) is HandleResult.HANDLED
    assert embedder.calls == []
    document = await runtime.read(document_id)
    assert (document.status, document.version, document.active_version) == ("pending", 2, 1)
    # A forged event naming the new version but the old event id claims nothing.
    forged = stale.model_copy(update={"document_version": 2})
    assert await runtime.ingest(forged, embedder) is HandleResult.HANDLED
    assert embedder.calls == [] and (await runtime.read(document_id)).status == "pending"


async def test_delete_leaves_tombstone_then_event_purges(runtime):
    document_id = await ready_document(runtime)
    stale = await runtime.event_for(document_id, version=1)
    assert stale.type is DocumentEventType.CREATED
    await remove(runtime, document_id)

    document = await runtime.read(document_id)
    assert (document.status, document.active_version, document.chunk_count) == (
        "deleted",
        None,
        None,
    )
    assert document.source is None and document.title != "Doc"
    assert await runtime.chunks(document_id) == []
    assert await search(runtime) == []
    async with runtime.factory() as session:
        with pytest.raises(HTTPException) as gone:
            await get_document(session=session, owner_id=runtime.owner_id, document_id=document_id)
    assert gone.value.status_code == 404
    with pytest.raises(HTTPException) as again:
        await remove(runtime, document_id)
    assert again.value.status_code == 404
    with pytest.raises(HTTPException) as no_update:
        await put_content(runtime, document_id)
    assert no_update.value.status_code == 404

    # A stale create after the delete settles; the delete event purges the row.
    assert await runtime.ingest(stale, ControlledEmbedder()) is HandleResult.HANDLED
    deleted = await _deleted_event(runtime, document_id)
    assert await runtime.ingest(deleted, ControlledEmbedder()) is HandleResult.HANDLED
    assert await runtime.read(document_id) is None
    # Redelivery after the purge is settled too.
    assert await runtime.ingest(deleted, ControlledEmbedder()) is HandleResult.HANDLED


async def _deleted_event(runtime, document_id):
    async with runtime.factory() as session:
        row = await session.scalar(
            select(RAGOutbox).where(
                RAGOutbox.document_id == document_id, RAGOutbox.event_kind == "deleted"
            )
        )
    event = await runtime.event_for(document_id, version=row.version)
    assert event.type is DocumentEventType.DELETED
    return event


async def test_delete_during_ingestion_stops_the_worker(runtime):
    document_id = await ready_document(runtime)
    await put_content(runtime, document_id)
    event = await runtime.event_for(document_id, version=2)

    async def delete_on_second(call: int) -> None:
        if call == 1:
            await remove(runtime, document_id)

    embedder = ControlledEmbedder(on_call=delete_on_second)
    assert await runtime.ingest(event, embedder) is HandleResult.HANDLED
    document = await runtime.read(document_id)
    assert document.status == "deleted"
    assert await runtime.chunks(document_id) == []


async def test_delete_frees_the_idempotency_key(runtime):
    event = await runtime.submit(OLD)
    async with runtime.factory() as session:
        key_hash = await session.scalar(
            select(RAGDocument.idempotency_key_hash).where(RAGDocument.id == event.document_id)
        )
    await remove(runtime, event.document_id)
    document = await runtime.read(event.document_id)
    assert document.idempotency_key_hash != key_hash


async def test_recovery_purges_old_tombstones_only(runtime):
    old = await ready_document(runtime)
    recent = await ready_document(runtime, b"Recent document about valves.")
    await remove(runtime, old)
    await remove(runtime, recent)
    async with runtime.factory() as session:
        await session.execute(
            update(RAGDocument)
            .where(RAGDocument.id == old)
            .values(
                finished_at=func.clock_timestamp() - timedelta(seconds=TOMBSTONE_PURGE_SECONDS + 1)
            )
        )
        await session.commit()
    await recover_documents(session_factory=runtime.factory, settings=runtime.settings)
    assert await runtime.read(old) is None
    assert (await runtime.read(recent)).status == "deleted"


async def test_recovery_of_exhausted_update_keeps_active_version(runtime):
    document_id = await ready_document(runtime)
    await put_content(runtime, document_id)
    event = await runtime.event_for(document_id, version=2)
    claimed = await claim_document(
        session_factory=runtime.factory, settings=runtime.settings, event=event
    )
    assert claimed is not None
    async with runtime.factory() as session:
        await session.execute(
            update(RAGDocument)
            .where(RAGDocument.id == document_id)
            .values(attempt=3, lease_expires_at=func.clock_timestamp() - timedelta(seconds=1))
        )
        await session.commit()
    assert await recover_documents(session_factory=runtime.factory, settings=runtime.settings)
    document = await runtime.read(document_id)
    assert (document.status, document.active_version) == ("ready", 1)
    assert document.update_error_code == IngestError.ATTEMPTS_EXHAUSTED
    assert await search(runtime) == [OLD.decode()]


async def test_retry_of_update_keeps_updated_kind(runtime):
    document_id = await ready_document(runtime)
    await put_content(runtime, document_id)
    event = await runtime.event_for(document_id, version=2)
    embedder = ControlledEmbedder(failures={1: ProviderFailure("provider_timeout", ambiguous=True)})
    assert await runtime.ingest(event, embedder) is HandleResult.HANDLED
    assert await runtime.outbox(document_id) == [("created", 1), ("updated", 1), ("updated", 2)]
    # The retry waits out its backoff; make it due now.
    async with runtime.factory() as session:
        await session.execute(
            update(RAGDocument)
            .where(RAGDocument.id == document_id)
            .values(next_attempt_at=func.clock_timestamp())
        )
        await session.commit()
    retry = await runtime.event_for(document_id, attempt=2, version=2)
    assert await runtime.ingest(retry, ControlledEmbedder()) is HandleResult.HANDLED
    assert (await runtime.read(document_id)).active_version == 2


async def test_same_bytes_with_another_type_is_a_new_version(runtime):
    document_id = await ready_document(runtime)
    async with runtime.factory() as session:
        response, status = await update_document(
            session=session,
            settings=runtime.settings,
            owner_id=runtime.owner_id,
            document_id=document_id,
            upload=Upload(title=None, content_type=ContentType.MARKDOWN, data=OLD),
            idempotency_key=uuid4().hex,
            trace_context={},
        )
    assert (status, response.version, response.content_type) == (202, 2, ContentType.MARKDOWN)


async def test_update_uses_the_current_embedding_model(runtime):
    document_id = await ready_document(runtime)
    async with runtime.factory() as session:
        await session.execute(
            update(RAGDocument)
            .where(RAGDocument.id == document_id)
            .values(embedding_model="retired-model", chunker_version="old")
        )
        await session.commit()
    await put_content(runtime, document_id)
    document = await runtime.read(document_id)
    assert document.embedding_model == runtime.settings.ai_embedding_model
    assert document.chunker_version != "old"
