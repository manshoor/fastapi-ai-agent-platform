"""Owner-scoped document SQL. No provider, Kafka, or parsing I/O inside these transactions."""

import hashlib
import json
from datetime import datetime, timedelta
from uuid import UUID, uuid4

from fastapi import HTTPException
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.assistant.residence import is_manager
from app.models import RAG_EMBEDDING_DIMENSIONS, RAGChunk, RAGDocument, RAGOutbox
from app.rag.config import RAGSettings
from app.rag.contracts import (
    CHUNKER_VERSION,
    TERMINAL_STATES,
    UNSETTLED_STATES,
    DocumentResponse,
    ingest_kind,
)
from app.rag.uploads import Upload
from app.telemetry import Telemetry

# Transaction-scoped advisory lock: serializes upload admission so the owner and
# global quotas are checked and consumed atomically. Value spells "RAG1".
ADMISSION_LOCK_KEY = 0x52414731
# A tombstone's title: the uploaded one goes with the delete.
DELETED_TITLE = "deleted"
# Readers never see a tombstone.
VISIBLE = RAGDocument.status != "deleted"


def document_response(document: RAGDocument) -> DocumentResponse:
    return DocumentResponse(
        id=document.id,
        building_id=document.building_id,
        title=document.title,
        content_type=document.content_type,
        size_bytes=document.size_bytes,
        status=document.status,
        chunk_count=document.chunk_count,
        page_count=document.page_count,
        error_code=document.error_code,
        version=document.version,
        active_version=document.active_version,
        update_error_code=document.update_error_code,
        source_kind=document.source_kind,
        created_at=document.created_at,
        updated_at=document.updated_at,
        finished_at=document.finished_at,
        status_url=f"/api/v1/rag/documents/{document.id}",
    )


def _fingerprint(upload: Upload, building_id: UUID | None) -> str:
    fields = [
        "rag-upload-v1",
        upload.title,
        str(upload.content_type),
        hashlib.sha256(upload.data).hexdigest(),
    ]
    # Owner-only uploads keep the original fingerprint, so earlier replays still match.
    if building_id is not None:
        fields.append(str(building_id))
    canonical = json.dumps(
        fields,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


async def _count(session: AsyncSession, *conditions) -> int:
    return await session.scalar(select(func.count()).select_from(RAGDocument).where(*conditions))


async def _check_unsettled(session: AsyncSession, settings: RAGSettings, owner_id: UUID) -> None:
    """Global and per-owner in-progress quotas. Call under the admission lock."""
    unsettled = RAGDocument.status.in_(UNSETTLED_STATES)
    if await _count(session, unsettled) >= settings.rag_global_unsettled_limit:
        await session.rollback()
        raise HTTPException(503, "Ingestion capacity unavailable", headers={"Retry-After": "5"})
    if (
        await _count(session, unsettled, RAGDocument.owner_id == owner_id)
        >= settings.rag_owner_unsettled_limit
    ):
        await session.rollback()
        raise HTTPException(429, "Too many documents in progress", headers={"Retry-After": "5"})


async def submit_document(
    *,
    session: AsyncSession,
    settings: RAGSettings,
    owner_id: UUID,
    upload: Upload,
    idempotency_key: str,
    trace_context: dict[str, str],
    telemetry: Telemetry | None = None,
    building_id: UUID | None = None,
) -> tuple[DocumentResponse, int]:
    key_hash = hashlib.sha256(idempotency_key.encode("ascii")).hexdigest()
    fingerprint = _fingerprint(upload, building_id)

    # Replay: same key and same content returns the original document.
    async def existing() -> tuple[DocumentResponse, int] | None:
        document = await session.scalar(
            select(RAGDocument).where(
                RAGDocument.owner_id == owner_id,
                RAGDocument.idempotency_key_hash == key_hash,
                VISIBLE,
            )
        )
        if document is None:
            return None
        if document.request_fingerprint != fingerprint:
            await session.rollback()
            raise HTTPException(409, "Idempotency key conflict")
        # Read everything before rollback: rollback expires the loaded row.
        response = document_response(document)
        status_code = 200 if document.status in TERMINAL_STATES else 202
        await session.rollback()
        return response, status_code

    replay = await existing()
    if replay is not None:
        return replay

    # Admission: lock, re-check replay, then quotas, all in one transaction.
    await session.execute(select(func.pg_advisory_xact_lock(ADMISSION_LOCK_KEY)))
    replay = await existing()
    if replay is not None:
        return replay
    # Re-check under the lock: a manager removed since the route's check cannot upload.
    if building_id is not None and not await is_manager(session, owner_id, building_id):
        await session.rollback()
        raise HTTPException(404, "Building not found")
    await _check_unsettled(session, settings, owner_id)
    if (
        await _count(session, RAGDocument.owner_id == owner_id, VISIBLE)
        >= settings.rag_owner_document_limit
    ):
        await session.rollback()
        raise HTTPException(429, "Document limit reached")

    # Insert the document and its first ingestion event together.
    now: datetime = await session.scalar(select(func.clock_timestamp()))
    document_id = uuid4()
    document = RAGDocument(
        id=document_id,
        owner_id=owner_id,
        building_id=building_id,
        title=upload.title,
        content_type=str(upload.content_type),
        size_bytes=len(upload.data),
        content_sha256=hashlib.sha256(upload.data).hexdigest(),
        source=upload.data,
        idempotency_key_hash=key_hash,
        request_fingerprint=fingerprint,
        status="pending",
        attempt=1,
        next_attempt_at=now,
        deadline_at=now + timedelta(seconds=settings.rag_queue_timeout_seconds),
        embedding_model=settings.ai_embedding_model,
        embedding_dimensions=RAG_EMBEDDING_DIMENSIONS,
        chunker_version=CHUNKER_VERSION,
        trace_context=trace_context,
        created_at=now,
        updated_at=now,
    )
    session.add(document)
    await session.flush()
    session.add(
        RAGOutbox(
            document_id=document_id,
            event_kind=ingest_kind(1),
            version=1,
            attempt=1,
            available_at=now,
        )
    )
    await session.flush()
    response = document_response(document)
    await session.commit()
    if telemetry is not None:
        telemetry.increment("rag.documents", status="pending")
    return response, 202


async def list_documents(
    *, session: AsyncSession, settings: RAGSettings, owner_id: UUID
) -> list[DocumentResponse]:
    # The per-owner document quota bounds this list, so no cursor is needed.
    documents = await session.scalars(
        select(RAGDocument)
        .where(RAGDocument.owner_id == owner_id, VISIBLE)
        .order_by(RAGDocument.created_at.desc(), RAGDocument.id.desc())
        .limit(settings.rag_owner_document_limit)
    )
    result = [document_response(document) for document in documents]
    await session.rollback()
    return result


async def get_document(
    *, session: AsyncSession, owner_id: UUID, document_id: UUID
) -> DocumentResponse:
    document = await session.scalar(
        select(RAGDocument).where(
            RAGDocument.id == document_id, RAGDocument.owner_id == owner_id, VISIBLE
        )
    )
    if document is None:
        await session.rollback()
        raise HTTPException(404, "Document not found")
    response = document_response(document)
    await session.rollback()
    return response


def _update_fingerprint(upload: Upload, document_id: UUID) -> str:
    canonical = json.dumps(
        [
            "rag-update-v1",
            str(document_id),
            str(upload.content_type),
            hashlib.sha256(upload.data).hexdigest(),
        ],
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


async def _lock_visible(
    session: AsyncSession, owner_id: UUID, document_id: UUID
) -> RAGDocument | None:
    # Row lock first: a worker's fenced write holds the same lock while it stores chunks.
    return await session.scalar(
        select(RAGDocument)
        .where(RAGDocument.id == document_id, RAGDocument.owner_id == owner_id, VISIBLE)
        .with_for_update()
    )


async def update_document(
    *,
    session: AsyncSession,
    settings: RAGSettings,
    owner_id: UUID,
    document_id: UUID,
    upload: Upload,
    idempotency_key: str,
    trace_context: dict[str, str],
    telemetry: Telemetry | None = None,
) -> tuple[DocumentResponse, int]:
    """Store new content as the next version. The active version is served until it swaps.

    200: same content as the newest ready version (no new version), or a replay of a
    finished update. 202: new version queued, or a replay of one in progress.
    409: another version is in progress, or the key was used for other content.
    """
    key_hash = hashlib.sha256(idempotency_key.encode("ascii")).hexdigest()
    fingerprint = _update_fingerprint(upload, document_id)
    content_sha256 = hashlib.sha256(upload.data).hexdigest()

    # Row lock first; the global admission lock only once a new version is due, so a
    # replay, 404 or 409 never queues other owners' uploads behind a busy row.
    document = await _lock_visible(session, owner_id, document_id)
    if document is None:
        await session.rollback()
        raise HTTPException(404, "Document not found")
    # Building documents: only a current manager of that building may change them.
    if document.building_id is not None and not await is_manager(
        session, owner_id, document.building_id
    ):
        await session.rollback()
        raise HTTPException(404, "Document not found")

    # Replay of the latest update: same key and content returns the current state.
    if document.update_key_hash == key_hash:
        if document.update_fingerprint != fingerprint:
            await session.rollback()
            raise HTTPException(409, "Idempotency key conflict")
        return await _current(session, document)
    if document.status in UNSETTLED_STATES:
        await session.rollback()
        raise HTTPException(409, "Another version is in progress", headers={"Retry-After": "5"})
    # Same content as the version already served: nothing to ingest.
    if (
        document.status == "ready"
        and document.update_error_code is None
        and document.content_sha256 == content_sha256
        and document.content_type == str(upload.content_type)
    ):
        return await _current(session, document)
    # Same admission lock as uploads: quotas are checked and consumed atomically.
    await session.execute(select(func.pg_advisory_xact_lock(ADMISSION_LOCK_KEY)))
    await _check_unsettled(session, settings, owner_id)

    # New version and its event, in one transaction.
    now: datetime = await session.scalar(select(func.clock_timestamp()))
    version = document.version + 1
    document.version = version
    document.status = "pending"
    document.content_type = str(upload.content_type)
    document.size_bytes = len(upload.data)
    document.content_sha256 = content_sha256
    document.source = upload.data
    document.update_key_hash = key_hash
    document.update_fingerprint = fingerprint
    document.attempt = 1
    document.next_attempt_at = now
    document.deadline_at = now + timedelta(seconds=settings.rag_queue_timeout_seconds)
    document.error_code = None
    document.update_error_code = None
    document.embedding_tokens = None
    # A new version is embedded and chunked with the current settings.
    document.embedding_model = settings.ai_embedding_model
    document.chunker_version = CHUNKER_VERSION
    document.trace_context = trace_context
    document.updated_at = now
    document.finished_at = None
    session.add(
        RAGOutbox(
            document_id=document.id,
            event_kind=ingest_kind(version),
            version=version,
            attempt=1,
            available_at=now,
        )
    )
    await session.flush()
    response = document_response(document)
    await session.commit()
    if telemetry is not None:
        telemetry.increment("rag.documents", status="pending")
    return response, 202


async def _current(session: AsyncSession, document: RAGDocument) -> tuple[DocumentResponse, int]:
    response = document_response(document)
    status_code = 200 if document.status in TERMINAL_STATES else 202
    await session.rollback()
    return response, status_code


async def delete_document(*, session: AsyncSession, owner_id: UUID, document_id: UUID) -> None:
    """Chunks and source go now; a tombstone stays until the deleted event purges it.

    One transaction: chunks deleted, source and title cleared, status deleted (404 to
    every reader), and a deleted event. A worker mid-ingestion loses its fence and stops.
    """
    document = await _lock_visible(session, owner_id, document_id)
    if document is None:
        await session.rollback()
        raise HTTPException(404, "Document not found")
    now: datetime = await session.scalar(select(func.clock_timestamp()))
    await session.execute(delete(RAGChunk).where(RAGChunk.document_id == document_id))
    await session.execute(
        update(RAGDocument)
        .where(RAGDocument.id == document_id)
        .values(
            status="deleted",
            title=DELETED_TITLE,
            source=None,
            active_version=None,
            chunk_count=None,
            page_count=None,
            embedding_tokens=None,
            update_error_code=None,
            # Frees the upload key: the owner may reuse it for a new document.
            idempotency_key_hash=hashlib.sha256(f"deleted:{document_id}".encode()).hexdigest(),
            lease_token=None,
            lease_expires_at=None,
            finished_at=now,
            updated_at=now,
        )
        .execution_options(synchronize_session=False)
    )
    session.add(
        RAGOutbox(
            document_id=document_id,
            event_kind="deleted",
            version=document.version,
            attempt=1,
            available_at=now,
        )
    )
    await session.commit()
