"""Fenced document ingestion. No transaction spans parsing, provider, or Kafka I/O.

    doc.created / doc.updated (one version in flight per document)
      claim (pending -> processing, new lease)
        -> parse (child process for PDF) -> chunk (thread)
        -> per batch: embed (provider) -> store batch of this version + renew lease
        -> swap: active_version = version, other versions' chunks deleted (one transaction)
      failure -> retry (pending, attempt + 1, this version's chunks deleted), or
                 ready on the old active version with update_error_code, or
                 failed when no version is active; both with a dead letter
    doc.deleted
      purge the tombstone (cascades to its outbox rows)

Every write after the claim carries the fence: status, version, attempt, lease token,
lease generation, and an unexpired database-clock lease. A stale worker, or one
whose document was deleted or updated, changes nothing.
"""

import asyncio
import logging
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import UUID, uuid4

from sqlalchemy import delete, func, insert, or_, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import undefer

from app.ai.processing import HandleResult, SessionFactory
from app.ai.provider import Embeddings, ProviderFailure
from app.models import RAGChunk, RAGDocument, RAGOutbox
from app.rag.chunking import Chunk, chunk_document
from app.rag.config import RAGSettings
from app.rag.contracts import (
    EVENT_KINDS,
    INGEST_TYPES,
    TERMINAL_STATES,
    ContentType,
    DocumentEvent,
    DocumentEventType,
    IngestError,
    ingest_kind,
)
from app.rag.parsing import ParseFailure, parse_document
from app.telemetry import Telemetry

logger = logging.getLogger(__name__)

# Provider failures worth another attempt. Embedding is a pure function, so a replay
# after an ambiguous timeout or 5xx can only repeat a small charge (plan decision).
_RETRY_CODES = frozenset(
    {"provider_timeout", "provider_connection", "provider_unavailable", "provider_error"}
)
_MAX_BACKOFF_SECONDS = 60
# Recovery purges tombstones whose deleted event was never consumed.
TOMBSTONE_PURGE_SECONDS = 3600


class Embedder(Protocol):
    async def embed(self, texts: Sequence[str], *, model: str | None = None) -> Embeddings: ...


@dataclass(frozen=True)
class DocumentLease:
    document_id: UUID
    owner_id: UUID
    version: int
    attempt: int
    token: UUID
    generation: int
    content_type: ContentType
    embedding_model: str
    trace_context: dict[str, str]
    created_at: datetime
    claimed_at: datetime


def _fence(lease: DocumentLease):
    return (
        RAGDocument.id == lease.document_id,
        RAGDocument.owner_id == lease.owner_id,
        RAGDocument.status == "processing",
        RAGDocument.version == lease.version,
        RAGDocument.attempt == lease.attempt,
        RAGDocument.lease_token == lease.token,
        RAGDocument.lease_generation == lease.generation,
        RAGDocument.lease_expires_at > func.clock_timestamp(),
    )


def _record_outcome(
    telemetry: Telemetry | None, status: str, created_at: datetime, code: str | None = None
) -> None:
    if telemetry is None:
        return
    attrs = {"status": status}
    if code is not None:
        attrs["error_code"] = code
    telemetry.increment("rag.documents", **attrs)
    telemetry.observe(
        "rag.ingest.duration", max(0, (datetime.now(UTC) - created_at).total_seconds()), **attrs
    )


def _authentic(event: DocumentEvent):
    """The persisted outbox row this event claims to be: id, document, kind, version, attempt."""
    return select(RAGOutbox.id).where(
        RAGOutbox.id == event.event_id,
        RAGOutbox.document_id == event.document_id,
        RAGOutbox.event_kind == EVENT_KINDS[event.type],
        RAGOutbox.version == event.document_version,
        RAGOutbox.attempt == event.attempt,
    )


async def claim_document(
    *, session_factory: SessionFactory, settings: RAGSettings, event: DocumentEvent
) -> tuple[DocumentLease, bytes] | None:
    """Commit a short lease, accepting only an authentic persisted outbox identity."""
    if event.type not in INGEST_TYPES:
        return None
    async with session_factory() as session:
        async with session.begin():
            document = await session.scalar(
                select(RAGDocument)
                .options(undefer(RAGDocument.source))
                .where(RAGDocument.id == event.document_id)
                .with_for_update(skip_locked=True)
            )
            if (
                document is None
                or document.status != "pending"
                or document.version != event.document_version
                or document.attempt != event.attempt
                or document.source is None
            ):
                return None
            authentic = await session.scalar(_authentic(event))
            if authentic is None:
                return None
            now = await session.scalar(select(func.clock_timestamp()))
            if document.next_attempt_at > now or document.deadline_at <= now:
                return None
            document.status = "processing"
            document.lease_token = uuid4()
            document.lease_generation += 1
            document.lease_expires_at = now + timedelta(seconds=settings.rag_lease_seconds)
            document.updated_at = now
            lease = DocumentLease(
                document_id=document.id,
                owner_id=document.owner_id,
                version=document.version,
                attempt=document.attempt,
                token=document.lease_token,
                generation=document.lease_generation,
                content_type=ContentType(document.content_type),
                embedding_model=document.embedding_model,
                trace_context=dict(document.trace_context),
                created_at=document.created_at,
                claimed_at=now,
            )
            source = document.source
        return lease, source


async def store_batch(
    *,
    session_factory: SessionFactory,
    settings: RAGSettings,
    lease: DocumentLease,
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
) -> bool:
    """Insert one embedded batch and renew the lease, only while the fence holds."""
    async with session_factory() as session:
        async with session.begin():
            # Row lock: a concurrent reset or delete waits for, or blocks, this insert.
            held = await session.scalar(
                select(RAGDocument.id).where(*_fence(lease)).with_for_update()
            )
            if held is None:
                return False
            await session.execute(
                insert(RAGChunk),
                [
                    {
                        "document_id": lease.document_id,
                        "version": lease.version,
                        "owner_id": lease.owner_id,
                        "ordinal": chunk.ordinal,
                        "page": chunk.page,
                        "content": chunk.content,
                        "heading": chunk.heading,
                        "embedding": list(vector),
                    }
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )
            await session.execute(
                update(RAGDocument)
                .where(RAGDocument.id == lease.document_id)
                .values(
                    lease_expires_at=func.clock_timestamp()
                    + timedelta(seconds=settings.rag_lease_seconds),
                    updated_at=func.clock_timestamp(),
                )
            )
        return True


async def finish_ready(
    *,
    session_factory: SessionFactory,
    lease: DocumentLease,
    chunk_count: int,
    page_count: int | None,
    embedding_tokens: int | None,
    telemetry: Telemetry | None = None,
) -> bool:
    """The swap: this version becomes active and every other version's chunks go."""
    async with session_factory() as session:
        async with session.begin():
            held = await session.scalar(
                select(RAGDocument.id).where(*_fence(lease)).with_for_update()
            )
            if held is None:
                return False
            await session.execute(
                delete(RAGChunk).where(
                    RAGChunk.document_id == lease.document_id, RAGChunk.version != lease.version
                )
            )
            await session.execute(
                update(RAGDocument)
                .where(RAGDocument.id == lease.document_id)
                .values(
                    status="ready",
                    source=None,
                    active_version=lease.version,
                    chunk_count=chunk_count,
                    page_count=page_count,
                    embedding_tokens=embedding_tokens,
                    error_code=None,
                    update_error_code=None,
                    finished_at=func.clock_timestamp(),
                    updated_at=func.clock_timestamp(),
                    lease_token=None,
                    lease_expires_at=None,
                )
            )
    _record_outcome(telemetry, "ready", lease.created_at)
    return True


def _terminal_values(code: str, active_version: int | None) -> dict:
    """A version's last failure. With an older version active, that one stays served."""
    values = {
        "source": None,
        "finished_at": func.clock_timestamp(),
        "updated_at": func.clock_timestamp(),
        "lease_token": None,
        "lease_expires_at": None,
    }
    if active_version is None:
        return {**values, "status": "failed", "error_code": code}
    return {**values, "status": "ready", "error_code": None, "update_error_code": code}


def _version_chunks(document: RAGDocument):
    """Chunks of the version being ingested; never the active version's."""
    return delete(RAGChunk).where(
        RAGChunk.document_id == document.id, RAGChunk.version == document.version
    )


async def finish_failure(
    *,
    session_factory: SessionFactory,
    settings: RAGSettings,
    lease: DocumentLease,
    code: str,
    retry: bool,
    telemetry: Telemetry | None = None,
) -> bool:
    """Retry as attempt + 1, or fail with a dead letter. Partial chunks are always removed."""
    async with session_factory() as session:
        async with session.begin():
            document = await session.scalar(
                select(RAGDocument).where(*_fence(lease)).with_for_update()
            )
            if document is None:
                return False
            now = await session.scalar(select(func.clock_timestamp()))
            delay = timedelta(seconds=min(2 ** (document.attempt - 1), _MAX_BACKOFF_SECONDS))
            may_retry = (
                document.attempt < settings.rag_max_attempts and now + delay < document.deadline_at
            )
            await session.execute(_version_chunks(document))
            if retry and may_retry:
                values = {
                    "status": "pending",
                    "attempt": document.attempt + 1,
                    "next_attempt_at": now + delay,
                    "error_code": code,
                    "updated_at": now,
                    "lease_token": None,
                    "lease_expires_at": None,
                }
                event = RAGOutbox(
                    document_id=document.id,
                    event_kind=ingest_kind(document.version),
                    version=document.version,
                    attempt=document.attempt + 1,
                    available_at=now + delay,
                )
            else:
                final = IngestError.ATTEMPTS_EXHAUSTED if retry else code
                values = _terminal_values(final, document.active_version)
                event = RAGOutbox(
                    document_id=document.id,
                    event_kind="dead_letter",
                    version=document.version,
                    attempt=document.attempt,
                    available_at=now,
                )
            await session.execute(
                update(RAGDocument)
                .where(RAGDocument.id == document.id)
                .values(**values)
                .execution_options(synchronize_session=False)
            )
            session.add(event)
    if values["status"] == "pending":
        if telemetry is not None:
            telemetry.increment("ai.retries", error_code=code, operation="rag_ingest")
    else:
        _record_outcome(telemetry, "failed", lease.created_at, final)
    return True


async def _durable_without_claim(
    session_factory: SessionFactory, event: DocumentEvent
) -> HandleResult:
    async with session_factory() as session:
        row = (
            await session.execute(
                select(RAGDocument.status, RAGDocument.version, RAGDocument.attempt).where(
                    RAGDocument.id == event.document_id
                )
            )
        ).one_or_none()
        authentic = await session.scalar(_authentic(event))
    if row is None or authentic is None or row.status in TERMINAL_STATES:
        return HandleResult.HANDLED
    # An older version, or an older attempt of this one, is settled for good.
    if (row.version, row.attempt) != (event.document_version, event.attempt):
        return HandleResult.HANDLED
    # Another holder, a future retry, or an expired lease must reach recovery first.
    return HandleResult.DEFERRED


def _span(telemetry: Telemetry | None, name: str, **attrs):
    return telemetry.span(name, **attrs) if telemetry is not None else nullcontext()


async def _embed_and_store(
    *,
    session_factory: SessionFactory,
    settings: RAGSettings,
    lease: DocumentLease,
    chunks: list[Chunk],
    embedder: Embedder,
    telemetry: Telemetry | None,
) -> tuple[bool, int | None]:
    """Embed and store batch by batch. Returns (fence held, total embedding tokens)."""
    tokens: int | None = 0
    size = settings.rag_embedding_batch_size
    for start in range(0, len(chunks), size):
        batch = chunks[start : start + size]
        result = await embedder.embed(
            [chunk.content for chunk in batch], model=lease.embedding_model
        )
        tokens = (
            None if tokens is None or result.input_tokens is None else tokens + result.input_tokens
        )
        with _span(telemetry, "rag.store", document_id=str(lease.document_id)):
            stored = await store_batch(
                session_factory=session_factory,
                settings=settings,
                lease=lease,
                chunks=batch,
                vectors=result.vectors,
            )
        if not stored:
            return False, tokens
    return True, tokens


async def purge_document(*, session_factory: SessionFactory, event: DocumentEvent) -> None:
    """Remove a tombstone named by its own authentic deleted event; anything else is a no-op."""
    async with session_factory() as session:
        async with session.begin():
            authentic = await session.scalar(_authentic(event))
            if authentic is None:
                return
            await session.execute(
                delete(RAGDocument).where(
                    RAGDocument.id == event.document_id, RAGDocument.status == "deleted"
                )
            )


async def process_ingest(
    *,
    session_factory: SessionFactory,
    settings: RAGSettings,
    event: DocumentEvent,
    embedder: Embedder,
    telemetry: Telemetry | None = None,
) -> HandleResult:
    if event.type is DocumentEventType.DELETED:
        await purge_document(session_factory=session_factory, event=event)
        return HandleResult.HANDLED
    claimed = await claim_document(session_factory=session_factory, settings=settings, event=event)
    if claimed is None:
        return await _durable_without_claim(session_factory, event)
    lease, source = claimed
    try:
        return await _ingest_claimed(
            session_factory=session_factory,
            settings=settings,
            event=event,
            embedder=embedder,
            telemetry=telemetry,
            lease=lease,
            source=source,
        )
    except asyncio.CancelledError:
        # Shutdown or rebalance: hand the document back at the same attempt. The Kafka
        # offset was not committed, so the same event is delivered again.
        await asyncio.shield(_release_lease(session_factory=session_factory, lease=lease))
        raise


async def _release_lease(*, session_factory: SessionFactory, lease: DocumentLease) -> bool:
    """Fenced: back to pending, partial chunks removed, attempt unchanged."""
    try:
        async with session_factory() as session:
            async with session.begin():
                document = await session.scalar(
                    select(RAGDocument.id).where(*_fence(lease)).with_for_update()
                )
                if document is None:
                    return False
                await session.execute(
                    delete(RAGChunk).where(
                        RAGChunk.document_id == document, RAGChunk.version == lease.version
                    )
                )
                await session.execute(
                    update(RAGDocument)
                    .where(RAGDocument.id == document)
                    .values(
                        status="pending",
                        next_attempt_at=func.clock_timestamp(),
                        updated_at=func.clock_timestamp(),
                        lease_token=None,
                        lease_expires_at=None,
                    )
                )
    except (SQLAlchemyError, OSError) as exc:
        # Lease expiry recovery still covers the document.
        logger.error("RAG lease release failure exception=%s", type(exc).__name__)
        return False
    return True


async def _ingest_claimed(
    *,
    session_factory: SessionFactory,
    settings: RAGSettings,
    event: DocumentEvent,
    embedder: Embedder,
    telemetry: Telemetry | None,
    lease: DocumentLease,
    source: bytes,
) -> HandleResult:
    if telemetry is not None:
        telemetry.observe(
            "ai.queue.wait",
            max(0, (lease.claimed_at - lease.created_at).total_seconds()),
            operation="rag_ingest",
        )
    context = telemetry.extract_context(lease.trace_context) if telemetry else None
    span = (
        telemetry.span(
            "rag.ingest",
            context=context,
            document_id=str(lease.document_id),
            event_id=str(event.event_id),
            attempt=lease.attempt,
            lease_generation=lease.generation,
        )
        if telemetry
        else nullcontext()
    )
    with span:
        try:
            with _span(telemetry, "rag.parse", document_id=str(lease.document_id)):
                parsed = await parse_document(source, lease.content_type, settings)
            del source
            with _span(telemetry, "rag.chunk", document_id=str(lease.document_id)):
                chunks = await asyncio.to_thread(
                    chunk_document, parsed, lease.content_type, settings
                )
        except ParseFailure as failure:
            committed = await finish_failure(
                session_factory=session_factory,
                settings=settings,
                lease=lease,
                code=failure.code,
                retry=False,
                telemetry=telemetry,
            )
            return HandleResult.HANDLED if committed else HandleResult.DEFERRED
        try:
            held, tokens = await _embed_and_store(
                session_factory=session_factory,
                settings=settings,
                lease=lease,
                chunks=chunks,
                embedder=embedder,
                telemetry=telemetry,
            )
        except ProviderFailure as failure:
            retry = failure.retryable or failure.code in _RETRY_CODES
            committed = await finish_failure(
                session_factory=session_factory,
                settings=settings,
                lease=lease,
                code=failure.code if retry else IngestError.EMBEDDING_REJECTED,
                retry=retry,
                telemetry=telemetry,
            )
            return HandleResult.HANDLED if committed else HandleResult.DEFERRED
        if not held:
            # Lost the fence: deleted, recovered, or reclaimed. Nothing else to write.
            return HandleResult.HANDLED
        if telemetry is not None:
            telemetry.annotate(chunk_count=len(chunks), page_count=parsed.page_count)
        committed = await finish_ready(
            session_factory=session_factory,
            lease=lease,
            chunk_count=len(chunks),
            page_count=parsed.page_count,
            embedding_tokens=tokens,
            telemetry=telemetry,
        )
    return HandleResult.HANDLED if committed else HandleResult.DEFERRED


async def recover_documents(
    *,
    session_factory: SessionFactory,
    settings: RAGSettings,
    batch_size: int = 100,
    telemetry: Telemetry | None = None,
) -> int:
    """Expire stale queue entries, move dead leases to their next attempt, and purge
    tombstones whose deleted event was lost."""
    async with session_factory() as session:
        async with session.begin():
            await session.execute(
                delete(RAGDocument).where(
                    RAGDocument.status == "deleted",
                    RAGDocument.finished_at
                    <= func.clock_timestamp() - timedelta(seconds=TOMBSTONE_PURGE_SECONDS),
                )
            )
            documents = (
                await session.scalars(
                    select(RAGDocument)
                    .where(
                        or_(
                            (RAGDocument.status == "pending")
                            & (RAGDocument.deadline_at <= func.clock_timestamp()),
                            (RAGDocument.status == "processing")
                            & (RAGDocument.lease_expires_at <= func.clock_timestamp()),
                        )
                    )
                    .order_by(RAGDocument.updated_at, RAGDocument.id)
                    .limit(batch_size)
                    .with_for_update(skip_locked=True)
                )
            ).all()
            now = await session.scalar(select(func.clock_timestamp()))
            outcomes = []
            for document in documents:
                await session.execute(_version_chunks(document))
                if document.status == "pending":
                    code = IngestError.QUEUE_EXPIRED
                elif document.attempt < settings.rag_max_attempts and document.deadline_at > now:
                    code = None
                else:
                    code = IngestError.ATTEMPTS_EXHAUSTED
                document.lease_token = None
                document.lease_expires_at = None
                document.updated_at = now
                if code is None:
                    # A dead worker cannot prove how far it got: start the next attempt.
                    document.status = "pending"
                    document.attempt += 1
                    document.next_attempt_at = now
                    session.add(
                        RAGOutbox(
                            document_id=document.id,
                            event_kind=ingest_kind(document.version),
                            version=document.version,
                            attempt=document.attempt,
                            available_at=now,
                        )
                    )
                    continue
                for name, value in _terminal_values(code, document.active_version).items():
                    setattr(document, name, value)
                document.finished_at = now
                document.updated_at = now
                session.add(
                    RAGOutbox(
                        document_id=document.id,
                        event_kind="dead_letter",
                        version=document.version,
                        attempt=document.attempt,
                        available_at=now,
                    )
                )
                outcomes.append((document.created_at, code))
            depth = (
                await session.scalar(
                    select(func.count())
                    .select_from(RAGDocument)
                    .where(RAGDocument.status.in_(("pending", "processing")))
                )
                if telemetry is not None
                else 0
            )
    for created_at, code in outcomes:
        _record_outcome(telemetry, "failed", created_at, code)
    if telemetry is not None:
        telemetry.observe("rag.queue.depth", depth)
    return len(documents)
