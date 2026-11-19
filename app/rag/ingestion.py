"""Fenced document ingestion. No transaction spans parsing, provider, or Kafka I/O.

    claim (pending -> processing, new lease)
      -> parse (child process for PDF) -> chunk (thread)
      -> per batch: embed (provider) -> store batch + renew lease (one transaction)
      -> finish ready (source dropped)
    failure -> retry (pending, attempt + 1, chunks deleted) or failed + dead letter

Every write after the claim carries the fence: status, attempt, lease token,
lease generation, and an unexpired database-clock lease. A stale worker, or one
whose document was deleted, changes nothing.
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
from app.rag.contracts import TERMINAL_STATES, ContentType, IngestError, IngestEvent
from app.rag.parsing import ParseFailure, parse_document
from app.telemetry import Telemetry

logger = logging.getLogger(__name__)

# Provider failures worth another attempt. Embedding is a pure function, so a replay
# after an ambiguous timeout or 5xx can only repeat a small charge (plan decision).
_RETRY_CODES = frozenset(
    {"provider_timeout", "provider_connection", "provider_unavailable", "provider_error"}
)
_MAX_BACKOFF_SECONDS = 60


class Embedder(Protocol):
    async def embed(self, texts: Sequence[str], *, model: str | None = None) -> Embeddings: ...


@dataclass(frozen=True)
class DocumentLease:
    document_id: UUID
    owner_id: UUID
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


async def claim_document(
    *, session_factory: SessionFactory, settings: RAGSettings, event: IngestEvent
) -> tuple[DocumentLease, bytes] | None:
    """Commit a short lease, accepting only an authentic persisted outbox identity."""
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
                or document.attempt != event.attempt
                or document.source is None
            ):
                return None
            authentic = await session.scalar(
                select(RAGOutbox.id).where(
                    RAGOutbox.id == event.event_id,
                    RAGOutbox.document_id == document.id,
                    RAGOutbox.attempt == event.attempt,
                    RAGOutbox.event_kind == "requested",
                )
            )
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
    async with session_factory() as session:
        result = await session.execute(
            update(RAGDocument)
            .where(*_fence(lease))
            .values(
                status="ready",
                source=None,
                chunk_count=chunk_count,
                page_count=page_count,
                embedding_tokens=embedding_tokens,
                error_code=None,
                finished_at=func.clock_timestamp(),
                updated_at=func.clock_timestamp(),
                lease_token=None,
                lease_expires_at=None,
            )
        )
        await session.commit()
    if result.rowcount == 1:
        _record_outcome(telemetry, "ready", lease.created_at)
    return result.rowcount == 1


def _terminal_values(code: str) -> dict:
    return {
        "status": "failed",
        "source": None,
        "error_code": code,
        "finished_at": func.clock_timestamp(),
        "updated_at": func.clock_timestamp(),
        "lease_token": None,
        "lease_expires_at": None,
    }


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
            await session.execute(delete(RAGChunk).where(RAGChunk.document_id == document.id))
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
                    event_kind="requested",
                    attempt=document.attempt + 1,
                    available_at=now + delay,
                )
            else:
                final = IngestError.ATTEMPTS_EXHAUSTED if retry else code
                values = _terminal_values(final)
                event = RAGOutbox(
                    document_id=document.id,
                    event_kind="dead_letter",
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
        _record_outcome(telemetry, "failed", lease.created_at, values["error_code"])
    return True


async def _durable_without_claim(
    session_factory: SessionFactory, event: IngestEvent
) -> HandleResult:
    async with session_factory() as session:
        row = (
            await session.execute(
                select(RAGDocument.status, RAGDocument.attempt).where(
                    RAGDocument.id == event.document_id
                )
            )
        ).one_or_none()
        authentic = await session.scalar(
            select(RAGOutbox.id).where(
                RAGOutbox.id == event.event_id,
                RAGOutbox.document_id == event.document_id,
                RAGOutbox.attempt == event.attempt,
                RAGOutbox.event_kind == "requested",
            )
        )
    if row is None or authentic is None or row.status in TERMINAL_STATES:
        return HandleResult.HANDLED
    if row.attempt != event.attempt:
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


async def process_ingest(
    *,
    session_factory: SessionFactory,
    settings: RAGSettings,
    event: IngestEvent,
    embedder: Embedder,
    telemetry: Telemetry | None = None,
) -> HandleResult:
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
                await session.execute(delete(RAGChunk).where(RAGChunk.document_id == document))
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
    event: IngestEvent,
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
    """Expire stale queue entries and move dead leases to their next attempt."""
    async with session_factory() as session:
        async with session.begin():
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
                await session.execute(delete(RAGChunk).where(RAGChunk.document_id == document.id))
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
                            event_kind="requested",
                            attempt=document.attempt,
                            available_at=now,
                        )
                    )
                    continue
                document.status = "failed"
                document.source = None
                document.error_code = code
                document.finished_at = now
                session.add(
                    RAGOutbox(
                        document_id=document.id,
                        event_kind="dead_letter",
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
