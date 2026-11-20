"""One connector run: lock, fetch, apply each record, delete what a complete run missed.

    pg_try_advisory_lock(connector)       one runner per connector across processes
      close a stale "running" row; skip unless due; insert run row
      owner must still manage the building, or the run fails before any write
      fetch (no transaction open)
      per record, one short transaction:
        new id           -> document + doc.created     (admission lock, connector cap)
        version busy     -> unchanged
        same content     -> unchanged (title updated in place); a version that failed
                            for lack of time or attempts gets a new version
        changed content  -> next version + doc.updated (admission lock, connector cap)
      first, documents left in another building or owner by an earlier config: tombstone
      duplicate ids in one fetch -> the first applies, the rest are skipped
      complete, non-empty fetch and no cap hit -> ids not seen: tombstone + doc.deleted
    pg_advisory_unlock

Run rows hold counts and fixed codes only. Nothing here logs content, queries or URLs.
"""

import hashlib
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timedelta
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import ValidationError
from sqlalchemy import func, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.ai.processing import SessionFactory
from app.assistant.residence import is_manager
from app.models import RAG_EMBEDDING_DIMENSIONS, RAGConnectorRun, RAGDocument, RAGOutbox
from app.rag.config import RAGSettings
from app.rag.connectors.config import Connector, HTTPSource
from app.rag.connectors.sources import Fetch, Record, fetch_http, fetch_postgres
from app.rag.contracts import (
    CHUNKER_VERSION,
    UNSETTLED_STATES,
    ContentType,
    IngestError,
    UploadTitle,
    ingest_kind,
)
from app.rag.documents import ADMISSION_LOCK_KEY, VISIBLE, start_version, tombstone
from app.rag.uploads import Upload

logger = logging.getLogger(__name__)

# Connector records are plain text.
CONTENT_TYPE = ContentType.TEXT
# Final failures caused by time, not content: the same content gets another version.
RETRY_CODES = frozenset({IngestError.ATTEMPTS_EXHAUSTED, IngestError.QUEUE_EXPIRED})
# Codes that end a run with nothing applied.
_FETCH_PARTIAL = frozenset({"record_limit", "page_limit", "invalid_record_id"})

Fetcher = Callable[[Connector, Mapping[str, str]], Awaitable[Fetch]]


class Outcome(StrEnum):
    CREATED = "created"
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    SKIPPED = "skipped"


class _Capacity(Exception):
    """The connector or global in-progress cap is full: stop, delete nothing."""


def lock_key(name: str) -> int:
    """A stable signed 64-bit advisory lock key per connector name."""
    digest = hashlib.sha256(f"rag-connector:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=True)


async def default_fetch(connector: Connector, environ: Mapping[str, str]) -> Fetch:
    if isinstance(connector.source, HTTPSource):
        return await fetch_http(connector.source, connector.max_records, environ)
    return await fetch_postgres(connector.source, connector.max_records, environ)


def _upload(record: Record, settings: RAGSettings) -> tuple[str, Upload] | None:
    """A usable title and text, or None: the record is skipped but still counts as seen."""
    if not isinstance(record.content, str) or "\x00" in record.content:
        return None
    try:
        data = record.content.encode("utf-8", "strict") if record.content.strip() else b""
    except UnicodeEncodeError:
        # A lone surrogate from JSON cannot be stored as text.
        return None
    if not data or len(data) > settings.rag_text_max_bytes:
        return None
    try:
        title = UploadTitle.model_validate({"title": record.title}).title
    except ValidationError:
        return None
    return title, Upload(title=title, content_type=CONTENT_TYPE, data=data)


async def _admit(session: AsyncSession, settings: RAGSettings) -> None:
    """Uploads' admission lock, then the global and the connectors' in-progress caps."""
    await session.execute(select(func.pg_advisory_xact_lock(ADMISSION_LOCK_KEY)))
    unsettled = RAGDocument.status.in_(UNSETTLED_STATES)
    count = select(func.count()).select_from(RAGDocument).where(unsettled)
    if await session.scalar(count) >= settings.rag_global_unsettled_limit:
        raise _Capacity
    connectors = count.where(RAGDocument.source_kind == "connector")
    if await session.scalar(connectors) >= settings.rag_connector_unsettled_limit:
        raise _Capacity


def _record_key(connector: Connector, external_id: str) -> str:
    return hashlib.sha256(f"connector:{connector.name}:{external_id}".encode()).hexdigest()


async def _create(
    session: AsyncSession,
    settings: RAGSettings,
    connector: Connector,
    record: Record,
    upload: Upload,
) -> None:
    now: datetime = await session.scalar(select(func.clock_timestamp()))
    digest = hashlib.sha256(upload.data).hexdigest()
    document = RAGDocument(
        id=uuid4(),
        owner_id=connector.owner_user_id,
        building_id=connector.building_id,
        title=upload.title,
        content_type=str(upload.content_type),
        size_bytes=len(upload.data),
        content_sha256=digest,
        source=upload.data,
        idempotency_key_hash=_record_key(connector, record.external_id),
        request_fingerprint=hashlib.sha256(
            json.dumps(["rag-connector-v1", connector.name, record.external_id, digest]).encode()
        ).hexdigest(),
        status="pending",
        version=1,
        attempt=1,
        next_attempt_at=now,
        deadline_at=now + timedelta(seconds=settings.rag_queue_timeout_seconds),
        embedding_model=settings.ai_embedding_model,
        embedding_dimensions=RAG_EMBEDDING_DIMENSIONS,
        chunker_version=CHUNKER_VERSION,
        trace_context={},
        source_kind="connector",
        connector_name=connector.name,
        external_id=record.external_id,
        created_at=now,
        updated_at=now,
    )
    session.add(document)
    await session.flush()
    session.add(
        RAGOutbox(
            document_id=document.id,
            event_kind=ingest_kind(1),
            version=1,
            attempt=1,
            available_at=now,
        )
    )


async def apply_record(
    *,
    session_factory: SessionFactory,
    settings: RAGSettings,
    connector: Connector,
    record: Record,
) -> Outcome:
    valid = _upload(record, settings)
    if valid is None:
        return Outcome.SKIPPED
    title, upload = valid
    async with session_factory() as session:
        async with session.begin():
            # Row lock first, then the admission lock: the order update_document uses.
            document = await session.scalar(
                select(RAGDocument)
                .where(
                    RAGDocument.connector_name == connector.name,
                    RAGDocument.external_id == record.external_id,
                    VISIBLE,
                )
                .with_for_update()
            )
            if document is None:
                await _admit(session, settings)
                await _create(session, settings, connector, record, upload)
                return Outcome.CREATED
            if document.status in UNSETTLED_STATES:
                # A version is being ingested; the next run compares again.
                return Outcome.UNCHANGED
            same = document.content_sha256 == hashlib.sha256(upload.data).hexdigest()
            timed_out = (document.error_code in RETRY_CODES) or (
                document.update_error_code in RETRY_CODES
            )
            if same and not timed_out:
                # Content failures (no text, limits) are not retried on every run.
                if document.title == title:
                    return Outcome.UNCHANGED
                document.title = title
                document.updated_at = func.clock_timestamp()
                return Outcome.UPDATED
            await _admit(session, settings)
            document.title = title
            await start_version(session, settings, document, upload, {})
            return Outcome.UPDATED


async def _tombstone_each(
    session_factory: SessionFactory, connector: Connector, document_ids: list[UUID]
) -> int:
    """Tombstone each still-visible document of this connector, one transaction each."""
    removed = 0
    for document_id in document_ids:
        async with session_factory() as session:
            async with session.begin():
                document = await session.scalar(
                    select(RAGDocument)
                    .where(
                        RAGDocument.id == document_id,
                        RAGDocument.connector_name == connector.name,
                        VISIBLE,
                    )
                    .with_for_update()
                )
                if document is None:
                    continue
                await tombstone(session, document)
                removed += 1
    return removed


async def _visible(session_factory: SessionFactory, connector: Connector) -> list:
    async with session_factory() as session:
        rows = (
            await session.execute(
                select(
                    RAGDocument.id,
                    RAGDocument.external_id,
                    RAGDocument.building_id,
                    RAGDocument.owner_id,
                ).where(RAGDocument.connector_name == connector.name, VISIBLE)
            )
        ).all()
        await session.rollback()
    return rows


async def remove_missing(
    *, session_factory: SessionFactory, connector: Connector, seen: set[str]
) -> int:
    """Tombstone this connector's live documents whose record ids the run did not see."""
    rows = await _visible(session_factory, connector)
    missing = [row.id for row in rows if row.external_id not in seen]
    return await _tombstone_each(session_factory, connector, missing)


async def remove_reassigned(*, session_factory: SessionFactory, connector: Connector) -> int:
    """Tombstone documents left in another building or owner by an earlier config.

    They would otherwise keep getting this source's versions while served to the old
    building's residents.
    """
    rows = await _visible(session_factory, connector)
    moved = [
        row.id
        for row in rows
        if (row.building_id, row.owner_id) != (connector.building_id, connector.owner_user_id)
    ]
    return await _tombstone_each(session_factory, connector, moved)


async def _finish(
    session_factory: SessionFactory,
    run_id: UUID,
    status: str,
    counts: dict[str, int],
    error_code: str | None,
) -> None:
    async with session_factory() as session:
        await session.execute(
            update(RAGConnectorRun)
            .where(RAGConnectorRun.id == run_id, RAGConnectorRun.status == "running")
            .values(
                status=status,
                error_code=error_code,
                finished_at=func.clock_timestamp(),
                **counts,
            )
        )
        await session.commit()


async def _start(session_factory: SessionFactory, connector: Connector) -> tuple[UUID, bool] | None:
    """Close a stale run, check the interval and the owner, insert the run row."""
    async with session_factory() as session:
        async with session.begin():
            # Holding the lock proves no other runner: a running row is a crashed run.
            await session.execute(
                update(RAGConnectorRun)
                .where(
                    RAGConnectorRun.connector_name == connector.name,
                    RAGConnectorRun.status == "running",
                )
                .values(
                    status="failed",
                    error_code="interrupted",
                    finished_at=func.clock_timestamp(),
                )
            )
            last = await session.scalar(
                select(func.max(RAGConnectorRun.started_at)).where(
                    RAGConnectorRun.connector_name == connector.name
                )
            )
            now: datetime = await session.scalar(select(func.clock_timestamp()))
            if last is not None and now - last < timedelta(seconds=connector.interval_seconds):
                return None
            run = RAGConnectorRun(id=uuid4(), connector_name=connector.name, started_at=now)
            session.add(run)
            allowed = await is_manager(session, connector.owner_user_id, connector.building_id)
            return run.id, allowed


async def _run_locked(
    *,
    session_factory: SessionFactory,
    settings: RAGSettings,
    connector: Connector,
    environ: Mapping[str, str],
    fetcher: Fetcher,
) -> UUID | None:
    started = await _start(session_factory, connector)
    if started is None:
        return None
    run_id, allowed = started
    counts = {outcome.value: 0 for outcome in Outcome} | {"fetched": 0, "deleted": 0}
    if not allowed:
        await _finish(session_factory, run_id, "failed", counts, "owner_not_manager")
        return run_id
    try:
        counts["deleted"] = await remove_reassigned(
            session_factory=session_factory, connector=connector
        )
        fetch = await fetcher(connector, environ)
        counts["fetched"] = len(fetch.records)
        if fetch.error_code is not None and fetch.error_code not in _FETCH_PARTIAL:
            await _finish(session_factory, run_id, "failed", counts, fetch.error_code)
            return run_id
        seen: set[str] = set()
        stopped = None
        for record in fetch.records:
            if record.external_id in seen:
                # Two records with one id: the first wins, or each run would swap them.
                counts[Outcome.SKIPPED.value] += 1
                continue
            seen.add(record.external_id)
            try:
                outcome = await apply_record(
                    session_factory=session_factory,
                    settings=settings,
                    connector=connector,
                    record=record,
                )
            except _Capacity:
                stopped = "capacity"
                break
            counts[outcome.value] += 1
        if fetch.complete and not fetch.records:
            # An empty answer is more often an outage or reload than a real wipe.
            stopped = "empty_source"
        if fetch.complete and stopped is None:
            counts["deleted"] += await remove_missing(
                session_factory=session_factory, connector=connector, seen=seen
            )
            await _finish(session_factory, run_id, "complete", counts, None)
        else:
            await _finish(session_factory, run_id, "partial", counts, stopped or fetch.error_code)
    except SQLAlchemyError:
        # Best effort: the next run closes the row as interrupted if this fails too.
        try:
            await _finish(session_factory, run_id, "failed", counts, "database_error")
        except SQLAlchemyError:
            pass
        raise
    return run_id


async def run_connector(
    *,
    engine: AsyncEngine,
    session_factory: SessionFactory,
    settings: RAGSettings,
    connector: Connector,
    environ: Mapping[str, str],
    fetcher: Fetcher = default_fetch,
) -> UUID | None:
    """Run once if due and unlocked. Returns the run id, or None when skipped."""
    key = lock_key(connector.name)
    async with engine.connect() as lock:
        # A session-level lock on its own connection: held across the run's transactions.
        held = await lock.scalar(select(func.pg_try_advisory_lock(key)))
        await lock.commit()
        if not held:
            return None
        try:
            return await _run_locked(
                session_factory=session_factory,
                settings=settings,
                connector=connector,
                environ=environ,
                fetcher=fetcher,
            )
        finally:
            try:
                await lock.execute(select(func.pg_advisory_unlock(key)))
                await lock.commit()
            except (SQLAlchemyError, OSError):
                # A discarded connection ends its session, and the lock with it.
                await lock.invalidate()
