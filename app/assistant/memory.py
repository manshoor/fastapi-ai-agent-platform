"""Notes the resident asked the assistant to keep: sealed, scoped, limited and forgotten.

    write   lock user row -> residence scope needs a current residence (403)
            -> active memories < ASSISTANT_MEMORY_LIMIT, else 429
            -> writes in the last day < ASSISTANT_MEMORY_WRITES_PER_DAY, else 429
            -> seal -> insert memory + write row, one transaction, database clock
            The route and the `remember` tool share this core; each commits itself.
    read    unexpired, and residence memories only while the user lives in that building
    recall  a turn reads at most MAX_RECALLED of them, newest first; a row that does not
            open is skipped and logged, never fatal
    forget  hard delete: one memory (404 if not the caller's) or all of them
    purge   the dispatcher tick deletes expired and moved-out memories, and day-old writes

Text is sealed with AES-256-GCM under ASSISTANT_MEMORY_KEY. The associated data binds a
ciphertext to its purpose, user and row, so a row copied to another user or id does not
open. Memory text is never logged, traced, or stored anywhere but the sealed column.
"""

import base64
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import HTTPException
from sqlalchemy import and_, delete, exists, func, insert, or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.processing import SessionFactory
from app.assistant.config import DAY_SECONDS, AssistantSettings
from app.assistant.contracts import MemoryList, MemoryScope, MemoryView, Role
from app.assistant.residence import load_resident
from app.models import AssistantMemory, AssistantMemoryWrite, Membership, User

logger = logging.getLogger(__name__)
NONCE_BYTES = 12
# Purpose label in the associated data: this key seals memories and nothing else.
ASSOCIATED_PREFIX = b"memory-v1"
KEY_VERSION = 1
# Rows each purge statement removes per dispatcher tick.
PURGE_BATCH = 500
# A turn recalls at most this many memories, newest first, within this many characters.
MAX_RECALLED = 20
MAX_RECALLED_CHARACTERS = 4000
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_TOO_MANY_REQUESTS = 429
HTTP_INTERNAL_ERROR = 500
NO_RESIDENCE = "No residence for this account"
LIMIT_REACHED = "Memory limit reached"
WRITES_EXHAUSTED = "Memory write limit reached"
NOT_FOUND = "Memory not found"
UNAVAILABLE = "Memory unavailable"


class MemoryBox:
    """AES-256-GCM sealing for memory text, bound to its user and row."""

    def __init__(self, key: bytes) -> None:
        self._aead = AESGCM(key)

    @classmethod
    def from_settings(cls, settings: AssistantSettings) -> "MemoryBox":
        # The settings validator already checked: base64 for exactly 32 bytes.
        key = settings.assistant_memory_key.get_secret_value()
        return cls(base64.b64decode(key, validate=True))

    @staticmethod
    def _associated(user_id: UUID, memory_id: UUID) -> bytes:
        return ASSOCIATED_PREFIX + user_id.bytes + memory_id.bytes

    def seal(self, user_id: UUID, memory_id: UUID, text: str) -> tuple[bytes, bytes]:
        nonce = os.urandom(NONCE_BYTES)
        sealed = self._aead.encrypt(nonce, text.encode(), self._associated(user_id, memory_id))
        return nonce, sealed

    def open(self, user_id: UUID, memory_id: UUID, nonce: bytes, ciphertext: bytes) -> str:
        # Raises InvalidTag for another user's or row's ciphertext, a change, or another key.
        associated = self._associated(user_id, memory_id)
        return self._aead.decrypt(nonce, ciphertext, associated).decode()


def _lives_in_its_building():
    """SQL: the memory's user is a resident of the memory's building (residence scope)."""
    return exists().where(
        Membership.user_id == AssistantMemory.user_id,
        Membership.building_id == AssistantMemory.building_id,
        Membership.role == Role.RESIDENT,
    )


def _servable(user_id: UUID, building_id: UUID | None) -> list:
    """SQL conditions: the user's memories that may be read now."""
    current = [AssistantMemory.user_id == user_id, AssistantMemory.expires_at > func.now()]
    if building_id is None:
        return [*current, AssistantMemory.scope == MemoryScope.RESIDENT]
    lives_there = exists().where(
        Membership.user_id == user_id,
        Membership.building_id == building_id,
        Membership.role == Role.RESIDENT,
    )
    home = and_(
        AssistantMemory.scope == MemoryScope.RESIDENCE,
        AssistantMemory.building_id == building_id,
        lives_there,
    )
    return [*current, or_(AssistantMemory.scope == MemoryScope.RESIDENT, home)]


class Refusal(StrEnum):
    """Why a memory write was refused. Each route and tool maps these to its own reply."""

    NO_RESIDENCE = "no_residence"
    LIMIT_REACHED = "limit_reached"
    WRITES_EXHAUSTED = "writes_exhausted"


@dataclass(frozen=True)
class Refused:
    reason: Refusal
    # Seconds until the oldest write leaves the one-day window (WRITES_EXHAUSTED only).
    retry_after: int | None = None


@dataclass(frozen=True)
class Recalled:
    """One memory opened for a turn. The turn labels them M1..Mn in this order."""

    memory_id: UUID
    text: str = field(repr=False)


async def lock_writer(session: AsyncSession, user_id: UUID) -> datetime:
    """Lock the user row and return the database clock. Take it before any other lock.

    The lock serializes every memory write of one user, from the route and the tool
    alike, so the limits hold under races.
    """
    await session.execute(select(User.id).where(User.id == user_id).with_for_update())
    return await session.scalar(select(func.clock_timestamp()))


async def write_memory(
    session: AsyncSession,
    box: MemoryBox,
    settings: AssistantSettings,
    user_id: UUID,
    scope: MemoryScope,
    text: str,
    now: datetime,
) -> MemoryView | Refused:
    """Check the limits and insert one memory and its write row. Commits nothing.

    The caller holds `lock_writer` in this transaction and commits or rolls back.
    """
    # Residence scope is tied to the building the user lives in now. The share lock
    # makes a move-out wait until this write commits.
    building_id = None
    if scope is MemoryScope.RESIDENCE:
        building_id = await session.scalar(
            select(Membership.building_id)
            .where(Membership.user_id == user_id, Membership.role == Role.RESIDENT)
            .with_for_update(read=True)
        )
        if building_id is None:
            return Refused(Refusal.NO_RESIDENCE)

    # Active memories: unexpired, and residence ones only while the user lives there,
    # so rows a read would hide never block a write before the purge removes them.
    active = await session.scalar(
        select(func.count())
        .select_from(AssistantMemory)
        .where(
            AssistantMemory.user_id == user_id,
            AssistantMemory.expires_at > now,
            or_(AssistantMemory.scope == MemoryScope.RESIDENT, _lives_in_its_building()),
        )
    )
    if active >= settings.assistant_memory_limit:
        return Refused(Refusal.LIMIT_REACHED)

    # Writes in the last day; the retry time is when the oldest one leaves the window.
    window_start = now - func.make_interval(0, 0, 0, 0, 0, 0, DAY_SECONDS)
    writes, oldest = (
        await session.execute(
            select(func.count(), func.min(AssistantMemoryWrite.created_at)).where(
                AssistantMemoryWrite.user_id == user_id,
                AssistantMemoryWrite.created_at > window_start,
            )
        )
    ).one()
    if writes >= settings.assistant_memory_writes_per_day:
        retry = await session.scalar(
            select(
                func.ceil(
                    func.extract(
                        "epoch",
                        oldest + func.make_interval(0, 0, 0, 0, 0, 0, DAY_SECONDS) - now,
                    )
                )
            )
        )
        return Refused(Refusal.WRITES_EXHAUSTED, retry_after=max(int(retry), 1))

    # Seal with the row's own id, then write the memory and its write row together.
    memory_id = uuid4()
    nonce, ciphertext = box.seal(user_id, memory_id, text)
    retention = func.make_interval(0, 0, 0, 0, 0, 0, settings.assistant_memory_retention_seconds)
    created_at, expires_at = (
        await session.execute(
            insert(AssistantMemory)
            .values(
                id=memory_id,
                user_id=user_id,
                scope=str(scope),
                building_id=building_id,
                ciphertext=ciphertext,
                nonce=nonce,
                key_version=KEY_VERSION,
                created_at=now,
                expires_at=now + retention,
            )
            .returning(AssistantMemory.created_at, AssistantMemory.expires_at)
        )
    ).one()
    await session.execute(insert(AssistantMemoryWrite).values(user_id=user_id, created_at=now))
    return MemoryView(
        id=memory_id, scope=scope, text=text, created_at=created_at, expires_at=expires_at
    )


async def add_memory(
    session: AsyncSession,
    box: MemoryBox,
    settings: AssistantSettings,
    user_id: UUID,
    scope: MemoryScope,
    text: str,
) -> MemoryView:
    """The route's write: one memory under the user's lock, committed, or an HTTP error."""
    now = await lock_writer(session, user_id)
    result = await write_memory(session, box, settings, user_id, scope, text, now)
    if not isinstance(result, Refused):
        await session.commit()
        return result
    await session.rollback()
    if result.reason is Refusal.NO_RESIDENCE:
        raise HTTPException(HTTP_FORBIDDEN, NO_RESIDENCE)
    if result.reason is Refusal.LIMIT_REACHED:
        raise HTTPException(HTTP_TOO_MANY_REQUESTS, LIMIT_REACHED)
    raise HTTPException(
        HTTP_TOO_MANY_REQUESTS,
        WRITES_EXHAUSTED,
        headers={"Retry-After": str(result.retry_after)},
    )


async def recall_memories(
    session_factory: SessionFactory, box: MemoryBox, user_id: UUID, building_id: UUID
) -> list[Recalled]:
    """Servable memories for one turn, newest first, within the count and size caps.

    A row that does not open is skipped, so one bad row never fails every turn.
    """
    async with session_factory() as session:
        rows = (
            await session.execute(
                select(AssistantMemory.id, AssistantMemory.nonce, AssistantMemory.ciphertext)
                .where(*_servable(user_id, building_id))
                .order_by(AssistantMemory.created_at.desc(), AssistantMemory.id.desc())
                .limit(MAX_RECALLED)
            )
        ).all()
        await session.rollback()
    recalled: list[Recalled] = []
    characters = 0
    for row in rows:
        try:
            text = box.open(user_id, row.id, row.nonce, row.ciphertext)
        except InvalidTag:
            # A changed key or row. Logged without the id or any text.
            logger.error("Assistant memory failed to open")
            continue
        # Newest first: the oldest notes are the ones left out.
        if characters + len(text) > MAX_RECALLED_CHARACTERS:
            break
        characters += len(text)
        recalled.append(Recalled(memory_id=row.id, text=text))
    return recalled


async def list_memories(session: AsyncSession, box: MemoryBox, user_id: UUID) -> MemoryList:
    """Servable memories, newest first. A row that does not open is a server fault (500)."""
    resident = await load_resident(session, user_id)
    building_id = resident.building_id if resident is not None else None
    # Plain columns: ORM rows would expire on the rollback below.
    rows = (
        await session.execute(
            select(
                AssistantMemory.id,
                AssistantMemory.scope,
                AssistantMemory.nonce,
                AssistantMemory.ciphertext,
                AssistantMemory.created_at,
                AssistantMemory.expires_at,
            )
            .where(*_servable(user_id, building_id))
            .order_by(AssistantMemory.created_at.desc(), AssistantMemory.id.desc())
        )
    ).all()
    await session.rollback()
    memories = []
    for row in rows:
        try:
            text = box.open(user_id, row.id, row.nonce, row.ciphertext)
        except InvalidTag:
            # A changed key or row. Logged without the id or any text.
            logger.error("Assistant memory failed to open")
            raise HTTPException(HTTP_INTERNAL_ERROR, UNAVAILABLE) from None
        memories.append(
            MemoryView(
                id=row.id,
                scope=MemoryScope(row.scope),
                text=text,
                created_at=row.created_at,
                expires_at=row.expires_at,
            )
        )
    return MemoryList(memories=memories)


async def delete_memory(session: AsyncSession, user_id: UUID, memory_id: UUID) -> bool:
    """Hard delete one of the user's memories. Commits nothing; False if none matched."""
    deleted = await session.scalar(
        delete(AssistantMemory)
        .where(AssistantMemory.id == memory_id, AssistantMemory.user_id == user_id)
        .returning(AssistantMemory.id)
    )
    return deleted is not None


async def forget_memory(session: AsyncSession, user_id: UUID, memory_id: UUID) -> None:
    """Hard delete one of the caller's memories; another user's and a missing one are 404."""
    if not await delete_memory(session, user_id, memory_id):
        await session.rollback()
        raise HTTPException(HTTP_NOT_FOUND, NOT_FOUND)
    await session.commit()


async def forget_all(session: AsyncSession, user_id: UUID) -> None:
    """Hard delete every memory of the caller, servable or not. The write log stays."""
    await session.execute(delete(AssistantMemory).where(AssistantMemory.user_id == user_id))
    await session.commit()


async def purge_memories(*, session_factory: SessionFactory, batch: int = PURGE_BATCH) -> int:
    """Delete expired and moved-out memories and day-old write rows; returns memories removed.

    Reads already skip these rows, so the purge bounds storage, not exposure.
    """
    # Two statements, not one OR: the expiry one uses its index on expires_at.
    expired = (
        select(AssistantMemory.id)
        .where(AssistantMemory.expires_at <= func.now())
        .limit(batch)
        .with_for_update(skip_locked=True)
    )
    moved_out = (
        select(AssistantMemory.id)
        .where(AssistantMemory.scope == MemoryScope.RESIDENCE, ~_lives_in_its_building())
        .limit(batch)
        .with_for_update(skip_locked=True)
    )
    window_start = func.now() - func.make_interval(0, 0, 0, 0, 0, 0, DAY_SECONDS)
    old_writes = (
        select(AssistantMemoryWrite.id)
        .where(AssistantMemoryWrite.created_at <= window_start)
        .limit(batch)
        .with_for_update(skip_locked=True)
    )
    async with session_factory() as session:
        async with session.begin():
            removed = 0
            for stale in (expired, moved_out):
                removed += len(
                    (
                        await session.scalars(
                            delete(AssistantMemory)
                            .where(AssistantMemory.id.in_(stale))
                            .returning(AssistantMemory.id)
                        )
                    ).all()
                )
            await session.execute(
                delete(AssistantMemoryWrite).where(AssistantMemoryWrite.id.in_(old_writes))
            )
    return removed


async def purge_tick(session_factory: SessionFactory) -> int:
    """The dispatcher's purge step: a database failure is logged and the tick goes on,
    so a slow purge never stops job and event publishing. Reads never need the purge."""
    try:
        return await purge_memories(session_factory=session_factory)
    except SQLAlchemyError as error:
        logger.error("Assistant memory purge failed exception=%s", type(error).__name__)
        return 0
