"""Per-process RAG objects built and closed by the API lifespan."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import HTTPException

from app.rag.config import RAGSettings
from app.telemetry import Telemetry


@asynccontextmanager
async def _slot(slots: asyncio.Semaphore, detail: str) -> AsyncIterator[None]:
    # Never queue: a full process sheds load at once instead of holding sockets open.
    if slots.locked():
        raise HTTPException(503, detail, headers={"Retry-After": "1"})
    async with slots:
        yield


@asynccontextmanager
async def _owner_slot(active: set[UUID], owner_id: UUID) -> AsyncIterator[None]:
    # One upload per owner per process: one account cannot hold every global slot.
    if owner_id in active:
        raise HTTPException(429, "Upload already in progress", headers={"Retry-After": "1"})
    active.add(owner_id)
    try:
        yield
    finally:
        active.discard(owner_id)


class RAGRuntime:
    def __init__(self, settings: RAGSettings, telemetry: Telemetry):
        self.settings = settings
        self.telemetry = telemetry
        self._upload_slots = asyncio.Semaphore(settings.rag_upload_concurrency)
        self._query_slots = asyncio.Semaphore(settings.rag_query_concurrency)
        self._uploading: set[UUID] = set()

    @asynccontextmanager
    async def upload_slot(self, owner_id: UUID) -> AsyncIterator[None]:
        async with _owner_slot(self._uploading, owner_id):
            async with _slot(self._upload_slots, "Upload capacity busy"):
                yield

    def query_slot(self):
        return _slot(self._query_slots, "Query capacity busy")

    async def aclose(self) -> None:
        return None
