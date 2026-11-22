"""Per-process assistant objects built by the API lifespan."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import HTTPException

from app.ai.provider import OpenAIAgentModel
from app.assistant.config import AssistantSettings
from app.assistant.graph import build_graph
from app.assistant.memory import MemoryBox
from app.assistant.step_up import SecretBox
from app.rag.runtime import RAGRuntime
from app.telemetry import Telemetry


class AssistantRuntime:
    def __init__(self, settings: AssistantSettings, telemetry: Telemetry, rag: RAGRuntime):
        self.settings = settings
        self.telemetry = telemetry
        self._slots = asyncio.Semaphore(settings.assistant_concurrency)
        self._running: set[UUID] = set()
        # Borrow the RAG runtime's HTTP client and embedder: one client per process.
        self.embedder = rag.embedder
        self.model = OpenAIAgentModel(settings, telemetry, client=rag.embedder.client)
        self.graph = build_graph(settings)
        # Seals and opens authenticator secrets for unlock step-up.
        self.secret_box = SecretBox.from_settings(settings)
        # Seals and opens memory text; None while memory is off.
        self.memory_box = (
            MemoryBox.from_settings(settings) if settings.assistant_memory_enabled else None
        )

    @asynccontextmanager
    async def turn_slot(self, user_id: UUID) -> AsyncIterator[None]:
        # Never queue: shed load at once. This set only catches a second turn in the
        # same process early; the admitted turn row decides across processes.
        if user_id in self._running:
            raise HTTPException(429, "A turn is already running", headers={"Retry-After": "1"})
        if self._slots.locked():
            raise HTTPException(503, "Assistant capacity busy", headers={"Retry-After": "1"})
        self._running.add(user_id)
        try:
            async with self._slots:
                yield
        finally:
            self._running.discard(user_id)
