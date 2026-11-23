"""Shared helpers for resident assistant tests: simulator, residences, ready chunks."""

import asyncio
import base64
import importlib.util
import json
import socket
import sys
from contextlib import asynccontextmanager
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from uuid import UUID, uuid4

import uvicorn
from fastapi import FastAPI
from sqlalchemy import func, insert

from app.ai.provider import ToolCall
from app.assistant.residence import load_resident
from app.assistant.tools import ToolContext, execute
from app.assistant.totp import code_at
from app.models import (
    RAG_EMBEDDING_DIMENSIONS,
    Apartment,
    AssistantTurn,
    Membership,
    RAGChunk,
    RAGDocument,
)

EMBEDDING_MODEL = "text-embedding-3-small"
# Test-only literal, as in compose.ai.test.yaml: base64 of "test-only-totp-key-not-a-secret!".
TOTP_KEY = "dGVzdC1vbmx5LXRvdHAta2V5LW5vdC1hLXNlY3JldCE="
# The `account` fixture's default password.
ACCOUNT_PASSWORD = "a-secure-password-123"
TOTP_PATH = "/api/v1/assistant/totp"
FRONT_DOOR = "Front door"


def _load_simulator():
    path = Path(__file__).parent / "provider_simulator.py"
    spec = importlib.util.spec_from_file_location("provider_simulator_assistant", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["provider_simulator_assistant"] = module
    spec.loader.exec_module(module)
    return module


SIMULATOR = _load_simulator()


@asynccontextmanager
async def serve(application):
    """Serve an ASGI app on a free loopback port for the length of one test."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    server = uvicorn.Server(
        uvicorn.Config(application, access_log=False, log_level="critical", lifespan="off")
    )
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(3):
            while not server.started:
                if task.done():
                    await task
                await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 3)
        sock.close()


class StepClock:
    """Stands in for the database clock's TOTP step, so a test decides which codes are fresh.

    Patch it over app.assistant.step_up._database_step.
    """

    def __init__(self) -> None:
        self.step = 59_000_000

    async def __call__(self, session) -> int:
        return self.step


async def enable_authenticator(client, owner: dict, clock: StepClock) -> bytes:
    """Enroll and enable an authenticator; return its secret.

    The first code spends the current step, so the clock moves on to a fresh one.
    """
    enrolled = await client.post(
        f"{TOTP_PATH}/enroll", headers=owner["headers"], json={"password": ACCOUNT_PASSWORD}
    )
    assert enrolled.status_code == 201
    secret = base64.b32decode(enrolled.json()["secret"])
    code = code_at(secret, clock.step)
    activated = await client.post(
        f"{TOTP_PATH}/activate", headers=owner["headers"], json={"code": code}
    )
    assert activated.status_code == 200
    clock.step += 1
    return secret


async def pending_unlock(app: FastAPI, owner: dict, device: str = FRONT_DOOR) -> UUID:
    """A pending unlock of the owner's door, made by the real tool."""
    factory = app.state.session_factory
    async with factory() as session:
        resident = await load_resident(session, UUID(owner["id"]))
        turn_id = await session.scalar(
            insert(AssistantTurn)
            .values(user_id=resident.user_id, model="test", prompt_version="assistant-v1")
            .returning(AssistantTurn.id)
        )
        await session.commit()
    context = ToolContext(
        session_factory=factory,
        settings=app.state.assistant_settings,
        resident=resident,
        turn_id=turn_id,
        embedder=None,
    )
    call = ToolCall(name="unlock_door", call_id="call_1", arguments=json.dumps({"device": device}))
    outcome = await execute(context, call)
    return outcome.action.pending_action_id


async def demo_residence(client, owner: dict) -> dict:
    response = await client.post("/api/v1/residence/demo", headers=owner["headers"])
    assert response.status_code == 201
    return response.json()


async def add_resident(app: FastAPI, user_id: UUID, building_id: UUID, label: str) -> UUID:
    """A plain resident (not a manager) in a new apartment of an existing building."""
    apartment_id = uuid4()
    async with app.state.session_factory() as session:
        await session.execute(
            insert(Apartment).values(id=apartment_id, building_id=building_id, label=label)
        )
        await session.execute(
            insert(Membership).values(
                user_id=user_id, building_id=building_id, apartment_id=apartment_id, role="resident"
            )
        )
        await session.commit()
    return apartment_id


async def add_ready_document(
    app: FastAPI,
    owner_id: UUID,
    title: str,
    texts: list[str],
    *,
    building_id: UUID | None = None,
) -> UUID:
    """Store ready chunks embedded exactly as the simulator embeds questions."""
    document_id = uuid4()
    joined = "\n".join(texts).encode()
    async with app.state.session_factory() as session:
        await session.execute(
            insert(RAGDocument).values(
                id=document_id,
                owner_id=owner_id,
                building_id=building_id,
                title=title,
                content_type="text/plain",
                size_bytes=len(joined),
                content_sha256=sha256(joined).hexdigest(),
                source=None,
                idempotency_key_hash=uuid4().hex,
                request_fingerprint=uuid4().hex,
                status="ready",
                deadline_at=func.now() + timedelta(hours=1),
                embedding_model=EMBEDDING_MODEL,
                embedding_dimensions=RAG_EMBEDDING_DIMENSIONS,
                chunker_version="recursive-v1",
                chunk_count=len(texts),
                page_count=1,
                finished_at=func.now(),
            )
        )
        await session.execute(
            insert(RAGChunk),
            [
                {
                    "document_id": document_id,
                    "owner_id": owner_id,
                    "ordinal": ordinal,
                    "page": None,
                    "content": text,
                    "embedding": SIMULATOR.embed_text(text, RAG_EMBEDDING_DIMENSIONS),
                }
                for ordinal, text in enumerate(texts)
            ],
        )
        await session.commit()
    return document_id
