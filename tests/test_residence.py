"""Residences, demo setup, building document uploads, and building-scoped search."""

import hashlib
import json
from uuid import UUID, uuid4

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI

from app.ai.provider import OpenAIEmbedder
from app.config import Settings
from app.rag.contracts import ContentType
from app.rag.documents import _fingerprint
from app.rag.retrieval import BuildingScopedRetriever, has_building_context
from app.rag.uploads import Upload
from tests.support.assistant import (
    EMBEDDING_MODEL,
    SIMULATOR,
    add_ready_document,
    add_resident,
    demo_residence,
    serve,
)

UPLOAD = "/api/v1/rag/documents"
RESIDENCE = "/api/v1/residence"
DEMO = "/api/v1/residence/demo"


@pytest.fixture
async def simulator():
    application = SIMULATOR.create_app()
    async with serve(application) as base_url:
        yield base_url


@pytest.fixture
def settings(monkeypatch, simulator) -> Settings:
    monkeypatch.setenv("OPENAI_BASE_URL", simulator + "/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key-never-external")
    return Settings(
        ai_enabled=True, rag_enabled=True, assistant_enabled=True, assistant_demo_enabled=True
    )


async def upload(client, owner: dict, building_id, *, key: str | None = None):
    params = {"building_id": str(building_id)} if building_id is not None else None
    return await client.post(
        UPLOAD,
        params=params,
        headers={**owner["headers"], "Idempotency-Key": key or uuid4().hex},
        data={"title": "House rules"},
        files={"file": ("rules.md", b"# Rules\n\nQuiet hours start at 22:00.")},
    )


async def test_demo_residence_is_created_once(client: httpx.AsyncClient, account):
    owner = await account()
    view = await demo_residence(client, owner)
    assert view["manager"] is True
    assert view["apartment"]["label"] == "1A"
    kinds = sorted(device["kind"] for device in view["devices"])
    assert kinds == ["alarm", "door_lock", "door_lock", "thermostat", "thermostat"]
    assert [booking["amenity"] for booking in view["bookings"]] == ["Gym", "Party room"]
    assert view["tickets"] == [] and view["pending_actions"] == []

    again = await client.post(DEMO, headers=owner["headers"])
    assert again.status_code == 409
    read = await client.get(RESIDENCE, headers=owner["headers"])
    assert read.status_code == 200 and read.json() == view
    assert read.headers["cache-control"] == "no-store"


async def test_residence_shows_only_the_callers_data(client: httpx.AsyncClient, account):
    first, second = await account(), await account()
    mine = await demo_residence(client, first)
    theirs = await demo_residence(client, second)
    assert mine["building"]["id"] != theirs["building"]["id"]
    mine_devices = {device["id"] for device in mine["devices"]}
    assert mine_devices.isdisjoint(device["id"] for device in theirs["devices"])


async def test_account_without_residence_gets_403(client: httpx.AsyncClient, account):
    owner = await account()
    response = await client.get(RESIDENCE, headers=owner["headers"])
    assert response.status_code == 403


async def test_unauthenticated_residence_is_401(client: httpx.AsyncClient):
    assert (await client.get(RESIDENCE)).status_code == 401
    assert (await client.post(DEMO)).status_code == 401


@pytest.mark.parametrize(
    ("flags", "demo_status", "residence_status"),
    [
        ({"assistant_enabled": True, "assistant_demo_enabled": False}, 404, 403),
        ({"assistant_enabled": False, "assistant_demo_enabled": False}, 404, 503),
    ],
    ids=["demo-off", "assistant-off"],
)
async def test_disabled_flags(monkeypatch, simulator, flags, demo_status, residence_status):
    monkeypatch.setenv("OPENAI_BASE_URL", simulator + "/v1")
    from app.main import create_app

    settings = Settings(ai_enabled=True, rag_enabled=True, **flags)
    application = create_app(settings)
    async with LifespanManager(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            address = f"test-{uuid4().hex}@example.com"
            registered = await client.post(
                "/api/v1/auth/register",
                json={"email": address, "password": "a-secure-password-123"},
            )
            assert registered.status_code == 201
            login = await client.post(
                "/api/v1/auth/login",
                data={"username": address, "password": "a-secure-password-123"},
            )
            headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
            assert (await client.post(DEMO, headers=headers)).status_code == demo_status
            assert (await client.get(RESIDENCE, headers=headers)).status_code == residence_status


async def test_only_a_manager_uploads_building_documents(
    app: FastAPI, client: httpx.AsyncClient, account
):
    manager, neighbour, stranger = await account(), await account(), await account()
    building_id = UUID((await demo_residence(client, manager))["building"]["id"])
    await add_resident(app, UUID(neighbour["id"]), building_id, "2B")
    await demo_residence(client, stranger)

    accepted = await upload(client, manager, building_id)
    assert accepted.status_code == 202
    assert accepted.json()["building_id"] == str(building_id)

    # A resident who is not a manager, a manager of another building, and a missing
    # building all get the same 404.
    for caller, target in ((neighbour, building_id), (stranger, building_id), (manager, uuid4())):
        assert (await upload(client, caller, target)).status_code == 404

    # An owner-only upload still works and has no building.
    private = await upload(client, neighbour, None)
    assert private.status_code == 202 and private.json()["building_id"] is None


async def test_same_key_for_another_building_conflicts(client: httpx.AsyncClient, account):
    manager = await account()
    building_id = UUID((await demo_residence(client, manager))["building"]["id"])
    key = uuid4().hex
    assert (await upload(client, manager, None, key=key)).status_code == 202
    assert (await upload(client, manager, building_id, key=key)).status_code == 409


def test_owner_only_fingerprint_is_unchanged():
    upload = Upload(title="Notes", content_type=ContentType.MARKDOWN, data=b"# Notes")
    original = hashlib.sha256(
        json.dumps(
            [
                "rag-upload-v1",
                "Notes",
                "text/markdown",
                hashlib.sha256(b"# Notes").hexdigest(),
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert _fingerprint(upload, None) == original
    assert _fingerprint(upload, uuid4()) != original


async def test_building_search_never_returns_other_chunks(
    app: FastAPI, client: httpx.AsyncClient, account
):
    manager, other = await account(), await account()
    building_id = UUID((await demo_residence(client, manager))["building"]["id"])
    other_building = UUID((await demo_residence(client, other))["building"]["id"])
    manager_id, other_id = UUID(manager["id"]), UUID(other["id"])
    text = "Quiet hours start at 22:00 in the building."
    mine = await add_ready_document(app, manager_id, "Rules", [text], building_id=building_id)
    # Same words elsewhere: the manager's private note and another building's rules.
    await add_ready_document(app, manager_id, "Private", [text])
    await add_ready_document(app, other_id, "Other rules", [text], building_id=other_building)

    factory = app.state.session_factory
    assert await has_building_context(
        session_factory=factory, building_id=building_id, embedding_model=EMBEDDING_MODEL
    )
    assert not await has_building_context(
        session_factory=factory, building_id=uuid4(), embedding_model=EMBEDDING_MODEL
    )
    embedder = OpenAIEmbedder(app.state.rag.settings)
    try:
        retriever = BuildingScopedRetriever(
            session_factory=factory,
            embedder=embedder,
            building_id=building_id,
            embedding_model=EMBEDDING_MODEL,
            top_k=8,
        )
        documents = await retriever.ainvoke("When do quiet hours start?")
    finally:
        await embedder.aclose()
    assert [document.metadata["document_id"] for document in documents] == [mine]
