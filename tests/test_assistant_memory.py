"""Assistant memory: sealing, scopes, limits, retention and forgetting on real PostgreSQL."""

import asyncio
import base64
from uuid import UUID, uuid4

import httpx
import pytest
from cryptography.exceptions import InvalidTag
from pydantic import ValidationError
from sqlalchemy import delete, select, text, update
from sqlalchemy.exc import OperationalError

from app.assistant.config import AssistantSettings
from app.assistant.memory import MemoryBox, purge_memories, purge_tick
from app.config import Settings
from app.models import AssistantMemory, Membership
from tests.support.assistant import SIMULATOR, TOTP_KEY, demo_residence, serve

URL = "/api/v1/assistant/memories"
# Test-only literal, as in compose.ai.test.yaml: base64 of "test-only-memory-key-not-secret!".
MEMORY_KEY = "dGVzdC1vbmx5LW1lbW9yeS1rZXktbm90LXNlY3JldCE="
LIMIT = 3
WRITES_PER_DAY = 5


@pytest.fixture
async def simulator():
    async with serve(SIMULATOR.create_app()) as base_url:
        yield base_url


@pytest.fixture
def settings(monkeypatch, simulator) -> Settings:
    for name, value in {
        "OPENAI_BASE_URL": simulator + "/v1",
        "OPENAI_API_KEY": "synthetic-key-never-external",
        "ASSISTANT_TOTP_KEY": TOTP_KEY,
        "ASSISTANT_MEMORY_ENABLED": "true",
        "ASSISTANT_MEMORY_KEY": MEMORY_KEY,
        "ASSISTANT_MEMORY_LIMIT": str(LIMIT),
        "ASSISTANT_MEMORY_WRITES_PER_DAY": str(WRITES_PER_DAY),
    }.items():
        monkeypatch.setenv(name, value)
    return Settings(
        ai_enabled=True, rag_enabled=True, assistant_enabled=True, assistant_demo_enabled=True
    )


async def remember(client, owner, text="I prefer the gym after 19:00", scope="resident"):
    return await client.post(URL, headers=owner["headers"], json={"text": text, "scope": scope})


async def listed(client, owner) -> list[dict]:
    response = await client.get(URL, headers=owner["headers"])
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    return response.json()["memories"]


# Settings and sealing


def _settings(**values) -> AssistantSettings:
    base = {
        "ai_enabled": True,
        "rag_enabled": True,
        "assistant_enabled": True,
        "assistant_totp_key": TOTP_KEY,
        "assistant_memory_enabled": True,
        "assistant_memory_key": MEMORY_KEY,
    }
    return AssistantSettings.model_validate({**base, **values})


@pytest.mark.parametrize(
    "values",
    [
        {"assistant_memory_key": None},
        {"assistant_memory_key": base64.b64encode(b"short").decode()},
        # The TOTP key must not be reused for another purpose.
        {"assistant_memory_key": TOTP_KEY},
        # The same 32 bytes in another spelling: the last character's two spare bits.
        {"assistant_memory_key": TOTP_KEY[:-2] + "F="},
        {"assistant_memory_retention_seconds": 3600},
        {"assistant_memory_retention_seconds": 366 * 86400},
    ],
)
def test_memory_settings_are_validated(values):
    with pytest.raises(ValidationError):
        _settings(**values)


def test_memory_needs_the_assistant_at_startup():
    with pytest.raises(ValidationError, match="ASSISTANT_MEMORY_ENABLED"):
        Settings(assistant_memory_enabled=True)


def test_memory_needs_the_assistant():
    with pytest.raises(ValidationError, match="ASSISTANT_MEMORY_ENABLED"):
        AssistantSettings.model_validate(
            {"assistant_memory_enabled": True, "assistant_memory_key": MEMORY_KEY}
        )


def test_sealed_text_is_bound_to_its_user_and_row():
    box = MemoryBox.from_settings(_settings())
    user, row = uuid4(), uuid4()
    nonce, ciphertext = box.seal(user, row, "Water the plants on Fridays")
    assert box.open(user, row, nonce, ciphertext) == "Water the plants on Fridays"
    for other_user, other_row in ((uuid4(), row), (user, uuid4())):
        with pytest.raises(InvalidTag):
            box.open(other_user, other_row, nonce, ciphertext)


# Routes


async def test_routes_are_absent_while_memory_is_off(client, monkeypatch, app):
    monkeypatch.setattr(
        app.state,
        "assistant_settings",
        app.state.assistant_settings.model_copy(update={"assistant_memory_enabled": False}),
    )
    # 404 before authentication: no token is needed to see the route is off.
    assert (await client.get(URL)).status_code == 404
    body = {"text": "a valid note", "scope": "resident"}
    assert (await client.post(URL, json=body)).status_code == 404
    assert (await client.delete(URL)).status_code == 404


def test_memory_routes_are_not_in_the_schema(app):
    assert not any("memories" in path for path in app.openapi()["paths"])


async def test_remember_list_and_forget_one(app, client: httpx.AsyncClient, account):
    owner, stranger = await account(), await account()
    created = await remember(client, owner)
    assert created.status_code == 201 and created.headers["cache-control"] == "no-store"
    memory = created.json()
    assert (memory["scope"], memory["text"]) == ("resident", "I prefer the gym after 19:00")
    assert [m["id"] for m in await listed(client, owner)] == [memory["id"]]
    assert await listed(client, stranger) == []

    # The text is stored sealed: no column holds it.
    async with app.state.session_factory() as session:
        row = await session.get(AssistantMemory, UUID(memory["id"]))
        assert b"gym" not in row.ciphertext and row.key_version == 1

    foreign = await client.delete(f"{URL}/{memory['id']}", headers=stranger["headers"])
    missing = await client.delete(f"{URL}/{uuid4()}", headers=owner["headers"])
    assert foreign.status_code == missing.status_code == 404
    gone = await client.delete(f"{URL}/{memory['id']}", headers=owner["headers"])
    assert gone.status_code == 204 and await listed(client, owner) == []


async def test_forget_everything(client, account):
    owner, other = await account(), await account()
    await remember(client, owner)
    await remember(client, owner, "My parcel locker code is on the fridge")
    await remember(client, other)
    assert (await client.delete(URL, headers=owner["headers"])).status_code == 204
    assert await listed(client, owner) == []
    assert len(await listed(client, other)) == 1


@pytest.mark.parametrize(
    "body",
    [
        {"text": "ab", "scope": "resident"},
        {"text": "x" * 301, "scope": "resident"},
        {"text": "nul \x00 inside", "scope": "resident"},
        {"text": "a valid note", "scope": "building"},
        {"text": "a valid note", "scope": "resident", "user_id": str(uuid4())},
    ],
    ids=["short", "long", "nul", "scope", "extra"],
)
async def test_invalid_memories_are_refused(client, account, body):
    owner = await account()
    response = await client.post(URL, headers=owner["headers"], json=body)
    assert response.status_code == 422
    assert "nul" not in response.text and "x" * 50 not in response.text


async def test_residence_memory_needs_a_residence_and_ends_with_it(app, client, account):
    owner = await account()
    without = await remember(client, owner, "The bike room is my favorite", scope="residence")
    assert without.status_code == 403

    view = await demo_residence(client, owner)
    created = await remember(client, owner, "The bike room is my favorite", scope="residence")
    assert created.status_code == 201
    assert [m["scope"] for m in await listed(client, owner)] == ["residence"]

    # Moving out hides the memory at once; the purge then removes it.
    async with app.state.session_factory() as session:
        await session.execute(
            delete(Membership).where(
                Membership.user_id == UUID(owner["id"]), Membership.role == "resident"
            )
        )
        await session.commit()
    assert await listed(client, owner) == []
    assert await purge_memories(session_factory=app.state.session_factory) == 1
    async with app.state.session_factory() as session:
        assert await session.scalar(select(AssistantMemory.id)) is None
    assert view["apartment"]["label"] == "1A"


async def test_moved_out_memories_do_not_count_against_the_limit(app, client, account):
    owner = await account()
    await demo_residence(client, owner)
    for index in range(LIMIT):
        created = await remember(client, owner, f"Home note {index} here", scope="residence")
        assert created.status_code == 201
    async with app.state.session_factory() as session:
        await session.execute(
            delete(Membership).where(
                Membership.user_id == UUID(owner["id"]), Membership.role == "resident"
            )
        )
        await session.commit()
    # Before any purge: the hidden rows leave room for a new memory.
    assert (await remember(client, owner)).status_code == 201


async def test_expired_memories_are_never_listed_and_are_purged(app, client, account):
    owner = await account()
    kept = (await remember(client, owner, "Keep this one around")).json()
    old = (await remember(client, owner, "This one is past its time")).json()
    async with app.state.session_factory() as session:
        await session.execute(
            update(AssistantMemory)
            .where(AssistantMemory.id == UUID(old["id"]))
            .values(
                created_at=text("clock_timestamp() - interval '2 seconds'"),
                expires_at=text("clock_timestamp() - interval '1 second'"),
            )
        )
        await session.commit()
    assert [m["id"] for m in await listed(client, owner)] == [kept["id"]]
    assert await purge_memories(session_factory=app.state.session_factory) == 1
    assert [m["id"] for m in await listed(client, owner)] == [kept["id"]]


async def test_retention_sets_the_expiry_on_the_database_clock(app, client, account):
    owner = await account()
    memory = (await remember(client, owner)).json()
    async with app.state.session_factory() as session:
        seconds = await session.scalar(
            select(text("extract(epoch FROM expires_at - created_at)"))
            .select_from(AssistantMemory)
            .where(AssistantMemory.id == UUID(memory["id"]))
        )
    assert float(seconds) == pytest.approx(90 * 86400, abs=1)


async def test_active_limit_holds_under_concurrency(client, account):
    owner = await account()
    responses = await asyncio.gather(
        *(remember(client, owner, f"Note number {index} for later") for index in range(LIMIT + 3))
    )
    codes = sorted(response.status_code for response in responses)
    assert codes == [201] * LIMIT + [429] * 3
    assert len(await listed(client, owner)) == LIMIT


async def test_daily_writes_are_limited(client, account):
    owner = await account()
    for index in range(WRITES_PER_DAY):
        created = (await remember(client, owner, f"Note number {index} for later")).json()
        # Forgetting frees an active slot, not a write.
        forgotten = await client.delete(f"{URL}/{created['id']}", headers=owner["headers"])
        assert forgotten.status_code == 204
    refused = await remember(client, owner, "One write too many today")
    assert refused.status_code == 429 and "retry-after" in refused.headers


async def test_a_failed_purge_does_not_stop_the_dispatcher_tick(caplog):
    def failing_factory():
        # A statement timeout or lost connection, with a detail that must not be logged.
        raise OperationalError("DELETE ... secret-detail", {}, Exception("timeout"))

    assert await purge_tick(failing_factory) == 0
    assert "secret-detail" not in caplog.text
    assert "OperationalError" in caplog.text
