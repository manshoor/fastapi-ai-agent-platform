"""Exercise real detail-cache reads, absolute expiry, and dependency failures."""

import asyncio
import socket
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from redis.exceptions import ConnectionError, ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError
from sqlalchemy import Select, event, select
from sqlalchemy.engine import Connection, ExecutionContext

from app.config import Settings
from app.models import Item
from app.schemas import ItemResponse


@pytest.fixture
def settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"cache_ttl_ms": 1000, "redis_max_connections": 2})


@pytest.fixture
def item_selects(app: FastAPI) -> Iterator[list[Select]]:
    statements: list[Select] = []

    def record(
        connection: Connection,
        cursor: object,
        statement: str,
        parameters: object,
        context: ExecutionContext,
        executemany: bool,
    ) -> None:
        compiled = context.compiled
        if compiled is not None and isinstance(compiled.statement, Select):
            query = compiled.statement
            if any(getattr(table, "name", None) == "items" for table in query.get_final_froms()):
                statements.append(query)

    engine = app.state.engine.sync_engine
    event.listen(engine, "before_cursor_execute", record)
    try:
        yield statements
    finally:
        event.remove(engine, "before_cursor_execute", record)


def cache_key(owner: dict, item_id: str) -> str:
    return f"item:v1:{owner['id']}:{item_id}"


async def redis_now_ms(redis: Redis) -> int:
    seconds, microseconds = await redis.time()
    return seconds * 1000 + microseconds // 1000


async def wait_for_redis_time(redis: Redis, deadline_ms: int) -> None:
    async with asyncio.timeout(3):
        while await redis_now_ms(redis) < deadline_ms:
            await asyncio.sleep(0.01)


async def wait_for_expiry(redis: Redis, key: str, deadline_ms: int) -> None:
    async with asyncio.timeout(3):
        while True:
            ttl = await redis.pttl(key)
            if ttl == -2:
                break
            assert ttl >= 0, "Detail cache entries must always have an expiration"
            assert await redis.pexpiretime(key) in (deadline_ms, -2)
            await asyncio.sleep(0.01)
    assert await redis_now_ms(redis) >= deadline_ms


@asynccontextmanager
async def paused_detail_fill(
    app: FastAPI,
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    owner: dict,
    item_id: str,
) -> AsyncIterator[tuple[asyncio.Task[httpx.Response], asyncio.Event, ItemResponse, int]]:
    """Pause only the first real fill, after the endpoint has read and released PostgreSQL."""
    cache = app.state.item_cache
    original_fill = cache.fill
    entered = asyncio.Event()
    release = asyncio.Event()
    captured: tuple[ItemResponse, int] | None = None

    async def pause(owner_id: UUID, response: ItemResponse, expires_at_ms: int) -> None:
        nonlocal captured
        if not entered.is_set():
            captured = response, expires_at_ms
            entered.set()
            await release.wait()
        await original_fill(owner_id, response, expires_at_ms)

    monkeypatch.setattr(cache, "fill", pause)
    request = asyncio.create_task(client.get(f"/api/v1/items/{item_id}", headers=owner["headers"]))
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        assert captured is not None
        yield request, release, captured[0], captured[1]
    finally:
        release.set()
        await asyncio.wait_for(request, timeout=2)


async def assert_persisted_description(app: FastAPI, item_id: str, expected: str | None) -> None:
    async with app.state.engine.connect() as connection:
        description = await connection.scalar(
            select(Item.description).where(Item.id == UUID(item_id))
        )
    assert description == expected


async def test_cache_hit_avoids_item_select_and_authentication_lookup(
    app: FastAPI, client: httpx.AsyncClient, account: Callable, item: Callable, item_selects: list
) -> None:
    owner = await account()
    created = await item(owner)
    key = cache_key(owner, created["id"])
    assert await app.state.redis.get(key) is None
    item_selects.clear()

    first = await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
    assert first.status_code == 200
    assert first.json() == created
    assert len(item_selects) == 1
    assert ItemResponse.model_validate_json(await app.state.redis.get(key)).id == UUID(
        created["id"]
    )
    item_selects.clear()
    selected: list[str] = []

    def record_select(
        connection: Connection,
        cursor: object,
        statement: str,
        parameters: object,
        context: ExecutionContext,
        executemany: bool,
    ) -> None:
        if context.compiled is not None and isinstance(context.compiled.statement, Select):
            selected.append(statement)

    engine = app.state.engine.sync_engine
    event.listen(engine, "before_cursor_execute", record_select)
    try:
        second = await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
    finally:
        event.remove(engine, "before_cursor_execute", record_select)
    assert second.status_code == 200
    assert second.json() == created
    assert item_selects == []
    assert selected == [], "A cache hit must not look up either the item or the JWT's user"


async def test_real_redis_expiration_uses_time_before_database_read(
    app: FastAPI,
    client: httpx.AsyncClient,
    account: Callable,
    item: Callable,
    item_selects: list,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = await account()
    created = await item(owner)
    cache = app.state.item_cache
    cache.ttl_ms = 200
    redis = app.state.redis
    key = cache_key(owner, created["id"])
    original_time = redis.time
    observed_times: list[int] = []
    select_after_time: list[bool] = []

    async def record_time() -> tuple[int, int]:
        seconds, microseconds = await original_time()
        observed_times.append(seconds * 1000 + microseconds // 1000)
        return seconds, microseconds

    def record_order(
        connection: Connection,
        cursor: object,
        statement: str,
        parameters: object,
        context: ExecutionContext,
        executemany: bool,
    ) -> None:
        if context.compiled is not None and isinstance(context.compiled.statement, Select):
            if any(
                getattr(table, "name", None) == "items"
                for table in context.compiled.statement.get_final_froms()
            ):
                select_after_time.append(bool(observed_times))

    monkeypatch.setattr(redis, "time", record_time)
    event.listen(app.state.engine.sync_engine, "before_cursor_execute", record_order)
    try:
        response = await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
    finally:
        event.remove(app.state.engine.sync_engine, "before_cursor_execute", record_order)
    assert response.status_code == 200
    assert select_after_time == [True]
    assert len(observed_times) == 1
    deadline = observed_times[0] + cache.ttl_ms
    assert await redis.pexpiretime(key) == deadline
    assert 0 <= await redis.pttl(key) <= cache.ttl_ms
    await wait_for_expiry(redis, key, deadline)
    item_selects.clear()
    fresh = await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
    assert fresh.status_code == 200
    assert fresh.json() == created
    assert len(item_selects) == 1


async def test_owner_keys_and_uncached_missing_results(
    app: FastAPI, client: httpx.AsyncClient, account: Callable, item: Callable, item_selects: list
) -> None:
    alice = await account()
    bob = await account()
    alice_item = await item(alice, title="Alice's private item")
    bob_item = await item(bob, title="Bob's private item")
    for owner, created in ((alice, alice_item), (bob, bob_item)):
        listed = await client.get("/api/v1/items", headers=owner["headers"])
        assert listed.status_code == 200
        assert [entry["id"] for entry in listed.json()["items"]] == [created["id"]]
        assert await app.state.redis.get(cache_key(owner, created["id"])) is None
        read = await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
        assert read.status_code == 200
        assert read.json() == created
        assert await app.state.redis.get(cache_key(owner, created["id"])) is not None

    for owner, foreign in ((alice, bob_item), (bob, alice_item)):
        response = await client.get(f"/api/v1/items/{foreign['id']}", headers=owner["headers"])
        assert response.status_code == 404
        assert response.json() == {"detail": "Item not found"}
        assert await app.state.redis.get(cache_key(owner, foreign["id"])) is None

    missing_id = str(uuid4())
    item_selects.clear()
    for _ in range(2):
        missing = await client.get(f"/api/v1/items/{missing_id}", headers=alice["headers"])
        assert missing.status_code == 404
        assert missing.json() == {"detail": "Item not found"}
    assert len(item_selects) == 2
    assert await app.state.redis.get(cache_key(alice, missing_id)) is None


@pytest.mark.parametrize("mutation", ["patch", "delete"])
async def test_stale_fill_after_committed_invalidation_keeps_original_deadline(
    mutation: str,
    app: FastAPI,
    client: httpx.AsyncClient,
    account: Callable,
    item: Callable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = await account()
    created = await item(owner)
    cache = app.state.item_cache
    cache.ttl_ms = 1500
    redis = app.state.redis
    key = cache_key(owner, created["id"])
    before = await redis_now_ms(redis)

    async with paused_detail_fill(app, client, monkeypatch, owner, created["id"]) as paused:
        request, release, snapshot, deadline = paused
        assert snapshot.description == "before"
        assert before + cache.ttl_ms <= deadline <= await redis_now_ms(redis) + cache.ttl_ms
        if mutation == "patch":
            changed = await client.patch(
                f"/api/v1/items/{created['id']}",
                headers=owner["headers"],
                json={"description": "after"},
            )
            assert changed.status_code == 200
            assert changed.json()["description"] == "after"
        else:
            changed = await client.delete(
                f"/api/v1/items/{created['id']}", headers=owner["headers"]
            )
            assert changed.status_code == 204
            assert changed.content == b""
        await assert_persisted_description(
            app, created["id"], "after" if mutation == "patch" else None
        )
        assert await redis.get(key) is None
        await wait_for_redis_time(redis, deadline - cache.ttl_ms + 150)
        release.set()
        stale = await request
        assert stale.status_code == 200
        assert stale.json()["description"] == "before"
        assert await redis.pexpiretime(key) == deadline
        assert 0 <= await redis.pttl(key) <= cache.ttl_ms - 150
        assert ItemResponse.model_validate_json(await redis.get(key)).description == "before"
        await wait_for_expiry(redis, key, deadline)
        fresh = await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
        if mutation == "patch":
            assert fresh.status_code == 200
            assert fresh.json()["description"] == "after"
        else:
            assert fresh.status_code == 404
            assert fresh.json() == {"detail": "Item not found"}
            assert await redis.get(key) is None


@pytest.mark.parametrize("mutation", ["patch", "delete"])
async def test_fill_resuming_after_original_deadline_cannot_resurrect_stale_data(
    mutation: str,
    app: FastAPI,
    client: httpx.AsyncClient,
    account: Callable,
    item: Callable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = await account()
    created = await item(owner)
    app.state.item_cache.ttl_ms = 200
    redis = app.state.redis
    key = cache_key(owner, created["id"])
    async with paused_detail_fill(app, client, monkeypatch, owner, created["id"]) as paused:
        request, release, snapshot, deadline = paused
        assert snapshot.description == "before"
        if mutation == "patch":
            changed = await client.patch(
                f"/api/v1/items/{created['id']}",
                headers=owner["headers"],
                json={"description": "after"},
            )
            assert changed.status_code == 200
            assert changed.json()["description"] == "after"
        else:
            changed = await client.delete(
                f"/api/v1/items/{created['id']}", headers=owner["headers"]
            )
            assert changed.status_code == 204
        await wait_for_redis_time(redis, deadline + 1)
        release.set()
        old_response = await request
        assert old_response.status_code == 200
        assert old_response.json()["description"] == "before"
        assert await redis.pttl(key) == -2
        assert await redis.pexpiretime(key) == -2
        fresh = await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
        assert fresh.status_code == (200 if mutation == "patch" else 404)
        assert fresh.json() == (
            changed.json() if mutation == "patch" else {"detail": "Item not found"}
        )


@pytest.mark.parametrize("mutation", ["patch", "delete"])
async def test_failed_invalidation_preserves_commit_but_not_staleness_beyond_deadline(
    mutation: str,
    app: FastAPI,
    client: httpx.AsyncClient,
    account: Callable,
    item: Callable,
    item_selects: list,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = await account()
    created = await item(owner)
    app.state.item_cache.ttl_ms = 1500
    redis = app.state.redis
    key = cache_key(owner, created["id"])
    warmed = await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
    assert warmed.status_code == 200
    deadline = await redis.pexpiretime(key)
    attempts: list[tuple] = []

    async def fail_delete(*keys: str) -> int:
        attempts.append(keys)
        raise ConnectionError("injected Redis invalidation failure")

    with monkeypatch.context() as faults:
        faults.setattr(redis, "delete", fail_delete)
        if mutation == "patch":
            changed = await client.patch(
                f"/api/v1/items/{created['id']}",
                headers=owner["headers"],
                json={"description": "after"},
            )
            assert changed.status_code == 200
            assert changed.json()["description"] == "after"
        else:
            changed = await client.delete(
                f"/api/v1/items/{created['id']}", headers=owner["headers"]
            )
            assert changed.status_code == 204
            assert changed.content == b""
    assert attempts == [(key,)]
    await assert_persisted_description(app, created["id"], "after" if mutation == "patch" else None)
    assert await redis.pexpiretime(key) == deadline
    item_selects.clear()
    stale = await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
    assert stale.status_code == 200
    assert stale.json() == created
    assert item_selects == []
    assert await redis.pexpiretime(key) == deadline
    await wait_for_expiry(redis, key, deadline)
    fresh = await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
    assert fresh.status_code == (200 if mutation == "patch" else 404)
    assert fresh.json() == (changed.json() if mutation == "patch" else {"detail": "Item not found"})
    assert len(item_selects) == 1


@pytest.mark.parametrize("corruption", ["invalid-json", "invalid-schema", "different-item-id"])
async def test_corrupt_values_are_deleted_and_read_from_postgres(
    corruption: str,
    app: FastAPI,
    client: httpx.AsyncClient,
    account: Callable,
    item: Callable,
    item_selects: list,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = await account()
    created = await item(owner)
    redis = app.state.redis
    key = cache_key(owner, created["id"])
    values = {
        "invalid-json": b"not-json",
        "invalid-schema": b'{"id":"not-a-uuid","description":null}',
        "different-item-id": ItemResponse.model_validate(
            {**created, "id": str(uuid4()), "description": "poisoned"}
        ).model_dump_json(),
    }
    await redis.set(key, values[corruption], px=5000)

    async def unavailable_time() -> None:
        raise ConnectionError("injected Redis TIME failure prevents replacement")

    item_selects.clear()
    with monkeypatch.context() as faults:
        faults.setattr(redis, "time", unavailable_time)
        response = await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
    assert response.status_code == 200
    assert response.json() == created
    assert len(item_selects) == 1
    assert await redis.get(key) is None, "The corrupt value must be deleted, not merely ignored"
    recovered = await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
    assert recovered.status_code == 200
    assert ItemResponse.model_validate_json(await redis.get(key)).id == UUID(created["id"])


@pytest.mark.parametrize("operation", ["get", "time", "set"])
@pytest.mark.parametrize("error_type", [ConnectionError, RedisTimeoutError, ResponseError])
async def test_expected_redis_failures_fall_back_to_database_and_recover(
    operation: str,
    error_type: type[Exception],
    app: FastAPI,
    client: httpx.AsyncClient,
    account: Callable,
    item: Callable,
    item_selects: list,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = await account()
    created = await item(owner)
    redis = app.state.redis
    attempts = 0

    async def fail(*args: object, **kwargs: object) -> None:
        nonlocal attempts
        attempts += 1
        raise error_type("injected expected Redis boundary failure")

    item_selects.clear()
    with monkeypatch.context() as faults:
        faults.setattr(redis, operation, fail)
        async with asyncio.timeout(2):
            response = await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
    assert response.status_code == 200
    assert response.json() == created
    assert len(item_selects) == 1
    assert attempts == 1
    if operation in ("time", "set"):
        assert await redis.get(cache_key(owner, created["id"])) is None
    recovered = await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
    assert recovered.status_code == 200
    assert recovered.json() == created
    assert await redis.get(cache_key(owner, created["id"])) is not None
    item_selects.clear()
    hit = await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
    assert hit.status_code == 200
    assert item_selects == []


@pytest.mark.parametrize("operation", ["get", "time", "set"])
async def test_cache_operation_deadline_cancels_stalled_dependency_and_falls_back(
    operation: str,
    app: FastAPI,
    client: httpx.AsyncClient,
    account: Callable,
    item: Callable,
    item_selects: list,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = await account()
    created = await item(owner)
    cancelled = asyncio.Event()
    never = asyncio.Event()

    async def stall(*args: object, **kwargs: object) -> None:
        try:
            await never.wait()
        finally:
            cancelled.set()

    item_selects.clear()
    with monkeypatch.context() as faults:
        faults.setattr(app.state.redis, operation, stall)
        async with asyncio.timeout(1):
            response = await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
    assert response.status_code == 200
    assert response.json() == created
    assert cancelled.is_set()
    assert len(item_selects) == 1
    recovered = await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
    assert recovered.status_code == 200


async def test_real_redis_connection_refusal_falls_back_and_recovers(
    app: FastAPI,
    client: httpx.AsyncClient,
    account: Callable,
    item: Callable,
    item_selects: list,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = await account()
    created = await item(owner)
    # Binding without listening reserves an actual unreachable local TCP endpoint.
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        unavailable = Redis(
            host="127.0.0.1",
            port=reserved.getsockname()[1],
            db=1,
            socket_connect_timeout=0.05,
            socket_timeout=0.05,
            retry=Retry(NoBackoff(), 0),
        )
        try:
            item_selects.clear()
            with monkeypatch.context() as faults:
                faults.setattr(app.state.item_cache, "redis", unavailable)
                async with asyncio.timeout(1):
                    response = await client.get(
                        f"/api/v1/items/{created['id']}", headers=owner["headers"]
                    )
            assert response.status_code == 200
            assert response.json() == created
            assert len(item_selects) == 1
        finally:
            await unavailable.aclose()
    recovered = await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
    assert recovered.status_code == 200
    assert await app.state.redis.get(cache_key(owner, created["id"])) is not None


async def test_real_redis_pool_exhaustion_falls_back_without_extra_connections(
    app: FastAPI, client: httpx.AsyncClient, account: Callable, item: Callable, item_selects: list
) -> None:
    owner = await account()
    created = await item(owner)
    pool = app.state.redis.connection_pool
    held = []
    try:
        for _ in range(app.state.settings.redis_max_connections):
            held.append(await pool.get_connection())
        with pytest.raises(ConnectionError):
            await pool.get_connection()
        item_selects.clear()
        async with asyncio.timeout(1):
            response = await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
        assert response.status_code == 200
        assert response.json() == created
        assert len(item_selects) == 1
        changed = await client.patch(
            f"/api/v1/items/{created['id']}",
            headers=owner["headers"],
            json={"description": "committed while Redis pool is full"},
        )
        assert changed.status_code == 200
        await assert_persisted_description(app, created["id"], "committed while Redis pool is full")
    finally:
        for connection in held:
            await pool.release(connection)
    recovered = await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
    assert recovered.status_code == 200
    assert recovered.json()["description"] == "committed while Redis pool is full"
    item_selects.clear()
    hit = await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
    assert hit.status_code == 200
    assert item_selects == []


@pytest.mark.parametrize("mutation", ["patch", "delete"])
async def test_real_deferred_commit_failure_does_not_invalidate_committed_cache(
    mutation: str,
    app: FastAPI,
    client: httpx.AsyncClient,
    account: Callable,
    item: Callable,
    item_selects: list,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = await account()
    created = await item(owner)
    app.state.item_cache.ttl_ms = 5000
    redis = app.state.redis
    key = cache_key(owner, created["id"])
    warmed = await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
    assert warmed.status_code == 200
    value = await redis.get(key)
    deadline = await redis.pexpiretime(key)
    deleted_keys: list[tuple] = []
    original_delete = redis.delete

    async def record_delete(*keys: str) -> int:
        deleted_keys.append(keys)
        return await original_delete(*keys)

    # This deferred PostgreSQL trigger fails COMMIT, not the preceding UPDATE/DELETE.
    async with app.state.engine.begin() as connection:
        await connection.exec_driver_sql(
            "CREATE FUNCTION test_cache_reject_commit() RETURNS trigger LANGUAGE plpgsql AS $$ "
            "BEGIN RAISE EXCEPTION 'injected deferred commit failure' USING ERRCODE = '23514'; "
            "END $$"
        )
        await connection.exec_driver_sql(
            "CREATE CONSTRAINT TRIGGER test_cache_reject_commit "
            "AFTER UPDATE OR DELETE ON items DEFERRABLE INITIALLY DEFERRED "
            "FOR EACH ROW EXECUTE FUNCTION test_cache_reject_commit()"
        )
    try:
        monkeypatch.setattr(redis, "delete", record_delete)
        if mutation == "patch":
            response = await client.patch(
                f"/api/v1/items/{created['id']}",
                headers=owner["headers"],
                json={"description": "must roll back"},
            )
        else:
            response = await client.delete(
                f"/api/v1/items/{created['id']}", headers=owner["headers"]
            )
        assert response.status_code == 500
        assert response.json() == {"detail": "Internal Server Error"}
        await assert_persisted_description(app, created["id"], "before")
        assert deleted_keys == []
        assert await redis.get(key) == value
        assert await redis.pexpiretime(key) == deadline
        item_selects.clear()
        read = await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
        assert read.status_code == 200
        assert read.json() == created
        assert item_selects == []
    finally:
        async with app.state.engine.begin() as connection:
            await connection.exec_driver_sql("DROP TRIGGER test_cache_reject_commit ON items")
            await connection.exec_driver_sql("DROP FUNCTION test_cache_reject_commit()")
