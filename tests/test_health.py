"""Readiness degrades on cache loss and bounds real PostgreSQL resource pressure."""

import asyncio
import errno
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from time import monotonic

import httpx
import pytest
from fastapi import FastAPI
from redis.exceptions import ConnectionError, ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from app.config import Settings


@pytest.fixture
def settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "db_pool_size": 2,
            "db_pool_timeout_seconds": 0.5,
            "db_statement_timeout_ms": 200,
        }
    )


async def test_healthy_readiness_checks_real_dependencies(client: httpx.AsyncClient) -> None:
    response = await client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok", "cache": "ok"}


async def test_liveness_does_not_touch_database_or_redis(
    app: FastAPI, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_connect = AsyncEngine.connect

    def forbidden_database_connection(engine: AsyncEngine) -> AsyncConnection:
        if engine is app.state.engine:
            raise AssertionError("Liveness must not acquire a database connection")
        return original_connect(engine)

    async def forbidden_cache_ping() -> None:
        raise AssertionError("Liveness must not contact Redis")

    with monkeypatch.context() as faults:
        faults.setattr(AsyncEngine, "connect", forbidden_database_connection)
        faults.setattr(app.state.redis, "ping", forbidden_cache_ping)
        response = await client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(
    ("database_available", "cache_available"), [(True, False), (False, True), (False, False)]
)
async def test_readiness_dependency_failures_and_recovery(
    database_available: bool,
    cache_available: bool,
    app: FastAPI,
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_connect = AsyncEngine.connect

    def refused_database_connection(engine: AsyncEngine) -> AsyncConnection:
        if engine is app.state.engine:
            raise OSError(errno.ECONNREFUSED, "injected database connection refusal")
        return original_connect(engine)

    async def refused_cache_ping() -> None:
        raise ConnectionError("injected Redis connection refusal")

    with monkeypatch.context() as faults:
        if not database_available:
            faults.setattr(AsyncEngine, "connect", refused_database_connection)
        if not cache_available:
            faults.setattr(app.state.redis, "ping", refused_cache_ping)
        async with asyncio.timeout(1.5):
            response = await client.get("/health/ready")
    assert response.status_code == (200 if database_available else 503)
    assert response.json() == {
        "status": "degraded" if database_available else "unavailable",
        "database": "ok" if database_available else "unavailable",
        "cache": "ok" if cache_available else "unavailable",
    }
    recovered = await client.get("/health/ready")
    assert recovered.status_code == 200
    assert recovered.json() == {"status": "ok", "database": "ok", "cache": "ok"}


@pytest.mark.parametrize("error_type", [RedisTimeoutError, ResponseError])
async def test_redis_timeout_or_server_error_is_degraded_not_unready(
    error_type: type[Exception],
    app: FastAPI,
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failed_ping() -> None:
        raise error_type("injected Redis PING failure")

    monkeypatch.setattr(app.state.redis, "ping", failed_ping)
    response = await client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "database": "ok", "cache": "unavailable"}


@pytest.mark.parametrize(
    ("stall_database", "stall_cache"), [(True, False), (False, True), (True, True)]
)
async def test_readiness_deadline_cancels_unfinished_dependency_work(
    stall_database: bool,
    stall_cache: bool,
    app: FastAPI,
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    never = asyncio.Event()
    database_cancelled = asyncio.Event()
    cache_cancelled = asyncio.Event()
    original_connect = app.state.engine.connect
    original_class_connect = AsyncEngine.connect

    @asynccontextmanager
    async def blocked_database_connection(engine: AsyncEngine) -> AsyncIterator[AsyncConnection]:
        if engine is not app.state.engine:
            async with original_class_connect(engine) as connection:
                yield connection
            return
        try:
            await never.wait()
            async with original_connect() as connection:
                yield connection
        finally:
            database_cancelled.set()

    async def blocked_cache_ping() -> bool:
        try:
            await never.wait()
            return True
        finally:
            cache_cancelled.set()

    with monkeypatch.context() as faults:
        if stall_database:
            faults.setattr(AsyncEngine, "connect", blocked_database_connection)
        if stall_cache:
            faults.setattr(app.state.redis, "ping", blocked_cache_ping)
        started = monotonic()
        async with asyncio.timeout(1.5):
            response = await client.get("/health/ready")
        assert monotonic() - started < 1.5
        assert database_cancelled.is_set() == stall_database
        assert cache_cancelled.is_set() == stall_cache
    assert response.status_code == (503 if stall_database else 200)
    assert response.json() == {
        "status": "unavailable" if stall_database else "degraded",
        "database": "unavailable" if stall_database else "ok",
        "cache": "unavailable" if stall_cache else "ok",
    }
    recovered = await client.get("/health/ready")
    assert recovered.status_code == 200
    assert recovered.json() == {"status": "ok", "database": "ok", "cache": "ok"}


async def test_real_database_pool_exhaustion_returns_503_then_recovers(
    app: FastAPI, client: httpx.AsyncClient, account: Callable, item: Callable
) -> None:
    owner = await account()
    created = await item(owner)
    warmed = await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
    assert warmed.status_code == 200
    held: list[AsyncConnection] = []
    try:
        for _ in range(app.state.settings.db_pool_size):
            held.append(await app.state.engine.connect())
        async with asyncio.timeout(1.5):
            ready = await client.get("/health/ready")
        assert ready.status_code == 503
        assert ready.json() == {
            "status": "unavailable",
            "database": "unavailable",
            "cache": "ok",
        }
        live = await client.get("/health/live")
        assert live.status_code == 200
        assert live.json() == {"status": "ok"}
        async with asyncio.timeout(1.5):
            blocked = await client.get("/api/v1/items", headers=owner["headers"])
        assert blocked.status_code == 503
        assert blocked.json() == {"detail": "Service temporarily unavailable"}
        assert blocked.headers["Retry-After"] == "1"
        cached = await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
        assert cached.status_code == 200
        assert cached.json() == created

        # Releasing one real checkout must restore service while another remains held.
        await held.pop().close()
        recovered = await client.get("/api/v1/items", headers=owner["headers"])
        assert recovered.status_code == 200
        assert [entry["id"] for entry in recovered.json()["items"]] == [created["id"]]
        written = await client.post(
            "/api/v1/items", headers=owner["headers"], json={"title": "After pool recovery"}
        )
        assert written.status_code == 201
        read = await client.get(f"/api/v1/items/{written.json()['id']}", headers=owner["headers"])
        assert read.status_code == 200
        assert read.json() == written.json()
    finally:
        for connection in held:
            await connection.close()
    ready = await client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json() == {"status": "ok", "database": "ok", "cache": "ok"}


async def test_real_postgres_statement_timeout_is_503_and_next_read_recovers(
    app: FastAPI, client: httpx.AsyncClient, account: Callable, item: Callable
) -> None:
    owner = await account()
    created = await item(owner)
    async with app.state.engine.connect() as locker:
        transaction = await locker.begin()
        try:
            await locker.exec_driver_sql("LOCK TABLE items IN ACCESS EXCLUSIVE MODE")
            async with asyncio.timeout(1.5):
                blocked = await client.get(
                    f"/api/v1/items/{created['id']}", headers=owner["headers"]
                )
            assert blocked.status_code == 503
            assert blocked.json() == {"detail": "Service temporarily unavailable"}
            assert blocked.headers["Retry-After"] == "1"
            live = await client.get("/health/live")
            assert live.status_code == 200
            ready = await client.get("/health/ready")
            assert ready.status_code == 200
            assert ready.json() == {"status": "ok", "database": "ok", "cache": "ok"}
        finally:
            await transaction.rollback()
    recovered = await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
    assert recovered.status_code == 200
    assert recovered.json() == created
