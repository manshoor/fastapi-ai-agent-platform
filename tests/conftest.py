"""Integration fixtures clean only explicitly dedicated test destinations."""

from collections.abc import AsyncIterator, Awaitable, Callable
from uuid import uuid4

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from redis.asyncio.connection import parse_url
from sqlalchemy import text

from app.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
async def app(settings: Settings) -> AsyncIterator[FastAPI]:
    url = settings.sqlalchemy_url
    _, connect_args = url.get_dialect()().create_connect_args(url)
    database = connect_args.get("database")
    if not isinstance(database, str) or database != "app_test":
        raise RuntimeError("Refusing integration cleanup outside database app_test")
    if parse_url(settings.redis_url).get("db", 0) != 1:
        raise RuntimeError("Refusing integration cleanup outside Redis database 1")
    from app.main import create_app

    application = create_app(settings)
    async with LifespanManager(application):
        async with application.state.engine.begin() as connection:
            await connection.execute(
                text(
                    "TRUNCATE TABLE assistant_tool_calls, pending_actions, assistant_turns, "
                    "tickets, bookings, devices, memberships, apartments, buildings, "
                    "rag_connector_runs, rag_queries, rag_outbox, rag_chunks, rag_documents, "
                    "ai_outbox, ai_jobs, items, users CASCADE"
                )
            )
        await application.state.redis.flushdb()
        yield application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as http:
        yield http


@pytest.fixture
def account(client: httpx.AsyncClient) -> Callable[..., Awaitable[dict]]:
    async def create(*, email: str | None = None, password: str = "a-secure-password-123") -> dict:
        address = email or f"test-{uuid4().hex}@example.com"
        registered = await client.post(
            "/api/v1/auth/register", json={"email": address, "password": password}
        )
        assert registered.status_code == 201
        user = registered.json()
        login = await client.post(
            "/api/v1/auth/login", data={"username": address, "password": password}
        )
        assert login.status_code == 200
        token = login.json()["access_token"]
        return {**user, "access_token": token, "headers": {"Authorization": f"Bearer {token}"}}

    return create


@pytest.fixture
def item(client: httpx.AsyncClient) -> Callable[..., Awaitable[dict]]:
    async def create(owner: dict, *, title: str = "Test item", description: str = "before") -> dict:
        response = await client.post(
            "/api/v1/items",
            headers=owner["headers"],
            json={"title": title, "description": description},
        )
        assert response.status_code == 201
        return response.json()

    return create
