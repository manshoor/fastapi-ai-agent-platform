import asyncio
from datetime import timedelta
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import func, select, update

from app.ai.config import AISettings
from app.config import Settings
from app.models import AIJob, AIOutbox, Item


@pytest.fixture
def settings() -> Settings:
    return Settings(ai_enabled=True)


def submission(owner: dict, key: str | None = None) -> dict:
    return {**owner["headers"], "Idempotency-Key": key or uuid4().hex}


async def submit(client: httpx.AsyncClient, owner: dict, created: dict, **kwargs):
    return await client.post(
        f"/api/v1/items/{created['id']}/ai/rewrite",
        headers=kwargs.pop("headers", submission(owner)),
        json=kwargs.pop("json", {"instruction": "Make it clearer"}),
        **kwargs,
    )


async def test_submission_snapshots_database_not_stale_cache_and_never_mutates(
    app: FastAPI, client: httpx.AsyncClient, account, item
):
    owner = await account()
    created = await item(owner, description="保留原意")
    await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
    async with app.state.session_factory() as session:
        await session.execute(
            update(Item).where(Item.id == UUID(created["id"])).values(title="Database snapshot")
        )
        await session.commit()
    response = await submit(client, owner, created)
    assert response.status_code == 202
    result = response.json()
    assert response.headers["location"] == result["status_url"]
    assert response.headers["cache-control"] == "no-store"
    assert result["status"] == "pending" and result["suggestion"] is None
    async with app.state.session_factory() as session:
        job = await session.get(AIJob, UUID(result["id"]))
        assert (job.source_title, job.source_description, job.instruction) == (
            "Database snapshot",
            "保留原意",
            "Make it clearer",
        )
        persisted = await session.get(Item, UUID(created["id"]))
        assert persisted.title == "Database snapshot"
        event = await session.scalar(select(AIOutbox).where(AIOutbox.job_id == job.id))
        assert event.event_kind == "requested" and event.attempt == 1
        assert event.published_at is None
    assert "instruction" not in result and "source_title" not in result
    polled = await client.get(result["status_url"], headers=owner["headers"])
    assert polled.status_code == 200 and polled.json() == result


async def test_job_and_submission_ownership_and_authentication(client, account, item):
    owner, foreign = await account(), await account()
    created = await item(owner)
    response = await submit(client, owner, created)
    url = response.json()["status_url"]
    for target in (created, {"id": str(uuid4())}):
        denied = await submit(client, foreign, target)
        assert denied.status_code == 404 and denied.json() == {"detail": "Item not found"}
    for target in (url, f"/api/v1/ai/jobs/{uuid4()}"):
        denied = await client.get(target, headers=foreign["headers"])
        assert denied.status_code == 404 and denied.json() == {"detail": "Job not found"}
    assert (await client.get(url)).status_code == 401
    assert (
        await client.post(
            f"/api/v1/items/{created['id']}/ai/rewrite",
            headers={"Idempotency-Key": uuid4().hex},
            json={"instruction": "Shorten"},
        )
    ).status_code == 401


async def test_concurrent_idempotency_conflicts_and_replay_after_item_deletion(
    app, client, account, item
):
    owner = await account()
    created = await item(owner)
    headers = submission(owner)
    responses = await asyncio.gather(
        *(submit(client, owner, created, headers=headers) for _ in range(6))
    )
    assert [r.status_code for r in responses] == [202] * 6
    ids = {r.json()["id"] for r in responses}
    assert len(ids) == 1
    async with app.state.session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(AIJob)) == 1
        assert await session.scalar(select(func.count()).select_from(AIOutbox)) == 1
    conflict = await submit(
        client, owner, created, headers=headers, json={"instruction": "Different instruction"}
    )
    assert conflict.status_code == 409
    await client.delete(f"/api/v1/items/{created['id']}", headers=owner["headers"])
    replay = await submit(client, owner, created, headers=headers)
    assert replay.status_code == 202 and replay.json()["id"] in ids
    foreign = await account()
    other_item = await item(foreign)
    independent = await submit(
        client, foreign, other_item, headers=submission(foreign, headers["Idempotency-Key"])
    )
    assert independent.status_code == 202 and independent.json()["id"] not in ids


async def test_concurrent_global_and_owner_admission_is_bounded(app, client, account, item):
    owner = await account()
    created = await item(owner)
    app.state.ai_settings = app.state.ai_settings.model_copy(
        update={"ai_owner_limit": 2, "ai_global_limit": 1000}
    )
    responses = await asyncio.gather(*(submit(client, owner, created) for _ in range(5)))
    assert sorted(r.status_code for r in responses) == [202, 202, 429, 429, 429]
    app.state.ai_settings = app.state.ai_settings.model_copy(update={"ai_global_limit": 3})
    foreign = await account()
    other = await item(foreign)
    responses = await asyncio.gather(*(submit(client, foreign, other) for _ in range(3)))
    assert sorted(r.status_code for r in responses) == [202, 503, 503]
    async with app.state.session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(AIJob)) == 3


@pytest.mark.parametrize(
    "payload",
    [
        {"instruction": "private\\secret\x00"},
        {"instruction": " "},
        {"instruction": "private" * 200},
        {"instruction": "private", "owner_id": "private"},
    ],
)
async def test_rewrite_validation_never_echoes_input_or_inserts_job(
    app, client, account, item, payload
):
    owner = await account()
    created = await item(owner)
    response = await submit(client, owner, created, json=payload)
    assert response.status_code == 422
    assert "private" not in response.text
    assert all("input" not in error for error in response.json()["detail"])
    async with app.state.session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(AIJob)) == 0


async def test_terminal_suggestion_replay_and_logical_expiry_before_purge(
    app, client, account, item
):
    owner = await account()
    created = await item(owner)
    headers = submission(owner)
    response = await submit(client, owner, created, headers=headers)
    job_id = UUID(response.json()["id"])
    async with app.state.session_factory() as session:
        now = await session.scalar(select(func.clock_timestamp()))
        await session.execute(
            update(AIJob)
            .where(AIJob.id == job_id)
            .values(
                status="succeeded",
                result_title="Clear title",
                result_description="Clear text",
                finished_at=now,
                content_expires_at=now + timedelta(days=7),
            )
        )
        await session.commit()
    replay = await submit(client, owner, created, headers=headers)
    assert replay.status_code == 200
    assert replay.json()["suggestion"] == {"title": "Clear title", "description": "Clear text"}
    unchanged = await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
    assert unchanged.json()["description"] == "before"
    async with app.state.session_factory() as session:
        await session.execute(
            update(AIJob)
            .where(AIJob.id == job_id)
            .values(content_expires_at=func.clock_timestamp() - timedelta(seconds=1))
        )
        await session.commit()
    assert (
        await client.get(response.json()["status_url"], headers=owner["headers"])
    ).status_code == 410
    assert (await submit(client, owner, created, headers=headers)).status_code == 410
    async with app.state.session_factory() as session:
        assert (await session.get(AIJob, job_id)).result_description == "Clear text"


async def test_deferred_submission_commit_failure_has_no_job_or_publishable_event(
    app, client, account, item
):
    owner = await account()
    created = await item(owner)
    async with app.state.engine.begin() as connection:
        await connection.exec_driver_sql(
            "CREATE FUNCTION test_ai_reject_commit() RETURNS trigger LANGUAGE plpgsql AS $$ "
            "BEGIN RAISE EXCEPTION 'deferred failure' USING ERRCODE = '23514'; END $$"
        )
        await connection.exec_driver_sql(
            "CREATE CONSTRAINT TRIGGER test_ai_reject_commit AFTER INSERT ON ai_outbox "
            "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION test_ai_reject_commit()"
        )
    try:
        response = await submit(client, owner, created)
        assert response.status_code == 500 and response.json() == {
            "detail": "Internal Server Error"
        }
        async with app.state.session_factory() as session:
            assert await session.scalar(select(func.count()).select_from(AIJob)) == 0
            assert await session.scalar(select(func.count()).select_from(AIOutbox)) == 0
    finally:
        async with app.state.engine.begin() as connection:
            await connection.exec_driver_sql("DROP TRIGGER test_ai_reject_commit ON ai_outbox")
            await connection.exec_driver_sql("DROP FUNCTION test_ai_reject_commit()")


async def test_disabled_rewriting_does_not_disable_crud(app, client, account, item):
    owner = await account()
    created = await item(owner)
    app.state.ai_settings = None
    assert (await submit(client, owner, created)).status_code == 503
    assert (
        await client.get(f"/api/v1/items/{created['id']}", headers=owner["headers"])
    ).status_code == 200


def test_daemon_database_settings_do_not_require_api_credentials(monkeypatch):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    settings = AISettings(database_url="postgresql+asyncpg://u:p@localhost/app_test")
    assert settings.sqlalchemy_url.database == "app_test"
