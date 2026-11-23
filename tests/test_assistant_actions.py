"""Confirming a pending unlock: own row only, once, in time, while still a resident."""

import asyncio
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from sqlalchemy import delete, func, insert, select, update

import app.assistant.actions as actions
import app.assistant.step_up as step_up
from app.assistant.residence import load_resident
from app.assistant.totp import code_at
from app.config import Settings
from app.models import AssistantToolCall, Device, Membership, PendingAction
from tests.support.assistant import (
    FRONT_DOOR,
    SIMULATOR,
    TOTP_KEY,
    StepClock,
    demo_residence,
    enable_authenticator,
    pending_unlock,
    serve,
)


def confirm_path(action_id) -> str:
    return f"/api/v1/assistant/actions/{action_id}/confirm"


@pytest.fixture
async def simulator():
    application = SIMULATOR.create_app()
    async with serve(application) as base_url:
        yield base_url


@pytest.fixture
def settings(monkeypatch, simulator) -> Settings:
    monkeypatch.setenv("OPENAI_BASE_URL", simulator + "/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key-never-external")
    monkeypatch.setenv("ASSISTANT_TOTP_KEY", TOTP_KEY)
    return Settings(
        ai_enabled=True, rag_enabled=True, assistant_enabled=True, assistant_demo_enabled=True
    )


@pytest.fixture
def clock(monkeypatch) -> StepClock:
    clock = StepClock()
    monkeypatch.setattr(step_up, "_database_step", clock)
    return clock


@pytest.fixture
def enabled(client: httpx.AsyncClient, account, clock):
    """A demo resident with an enabled authenticator."""

    async def create() -> dict:
        owner = await account()
        await demo_residence(client, owner)
        return {**owner, "totp": await enable_authenticator(client, owner, clock)}

    return create


@pytest.fixture
def pending(app: FastAPI, enabled):
    """A demo resident with an enabled authenticator and a pending front-door unlock."""

    async def create() -> tuple[dict, UUID]:
        owner = await enabled()
        return owner, await pending_unlock(app, owner)

    return create


async def confirm(client, clock: StepClock, owner: dict, action_id) -> httpx.Response:
    """Confirm with the owner's code for the current step, then move to a fresh step."""
    code = code_at(owner["totp"], clock.step)
    response = await client.post(
        confirm_path(action_id), headers=owner["headers"], json={"code": code}
    )
    clock.step += 1
    return response


async def door_locked(app: FastAPI, owner: dict) -> bool:
    async with app.state.session_factory() as session:
        resident = await load_resident(session, UUID(owner["id"]))
        return await session.scalar(
            select(Device.locked).where(
                Device.apartment_id == resident.apartment_id, Device.name == FRONT_DOOR
            )
        )


async def test_owner_confirms_once_and_the_door_unlocks(
    app: FastAPI, client: httpx.AsyncClient, clock, pending
):
    owner, action_id = await pending()
    response = await confirm(client, clock, owner, action_id)

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {
        "id": str(action_id),
        "action": "unlock_door",
        "status": "confirmed",
        "device": {"name": FRONT_DOOR, "locked": False},
    }
    assert await door_locked(app, owner) is False
    async with app.state.session_factory() as session:
        row = await session.scalar(
            select(AssistantToolCall).where(AssistantToolCall.source == "confirmation")
        )
    assert (row.tool, row.decision, row.result_code, row.turn_id) == (
        "unlock_door",
        "confirmed",
        "ok",
        None,
    )
    assert row.pending_action_id == action_id

    again = await confirm(client, clock, owner, action_id)
    assert again.status_code == 409


async def test_foreign_or_missing_action_is_not_found(
    app: FastAPI, client: httpx.AsyncClient, clock, enabled, pending
):
    owner, action_id = await pending()
    stranger = await enabled()

    foreign = await confirm(client, clock, stranger, action_id)
    missing = await confirm(client, clock, owner, uuid4())
    assert (foreign.status_code, missing.status_code) == (404, 404)
    assert foreign.json() == missing.json()
    assert await door_locked(app, owner) is True


async def test_late_confirm_marks_the_action_expired(
    app: FastAPI, client: httpx.AsyncClient, clock, pending
):
    owner, action_id = await pending()
    async with app.state.session_factory() as session:
        await session.execute(
            update(PendingAction)
            .where(PendingAction.id == action_id)
            .values(
                created_at=func.now() - func.make_interval(0, 0, 0, 0, 0, 2),
                expires_at=func.now() - func.make_interval(0, 0, 0, 0, 0, 1),
            )
        )
        await session.commit()

    response = await confirm(client, clock, owner, action_id)
    assert response.status_code == 409
    assert await door_locked(app, owner) is True
    async with app.state.session_factory() as session:
        status = await session.scalar(
            select(PendingAction.status).where(PendingAction.id == action_id)
        )
    assert status == "expired"
    assert (await confirm(client, clock, owner, action_id)).status_code == 409


async def test_concurrent_claims_have_exactly_one_winner(app: FastAPI, pending):
    """The claim itself, after step-up: four racing confirms unlock exactly once.

    Over HTTP, step-up already lets one code through per step; the claim must still hold
    when two valid codes race for one action.
    """
    owner, action_id = await pending()
    async with app.state.session_factory() as session:
        resident = await load_resident(session, UUID(owner["id"]))

    async def claim() -> int:
        async with app.state.session_factory() as session:
            try:
                await actions.confirm_action(session, resident, action_id)
            except HTTPException as error:
                return error.status_code
        return 200

    statuses = await asyncio.gather(*[claim() for _ in range(4)])

    assert sorted(statuses) == [200, 409, 409, 409]
    async with app.state.session_factory() as session:
        confirmations = await session.scalar(
            select(func.count())
            .select_from(AssistantToolCall)
            .where(AssistantToolCall.source == "confirmation")
        )
    assert confirmations == 1


async def test_revoked_membership_blocks_the_confirm(
    app: FastAPI, client: httpx.AsyncClient, clock, pending
):
    owner, action_id = await pending()
    async with app.state.session_factory() as session:
        await session.execute(
            delete(Membership).where(
                Membership.user_id == UUID(owner["id"]), Membership.role == "resident"
            )
        )
        await session.commit()

    response = await confirm(client, clock, owner, action_id)
    assert response.status_code == 403
    async with app.state.session_factory() as session:
        status = await session.scalar(
            select(PendingAction.status).where(PendingAction.id == action_id)
        )
    assert status == "pending"


async def test_confirm_requires_a_token_and_the_feature(app: FastAPI, client, clock, pending):
    assert (await client.post(confirm_path(uuid4()))).status_code == 401
    owner, action_id = await pending()
    app.state.assistant_settings = None
    assert (await confirm(client, clock, owner, action_id)).status_code == 503


async def test_co_resident_cannot_confirm_anothers_action(
    app: FastAPI, client: httpx.AsyncClient, clock, account, pending
):
    owner, action_id = await pending()
    housemate = await account()
    async with app.state.session_factory() as session:
        resident = await load_resident(session, UUID(owner["id"]))
        await session.execute(
            insert(Membership).values(
                user_id=UUID(housemate["id"]),
                building_id=resident.building_id,
                apartment_id=resident.apartment_id,
                role="resident",
            )
        )
        await session.commit()
    housemate = {**housemate, "totp": await enable_authenticator(client, housemate, clock)}

    response = await confirm(client, clock, housemate, action_id)
    assert response.status_code == 404
    assert await door_locked(app, owner) is True


async def test_failed_confirmation_audit_keeps_the_door_locked(
    app: FastAPI, client: httpx.AsyncClient, pending, monkeypatch
):
    owner, action_id = await pending()
    # A tool name outside the audit CHECK makes the audit insert fail.
    monkeypatch.setattr(actions, "ToolName", SimpleNamespace(UNLOCK_DOOR="bogus"))
    async with app.state.session_factory() as session:
        resident = await load_resident(session, UUID(owner["id"]))
        with pytest.raises(Exception, match="ck_assistant_tool_calls_tool"):
            await actions.confirm_action(session, resident, action_id)
    assert await door_locked(app, owner) is True
    async with app.state.session_factory() as session:
        status = await session.scalar(
            select(PendingAction.status).where(PendingAction.id == action_id)
        )
    assert status == "pending"
