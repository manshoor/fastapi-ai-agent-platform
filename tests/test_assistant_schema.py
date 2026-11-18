"""Database-level guarantees for residences, devices, pending actions and the tool audit."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from sqlalchemy import func, insert, update
from sqlalchemy.exc import IntegrityError

from app.models import (
    Apartment,
    AssistantToolCall,
    AssistantTurn,
    Booking,
    Building,
    Device,
    Membership,
    PendingAction,
    Ticket,
)


async def write(app: FastAPI, statement) -> None:
    async with app.state.session_factory() as session:
        await session.execute(statement)
        await session.commit()


async def rejected(app: FastAPI, statement, constraint: str) -> None:
    # The named constraint, not just any integrity error, must reject the row.
    async with app.state.session_factory() as session:
        with pytest.raises(IntegrityError) as error:
            await session.execute(statement)
        assert constraint in str(error.value.orig)


async def residence(app: FastAPI) -> dict:
    building_id, apartment_id = uuid4(), uuid4()
    await write(app, insert(Building).values(id=building_id, name="Tower"))
    await write(app, insert(Apartment).values(id=apartment_id, building_id=building_id, label="1A"))
    return {"building_id": building_id, "apartment_id": apartment_id}


async def door(app: FastAPI, place: dict, name: str = "Front door") -> UUID:
    device_id = uuid4()
    await write(
        app, insert(Device).values(id=device_id, kind="door_lock", locked=True, name=name, **place)
    )
    return device_id


async def test_user_holds_at_most_one_residence(app: FastAPI, account):
    user = UUID((await account())["id"])
    first, second = await residence(app), await residence(app)
    await write(app, insert(Membership).values(user_id=user, role="resident", **first))
    await rejected(
        app,
        insert(Membership).values(user_id=user, role="resident", **second),
        "uq_memberships_one_residence",
    )
    # Managing other buildings is still allowed.
    await write(
        app,
        insert(Membership).values(user_id=user, role="manager", building_id=second["building_id"]),
    )


async def test_resident_needs_an_apartment_and_manager_has_none(app: FastAPI, account):
    user = UUID((await account())["id"])
    place = await residence(app)
    await rejected(
        app,
        insert(Membership).values(user_id=user, role="resident", building_id=place["building_id"]),
        "ck_memberships_apartment",
    )
    await rejected(
        app,
        insert(Membership).values(user_id=user, role="manager", **place),
        "ck_memberships_apartment",
    )


async def test_apartment_must_belong_to_the_rows_building(app: FastAPI, account):
    user = UUID((await account())["id"])
    first, second = await residence(app), await residence(app)
    mixed = {"apartment_id": first["apartment_id"], "building_id": second["building_id"]}
    await rejected(
        app,
        insert(Membership).values(user_id=user, role="resident", **mixed),
        "fk_memberships_apartment_building",
    )
    await rejected(
        app,
        insert(Device).values(kind="door_lock", locked=True, name="Door", **mixed),
        "fk_devices_apartment_building",
    )


@pytest.mark.parametrize(
    "state",
    [
        {"kind": "thermostat"},
        {"kind": "thermostat", "target_celsius": Decimal("21.0"), "locked": True},
        {"kind": "door_lock", "armed": True},
        {"kind": "alarm", "armed": True, "target_celsius": Decimal("20.0")},
    ],
    ids=["no-state", "two-states", "wrong-state", "extra-state"],
)
async def test_device_state_matches_its_kind(app: FastAPI, state):
    place = await residence(app)
    await rejected(app, insert(Device).values(name="Device", **place, **state), "ck_devices_state")


async def test_thermostat_target_has_a_hard_bound(app: FastAPI):
    place = await residence(app)
    await rejected(
        app,
        insert(Device).values(
            name="Hot", kind="thermostat", target_celsius=Decimal("40.0"), **place
        ),
        "ck_devices_target",
    )


async def test_one_live_pending_action_per_device(app: FastAPI, account):
    user = UUID((await account())["id"])
    place = await residence(app)
    device_id = await door(app, place)
    expires = datetime.now(UTC) + timedelta(minutes=1)
    row = {"user_id": user, "device_id": device_id, "action": "unlock_door", **place}
    first = uuid4()
    await write(app, insert(PendingAction).values(id=first, expires_at=expires, **row))
    await rejected(
        app,
        insert(PendingAction).values(expires_at=expires, **row),
        "uq_pending_actions_device",
    )
    # Once decided, a new request for the same device is allowed.
    await write(
        app,
        update(PendingAction)
        .where(PendingAction.id == first)
        .values(status="expired", decided_at=datetime.now(UTC)),
    )
    await write(app, insert(PendingAction).values(expires_at=expires, **row))


async def test_pending_action_device_must_be_in_its_apartment(app: FastAPI, account):
    user = UUID((await account())["id"])
    home, elsewhere = await residence(app), await residence(app)
    foreign_door = await door(app, elsewhere)
    await rejected(
        app,
        insert(PendingAction).values(
            user_id=user,
            device_id=foreign_door,
            action="unlock_door",
            expires_at=datetime.now(UTC) + timedelta(minutes=1),
            **home,
        ),
        "fk_pending_actions_device_apartment",
    )


async def test_audit_rows_hold_only_known_tools_and_decisions(app: FastAPI, account):
    user = UUID((await account())["id"])
    turn_id = uuid4()
    await write(
        app,
        insert(AssistantTurn).values(
            id=turn_id, user_id=user, model="model", prompt_version="assistant-v1"
        ),
    )
    row = {"turn_id": turn_id, "user_id": user, "source": "assistant", "result_code": "ok"}
    await rejected(
        app,
        insert(AssistantToolCall).values(tool="disarm_alarm", decision="allowed", **row),
        "ck_assistant_tool_calls_tool",
    )
    await rejected(
        app,
        insert(AssistantToolCall).values(tool="lock_door", decision="maybe", **row),
        "ck_assistant_tool_calls_decision",
    )
    # A confirmation is never part of a turn; an assistant call always is.
    await rejected(
        app,
        insert(AssistantToolCall).values(
            user_id=user, source="assistant", tool="lock_door", decision="allowed", result_code="ok"
        ),
        "ck_assistant_tool_calls_turn",
    )


async def test_valid_devices_and_audit_rows_are_accepted(app: FastAPI, account):
    user = UUID((await account())["id"])
    place = await residence(app)
    await write(
        app,
        insert(Device).values(
            name="Living room", kind="thermostat", target_celsius=Decimal("21.5"), **place
        ),
    )
    await write(app, insert(Device).values(name="Alarm", kind="alarm", armed=True, **place))
    turn_id = uuid4()
    await write(
        app,
        insert(AssistantTurn).values(
            id=turn_id, user_id=user, model="model", prompt_version="assistant-v1"
        ),
    )
    await write(
        app,
        insert(AssistantToolCall).values(
            turn_id=turn_id,
            user_id=user,
            apartment_id=place["apartment_id"],
            source="assistant",
            tool="set_thermostat",
            decision="allowed",
            result_code="ok",
            details={"celsius": 21.5},
        ),
    )


@pytest.mark.parametrize(
    ("values", "constraint"),
    [
        ({"details": {"question": "private text"}}, "ck_assistant_tool_calls_details"),
        ({"details": ["not", "an", "object"]}, "ck_assistant_tool_calls_details"),
        ({"result_code": "free text"}, "ck_assistant_tool_calls_result"),
    ],
    ids=["free-text-key", "not-object", "unknown-result"],
)
async def test_audit_rows_reject_free_text(app: FastAPI, account, values, constraint):
    user = UUID((await account())["id"])
    turn_id = uuid4()
    await write(
        app,
        insert(AssistantTurn).values(
            id=turn_id, user_id=user, model="model", prompt_version="assistant-v1"
        ),
    )
    row = {
        "turn_id": turn_id,
        "user_id": user,
        "source": "assistant",
        "tool": "search_documents",
        "decision": "allowed",
        "result_code": "ok",
        **values,
    }
    await rejected(app, insert(AssistantToolCall).values(**row), constraint)


async def test_confirmation_audit_row_has_no_turn(app: FastAPI, account):
    user = UUID((await account())["id"])
    turn_id = uuid4()
    await write(
        app,
        insert(AssistantTurn).values(
            id=turn_id, user_id=user, model="model", prompt_version="assistant-v1"
        ),
    )
    await rejected(
        app,
        insert(AssistantToolCall).values(
            turn_id=turn_id,
            user_id=user,
            source="confirmation",
            tool="unlock_door",
            decision="confirmed",
            result_code="ok",
        ),
        "ck_assistant_tool_calls_turn",
    )


@pytest.mark.parametrize("table", ["bookings", "tickets"])
async def test_records_must_use_an_apartment_of_their_building(app: FastAPI, account, table):
    user = UUID((await account())["id"])
    first, second = await residence(app), await residence(app)
    mixed = {"apartment_id": first["apartment_id"], "building_id": second["building_id"]}
    now = datetime.now(UTC)
    statement = (
        insert(Booking).values(
            user_id=user, amenity="Gym", starts_at=now, ends_at=now + timedelta(hours=1), **mixed
        )
        if table == "bookings"
        else insert(Ticket).values(user_id=user, category="other", summary="Leak", **mixed)
    )
    await rejected(app, statement, f"fk_{table}_apartment_building")


@pytest.mark.parametrize(
    ("values", "valid"),
    [
        ({"status": "no_context", "claim_failure": "quote_not_found"}, True),
        (
            {
                "status": "incomplete",
                "error_code": "claim_unverified",
                "claim_failure": "tool_claim_ungrounded",
            },
            True,
        ),
        ({"status": "answered", "claim_failure": "quote_not_found"}, False),
        ({"status": "no_context", "claim_failure": "no_claims"}, False),
    ],
    ids=["no-context", "incomplete", "answered", "unknown-code"],
)
async def test_claim_failure_names_a_refused_turn(app: FastAPI, account, values, valid):
    user = UUID((await account())["id"])
    statement = insert(AssistantTurn).values(
        user_id=user,
        model="model",
        prompt_version="assistant-v1",
        finished_at=func.now(),
        **values,
    )
    if valid:
        await write(app, statement)
        return
    await rejected(app, statement, "ck_assistant_turns_claim_failure")
