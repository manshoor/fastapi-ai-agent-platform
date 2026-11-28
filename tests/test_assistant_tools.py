"""Assistant tools: allowlist, argument checks, ownership, limits, and the audit trail."""

import asyncio
import json
import re
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import delete, func, insert, select, update

import app.assistant.tools as tools
from app.ai.provider import OpenAIEmbedder, ToolCall
from app.assistant.contracts import MEMORY_TOOLS, Decision, ResultCode, ToolName
from app.assistant.residence import load_resident
from app.assistant.tools import ToolContext, execute
from app.config import Settings
from app.models import AssistantToolCall, AssistantTurn, Device, Membership, PendingAction, Ticket
from tests.support.assistant import SIMULATOR, TOTP_KEY, add_ready_document, demo_residence, serve

LIVING_ROOM = "Living room thermostat"
FRONT_DOOR = "Front door"
RULE_TEXT = "Quiet hours start at 22:00."


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
async def resident_context(app: FastAPI, client: httpx.AsyncClient, account):
    """A demo resident with one admitted turn, and a factory for tool contexts."""
    embedders = []

    async def create(**limits) -> tuple[ToolContext, dict]:
        owner = await account()
        view = await demo_residence(client, owner)
        factory = app.state.session_factory
        async with factory() as session:
            resident = await load_resident(session, UUID(owner["id"]))
            turn_id = await session.scalar(
                insert(AssistantTurn)
                .values(user_id=resident.user_id, model="test", prompt_version="assistant-v1")
                .returning(AssistantTurn.id)
            )
            await session.commit()
        embedder = OpenAIEmbedder(app.state.rag.settings)
        embedders.append(embedder)
        settings = app.state.assistant_settings.model_copy(update=limits)
        context = ToolContext(
            session_factory=factory,
            settings=settings,
            resident=resident,
            turn_id=turn_id,
            embedder=embedder,
        )
        return context, {**owner, "view": view}

    yield create
    for embedder in embedders:
        await embedder.aclose()


def call(name: str, **arguments) -> ToolCall:
    return ToolCall(name=name, call_id=f"call_{uuid4().hex}", arguments=json.dumps(arguments))


async def device(app: FastAPI, ctx: ToolContext, name: str) -> Device:
    async with app.state.session_factory() as session:
        return await session.scalar(
            select(Device).where(
                Device.apartment_id == ctx.resident.apartment_id, Device.name == name
            )
        )


async def audit(app: FastAPI, ctx: ToolContext) -> list[AssistantToolCall]:
    async with app.state.session_factory() as session:
        rows = await session.scalars(
            select(AssistantToolCall)
            .where(AssistantToolCall.user_id == ctx.resident.user_id)
            .order_by(AssistantToolCall.created_at)
        )
        return list(rows)


async def simulator_calls(simulator: str) -> int:
    async with httpx.AsyncClient() as http:
        return (await http.get(simulator + "/__state")).json()["calls"]


async def test_set_thermostat_changes_only_that_device_and_audits_the_value(
    app: FastAPI, resident_context
):
    ctx, _ = await resident_context()
    outcome = await execute(ctx, call("set_thermostat", device=LIVING_ROOM, celsius=22.5))

    assert (outcome.action.decision, outcome.action.result) == (Decision.ALLOWED, ResultCode.OK)
    assert outcome.output == {
        "result": "ok",
        "device": LIVING_ROOM,
        "label": "T1",
        "fact": f"{LIVING_ROOM} target set to 22.5 °C.",
    }
    assert float((await device(app, ctx, LIVING_ROOM)).target_celsius) == 22.5
    assert float((await device(app, ctx, "Bedroom thermostat")).target_celsius) == 19.5
    [row] = await audit(app, ctx)
    assert (row.tool, row.decision, row.result_code) == ("set_thermostat", "allowed", "ok")
    assert row.details == {"celsius": 22.5}
    assert row.turn_id == ctx.turn_id and row.source == "assistant"


@pytest.mark.parametrize(
    "arguments",
    [
        {"device": LIVING_ROOM, "celsius": 35},
        {"device": LIVING_ROOM, "celsius": 21.3},
        {"device": LIVING_ROOM, "celsius": "21"},
        {"device": LIVING_ROOM},
        {"device": LIVING_ROOM, "celsius": 21, "apartment_id": str(uuid4())},
        {"device": "", "celsius": 21},
        {"device": "a\x00b", "celsius": 21},
    ],
    ids=["too-hot", "off-step", "string", "missing", "extra-field", "empty", "nul"],
)
async def test_invalid_arguments_are_denied_and_change_nothing(
    app: FastAPI, resident_context, arguments
):
    ctx, _ = await resident_context()
    outcome = await execute(ctx, call("set_thermostat", **arguments))

    assert outcome.action.result == ResultCode.INVALID_ARGUMENTS
    assert outcome.output == {"result": "invalid_arguments"}
    assert float((await device(app, ctx, LIVING_ROOM)).target_celsius) == 21.0
    [row] = await audit(app, ctx)
    assert (row.decision, row.result_code, row.details) == ("denied", "invalid_arguments", {})


async def test_malformed_json_arguments_are_invalid(resident_context):
    ctx, _ = await resident_context()
    bad = ToolCall(name="lock_door", call_id="call_x", arguments="{not json")
    assert (await execute(ctx, bad)).action.result == ResultCode.INVALID_ARGUMENTS


async def test_unknown_tool_is_audited_without_the_models_name(app: FastAPI, resident_context):
    ctx, _ = await resident_context()
    outcome = await execute(ctx, call("disarm_alarm", device="Alarm"))

    assert (outcome.action.tool, outcome.action.result) == ("unknown", ResultCode.UNKNOWN_TOOL)
    assert (await device(app, ctx, "Alarm")).armed is True
    [row] = await audit(app, ctx)
    assert (row.tool, row.decision, row.result_code) == ("unknown", "denied", "unknown_tool")


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("set_thermostat", {"device": "Alarm", "celsius": 21}),
        ("lock_door", {"device": "Alarm"}),
        ("unlock_door", {"device": "Alarm"}),
        ("set_thermostat", {"device": FRONT_DOOR, "celsius": 21}),
        ("unlock_door", {"device": LIVING_ROOM}),
        ("lock_door", {"device": "Garage"}),
    ],
    ids=["alarm-thermostat", "alarm-lock", "alarm-unlock", "wrong-kind", "thermostat", "missing"],
)
async def test_wrong_kind_or_unknown_device_is_not_found(
    app: FastAPI, resident_context, name, arguments
):
    ctx, _ = await resident_context()
    outcome = await execute(ctx, call(name, **arguments))

    assert outcome.action.result == ResultCode.NOT_FOUND
    alarm = await device(app, ctx, "Alarm")
    assert alarm.armed is True
    assert (await device(app, ctx, FRONT_DOOR)).locked is True
    async with app.state.session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(PendingAction)) == 0


async def test_another_apartments_device_is_not_found(app: FastAPI, resident_context):
    mine, _ = await resident_context()
    theirs, _ = await resident_context()
    # Both demo apartments have a "Front door"; the name resolves only in the caller's apartment.
    async with app.state.session_factory() as session:
        await session.execute(
            update(Device)
            .where(Device.apartment_id == theirs.resident.apartment_id, Device.name == FRONT_DOOR)
            .values(name="Their door", locked=False)
        )
        await session.commit()

    outcome = await execute(mine, call("lock_door", device="Their door"))
    assert outcome.action.result == ResultCode.NOT_FOUND
    assert (await device(app, theirs, "Their door")).locked is False


async def test_revoked_membership_blocks_every_write(app: FastAPI, resident_context):
    ctx, _ = await resident_context()
    async with app.state.session_factory() as session:
        await session.execute(
            delete(Membership).where(
                Membership.user_id == ctx.resident.user_id, Membership.role == "resident"
            )
        )
        await session.commit()

    attempts = [
        call("set_thermostat", device=LIVING_ROOM, celsius=24),
        call("lock_door", device=FRONT_DOOR),
        call("unlock_door", device=FRONT_DOOR),
        call("create_ticket", category="plumbing", summary="Leaking tap"),
        call("list_bookings"),
    ]
    for attempt in attempts:
        outcome = await execute(ctx, attempt)
        assert outcome.action.result == ResultCode.NOT_PERMITTED, attempt.name
    assert float((await device(app, ctx, LIVING_ROOM)).target_celsius) == 21.0
    async with app.state.session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(Ticket)) == 0
        assert await session.scalar(select(func.count()).select_from(PendingAction)) == 0


async def test_device_action_limit_admits_exactly_n_under_concurrency(
    app: FastAPI, resident_context
):
    ctx, _ = await resident_context(assistant_device_action_limit=3)
    attempts = [
        execute(ctx, call("set_thermostat", device=LIVING_ROOM, celsius=17 + index))
        for index in range(5)
    ]
    outcomes = await asyncio.gather(*attempts)

    results = sorted(str(outcome.action.result) for outcome in outcomes)
    assert results == ["limit_exceeded", "limit_exceeded", "ok", "ok", "ok"]
    # The limit is shared by every device tool.
    blocked = await execute(ctx, call("lock_door", device=FRONT_DOOR))
    assert blocked.action.result == ResultCode.LIMIT_EXCEEDED


async def test_failed_audit_insert_rolls_back_the_device_write(
    app: FastAPI, resident_context, monkeypatch
):
    ctx, _ = await resident_context()
    original = tools._audit

    def broken(*args, **kwargs):
        # A free-text details key violates the audit CHECK.
        return original(*args, **{**kwargs, "details": {"note": "text"}})

    monkeypatch.setattr(tools, "_audit", broken)
    with pytest.raises(Exception, match="ck_assistant_tool_calls_details"):
        await execute(ctx, call("set_thermostat", device=LIVING_ROOM, celsius=25))
    assert float((await device(app, ctx, LIVING_ROOM)).target_celsius) == 21.0
    assert await audit(app, ctx) == []


async def test_ticket_is_deduplicated_and_limited(app: FastAPI, resident_context):
    ctx, _ = await resident_context(assistant_ticket_limit=2)
    first = await execute(ctx, call("create_ticket", category="plumbing", summary="Leaking tap"))
    retry = await execute(ctx, call("create_ticket", category="plumbing", summary="Tap leaks"))
    second = await execute(ctx, call("create_ticket", category="heating", summary="Cold radiator"))
    third = await execute(ctx, call("create_ticket", category="other", summary="Squeaky door"))

    assert first.action.result == ResultCode.OK
    assert (retry.action.result, retry.action.ticket_id) == (
        ResultCode.DUPLICATE,
        first.action.ticket_id,
    )
    assert second.action.result == ResultCode.OK
    assert third.action.result == ResultCode.LIMIT_EXCEEDED
    assert [outcome.output.get("fact") for outcome in (first, retry, second, third)] == [
        "Maintenance ticket opened in category plumbing.",
        "An open plumbing ticket from the last 10 minutes already exists.",
        "Maintenance ticket opened in category heating.",
        None,
    ]
    async with app.state.session_factory() as session:
        summaries = await session.scalars(select(Ticket.summary).order_by(Ticket.created_at))
        assert list(summaries) == ["Leaking tap", "Cold radiator"]
    rows = await audit(app, ctx)
    assert [row.details for row in rows] == [
        {"category": "plumbing"},
        {"category": "plumbing"},
        {"category": "heating"},
        {},
    ]


async def test_unlock_creates_a_pending_action_and_never_unlocks(app: FastAPI, resident_context):
    ctx, _ = await resident_context(assistant_unlock_limit=2)
    first = await execute(ctx, call("unlock_door", device=FRONT_DOOR))
    again = await execute(ctx, call("unlock_door", device=FRONT_DOOR))

    assert (first.action.decision, first.action.result) == (
        Decision.PENDING_CONFIRMATION,
        ResultCode.OK,
    )
    assert first.output == {
        "result": "ok",
        "device": FRONT_DOOR,
        "status": "confirmation_required",
        "label": "T1",
        "fact": f"Unlock request for {FRONT_DOOR} is waiting for your confirmation in the app.",
    }
    assert again.action.result == ResultCode.ALREADY_PENDING
    assert again.action.pending_action_id == first.action.pending_action_id
    assert (await device(app, ctx, FRONT_DOOR)).locked is True

    # A stale request no longer blocks a new one.
    async with app.state.session_factory() as session:
        await session.execute(
            update(PendingAction)
            .where(PendingAction.id == first.action.pending_action_id)
            .values(
                created_at=func.now() - func.make_interval(0, 0, 0, 0, 0, 2),
                expires_at=func.now() - func.make_interval(0, 0, 0, 0, 0, 1),
            )
        )
        await session.commit()
    renewed = await execute(ctx, call("unlock_door", device=FRONT_DOOR))
    assert renewed.action.result == ResultCode.OK
    assert renewed.action.pending_action_id != first.action.pending_action_id
    async with app.state.session_factory() as session:
        stale = await session.scalar(
            select(PendingAction.status).where(PendingAction.id == first.action.pending_action_id)
        )
        assert stale == "expired"

    blocked = await execute(ctx, call("unlock_door", device="Balcony door"))
    assert blocked.action.result == ResultCode.LIMIT_EXCEEDED


async def test_list_bookings_returns_only_the_apartments_upcoming_bookings(resident_context):
    ctx, _ = await resident_context()
    await resident_context()
    outcome = await execute(ctx, call("list_bookings"))

    assert outcome.action.result == ResultCode.OK
    assert [booking["amenity"] for booking in outcome.output["bookings"]] == ["Gym", "Party room"]


# One booking clause; the end carries a date only when it falls on another day.
BOOKING_CLAUSE = r"\d{4}-\d{2}-\d{2} from \d{2}:\d{2} to (?:\d{4}-\d{2}-\d{2} )?\d{2}:\d{2} UTC"


async def test_grounding_results_get_fact_labels_in_order(resident_context):
    ctx, _ = await resident_context()
    warmer = await execute(ctx, call("set_thermostat", device=LIVING_ROOM, celsius=22.5))
    missing = await execute(ctx, call("lock_door", device="Garage door"))
    locked = await execute(ctx, call("lock_door", device=FRONT_DOOR))
    bookings = await execute(ctx, call("list_bookings"))

    # A denial gets no label, so no claim can cite it.
    assert missing.action.decision == Decision.DENIED and "label" not in missing.output
    labelled = [warmer.output, locked.output, bookings.output]
    assert [output["label"] for output in labelled] == ["T1", "T2", "T3"]
    assert ctx.facts == [output["fact"] for output in labelled]
    assert locked.output["fact"] == f"{FRONT_DOOR} is locked."
    assert re.fullmatch(
        rf"Upcoming bookings: Gym on {BOOKING_CLAUSE}; Party room on {BOOKING_CLAUSE}\.",
        bookings.output["fact"],
    )


async def test_later_result_for_a_device_blanks_its_earlier_fact(resident_context):
    ctx, _ = await resident_context()
    await execute(ctx, call("set_thermostat", device=LIVING_ROOM, celsius=21))
    locked = await execute(ctx, call("lock_door", device=FRONT_DOOR))
    pending = await execute(ctx, call("unlock_door", device=FRONT_DOOR))
    opened = await execute(ctx, call("create_ticket", category="heating", summary="Cold"))
    again = await execute(ctx, call("create_ticket", category="heating", summary="Still cold"))
    warmer = await execute(ctx, call("set_thermostat", device=LIVING_ROOM, celsius=23))

    # Labels never move. Only the replaced thermostat fact is blank: the door stays
    # locked until the resident confirms, and the first ticket was still opened.
    assert warmer.output["label"] == "T6"
    assert ctx.facts == [
        "",
        locked.output["fact"],
        pending.output["fact"],
        opened.output["fact"],
        again.output["fact"],
        f"{LIVING_ROOM} target set to 23 °C.",
    ]


async def test_search_without_building_documents_makes_no_provider_call(
    resident_context, simulator
):
    ctx, _ = await resident_context()
    outcome = await execute(ctx, call("search_documents", question="When are quiet hours?"))

    assert outcome.action.result == ResultCode.NO_DOCUMENTS
    assert await simulator_calls(simulator) == 0


async def test_search_labels_continue_across_calls_and_stop_at_eight(
    app: FastAPI, resident_context
):
    ctx, owner = await resident_context()
    texts = [f"Rule {index}: quiet hours start at {20 + index % 3}:00." for index in range(10)]
    await add_ready_document(
        app, ctx.resident.user_id, "House rules", texts, building_id=ctx.resident.building_id
    )
    # Private documents of the same user stay out of building search.
    await add_ready_document(app, ctx.resident.user_id, "Private", ["Private note."])

    labels = []
    for _ in range(2):
        outcome = await execute(ctx, call("search_documents", question="quiet hours"))
        assert outcome.action.result == ResultCode.OK
        labels += [passage["label"] for passage in outcome.output["passages"]]
        assert {passage["title"] for passage in outcome.output["passages"]} == {"House rules"}
    assert labels == [f"P{index}" for index in range(1, 9)]

    capped = await execute(ctx, call("search_documents", question="quiet hours"))
    assert capped.action.result == ResultCode.LIMIT_EXCEEDED
    rows = await audit(app, ctx)
    assert [row.details for row in rows] == [{"passage_count": 4}, {"passage_count": 4}, {}]


async def test_every_enum_value_is_accepted_by_the_audit_table(app: FastAPI, resident_context):
    ctx, _ = await resident_context()
    values = [
        *[{"tool": str(tool)} for tool in ToolName],
        {"tool": tools.UNKNOWN_TOOL},
        *[{"decision": str(decision)} for decision in Decision],
        *[{"result_code": str(result)} for result in ResultCode],
    ]
    async with app.state.session_factory() as session:
        for value in values:
            row = {
                "turn_id": ctx.turn_id,
                "user_id": ctx.resident.user_id,
                "source": "assistant",
                "tool": "lock_door",
                "decision": "allowed",
                "result_code": "ok",
                **value,
            }
            await session.execute(insert(AssistantToolCall).values(**row))
        await session.commit()


def test_tool_schemas_are_strict_and_cover_the_allowlist(app):
    # Memory is off in this suite: its tools are not offered.
    schemas = tools.tool_schemas(app.state.assistant_settings)
    names = [schema["name"] for schema in schemas]
    assert names == [str(tool) for tool in ToolName if tool not in MEMORY_TOOLS]
    for schema in schemas:
        parameters = schema["parameters"]
        assert schema["strict"] is True
        assert parameters["additionalProperties"] is False
        assert sorted(parameters["required"]) == sorted(parameters["properties"])


async def co_resident(app: FastAPI, ctx: ToolContext, factory) -> ToolContext:
    """A second resident of the same apartment, with their own turn."""
    other, _ = await factory()
    async with app.state.session_factory() as session:
        await session.execute(
            update(Membership)
            .where(Membership.user_id == other.resident.user_id, Membership.role == "resident")
            .values(apartment_id=ctx.resident.apartment_id, building_id=ctx.resident.building_id)
        )
        await session.commit()
        resident = await load_resident(session, other.resident.user_id)
    other.resident = resident
    return other


async def test_co_residents_pending_unlock_is_not_handed_out(app: FastAPI, resident_context):
    mine, _ = await resident_context()
    theirs = await co_resident(app, mine, resident_context)
    first = await execute(mine, call("unlock_door", device=FRONT_DOOR))
    second = await execute(theirs, call("unlock_door", device=FRONT_DOOR))

    assert first.action.result == ResultCode.OK
    # The live request is someone else's: no id, and nothing to confirm.
    assert (second.action.decision, second.action.result) == (
        Decision.DENIED,
        ResultCode.ALREADY_PENDING,
    )
    assert second.action.pending_action_id is None
    rows = await audit(app, theirs)
    assert [(row.decision, row.pending_action_id) for row in rows] == [("denied", None)]


async def test_search_rechecks_membership(app: FastAPI, resident_context, simulator):
    ctx, _ = await resident_context()
    await add_ready_document(
        app, ctx.resident.user_id, "House rules", [RULE_TEXT], building_id=ctx.resident.building_id
    )
    async with app.state.session_factory() as session:
        await session.execute(
            delete(Membership).where(
                Membership.user_id == ctx.resident.user_id, Membership.role == "resident"
            )
        )
        await session.commit()

    outcome = await execute(ctx, call("search_documents", question="quiet hours"))
    assert outcome.action.result == ResultCode.NOT_PERMITTED
    assert "passages" not in outcome.output
    assert await simulator_calls(simulator) == 0


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("create_ticket", {"category": "plumbing", "summary": "Leaking tap"}),
        ("unlock_door", {"device": FRONT_DOOR}),
    ],
)
async def test_failed_audit_insert_rolls_back_other_writes(
    app: FastAPI, resident_context, monkeypatch, name, arguments
):
    ctx, _ = await resident_context()
    original = tools._audit

    def broken(*args, **kwargs):
        return original(*args, **{**kwargs, "details": {"note": "text"}})

    monkeypatch.setattr(tools, "_audit", broken)
    with pytest.raises(Exception, match="ck_assistant_tool_calls_details"):
        await execute(ctx, call(name, **arguments))
    async with app.state.session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(Ticket)) == 0
        assert await session.scalar(select(func.count()).select_from(PendingAction)) == 0


@pytest.mark.parametrize(
    ("name", "arguments", "limit"),
    [
        ("create_ticket", None, {"assistant_ticket_limit": 2}),
        ("unlock_door", None, {"assistant_unlock_limit": 1}),
    ],
)
async def test_ticket_and_unlock_limits_hold_under_concurrency(
    app: FastAPI, resident_context, name, arguments, limit
):
    ctx, _ = await resident_context(**limit)
    if name == "create_ticket":
        categories = ["plumbing", "electrical", "heating", "appliance"]
        attempts = [call(name, category=c, summary="Broken") for c in categories]
    else:
        attempts = [call(name, device=FRONT_DOOR), call(name, device="Balcony door")] * 2
    outcomes = await asyncio.gather(*(execute(ctx, attempt) for attempt in attempts))

    allowed = [o for o in outcomes if o.action.result == ResultCode.OK]
    assert len(allowed) == next(iter(limit.values()))


async def test_refusal_after_a_search_tells_the_model_to_answer(
    app: FastAPI, resident_context, simulator
):
    ctx, _ = await resident_context()
    await add_ready_document(
        app, ctx.resident.user_id, "House rules", [RULE_TEXT], building_id=ctx.resident.building_id
    )
    await execute(ctx, call("search_documents", question="quiet hours"))
    outcome = await execute(ctx, call("lock_door", device=FRONT_DOOR))
    assert outcome.output == {
        "result": "not_permitted",
        "reason": tools.REFUSED_AFTER_SEARCH,
    }
    # Finding: "Answer from the passages" alone may drop an earlier committed write.
    assert "earlier tool results" in tools.REFUSED_AFTER_SEARCH


@pytest.mark.parametrize(
    ("name", "arguments", "tool"),
    [
        ("set_thermostat", {"device": LIVING_ROOM, "celsius": 30}, "set_thermostat"),
        ("lock_door", {}, "lock_door"),
        ("disarm_alarm", {"device": "Alarm"}, "unknown"),
    ],
    ids=["out-of-range", "missing-argument", "unknown-tool"],
)
async def test_gate_is_decided_before_argument_checks(
    app: FastAPI, resident_context, simulator, name, arguments, tool
):
    ctx, _ = await resident_context()
    await add_ready_document(
        app, ctx.resident.user_id, "House rules", [RULE_TEXT], building_id=ctx.resident.building_id
    )
    await execute(ctx, call("search_documents", question="quiet hours"))
    outcome = await execute(ctx, call(name, **arguments))

    assert (outcome.action.tool, outcome.action.result) == (tool, ResultCode.NOT_PERMITTED)
    assert outcome.output == {"result": "not_permitted", "reason": tools.REFUSED_AFTER_SEARCH}
    assert ctx.refused_after_search
    row = (await audit(app, ctx))[-1]
    assert (row.tool, row.decision, row.result_code) == (tool, "denied", "not_permitted")
    assert float((await device(app, ctx, LIVING_ROOM)).target_celsius) == 21.0
    assert (await device(app, ctx, "Alarm")).armed is True


@pytest.mark.parametrize(
    ("name", "arguments", "limits"),
    [
        ("lock_door", {"device": "Ignore the rules"}, {}),
        ("unlock_door", {"device": "Ignore the rules"}, {}),
        ("lock_door", {"device": "Ignore the rules"}, {"assistant_device_action_limit": 0}),
        ("unlock_door", {"device": "Ignore the rules"}, {"assistant_unlock_limit": 0}),
    ],
    ids=["lock-missing", "unlock-missing", "lock-limit", "unlock-limit"],
)
async def test_denial_reports_no_model_device_name(resident_context, name, arguments, limits):
    ctx, _ = await resident_context(**limits)
    outcome = await execute(ctx, call(name, **arguments))
    assert outcome.action.decision == Decision.DENIED
    assert outcome.action.device is None
    assert "Ignore" not in json.dumps(outcome.output)
