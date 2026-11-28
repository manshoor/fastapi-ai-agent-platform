"""Assistant turns: the agent loop, budgets, grounding, failures, limits, and privacy."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import timedelta
from uuid import UUID

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_core.runnables import RunnableLambda
from langchain_core.tracers.langchain import wait_for_all_tracers
from sqlalchemy import func, select
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

import app.assistant.tools as tools
import app.assistant.turns as turns_module
from app.assistant.residence import load_resident
from app.assistant.turns import INCOMPLETE_ANSWER, NO_CONTEXT_ANSWER, REFUSED_NOTICE, run_turn
from app.config import Settings
from app.models import AssistantToolCall, AssistantTurn, Device
from tests.support.assistant import SIMULATOR, TOTP_KEY, add_ready_document, demo_residence, serve

URL = "/api/v1/assistant/messages"
TURN_LIMIT = 3
SECRET = "private-message-7f3a"
# No digits: every number in a statement must appear in the fact it cites.
REPLY_SECRET = "private-reply-marker"
RULE = "Quiet hours start at 22:00 and end at 07:00."


# Quoted from no tool fact: cited as T1, it grounds nothing.
UNGROUNDED_QUOTE = "the request is done"


def tool_reply(statement: str, quote: str = UNGROUNDED_QUOTE, source: str = "T1") -> dict:
    return {"claims": [{"statement": statement, "source": source, "quote": quote}]}


def passage_reply(statement: str, quote: str, source: str = "P1") -> dict:
    return {"claims": [{"statement": statement, "source": source, "quote": quote}]}


@pytest.fixture
async def simulator():
    application = SIMULATOR.create_app()
    async with serve(application) as base_url:
        yield application.state.simulator, base_url


@pytest.fixture
def settings(monkeypatch, simulator) -> Settings:
    _, base_url = simulator
    for name, value in {
        "OPENAI_BASE_URL": base_url + "/v1",
        "OPENAI_API_KEY": "synthetic-key-never-external",
        "ASSISTANT_TOTP_KEY": TOTP_KEY,
        "AI_PROVIDER_TIMEOUT_SECONDS": 1,
        "AI_EMBEDDING_TIMEOUT_SECONDS": 1,
        "ASSISTANT_TURN_LIMIT": TURN_LIMIT,
    }.items():
        monkeypatch.setenv(name, str(value))
    return Settings(
        ai_enabled=True, rag_enabled=True, assistant_enabled=True, assistant_demo_enabled=True
    )


@pytest.fixture
def resident(client: httpx.AsyncClient, account):
    async def create() -> dict:
        owner = await account()
        view = await demo_residence(client, owner)
        return {**owner, "view": view}

    return create


def script(simulator, *steps, modes=()):
    state, _ = simulator
    state.script = list(steps)
    state.modes = list(modes)
    state.delay_seconds = 2


async def send(client, owner, message: str = f"Please help {SECRET}"):
    return await client.post(URL, headers=owner["headers"], json={"message": message})


async def turns(app: FastAPI) -> list[AssistantTurn]:
    async with app.state.session_factory() as session:
        return list(await session.scalars(select(AssistantTurn).order_by(AssistantTurn.created_at)))


async def target(app: FastAPI, owner: dict, name: str = "Living room thermostat") -> float:
    async with app.state.session_factory() as session:
        resident = await load_resident(session, UUID(owner["id"]))
        value = await session.scalar(
            select(Device.target_celsius).where(
                Device.apartment_id == resident.apartment_id, Device.name == name
            )
        )
    return float(value)


async def test_tool_then_reply_is_answered_and_recorded_without_text(
    app: FastAPI, client, resident, simulator
):
    owner = await resident()
    script(
        simulator,
        {
            "tool": "set_thermostat",
            "arguments": {"device": "Living room thermostat", "celsius": 23},
        },
        tool_reply(f"Set to 23 degrees {REPLY_SECRET}", "target set to 23"),
    )
    response = await send(client, owner)

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["status"] == "answered"
    assert body["citations"] == [] and body["pending_actions"] == []
    assert body["actions"] == [
        {
            "tool": "set_thermostat",
            "decision": "allowed",
            "result": "ok",
            "device": "Living room thermostat",
            "ticket_id": None,
            "pending_action_id": None,
        }
    ]
    assert await target(app, owner) == 23.0
    (row,) = await turns(app)
    assert (row.status, row.model_calls, row.tool_calls) == ("answered", 2, 1)
    assert (row.input_tokens, row.output_tokens, row.error_code) == (80, 40, None)
    assert row.claim_failure is None and row.prompt_version == "assistant-v7"
    assert row.model == "simulator-assistant" and row.finished_at is not None
    async with app.state.session_factory() as session:
        stored = [
            {column: str(getattr(item, column)) for column in item.__table__.columns.keys()}
            for item in [*(await turns(app)), *await session.scalars(select(AssistantToolCall))]
        ]
    assert SECRET not in json.dumps(stored)
    assert REPLY_SECRET not in json.dumps(stored)


async def test_the_model_sees_context_without_the_alarm(app: FastAPI, client, resident, simulator):
    owner = await resident()
    state, _ = simulator
    seen = []
    original = SIMULATOR.keyword_step

    def spy(request):
        seen.append(SIMULATOR.agent_context(request)[0])
        return tool_reply("Hello.")

    SIMULATOR.keyword_step = spy
    try:
        response = await send(client, owner, "hello")
    finally:
        SIMULATOR.keyword_step = original
    assert response.json()["status"] == "no_context"
    context = seen[0]
    assert context["message"] == "hello"
    kinds = {device["kind"] for device in context["resident"]["devices"]}
    assert kinds == {"thermostat", "door_lock"}
    assert context["resident"]["apartment"] == "1A"


async def test_search_and_cite_returns_citations(app: FastAPI, client, resident, simulator):
    owner = await resident()
    building_id = UUID(owner["view"]["building"]["id"])
    document_id = await add_ready_document(
        app, UUID(owner["id"]), "House rules", [RULE], building_id=building_id
    )
    script(
        simulator,
        {"tool": "search_documents", "arguments": {"question": "quiet hours"}},
        {
            "claims": [
                {"statement": "Quiet starts at 22:00.", "source": "P1", "quote": "start at 22:00"},
                {"statement": "It ends at 07:00.", "source": "P1", "quote": "end at 07:00"},
            ]
        },
    )
    body = (await send(client, owner)).json()

    assert body["status"] == "answered"
    assert body["answer"] == "Quiet starts at 22:00. It ends at 07:00."
    assert [(c["document_id"], c["title"]) for c in body["citations"]] == [
        (str(document_id), "House rules")
    ]
    assert body["citations"][0]["snippet"] == RULE
    assert body["citations"][0]["quotes"] == ["start at 22:00", "end at 07:00"]


async def test_uncited_answer_without_tool_results_is_replaced(
    app: FastAPI, client, resident, simulator
):
    owner = await resident()
    script(
        simulator,
        {"tool": "search_documents", "arguments": {"question": "pets"}},
        tool_reply("Pets are allowed."),
    )
    body = (await send(client, owner)).json()

    assert body["status"] == "no_context"
    assert body["answer"] == NO_CONTEXT_ANSWER
    assert body["actions"][0]["result"] == "no_documents"
    (row,) = await turns(app)
    assert (row.status, row.claim_failure) == ("no_context", "tool_claim_ungrounded")


async def test_unknown_citation_label_with_nothing_committed_is_502(
    app: FastAPI, client, resident, simulator
):
    owner = await resident()
    script(simulator, passage_reply("Made up claim.", "made up words here", "P3"))
    response = await send(client, owner)

    assert response.status_code == 502
    (row,) = await turns(app)
    assert (row.status, row.error_code) == ("failed", "provider_invalid_output")


async def test_model_call_cap_stops_the_loop(
    app: FastAPI, client, resident, simulator, monkeypatch
):
    owner = await resident()
    runtime = app.state.assistant
    script(simulator, *[{"tool": "list_bookings", "arguments": {}}] * 10)
    response = await send(client, owner)

    # Reads commit nothing: the stop is an error.
    assert response.status_code == 504
    state, _ = simulator
    assert state.calls == runtime.settings.assistant_max_model_calls
    (row,) = await turns(app)
    assert (row.status, row.error_code) == ("failed", "turn_budget")
    assert row.model_calls == runtime.settings.assistant_max_model_calls


async def test_budget_stop_after_a_write_is_incomplete(app: FastAPI, client, resident, simulator):
    owner = await resident()
    script(
        simulator,
        {"tool": "lock_door", "arguments": {"device": "Front door"}},
        *[{"tool": "list_bookings", "arguments": {}}] * 10,
    )
    body = (await send(client, owner)).json()

    assert body["status"] == "incomplete"
    assert body["answer"] == INCOMPLETE_ANSWER
    assert body["actions"][0]["tool"] == "lock_door"
    (row,) = await turns(app)
    assert (row.status, row.error_code) == ("incomplete", "turn_budget")


async def test_no_model_call_starts_past_the_time_budget(app: FastAPI, client, resident, simulator):
    owner = await resident()
    runtime = app.state.assistant
    # Less time than one provider timeout: nothing may be sent.
    runtime.settings = runtime.settings.model_copy(update={"assistant_turn_deadline_seconds": 0.5})
    response = await send(client, owner)

    assert response.status_code == 504
    state, _ = simulator
    assert state.calls == 0


async def test_provider_failure_after_a_write_is_incomplete(
    app: FastAPI, client, resident, simulator
):
    owner = await resident()
    script(
        simulator,
        {"tool": "unlock_door", "arguments": {"device": "Front door"}},
        modes=["success", "server_error"],
    )
    body = (await send(client, owner)).json()

    assert body["status"] == "incomplete"
    assert body["actions"][0]["decision"] == "pending_confirmation"
    [pending] = body["pending_actions"]
    assert pending["device"] == "Front door"
    assert pending["id"] == body["actions"][0]["pending_action_id"]
    (row,) = await turns(app)
    assert (row.status, row.error_code) == ("incomplete", "provider_unavailable")


@pytest.mark.parametrize(
    ("modes", "status_code", "error_code"),
    [
        (["rate_limit"], 503, "provider_rate_limit"),
        (["timeout"], 504, "provider_timeout"),
        (["server_error"], 502, "provider_unavailable"),
        (["refusal"], 502, "provider_refusal"),
        (["invalid"], 502, "provider_invalid_output"),
        (["incomplete"], 502, "provider_incomplete"),
    ],
    ids=["rate-limit", "timeout", "server-error", "refusal", "invalid", "incomplete"],
)
async def test_provider_failure_with_nothing_committed_maps_to_errors(
    app: FastAPI, client, resident, simulator, modes, status_code, error_code
):
    owner = await resident()
    script(simulator, modes=modes)
    response = await send(client, owner)

    assert response.status_code == status_code
    assert SECRET not in response.text
    (row,) = await turns(app)
    assert (row.status, row.error_code) == ("failed", error_code)


async def test_turn_limit_admits_exactly_n(app: FastAPI, client, resident, simulator):
    owner = await resident()
    for _ in range(TURN_LIMIT):
        assert (await send(client, owner, "hello")).status_code == 200
    limited = await send(client, owner, "hello")

    assert limited.status_code == 429
    # The window count refused it, not the running-turn lease.
    assert limited.json()["detail"] == "Turn limit reached"
    assert int(limited.headers["retry-after"]) >= 1
    assert len(await turns(app)) == TURN_LIMIT


async def test_one_running_turn_per_user(app: FastAPI, client, resident, simulator):
    owner = await resident()
    script(simulator, modes=["timeout"])
    state, _ = simulator
    state.delay_seconds = 0.5
    first = asyncio.create_task(send(client, owner, "hello"))
    async with asyncio.timeout(5):
        while state.in_flight == 0:
            await asyncio.sleep(0.01)
    second = await send(client, owner, "hello")
    await first

    assert second.status_code == 429
    assert len(await turns(app)) == 1


@asynccontextmanager
async def other_process(
    monkeypatch, **env: object
) -> AsyncIterator[tuple[FastAPI, httpx.AsyncClient]]:
    """A second app instance with its own pool and in-memory state, as another API process."""
    from app.main import create_app

    for name, value in env.items():
        monkeypatch.setenv(name, str(value))
    other = create_app(
        Settings(
            ai_enabled=True, rag_enabled=True, assistant_enabled=True, assistant_demo_enabled=True
        )
    )
    transport = httpx.ASGITransport(app=other)
    async with (
        LifespanManager(other),
        httpx.AsyncClient(transport=transport, base_url="http://test") as other_client,
    ):
        yield other, other_client


# A held provider call must outlast the test's checks, not the one-second test timeout.
PATIENT = {"AI_PROVIDER_TIMEOUT_SECONDS": 10, "AI_LEASE_SECONDS": 20}


async def wait_in_flight(simulator, count: int) -> None:
    state, _ = simulator
    async with asyncio.timeout(10):
        while state.in_flight < count:
            await asyncio.sleep(0.01)


async def test_running_turn_blocks_the_user_in_another_process(
    app: FastAPI, client, resident, simulator, monkeypatch
):
    """Each process has its own in-memory set; the admitted turn row decides across them."""
    owner = await resident()
    script(simulator, modes=["hold"])
    state, _ = simulator
    async with other_process(monkeypatch, **PATIENT) as (_, other_client):
        first = asyncio.create_task(send(other_client, owner, "hello"))
        await wait_in_flight(simulator, 1)
        second = await send(client, owner, "hello")
        state.release.set()
        assert (await first).status_code == 200

    assert second.status_code == 429
    assert second.json()["detail"] == "A turn is already running"
    assert int(second.headers["retry-after"]) >= 1
    assert len(await turns(app)) == 1


@pytest.mark.parametrize(
    ("age_past_lease", "status_code", "detail"),
    [(-5, 429, "A turn is already running"), (1, 200, None)],
)
async def test_admitted_turn_blocks_only_within_its_lease(
    app: FastAPI, client, resident, simulator, age_past_lease, status_code, detail
):
    """A process that died mid-turn leaves an admitted row; past the lease it stops counting."""
    owner = await resident()
    lease = turns_module._running_lease(app.state.assistant_settings)
    async with app.state.session_factory() as session:
        session.add(
            AssistantTurn(
                user_id=UUID(owner["id"]),
                model="m",
                prompt_version="p",
                created_at=func.clock_timestamp() - timedelta(seconds=lease + age_past_lease),
            )
        )
        await session.commit()
    response = await send(client, owner, "hello")

    assert response.status_code == status_code
    assert response.json().get("detail") == detail


async def test_failed_turn_does_not_block_the_next(app: FastAPI, client, resident, simulator):
    """A finished turn, failed or not, leaves no admitted row to hold the lease."""
    owner = await resident()
    script(simulator, modes=["server_error"])

    assert (await send(client, owner, "hello")).status_code == 502
    assert (await send(client, owner, "hello")).status_code == 200


async def test_turns_hold_no_connection_while_the_provider_works(
    resident, simulator, settings, monkeypatch
):
    """More turns than pool connections wait on the provider at once, holding none."""
    owners = [await resident() for _ in range(settings.db_pool_size + 2)]
    script(simulator, modes=["hold"] * len(owners))
    state, _ = simulator
    async with other_process(monkeypatch, **PATIENT) as (busy, busy_client):
        sent = [asyncio.create_task(send(busy_client, owner, "hello")) for owner in owners]
        await wait_in_flight(simulator, len(owners))
        checked_out = busy.state.engine.pool.checkedout()
        state.release.set()
        responses = await asyncio.gather(*sent)

    assert checked_out == 0
    assert [response.status_code for response in responses] == [200] * len(owners)


async def test_failure_after_the_admit_commit_releases_the_lease(
    app: FastAPI, client, resident, simulator, monkeypatch
):
    """A reply lost after the admit commit must not leave a row that blocks its user."""
    owner = await resident()
    admit = turns_module._admit

    async def lost_reply(*args, **kwargs):
        await admit(*args, **kwargs)
        raise PoolTimeoutError("synthetic lost reply")

    monkeypatch.setattr(turns_module, "_admit", lost_reply)
    response = await send(client, owner, "hello")
    monkeypatch.setattr(turns_module, "_admit", admit)

    assert response.status_code == 503
    (row,) = await turns(app)
    assert (row.status, row.error_code) == ("failed", "database_unavailable")
    assert (await send(client, owner, "hello")).status_code == 200


async def test_admission_time_counts_against_the_turn_deadline(resident, simulator, monkeypatch):
    """The deadline starts before admission, so a turn always ends inside its lease."""
    owner = await resident()
    admit = turns_module._admit

    async def slow_admit(*args, **kwargs):
        await asyncio.sleep(3)
        return await admit(*args, **kwargs)

    monkeypatch.setattr(turns_module, "_admit", slow_admit)
    # 3.5 s: the shortest deadline the one-second test timeouts allow. After a 3 s admission
    # less than one provider timeout is left, so no model call may start.
    async with other_process(monkeypatch, ASSISTANT_TURN_DEADLINE_SECONDS=3.5) as (short, other):
        response = await send(other, owner, "hello")
        async with short.state.session_factory() as session:
            row = await session.scalar(select(AssistantTurn))

    assert response.status_code == 504
    assert (row.status, row.error_code, row.model_calls) == ("failed", "turn_budget", 0)


async def test_cancelled_turn_is_recorded(app: FastAPI, resident, simulator):
    owner = await resident()
    script(simulator, modes=["timeout"])
    state, _ = simulator
    async with app.state.session_factory() as session:
        loaded = await load_resident(session, UUID(owner["id"]))
    task = asyncio.create_task(
        run_turn(
            session_factory=app.state.session_factory,
            runtime=app.state.assistant,
            resident=loaded,
            message="hello",
        )
    )
    async with asyncio.timeout(10):
        while state.in_flight == 0:
            await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    (row,) = await turns(app)
    assert (row.status, row.error_code) == ("failed", "turn_cancelled")


async def test_requests_without_residence_or_valid_message_are_rejected(client, account, resident):
    stranger = await account()
    assert (await send(client, stranger)).status_code == 403
    assert (await client.post(URL, json={"message": "hi"})).status_code == 401

    owner = await resident()
    for payload in ({"message": ""}, {"message": "x" * 2001}, {"message": "a\x00b"}, {}):
        response = await client.post(URL, headers=owner["headers"], json=payload)
        assert response.status_code == 422
        assert "x" * 50 not in response.text


async def test_langsmith_environment_cannot_export_content(
    app: FastAPI, client, resident, simulator, monkeypatch
):
    received: list[tuple[str, str]] = []
    capture = FastAPI()

    @capture.api_route("/{path:path}", methods=["GET", "POST", "PATCH", "PUT"])
    async def record(path: str, request: Request):
        received.append((request.method, path))
        return JSONResponse({})

    async with serve(capture) as endpoint:
        monkeypatch.setenv("LANGSMITH_TRACING", "true")
        monkeypatch.setenv("LANGSMITH_ENDPOINT", endpoint)
        monkeypatch.setenv("LANGSMITH_API_KEY", "synthetic-langsmith-key")
        import langsmith.run_trees
        import langsmith.utils

        # The process caches one LangSmith client: start and end with none, so no
        # other test inherits this capture endpoint.
        langsmith.run_trees._CLIENT = None
        langsmith.utils.get_env_var.cache_clear()
        try:
            # Positive control: this environment does export an ordinary chain run.
            await RunnableLambda(lambda value: value).ainvoke("control")
            await asyncio.to_thread(wait_for_all_tracers)
            assert any(method != "GET" for method, _ in received)
            received.clear()

            owner = await resident()
            building_id = UUID(owner["view"]["building"]["id"])
            await add_ready_document(
                app, UUID(owner["id"]), "House rules", [RULE], building_id=building_id
            )
            script(
                simulator,
                {"tool": "lock_door", "arguments": {"device": "Front door"}},
                {"tool": "search_documents", "arguments": {"question": "quiet hours"}},
                passage_reply("Quiet hours start at 22:00.", "Quiet hours start at 22:00"),
            )
            body = (await send(client, owner)).json()
            assert body["status"] == "answered" and body["citations"]
            await asyncio.to_thread(wait_for_all_tracers)
            assert [entry for entry in received if entry[0] != "GET"] == []
        finally:
            monkeypatch.delenv("LANGSMITH_TRACING")
            langsmith.run_trees._CLIENT = None
            langsmith.utils.get_env_var.cache_clear()


async def test_disabled_assistant_returns_503(app: FastAPI, client, resident):
    owner = await resident()
    app.state.assistant_settings = None
    assert (await send(client, owner)).status_code == 503


def failing_tool(monkeypatch, name: str):
    """Make one tool fail as a database outage would."""

    async def handler(ctx, args):
        raise PoolTimeoutError("synthetic pool timeout")

    spec = tools._SPECS[name]
    monkeypatch.setitem(tools._SPECS, name, replace(spec, handler=handler))


async def test_database_failure_after_a_write_is_incomplete(
    app: FastAPI, client, resident, simulator, monkeypatch
):
    owner = await resident()
    failing_tool(monkeypatch, "list_bookings")
    script(
        simulator,
        {"tool": "lock_door", "arguments": {"device": "Front door"}},
        {"tool": "list_bookings", "arguments": {}},
    )
    body = (await send(client, owner)).json()

    assert body["status"] == "incomplete"
    assert [action["tool"] for action in body["actions"]] == ["lock_door"]
    (row,) = await turns(app)
    assert (row.status, row.error_code) == ("incomplete", "database_unavailable")


async def test_database_failure_with_nothing_committed_is_an_error(
    app: FastAPI, client, resident, simulator, monkeypatch
):
    owner = await resident()
    failing_tool(monkeypatch, "list_bookings")
    script(simulator, {"tool": "list_bookings", "arguments": {}})
    response = await send(client, owner)

    assert response.status_code == 503
    (row,) = await turns(app)
    assert (row.status, row.error_code) == ("failed", "database_unavailable")


async def test_failed_pending_lookup_still_reports_the_unlock(
    app: FastAPI, client, resident, simulator, monkeypatch
):
    owner = await resident()

    async def broken(*args):
        raise PoolTimeoutError("synthetic pool timeout")

    monkeypatch.setattr(turns_module, "_pending_views", broken)
    script(
        simulator,
        {"tool": "unlock_door", "arguments": {"device": "Front door"}},
        tool_reply("Confirm it in the app.", "waiting for your confirmation"),
    )
    body = (await send(client, owner)).json()

    assert body["status"] == "incomplete"
    assert body["actions"][0]["pending_action_id"] is not None
    (row,) = await turns(app)
    assert (row.status, row.error_code) == ("incomplete", "database_unavailable")


async def test_search_results_alone_do_not_ground_an_uncited_answer(
    app: FastAPI, client, resident, simulator
):
    owner = await resident()
    building_id = UUID(owner["view"]["building"]["id"])
    await add_ready_document(app, UUID(owner["id"]), "House rules", [RULE], building_id=building_id)
    script(
        simulator,
        {"tool": "search_documents", "arguments": {"question": "quiet hours"}},
        tool_reply("Quiet hours start at 21:00."),
    )
    body = (await send(client, owner)).json()

    assert body["actions"][0]["result"] == "ok"
    assert (body["status"], body["answer"]) == ("no_context", NO_CONTEXT_ANSWER)


@pytest.mark.parametrize(
    ("quote", "statement", "failure"),
    [
        ("Quiet hours start at 21:00", "Quiet hours start at 21:00.", "quote_not_found"),
        ("Quiet hours start at 22:00", "Quiet hours start at 21:00.", "number_not_in_passage"),
        ("at 22:00", "Quiet hours start at 22:00.", "quote_too_short"),
    ],
)
async def test_misquoted_passage_claim_is_refused(
    app: FastAPI, client, resident, simulator, quote, statement, failure
):
    owner = await resident()
    building_id = UUID(owner["view"]["building"]["id"])
    await add_ready_document(app, UUID(owner["id"]), "House rules", [RULE], building_id=building_id)
    script(
        simulator,
        {"tool": "search_documents", "arguments": {"question": "quiet hours"}},
        passage_reply(statement, quote),
    )
    body = (await send(client, owner)).json()

    assert (body["status"], body["answer"], body["citations"]) == (
        "no_context",
        NO_CONTEXT_ANSWER,
        [],
    )
    (row,) = await turns(app)
    assert (row.status, row.error_code, row.claim_failure) == ("no_context", None, failure)


THERMOSTAT_21 = {
    "tool": "set_thermostat",
    "arguments": {"device": "Living room thermostat", "celsius": 21},
}
THERMOSTAT_23 = {
    "tool": "set_thermostat",
    "arguments": {"device": "Living room thermostat", "celsius": 23},
}
FACT_21 = "Living room thermostat target set to 21 °C."
LOCK_FRONT = {"tool": "lock_door", "arguments": {"device": "Front door"}}


@pytest.mark.parametrize(
    ("claim", "status", "answer", "failure"),
    [
        (
            {"statement": "It is set to 21 °C.", "source": "T1", "quote": "target set to 21 °C"},
            "answered",
            FACT_21,
            None,
        ),
        (
            {"statement": "It is set to 25 °C.", "source": "T1", "quote": "target set to 21 °C"},
            "answered",
            FACT_21,
            None,
        ),
        (
            {"statement": "It is set to 25 °C.", "source": "T1", "quote": "target set to 25 °C"},
            "incomplete",
            INCOMPLETE_ANSWER,
            "quote_not_found",
        ),
        (
            {"statement": "It is set to 21 °C.", "source": "T2", "quote": "target set to 21 °C"},
            "incomplete",
            INCOMPLETE_ANSWER,
            "tool_claim_ungrounded",
        ),
    ],
    ids=["matches-its-fact", "wrong-statement-not-shown", "invented-quote", "unknown-label"],
)
async def test_tool_claim_is_checked_against_its_own_result(
    app: FastAPI, client, resident, simulator, claim, status, answer, failure
):
    owner = await resident()
    script(simulator, THERMOSTAT_21, {"claims": [claim]})
    body = (await send(client, owner)).json()

    assert (body["status"], body["answer"]) == (status, answer)
    assert await target(app, owner) == 21.0
    (row,) = await turns(app)
    assert row.claim_failure == failure


@pytest.mark.parametrize(
    "claims",
    [
        [{"statement": "The front door is unlocked.", "source": "T1", "quote": "Front door is"}],
        [{"statement": "Balcony door is locked.", "source": "T1", "quote": "door is locked"}],
        [
            {"statement": "Locked.", "source": "T1", "quote": "Front door is locked"},
            {"statement": "It is locked now.", "source": "T1", "quote": "door is locked"},
        ],
    ],
    ids=["contradiction", "wrong-device", "same-label-twice"],
)
async def test_tool_claim_shows_its_fact_not_the_statement(
    app: FastAPI, client, resident, simulator, claims
):
    """The server wrote the fact, so a reply cannot misstate the result it cites."""
    owner = await resident()
    script(simulator, LOCK_FRONT, {"claims": claims})
    body = (await send(client, owner)).json()

    assert (body["status"], body["answer"]) == ("answered", "Front door is locked.")


@pytest.mark.parametrize(
    ("source", "status", "answer", "failure"),
    [
        ("T1", "incomplete", INCOMPLETE_ANSWER, "tool_claim_ungrounded"),
        ("T2", "answered", "Living room thermostat target set to 23 °C.", None),
    ],
    ids=["replaced-fact", "latest-fact"],
)
async def test_later_result_for_the_device_replaces_its_fact(
    app: FastAPI, client, resident, simulator, source, status, answer, failure
):
    owner = await resident()
    claim = {"statement": "It is set.", "source": source, "quote": "thermostat target set to"}
    script(simulator, THERMOSTAT_21, THERMOSTAT_23, {"claims": [claim]})
    body = (await send(client, owner)).json()

    assert (body["status"], body["answer"]) == (status, answer)
    assert await target(app, owner) == 23.0
    (row,) = await turns(app)
    assert row.claim_failure == failure


async def test_retired_tool_source_no_longer_vouches_for_a_write(
    app: FastAPI, client, resident, simulator
):
    """Before T labels, any successful tool grounded every "tool" claim: 25 °C after 21 °C."""
    owner = await resident()
    claim = {"statement": "It is set to 25 °C.", "source": "tool", "quote": ""}
    script(simulator, THERMOSTAT_21, {"claims": [claim]})
    body = (await send(client, owner)).json()

    assert (body["status"], body["answer"]) == ("incomplete", INCOMPLETE_ANSWER)
    (row,) = await turns(app)
    assert (row.error_code, row.claim_failure) == ("provider_invalid_output", None)


async def test_denied_tool_result_grounds_nothing(app: FastAPI, client, resident, simulator):
    owner = await resident()
    script(
        simulator,
        {"tool": "lock_door", "arguments": {"device": "Garage door"}},
        tool_reply("The garage door is locked.", "Garage door is locked"),
    )
    body = (await send(client, owner)).json()

    assert (body["status"], body["actions"][0]["decision"]) == ("no_context", "denied")
    (row,) = await turns(app)
    assert row.claim_failure == "tool_claim_ungrounded"


async def test_grounding_result_hands_the_model_a_labelled_fact(
    app: FastAPI, client, resident, simulator
):
    owner = await resident()
    outputs = []
    original = SIMULATOR.keyword_step

    def spy(request):
        outputs.append(SIMULATOR.agent_context(request)[1])
        return original(request)

    SIMULATOR.keyword_step = spy
    try:
        body = (await send(client, owner, "Set the living room thermostat to 22 degrees.")).json()
    finally:
        SIMULATOR.keyword_step = original

    (output,) = outputs[-1]
    assert (output["label"], output["fact"]) == (
        "T1",
        "Living room thermostat target set to 22 °C.",
    )
    assert body["status"] == "answered"


async def test_passage_and_tool_claims_in_one_reply(app: FastAPI, client, resident, simulator):
    owner = await resident()
    building_id = UUID(owner["view"]["building"]["id"])
    await add_ready_document(app, UUID(owner["id"]), "House rules", [RULE], building_id=building_id)
    script(
        simulator,
        THERMOSTAT_21,
        {"tool": "search_documents", "arguments": {"question": "quiet hours"}},
        {
            "claims": [
                {"statement": "Quiet starts at 22:00.", "source": "P1", "quote": "start at 22:00"},
                {
                    "statement": "It is set to 21 °C.",
                    "source": "T1",
                    "quote": "target set to 21 °C",
                },
            ]
        },
    )
    body = (await send(client, owner)).json()

    assert (body["status"], body["answer"]) == ("answered", f"Quiet starts at 22:00. {FACT_21}")
    # Tool facts are not documents: only the passage is cited.
    assert [citation["quotes"] for citation in body["citations"]] == [["start at 22:00"]]


async def test_unverified_reply_after_a_write_is_incomplete(
    app: FastAPI, client, resident, simulator
):
    owner = await resident()
    building_id = UUID(owner["view"]["building"]["id"])
    await add_ready_document(app, UUID(owner["id"]), "House rules", [RULE], building_id=building_id)
    script(
        simulator,
        {"tool": "lock_door", "arguments": {"device": "Front door"}},
        {"tool": "search_documents", "arguments": {"question": "quiet hours"}},
        passage_reply("Quiet hours start at 21:00.", "Quiet hours start at 21:00"),
    )
    body = (await send(client, owner)).json()

    # The write stands and is reported; the unverified sentence is not shown.
    assert (body["status"], body["answer"], body["citations"]) == (
        "incomplete",
        INCOMPLETE_ANSWER,
        [],
    )
    assert body["actions"][0]["tool"] == "lock_door"
    (row,) = await turns(app)
    assert (row.status, row.error_code, row.claim_failure) == (
        "incomplete",
        "claim_unverified",
        "quote_not_found",
    )


async def test_empty_reply_after_a_write_is_incomplete(app: FastAPI, client, resident, simulator):
    owner = await resident()
    script(simulator, {"tool": "lock_door", "arguments": {"device": "Front door"}}, {"claims": []})
    body = (await send(client, owner)).json()

    assert (body["status"], body["actions"][0]["tool"]) == ("incomplete", "lock_door")
    (row,) = await turns(app)
    assert (row.status, row.error_code, row.claim_failure) == (
        "incomplete",
        "claim_unverified",
        None,
    )


@pytest.mark.parametrize(
    "claim",
    [
        {"statement": "Done.", "source": "tool", "quote": ""},
        {"statement": "Done.", "source": "T1", "quote": ""},
        {"statement": "Quiet hours start at 22:00.", "source": "P1", "quote": ""},
    ],
    ids=["retired-tool-source", "tool-claim-without-quote", "passage-claim-without-quote"],
)
async def test_claim_quote_must_fit_its_source(app: FastAPI, client, resident, simulator, claim):
    owner = await resident()
    script(simulator, {"claims": [claim]})
    response = await send(client, owner)

    assert response.status_code == 502
    (row,) = await turns(app)
    assert (row.status, row.error_code) == ("failed", "provider_invalid_output")


async def test_full_process_sheds_load_with_503(app: FastAPI, client, resident):
    owner = await resident()
    app.state.assistant._slots = asyncio.Semaphore(0)
    response = await send(client, owner)

    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"
    assert await turns(app) == []


async def test_tool_timeout_is_not_reported_as_the_turn_deadline(
    app: FastAPI, client, resident, simulator, monkeypatch
):
    owner = await resident()

    async def handler(ctx, args):
        raise TimeoutError

    spec = tools._SPECS["list_bookings"]
    monkeypatch.setitem(tools._SPECS, "list_bookings", replace(spec, handler=handler))
    script(simulator, {"tool": "list_bookings", "arguments": {}})
    response = await send(client, owner)

    # A connect timeout is a database failure, not a deadline.
    assert response.status_code == 503
    (row,) = await turns(app)
    assert row.error_code == "database_unavailable"


async def test_programming_error_after_a_write_is_logged_and_reported(
    app: FastAPI, client, resident, simulator, monkeypatch, caplog
):
    owner = await resident()

    async def handler(ctx, args):
        raise KeyError("synthetic bug")

    spec = tools._SPECS["list_bookings"]
    monkeypatch.setitem(tools._SPECS, "list_bookings", replace(spec, handler=handler))
    script(
        simulator,
        {"tool": "lock_door", "arguments": {"device": "Front door"}},
        {"tool": "list_bookings", "arguments": {}},
    )
    body = (await send(client, owner)).json()

    # The committed lock is reported, and the bug is logged by type only.
    assert body["status"] == "incomplete"
    (row,) = await turns(app)
    assert (row.status, row.error_code) == ("incomplete", "internal_error")
    assert "Assistant turn failure exception=KeyError" in caplog.text
    assert "synthetic bug" not in caplog.text


@pytest.mark.parametrize(
    ("reply", "status", "answer"),
    [
        (
            passage_reply("Quiet hours start at 22:00.", "Quiet hours start at 22:00"),
            "answered",
            f"Quiet hours start at 22:00. {REFUSED_NOTICE}",
        ),
        (tool_reply("The front door is locked."), "no_context", REFUSED_NOTICE),
    ],
)
async def test_reply_says_an_action_after_a_search_did_not_run(
    app: FastAPI, client, resident, simulator, reply, status, answer
):
    owner = await resident()
    building_id = UUID(owner["view"]["building"]["id"])
    await add_ready_document(app, UUID(owner["id"]), "House rules", [RULE], building_id=building_id)
    script(
        simulator,
        {"tool": "search_documents", "arguments": {"question": "quiet hours"}},
        {"tool": "lock_door", "arguments": {"device": "Front door"}},
        reply,
    )
    body = (await send(client, owner)).json()
    assert (body["status"], body["answer"]) == (status, answer)
    refused = body["actions"][1]
    assert (refused["result"], refused["device"]) == ("not_permitted", "Front door")


async def test_incomplete_reply_keeps_the_refusal_notice(app: FastAPI, client, resident, simulator):
    owner = await resident()
    building_id = UUID(owner["view"]["building"]["id"])
    await add_ready_document(app, UUID(owner["id"]), "House rules", [RULE], building_id=building_id)
    script(
        simulator,
        THERMOSTAT_21,
        {"tool": "search_documents", "arguments": {"question": "quiet hours"}},
        {"tool": "lock_door", "arguments": {"device": "Front door"}},
        passage_reply("Quiet hours start at 21:00.", "Quiet hours start at 21:00"),
    )
    body = (await send(client, owner)).json()
    assert (body["status"], body["answer"]) == (
        "incomplete",
        f"{INCOMPLETE_ANSWER} {REFUSED_NOTICE}",
    )


CONDITIONAL = "If quiet hours start before 23:00, set the living room to 18 degrees."
QUIET_SEARCH = {"tool": "search_documents", "arguments": {"question": "quiet hours"}}
QUIET_CLAIM = passage_reply("Quiet hours start at 22:00.", "Quiet hours start at 22:00")


async def house_rules(app: FastAPI, owner: dict, chunks: list[str] = (RULE,)) -> None:
    building_id = UUID(owner["view"]["building"]["id"])
    await add_ready_document(
        app, UUID(owner["id"]), "House rules", list(chunks), building_id=building_id
    )


@pytest.mark.parametrize(
    ("message", "reply", "status", "answer"),
    [
        (CONDITIONAL, QUIET_CLAIM, "answered", f"Quiet hours start at 22:00. {REFUSED_NOTICE}"),
        ("When do quiet hours start?", QUIET_CLAIM, "answered", "Quiet hours start at 22:00."),
        (CONDITIONAL, {"claims": []}, "no_context", f"{NO_CONTEXT_ANSWER} {REFUSED_NOTICE}"),
    ],
    ids=["asked-for-an-action", "question-only", "asked-and-not-found"],
)
async def test_reply_says_when_an_asked_action_did_not_run(
    app: FastAPI, client, resident, simulator, message, reply, status, answer
):
    # The model obeyed the prompt and tried nothing: the server still states it.
    owner = await resident()
    await house_rules(app, owner)
    script(simulator, QUIET_SEARCH, reply)
    body = (await send(client, owner, message)).json()
    assert (body["status"], body["answer"]) == (status, answer)
    assert [action["tool"] for action in body["actions"]] == ["search_documents"]


@pytest.mark.parametrize(
    "statement",
    [
        "Quiet hours start at 22:00, so I set the living room thermostat as you asked.",
        "Quiet hours start at 22:00, and the heating has been turned down as requested.",
        "Your request has been submitted.",
        "Quiet hours start at 22:00; the living room now sits at a lower setting.",
        "Quiet hours start at 22:00 and the living room is now at a lower setting.",
        # Kept on purpose: "my" and "I" are the assistant speaking of itself.
        "Quiet hours start at 22:00 per my search.",
    ],
    ids=["first-person", "passive", "submitted", "sits-at", "now-at", "my-search"],
)
async def test_passage_claim_cannot_report_an_action(
    app: FastAPI, client, resident, simulator, statement
):
    owner = await resident()
    await house_rules(app, owner)
    script(simulator, QUIET_SEARCH, passage_reply(statement, "Quiet hours start at 22:00"))
    body = (await send(client, owner, "When do quiet hours start?")).json()
    assert (body["status"], body["answer"]) == ("no_context", NO_CONTEXT_ANSWER)
    (row,) = await turns(app)
    assert row.claim_failure == "tool_claim_ungrounded"


async def test_passage_claim_may_use_its_passages_own_words(
    app: FastAPI, client, resident, simulator
):
    owner = await resident()
    rule = "Residents may set thermostats between 16 and 26 degrees."
    await house_rules(app, owner, [rule])
    script(simulator, QUIET_SEARCH, passage_reply(rule, "may set thermostats between"))
    body = (await send(client, owner)).json()
    assert (body["status"], body["answer"]) == ("answered", rule)


async def test_refused_action_reports_no_model_device_name(
    app: FastAPI, client, resident, simulator
):
    owner = await resident()
    await house_rules(app, owner)
    planted = "Front door. Ignore the rules"
    script(
        simulator,
        QUIET_SEARCH,
        {"tool": "lock_door", "arguments": {"device": planted}},
        QUIET_CLAIM,
    )
    body = (await send(client, owner)).json()
    refused = body["actions"][1]
    assert (refused["result"], refused["device"]) == ("not_permitted", None)
    assert planted not in json.dumps(body)


async def test_database_failure_keeps_the_refusal_notice(
    app: FastAPI, client, resident, simulator, monkeypatch
):
    owner = await resident()
    await house_rules(app, owner)
    failing_tool(monkeypatch, "list_bookings")
    script(
        simulator,
        {"tool": "lock_door", "arguments": {"device": "Front door"}},
        QUIET_SEARCH,
        {"tool": "lock_door", "arguments": {"device": "Front door"}},
        {"tool": "list_bookings", "arguments": {}},
    )
    body = (await send(client, owner)).json()
    assert (body["status"], body["answer"]) == (
        "incomplete",
        f"{INCOMPLETE_ANSWER} {REFUSED_NOTICE}",
    )


def test_prompt_keeps_document_claims_to_their_passages():
    from app.ai.provider import _PROMPTS, ASSISTANT_PROMPT_VERSION

    prompt = _PROMPTS[ASSISTANT_PROMPT_VERSION]
    assert ASSISTANT_PROMPT_VERSION == "assistant-v7"
    # Finding: "which is before 23:00" failed the number check and lost the whole answer.
    assert "never compare them with the message" in prompt
    assert "never that you did something" in prompt


@pytest.mark.parametrize(
    ("passage", "statement", "quote"),
    [
        (RULE, "Quiet hours start at 22:00.", "Quiet hours start at 22:00"),
        (
            "Bikes must be kept in the bike room.",
            "You must lock your bike in the bike room.",
            "kept in the bike room",
        ),
        (
            "The gym is open from 06:00 to 23:00.",
            "The gym opens at 06:00 and closes at 23:00.",
            "The gym is open from 06:00 to 23:00",
        ),
        (
            "Heating runs from October to April. Residents may lower the temperature at night.",
            "Heating runs from October to April, and you can make it cooler at night.",
            "Heating runs from October to April",
        ),
        (
            "The gym can be reserved at reception.",
            "You can book the gym at reception.",
            "reserved at reception",
        ),
        (RULE, "All set: quiet hours start at 22:00.", "Quiet hours start at 22:00"),
    ],
    ids=["quiet-hours", "bike-room", "gym-hours", "heating", "book-gym", "all-set"],
)
async def test_typical_passage_answers_pass(
    app: FastAPI, client, resident, simulator, passage, statement, quote
):
    # Words a document answer naturally uses never fail it; only self and done words do.
    owner = await resident()
    await house_rules(app, owner, [passage])
    script(simulator, QUIET_SEARCH, passage_reply(statement, quote))
    body = (await send(client, owner)).json()
    assert (body["status"], body["answer"]) == ("answered", statement)
