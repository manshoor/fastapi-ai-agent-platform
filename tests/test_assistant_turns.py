"""Assistant turns: the agent loop, budgets, grounding, failures, limits, and privacy."""

import asyncio
import json
from dataclasses import replace
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_core.runnables import RunnableLambda
from langchain_core.tracers.langchain import wait_for_all_tracers
from sqlalchemy import select
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

import app.assistant.tools as tools
import app.assistant.turns as turns_module
from app.assistant.residence import load_resident
from app.assistant.turns import INCOMPLETE_ANSWER, NO_CONTEXT_ANSWER, run_turn
from app.config import Settings
from app.models import AssistantToolCall, AssistantTurn, Device
from tests.support.assistant import SIMULATOR, add_ready_document, demo_residence, serve

URL = "/api/v1/assistant/messages"
TURN_LIMIT = 3
SECRET = "private-message-7f3a"
RULE = "Quiet hours start at 22:00 and end at 07:00."


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
        {"answer": f"Set to 23 degrees {SECRET}", "citations": []},
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
    assert row.model == "simulator-assistant" and row.finished_at is not None
    async with app.state.session_factory() as session:
        stored = [
            {column: str(getattr(item, column)) for column in item.__table__.columns.keys()}
            for item in [*(await turns(app)), *await session.scalars(select(AssistantToolCall))]
        ]
    assert SECRET not in json.dumps(stored)


async def test_the_model_sees_context_without_the_alarm(app: FastAPI, client, resident, simulator):
    owner = await resident()
    state, _ = simulator
    seen = []
    original = SIMULATOR.keyword_step

    def spy(request):
        seen.append(SIMULATOR.agent_context(request)[0])
        return {"answer": "Hello.", "citations": []}

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
        {"answer": "Quiet hours start at 22:00.", "citations": ["P1", "P1"]},
    )
    body = (await send(client, owner)).json()

    assert body["status"] == "answered"
    assert [(c["document_id"], c["title"]) for c in body["citations"]] == [
        (str(document_id), "House rules")
    ]
    assert body["citations"][0]["snippet"] == RULE


async def test_uncited_answer_without_tool_results_is_replaced(client, resident, simulator):
    owner = await resident()
    script(
        simulator,
        {"tool": "search_documents", "arguments": {"question": "pets"}},
        {"answer": "Pets are allowed.", "citations": []},
    )
    body = (await send(client, owner)).json()

    assert body["status"] == "no_context"
    assert body["answer"] == NO_CONTEXT_ANSWER
    assert body["actions"][0]["result"] == "no_documents"


async def test_unknown_citation_label_with_nothing_committed_is_502(
    app: FastAPI, client, resident, simulator
):
    owner = await resident()
    script(simulator, {"answer": "Made up.", "citations": ["P3"]})
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
        import langsmith.utils

        langsmith.utils.get_env_var.cache_clear()
        try:
            # Positive control: this environment does export an ordinary chain run.
            await RunnableLambda(lambda value: value).ainvoke("control")
            await asyncio.to_thread(wait_for_all_tracers)
            assert any(method != "GET" for method, _ in received)
            received.clear()

            owner = await resident()
            script(
                simulator,
                {"tool": "lock_door", "arguments": {"device": "Front door"}},
                {"answer": "Locked.", "citations": []},
            )
            assert (await send(client, owner)).json()["status"] == "answered"
            await asyncio.to_thread(wait_for_all_tracers)
            assert [entry for entry in received if entry[0] != "GET"] == []
        finally:
            monkeypatch.delenv("LANGSMITH_TRACING")
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
        {"answer": "Confirm it in the app.", "citations": []},
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
        {"answer": "Quiet hours start at 21:00.", "citations": []},
    )
    body = (await send(client, owner)).json()

    assert body["actions"][0]["result"] == "ok"
    assert (body["status"], body["answer"]) == ("no_context", NO_CONTEXT_ANSWER)


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
