"""Agent-step and judge adapters over real TCP with the installed SDK and the simulator."""

import asyncio
import json
import socket
from contextlib import asynccontextmanager

import httpx
import pytest
import uvicorn
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import SecretStr

from app.ai.config import AISettings
from app.ai.provider import (
    ASSISTANT_PROMPT_VERSION,
    OpenAIAgentModel,
    OpenAIJudge,
    OpenAIRewriter,
    ProviderFailure,
)
from tests.support.assistant import SIMULATOR

TOOLS = [
    {
        "type": "function",
        "name": "lock_door",
        "description": "Lock a door.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"device": {"type": "string"}},
            "required": ["device"],
            "additionalProperties": False,
        },
    }
]
CONTEXT = {
    "resident": {"devices": [{"name": "Front door", "kind": "door_lock", "locked": False}]},
    "message": "Lock the front door",
}
INPUT = [{"role": "user", "content": json.dumps(CONTEXT)}]


@asynccontextmanager
async def running(app):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    server = uvicorn.Server(
        uvicorn.Config(app, access_log=False, log_level="critical", lifespan="off")
    )
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(3):
            while not server.started:
                await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 3)
        sock.close()


@pytest.fixture
async def simulator():
    app = SIMULATOR.create_app()
    captured = []

    # Capture request bodies only inside this test process to assert the wire contract.
    @app.middleware("http")
    async def capture(request, call_next):
        if request.url.path == "/v1/responses":
            captured.append(json.loads(await request.body()))
        return await call_next(request)

    async with running(app) as url:
        yield url, app, captured


def settings_for(url: str) -> AISettings:
    return AISettings(
        openai_api_key=SecretStr("synthetic-key-never-external"),
        openai_base_url=url + "/v1",
        ai_provider_timeout_seconds=1,
    )


async def control(url: str, **body) -> None:
    async with httpx.AsyncClient(base_url=url, trust_env=False) as client:
        response = await client.post("/__control", json=body)
        assert response.status_code == 200


def replace_route(app, path: str, handler) -> None:
    original = next(route for route in app.routes if getattr(route, "path", "") == path)
    app.routes.remove(original)
    app.post(path)(handler)


async def step(url: str):
    model = OpenAIAgentModel(settings_for(url))
    try:
        return await model.step(INPUT, TOOLS)
    finally:
        await model.aclose()


async def test_tool_call_wire_contract(simulator):
    url, _, captured = simulator
    result = await step(url)
    assert result.reply is None
    assert result.call.name == "lock_door"
    assert json.loads(result.call.arguments) == {"device": "Front door"}
    assert result.call.call_id.startswith("call_sim_")
    assert result.model == "simulator-assistant" and result.input_tokens == 40
    (request,) = captured
    assert request["tools"] == TOOLS
    assert request["parallel_tool_calls"] is False
    assert request["store"] is False and request["tool_choice"] == "auto"
    assert request["reasoning"] == {"effort": "none"}
    schema = request["text"]["format"]["schema"]
    assert set(schema["required"]) == {"claims"}
    assert set(schema["$defs"]["AssistantClaim"]["required"]) == {"statement", "source", "quote"}


async def test_scripted_reply_and_unknown_tool_pass_through(simulator):
    url, _, _ = simulator
    await control(
        url,
        script=[
            {"tool": "disarm_alarm", "arguments": {}},
            {"tool": "lock_door", "raw_arguments": "{not json"},
            {
                "claims": [
                    {"statement": "The door is locked.", "source": "tool", "quote": ""},
                    {"statement": "Doors lock at night.", "source": "P1", "quote": "doors lock at"},
                ]
            },
        ],
    )
    # Unknown names and malformed arguments are the executor's to deny and audit.
    assert (await step(url)).call.name == "disarm_alarm"
    assert (await step(url)).call.arguments == "{not json"
    reply = (await step(url)).reply
    assert [(claim.source, claim.quote) for claim in reply.claims] == [
        ("tool", ""),
        ("P1", "doors lock at"),
    ]


def envelope(*items, status="completed") -> dict:
    return {
        "id": "resp_x",
        "object": "response",
        "created_at": 0,
        "status": status,
        "model": "simulator-assistant",
        "error": None,
        "incomplete_details": None,
        "output": list(items),
        "usage": None,
        "parallel_tool_calls": False,
        "tools": [],
        "tool_choice": "auto",
        "store": False,
    }


def call(**overrides) -> dict:
    item = {
        "id": "fc_x",
        "type": "function_call",
        "status": "completed",
        "call_id": "call_x",
        "name": "lock_door",
        "arguments": '{"device": "Front door"}',
    }
    return {**item, **overrides}


REASONING = {"id": "rs_x", "type": "reasoning", "summary": []}


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (envelope(call(), call(call_id="call_y")), "provider_invalid_output"),
        (envelope(REASONING, call()), "provider_invalid_output"),
        (envelope(REASONING), "provider_invalid_output"),
        (envelope(), "provider_invalid_output"),
        (envelope(call(status="in_progress")), "provider_invalid_output"),
        (envelope(call(name="Disarm Alarm!")), "provider_invalid_output"),
        (envelope(call(call_id="bad id with spaces")), "provider_invalid_output"),
        (envelope(call(arguments="x" * 9000)), "provider_invalid_output"),
        (envelope(call(), status="incomplete"), "provider_incomplete"),
        (envelope(call(), status="failed"), "provider_error"),
    ],
    ids=[
        "two-calls",
        "reasoning-and-call",
        "reasoning-only",
        "empty",
        "unfinished-call",
        "unsafe-name",
        "unsafe-call-id",
        "huge-arguments",
        "incomplete",
        "failed",
    ],
)
async def test_invalid_agent_output_maps_to_one_code(simulator, body, code):
    url, app, _ = simulator
    calls = []

    async def malformed(request: Request):
        calls.append(1)
        return JSONResponse(body)

    replace_route(app, "/v1/responses", malformed)
    with pytest.raises(ProviderFailure) as error:
        await step(url)
    assert error.value.code == code
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("mode", "code"),
    [
        ("refusal", "provider_refusal"),
        ("invalid", "provider_invalid_output"),
        ("rate_limit", "provider_rate_limit"),
        ("server_error", "provider_unavailable"),
    ],
)
async def test_simulated_failures_are_classified(simulator, mode, code):
    url, _, _ = simulator
    await control(url, modes=[mode])
    with pytest.raises(ProviderFailure) as error:
        await step(url)
    assert error.value.code == code


async def test_judge_returns_a_verdict(simulator):
    url, _, captured = simulator
    judge = OpenAIJudge(settings_for(url))
    try:
        verdict = await judge.judge("When are quiet hours?", "From 22:00.", ["Quiet from 22:00."])
    finally:
        await judge.aclose()
    assert verdict.grounded and verdict.answers_question
    assert captured[0]["tools"] == [] and captured[0]["store"] is False


async def test_rewriter_refuses_the_assistant_prompt(simulator):
    url, _, captured = simulator
    rewriter = OpenAIRewriter(settings_for(url))
    try:
        with pytest.raises(ProviderFailure) as error:
            await rewriter.rewrite("t", "d", "i", prompt_version=ASSISTANT_PROMPT_VERSION)
    finally:
        await rewriter.aclose()
    assert error.value.code == "provider_prompt_version"
    assert captured == []


def test_keyword_policy_picks_named_devices():
    devices = [
        {"name": "Living room thermostat", "kind": "thermostat"},
        {"name": "Bedroom thermostat", "kind": "thermostat"},
        {"name": "Front door", "kind": "door_lock"},
    ]

    def first_step(message: str) -> dict:
        context = {"resident": {"devices": devices}, "message": message}
        return SIMULATOR.keyword_step({"input": [{"content": json.dumps(context)}]})

    assert first_step("Set the bedroom thermostat to 20.5 degrees") == {
        "tool": "set_thermostat",
        "arguments": {"device": "Bedroom thermostat", "celsius": 20.5},
    }
    assert first_step("Please unlock the front door")["tool"] == "unlock_door"
    assert first_step("When are quiet hours?")["tool"] == "search_documents"
