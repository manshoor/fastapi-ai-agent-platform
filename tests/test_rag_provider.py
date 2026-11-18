"""Embedding and quoted-answer adapters over real TCP with the installed SDK."""

import asyncio
import importlib.util
import json
import math
import socket
import sys
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi import Request
from fastapi.responses import JSONResponse, Response
from pydantic import SecretStr

from app.ai.config import AISettings
from app.ai.provider import (
    RAG_PROMPT_VERSION,
    OpenAIAnswerer,
    OpenAIEmbedder,
    OpenAIReranker,
    OpenAIRewriter,
    Passage,
    ProviderFailure,
)

DIMENSIONS = 1536
PASSAGES = [
    Passage(title="Pump manual", page=3, text="The pump runs at 40 litres per minute."),
    Passage(title="Pump manual", page=4, text="Clean the filter every month."),
]


def load_simulator():
    path = Path(__file__).parent / "support/provider_simulator.py"
    spec = importlib.util.spec_from_file_location("provider_simulator_rag", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["provider_simulator_rag"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
async def simulator():
    app = load_simulator().create_app()
    # Capture request bodies only inside this test process to assert the wire contract.
    captured = []

    @app.middleware("http")
    async def capture(request, call_next):
        if request.url.path in {"/v1/responses", "/v1/embeddings"}:
            captured.append((request.url.path, json.loads(await request.body())))
        return await call_next(request)

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
                if task.done():
                    await task
                await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{sock.getsockname()[1]}", app, captured
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 3)
        sock.close()


def settings_for(base_url: str, **updates) -> AISettings:
    return AISettings(
        openai_api_key=SecretStr("synthetic-key-never-external"),
        openai_base_url=base_url + "/v1",
        ai_provider_timeout_seconds=1,
        ai_embedding_timeout_seconds=1,
    ).model_copy(update=updates)


async def configure(base_url: str, mode: str, delay: float = 0.4) -> None:
    async with httpx.AsyncClient(base_url=base_url, trust_env=False) as client:
        response = await client.post("/__control", json={"modes": [mode], "delay_seconds": delay})
        assert response.status_code == 200


def replace_route(app, path: str, handler) -> None:
    # Swap only the HTTP route: the SDK still parses a real wire response.
    original = next(route for route in app.routes if getattr(route, "path", "") == path)
    app.routes.remove(original)
    app.post(path)(handler)


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    return dot / (math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right)))


async def test_embedding_wire_contract_order_and_similarity(simulator):
    url, app, captured = simulator
    embedder = OpenAIEmbedder(settings_for(url, ai_embedding_model="pinned-embedding"))
    texts = ["pump flow rate litres", "pump flow rate", "holiday calendar policy"]
    try:
        result = await embedder.embed(texts)
        again = await embedder.embed([texts[0]])
    finally:
        await embedder.aclose()
    path, wire = captured[0]
    assert path == "/v1/embeddings"
    assert wire == {
        "model": "pinned-embedding",
        "input": texts,
        "dimensions": DIMENSIONS,
        "encoding_format": "float",
    }
    assert app.state.simulator.calls == 2
    assert result.model == "simulator-embedding"
    assert result.input_tokens == 10 and result.request_id.startswith("req_sim_")
    assert [len(vector) for vector in result.vectors] == [DIMENSIONS] * 3
    assert again.vectors[0] == result.vectors[0]
    assert cosine(result.vectors[0], result.vectors[1]) > cosine(
        result.vectors[0], result.vectors[2]
    )


@pytest.mark.parametrize(
    ("mode", "code", "ambiguous", "retryable"),
    [
        ("rate_limit", "provider_rate_limit", False, True),
        ("server_error", "provider_unavailable", True, False),
        ("timeout", "provider_timeout", True, False),
        ("invalid", "provider_invalid_output", False, False),
    ],
)
async def test_embedding_failures_are_classified_once(simulator, mode, code, ambiguous, retryable):
    url, app, captured = simulator
    await configure(url, mode)
    embedder = OpenAIEmbedder(
        settings_for(url, ai_embedding_timeout_seconds=0.1 if mode == "timeout" else 1)
    )
    try:
        with pytest.raises(ProviderFailure) as error:
            await embedder.embed(["one", "two"])
    finally:
        await embedder.aclose()
    assert (error.value.code, error.value.ambiguous, error.value.retryable) == (
        code,
        ambiguous,
        retryable,
    )
    assert error.value.__cause__ is None
    assert app.state.simulator.calls == 1 and len(captured) == 1


def vector_body(**overrides) -> dict:
    body = {
        "object": "list",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.5] * DIMENSIONS},
            {"object": "embedding", "index": 1, "embedding": [0.25] * DIMENSIONS},
        ],
        "model": "synthetic-embedding",
        "usage": {"prompt_tokens": 2, "total_tokens": 2},
    }
    return {**body, **overrides}


@pytest.mark.parametrize(
    "body",
    [
        json.dumps(vector_body(data=vector_body()["data"][:1])),
        json.dumps(
            vector_body(
                data=[
                    {"index": 0, "embedding": [0.5] * DIMENSIONS},
                    {"index": 0, "embedding": [0.5] * DIMENSIONS},
                ]
            )
        ),
        json.dumps(
            vector_body(
                data=[
                    {"index": 0, "embedding": [0.5] * (DIMENSIONS - 1)},
                    {"index": 1, "embedding": [0.5] * DIMENSIONS},
                ]
            )
        ),
        json.dumps(
            vector_body(
                data=[
                    {"index": 0, "embedding": [True] * DIMENSIONS},
                    {"index": 1, "embedding": [0.5] * DIMENSIONS},
                ]
            )
        ),
        json.dumps(vector_body(model="bad model with spaces")),
        # Python's JSON parser accepts NaN, so the adapter must reject it explicitly.
        json.dumps(vector_body()).replace("0.25", "NaN", 1),
        json.dumps(vector_body()).replace("0.25", "Infinity", 1),
        json.dumps(vector_body()).replace("0.25", "1" + "0" * 400, 1),
        json.dumps(vector_body()).replace("0.25", "1e300", 1),
        json.dumps(
            vector_body(
                data=[
                    {"index": 0, "embedding": [0.0] * DIMENSIONS},
                    {"index": 1, "embedding": [0.5] * DIMENSIONS},
                ]
            )
        ),
    ],
    ids=[
        "count",
        "duplicate-index",
        "dimensions",
        "boolean",
        "model",
        "nan",
        "infinity",
        "huge-integer",
        "beyond-float4",
        "zero-vector",
    ],
)
async def test_embedding_rejects_malformed_vectors(simulator, body):
    url, app, _ = simulator

    calls = []

    async def malformed(request: Request):
        calls.append(1)
        return Response(body, media_type="application/json")

    replace_route(app, "/v1/embeddings", malformed)
    embedder = OpenAIEmbedder(settings_for(url))
    try:
        with pytest.raises(ProviderFailure) as error:
            await embedder.embed(["one", "two"])
    finally:
        await embedder.aclose()
    assert error.value.code == "provider_invalid_output"
    assert not error.value.retryable and not error.value.ambiguous
    assert len(calls) == 1


async def test_embedding_returns_vectors_in_input_order(simulator):
    url, app, _ = simulator

    async def reversed_order(request: Request):
        return JSONResponse(vector_body(data=list(reversed(vector_body()["data"]))))

    replace_route(app, "/v1/embeddings", reversed_order)
    embedder = OpenAIEmbedder(settings_for(url))
    try:
        result = await embedder.embed(["one", "two"])
    finally:
        await embedder.aclose()
    assert [vector[0] for vector in result.vectors] == [0.5, 0.25]


async def test_quoted_answer_wire_contract_and_claims(simulator):
    url, _, captured = simulator
    answerer = OpenAIAnswerer(settings_for(url, ai_model="pinned-answer-model"))
    try:
        result = await answerer.answer("How fast is the pump?", PASSAGES)
    finally:
        await answerer.aclose()
    path, wire = captured[0]
    assert path == "/v1/responses"
    assert wire["model"] == "pinned-answer-model"
    assert wire["store"] is False and wire["tools"] == [] and wire["tool_choice"] == "none"
    assert json.loads(wire["input"]) == {
        "question": "How fast is the pump?",
        "passages": [
            {
                "label": "P1",
                "title": "Pump manual",
                "section": None,
                "page": 3,
                "text": PASSAGES[0].text,
            },
            {
                "label": "P2",
                "title": "Pump manual",
                "section": None,
                "page": 4,
                "text": PASSAGES[1].text,
            },
        ],
    }
    schema = wire["text"]["format"]
    assert schema["strict"] is True and set(schema["schema"]["required"]) == {"claims"}
    claim_schema = schema["schema"]["$defs"]["Claim"]
    assert set(claim_schema["required"]) == {"statement", "citation", "quote"}
    assert claim_schema["additionalProperties"] is False
    assert result.citations == (0,)
    (claim,) = result.claims
    assert claim.citation == "P1" and claim.quote == PASSAGES[0].text
    assert result.answer == claim.statement
    assert result.model == "simulator-item-rewrite"


CLAIM = {"statement": "It runs fast.", "citation": "P1", "quote": "runs at 40 litres"}


def answer_body(answer: dict) -> dict:
    return {
        "id": "resp_answer",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "synthetic-model",
        "output": [
            {
                "id": "msg_answer",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": json.dumps(answer), "annotations": []}],
            }
        ],
    }


@pytest.mark.parametrize(
    "answer",
    [
        {"claims": [CLAIM | {"citation": "P3"}]},
        {"claims": [CLAIM | {"citation": "S1"}]},
        {"claims": [CLAIM | {"statement": ""}]},
        {"claims": [CLAIM | {"quote": ""}]},
        {"claims": [CLAIM | {"quote": "x" * 301}]},
        {"claims": [CLAIM | {"statement": "Nul\u0000"}]},
        {"claims": [CLAIM] * 9},
        {"claims": [CLAIM | {"confidence": 1}]},
        {"claims": [], "answer": "Extra"},
        {"answer": "Old shape", "citations": ["P1"]},
    ],
    ids=[
        "outside-passages",
        "label",
        "empty-statement",
        "empty-quote",
        "long-quote",
        "nul",
        "too-many",
        "extra-claim-field",
        "extra-field",
        "v1-shape",
    ],
)
async def test_quoted_answer_rejects_invalid_output(simulator, answer):
    url, app, _ = simulator

    async def malformed(request: Request):
        return JSONResponse(answer_body(answer))

    replace_route(app, "/v1/responses", malformed)
    answerer = OpenAIAnswerer(settings_for(url))
    try:
        with pytest.raises(ProviderFailure) as error:
            await answerer.answer("Question?", PASSAGES)
    finally:
        await answerer.aclose()
    assert error.value.code == "provider_invalid_output"


async def test_quoted_answer_joins_claims_and_deduplicates_citations(simulator):
    url, app, _ = simulator
    claims = [
        CLAIM | {"citation": "P2", "statement": "Clean it monthly."},
        CLAIM | {"statement": "It runs fast."},
        CLAIM | {"citation": "P2", "statement": "Use a brush."},
    ]

    async def repeated(request: Request):
        return JSONResponse(answer_body({"claims": claims}))

    replace_route(app, "/v1/responses", repeated)
    answerer = OpenAIAnswerer(settings_for(url))
    try:
        result = await answerer.answer("Question?", PASSAGES)
    finally:
        await answerer.aclose()
    assert result.citations == (1, 0)
    assert result.answer == "Clean it monthly. It runs fast. Use a brush."
    assert [claim.citation for claim in result.claims] == ["P2", "P1", "P2"]


async def test_no_claims_means_not_in_passages(simulator):
    url, _, _ = simulator
    await configure(url, "uncited")
    answerer = OpenAIAnswerer(settings_for(url))
    try:
        result = await answerer.answer("Question?", PASSAGES)
    finally:
        await answerer.aclose()
    assert (result.answer, result.citations, result.claims) == ("", (), ())


async def test_out_of_range_batches_never_send(simulator):
    url, _, captured = simulator
    embedder = OpenAIEmbedder(settings_for(url))
    answerer = OpenAIAnswerer(settings_for(url), client=embedder.client)
    try:
        with pytest.raises(ValueError):
            await embedder.embed([])
        with pytest.raises(ValueError):
            await answerer.answer("Question?", [])
        with pytest.raises(ValueError):
            await answerer.answer("Question?", PASSAGES * 5)
    finally:
        await answerer.aclose()
        await embedder.aclose()
    assert captured == []


async def test_rewrite_refuses_the_rag_prompt(simulator):
    url, _, captured = simulator
    rewriter = OpenAIRewriter(settings_for(url))
    try:
        with pytest.raises(ProviderFailure) as error:
            await rewriter.rewrite("Title", "", "Polish", prompt_version=RAG_PROMPT_VERSION)
    finally:
        await rewriter.aclose()
    assert error.value.code == "provider_prompt_version" and captured == []


@pytest.mark.parametrize(
    ("mode", "code"),
    [
        ("refusal", "provider_refusal"),
        ("incomplete", "provider_incomplete"),
        ("rate_limit", "provider_rate_limit"),
        ("server_error", "provider_unavailable"),
        ("timeout", "provider_timeout"),
    ],
)
async def test_quoted_answer_failures_are_classified_once(simulator, mode, code):
    url, app, captured = simulator
    await configure(url, mode)
    answerer = OpenAIAnswerer(
        settings_for(url, ai_provider_timeout_seconds=0.1 if mode == "timeout" else 1)
    )
    try:
        with pytest.raises(ProviderFailure) as error:
            await answerer.answer("Question?", PASSAGES)
    finally:
        await answerer.aclose()
    assert error.value.code == code
    assert app.state.simulator.calls == 1 and len(captured) == 1


async def test_embedding_cancellation_propagates_without_replay(simulator):
    url, app, _ = simulator
    await configure(url, "timeout", delay=1)
    embedder = OpenAIEmbedder(settings_for(url, ai_embedding_timeout_seconds=10))
    task = asyncio.create_task(embedder.embed(["slow"]))
    try:
        async with asyncio.timeout(2):
            while not app.state.simulator.in_flight:
                await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert app.state.simulator.calls == 1
    finally:
        await embedder.aclose()


async def test_borrowed_client_with_sdk_retries_is_rejected(simulator):
    from openai import AsyncOpenAI

    url, _, _ = simulator
    client = AsyncOpenAI(api_key="synthetic", base_url=url + "/v1")
    try:
        with pytest.raises(ValueError):
            OpenAIEmbedder(settings_for(url), client=client)
    finally:
        await client.close()


async def test_rerank_wire_contract_and_order(simulator):
    url, _, captured = simulator
    reranker = OpenAIReranker(settings_for(url))
    try:
        result = await reranker.rerank("How fast?", PASSAGES, 1)
    finally:
        await reranker.aclose()
    (_, wire) = captured[0]
    payload = json.loads(wire["input"])
    assert payload["limit"] == 1 and [item["label"] for item in payload["passages"]] == ["P1", "P2"]
    assert set(wire["text"]["format"]["schema"]["required"]) == {"passages"}
    assert result.order == (0,)


@pytest.mark.parametrize(
    "labels",
    [["P1", "P1"], ["P3"], ["P1", "P2", "P1"], ["P2", "P1", "P2"]],
    ids=["duplicate", "outside-passages", "over-limit", "duplicate-over-limit"],
)
async def test_rerank_rejects_invalid_rankings(simulator, labels):
    url, app, _ = simulator

    async def malformed(request: Request):
        return JSONResponse(answer_body({"passages": labels}))

    replace_route(app, "/v1/responses", malformed)
    reranker = OpenAIReranker(settings_for(url))
    try:
        with pytest.raises(ProviderFailure) as error:
            await reranker.rerank("Question?", PASSAGES, 2)
    finally:
        await reranker.aclose()
    assert error.value.code == "provider_invalid_output"


async def test_section_reaches_the_answer_prompt(simulator):
    url, _, captured = simulator
    answerer = OpenAIAnswerer(settings_for(url))
    try:
        await answerer.answer(
            "Q?", [Passage(title="T", page=None, text="a b c d", section="A > B")]
        )
    finally:
        await answerer.aclose()
    (_, wire) = captured[0]
    assert json.loads(wire["input"])["passages"][0]["section"] == "A > B"
