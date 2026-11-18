"""Synchronous RAG queries over real PostgreSQL, pgvector and the SDK against a simulator."""

import asyncio
import importlib.util
import socket
import sys
from contextlib import asynccontextmanager
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_core.runnables import RunnableLambda
from langchain_core.tracers.langchain import wait_for_all_tracers
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from sqlalchemy import func, insert, select
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from app.config import Settings
from app.models import RAG_EMBEDDING_DIMENSIONS, RAGChunk, RAGDocument, RAGQuery
from app.rag.contracts import QueryRequest
from app.rag.pipeline import NO_CONTEXT_ANSWER, answer_query
from app.telemetry import Telemetry

URL = "/api/v1/rag/query"
DIAGNOSTICS = "/api/v1/rag/query/diagnostics"
QUERY_LIMIT = 3
EMBEDDING_MODEL = "text-embedding-3-small"
PUMP = "The pump runs at 40 litres per minute when the valve is open."
GARDEN = "Water the tomato plants every morning in the summer garden."
QUESTION = "How many litres per minute does the pump run?"


def _load_simulator():
    path = Path(__file__).parent / "support/provider_simulator.py"
    spec = importlib.util.spec_from_file_location("provider_simulator_query", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["provider_simulator_query"] = module
    spec.loader.exec_module(module)
    return module


SIMULATOR = _load_simulator()


@asynccontextmanager
async def serve(application):
    """Serve an ASGI app on a free loopback port for the length of one test."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    server = uvicorn.Server(
        uvicorn.Config(application, access_log=False, log_level="critical", lifespan="off")
    )
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(3):
            while not server.started:
                if task.done():
                    await task
                await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 3)
        sock.close()


@pytest.fixture
async def simulator():
    application = SIMULATOR.create_app()
    async with serve(application) as base_url:
        yield application.state.simulator, base_url


@pytest.fixture
def settings(monkeypatch, simulator) -> Settings:
    # The lifespan reads RAG and provider settings from the environment.
    _, base_url = simulator
    for name, value in {
        "OPENAI_BASE_URL": base_url + "/v1",
        "OPENAI_API_KEY": "synthetic-key-never-external",
        "AI_PROVIDER_TIMEOUT_SECONDS": 1,
        "AI_EMBEDDING_TIMEOUT_SECONDS": 1,
        "RAG_QUERY_LIMIT": QUERY_LIMIT,
    }.items():
        monkeypatch.setenv(name, str(value))
    return Settings(ai_enabled=True, rag_enabled=True)


async def add_ready_document(
    app: FastAPI,
    owner_id: UUID,
    title: str,
    text: str,
    *,
    status: str = "ready",
    embedding_model: str = EMBEDDING_MODEL,
) -> UUID:
    """Store one chunk embedded exactly as the simulator embeds questions."""
    document_id = uuid4()
    ready = status == "ready"
    async with app.state.session_factory() as session:
        await session.execute(
            insert(RAGDocument).values(
                id=document_id,
                owner_id=owner_id,
                title=title,
                content_type="text/plain",
                size_bytes=len(text.encode()),
                content_sha256=sha256(text.encode()).hexdigest(),
                source=None if ready else text.encode(),
                idempotency_key_hash=uuid4().hex,
                request_fingerprint=uuid4().hex,
                status=status,
                deadline_at=func.now() + timedelta(hours=1),
                embedding_model=embedding_model,
                embedding_dimensions=RAG_EMBEDDING_DIMENSIONS,
                chunker_version="recursive-v1",
                chunk_count=1 if ready else None,
                page_count=1 if ready else None,
                finished_at=func.now() if ready else None,
            )
        )
        await session.commit()
        session.add(
            RAGChunk(
                document_id=document_id,
                owner_id=owner_id,
                ordinal=0,
                page=None,
                content=text,
                embedding=SIMULATOR.embed_text(text, RAG_EMBEDDING_DIMENSIONS),
            )
        )
        await session.commit()
    return document_id


async def ask(client: httpx.AsyncClient, owner: dict, **body) -> httpx.Response:
    return await client.post(URL, headers=owner["headers"], json={"question": QUESTION, **body})


async def query_rows(app: FastAPI) -> list[RAGQuery]:
    async with app.state.session_factory() as session:
        return list((await session.scalars(select(RAGQuery))).all())


async def test_answer_cites_the_matching_document(app, client, account, simulator):
    state, _ = simulator
    owner = await account()
    owner_id = UUID(owner["id"])
    pump = await add_ready_document(app, owner_id, "Pump manual", PUMP)
    await add_ready_document(app, owner_id, "Garden notes", GARDEN)

    response = await ask(client, owner, top_k=1)

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["status"] == "answered"
    assert [citation["document_id"] for citation in body["citations"]] == [str(pump)]
    assert body["citations"][0]["title"] == "Pump manual"
    assert body["citations"][0]["snippet"] == PUMP
    # One question embedding, one answer.
    assert state.calls == 2
    (row,) = await query_rows(app)
    assert (row.status, row.retrieved_count, row.top_k) == ("answered", 1, 1)
    assert row.finished_at is not None and row.input_tokens == 25 and row.output_tokens == 15
    assert row.provider_request_id.startswith("req_sim_")


async def test_other_owner_gets_no_context_without_provider_calls(app, client, account, simulator):
    state, _ = simulator
    owner, stranger = await account(), await account()
    pump = await add_ready_document(app, UUID(owner["id"]), "Pump manual", PUMP)

    unfiltered = await ask(client, stranger)
    foreign_filter = await ask(client, stranger, document_ids=[str(pump)])

    for response in (unfiltered, foreign_filter):
        assert response.status_code == 200
        assert response.json() == {
            "status": "no_context",
            "answer": NO_CONTEXT_ANSWER,
            "citations": [],
        }
    assert state.calls == 0
    rows = await query_rows(app)
    assert {(row.status, row.retrieved_count) for row in rows} == {("no_context", 0)}


async def test_document_filter_limits_the_search(app, client, account):
    owner = await account()
    owner_id = UUID(owner["id"])
    await add_ready_document(app, owner_id, "Pump manual", PUMP)
    garden = await add_ready_document(app, owner_id, "Garden notes", GARDEN)

    response = await ask(client, owner, document_ids=[str(garden)])

    assert response.status_code == 200
    assert {citation["document_id"] for citation in response.json()["citations"]} == {str(garden)}


@pytest.mark.parametrize(
    "overrides",
    [{"embedding_model": "other-embedding-model"}, {"status": "pending"}],
    ids=["other-embedding-model", "not-ready"],
)
async def test_unsearchable_documents_are_excluded(app, client, account, simulator, overrides):
    state, _ = simulator
    owner = await account()
    await add_ready_document(app, UUID(owner["id"]), "Pump manual", PUMP, **overrides)

    response = await ask(client, owner)

    assert response.json()["status"] == "no_context"
    assert state.calls == 0


async def test_uncited_answer_is_never_shown(app, client, account, simulator):
    state, _ = simulator
    state.modes = ["success", "uncited"]
    owner = await account()
    await add_ready_document(app, UUID(owner["id"]), "Pump manual", PUMP)

    response = await ask(client, owner)

    assert response.status_code == 200
    assert response.json() == {"status": "no_context", "answer": NO_CONTEXT_ANSWER, "citations": []}
    assert "Unsupported" not in response.text
    (row,) = await query_rows(app)
    assert (row.status, row.retrieved_count, row.output_tokens) == ("no_context", 1, 15)


async def test_parallel_queries_admit_exactly_the_limit(app, client, account):
    owner = await account()

    responses = await asyncio.gather(*(ask(client, owner) for _ in range(QUERY_LIMIT + 1)))

    statuses = sorted(response.status_code for response in responses)
    assert statuses == [200] * QUERY_LIMIT + [429]
    limited = next(response for response in responses if response.status_code == 429)
    assert int(limited.headers["retry-after"]) >= 1
    assert len(await query_rows(app)) == QUERY_LIMIT


@pytest.mark.parametrize(
    ("modes", "status_code", "error_code"),
    [
        (["rate_limit"], 503, "provider_rate_limit"),
        (["success", "rate_limit"], 503, "provider_rate_limit"),
        (["success", "timeout"], 504, "provider_timeout"),
        (["success", "server_error"], 502, "provider_unavailable"),
        (["success", "refusal"], 502, "provider_refusal"),
        (["success", "invalid"], 502, "provider_invalid_output"),
    ],
    ids=["embed-rate-limit", "answer-rate-limit", "timeout", "server-error", "refusal", "invalid"],
)
async def test_provider_failures_map_to_fixed_errors(
    app, client, account, simulator, modes, status_code, error_code
):
    state, _ = simulator
    state.modes = list(modes)
    state.delay_seconds = 2
    owner = await account()
    await add_ready_document(app, UUID(owner["id"]), "Pump manual", PUMP)

    response = await ask(client, owner)

    assert response.status_code == status_code
    assert QUESTION not in response.text and PUMP not in response.text
    (row,) = await query_rows(app)
    assert (row.status, row.error_code) == ("failed", error_code)
    assert row.finished_at is not None


async def test_query_deadline_returns_504(app, client, account, simulator):
    state, _ = simulator
    state.modes = ["timeout"]
    state.delay_seconds = 2
    runtime = app.state.rag
    runtime.settings = runtime.settings.model_copy(update={"rag_query_deadline_seconds": 0.3})
    owner = await account()
    await add_ready_document(app, UUID(owner["id"]), "Pump manual", PUMP)

    response = await ask(client, owner)

    assert response.status_code == 504
    (row,) = await query_rows(app)
    assert (row.status, row.error_code) == ("failed", "query_deadline")


async def test_langsmith_environment_cannot_export_content(app, client, account, monkeypatch):
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

            owner = await account()
            await add_ready_document(app, UUID(owner["id"]), "Pump manual", PUMP)
            response = await ask(client, owner)
            assert response.json()["status"] == "answered"
            await asyncio.to_thread(wait_for_all_tracers)
            assert [entry for entry in received if entry[0] != "GET"] == []
        finally:
            monkeypatch.delenv("LANGSMITH_TRACING")
            langsmith.utils.get_env_var.cache_clear()


async def test_invalid_query_is_rejected_without_echo(client, account):
    owner = await account()
    secret = "sk-never-echo-in-errors"
    for body in (
        {"question": secret, "top_k": 9},
        {"question": secret, "extra": True},
        {"question": "", "top_k": 1},
        {"question": secret, "document_ids": []},
    ):
        response = await client.post(URL, headers=owner["headers"], json=body)
        assert response.status_code == 422
        assert secret not in response.text


async def test_query_requires_authentication(client):
    response = await client.post(URL, json={"question": QUESTION})
    assert response.status_code == 401


async def test_disabled_rag_returns_503(monkeypatch, account, client, app):
    owner = await account()
    app.state.rag = None
    response = await ask(client, owner)
    assert response.status_code == 503


async def test_foreign_chunks_never_reach_citations(app, client, account):
    owner, other = await account(), await account()
    garden = await add_ready_document(app, UUID(owner["id"]), "Garden notes", GARDEN)
    # The other owner's chunk is the closest match to the question.
    await add_ready_document(app, UUID(other["id"]), "Pump manual", PUMP)

    response = await ask(client, owner, top_k=8)

    assert response.status_code == 200
    assert {citation["document_id"] for citation in response.json()["citations"]} <= {str(garden)}
    assert PUMP not in response.text and "Pump manual" not in response.text


@pytest.mark.parametrize("failing_use", [2, 3], ids=["context-check", "vector-search"])
async def test_database_failure_after_admission_is_recorded(app, client, account, failing_use):
    owner = await account()
    await add_ready_document(app, UUID(owner["id"]), "Pump manual", PUMP)
    real_factory = app.state.session_factory
    uses = 0

    def factory():
        # Session uses per query: 1 admit, 2 context check, 3 vector search, 4 usage row.
        nonlocal uses
        uses += 1
        if uses == failing_use:
            raise PoolTimeoutError("synthetic pool timeout")
        return real_factory()

    app.state.session_factory = factory
    try:
        response = await ask(client, owner)
    finally:
        app.state.session_factory = real_factory

    assert response.status_code == 503
    (row,) = await query_rows(app)
    assert (row.status, row.error_code) == ("failed", "database_unavailable")
    # The question embedding was paid before the vector search failed.
    assert (row.embedding_tokens is not None) is (failing_use == 3)


async def test_cancelled_query_is_recorded(app, account, simulator):
    state, _ = simulator
    state.modes = ["success", "timeout"]
    state.delay_seconds = 2
    owner = await account()
    await add_ready_document(app, UUID(owner["id"]), "Pump manual", PUMP)
    task = asyncio.create_task(
        answer_query(
            session_factory=app.state.session_factory,
            runtime=app.state.rag,
            owner_id=UUID(owner["id"]),
            request=QueryRequest(question=QUESTION),
        )
    )

    async def answering() -> bool:
        return state.in_flight == 1 and state.calls == 2

    async with asyncio.timeout(10):
        while not await answering():
            await asyncio.sleep(0.01)
    task.cancel()
    async with asyncio.timeout(10):
        with pytest.raises(asyncio.CancelledError):
            await task

    (row,) = await query_rows(app)
    assert (row.status, row.error_code) == ("failed", "query_cancelled")
    assert row.embedding_tokens is not None


async def test_diagnostics_route_is_absent_unless_enabled(app, client, account):
    owner = await account()
    body = {"question": QUESTION}
    # Absent, not just refused: no auth challenge, no schema entry, no usage row.
    assert (await client.post(DIAGNOSTICS, json=body)).status_code == 404
    assert (await client.post(DIAGNOSTICS, headers=owner["headers"], json=body)).status_code == 404
    assert DIAGNOSTICS not in (await client.get("/openapi.json")).json()["paths"]
    assert await query_rows(app) == []


async def test_diagnostics_return_the_passages_sent_to_the_model(
    app, client, account, settings, simulator, monkeypatch
):
    settings.rag_diagnostics_enabled = True
    exporter = InMemorySpanExporter()
    telemetry = Telemetry("rag-diagnostics-test", trace_exporter=exporter)
    runtime = app.state.rag
    for holder in (runtime, runtime.embedder, runtime.answerer):
        monkeypatch.setattr(holder, "telemetry", telemetry)
    owner = await account()
    owner_id = UUID(owner["id"])
    pump = await add_ready_document(app, owner_id, "Pump manual", PUMP)
    await add_ready_document(app, owner_id, "Garden notes", GARDEN)
    stranger = await account()
    await add_ready_document(app, UUID(stranger["id"]), "Stranger manual", PUMP)
    try:
        unauthenticated = await client.post(DIAGNOSTICS, json={"question": QUESTION})
        response = await client.post(
            DIAGNOSTICS, headers=owner["headers"], json={"question": QUESTION, "top_k": 3}
        )
        assert telemetry.provider.force_flush(timeout_millis=1000)
        spans = exporter.get_finished_spans()
    finally:
        telemetry.shutdown()

    assert unauthenticated.status_code == 401
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["status"] == "answered"
    # Only the owner's two documents, closest first, exactly as the model received them.
    assert [passage["label"] for passage in body["passages"]] == ["P1", "P2"]
    first = body["passages"][0]
    assert (first["document_id"], first["title"], first["ordinal"]) == (str(pump), "Pump manual", 0)
    assert (first["text"], first["section"], first["page"]) == (PUMP, None, None)
    assert body["passages"][1]["title"] == "Garden notes"
    assert body["passages"][0]["distance"] <= body["passages"][1]["distance"]
    assert body["citations"][0]["document_id"] == str(pump)
    assert (body["no_context_reason"], body["claims"], body["verdicts"]) == (None, None, None)
    (row,) = await query_rows(app)
    assert row.status == "answered"
    assert body["usage"] == {
        "embedding_tokens": row.embedding_tokens,
        "input_tokens": 25,
        "output_tokens": 15,
        "chat_calls": None,
        "reasoning_effort": "none",
    }
    # Passage and question text return only in the response body, never in a span.
    assert {"rag.query", "rag.retrieve"} <= {span.name for span in spans}
    assert not any(PUMP in span.to_json() or QUESTION in span.to_json() for span in spans)


async def test_diagnostics_count_toward_the_query_limit(app, client, account, settings):
    settings.rag_diagnostics_enabled = True
    owner = await account()
    for _ in range(QUERY_LIMIT - 1):
        assert (await ask(client, owner)).status_code == 200
    body = {"question": QUESTION}
    first = await client.post(DIAGNOSTICS, headers=owner["headers"], json=body)
    assert first.status_code == 200
    assert (first.json()["status"], first.json()["passages"]) == ("no_context", [])
    limited = await client.post(DIAGNOSTICS, headers=owner["headers"], json=body)
    assert limited.status_code == 429
    assert len(await query_rows(app)) == QUERY_LIMIT
