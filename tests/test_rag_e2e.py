"""Real HTTP -> PostgreSQL -> Kafka -> rag-worker -> query scenarios; opt in via AI_QA=1."""

import asyncio
import json
import os
from uuid import UUID, uuid4

import pytest
from aiokafka import AIOKafkaConsumer
from sqlalchemy import select

from app.config import Settings
from app.models import RAGDocument
from app.rag.config import RAGSettings
from tests.support.e2e import eventually, isolated_runtime

pytestmark = pytest.mark.skipif(
    os.environ.get("AI_QA") != "1", reason="requires isolated AI QA stack"
)
DOCUMENTS = "/api/v1/rag/documents"
QUERY = "/api/v1/rag/query"
QUESTION = "How many litres per minute does the pump run?"
SETTLED = {"ready", "failed"}
# The worker's embedding call can hang this long before it times out; crash tests kill first.
EMBEDDING_TIMEOUT = 5
LEASE_SECONDS = 16


@pytest.fixture
def settings() -> Settings:
    # The QA container disables RAG for other suites; this module's API enables it.
    return Settings(ai_enabled=True, rag_enabled=True)


@pytest.fixture
async def runtime(app):
    prefix = f"rag.qa.{uuid4().hex}"
    overrides = {
        "RAG_ENABLED": "true",
        "KAFKA_RAG_REQUEST_TOPIC": f"{prefix}.requests",
        "KAFKA_RAG_DEAD_LETTER_TOPIC": f"{prefix}.dead-letter",
        "KAFKA_RAG_GROUP_ID": prefix,
        "AI_EMBEDDING_TIMEOUT_SECONDS": str(EMBEDDING_TIMEOUT),
        "RAG_PDF_TIMEOUT_SECONDS": str(EMBEDDING_TIMEOUT),
        "RAG_LEASE_SECONDS": str(LEASE_SECONDS),
    }
    async with isolated_runtime(
        app,
        settings_type=RAGSettings,
        overrides=overrides,
        topics=[f"{prefix}.requests", f"{prefix}.dead-letter"],
        roles=["dispatcher", "rag-worker"],
    ) as children:
        yield children


async def upload(client, owner, text: str, *, name: str = "pump.md") -> dict:
    response = await client.post(
        DOCUMENTS,
        headers={**owner["headers"], "Idempotency-Key": f"e2e-{uuid4().hex}"},
        data={"title": "Pump manual"},
        files={"file": (name, text.encode())},
    )
    assert response.status_code == 202
    return response.json()


async def settled(client, owner, document, runtime) -> dict:
    async def read():
        runtime.healthy()
        response = await client.get(document["status_url"], headers=owner["headers"])
        assert response.status_code == 200
        result = response.json()
        return result if result["status"] in SETTLED else None

    return await eventually(read, message="document did not settle")


async def attempt_of(app, document) -> int:
    async with app.state.session_factory() as session:
        return await session.scalar(
            select(RAGDocument.attempt).where(RAGDocument.id == UUID(document["id"]))
        )


async def test_upload_ingest_query_isolation_and_content_free_telemetry(
    app, client, account, runtime
):
    owner, stranger = await account(), await account()
    secret = f"private-rag-{uuid4().hex}"
    text = f"# Pump manual\n\nThe pump runs at 40 litres per minute. Reference {secret}."
    document = await upload(client, owner, text)

    result = await settled(client, owner, document, runtime)
    assert result["status"] == "ready" and result["chunk_count"] == 1

    answer = await client.post(QUERY, headers=owner["headers"], json={"question": QUESTION})
    assert answer.status_code == 200
    body = answer.json()
    assert body["status"] == "answered"
    assert [citation["document_id"] for citation in body["citations"]] == [document["id"]]
    foreign = await client.post(QUERY, headers=stranger["headers"], json={"question": QUESTION})
    assert foreign.json()["status"] == "no_context"
    # One embedding batch for ingestion, then the question embedding and one answer.
    # The stranger's query makes no call.
    assert await runtime.calls() == 3

    # The API's own spans are exported by this process; flush before reading.
    assert await asyncio.to_thread(app.state.telemetry.provider.force_flush, 2000)

    async def routed():
        response = await runtime.receiver.get("/__state")
        assert response.status_code == 200
        spans = response.json()["spans"]
        names = {(span["channel"], span["name"]) for span in spans}
        wanted = {
            (channel, name)
            for channel in ("langfuse", "langsmith")
            for name in ("rag.ingest", "ai.embed", "rag.query", "ai.generate")
        }
        return spans if wanted <= names else None

    spans = await eventually(routed, seconds=15, message="RAG spans not observed on both routes")
    serialized = json.dumps(spans)
    for private in (secret, text, QUESTION, body["answer"], owner["access_token"]):
        assert private not in serialized, "Document, question or answer text escaped"
    for span in spans:
        attributes = span["attributes"]
        if span["channel"] == "langfuse" and span["name"] == "ai.embed":
            assert attributes["langfuse.observation.type"] == "embedding"
        if span["channel"] == "langfuse" and span["name"] == "rag.query":
            assert attributes["langfuse.observation.type"] == "chain"
        if span["channel"] == "langsmith" and span["name"] == "ai.embed":
            assert attributes["langsmith.span.kind"] == "embedding"
    ingest = next(span for span in spans if span["name"] == "rag.ingest")
    assert ingest["attributes"]["document_id"] == document["id"]


async def test_worker_crash_mid_embedding_recovers_on_the_next_attempt(
    app, client, account, runtime
):
    owner = await account()
    # The first embedding call hangs; the worker is killed while it waits.
    await runtime.control(["timeout"], delay_seconds=EMBEDDING_TIMEOUT * 2)
    document = await upload(client, owner, "The pump runs at 40 litres per minute.")

    async def embedding_in_flight():
        runtime.healthy()
        response = await runtime.simulator.get("/__state")
        return response.json()["in_flight"] == 1

    await eventually(embedding_in_flight, message="worker never called the embedder")
    await runtime.stop("rag-worker", crash=True)
    runtime.start("rag-worker")

    result = await settled(client, owner, document, runtime)
    assert result["status"] == "ready"
    # The dispatcher recovered the expired lease as attempt 2; attempt 1 never wrote.
    assert await attempt_of(app, document) == 2


async def test_unusable_document_is_dead_lettered_without_content(app, client, account, runtime):
    owner = await account()
    document = await upload(client, owner, "   \n\t  ", name="blank.txt")

    result = await settled(client, owner, document, runtime)
    assert (result["status"], result["error_code"]) == ("failed", "no_text")
    assert await runtime.calls() == 0

    consumer = AIOKafkaConsumer(
        runtime.settings.kafka_rag_dead_letter_topic,
        bootstrap_servers=runtime.settings.kafka_bootstrap_servers,
        group_id=f"qa-check-{uuid4().hex}",
        auto_offset_reset="earliest",
        enable_auto_commit=False,
    )
    await consumer.start()
    try:
        async with asyncio.timeout(20):
            while True:
                payload = json.loads((await consumer.getone()).value)
                if payload.get("document_id") == document["id"]:
                    break
    finally:
        await consumer.stop()
    assert set(payload) == {
        "version",
        "event_id",
        "document_id",
        "attempt",
        "created_at",
        "status",
        "error_code",
    }
    assert (payload["status"], payload["error_code"]) == ("failed", "no_text")
