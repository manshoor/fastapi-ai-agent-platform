"""Real HTTP -> upload -> rag-worker -> assistant turn, with content-free spans; AI_QA=1."""

import asyncio
import json
import os
from uuid import uuid4

import pytest

from app.config import Settings
from app.rag.config import RAGSettings
from tests.support.e2e import eventually, isolated_runtime

pytestmark = pytest.mark.skipif(
    os.environ.get("AI_QA") != "1", reason="requires isolated AI QA stack"
)
DOCUMENTS = "/api/v1/rag/documents"
MESSAGES = "/api/v1/assistant/messages"
SETTLED = {"ready", "failed"}


@pytest.fixture
def settings() -> Settings:
    # The QA container disables RAG and the assistant for other suites.
    return Settings(
        ai_enabled=True, rag_enabled=True, assistant_enabled=True, assistant_demo_enabled=True
    )


@pytest.fixture
async def runtime(app):
    prefix = f"rag.qa.{uuid4().hex}"
    overrides = {
        "RAG_ENABLED": "true",
        "KAFKA_RAG_EVENT_TOPIC": f"{prefix}.events",
        "KAFKA_RAG_DEAD_LETTER_TOPIC": f"{prefix}.dead-letter",
        "KAFKA_RAG_GROUP_ID": prefix,
    }
    async with isolated_runtime(
        app,
        settings_type=RAGSettings,
        overrides=overrides,
        topics=[f"{prefix}.events", f"{prefix}.dead-letter"],
        roles=["dispatcher", "rag-worker"],
    ) as children:
        yield children


async def test_building_document_is_cited_and_spans_hold_no_content(app, client, account, runtime):
    owner = await account()
    view = (await client.post("/api/v1/residence/demo", headers=owner["headers"])).json()
    secret = f"private-rule-{uuid4().hex}"
    text = f"# House rules\n\nQuiet hours start at 22:00. Reference {secret}."
    upload = await client.post(
        DOCUMENTS,
        params={"building_id": view["building"]["id"]},
        headers={**owner["headers"], "Idempotency-Key": f"e2e-{uuid4().hex}"},
        data={"title": "House rules"},
        files={"file": ("rules.md", text.encode())},
    )
    assert upload.status_code == 202
    document = upload.json()

    async def ready():
        runtime.healthy()
        response = await client.get(document["status_url"], headers=owner["headers"])
        status = response.json()["status"]
        return status if status in SETTLED else None

    assert await eventually(ready, message="document did not settle") == "ready"

    # The simulator's keyword policy searches, then cites the first passage.
    message = f"When do quiet hours start? {secret}"
    answer = await client.post(MESSAGES, headers=owner["headers"], json={"message": message})
    assert answer.status_code == 200
    body = answer.json()
    assert body["status"] == "answered"
    assert [citation["document_id"] for citation in body["citations"]] == [document["id"]]
    assert [action["tool"] for action in body["actions"]] == ["search_documents"]

    assert await asyncio.to_thread(app.state.telemetry.provider.force_flush, 2000)

    async def routed():
        response = await runtime.receiver.get("/__state")
        spans = response.json()["spans"]
        names = {(span["channel"], span["name"]) for span in spans}
        wanted = {
            (channel, name)
            for channel in ("langfuse", "langsmith")
            for name in ("assistant.turn", "assistant.tool", "ai.generate", "ai.embed")
        }
        return spans if wanted <= names else None

    spans = await eventually(routed, seconds=15, message="assistant spans not observed")
    serialized = json.dumps(spans)
    for private in (secret, text, message, body["answer"], owner["access_token"]):
        assert private not in serialized, "Message, document or answer text escaped"
    kinds = {
        (span["channel"], span["name"]): span["attributes"].get(
            "langfuse.observation.type" if span["channel"] == "langfuse" else "langsmith.span.kind"
        )
        for span in spans
    }
    assert kinds[("langfuse", "assistant.turn")] == "agent"
    assert kinds[("langfuse", "assistant.tool")] == "tool"
    assert kinds[("langsmith", "assistant.tool")] == "tool"
    turn = next(span for span in spans if span["name"] == "assistant.turn")
    assert turn["attributes"]["turn_id"] == body["turn_id"]
