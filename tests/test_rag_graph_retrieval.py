"""Graph-augmented retrieval and the graph read, on real PostgreSQL."""

from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.ai.processing import HandleResult
from app.ai.provider import Embeddings
from app.models import User
from app.rag import retrieval as retrieval_module
from app.rag.contracts import ContentType
from app.rag.documents import document_graph, submit_document
from app.rag.retrieval import GraphSearch, OwnerScopedRetriever
from app.rag.uploads import Upload
from tests.support.provider_simulator import EMBEDDING_DIMENSIONS, embed_text
from tests.test_rag_graph import ControlledExtractor, ingest, rag_graph_runtime  # noqa: F401
from tests.test_rag_ingestion import rag_runtime  # noqa: F401
from tests.test_rag_versions import put_content

GYM = b"The gym is located in the basement. Bring a towel to the gym."
RULES = b"The basement is governed by the noise rules. Quiet hours start at 22:00."
OTHER = b"The pool is located in the courtyard. The courtyard opens at 07:00."


class BagEmbedder:
    """The simulator's bag-of-words vectors: shared words give nearby vectors."""

    async def embed(self, texts, *, model=None):
        vectors = [embed_text(text, EMBEDDING_DIMENSIONS) for text in texts]
        return Embeddings(vectors, "simulated-embedding", len(texts))


async def ready(runtime, data: bytes, owner_id=None):
    if owner_id is None:
        event = await runtime.submit(data)
    else:
        async with runtime.factory() as session:
            response, _ = await submit_document(
                session=session,
                settings=runtime.settings,
                owner_id=owner_id,
                upload=Upload(title="Doc", content_type=ContentType.TEXT, data=data),
                idempotency_key=uuid4().hex,
                trace_context={},
            )
        event = await runtime.event_for(response.id)
    result = await ingest(runtime, event, ControlledExtractor(), BagEmbedder())
    assert result is HandleResult.HANDLED
    return event.document_id


ONE_HOP = GraphSearch(hops=1, extra_passages=2)


def retriever(runtime, *, graph=ONE_HOP, top_k=1, **kwargs):
    return OwnerScopedRetriever(
        session_factory=runtime.factory,
        embedder=BagEmbedder(),
        embedding_model=runtime.settings.ai_embedding_model,
        top_k=top_k,
        owner_id=kwargs.pop("owner_id", runtime.owner_id),
        graph=graph,
        **kwargs,
    )


def sources(documents) -> list[tuple[str, str]]:
    return [(d.metadata["source"], d.page_content.split(".")[0]) for d in documents]


async def new_owner(runtime):
    async with runtime.factory() as session:
        user = User(email=f"rag-{uuid4().hex}@example.com", password_hash="unused-test-hash")
        session.add(user)
        await session.commit()
        return user.id


async def test_one_hop_reaches_a_fact_vector_search_misses(graph_runtime):
    await ready(graph_runtime, GYM)
    await ready(graph_runtime, RULES)
    found = await retriever(graph_runtime).ainvoke("When can I use the gym with a towel?")
    assert sources(found) == [
        ("vector", "The gym is located in the basement"),
        ("graph", "The basement is governed by the noise rules"),
    ]
    # Vector search alone returns only the gym chunk.
    alone = await retriever(graph_runtime, graph=None).ainvoke("When can I use the gym?")
    assert [d.metadata["source"] for d in alone] == ["vector"]


async def test_two_hops_follow_the_chain(graph_runtime):
    await ready(graph_runtime, b"The spa is part of the gym. Robes are provided at the spa.")
    await ready(graph_runtime, GYM)
    await ready(graph_runtime, RULES)
    question = "Are robes provided at the spa?"
    one = await retriever(graph_runtime).ainvoke(question)
    two = await retriever(graph_runtime, graph=GraphSearch(hops=2, extra_passages=2)).ainvoke(
        question
    )
    assert "The basement is governed by the noise rules" not in [s for _, s in sources(one)]
    assert ("graph", "The basement is governed by the noise rules") in sources(two)


async def test_question_names_seed_the_graph(graph_runtime):
    await ready(graph_runtime, RULES)
    await ready(graph_runtime, OTHER)
    # The vector seed is the pool chunk; "basement" in the question seeds the rules.
    found = await retriever(graph_runtime).ainvoke("Is the pool or the basement open at 07:00?")
    assert ("graph", "The basement is governed by the noise rules") in sources(found)
    # Names match whole words only: "basements" names no node.
    found = await retriever(graph_runtime).ainvoke("Are the pool or basements open at 07:00?")
    assert [d.metadata["source"] for d in found] == ["vector"]


async def test_question_names_seed_first_and_once(graph_runtime, monkeypatch):
    monkeypatch.setattr(retrieval_module, "MAX_SEED_NODES", 2)
    await ready(graph_runtime, RULES)
    await ready(graph_runtime, OTHER)
    await ready(graph_runtime, OTHER)
    # "pool" is a node in two chunks; "basement" still takes the second seed slot.
    found = await retriever(graph_runtime).ainvoke("Is the pool or the basement open at 07:00?")
    assert ("graph", "The basement is governed by the noise rules") in sources(found)


async def test_seed_chunk_edges_take_no_fan_out_slot(graph_runtime, monkeypatch):
    monkeypatch.setattr(retrieval_module, "FAN_OUT", 1)
    await ready(graph_runtime, GYM)
    await ready(graph_runtime, RULES)
    found = await retriever(graph_runtime).ainvoke("When can I use the gym with a towel?")
    assert ("graph", "The basement is governed by the noise rules") in sources(found)


async def test_second_hop_reads_only_new_edges(graph_runtime, monkeypatch):
    monkeypatch.setattr(retrieval_module, "FAN_OUT", 1)
    await ready(graph_runtime, b"The spa is part of the gym. Robes are provided at the spa.")
    await ready(graph_runtime, GYM)
    await ready(graph_runtime, RULES)
    two = GraphSearch(hops=2, extra_passages=2)
    found = await retriever(graph_runtime, graph=two).ainvoke("Are robes provided at the spa?")
    assert ("graph", "The basement is governed by the noise rules") in sources(found)


async def test_margin_drops_seeds_before_the_graph(graph_runtime):
    await ready(graph_runtime, GYM)
    await ready(graph_runtime, RULES)
    await ready(graph_runtime, OTHER)
    question = "Does the pool in the courtyard open at 07:00, and is there a towel?"
    # Vector: the pool chunk (0.26) and the gym chunk (0.47); the rules link through the gym.
    wide = await retriever(graph_runtime, top_k=2).ainvoke(question)
    assert ("graph", "The basement is governed by the noise rules") in sources(wide)
    narrow = await retriever(graph_runtime, top_k=2, relevance_margin=0.1).ainvoke(question)
    assert "graph" not in [d.metadata["source"] for d in narrow]


async def test_another_owners_graph_is_never_reached(graph_runtime):
    await ready(graph_runtime, GYM)
    other = await new_owner(graph_runtime)
    # Same node keys (location:basement) in another account's document.
    await ready(graph_runtime, RULES, owner_id=other)
    found = await retriever(graph_runtime).ainvoke("When can I use the gym?")
    assert [d.metadata["source"] for d in found] == ["vector"]


async def test_only_the_active_version_links(graph_runtime):
    await ready(graph_runtime, GYM)
    rules = await ready(graph_runtime, RULES)
    # Version 2 is pending: version 1 stays served, so its edge still links.
    await put_content(graph_runtime, rules, b"The lobby is governed by the dress rules.")
    found = await retriever(graph_runtime).ainvoke("When can I use the gym?")
    assert ("graph", "The basement is governed by the noise rules") in sources(found)
    # After the swap the new version has no basement edge.
    event = await graph_runtime.event_for(rules, version=2)
    await ingest(graph_runtime, event, ControlledExtractor(), BagEmbedder())
    found = await retriever(graph_runtime).ainvoke("When can I use the gym?")
    assert [d.metadata["source"] for d in found] == ["vector"]


@pytest.mark.parametrize(("max_passages", "expected"), [(1, 1), (2, 2), (8, 2)])
async def test_graph_passages_fit_the_passage_budget(graph_runtime, max_passages, expected):
    await ready(graph_runtime, GYM)
    await ready(graph_runtime, RULES)
    found = await retriever(graph_runtime, max_passages=max_passages).ainvoke("the gym towel")
    assert len(found) == expected


async def test_graph_read_is_owner_only(graph_runtime):
    document_id = await ready(graph_runtime, RULES)
    async with graph_runtime.factory() as session:
        graph = await document_graph(
            session=session, owner_id=graph_runtime.owner_id, document_id=document_id
        )
    assert graph.version == 1 and not graph.truncated
    assert [(e.source, e.relation, e.target) for e in graph.edges] == [
        ("location:basement", "governed_by", "rule:noise rules")
    ]
    other = await new_owner(graph_runtime)
    async with graph_runtime.factory() as session:
        with pytest.raises(HTTPException) as missing:
            await document_graph(session=session, owner_id=other, document_id=document_id)
    assert missing.value.status_code == 404
