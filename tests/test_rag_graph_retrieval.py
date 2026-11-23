"""Graph-augmented retrieval and the graph read, on real PostgreSQL."""

from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import func, update

from app.ai.contracts import GraphExtraction
from app.ai.processing import HandleResult
from app.ai.provider import Embeddings, GraphResult
from app.models import Building, RAGChunk, RAGDocument, RAGGraphEdge, RAGGraphNode, User
from app.rag import retrieval as retrieval_module
from app.rag.contracts import ContentType
from app.rag.documents import document_graph, submit_document
from app.rag.retrieval import BuildingScopedRetriever, GraphSearch, OwnerScopedRetriever
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


async def ready(runtime, data: bytes, owner_id=None, extractor=None):
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
    result = await ingest(runtime, event, extractor or ControlledExtractor(), BagEmbedder())
    assert result is HandleResult.HANDLED
    return event.document_id


class RetypingExtractor(ControlledExtractor):
    """Types one name differently, as a live model may from one document to the next."""

    def __init__(self, name: str, node_type: str):
        super().__init__()
        self.name, self.node_type = name, node_type

    async def extract(self, chunks, *, limit, model=None):
        result = await super().extract(chunks, limit=limit, model=model)
        value = result.extraction.model_dump()
        for node in value["nodes"]:
            if node["name"] == self.name:
                node["type"] = self.node_type
        for edge in value["edges"]:
            for end in ("source", "target"):
                if edge[end] == self.name:
                    edge[f"{end}_type"] = self.node_type
        extraction = GraphExtraction.model_validate(value)
        return GraphResult(extraction, result.model, 10, 5, None)


class EdgelessExtractor(ControlledExtractor):
    """Names entities but misses every relation, as a live model may."""

    async def extract(self, chunks, *, limit, model=None):
        result = await super().extract(chunks, limit=limit, model=model)
        extraction = result.extraction.model_copy(update={"edges": []})
        return GraphResult(extraction, result.model, 10, 5, None)


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


async def test_one_entity_joins_across_types(graph_runtime):
    # The gym document types "basement" as a location; the rules document, as a facility.
    await ready(graph_runtime, GYM)
    await ready(graph_runtime, RULES, extractor=RetypingExtractor("basement", "facility"))
    found = await retriever(graph_runtime).ainvoke("When can I use the gym with a towel?")
    assert ("graph", "The basement is governed by the noise rules") in sources(found)


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
    # Same node keys (basement) in another account's document.
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
        ("basement", "governed_by", "noise rules")
    ]
    other = await new_owner(graph_runtime)
    async with graph_runtime.factory() as session:
        with pytest.raises(HTTPException) as missing:
            await document_graph(session=session, owner_id=other, document_id=document_id)
    assert missing.value.status_code == 404


RULES_PASSAGE = ("graph", "The basement is governed by the noise rules")
GYM_QUESTION = "When can I use the gym with a towel?"


async def test_mention_reaches_a_chunk_without_edges(graph_runtime):
    await ready(graph_runtime, GYM)
    await ready(graph_runtime, RULES, extractor=EdgelessExtractor())
    await ready(graph_runtime, OTHER, extractor=EdgelessExtractor())
    found = await retriever(graph_runtime).ainvoke(GYM_QUESTION)
    # The rules chunk names the basement; the pool chunk names nothing linked.
    assert sources(found) == [("vector", "The gym is located in the basement"), RULES_PASSAGE]


async def test_edges_rank_before_mentions(graph_runtime):
    await ready(graph_runtime, GYM)
    # Closer to the question than the rules chunk, and it mentions the gym.
    await ready(
        graph_runtime,
        b"The gym is part of the club. Bring a robe.",
        extractor=EdgelessExtractor(),
    )
    await ready(graph_runtime, RULES)
    question = "Bring a towel to the gym in the basement"
    both = await retriever(graph_runtime).ainvoke(question)
    assert sources(both)[1:] == [RULES_PASSAGE, ("graph", "The gym is part of the club")]
    one = GraphSearch(hops=1, extra_passages=1)
    single = await retriever(graph_runtime, graph=one).ainvoke(question)
    assert sources(single)[1:] == [RULES_PASSAGE]


async def test_mentions_rank_by_linked_entities(graph_runtime):
    await ready(graph_runtime, GYM)
    both = b"The gym is located in the basement. Painters work nights."
    club = b"The gym is part of the club. Use a towel."
    await ready(graph_runtime, club, extractor=EdgelessExtractor())
    await ready(graph_runtime, both, extractor=EdgelessExtractor())
    one = GraphSearch(hops=1, extra_passages=1)
    found = await retriever(graph_runtime, graph=one).ainvoke("Bring a towel to the gym")
    assert found[1].page_content == both.decode()


async def test_mentions_add_no_hop(graph_runtime):
    await ready(graph_runtime, b"The spa is part of the gym. Robes are provided at the spa.")
    # Mentions the gym; its own basement node must not lead on to the rules.
    await ready(graph_runtime, GYM, extractor=EdgelessExtractor())
    await ready(graph_runtime, RULES)
    two = GraphSearch(hops=2, extra_passages=2)
    found = await retriever(graph_runtime, graph=two).ainvoke("Are robes provided at the spa?")
    assert sources(found)[1:] == [("graph", "The gym is located in the basement")]


async def test_second_hop_frontier_is_mentioned(graph_runtime):
    await ready(graph_runtime, b"The spa is part of the gym. Robes are provided at the spa.")
    await ready(graph_runtime, GYM)
    await ready(graph_runtime, RULES, extractor=EdgelessExtractor())
    question = "Are robes provided at the spa?"
    one = await retriever(graph_runtime).ainvoke(question)
    assert RULES_PASSAGE not in sources(one)
    two = GraphSearch(hops=2, extra_passages=2)
    assert RULES_PASSAGE in sources(await retriever(graph_runtime, graph=two).ainvoke(question))


async def test_mentions_stay_in_scope(graph_runtime):
    await ready(graph_runtime, GYM)
    other = await new_owner(graph_runtime)
    await ready(graph_runtime, RULES, owner_id=other, extractor=EdgelessExtractor())
    found = await retriever(graph_runtime).ainvoke(GYM_QUESTION)
    assert [d.metadata["source"] for d in found] == ["vector"]


async def test_mentions_use_the_active_version(graph_runtime):
    await ready(graph_runtime, GYM)
    rules = await ready(graph_runtime, RULES, extractor=EdgelessExtractor())
    await put_content(graph_runtime, rules, b"The lobby is governed by the dress rules.")
    assert RULES_PASSAGE in sources(await retriever(graph_runtime).ainvoke(GYM_QUESTION))
    event = await graph_runtime.event_for(rules, version=2)
    await ingest(graph_runtime, event, EdgelessExtractor(), BagEmbedder())
    found = await retriever(graph_runtime).ainvoke(GYM_QUESTION)
    assert [d.metadata["source"] for d in found] == ["vector"]


async def test_mentions_fit_the_passage_budget(graph_runtime):
    await ready(graph_runtime, GYM)
    for rule in (b"noise", b"dress", b"smoke"):
        text = b"The basement is governed by the %s rules. Painters work nights." % rule
        await ready(graph_runtime, text, extractor=EdgelessExtractor())
    found = await retriever(graph_runtime).ainvoke(GYM_QUESTION)
    assert len(found) == 3
    found = await retriever(graph_runtime, max_passages=2).ainvoke(GYM_QUESTION)
    assert len(found) == 2


async def add_chunk(runtime, document_id, text, names, *, ordinal=1, version=1, page=None):
    """One more stored chunk of a document, and the nodes it names."""
    place = {
        "document_id": document_id,
        "version": version,
        "ordinal": ordinal,
        "owner_id": runtime.owner_id,
    }
    async with runtime.factory() as session:
        vector = embed_text(text, EMBEDDING_DIMENSIONS)
        session.add(RAGChunk(**place, content=text, embedding=vector, page=page))
        await session.flush()
        for name in names:
            node = {"node_key": name, "match_name": name, "name": name, "node_type": "facility"}
            session.add(RAGGraphNode(**place, **node))
        await session.commit()


LOCKERS = "Lockers near the gym in the basement."


@pytest.mark.parametrize(
    ("neighbor_chunks", "page", "expected"),
    [
        # The vector hit's window already holds the next chunk: the slot goes to the rules.
        (1, None, RULES.decode()),
        # No window, or another page the window never covers: the next chunk is new text.
        (0, None, LOCKERS),
        (1, 2, LOCKERS),
    ],
)
async def test_mentions_skip_the_neighbor_window(graph_runtime, neighbor_chunks, page, expected):
    gym = await ready(graph_runtime, GYM)
    await add_chunk(graph_runtime, gym, LOCKERS, ["gym", "basement"], page=page)
    await ready(graph_runtime, RULES, extractor=EdgelessExtractor())
    one = GraphSearch(hops=1, extra_passages=1)
    search = retriever(graph_runtime, graph=one, neighbor_chunks=neighbor_chunks)
    found = await search.ainvoke("Bring a towel to the gym")
    assert [d.page_content for d in found] == [GYM.decode(), expected]


async def test_mentions_reach_past_typed_keys(graph_runtime):
    await ready(graph_runtime, GYM)
    await ready(graph_runtime, RULES, extractor=EdgelessExtractor())
    # Rows stored before keys became type-free.
    async with graph_runtime.factory() as session:
        typed = RAGGraphNode.node_type + ":" + RAGGraphNode.node_key
        await session.execute(update(RAGGraphNode).values(node_key=typed))
        await session.execute(
            update(RAGGraphEdge).values(
                source_key=func.concat("facility:", RAGGraphEdge.source_key),
                target_key=func.concat("location:", RAGGraphEdge.target_key),
            )
        )
        await session.commit()
    assert RULES_PASSAGE in sources(await retriever(graph_runtime).ainvoke(GYM_QUESTION))


async def test_mentions_ignore_a_version_in_flight(graph_runtime):
    await ready(graph_runtime, GYM)
    rules = await ready(graph_runtime, OTHER, extractor=EdgelessExtractor())
    await add_chunk(graph_runtime, rules, RULES.decode(), ["basement"], ordinal=0, version=2)
    # Same place, other version: the active chunk must not borrow the new version's node.
    found = await retriever(graph_runtime).ainvoke(GYM_QUESTION)
    assert [d.metadata["source"] for d in found] == ["vector"]


async def test_closest_mention_comes_first(graph_runtime):
    gym = await ready(graph_runtime, GYM)
    await add_chunk(graph_runtime, gym, "Painters work nights.", ["basement"], ordinal=8)
    await add_chunk(graph_runtime, gym, "Bring a robe to it.", ["basement"], ordinal=9)
    one = GraphSearch(hops=1, extra_passages=1)
    found = await retriever(graph_runtime, graph=one).ainvoke("Bring a towel to the gym")
    assert found[1].page_content == "Bring a robe to it."


async def test_building_mentions_stay_in_the_building(graph_runtime):
    gym = await ready(graph_runtime, GYM)
    await ready(graph_runtime, RULES, extractor=EdgelessExtractor())
    text = b"The basement is governed by the dress rules. Wear shoes."
    near = await ready(graph_runtime, text, extractor=EdgelessExtractor())
    async with graph_runtime.factory() as session:
        building = Building(name="Tower")
        session.add(building)
        await session.flush()
        await session.execute(
            update(RAGDocument)
            .where(RAGDocument.id.in_([gym, near]))
            .values(building_id=building.id)
        )
        building_id = building.id
        await session.commit()
    search = BuildingScopedRetriever(
        session_factory=graph_runtime.factory,
        embedder=BagEmbedder(),
        embedding_model=graph_runtime.settings.ai_embedding_model,
        top_k=1,
        building_id=building_id,
        graph=ONE_HOP,
    )
    found = await search.ainvoke(GYM_QUESTION)
    # The private rules document has the same owner but no building.
    assert [d.metadata["document_id"] for d in found] == [gym, near]
