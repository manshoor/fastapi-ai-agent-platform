"""Knowledge graph: ontology, grounding, and extraction at ingest on real PostgreSQL."""

import json

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from app.ai.contracts import GraphExtraction
from app.ai.processing import HandleResult
from app.ai.provider import GraphResult, ProviderFailure
from app.models import RAGDocument, RAGGraphEdge, RAGGraphNode
from app.rag.chunking import Chunk
from app.rag.config import RAGSettings
from app.rag.graph import ground, node_key
from app.rag.ingestion import claim_document, process_ingest, recover_documents, store_batch
from app.rag.ontology import NodeType, Relation, acceptable_name, allowed
from tests.support.provider_simulator import graph_extraction
from tests.test_rag_ingestion import ControlledEmbedder, rag_runtime, unit_vector  # noqa: F401
from tests.test_rag_versions import put_content, remove

GYM = "The gym is located in the basement. The basement is governed by the noise rules."


def chunk(text: str, ordinal: int = 0) -> Chunk:
    return Chunk(ordinal=ordinal, page=None, content=text)


def extraction(nodes=(), edges=()) -> GraphExtraction:
    return GraphExtraction.model_validate({"nodes": list(nodes), "edges": list(edges)})


def edge(source, source_type, relation, target, target_type, quote, label="C1") -> dict:
    return {
        "chunk": label,
        "source": source,
        "source_type": source_type,
        "relation": relation,
        "target": target,
        "target_type": target_type,
        "quote": quote,
    }


# Ontology


def test_allowed_triples():
    assert allowed(NodeType.FACILITY, Relation.LOCATED_IN, NodeType.LOCATION)
    assert not allowed(NodeType.LOCATION, Relation.LOCATED_IN, NodeType.RULE)
    assert allowed(NodeType.RULE, Relation.PROHIBITS, NodeType.FACILITY)
    assert not allowed(NodeType.FACILITY, Relation.PROHIBITS, NodeType.RULE)
    # replaces links two things of one type.
    assert allowed(NodeType.DEVICE, Relation.REPLACES, NodeType.DEVICE)
    assert not allowed(NodeType.DEVICE, Relation.REPLACES, NodeType.SERVICE)


@pytest.mark.parametrize(
    ("name", "ok"),
    [
        ("concierge", True),
        ("bike room", True),
        ("desk@example.com", False),
        ("+44 20 7946 0000", False),
        ("call 555-010-0199", False),
        ("1234", False),
        ("x" * 121, False),
        # Contact details in typographic forms the normalizer folds.
        ("desk\uff20example.com", False),
        ("call 555\u2011010\u20110199", False),
        ("call 555\u2013010\u20130199", False),
        # A name starts and ends on a letter or digit: no word-edge bypass, no quote marks.
        ("-legal parking", False),
        ("'gym'", False),
        ("level 2", True),
    ],
)
def test_names_are_things_and_roles_not_contacts(name, ok):
    assert acceptable_name(name) is ok


def test_schema_refuses_types_outside_the_ontology():
    with pytest.raises(ValidationError):
        extraction(nodes=[{"chunk": "C1", "type": "person", "name": "Alex"}])
    with pytest.raises(ValidationError):
        extraction(edges=[edge("gym", "facility", "owns", "basement", "location", "x y z")])


# Grounding


def test_grounded_edge_and_nodes_are_kept():
    rows = ground(
        extraction(
            nodes=[{"chunk": "C1", "type": "facility", "name": "gym"}],
            edges=[
                edge(
                    "gym",
                    "facility",
                    "located_in",
                    "basement",
                    "location",
                    "The gym is located in the basement",
                )
            ],
        ),
        [chunk(GYM)],
        12,
    )
    assert {node.node_key for node in rows.nodes} == {"gym", "basement"}
    assert [(e.source_key, e.relation, e.target_key) for e in rows.edges] == [
        ("gym", Relation.LOCATED_IN, "basement")
    ]
    assert rows.dropped == 0


@pytest.mark.parametrize(
    "item",
    [
        # The quote is not in the chunk.
        edge("gym", "facility", "located_in", "basement", "location", "the gym is on the roof"),
        # The quote does not name the target.
        edge("gym", "facility", "located_in", "noise rules", "location", "The gym is located"),
        # A triple outside the ontology.
        edge("gym", "facility", "governed_by", "basement", "location", "The gym is located in"),
        # A quote shorter than three words.
        edge("gym", "facility", "located_in", "basement", "location", "gym basement"),
        # A chunk label past the batch.
        edge("gym", "facility", "located_in", "basement", "location", "The gym is", "C2"),
    ],
)
def test_ungrounded_edges_are_dropped(item):
    rows = ground(extraction(edges=[item]), [chunk(GYM)], 12)
    assert rows.edges == [] and rows.dropped == 1


def test_names_must_appear_in_the_chunk_at_word_edges():
    rows = ground(
        extraction(
            nodes=[
                {"chunk": "C1", "type": "facility", "name": "sauna"},
                # "ym" sits inside "gym": not a word.
                {"chunk": "C1", "type": "facility", "name": "ym"},
                {"chunk": "C1", "type": "facility", "name": "GYM"},
            ]
        ),
        [chunk(GYM)],
        12,
    )
    assert [node.node_key for node in rows.nodes] == ["gym"]
    assert rows.dropped == 2


def test_caps_and_duplicates_per_chunk():
    text = "The gym, the pool, the sauna and the spa are open."
    names = ["gym", "pool", "sauna", "spa", "gym"]
    rows = ground(
        extraction(nodes=[{"chunk": "C1", "type": "facility", "name": n} for n in names]),
        [chunk(text)],
        3,
    )
    assert [node.name for node in rows.nodes] == ["gym", "pool", "sauna"]
    # "spa" is over the cap; the second "gym" is the same node.
    assert rows.dropped == 1


def test_names_that_share_a_key_make_no_edge():
    # Two 120-character names that differ only past the 160-character key.
    source, target = "\ufb03" * 55 + "a", "\ufb03" * 55 + "b"
    text = f"The {source} replaces the {target} today."
    item = edge(source, "device", "replaces", target, "device", f"{source} replaces the {target}")
    rows = ground(extraction(edges=[item]), [chunk(text)], 12)
    assert rows.edges == [] and rows.dropped == 1


def test_node_key_normalizes_case_and_spacing():
    assert node_key("  Bike   Room ") == "bike room"
    # The type is not part of the identity: one entity, typed two ways, is one key.
    assert node_key("Level-2 Gym") == node_key("level 2 gym") == "level 2 gym"


# Extraction at ingest


class ControlledExtractor:
    """GraphExtractor fake: the simulator's sentence rules, or a failure per call."""

    def __init__(self, *, failures=None, mode="success"):
        self.failures = failures or {}
        self.mode = mode
        self.calls: list[list[str]] = []
        self.limits: list[int] = []

    async def extract(self, chunks, *, limit, model=None):
        self.calls.append(list(chunks))
        self.limits.append(limit)
        failure = self.failures.get(len(self.calls))
        if failure:
            raise failure
        payload = {
            "input": json.dumps(
                {"chunks": [{"label": f"C{i}", "text": t} for i, t in enumerate(chunks, 1)]}
            )
        }
        value = GraphExtraction.model_validate(graph_extraction(payload, self.mode))
        return GraphResult(value, "simulated-graph", 10, 5, None)


@pytest.fixture(name="graph_runtime")
def rag_graph_runtime(runtime):
    runtime.settings = RAGSettings.model_validate(
        {**runtime.settings.model_dump(), "rag_graph_enabled": True, "rag_graph_batch_size": 2}
    )
    return runtime


async def ingest(runtime, event, extractor=None, embedder=None):
    return await process_ingest(
        session_factory=runtime.factory,
        settings=runtime.settings,
        event=event,
        embedder=embedder or ControlledEmbedder(),
        extractor=extractor,
    )


async def graph(runtime, document_id):
    async with runtime.factory() as session:
        nodes = list(
            await session.scalars(
                select(RAGGraphNode).where(RAGGraphNode.document_id == document_id)
            )
        )
        edges = list(
            await session.scalars(
                select(RAGGraphEdge).where(RAGGraphEdge.document_id == document_id)
            )
        )
    return nodes, edges


async def test_ingest_stores_grounded_graph_with_the_chunks(graph_runtime):
    event = await graph_runtime.submit(GYM.encode())
    extractor = ControlledExtractor()
    assert await ingest(graph_runtime, event, extractor) is HandleResult.HANDLED
    document = await graph_runtime.read(event.document_id)
    nodes, edges = await graph(graph_runtime, event.document_id)
    assert (document.status, document.graph_error_code) == ("ready", None)
    assert {(e.source_key, e.relation, e.target_key) for e in edges} == {
        ("gym", "located_in", "basement"),
        ("basement", "governed_by", "noise rules"),
    }
    assert {node.owner_id for node in nodes} == {graph_runtime.owner_id}
    assert {node.version for node in nodes} == {1}


async def test_graph_batches_follow_the_graph_batch_size(graph_runtime):
    graph_runtime.settings = graph_runtime.settings.model_copy(
        update={"rag_embedding_batch_size": 4}
    )
    text = " ".join(f"sentence{index} about pumps and filters." for index in range(30))
    event = await graph_runtime.submit(text.encode())
    extractor, embedder = ControlledExtractor(), ControlledEmbedder()
    assert await ingest(graph_runtime, event, extractor, embedder) is HandleResult.HANDLED
    assert [len(call) for call in extractor.calls] == [len(call) for call in embedder.calls]
    assert max(len(call) for call in extractor.calls) == 2


async def test_misquoted_graph_is_dropped_and_the_document_served(graph_runtime):
    event = await graph_runtime.submit(GYM.encode())
    assert await ingest(graph_runtime, event, ControlledExtractor(mode="misquote")) is (
        HandleResult.HANDLED
    )
    nodes, edges = await graph(graph_runtime, event.document_id)
    # Every edge quotes no chunk; the names themselves are in the chunk, so nodes stay.
    assert edges == []
    assert {node.node_key for node in nodes} == {
        "gym",
        "basement",
        "noise rules",
    }
    assert (await graph_runtime.read(event.document_id)).status == "ready"


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        (ProviderFailure("provider_rejected"), "graph_rejected"),
        (ProviderFailure("provider_invalid_output"), "graph_invalid"),
        # Output cut at the token budget: the batch has no graph, and says why.
        (ProviderFailure("provider_incomplete"), "graph_incomplete"),
    ],
)
async def test_rejected_extraction_keeps_the_document_ready(graph_runtime, failure, code):
    event = await graph_runtime.submit(GYM.encode())
    extractor = ControlledExtractor(failures={1: failure})
    assert await ingest(graph_runtime, event, extractor) is HandleResult.HANDLED
    document = await graph_runtime.read(event.document_id)
    assert (document.status, document.graph_error_code) == ("ready", code)
    assert await graph(graph_runtime, event.document_id) == ([], [])


async def test_transient_extraction_failure_retries_the_document(graph_runtime):
    event = await graph_runtime.submit(GYM.encode())
    failure = ProviderFailure("provider_timeout", ambiguous=True)
    assert await ingest(graph_runtime, event, ControlledExtractor(failures={1: failure})) is (
        HandleResult.HANDLED
    )
    document = await graph_runtime.read(event.document_id)
    assert (document.status, document.attempt, document.error_code) == (
        "pending",
        2,
        "provider_timeout",
    )
    assert await graph(graph_runtime, event.document_id) == ([], [])


async def test_retry_removes_graph_rows_stored_before_the_failure(graph_runtime):
    text = GYM + " " + " ".join(f"sentence{i} about pumps and filters." for i in range(30))
    event = await graph_runtime.submit(text.encode())
    failure = ProviderFailure("provider_timeout", ambiguous=True)
    extractor = ControlledExtractor(failures={2: failure})
    assert await ingest(graph_runtime, event, extractor) is HandleResult.HANDLED
    # Batch 1 stored its graph; the failure on batch 2 removed it with the chunks.
    assert len(extractor.calls) == 2
    assert (await graph_runtime.read(event.document_id)).status == "pending"
    assert await graph(graph_runtime, event.document_id) == ([], [])


async def test_extraction_is_told_the_per_chunk_cap(graph_runtime):
    event = await graph_runtime.submit(GYM.encode())
    extractor = ControlledExtractor()
    assert await ingest(graph_runtime, event, extractor) is HandleResult.HANDLED
    assert extractor.limits == [graph_runtime.settings.rag_graph_max_items_per_chunk]


async def test_delete_clears_the_graph_error(graph_runtime):
    event = await graph_runtime.submit(GYM.encode())
    extractor = ControlledExtractor(failures={1: ProviderFailure("provider_rejected")})
    assert await ingest(graph_runtime, event, extractor) is HandleResult.HANDLED
    await remove(graph_runtime, event.document_id)
    assert (await graph_runtime.read(event.document_id)).graph_error_code is None


async def test_stale_worker_writes_no_graph_rows(graph_runtime):
    from app.rag.graph import GraphRows, NodeRow

    event = await graph_runtime.submit(GYM.encode())
    lease, _ = await claim_document(
        session_factory=graph_runtime.factory, settings=graph_runtime.settings, event=event
    )
    await graph_runtime.expire(event.document_id)
    assert await recover_documents(
        session_factory=graph_runtime.factory, settings=graph_runtime.settings
    )
    stored = await store_batch(
        session_factory=graph_runtime.factory,
        settings=graph_runtime.settings,
        lease=lease,
        chunks=[chunk("gym")],
        vectors=[unit_vector("gym")],
        graph=GraphRows(nodes=[NodeRow(0, "gym", NodeType.FACILITY, "gym", "gym")]),
    )
    assert stored is False
    assert await graph(graph_runtime, event.document_id) == ([], [])


async def test_swap_and_delete_remove_old_graph_rows(graph_runtime):
    event = await graph_runtime.submit(GYM.encode())
    assert await ingest(graph_runtime, event, ControlledExtractor()) is HandleResult.HANDLED
    await put_content(graph_runtime, event.document_id, b"The pool is located in the courtyard.")
    updated = await graph_runtime.event_for(event.document_id, version=2)
    assert await ingest(graph_runtime, updated, ControlledExtractor()) is HandleResult.HANDLED
    nodes, edges = await graph(graph_runtime, event.document_id)
    assert {node.version for node in nodes} == {2}
    assert [(e.source_key, e.target_key) for e in edges] == [("pool", "courtyard")]
    await remove(graph_runtime, event.document_id)
    assert await graph(graph_runtime, event.document_id) == ([], [])


async def test_graph_off_extracts_nothing(runtime):
    event = await runtime.submit(GYM.encode())
    assert await ingest(runtime, event) is HandleResult.HANDLED
    assert await graph(runtime, event.document_id) == ([], [])
    async with runtime.factory() as session:
        document = await session.get(RAGDocument, event.document_id)
    assert document.graph_error_code is None


def test_lease_must_cover_the_extraction_call(settings):
    base = {
        **settings.model_dump(),
        "ai_enabled": True,
        "rag_enabled": True,
        "rag_lease_seconds": 60,
        "rag_pdf_timeout_seconds": 30,
        "ai_embedding_timeout_seconds": 20,
        "ai_provider_timeout_seconds": 30,
        "ai_lease_seconds": 60,
    }
    # 30 s parse + 20 s embedding + 5 s fits 60 s; the 30 s extraction call does not.
    RAGSettings.model_validate(base)
    with pytest.raises(ValidationError, match="graph extraction"):
        RAGSettings.model_validate({**base, "rag_graph_enabled": True})
