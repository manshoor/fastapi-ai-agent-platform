"""Owner- or building-scoped vector search exposed as LangChain retrievers.

Owner scope: an account's own documents (`/api/v1/rag/query`).
Building scope: documents a manager uploaded for a building; the resident assistant
passes the building from the resident's membership, never from model output.

Neighbor windows (`expand`) widen each retrieved chunk to the chunks around it on the
same page, under the same scope, so a fact split at a chunk boundary reaches the model:

    retrieved:  ordinal 4 (anchor)            radius 1
    window:     3 + 4 + 5, overlap removed    cites the anchor (ordinal 4)
    windows that touch or overlap merge; the anchor is the closest retrieved chunk

Graph expansion (`GraphSearch`, RAG_GRAPH_ENABLED) adds chunks the knowledge graph links
to the vector results, under the same scope and active versions:

    seeds:     scope nodes the question names (whole words), then nodes of the vector
               chunks within the relevance margin; at most MAX_SEED_NODES
    hop 1..n:  unread edges touching the frontier, outside the vector chunks, closest
               chunk first (at most FAN_OUT per hop)
    evidence:  the edges' chunks, not already retrieved, most links first
    passages:  vector chunks, then up to `extra_passages` graph chunks (source "graph")

Vector and graph reads run in one REPEATABLE READ snapshot. Graph passages are ordinary
chunk text: quotes are still checked against what was sent. With `max_passages` vector
rows already, the graph adds nothing.
"""

from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr
from sqlalchemy import and_, func, or_, select, text, tuple_

from app.ai.processing import SessionFactory
from app.models import RAGChunk, RAGDocument, RAGGraphEdge, RAGGraphNode
from app.rag.graph import question_phrases
from app.rag.ingestion import Embedder

# Shortest suffix/prefix match treated as the splitter's overlap, not a coincidence.
MIN_OVERLAP_CHARACTERS = 20
# The answer prompt's passage limit.
MAX_PASSAGES = 8
# Graph bounds per search: seed nodes, and edges read per hop.
MAX_SEED_NODES = 50
FAN_OUT = 50
VECTOR_SOURCE = "vector"
GRAPH_SOURCE = "graph"


@dataclass(frozen=True)
class GraphSearch:
    hops: int
    extra_passages: int


def _join_chunks(first: str, second: str) -> str:
    """Join neighbors, dropping the longest overlap the splitter repeated at the seam."""
    for size in range(min(len(first), len(second)), MIN_OVERLAP_CHARACTERS - 1, -1):
        if first.endswith(second[:size]):
            return first + second[size:]
    return f"{first}\n{second}"


def _windows(anchors: list[Document], radius: int) -> list[tuple[Document, int, int]]:
    """(anchor, first, last) ordinal ranges; touching ranges on one page merge.

    Anchors arrive closest first, and a merge keeps the earlier anchor: its closest.
    """
    windows = [
        (anchor, anchor.metadata["ordinal"] - radius, anchor.metadata["ordinal"] + radius)
        for anchor in anchors
    ]
    return _merged(windows)


def _merged(windows: list[tuple[Document, int, int]]) -> list[tuple[Document, int, int]]:
    # A widened range can now touch a later one: merge until nothing changes.
    changed = True
    while changed:
        changed = False
        for i, (kept, low, high) in enumerate(windows):
            for j in range(i + 1, len(windows)):
                other, other_low, other_high = windows[j]
                same = (kept.metadata["document_id"], kept.metadata["page"]) == (
                    other.metadata["document_id"],
                    other.metadata["page"],
                )
                if same and other_low <= high + 1 and low <= other_high + 1:
                    windows[i] = (kept, min(low, other_low), max(high, other_high))
                    del windows[j]
                    changed = True
                    break
            if changed:
                break
    return windows


def _ready(embedding_model: str) -> list:
    """Chunks of each document's active version only; a version being ingested is unseen."""
    return [
        RAGDocument.active_version.is_not(None),
        RAGChunk.version == RAGDocument.active_version,
        RAGDocument.embedding_model == embedding_model,
    ]


def _searchable(owner_id: UUID, embedding_model: str, document_ids: list[UUID] | None):
    """Chunks this owner may search: own, active version, same embedding model."""
    conditions = [
        RAGChunk.owner_id == owner_id,
        RAGDocument.owner_id == owner_id,
        *_ready(embedding_model),
    ]
    if document_ids:
        conditions.append(RAGChunk.document_id.in_(document_ids))
    return conditions


def _building_searchable(building_id: UUID, embedding_model: str) -> list:
    """Active chunks of one building's documents. The chunk FK pins chunk owner = document owner."""
    return [
        RAGDocument.building_id == building_id,
        RAGChunk.owner_id == RAGDocument.owner_id,
        *_ready(embedding_model),
    ]


async def _any_chunk(session_factory: SessionFactory, conditions: list) -> bool:
    async with session_factory() as session:
        found = await session.scalar(
            select(
                select(RAGChunk.ordinal)
                .join(RAGDocument, RAGDocument.id == RAGChunk.document_id)
                .where(*conditions)
                .exists()
            )
        )
        await session.rollback()
    return bool(found)


async def has_context(
    *,
    session_factory: SessionFactory,
    owner_id: UUID,
    embedding_model: str,
    document_ids: list[UUID] | None,
) -> bool:
    """Checked before any provider call: no searchable chunk means no paid request."""
    return await _any_chunk(session_factory, _searchable(owner_id, embedding_model, document_ids))


async def has_building_context(
    *, session_factory: SessionFactory, building_id: UUID, embedding_model: str
) -> bool:
    """Building variant of `has_context`, checked before the question is embedded."""
    return await _any_chunk(session_factory, _building_searchable(building_id, embedding_model))


class _ScopedRetriever(BaseRetriever):
    """Embeds the question, then runs cosine KNN over the chunks `_conditions` allows."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_factory: Any
    embedder: Any
    embedding_model: str
    top_k: int
    telemetry: Any = None
    # Knowledge graph expansion; None searches vectors only.
    graph: GraphSearch | None = None
    # Vector plus graph passages stay within this (the assistant passes its labels left).
    max_passages: int = MAX_PASSAGES
    # Vector rows farther than this behind the closest seed no graph search.
    relevance_margin: float | None = None
    # Usage of the last search, for the query's usage row.
    _embedding_tokens: int | None = PrivateAttr(default=None)

    @property
    def embedding_tokens(self) -> int | None:
        return self._embedding_tokens

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Document]:
        raise NotImplementedError("Use the async interface")

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        embedder: Embedder = self.embedder
        embedded = await embedder.embed([query], model=self.embedding_model)
        self._embedding_tokens = embedded.input_tokens
        vector = embedded.vectors[0]
        distance = RAGChunk.embedding.cosine_distance(vector)
        span = (
            self.telemetry.span("rag.retrieve", top_k=self.top_k)
            if self.telemetry is not None
            else nullcontext()
        )
        # Vector and graph reads share one snapshot: a version swap between them cannot
        # mix two versions of a chunk into one answer.
        async with self.session_factory() as session, session.begin():
            await session.connection(execution_options={"isolation_level": "REPEATABLE READ"})
            with span:
                rows = await self._search(session, distance)
                if self.telemetry is not None:
                    self.telemetry.annotate(retrieved_count=len(rows))
            limit = self.max_passages - len(rows)
            extra = min(self.graph.extra_passages, limit) if self.graph else 0
            if extra <= 0 or not rows:
                return [_document(row, VECTOR_SOURCE) for row in rows]
            span = (
                self.telemetry.span("rag.graph.expand", top_k=self.top_k)
                if self.telemetry is not None
                else nullcontext()
            )
            with span:
                linked = await self._graph_search(session, query, rows, distance, extra)
                if self.telemetry is not None:
                    self.telemetry.annotate(graph_passages=len(linked))
        return [_document(row, VECTOR_SOURCE) for row in rows] + [
            _document(row, GRAPH_SOURCE) for row in linked
        ]

    def _conditions(self) -> list:
        raise NotImplementedError

    async def expand(self, documents: list[Document], radius: int) -> list[Document]:
        """Widen each retrieved chunk to its neighbors; one query, same scope as search.

        The window's text is what the model reads and what quotes are checked against.
        Its metadata is the anchor's, so the citation points at the retrieved chunk;
        `anchor_text` keeps the anchor's own text for the snippet.
        """
        if radius <= 0 or not documents:
            return documents
        windows = _windows(documents, radius)
        ranges = [
            and_(
                RAGChunk.document_id == anchor.metadata["document_id"],
                RAGChunk.version == anchor.metadata["version"],
                RAGChunk.ordinal.between(first, last),
            )
            for anchor, first, last in windows
        ]
        async with self.session_factory() as session:
            rows = (
                await session.execute(
                    select(RAGChunk.document_id, RAGChunk.ordinal, RAGChunk.page, RAGChunk.content)
                    .join(RAGDocument, RAGDocument.id == RAGChunk.document_id)
                    .where(*self._conditions(), or_(*ranges))
                    .order_by(RAGChunk.document_id, RAGChunk.ordinal)
                )
            ).all()
            await session.rollback()
        expanded = []
        for anchor, first, last in windows:
            meta = anchor.metadata
            texts = [
                row.content
                for row in rows
                if row.document_id == meta["document_id"]
                and row.page == meta["page"]
                and first <= row.ordinal <= last
            ]
            window = texts[0] if texts else anchor.page_content
            for piece in texts[1:]:
                window = _join_chunks(window, piece)
            expanded.append(
                Document(
                    page_content=window,
                    metadata={**meta, "anchor_text": anchor.page_content},
                )
            )
        return expanded

    async def _graph_search(self, session, query: str, rows: list, distance, extra: int) -> list:
        """Chunks linked to the seeds through the graph, inside the caller's snapshot."""
        # Seeds: vector rows within the relevance margin. Every vector row is already a
        # passage, so none of them is evidence and their own edges are never read.
        closest = float(rows[0].distance)
        seeds = [
            row
            for row in rows
            if self.relevance_margin is None
            or float(row.distance) <= closest + self.relevance_margin
        ]
        retrieved = {_place(row) for row in rows}
        frontier = await self._seed_keys(session, query, {_place(row) for row in seeds}, distance)
        visited = set(frontier)
        read: set[tuple] = set()
        links: Counter = Counter()
        for _ in range(self.graph.hops):
            if not frontier:
                break
            edges = await self._edges(session, frontier, retrieved, read, distance)
            frontier = set()
            for edge in edges:
                read.add(_edge_identity(edge))
                links[_place(edge)] += 1
                for key in (edge.source_key, edge.target_key):
                    if key not in visited:
                        visited.add(key)
                        frontier.add(key)
        if not links:
            return []
        found = (
            await session.execute(
                self._columns(distance).where(
                    *self._conditions(),
                    tuple_(RAGChunk.document_id, RAGChunk.version, RAGChunk.ordinal).in_(
                        list(links)
                    ),
                )
            )
        ).all()
        # Most links first, then closest to the question.
        found.sort(key=lambda row: (-links[_place(row)], float(row.distance)))
        return found[:extra]

    def _nodes_in_scope(self, distance):
        """Distinct node keys in scope, closest chunk first; callers add the filter."""
        return (
            select(RAGGraphNode.node_key)
            .join(
                RAGChunk,
                and_(
                    RAGChunk.document_id == RAGGraphNode.document_id,
                    RAGChunk.version == RAGGraphNode.version,
                    RAGChunk.ordinal == RAGGraphNode.ordinal,
                ),
            )
            .join(RAGDocument, RAGDocument.id == RAGChunk.document_id)
            .where(*self._conditions())
            .group_by(RAGGraphNode.node_key)
            .order_by(func.min(distance), RAGGraphNode.node_key)
            .limit(MAX_SEED_NODES)
        )

    async def _seed_keys(self, session, query: str, seeds: set, distance) -> set[str]:
        """Nodes the question names (first), then nodes of the seed chunks."""
        keys: list[str] = []
        phrases = question_phrases(query)
        if phrases:
            # Index lookup on match_name, then the scope join.
            named = self._nodes_in_scope(distance).where(RAGGraphNode.match_name.in_(phrases))
            keys.extend(await session.scalars(named))
        if seeds:
            located = self._nodes_in_scope(distance).where(
                tuple_(RAGGraphNode.document_id, RAGGraphNode.version, RAGGraphNode.ordinal).in_(
                    list(seeds)
                )
            )
            keys.extend(key for key in await session.scalars(located) if key not in keys)
        return set(keys[:MAX_SEED_NODES])

    async def _edges(
        self, session, frontier: set[str], retrieved: set, read: set, distance
    ) -> list:
        """Unread edges in scope that touch the frontier, outside the retrieved chunks;
        at most FAN_OUT, closest chunk first."""
        keys = sorted(frontier)
        conditions = [
            *self._conditions(),
            or_(RAGGraphEdge.source_key.in_(keys), RAGGraphEdge.target_key.in_(keys)),
            tuple_(RAGGraphEdge.document_id, RAGGraphEdge.version, RAGGraphEdge.ordinal).not_in(
                list(retrieved)
            ),
        ]
        if read:
            conditions.append(
                tuple_(
                    RAGGraphEdge.document_id,
                    RAGGraphEdge.version,
                    RAGGraphEdge.ordinal,
                    RAGGraphEdge.source_key,
                    RAGGraphEdge.relation,
                    RAGGraphEdge.target_key,
                ).not_in(list(read))
            )
        return (
            await session.execute(
                select(
                    RAGGraphEdge.document_id,
                    RAGGraphEdge.version,
                    RAGGraphEdge.ordinal,
                    RAGGraphEdge.source_key,
                    RAGGraphEdge.relation,
                    RAGGraphEdge.target_key,
                )
                .join(
                    RAGChunk,
                    and_(
                        RAGChunk.document_id == RAGGraphEdge.document_id,
                        RAGChunk.version == RAGGraphEdge.version,
                        RAGChunk.ordinal == RAGGraphEdge.ordinal,
                    ),
                )
                .join(RAGDocument, RAGDocument.id == RAGChunk.document_id)
                .where(*conditions)
                .order_by(
                    distance,
                    RAGGraphEdge.document_id,
                    RAGGraphEdge.ordinal,
                    RAGGraphEdge.source_key,
                    RAGGraphEdge.relation,
                    RAGGraphEdge.target_key,
                )
                .limit(FAN_OUT)
            )
        ).all()

    def _columns(self, distance):
        return select(
            RAGChunk.document_id,
            RAGChunk.version,
            RAGDocument.title,
            RAGChunk.ordinal,
            RAGChunk.page,
            RAGChunk.content,
            RAGChunk.heading,
            distance.label("distance"),
        ).join(RAGDocument, RAGDocument.id == RAGChunk.document_id)

    async def _search(self, session, distance) -> list:
        # Filtered HNSW scans keep searching until enough rows pass the filter.
        await session.execute(text("SET LOCAL hnsw.iterative_scan = 'strict_order'"))
        return (
            await session.execute(
                self._columns(distance)
                .where(*self._conditions())
                .order_by(distance)
                .limit(self.top_k)
            )
        ).all()


def _place(row) -> tuple:
    return (row.document_id, row.version, row.ordinal)


def _edge_identity(edge) -> tuple:
    return (*_place(edge), edge.source_key, edge.relation, edge.target_key)


def _document(row, source: str) -> Document:
    return Document(
        page_content=row.content,
        metadata={
            "document_id": row.document_id,
            # Expansion reads only this version: after a swap it keeps the anchor.
            "version": row.version,
            "title": row.title,
            "ordinal": row.ordinal,
            "page": row.page,
            "heading": row.heading,
            "distance": float(row.distance),
            "source": source,
        },
    )


def graph_search(settings) -> GraphSearch | None:
    """The settings' graph expansion, or None when the graph is off."""
    if not settings.rag_graph_enabled:
        return None
    return GraphSearch(
        hops=settings.rag_graph_hops, extra_passages=settings.rag_graph_extra_passages
    )


class OwnerScopedRetriever(_ScopedRetriever):
    """One owner's ready chunks, optionally limited to some of their documents."""

    owner_id: UUID
    document_ids: list[UUID] | None = None

    def _conditions(self) -> list:
        return _searchable(self.owner_id, self.embedding_model, self.document_ids)


class BuildingScopedRetriever(_ScopedRetriever):
    """Ready chunks of one building's documents."""

    building_id: UUID

    def _conditions(self) -> list:
        return _building_searchable(self.building_id, self.embedding_model)
