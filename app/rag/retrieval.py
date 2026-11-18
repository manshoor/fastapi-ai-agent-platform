"""Owner- or building-scoped vector search exposed as LangChain retrievers.

Owner scope: an account's own documents (`/api/v1/rag/query`).
Building scope: documents a manager uploaded for a building; the resident assistant
passes the building from the resident's membership, never from model output.

Neighbor windows (`expand`) widen each retrieved chunk to the chunks around it on the
same page, under the same scope, so a fact split at a chunk boundary reaches the model:

    retrieved:  ordinal 4 (anchor)            radius 1
    window:     3 + 4 + 5, overlap removed    cites the anchor (ordinal 4)
    windows that touch or overlap merge; the anchor is the closest retrieved chunk
"""

from contextlib import nullcontext
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr
from sqlalchemy import and_, or_, select, text

from app.ai.processing import SessionFactory
from app.models import RAGChunk, RAGDocument
from app.rag.ingestion import Embedder

# Shortest suffix/prefix match treated as the splitter's overlap, not a coincidence.
MIN_OVERLAP_CHARACTERS = 20


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
    return [RAGDocument.status == "ready", RAGDocument.embedding_model == embedding_model]


def _searchable(owner_id: UUID, embedding_model: str, document_ids: list[UUID] | None):
    """Chunks this owner may search: own, from ready documents, same embedding model."""
    conditions = [
        RAGChunk.owner_id == owner_id,
        RAGDocument.owner_id == owner_id,
        *_ready(embedding_model),
    ]
    if document_ids:
        conditions.append(RAGChunk.document_id.in_(document_ids))
    return conditions


def _building_searchable(building_id: UUID, embedding_model: str) -> list:
    """Chunks of one building's ready documents. The chunk FK pins chunk owner = document owner."""
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
        with span:
            rows = await self._search(distance)
            if self.telemetry is not None:
                self.telemetry.annotate(retrieved_count=len(rows))
        return [
            Document(
                page_content=row.content,
                metadata={
                    "document_id": row.document_id,
                    "title": row.title,
                    "ordinal": row.ordinal,
                    "page": row.page,
                    "heading": row.heading,
                    "distance": float(row.distance),
                },
            )
            for row in rows
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

    async def _search(self, distance) -> list:
        async with self.session_factory() as session:
            async with session.begin():
                # Filtered HNSW scans keep searching until enough rows pass the filter.
                await session.execute(text("SET LOCAL hnsw.iterative_scan = 'strict_order'"))
                rows = (
                    await session.execute(
                        select(
                            RAGChunk.document_id,
                            RAGDocument.title,
                            RAGChunk.ordinal,
                            RAGChunk.page,
                            RAGChunk.content,
                            RAGChunk.heading,
                            distance.label("distance"),
                        )
                        .join(RAGDocument, RAGDocument.id == RAGChunk.document_id)
                        .where(*self._conditions())
                        .order_by(distance)
                        .limit(self.top_k)
                    )
                ).all()
        return rows


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
