"""Owner- or building-scoped vector search exposed as LangChain retrievers.

Owner scope: an account's own documents (`/api/v1/rag/query`).
Building scope: documents a manager uploaded for a building; the resident assistant
passes the building from the resident's membership, never from model output.
"""

from contextlib import nullcontext
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr
from sqlalchemy import select, text

from app.ai.processing import SessionFactory
from app.models import RAGChunk, RAGDocument
from app.rag.ingestion import Embedder


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
                    "distance": float(row.distance),
                },
            )
            for row in rows
        ]

    def _conditions(self) -> list:
        raise NotImplementedError

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
