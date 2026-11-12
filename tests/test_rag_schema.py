"""Database-level guarantees for RAG rows, independent of the API layer."""

from datetime import UTC, datetime, timedelta
from hashlib import sha256
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from sqlalchemy import delete, func, insert, select, text, update
from sqlalchemy.exc import DBAPIError, IntegrityError

from app.models import RAG_EMBEDDING_DIMENSIONS, RAGChunk, RAGDocument, RAGOutbox, RAGQuery, User

SOURCE = b"# Notes\n\nPostgreSQL stores the chunks."


def vector(index: int) -> list[float]:
    # Unit vector along one axis: distinct, valid cosine input.
    values = [0.0] * RAG_EMBEDDING_DIMENSIONS
    values[index] = 1.0
    return values


def document_row(owner_id: UUID, **overrides) -> dict:
    row = {
        "id": uuid4(),
        "owner_id": owner_id,
        "title": "Notes",
        "content_type": "text/markdown",
        "size_bytes": len(SOURCE),
        "content_sha256": sha256(SOURCE).hexdigest(),
        "source": SOURCE,
        "idempotency_key_hash": uuid4().hex,
        "request_fingerprint": uuid4().hex,
        "deadline_at": datetime.now(UTC) + timedelta(hours=1),
        "embedding_model": "text-embedding-3-small",
        "embedding_dimensions": RAG_EMBEDDING_DIMENSIONS,
        "chunker_version": "recursive-v1",
    }
    return {**row, **overrides}


async def owner_ids(app: FastAPI, account, count: int) -> list[UUID]:
    return [UUID((await account())["id"]) for _ in range(count)]


async def add_document(app: FastAPI, owner_id: UUID, **overrides) -> UUID:
    row = document_row(owner_id, **overrides)
    async with app.state.session_factory() as session:
        await session.execute(insert(RAGDocument).values(row))
        await session.commit()
    return row["id"]


async def test_chunk_cannot_belong_to_another_owner(app: FastAPI, account):
    owner, stranger = await owner_ids(app, account, 2)
    document_id = await add_document(app, owner)
    async with app.state.session_factory() as session:
        with pytest.raises(IntegrityError) as error:
            await session.execute(
                insert(RAGChunk).values(
                    document_id=document_id,
                    ordinal=0,
                    owner_id=stranger,
                    content="foreign",
                    embedding=vector(0),
                )
            )
        assert "fk_rag_chunks_document_owner" in str(error.value.orig)


async def test_document_delete_cascades_chunks_and_outbox(app: FastAPI, account):
    (owner,) = await owner_ids(app, account, 1)
    document_id = await add_document(app, owner)
    async with app.state.session_factory() as session:
        await session.execute(
            insert(RAGChunk),
            [
                {
                    "document_id": document_id,
                    "ordinal": ordinal,
                    "owner_id": owner,
                    "page": None,
                    "content": f"chunk {ordinal}",
                    "embedding": vector(ordinal),
                }
                for ordinal in range(3)
            ],
        )
        await session.execute(
            insert(RAGOutbox).values(
                id=uuid4(), document_id=document_id, event_kind="requested", attempt=1
            )
        )
        await session.commit()
        await session.execute(delete(RAGDocument).where(RAGDocument.id == document_id))
        await session.commit()
        assert await session.scalar(select(func.count()).select_from(RAGChunk)) == 0
        assert await session.scalar(select(func.count()).select_from(RAGOutbox)) == 0


async def test_user_delete_cascades_documents_chunks_and_queries(app: FastAPI, account):
    (owner,) = await owner_ids(app, account, 1)
    document_id = await add_document(app, owner)
    async with app.state.session_factory() as session:
        await session.execute(
            insert(RAGChunk).values(
                document_id=document_id,
                ordinal=0,
                owner_id=owner,
                content="kept until delete",
                embedding=vector(0),
            )
        )
        await session.execute(
            insert(RAGQuery).values(
                id=uuid4(),
                owner_id=owner,
                top_k=4,
                embedding_model="text-embedding-3-small",
                model="gpt-5.6-terra",
                prompt_version="rag-answer-v1",
            )
        )
        await session.commit()
        await session.execute(delete(User).where(User.id == owner))
        await session.commit()
        for model in (RAGDocument, RAGChunk, RAGQuery):
            assert await session.scalar(select(func.count()).select_from(model)) == 0


@pytest.mark.parametrize(
    ("overrides", "constraint"),
    [
        # Unfinished rows must keep their source bytes.
        ({"source": None}, "ck_rag_documents_source"),
        # Terminal rows must drop source bytes and record when they finished.
        (
            {"status": "ready", "chunk_count": 1, "finished_at": datetime.now(UTC)},
            "ck_rag_documents_source",
        ),
        (
            {"status": "ready", "source": None, "finished_at": datetime.now(UTC)},
            "ck_rag_documents_ready",
        ),
        (
            {"status": "failed", "source": None, "finished_at": datetime.now(UTC)},
            "ck_rag_documents_error",
        ),
        ({"status": "ready", "source": None, "chunk_count": 1}, "ck_rag_documents_terminal"),
        ({"content_type": "text/html"}, "ck_rag_documents_content_type"),
        ({"embedding_dimensions": 3072}, "ck_rag_documents_dimensions"),
        ({"attempt": 4}, "ck_rag_documents_attempt"),
        # Each biconditional, checked from the unfinished side as well.
        ({"chunk_count": 1}, "ck_rag_documents_ready"),
        ({"finished_at": datetime.now(UTC)}, "ck_rag_documents_terminal"),
        (
            {"status": "ready", "source": None, "chunk_count": 0, "finished_at": datetime.now(UTC)},
            "ck_rag_documents_ready",
        ),
        (
            {
                "status": "ready",
                "source": None,
                "chunk_count": 1,
                "error_code": "no_text",
                "finished_at": datetime.now(UTC),
            },
            "ck_rag_documents_error",
        ),
        ({"status": "processing"}, "ck_rag_documents_lease"),
        (
            {"lease_token": uuid4(), "lease_expires_at": datetime.now(UTC)},
            "ck_rag_documents_lease",
        ),
        ({"size_bytes": len(SOURCE) + 1}, "ck_rag_documents_size"),
        ({"page_count": 0}, "ck_rag_documents_counts"),
    ],
)
async def test_document_state_checks(app: FastAPI, account, overrides, constraint):
    (owner,) = await owner_ids(app, account, 1)
    with pytest.raises(IntegrityError) as error:
        await add_document(app, owner, **overrides)
    assert constraint in str(error.value.orig)


async def test_terminal_document_without_source_is_valid(app: FastAPI, account):
    (owner,) = await owner_ids(app, account, 1)
    document_id = await add_document(app, owner)
    async with app.state.session_factory() as session:
        await session.execute(
            update(RAGDocument)
            .where(RAGDocument.id == document_id)
            .values(status="ready", source=None, chunk_count=1, finished_at=func.now())
        )
        await session.commit()
        loaded = await session.get(RAGDocument, document_id)
        assert loaded.status == "ready" and loaded.chunk_count == 1


async def test_retry_and_lease_transitions_are_valid(app: FastAPI, account):
    (owner,) = await owner_ids(app, account, 1)
    document_id = await add_document(app, owner)
    transitions = [
        # Claim: processing always carries a lease.
        {"status": "processing", "lease_token": uuid4(), "lease_expires_at": func.now()},
        # Retry: back to pending, next attempt, lease cleared, last error kept.
        {
            "status": "pending",
            "attempt": 2,
            "lease_token": None,
            "lease_expires_at": None,
            "error_code": "provider_timeout",
        },
        # Terminal failure drops the source bytes.
        {"status": "failed", "source": None, "finished_at": func.now()},
    ]
    async with app.state.session_factory() as session:
        for values in transitions:
            await session.execute(
                update(RAGDocument).where(RAGDocument.id == document_id).values(**values)
            )
            await session.commit()
        loaded = await session.get(RAGDocument, document_id)
        assert (loaded.status, loaded.attempt, loaded.error_code) == (
            "failed",
            2,
            "provider_timeout",
        )


async def test_idempotency_key_is_unique_per_owner(app: FastAPI, account):
    owner, stranger = await owner_ids(app, account, 2)
    key = uuid4().hex
    await add_document(app, owner, idempotency_key_hash=key)
    await add_document(app, stranger, idempotency_key_hash=key)
    with pytest.raises(IntegrityError) as error:
        await add_document(app, owner, idempotency_key_hash=key)
    assert "uq_rag_documents_owner_key" in str(error.value.orig)


async def test_failed_query_requires_error_code(app: FastAPI, account):
    (owner,) = await owner_ids(app, account, 1)
    async with app.state.session_factory() as session:
        with pytest.raises(IntegrityError) as error:
            await session.execute(
                insert(RAGQuery).values(
                    id=uuid4(),
                    owner_id=owner,
                    status="failed",
                    top_k=4,
                    embedding_model="text-embedding-3-small",
                    model="gpt-5.6-terra",
                    prompt_version="rag-answer-v1",
                    finished_at=func.now(),
                )
            )
        assert "ck_rag_queries_error" in str(error.value.orig)


async def test_embedding_dimension_is_enforced(app: FastAPI, account):
    (owner,) = await owner_ids(app, account, 1)
    document_id = await add_document(app, owner)
    async with app.state.session_factory() as session:
        with pytest.raises(DBAPIError) as error:
            await session.execute(
                insert(RAGChunk).values(
                    document_id=document_id,
                    ordinal=0,
                    owner_id=owner,
                    content="short vector",
                    embedding=[1.0, 0.0, 0.0],
                )
            )
        assert f"expected {RAG_EMBEDDING_DIMENSIONS} dimensions" in str(error.value.orig)


async def test_zero_vector_is_rejected(app: FastAPI, account):
    (owner,) = await owner_ids(app, account, 1)
    document_id = await add_document(app, owner)
    async with app.state.session_factory() as session:
        with pytest.raises(IntegrityError) as error:
            await session.execute(
                insert(RAGChunk).values(
                    document_id=document_id,
                    ordinal=0,
                    owner_id=owner,
                    content="no direction",
                    embedding=[0.0] * RAG_EMBEDDING_DIMENSIONS,
                )
            )
        assert "ck_rag_chunks_norm" in str(error.value.orig)


async def test_cosine_order_can_use_the_hnsw_index(app: FastAPI):
    # Proves the operator class matches the <=> operator the retriever uses.
    async with app.state.session_factory() as session, session.begin():
        await session.execute(text("SET LOCAL enable_seqscan = off"))
        plan = (
            await session.execute(
                text("EXPLAIN SELECT ordinal FROM rag_chunks ORDER BY embedding <=> :v LIMIT 4"),
                {"v": str(vector(0))},
            )
        ).scalars()
        assert "ix_rag_chunks_embedding_hnsw" in "\n".join(plan)


async def test_owner_filtered_cosine_search_round_trips(app: FastAPI, account):
    owner, stranger = await owner_ids(app, account, 2)
    mine = await add_document(app, owner)
    theirs = await add_document(app, stranger)
    async with app.state.session_factory() as session:
        await session.execute(
            insert(RAGChunk),
            [
                {
                    "document_id": document_id,
                    "ordinal": 0,
                    "owner_id": owner_id,
                    "page": 1,
                    "content": label,
                    "embedding": vector(axis),
                }
                for document_id, owner_id, label, axis in (
                    (mine, owner, "mine", 1),
                    (theirs, stranger, "theirs", 0),
                )
            ],
        )
        await session.commit()
        async with session.begin():
            # Iterative scan keeps filtered HNSW searches from returning too few rows.
            await session.execute(text("SET LOCAL hnsw.iterative_scan = 'strict_order'"))
            rows = (
                await session.execute(
                    select(RAGChunk.content, RAGChunk.embedding)
                    .where(RAGChunk.owner_id == owner)
                    .order_by(RAGChunk.embedding.cosine_distance(vector(0)))
                    .limit(4)
                )
            ).all()
    # The closer foreign chunk is never visible to this owner.
    assert [content for content, _ in rows] == ["mine"]
    assert len(rows[0].embedding) == RAG_EMBEDDING_DIMENSIONS
