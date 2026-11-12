"""Persistent account and owner-scoped item schema."""

from datetime import datetime
from uuid import UUID, uuid4

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Fixed by the vector(1536) column; changing it needs a migration and re-embedding.
RAG_EMBEDDING_DIMENSIONS = 1536


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("email", name="uq_users_email"),)

    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(254), nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Item(Base):
    __tablename__ = "items"

    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    owner_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(
        String(5000), nullable=False, server_default=text("''")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("ix_items_owner_created_id", owner_id, created_at.desc(), id.desc()),)


class AIAdmission(Base):
    __tablename__ = "ai_admission"
    __table_args__ = (CheckConstraint("id = 1", name="ck_ai_admission_singleton"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)


class AIJob(Base):
    __tablename__ = "ai_jobs"

    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    owner_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    # No item FK: deleting an item cannot erase an accepted billable operation.
    item_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), nullable=False)
    source_item_updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    source_title: Mapped[str | None] = mapped_column(String(200))
    source_description: Mapped[str | None] = mapped_column(String(5000))
    instruction: Mapped[str | None] = mapped_column(String(1000))
    idempotency_key_hash: Mapped[str] = mapped_column(String(64))
    request_fingerprint: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), server_default="pending")
    attempt: Mapped[int] = mapped_column(Integer, server_default="1")
    next_attempt_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    deadline_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    lease_token: Mapped[UUID | None] = mapped_column(PostgreSQLUUID(as_uuid=True))
    lease_generation: Mapped[int] = mapped_column(Integer, server_default="0")
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    provider_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    model: Mapped[str] = mapped_column(String(128))
    prompt_version: Mapped[str] = mapped_column(String(64))
    schema_version: Mapped[str] = mapped_column(String(16))
    result_title: Mapped[str | None] = mapped_column(String(200))
    result_description: Mapped[str | None] = mapped_column(String(5000))
    response_model: Mapped[str | None] = mapped_column(String(128))
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    provider_request_id: Mapped[str | None] = mapped_column(String(128))
    error_code: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    content_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    content_purged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    admission_released_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    trace_context: Mapped[dict[str, str]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))

    __table_args__ = (
        UniqueConstraint("owner_id", "idempotency_key_hash", name="uq_ai_jobs_owner_key"),
        CheckConstraint(
            "status IN ('pending','leased','calling','succeeded','failed','needs_reconciliation')",
            name="ck_ai_jobs_status",
        ),
        CheckConstraint("attempt BETWEEN 1 AND 3", name="ck_ai_jobs_attempt"),
        CheckConstraint("lease_generation >= 0", name="ck_ai_jobs_generation"),
        CheckConstraint(
            "(status IN ('succeeded','failed','needs_reconciliation')) = "
            "(finished_at IS NOT NULL AND content_expires_at IS NOT NULL)",
            name="ck_ai_jobs_terminal",
        ),
        CheckConstraint(
            "(status = 'succeeded' AND ((result_title IS NOT NULL AND "
            "result_description IS NOT NULL) OR content_purged_at IS NOT NULL)) OR "
            "(status <> 'succeeded' AND result_title IS NULL AND result_description IS NULL)",
            name="ck_ai_jobs_result",
        ),
        Index("ix_ai_jobs_owner_id", "owner_id", "id"),
        Index(
            "ix_ai_jobs_admission",
            "owner_id",
            postgresql_where=text("admission_released_at IS NULL"),
        ),
        Index(
            "ix_ai_jobs_due",
            "next_attempt_at",
            postgresql_where=text("status = 'pending'"),
        ),
        Index(
            "ix_ai_jobs_lease",
            "lease_expires_at",
            postgresql_where=text("status IN ('leased','calling')"),
        ),
        Index("ix_ai_jobs_content_expiry", "content_expires_at"),
    )


class AIOutbox(Base):
    __tablename__ = "ai_outbox"

    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    job_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("ai_jobs.id", ondelete="CASCADE")
    )
    event_kind: Mapped[str] = mapped_column(String(16))
    attempt: Mapped[int] = mapped_column(Integer)
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    lease_token: Mapped[UUID | None] = mapped_column(PostgreSQLUUID(as_uuid=True))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    publish_attempts: Mapped[int] = mapped_column(Integer, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("job_id", "event_kind", "attempt", name="uq_ai_outbox_job_event_attempt"),
        CheckConstraint(
            "event_kind IN ('requested','dead_letter')", name="ck_ai_outbox_event_kind"
        ),
        CheckConstraint("attempt BETWEEN 1 AND 3", name="ck_ai_outbox_attempt"),
        Index(
            "ix_ai_outbox_due",
            "available_at",
            postgresql_where=text("published_at IS NULL"),
        ),
        Index("ix_ai_outbox_job_id", "job_id"),
    )


# RAG documents: one row per upload. Source bytes live here only until ingestion
# ends; chunk text and embeddings live in rag_chunks until the owner deletes.
class RAGDocument(Base):
    __tablename__ = "rag_documents"

    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    owner_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    title: Mapped[str] = mapped_column(String(200))
    content_type: Mapped[str] = mapped_column(String(32))
    size_bytes: Mapped[int] = mapped_column(Integer)
    content_sha256: Mapped[str] = mapped_column(String(64))
    # Deferred: list/get/claim queries never load the upload bytes by accident.
    source: Mapped[bytes | None] = mapped_column(LargeBinary, deferred=True)
    idempotency_key_hash: Mapped[str] = mapped_column(String(64))
    request_fingerprint: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), server_default="pending")
    attempt: Mapped[int] = mapped_column(Integer, server_default="1")
    next_attempt_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    deadline_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    lease_token: Mapped[UUID | None] = mapped_column(PostgreSQLUUID(as_uuid=True))
    lease_generation: Mapped[int] = mapped_column(Integer, server_default="0")
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    embedding_model: Mapped[str] = mapped_column(String(128))
    embedding_dimensions: Mapped[int] = mapped_column(Integer)
    chunker_version: Mapped[str] = mapped_column(String(32))
    chunk_count: Mapped[int | None] = mapped_column(Integer)
    page_count: Mapped[int | None] = mapped_column(Integer)
    embedding_tokens: Mapped[int | None] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(String(64))
    trace_context: Mapped[dict[str, str]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("owner_id", "idempotency_key_hash", name="uq_rag_documents_owner_key"),
        # Target of the rag_chunks composite FK: a chunk's owner must be its document's owner.
        UniqueConstraint("id", "owner_id", name="uq_rag_documents_id_owner"),
        CheckConstraint(
            "status IN ('pending','processing','ready','failed')", name="ck_rag_documents_status"
        ),
        CheckConstraint(
            "content_type IN ('text/plain','text/markdown','application/pdf')",
            name="ck_rag_documents_content_type",
        ),
        CheckConstraint(
            "size_bytes > 0 AND (source IS NULL OR octet_length(source) = size_bytes)",
            name="ck_rag_documents_size",
        ),
        # Only a processing row holds a lease, and it always has one.
        CheckConstraint(
            "(status = 'processing') = (lease_token IS NOT NULL AND lease_expires_at IS NOT NULL)",
            name="ck_rag_documents_lease",
        ),
        CheckConstraint(
            "(chunk_count IS NULL OR chunk_count >= 0) AND (page_count IS NULL OR page_count >= 1) "
            "AND (embedding_tokens IS NULL OR embedding_tokens >= 0)",
            name="ck_rag_documents_counts",
        ),
        CheckConstraint("attempt BETWEEN 1 AND 3", name="ck_rag_documents_attempt"),
        CheckConstraint("lease_generation >= 0", name="ck_rag_documents_generation"),
        CheckConstraint(
            f"embedding_dimensions = {RAG_EMBEDDING_DIMENSIONS}",
            name="ck_rag_documents_dimensions",
        ),
        # Terminal rows are finished and hold no source bytes; unfinished rows keep them.
        CheckConstraint(
            "(status IN ('ready','failed')) = (finished_at IS NOT NULL)",
            name="ck_rag_documents_terminal",
        ),
        CheckConstraint(
            "(status IN ('ready','failed')) = (source IS NULL)", name="ck_rag_documents_source"
        ),
        CheckConstraint(
            "(status = 'ready') = (chunk_count IS NOT NULL) "
            "AND (status <> 'ready' OR chunk_count >= 1)",
            name="ck_rag_documents_ready",
        ),
        CheckConstraint(
            "(status <> 'failed' OR error_code IS NOT NULL) "
            "AND (status <> 'ready' OR error_code IS NULL)",
            name="ck_rag_documents_error",
        ),
        Index("ix_rag_documents_owner_created", "owner_id", created_at.desc(), id.desc()),
        Index(
            "ix_rag_documents_unsettled",
            "owner_id",
            postgresql_where=text("status IN ('pending','processing')"),
        ),
        Index(
            # The reaper expires pending rows by deadline; claims go by primary key.
            "ix_rag_documents_pending_deadline",
            "deadline_at",
            postgresql_where=text("status = 'pending'"),
        ),
        Index(
            "ix_rag_documents_lease",
            "lease_expires_at",
            postgresql_where=text("status = 'processing'"),
        ),
    )


class RAGChunk(Base):
    __tablename__ = "rag_chunks"

    document_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True))
    page: Mapped[int | None] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(VECTOR(RAG_EMBEDDING_DIMENSIONS))

    __table_args__ = (
        ForeignKeyConstraint(
            ["document_id", "owner_id"],
            ["rag_documents.id", "rag_documents.owner_id"],
            ondelete="CASCADE",
            name="fk_rag_chunks_document_owner",
        ),
        CheckConstraint("ordinal >= 0", name="ck_rag_chunks_ordinal"),
        CheckConstraint("page IS NULL OR page >= 1", name="ck_rag_chunks_page"),
        CheckConstraint("vector_norm(embedding) > 0", name="ck_rag_chunks_norm"),
        CheckConstraint("char_length(content) BETWEEN 1 AND 4000", name="ck_rag_chunks_content"),
        Index("ix_rag_chunks_owner_id", "owner_id"),
        Index(
            "ix_rag_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class RAGOutbox(Base):
    __tablename__ = "rag_outbox"

    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("rag_documents.id", ondelete="CASCADE")
    )
    event_kind: Mapped[str] = mapped_column(String(16))
    attempt: Mapped[int] = mapped_column(Integer)
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    lease_token: Mapped[UUID | None] = mapped_column(PostgreSQLUUID(as_uuid=True))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    publish_attempts: Mapped[int] = mapped_column(Integer, server_default="0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint(
            "document_id", "event_kind", "attempt", name="uq_rag_outbox_document_event_attempt"
        ),
        CheckConstraint(
            "event_kind IN ('requested','dead_letter')", name="ck_rag_outbox_event_kind"
        ),
        CheckConstraint("attempt BETWEEN 1 AND 3", name="ck_rag_outbox_attempt"),
        Index(
            "ix_rag_outbox_due",
            "available_at",
            postgresql_where=text("published_at IS NULL"),
        ),
        Index("ix_rag_outbox_document_id", "document_id"),
    )


# Usage log and rate-limit source. Holds no question, passage or answer text.
class RAGQuery(Base):
    __tablename__ = "rag_queries"

    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    owner_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    status: Mapped[str] = mapped_column(String(16), server_default="admitted")
    top_k: Mapped[int] = mapped_column(Integer)
    retrieved_count: Mapped[int | None] = mapped_column(Integer)
    embedding_model: Mapped[str] = mapped_column(String(128))
    model: Mapped[str] = mapped_column(String(128))
    prompt_version: Mapped[str] = mapped_column(String(64))
    response_model: Mapped[str | None] = mapped_column(String(128))
    embedding_tokens: Mapped[int | None] = mapped_column(Integer)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    provider_request_id: Mapped[str | None] = mapped_column(String(128))
    error_code: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint(
            "status IN ('admitted','answered','no_context','failed')",
            name="ck_rag_queries_status",
        ),
        CheckConstraint("top_k BETWEEN 1 AND 8", name="ck_rag_queries_top_k"),
        CheckConstraint(
            "retrieved_count IS NULL OR retrieved_count >= 0", name="ck_rag_queries_retrieved"
        ),
        CheckConstraint(
            "(status = 'admitted') = (finished_at IS NULL)", name="ck_rag_queries_terminal"
        ),
        CheckConstraint(
            "(status = 'failed') = (error_code IS NOT NULL)", name="ck_rag_queries_error"
        ),
        CheckConstraint(
            "(embedding_tokens IS NULL OR embedding_tokens >= 0) "
            "AND (input_tokens IS NULL OR input_tokens >= 0) "
            "AND (output_tokens IS NULL OR output_tokens >= 0)",
            name="ck_rag_queries_tokens",
        ),
        Index("ix_rag_queries_owner_created", "owner_id", "created_at"),
    )
