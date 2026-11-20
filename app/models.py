"""Persistent account and owner-scoped item schema."""

from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    Numeric,
    SmallInteger,
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
    # Set for building documents a manager uploads; residents of that building search them.
    building_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("buildings.id", ondelete="CASCADE")
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
        Index(
            "ix_rag_documents_building",
            "building_id",
            postgresql_where=text("building_id IS NOT NULL"),
        ),
    )


# Why a query or turn ended without an answer; fixed codes only, never content.
RAG_NO_CONTEXT_REASONS = (
    "no_chunks",
    "no_passages",
    "no_claims",
    "quote_too_short",
    "quote_not_found",
    "number_not_in_passage",
    "rerank_empty",
    "verifier_rejected",
)
CLAIM_FAILURES = (
    "quote_too_short",
    "quote_not_found",
    "number_not_in_passage",
    "tool_claim_ungrounded",
)
REASONING_EFFORTS = ("none", "low", "medium")


def _sql_in(values: tuple[str, ...]) -> str:
    return ",".join(f"'{value}'" for value in values)


class RAGChunk(Base):
    __tablename__ = "rag_chunks"

    document_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True))
    page: Mapped[int | None] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    # Markdown heading path such as "Pets > Dogs"; null for text, PDF and older chunks.
    heading: Mapped[str | None] = mapped_column(Text)
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
        CheckConstraint(
            "heading IS NULL OR char_length(heading) BETWEEN 1 AND 300",
            name="ck_rag_chunks_heading",
        ),
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
    no_context_reason: Mapped[str | None] = mapped_column(String(32))
    reasoning_effort: Mapped[str] = mapped_column(String(16), server_default="none")
    # Completed chat calls; input_tokens and output_tokens are their sums.
    chat_calls: Mapped[int | None] = mapped_column(SmallInteger)
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
        CheckConstraint(
            "no_context_reason IS NULL OR (status = 'no_context' "
            f"AND no_context_reason IN ({_sql_in(RAG_NO_CONTEXT_REASONS)}))",
            name="ck_rag_queries_no_context_reason",
        ),
        CheckConstraint(
            f"reasoning_effort IN ({_sql_in(REASONING_EFFORTS)})",
            name="ck_rag_queries_reasoning_effort",
        ),
        CheckConstraint("chat_calls IS NULL OR chat_calls >= 0", name="ck_rag_queries_chat_calls"),
        Index("ix_rag_queries_owner_created", "owner_id", "created_at"),
    )


# Resident assistant: buildings, residences, simulated devices, and the agent's audit trail.
# A row's apartment must belong to its building: composite FKs target (id, building_id).
_APARTMENT_FK_COLUMNS = ["apartments.id", "apartments.building_id"]


class Building(Base):
    __tablename__ = "buildings"

    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (CheckConstraint("char_length(name) >= 1", name="ck_buildings_name"),)


class Apartment(Base):
    __tablename__ = "apartments"

    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    building_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("buildings.id", ondelete="CASCADE")
    )
    label: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("building_id", "label", name="uq_apartments_building_label"),
        UniqueConstraint("id", "building_id", name="uq_apartments_id_building"),
        CheckConstraint("char_length(label) >= 1", name="ck_apartments_label"),
    )


class Membership(Base):
    __tablename__ = "memberships"

    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    building_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("buildings.id", ondelete="CASCADE")
    )
    # Residents live in one apartment; managers act for the whole building.
    apartment_id: Mapped[UUID | None] = mapped_column(PostgreSQLUUID(as_uuid=True))
    role: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        ForeignKeyConstraint(
            ["apartment_id", "building_id"],
            _APARTMENT_FK_COLUMNS,
            ondelete="CASCADE",
            name="fk_memberships_apartment_building",
        ),
        CheckConstraint("role IN ('resident','manager')", name="ck_memberships_role"),
        CheckConstraint(
            "(role = 'resident') = (apartment_id IS NOT NULL)", name="ck_memberships_apartment"
        ),
        UniqueConstraint(
            "user_id", "building_id", "role", name="uq_memberships_user_building_role"
        ),
        Index(
            "uq_memberships_one_residence",
            "user_id",
            unique=True,
            postgresql_where=text("role = 'resident'"),
        ),
    )


class Device(Base):
    """Simulated device: its state is this row. Exactly one state column fits each kind."""

    __tablename__ = "devices"

    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    apartment_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True))
    building_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True))
    name: Mapped[str] = mapped_column(String(100))
    kind: Mapped[str] = mapped_column(String(16))
    target_celsius: Mapped[Decimal | None] = mapped_column(Numeric(3, 1))
    locked: Mapped[bool | None] = mapped_column(Boolean)
    armed: Mapped[bool | None] = mapped_column(Boolean)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        ForeignKeyConstraint(
            ["apartment_id", "building_id"],
            _APARTMENT_FK_COLUMNS,
            ondelete="CASCADE",
            name="fk_devices_apartment_building",
        ),
        UniqueConstraint("apartment_id", "name", name="uq_devices_apartment_name"),
        # Target of the pending_actions composite FK: an action's device is in its apartment.
        UniqueConstraint("id", "apartment_id", name="uq_devices_id_apartment"),
        CheckConstraint("kind IN ('thermostat','door_lock','alarm')", name="ck_devices_kind"),
        CheckConstraint(
            "(kind = 'thermostat') = (target_celsius IS NOT NULL) "
            "AND (kind = 'door_lock') = (locked IS NOT NULL) "
            "AND (kind = 'alarm') = (armed IS NOT NULL)",
            name="ck_devices_state",
        ),
        # Hard physical bound; the assistant's own limit is narrower.
        CheckConstraint(
            "target_celsius IS NULL OR target_celsius BETWEEN 5 AND 35", name="ck_devices_target"
        ),
        CheckConstraint("char_length(name) >= 1", name="ck_devices_name"),
    )


class Booking(Base):
    __tablename__ = "bookings"

    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    apartment_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True))
    building_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True))
    user_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    amenity: Mapped[str] = mapped_column(String(100))
    starts_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ends_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        ForeignKeyConstraint(
            ["apartment_id", "building_id"],
            _APARTMENT_FK_COLUMNS,
            ondelete="CASCADE",
            name="fk_bookings_apartment_building",
        ),
        CheckConstraint("starts_at < ends_at", name="ck_bookings_period"),
        CheckConstraint("char_length(amenity) >= 1", name="ck_bookings_amenity"),
        Index("ix_bookings_apartment_starts", "apartment_id", "starts_at"),
    )


class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    apartment_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True))
    building_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True))
    user_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    category: Mapped[str] = mapped_column(String(16))
    summary: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(16), server_default="open")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        ForeignKeyConstraint(
            ["apartment_id", "building_id"],
            _APARTMENT_FK_COLUMNS,
            ondelete="CASCADE",
            name="fk_tickets_apartment_building",
        ),
        CheckConstraint(
            "category IN ('plumbing','electrical','heating','appliance','other')",
            name="ck_tickets_category",
        ),
        CheckConstraint("status IN ('open','closed')", name="ck_tickets_status"),
        CheckConstraint("char_length(summary) >= 1", name="ck_tickets_summary"),
        Index("ix_tickets_apartment_created", "apartment_id", "created_at"),
        Index("ix_tickets_user_created", "user_id", "created_at"),
    )


# Usage log and rate-limit source for assistant turns. Holds no message or answer text.
class AssistantTurn(Base):
    __tablename__ = "assistant_turns"

    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    status: Mapped[str] = mapped_column(String(16), server_default="admitted")
    model: Mapped[str] = mapped_column(String(128))
    prompt_version: Mapped[str] = mapped_column(String(64))
    model_calls: Mapped[int] = mapped_column(Integer, server_default="0")
    tool_calls: Mapped[int] = mapped_column(Integer, server_default="0")
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    embedding_tokens: Mapped[int | None] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(String(64))
    claim_failure: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint(
            "status IN ('admitted','answered','no_context','incomplete','failed')",
            name="ck_assistant_turns_status",
        ),
        CheckConstraint(
            "(status = 'admitted') = (finished_at IS NULL)", name="ck_assistant_turns_terminal"
        ),
        CheckConstraint(
            "(status IN ('incomplete','failed')) = (error_code IS NOT NULL)",
            name="ck_assistant_turns_error",
        ),
        CheckConstraint(
            "model_calls >= 0 AND tool_calls >= 0 "
            "AND (input_tokens IS NULL OR input_tokens >= 0) "
            "AND (output_tokens IS NULL OR output_tokens >= 0) "
            "AND (embedding_tokens IS NULL OR embedding_tokens >= 0)",
            name="ck_assistant_turns_counts",
        ),
        CheckConstraint(
            "claim_failure IS NULL OR (status IN ('no_context','incomplete') "
            f"AND claim_failure IN ({_sql_in(CLAIM_FAILURES)}))",
            name="ck_assistant_turns_claim_failure",
        ),
        Index("ix_assistant_turns_user_created", "user_id", "created_at"),
    )


class PendingAction(Base):
    """A sensitive action the model asked for. Only the resident's confirm request runs it."""

    __tablename__ = "pending_actions"

    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    apartment_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True))
    building_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True))
    device_id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True))
    turn_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("assistant_turns.id", ondelete="SET NULL")
    )
    action: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), server_default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        ForeignKeyConstraint(
            ["apartment_id", "building_id"],
            _APARTMENT_FK_COLUMNS,
            ondelete="CASCADE",
            name="fk_pending_actions_apartment_building",
        ),
        ForeignKeyConstraint(
            ["device_id", "apartment_id"],
            ["devices.id", "devices.apartment_id"],
            ondelete="CASCADE",
            name="fk_pending_actions_device_apartment",
        ),
        CheckConstraint("action IN ('unlock_door')", name="ck_pending_actions_action"),
        CheckConstraint(
            "status IN ('pending','confirmed','expired')", name="ck_pending_actions_status"
        ),
        CheckConstraint(
            "(status = 'pending') = (decided_at IS NULL)", name="ck_pending_actions_decided"
        ),
        CheckConstraint("expires_at > created_at", name="ck_pending_actions_expiry"),
        Index(
            "uq_pending_actions_device",
            "device_id",
            unique=True,
            postgresql_where=text("status = 'pending'"),
        ),
        Index("ix_pending_actions_user_created", "user_id", "created_at"),
    )


# Audit trail of every tool call and confirmation. Typed values only, never free text.
class AssistantToolCall(Base):
    __tablename__ = "assistant_tool_calls"

    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    turn_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("assistant_turns.id", ondelete="CASCADE")
    )
    user_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    apartment_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("apartments.id", ondelete="SET NULL")
    )
    source: Mapped[str] = mapped_column(String(16))
    tool: Mapped[str] = mapped_column(String(32))
    decision: Mapped[str] = mapped_column(String(24))
    result_code: Mapped[str] = mapped_column(String(32))
    device_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("devices.id", ondelete="SET NULL")
    )
    ticket_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("tickets.id", ondelete="SET NULL")
    )
    pending_action_id: Mapped[UUID | None] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("pending_actions.id", ondelete="SET NULL")
    )
    details: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "source IN ('assistant','confirmation')", name="ck_assistant_tool_calls_source"
        ),
        CheckConstraint(
            "(source = 'assistant') = (turn_id IS NOT NULL)", name="ck_assistant_tool_calls_turn"
        ),
        CheckConstraint(
            "tool IN ('search_documents','list_bookings','create_ticket','set_thermostat',"
            "'lock_door','unlock_door','unknown')",
            name="ck_assistant_tool_calls_tool",
        ),
        CheckConstraint(
            "decision IN ('allowed','denied','pending_confirmation','confirmed')",
            name="ck_assistant_tool_calls_decision",
        ),
        CheckConstraint(
            "result_code IN ('ok','duplicate','already_pending','not_found','not_permitted',"
            "'invalid_arguments','unknown_tool','limit_exceeded','no_documents',"
            "'step_up_required','step_up_failed')",
            name="ck_assistant_tool_calls_result",
        ),
        # Typed values only: any other key (a question, a summary) is rejected.
        CheckConstraint(
            "jsonb_typeof(details) = 'object' "
            "AND (details - ARRAY['celsius','category','passage_count']) = '{}'::jsonb",
            name="ck_assistant_tool_calls_details",
        ),
        Index("ix_assistant_tool_calls_user_created", "user_id", "created_at"),
        Index("ix_assistant_tool_calls_turn", "turn_id"),
    )


class TotpCredential(Base):
    """One authenticator secret per user, sealed by the API (AES-256-GCM)."""

    __tablename__ = "totp_credentials"

    user_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    secret_ciphertext: Mapped[bytes] = mapped_column(LargeBinary)
    nonce: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # Set together by the first matching code; last_used_step stops a code's reuse.
    enabled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_used_step: Mapped[int | None] = mapped_column(BigInteger)

    __table_args__ = (
        CheckConstraint("octet_length(nonce) = 12", name="ck_totp_credentials_nonce"),
        # A 20-byte secret plus the 16-byte tag; room for longer secrets later.
        CheckConstraint(
            "octet_length(secret_ciphertext) BETWEEN 17 AND 64", name="ck_totp_credentials_secret"
        ),
        CheckConstraint(
            "(enabled_at IS NULL) = (last_used_step IS NULL)", name="ck_totp_credentials_enabled"
        ),
    )


class AuthFailure(Base):
    """One failed password or code check, counted per user over a sliding window."""

    __tablename__ = "auth_failures"

    id: Mapped[UUID] = mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    kind: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("clock_timestamp()")
    )

    __table_args__ = (
        CheckConstraint("kind IN ('password','totp')", name="ck_auth_failures_kind"),
        Index("ix_auth_failures_user_kind_created", "user_id", "kind", "created_at"),
    )
