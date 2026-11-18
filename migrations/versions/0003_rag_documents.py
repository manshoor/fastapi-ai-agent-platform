"""RAG documents, embedded chunks, ingestion outbox, and query usage log.

Revision ID: 0003_rag_documents
Revises: 0002_ai_rewrite_jobs
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import VECTOR
from sqlalchemy.dialects import postgresql

revision: str = "0003_rag_documents"
down_revision: str | None = "0002_ai_rewrite_jobs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Frozen here on purpose: a migration must not change when app constants change.
EMBEDDING_DIMENSIONS = 1536


def upgrade() -> None:
    # pgvector is not a trusted extension; the compose database role is the superuser.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "rag_documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("content_type", sa.String(32), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("source", sa.LargeBinary()),
        sa.Column("idempotency_key_hash", sa.String(64), nullable=False),
        sa.Column("request_fingerprint", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "next_attempt_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("deadline_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lease_token", postgresql.UUID(as_uuid=True)),
        sa.Column("lease_generation", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("embedding_model", sa.String(128), nullable=False),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=False),
        sa.Column("chunker_version", sa.String(32), nullable=False),
        sa.Column("chunk_count", sa.Integer()),
        sa.Column("page_count", sa.Integer()),
        sa.Column("embedding_tokens", sa.Integer()),
        sa.Column("error_code", sa.String(64)),
        sa.Column(
            "trace_context",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("owner_id", "idempotency_key_hash", name="uq_rag_documents_owner_key"),
        sa.UniqueConstraint("id", "owner_id", name="uq_rag_documents_id_owner"),
        sa.CheckConstraint(
            "status IN ('pending','processing','ready','failed')", name="ck_rag_documents_status"
        ),
        sa.CheckConstraint(
            "content_type IN ('text/plain','text/markdown','application/pdf')",
            name="ck_rag_documents_content_type",
        ),
        sa.CheckConstraint(
            "size_bytes > 0 AND (source IS NULL OR octet_length(source) = size_bytes)",
            name="ck_rag_documents_size",
        ),
        # Only a processing row holds a lease, and it always has one.
        sa.CheckConstraint(
            "(status = 'processing') = (lease_token IS NOT NULL AND lease_expires_at IS NOT NULL)",
            name="ck_rag_documents_lease",
        ),
        sa.CheckConstraint(
            "(chunk_count IS NULL OR chunk_count >= 0) AND (page_count IS NULL OR page_count >= 1) "
            "AND (embedding_tokens IS NULL OR embedding_tokens >= 0)",
            name="ck_rag_documents_counts",
        ),
        sa.CheckConstraint("attempt BETWEEN 1 AND 3", name="ck_rag_documents_attempt"),
        sa.CheckConstraint("lease_generation >= 0", name="ck_rag_documents_generation"),
        sa.CheckConstraint(
            f"embedding_dimensions = {EMBEDDING_DIMENSIONS}", name="ck_rag_documents_dimensions"
        ),
        sa.CheckConstraint(
            "(status IN ('ready','failed')) = (finished_at IS NOT NULL)",
            name="ck_rag_documents_terminal",
        ),
        sa.CheckConstraint(
            "(status IN ('ready','failed')) = (source IS NULL)", name="ck_rag_documents_source"
        ),
        sa.CheckConstraint(
            "(status = 'ready') = (chunk_count IS NOT NULL) "
            "AND (status <> 'ready' OR chunk_count >= 1)",
            name="ck_rag_documents_ready",
        ),
        sa.CheckConstraint(
            "(status <> 'failed' OR error_code IS NOT NULL) "
            "AND (status <> 'ready' OR error_code IS NULL)",
            name="ck_rag_documents_error",
        ),
    )
    op.create_index(
        "ix_rag_documents_owner_created",
        "rag_documents",
        ["owner_id", sa.text("created_at DESC"), sa.text("id DESC")],
    )
    op.create_index(
        "ix_rag_documents_unsettled",
        "rag_documents",
        ["owner_id"],
        postgresql_where=sa.text("status IN ('pending','processing')"),
    )
    op.create_index(
        "ix_rag_documents_pending_deadline",
        "rag_documents",
        ["deadline_at"],
        postgresql_where=sa.text("status = 'pending'"),
    )
    op.create_index(
        "ix_rag_documents_lease",
        "rag_documents",
        ["lease_expires_at"],
        postgresql_where=sa.text("status = 'processing'"),
    )

    op.create_table(
        "rag_chunks",
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("page", sa.Integer()),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", VECTOR(EMBEDDING_DIMENSIONS), nullable=False),
        sa.PrimaryKeyConstraint("document_id", "ordinal"),
        sa.ForeignKeyConstraint(
            ["document_id", "owner_id"],
            ["rag_documents.id", "rag_documents.owner_id"],
            ondelete="CASCADE",
            name="fk_rag_chunks_document_owner",
        ),
        sa.CheckConstraint("ordinal >= 0", name="ck_rag_chunks_ordinal"),
        sa.CheckConstraint("page IS NULL OR page >= 1", name="ck_rag_chunks_page"),
        sa.CheckConstraint("vector_norm(embedding) > 0", name="ck_rag_chunks_norm"),
        sa.CheckConstraint("char_length(content) BETWEEN 1 AND 4000", name="ck_rag_chunks_content"),
    )
    op.create_index("ix_rag_chunks_owner_id", "rag_chunks", ["owner_id"])
    op.create_index(
        "ix_rag_chunks_embedding_hnsw",
        "rag_chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    op.create_table(
        "rag_outbox",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("event_kind", sa.String(16), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column(
            "available_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("lease_token", postgresql.UUID(as_uuid=True)),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.Column("last_published_at", sa.DateTime(timezone=True)),
        sa.Column("publish_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(["document_id"], ["rag_documents.id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "document_id", "event_kind", "attempt", name="uq_rag_outbox_document_event_attempt"
        ),
        sa.CheckConstraint(
            "event_kind IN ('requested','dead_letter')", name="ck_rag_outbox_event_kind"
        ),
        sa.CheckConstraint("attempt BETWEEN 1 AND 3", name="ck_rag_outbox_attempt"),
    )
    op.create_index(
        "ix_rag_outbox_due",
        "rag_outbox",
        ["available_at"],
        postgresql_where=sa.text("published_at IS NULL"),
    )
    op.create_index("ix_rag_outbox_document_id", "rag_outbox", ["document_id"])

    op.create_table(
        "rag_queries",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="admitted"),
        sa.Column("top_k", sa.Integer(), nullable=False),
        sa.Column("retrieved_count", sa.Integer()),
        sa.Column("embedding_model", sa.String(128), nullable=False),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("prompt_version", sa.String(64), nullable=False),
        sa.Column("response_model", sa.String(128)),
        sa.Column("embedding_tokens", sa.Integer()),
        sa.Column("input_tokens", sa.Integer()),
        sa.Column("output_tokens", sa.Integer()),
        sa.Column("provider_request_id", sa.String(128)),
        sa.Column("error_code", sa.String(64)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="CASCADE"),
        sa.CheckConstraint(
            "status IN ('admitted','answered','no_context','failed')",
            name="ck_rag_queries_status",
        ),
        sa.CheckConstraint("top_k BETWEEN 1 AND 8", name="ck_rag_queries_top_k"),
        sa.CheckConstraint(
            "retrieved_count IS NULL OR retrieved_count >= 0", name="ck_rag_queries_retrieved"
        ),
        sa.CheckConstraint(
            "(status = 'admitted') = (finished_at IS NULL)", name="ck_rag_queries_terminal"
        ),
        sa.CheckConstraint(
            "(status = 'failed') = (error_code IS NOT NULL)", name="ck_rag_queries_error"
        ),
        sa.CheckConstraint(
            "(embedding_tokens IS NULL OR embedding_tokens >= 0) "
            "AND (input_tokens IS NULL OR input_tokens >= 0) "
            "AND (output_tokens IS NULL OR output_tokens >= 0)",
            name="ck_rag_queries_tokens",
        ),
    )
    op.create_index("ix_rag_queries_owner_created", "rag_queries", ["owner_id", "created_at"])


def downgrade() -> None:
    # The vector extension stays: other databases objects may use it, and dropping
    # it needs the same superuser right that created it.
    op.drop_table("rag_queries")
    op.drop_table("rag_outbox")
    op.drop_table("rag_chunks")
    op.drop_table("rag_documents")
