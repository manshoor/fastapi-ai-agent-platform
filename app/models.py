"""Persistent account and owner-scoped item schema."""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


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
