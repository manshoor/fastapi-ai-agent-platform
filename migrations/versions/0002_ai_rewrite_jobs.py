"""Durable owner-scoped rewrite jobs, publication outbox, and admission lock.

Revision ID: 0002_ai_rewrite_jobs
Revises: 0001_users_items
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002_ai_rewrite_jobs"
down_revision: str | None = "0001_users_items"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "ai_admission",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.CheckConstraint("id = 1", name="ck_ai_admission_singleton"),
    )
    op.execute("INSERT INTO ai_admission (id) VALUES (1)")
    op.create_table(
        "ai_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("item_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_item_updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source_title", sa.String(200)),
        sa.Column("source_description", sa.String(5000)),
        sa.Column("instruction", sa.String(1000)),
        sa.Column("idempotency_key_hash", sa.String(64), nullable=False),
        sa.Column("request_fingerprint", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
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
        sa.Column("provider_started_at", sa.DateTime(timezone=True)),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("prompt_version", sa.String(64), nullable=False),
        sa.Column("schema_version", sa.String(16), nullable=False),
        sa.Column("result_title", sa.String(200)),
        sa.Column("result_description", sa.String(5000)),
        sa.Column("response_model", sa.String(128)),
        sa.Column("input_tokens", sa.Integer()),
        sa.Column("output_tokens", sa.Integer()),
        sa.Column("provider_request_id", sa.String(128)),
        sa.Column("error_code", sa.String(64)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("content_expires_at", sa.DateTime(timezone=True)),
        sa.Column("content_purged_at", sa.DateTime(timezone=True)),
        sa.Column("admission_released_at", sa.DateTime(timezone=True)),
        sa.Column(
            "trace_context",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("owner_id", "idempotency_key_hash", name="uq_ai_jobs_owner_key"),
        sa.CheckConstraint(
            "status IN ('pending','leased','calling','succeeded','failed','needs_reconciliation')",
            name="ck_ai_jobs_status",
        ),
        sa.CheckConstraint("attempt BETWEEN 1 AND 3", name="ck_ai_jobs_attempt"),
        sa.CheckConstraint("lease_generation >= 0", name="ck_ai_jobs_generation"),
        sa.CheckConstraint(
            "(status IN ('succeeded','failed','needs_reconciliation')) = "
            "(finished_at IS NOT NULL AND content_expires_at IS NOT NULL)",
            name="ck_ai_jobs_terminal",
        ),
        sa.CheckConstraint(
            "(status = 'succeeded' AND ((result_title IS NOT NULL AND "
            "result_description IS NOT NULL) OR content_purged_at IS NOT NULL)) OR "
            "(status <> 'succeeded' AND result_title IS NULL AND result_description IS NULL)",
            name="ck_ai_jobs_result",
        ),
    )
    op.create_index("ix_ai_jobs_owner_id", "ai_jobs", ["owner_id", "id"])
    op.create_index(
        "ix_ai_jobs_admission",
        "ai_jobs",
        ["owner_id"],
        postgresql_where=sa.text("admission_released_at IS NULL"),
    )
    op.create_index(
        "ix_ai_jobs_due",
        "ai_jobs",
        ["next_attempt_at"],
        postgresql_where=sa.text("status = 'pending'"),
    )
    op.create_index(
        "ix_ai_jobs_lease",
        "ai_jobs",
        ["lease_expires_at"],
        postgresql_where=sa.text("status IN ('leased','calling')"),
    )
    op.create_index("ix_ai_jobs_content_expiry", "ai_jobs", ["content_expires_at"])
    op.create_table(
        "ai_outbox",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
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
        sa.ForeignKeyConstraint(["job_id"], ["ai_jobs.id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "job_id", "event_kind", "attempt", name="uq_ai_outbox_job_event_attempt"
        ),
        sa.CheckConstraint(
            "event_kind IN ('requested','dead_letter')", name="ck_ai_outbox_event_kind"
        ),
        sa.CheckConstraint("attempt BETWEEN 1 AND 3", name="ck_ai_outbox_attempt"),
    )
    op.create_index(
        "ix_ai_outbox_due",
        "ai_outbox",
        ["available_at"],
        postgresql_where=sa.text("published_at IS NULL"),
    )
    op.create_index("ix_ai_outbox_job_id", "ai_outbox", ["job_id"])


def downgrade() -> None:
    op.drop_table("ai_outbox")
    op.drop_table("ai_jobs")
    op.drop_table("ai_admission")
