"""Connector sync runs; a tombstone frees its connector record id.

Revision ID: 0009_connectors
Revises: 0008_document_formats
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0009_connectors"
down_revision: str | None = "0008_document_formats"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.drop_constraint("uq_rag_documents_connector_item", "rag_documents", type_="unique")
    op.create_index(
        "uq_rag_documents_connector_item",
        "rag_documents",
        ["connector_name", "external_id"],
        unique=True,
        postgresql_where=sa.text("connector_name IS NOT NULL AND status <> 'deleted'"),
    )
    op.create_table(
        "rag_connector_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("connector_name", sa.String(64), nullable=False),
        sa.Column(
            "started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("status", sa.String(16), nullable=False, server_default="running"),
        *(
            sa.Column(name, sa.Integer(), nullable=False, server_default="0")
            for name in ("fetched", "created", "updated", "deleted", "unchanged", "skipped")
        ),
        sa.Column("error_code", sa.String(64)),
        sa.CheckConstraint(
            "status IN ('running','complete','partial','failed')",
            name="ck_rag_connector_runs_status",
        ),
        sa.CheckConstraint(
            "(status = 'running') = (finished_at IS NULL)", name="ck_rag_connector_runs_finished"
        ),
        sa.CheckConstraint(
            "fetched >= 0 AND created >= 0 AND updated >= 0 AND deleted >= 0 "
            "AND unchanged >= 0 AND skipped >= 0",
            name="ck_rag_connector_runs_counts",
        ),
    )
    op.create_index(
        "ix_rag_connector_runs_connector",
        "rag_connector_runs",
        ["connector_name", sa.text("started_at DESC")],
    )


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.drop_index("ix_rag_connector_runs_connector", table_name="rag_connector_runs")
    op.drop_table("rag_connector_runs")
    op.drop_index("uq_rag_documents_connector_item", table_name="rag_documents")
    # Fails if a tombstone and a live row share a record id; purge tombstones first.
    op.create_unique_constraint(
        "uq_rag_documents_connector_item", "rag_documents", ["connector_name", "external_id"]
    )
