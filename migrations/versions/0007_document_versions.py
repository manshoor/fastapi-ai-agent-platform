"""Document versions, tombstone deletes and created/updated/deleted events.

Revision ID: 0007_document_versions
Revises: 0006_step_up
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007_document_versions"
down_revision: str | None = "0006_step_up"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Literal copies of the 0003 checks: a migration must not change when the models do.
OLD_STATUS = "status IN ('pending','processing','ready','failed')"
OLD_TERMINAL = "(status IN ('ready','failed')) = (finished_at IS NOT NULL)"
OLD_SOURCE = "(status IN ('ready','failed')) = (source IS NULL)"
OLD_READY = (
    "(status = 'ready') = (chunk_count IS NOT NULL) AND (status <> 'ready' OR chunk_count >= 1)"
)
OLD_KINDS = "event_kind IN ('requested','dead_letter')"

NEW_STATUS = "status IN ('pending','processing','ready','failed','deleted')"
NEW_TERMINAL = "(status IN ('ready','failed','deleted')) = (finished_at IS NOT NULL)"
NEW_SOURCE = "(status IN ('ready','failed','deleted')) = (source IS NULL)"
NEW_READY = (
    "(active_version IS NULL) = (chunk_count IS NULL) "
    "AND (chunk_count IS NULL OR chunk_count >= 1) "
    "AND (status <> 'ready' OR active_version IS NOT NULL) "
    "AND (status NOT IN ('failed','deleted') OR active_version IS NULL)"
)
NEW_KINDS = "event_kind IN ('created','updated','deleted','dead_letter')"


def _replace_check(name: str, table: str, condition: str) -> None:
    op.drop_constraint(name, table, type_="check")
    op.create_check_constraint(name, table, condition)


def upgrade() -> None:
    # Fail fast instead of queueing every API statement behind a blocked ALTER.
    op.execute("SET LOCAL lock_timeout = '5s'")

    # Documents: versions, source identity, update idempotency.
    op.add_column(
        "rag_documents", sa.Column("version", sa.Integer(), nullable=False, server_default="1")
    )
    op.add_column("rag_documents", sa.Column("active_version", sa.Integer(), nullable=True))
    op.add_column(
        "rag_documents",
        sa.Column("source_kind", sa.String(16), nullable=False, server_default="upload"),
    )
    op.add_column("rag_documents", sa.Column("connector_name", sa.String(64), nullable=True))
    op.add_column("rag_documents", sa.Column("external_id", sa.String(256), nullable=True))
    op.add_column("rag_documents", sa.Column("update_error_code", sa.String(64), nullable=True))
    op.add_column("rag_documents", sa.Column("update_key_hash", sa.String(64), nullable=True))
    op.add_column("rag_documents", sa.Column("update_fingerprint", sa.String(64), nullable=True))

    # Backfill: every ready document serves version 1.
    op.execute("UPDATE rag_documents SET active_version = 1 WHERE status = 'ready'")

    _replace_check("ck_rag_documents_status", "rag_documents", NEW_STATUS)
    _replace_check("ck_rag_documents_terminal", "rag_documents", NEW_TERMINAL)
    _replace_check("ck_rag_documents_source", "rag_documents", NEW_SOURCE)
    _replace_check("ck_rag_documents_ready", "rag_documents", NEW_READY)
    op.create_check_constraint(
        "ck_rag_documents_version",
        "rag_documents",
        "version >= 1 AND (active_version IS NULL OR active_version BETWEEN 1 AND version)",
    )
    op.create_check_constraint(
        "ck_rag_documents_source_kind",
        "rag_documents",
        "source_kind IN ('upload','connector') "
        "AND (source_kind = 'connector') = (connector_name IS NOT NULL) "
        "AND (connector_name IS NULL) = (external_id IS NULL)",
    )
    op.create_check_constraint(
        "ck_rag_documents_update_error",
        "rag_documents",
        "update_error_code IS NULL OR (status = 'ready' AND active_version IS NOT NULL)",
    )
    op.create_check_constraint(
        "ck_rag_documents_update_key",
        "rag_documents",
        "(update_key_hash IS NULL) = (update_fingerprint IS NULL)",
    )
    op.create_unique_constraint(
        "uq_rag_documents_connector_item", "rag_documents", ["connector_name", "external_id"]
    )
    op.create_index(
        "ix_rag_documents_tombstone",
        "rag_documents",
        ["finished_at"],
        postgresql_where=sa.text("status = 'deleted'"),
    )

    # Chunks: keyed by version, so a new version's chunks sit beside the active ones.
    op.add_column(
        "rag_chunks", sa.Column("version", sa.Integer(), nullable=False, server_default="1")
    )
    op.alter_column("rag_chunks", "version", server_default=None)
    op.create_check_constraint("ck_rag_chunks_version", "rag_chunks", "version >= 1")
    op.drop_constraint("rag_chunks_pkey", "rag_chunks", type_="primary")
    op.create_primary_key("rag_chunks_pkey", "rag_chunks", ["document_id", "version", "ordinal"])

    # Outbox: created/updated/deleted per version; requested rows were version 1 creates.
    op.add_column(
        "rag_outbox", sa.Column("version", sa.Integer(), nullable=False, server_default="1")
    )
    op.alter_column("rag_outbox", "version", server_default=None)
    op.drop_constraint("ck_rag_outbox_event_kind", "rag_outbox", type_="check")
    op.execute("UPDATE rag_outbox SET event_kind = 'created' WHERE event_kind = 'requested'")
    op.create_check_constraint("ck_rag_outbox_event_kind", "rag_outbox", NEW_KINDS)
    op.create_check_constraint("ck_rag_outbox_version", "rag_outbox", "version >= 1")
    op.drop_constraint("uq_rag_outbox_document_event_attempt", "rag_outbox", type_="unique")
    op.create_unique_constraint(
        "uq_rag_outbox_document_event_attempt",
        "rag_outbox",
        ["document_id", "event_kind", "version", "attempt"],
    )


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")

    # Refuse while any state exists that 0006 cannot represent.
    connection = op.get_bind()
    blocked = connection.scalar(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM rag_documents WHERE version > 1 "
            "OR status = 'deleted' OR source_kind <> 'upload') "
            "OR EXISTS (SELECT 1 FROM rag_outbox WHERE event_kind IN ('updated','deleted'))"
        )
    )
    if blocked:
        raise RuntimeError(
            "Downgrade refused: updated, deleted or connector documents exist. Delete "
            "them and wait until the worker or recovery purges every tombstone."
        )

    op.drop_constraint("uq_rag_outbox_document_event_attempt", "rag_outbox", type_="unique")
    op.create_unique_constraint(
        "uq_rag_outbox_document_event_attempt",
        "rag_outbox",
        ["document_id", "event_kind", "attempt"],
    )
    op.drop_constraint("ck_rag_outbox_version", "rag_outbox", type_="check")
    op.drop_constraint("ck_rag_outbox_event_kind", "rag_outbox", type_="check")
    op.execute("UPDATE rag_outbox SET event_kind = 'requested' WHERE event_kind = 'created'")
    op.create_check_constraint("ck_rag_outbox_event_kind", "rag_outbox", OLD_KINDS)
    op.drop_column("rag_outbox", "version")

    op.drop_constraint("rag_chunks_pkey", "rag_chunks", type_="primary")
    op.create_primary_key("rag_chunks_pkey", "rag_chunks", ["document_id", "ordinal"])
    op.drop_constraint("ck_rag_chunks_version", "rag_chunks", type_="check")
    op.drop_column("rag_chunks", "version")

    op.drop_index("ix_rag_documents_tombstone", table_name="rag_documents")
    op.drop_constraint("uq_rag_documents_connector_item", "rag_documents", type_="unique")
    for name in (
        "ck_rag_documents_update_key",
        "ck_rag_documents_update_error",
        "ck_rag_documents_source_kind",
        "ck_rag_documents_version",
    ):
        op.drop_constraint(name, "rag_documents", type_="check")
    _replace_check("ck_rag_documents_ready", "rag_documents", OLD_READY)
    _replace_check("ck_rag_documents_source", "rag_documents", OLD_SOURCE)
    _replace_check("ck_rag_documents_terminal", "rag_documents", OLD_TERMINAL)
    _replace_check("ck_rag_documents_status", "rag_documents", OLD_STATUS)
    for name in (
        "update_fingerprint",
        "update_key_hash",
        "update_error_code",
        "external_id",
        "connector_name",
        "source_kind",
        "active_version",
        "version",
    ):
        op.drop_column("rag_documents", name)
