"""Word and HTML documents: wider content_type and its allowed values.

Revision ID: 0008_document_formats
Revises: 0007_document_versions
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008_document_formats"
down_revision: str | None = "0007_document_versions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Literal copies: a migration must not change when the models do.
OLD_TYPES = "content_type IN ('text/plain','text/markdown','application/pdf')"
NEW_TYPES = (
    "content_type IN ('text/plain','text/markdown','application/pdf',"
    "'application/vnd.openxmlformats-officedocument.wordprocessingml.document','text/html')"
)


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.alter_column("rag_documents", "content_type", type_=sa.String(128), existing_nullable=False)
    op.drop_constraint("ck_rag_documents_content_type", "rag_documents", type_="check")
    op.create_check_constraint("ck_rag_documents_content_type", "rag_documents", NEW_TYPES)


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    blocked = op.get_bind().scalar(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM rag_documents WHERE content_type IN "
            "('application/vnd.openxmlformats-officedocument.wordprocessingml.document',"
            "'text/html'))"
        )
    )
    if blocked:
        raise RuntimeError(
            "Downgrade refused: Word or HTML documents exist. Delete them and wait until "
            "every tombstone is purged."
        )
    op.drop_constraint("ck_rag_documents_content_type", "rag_documents", type_="check")
    op.create_check_constraint("ck_rag_documents_content_type", "rag_documents", OLD_TYPES)
    op.alter_column("rag_documents", "content_type", type_=sa.String(32), existing_nullable=False)
