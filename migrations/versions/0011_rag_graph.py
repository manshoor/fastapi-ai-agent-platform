"""Knowledge graph rows per chunk, and why a version's graph is missing or partial.

    rag_graph_nodes   (document, version, ordinal, node_key) -> type, name, match_name
    rag_graph_edges   (document, version, ordinal, source, relation, target) -> quote
    rag_documents.graph_error_code

Both tables cascade from their chunk, so the version swap, retries and purges that
delete chunks delete the graph with them.

Revision ID: 0011_rag_graph
Revises: 0010_event_sinks
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0011_rag_graph"
down_revision: str | None = "0010_event_sinks"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Frozen copies: a migration never follows later code changes.
NODE_TYPES = (
    "building",
    "facility",
    "device",
    "service",
    "rule",
    "schedule",
    "contact_role",
    "location",
    "document_topic",
)
RELATIONS = (
    "located_in",
    "part_of",
    "operated_by",
    "governed_by",
    "scheduled_on",
    "contact_for",
    "requires",
    "prohibits",
    "allows",
    "replaces",
)
GRAPH_ERRORS = ("graph_rejected", "graph_invalid", "graph_incomplete")


def _in(values: tuple[str, ...]) -> str:
    return ",".join(f"'{value}'" for value in values)


def _chunk_keys() -> list:
    return [
        sa.Column("document_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("version", sa.Integer(), primary_key=True),
        sa.Column("ordinal", sa.Integer(), primary_key=True),
    ]


def _foreign_keys(table: str) -> list:
    return [
        sa.ForeignKeyConstraint(
            ["document_id", "version", "ordinal"],
            ["rag_chunks.document_id", "rag_chunks.version", "rag_chunks.ordinal"],
            ondelete="CASCADE",
            name=f"fk_{table}_chunk",
        ),
        sa.ForeignKeyConstraint(
            ["document_id", "owner_id"],
            ["rag_documents.id", "rag_documents.owner_id"],
            ondelete="CASCADE",
            name=f"fk_{table}_document_owner",
        ),
    ]


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.add_column("rag_documents", sa.Column("graph_error_code", sa.String(64)))
    op.create_check_constraint(
        "ck_rag_documents_graph_error",
        "rag_documents",
        f"graph_error_code IS NULL OR graph_error_code IN ({_in(GRAPH_ERRORS)})",
    )
    op.create_table(
        "rag_graph_nodes",
        *_chunk_keys(),
        sa.Column("node_key", sa.String(160), primary_key=True),
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("node_type", sa.String(32), nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("match_name", sa.String(160), nullable=False),
        *_foreign_keys("rag_graph_nodes"),
        sa.CheckConstraint(f"node_type IN ({_in(NODE_TYPES)})", name="ck_rag_graph_nodes_type"),
        sa.CheckConstraint("char_length(name) BETWEEN 1 AND 120", name="ck_rag_graph_nodes_name"),
    )
    op.create_index("ix_rag_graph_nodes_match", "rag_graph_nodes", ["match_name"])
    op.create_table(
        "rag_graph_edges",
        *_chunk_keys(),
        sa.Column("source_key", sa.String(160), primary_key=True),
        sa.Column("relation", sa.String(32), primary_key=True),
        sa.Column("target_key", sa.String(160), primary_key=True),
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("quote", sa.String(300), nullable=False),
        *_foreign_keys("rag_graph_edges"),
        sa.CheckConstraint(f"relation IN ({_in(RELATIONS)})", name="ck_rag_graph_edges_relation"),
        sa.CheckConstraint("char_length(quote) BETWEEN 1 AND 300", name="ck_rag_graph_edges_quote"),
        sa.CheckConstraint("source_key <> target_key", name="ck_rag_graph_edges_distinct"),
    )
    op.create_index("ix_rag_graph_edges_source", "rag_graph_edges", ["source_key"])
    op.create_index("ix_rag_graph_edges_target", "rag_graph_edges", ["target_key"])


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    blocked = op.get_bind().scalar(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM rag_graph_nodes) "
            "OR EXISTS (SELECT 1 FROM rag_documents WHERE graph_error_code IS NOT NULL)"
        )
    )
    if blocked:
        raise RuntimeError(
            "Downgrade refused: knowledge graph rows exist. Turn RAG_GRAPH_ENABLED off and "
            "update or delete the documents that have a graph first."
        )
    op.drop_table("rag_graph_edges")
    op.drop_table("rag_graph_nodes")
    op.drop_constraint("ck_rag_documents_graph_error", "rag_documents", type_="check")
    op.drop_column("rag_documents", "graph_error_code")
