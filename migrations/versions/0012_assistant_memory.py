"""Assistant memory: sealed notes per user, a one-day write log, and a recall count.

    assistant_memories        id -> user, scope, building (residence only), sealed text,
                              expires_at; served only before expiry and, for residence,
                              while the user lives in that building
    assistant_memory_writes   one row per write, for the daily write limit
    assistant_turns.memory_count

Revision ID: 0012_assistant_memory
Revises: 0011_rag_graph
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0012_assistant_memory"
down_revision: str | None = "0011_rag_graph"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Frozen copies: a migration never follows later code changes.
SCOPES = "'resident','residence'"
MAX_CIPHERTEXT_BYTES = 1216


def _user() -> sa.Column:
    return sa.Column(
        "user_id",
        postgresql.UUID(as_uuid=True),
        sa.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )


def _created_at() -> sa.Column:
    return sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        server_default=sa.text("clock_timestamp()"),
        nullable=False,
    )


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.create_table(
        "assistant_memories",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        _user(),
        sa.Column("scope", sa.String(16), nullable=False),
        sa.Column(
            "building_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("buildings.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("ciphertext", sa.LargeBinary(), nullable=False),
        sa.Column("nonce", sa.LargeBinary(), nullable=False),
        sa.Column("key_version", sa.SmallInteger(), server_default="1", nullable=False),
        _created_at(),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(f"scope IN ({SCOPES})", name="ck_assistant_memories_scope"),
        sa.CheckConstraint(
            "(scope = 'residence') = (building_id IS NOT NULL)",
            name="ck_assistant_memories_building",
        ),
        sa.CheckConstraint("octet_length(nonce) = 12", name="ck_assistant_memories_nonce"),
        sa.CheckConstraint(
            f"octet_length(ciphertext) BETWEEN 19 AND {MAX_CIPHERTEXT_BYTES}",
            name="ck_assistant_memories_ciphertext",
        ),
        sa.CheckConstraint("key_version = 1", name="ck_assistant_memories_key_version"),
        sa.CheckConstraint("expires_at > created_at", name="ck_assistant_memories_expiry"),
    )
    op.create_index(
        "ix_assistant_memories_user_expires", "assistant_memories", ["user_id", "expires_at"]
    )
    op.create_index("ix_assistant_memories_expires", "assistant_memories", ["expires_at"])
    op.create_table(
        "assistant_memory_writes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        _user(),
        _created_at(),
    )
    op.create_index(
        "ix_assistant_memory_writes_user_created",
        "assistant_memory_writes",
        ["user_id", "created_at"],
    )
    op.create_index("ix_assistant_memory_writes_created", "assistant_memory_writes", ["created_at"])
    op.add_column(
        "assistant_turns",
        sa.Column("memory_count", sa.Integer(), server_default="0", nullable=False),
    )
    op.create_check_constraint(
        "ck_assistant_turns_memory_count", "assistant_turns", "memory_count >= 0"
    )


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    # Memories are the residents' own data: never drop them silently. The lock keeps a
    # write from landing between the check and the drop.
    op.execute("LOCK TABLE assistant_memories IN ACCESS EXCLUSIVE MODE")
    if op.get_bind().scalar(sa.text("SELECT EXISTS (SELECT 1 FROM assistant_memories)")):
        raise RuntimeError(
            "Downgrade refused: assistant memories exist. They are residents' data. Stop the "
            "API, and only if losing them is intended, run DELETE FROM assistant_memories; "
            "then rerun the downgrade."
        )
    op.drop_constraint("ck_assistant_turns_memory_count", "assistant_turns", type_="check")
    op.drop_column("assistant_turns", "memory_count")
    op.drop_table("assistant_memory_writes")
    op.drop_table("assistant_memories")
