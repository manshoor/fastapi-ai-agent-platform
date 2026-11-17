"""Resident assistant: buildings, residences, simulated devices, turns, and tool audit.

Revision ID: 0004_resident_assistant
Revises: 0003_rag_documents
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_resident_assistant"
down_revision: str | None = "0003_rag_documents"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

APARTMENT = ["apartments.id", "apartments.building_id"]


def _uuid(name: str, *, nullable: bool = False) -> sa.Column:
    return sa.Column(name, postgresql.UUID(as_uuid=True), nullable=nullable)


def _timestamp(name: str) -> sa.Column:
    return sa.Column(name, sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())


def _apartment_fk(table: str) -> sa.ForeignKeyConstraint:
    # A row's apartment must belong to the row's building.
    return sa.ForeignKeyConstraint(
        ["apartment_id", "building_id"],
        APARTMENT,
        ondelete="CASCADE",
        name=f"fk_{table}_apartment_building",
    )


def upgrade() -> None:
    # Fail fast instead of queueing every API statement behind a blocked ALTER.
    op.execute("SET LOCAL lock_timeout = '5s'")

    op.create_table(
        "buildings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        _timestamp("created_at"),
        sa.CheckConstraint("char_length(name) >= 1", name="ck_buildings_name"),
    )
    op.create_table(
        "apartments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        _uuid("building_id"),
        sa.Column("label", sa.String(32), nullable=False),
        _timestamp("created_at"),
        sa.ForeignKeyConstraint(["building_id"], ["buildings.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("building_id", "label", name="uq_apartments_building_label"),
        sa.UniqueConstraint("id", "building_id", name="uq_apartments_id_building"),
        sa.CheckConstraint("char_length(label) >= 1", name="ck_apartments_label"),
    )

    # Membership: residents live in one apartment; managers act for a whole building.
    op.create_table(
        "memberships",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        _uuid("user_id"),
        _uuid("building_id"),
        _uuid("apartment_id", nullable=True),
        sa.Column("role", sa.String(16), nullable=False),
        _timestamp("created_at"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["building_id"], ["buildings.id"], ondelete="CASCADE"),
        _apartment_fk("memberships"),
        sa.CheckConstraint("role IN ('resident','manager')", name="ck_memberships_role"),
        sa.CheckConstraint(
            "(role = 'resident') = (apartment_id IS NOT NULL)", name="ck_memberships_apartment"
        ),
        sa.UniqueConstraint(
            "user_id", "building_id", "role", name="uq_memberships_user_building_role"
        ),
    )
    op.create_index(
        "uq_memberships_one_residence",
        "memberships",
        ["user_id"],
        unique=True,
        postgresql_where=sa.text("role = 'resident'"),
    )

    # Simulated devices: the row is the device state.
    op.create_table(
        "devices",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        _uuid("apartment_id"),
        _uuid("building_id"),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("target_celsius", sa.Numeric(3, 1)),
        sa.Column("locked", sa.Boolean()),
        sa.Column("armed", sa.Boolean()),
        _timestamp("updated_at"),
        _apartment_fk("devices"),
        sa.UniqueConstraint("apartment_id", "name", name="uq_devices_apartment_name"),
        sa.UniqueConstraint("id", "apartment_id", name="uq_devices_id_apartment"),
        sa.CheckConstraint("kind IN ('thermostat','door_lock','alarm')", name="ck_devices_kind"),
        sa.CheckConstraint(
            "(kind = 'thermostat') = (target_celsius IS NOT NULL) "
            "AND (kind = 'door_lock') = (locked IS NOT NULL) "
            "AND (kind = 'alarm') = (armed IS NOT NULL)",
            name="ck_devices_state",
        ),
        sa.CheckConstraint(
            "target_celsius IS NULL OR target_celsius BETWEEN 5 AND 35", name="ck_devices_target"
        ),
        sa.CheckConstraint("char_length(name) >= 1", name="ck_devices_name"),
    )

    op.create_table(
        "bookings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        _uuid("apartment_id"),
        _uuid("building_id"),
        _uuid("user_id"),
        sa.Column("amenity", sa.String(100), nullable=False),
        sa.Column("starts_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ends_at", sa.DateTime(timezone=True), nullable=False),
        _timestamp("created_at"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        _apartment_fk("bookings"),
        sa.CheckConstraint("starts_at < ends_at", name="ck_bookings_period"),
        sa.CheckConstraint("char_length(amenity) >= 1", name="ck_bookings_amenity"),
    )
    op.create_index("ix_bookings_apartment_starts", "bookings", ["apartment_id", "starts_at"])

    op.create_table(
        "tickets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        _uuid("apartment_id"),
        _uuid("building_id"),
        _uuid("user_id"),
        sa.Column("category", sa.String(16), nullable=False),
        sa.Column("summary", sa.String(200), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="open"),
        _timestamp("created_at"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        _apartment_fk("tickets"),
        sa.CheckConstraint(
            "category IN ('plumbing','electrical','heating','appliance','other')",
            name="ck_tickets_category",
        ),
        sa.CheckConstraint("status IN ('open','closed')", name="ck_tickets_status"),
        sa.CheckConstraint("char_length(summary) >= 1", name="ck_tickets_summary"),
    )
    op.create_index("ix_tickets_apartment_created", "tickets", ["apartment_id", "created_at"])
    op.create_index("ix_tickets_user_created", "tickets", ["user_id", "created_at"])

    # Turn usage log and rate-limit source. No message or answer text.
    op.create_table(
        "assistant_turns",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        _uuid("user_id"),
        sa.Column("status", sa.String(16), nullable=False, server_default="admitted"),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("prompt_version", sa.String(64), nullable=False),
        sa.Column("model_calls", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tool_calls", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("input_tokens", sa.Integer()),
        sa.Column("output_tokens", sa.Integer()),
        sa.Column("embedding_tokens", sa.Integer()),
        sa.Column("error_code", sa.String(64)),
        _timestamp("created_at"),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.CheckConstraint(
            "status IN ('admitted','answered','no_context','incomplete','failed')",
            name="ck_assistant_turns_status",
        ),
        sa.CheckConstraint(
            "(status = 'admitted') = (finished_at IS NULL)", name="ck_assistant_turns_terminal"
        ),
        sa.CheckConstraint(
            "(status IN ('incomplete','failed')) = (error_code IS NOT NULL)",
            name="ck_assistant_turns_error",
        ),
        sa.CheckConstraint(
            "model_calls >= 0 AND tool_calls >= 0 "
            "AND (input_tokens IS NULL OR input_tokens >= 0) "
            "AND (output_tokens IS NULL OR output_tokens >= 0) "
            "AND (embedding_tokens IS NULL OR embedding_tokens >= 0)",
            name="ck_assistant_turns_counts",
        ),
    )
    op.create_index("ix_assistant_turns_user_created", "assistant_turns", ["user_id", "created_at"])

    # Sensitive actions wait here until the resident confirms them.
    op.create_table(
        "pending_actions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        _uuid("user_id"),
        _uuid("apartment_id"),
        _uuid("building_id"),
        _uuid("device_id"),
        _uuid("turn_id", nullable=True),
        sa.Column("action", sa.String(16), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        _timestamp("created_at"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["turn_id"], ["assistant_turns.id"], ondelete="SET NULL"),
        _apartment_fk("pending_actions"),
        sa.ForeignKeyConstraint(
            ["device_id", "apartment_id"],
            ["devices.id", "devices.apartment_id"],
            ondelete="CASCADE",
            name="fk_pending_actions_device_apartment",
        ),
        sa.CheckConstraint("action IN ('unlock_door')", name="ck_pending_actions_action"),
        sa.CheckConstraint(
            "status IN ('pending','confirmed','expired')", name="ck_pending_actions_status"
        ),
        sa.CheckConstraint(
            "(status = 'pending') = (decided_at IS NULL)", name="ck_pending_actions_decided"
        ),
        sa.CheckConstraint("expires_at > created_at", name="ck_pending_actions_expiry"),
    )
    op.create_index(
        "uq_pending_actions_device",
        "pending_actions",
        ["device_id"],
        unique=True,
        postgresql_where=sa.text("status = 'pending'"),
    )
    op.create_index("ix_pending_actions_user_created", "pending_actions", ["user_id", "created_at"])

    # Audit trail: typed values only, never free text.
    op.create_table(
        "assistant_tool_calls",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        _uuid("turn_id", nullable=True),
        _uuid("user_id"),
        _uuid("apartment_id", nullable=True),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("tool", sa.String(32), nullable=False),
        sa.Column("decision", sa.String(24), nullable=False),
        sa.Column("result_code", sa.String(32), nullable=False),
        _uuid("device_id", nullable=True),
        _uuid("ticket_id", nullable=True),
        _uuid("pending_action_id", nullable=True),
        sa.Column(
            "details", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        _timestamp("created_at"),
        sa.ForeignKeyConstraint(["turn_id"], ["assistant_turns.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["apartment_id"], ["apartments.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["device_id"], ["devices.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["ticket_id"], ["tickets.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["pending_action_id"], ["pending_actions.id"], ondelete="SET NULL"),
        sa.CheckConstraint(
            "source IN ('assistant','confirmation')", name="ck_assistant_tool_calls_source"
        ),
        sa.CheckConstraint(
            "(source = 'assistant') = (turn_id IS NOT NULL)", name="ck_assistant_tool_calls_turn"
        ),
        sa.CheckConstraint(
            "tool IN ('search_documents','list_bookings','create_ticket','set_thermostat',"
            "'lock_door','unlock_door','unknown')",
            name="ck_assistant_tool_calls_tool",
        ),
        sa.CheckConstraint(
            "decision IN ('allowed','denied','pending_confirmation','confirmed')",
            name="ck_assistant_tool_calls_decision",
        ),
        sa.CheckConstraint(
            "result_code IN ('ok','duplicate','already_pending','not_found','not_permitted',"
            "'invalid_arguments','unknown_tool','limit_exceeded','no_documents')",
            name="ck_assistant_tool_calls_result",
        ),
        # Typed values only: any other key (a question, a summary) is rejected.
        sa.CheckConstraint(
            "jsonb_typeof(details) = 'object' "
            "AND (details - ARRAY['celsius','category','passage_count']) = '{}'::jsonb",
            name="ck_assistant_tool_calls_details",
        ),
    )
    op.create_index(
        "ix_assistant_tool_calls_user_created", "assistant_tool_calls", ["user_id", "created_at"]
    )
    op.create_index("ix_assistant_tool_calls_turn", "assistant_tool_calls", ["turn_id"])

    # Building documents: nullable, so existing rows stay owner-only documents.
    op.add_column("rag_documents", _uuid("building_id", nullable=True))
    op.create_foreign_key(
        "rag_documents_building_id_fkey",
        "rag_documents",
        "buildings",
        ["building_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(
        "ix_rag_documents_building",
        "rag_documents",
        ["building_id"],
        postgresql_where=sa.text("building_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_rag_documents_building", table_name="rag_documents")
    op.drop_constraint("rag_documents_building_id_fkey", "rag_documents", type_="foreignkey")
    op.drop_column("rag_documents", "building_id")
    for table in (
        "assistant_tool_calls",
        "pending_actions",
        "assistant_turns",
        "tickets",
        "bookings",
        "devices",
        "memberships",
        "apartments",
        "buildings",
    ):
        op.drop_table(table)
