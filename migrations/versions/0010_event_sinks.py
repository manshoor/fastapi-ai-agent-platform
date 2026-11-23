"""Per-sink outbox delivery: each event is published to Kafka and SQS on its own.

    rag_event_sinks          registered sinks ('kafka' seeded; the SQS dispatcher adds 'sqs')
    rag_outbox  --insert-->  trigger: one rag_outbox_deliveries row per registered sink
    rag_outbox_deliveries    lease, attempts and published state of one event in one sink

The publish columns move off rag_outbox into its Kafka delivery rows.

Revision ID: 0010_event_sinks
Revises: 0009_connectors
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0010_event_sinks"
down_revision: str | None = "0009_connectors"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

PUBLISH_COLUMNS = (
    "lease_token",
    "lease_expires_at",
    "published_at",
    "last_published_at",
    "publish_attempts",
)

FILL_FUNCTION = """
CREATE FUNCTION rag_outbox_fill_deliveries() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO rag_outbox_deliveries (outbox_id, sink, available_at)
    SELECT NEW.id, sink, NEW.available_at FROM rag_event_sinks;
    RETURN NULL;
END
$$
"""


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.create_table(
        "rag_event_sinks",
        sa.Column("sink", sa.String(8), primary_key=True),
        sa.Column(
            "registered_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint("sink IN ('kafka','sqs')", name="ck_rag_event_sinks_sink"),
    )
    op.execute("INSERT INTO rag_event_sinks (sink) VALUES ('kafka')")
    op.create_table(
        "rag_outbox_deliveries",
        sa.Column(
            "outbox_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("rag_outbox.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("sink", sa.String(8), primary_key=True),
        sa.Column(
            "available_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("lease_token", postgresql.UUID(as_uuid=True)),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.Column("last_published_at", sa.DateTime(timezone=True)),
        sa.Column("publish_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.CheckConstraint("sink IN ('kafka','sqs')", name="ck_rag_outbox_deliveries_sink"),
        sa.CheckConstraint(
            "(lease_token IS NULL) = (lease_expires_at IS NULL)",
            name="ck_rag_outbox_deliveries_lease",
        ),
        sa.CheckConstraint("publish_attempts >= 0", name="ck_rag_outbox_deliveries_attempts"),
    )
    op.create_index(
        "ix_rag_outbox_deliveries_due",
        "rag_outbox_deliveries",
        ["sink", "available_at"],
        postgresql_where=sa.text("published_at IS NULL"),
    )

    # Existing publish state becomes each row's Kafka delivery; SQS starts with new events.
    op.execute(
        "INSERT INTO rag_outbox_deliveries (outbox_id, sink, available_at, lease_token, "
        "lease_expires_at, published_at, last_published_at, publish_attempts) "
        "SELECT id, 'kafka', available_at, lease_token, lease_expires_at, published_at, "
        "last_published_at, publish_attempts FROM rag_outbox"
    )
    op.drop_index("ix_rag_outbox_due", table_name="rag_outbox")
    for column in PUBLISH_COLUMNS:
        op.drop_column("rag_outbox", column)
    op.execute(FILL_FUNCTION)
    op.execute(
        "CREATE TRIGGER rag_outbox_fill_deliveries AFTER INSERT ON rag_outbox "
        "FOR EACH ROW EXECUTE FUNCTION rag_outbox_fill_deliveries()"
    )


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("DROP TRIGGER rag_outbox_fill_deliveries ON rag_outbox")
    op.execute("DROP FUNCTION rag_outbox_fill_deliveries()")
    op.add_column("rag_outbox", sa.Column("lease_token", postgresql.UUID(as_uuid=True)))
    op.add_column("rag_outbox", sa.Column("lease_expires_at", sa.DateTime(timezone=True)))
    op.add_column("rag_outbox", sa.Column("published_at", sa.DateTime(timezone=True)))
    op.add_column("rag_outbox", sa.Column("last_published_at", sa.DateTime(timezone=True)))
    op.add_column(
        "rag_outbox",
        sa.Column("publish_attempts", sa.Integer(), nullable=False, server_default="0"),
    )
    # Kafka state folds back; SQS state has no home before this revision and is dropped.
    op.execute(
        "UPDATE rag_outbox AS o SET lease_token = d.lease_token, "
        "lease_expires_at = d.lease_expires_at, published_at = d.published_at, "
        "last_published_at = d.last_published_at, publish_attempts = d.publish_attempts "
        "FROM rag_outbox_deliveries AS d WHERE d.outbox_id = o.id AND d.sink = 'kafka'"
    )
    op.create_index(
        "ix_rag_outbox_due",
        "rag_outbox",
        ["available_at"],
        postgresql_where=sa.text("published_at IS NULL"),
    )
    op.drop_index("ix_rag_outbox_deliveries_due", table_name="rag_outbox_deliveries")
    op.drop_table("rag_outbox_deliveries")
    op.drop_table("rag_event_sinks")
