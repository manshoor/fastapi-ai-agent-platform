"""Step-up for unlock confirmation: TOTP credentials, auth failure counts, audit codes.

Revision ID: 0006_step_up
Revises: 0005_grounding
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006_step_up"
down_revision: str | None = "0005_grounding"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Literal copies of the model lists: a migration must not change when the models do.
RESULT_CODES = (
    "'ok','duplicate','already_pending','not_found','not_permitted',"
    "'invalid_arguments','unknown_tool','limit_exceeded','no_documents'"
)
STEP_UP_CODES = "'step_up_required','step_up_failed'"


def upgrade() -> None:
    # Fail fast instead of queueing every API statement behind a blocked ALTER.
    op.execute("SET LOCAL lock_timeout = '5s'")

    # One TOTP secret per user, sealed by the API. Enabled once a first code matched;
    # last_used_step stops a code from being used twice.
    op.create_table(
        "totp_credentials",
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("secret_ciphertext", sa.LargeBinary(), nullable=False),
        sa.Column("nonce", sa.LargeBinary(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("enabled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_used_step", sa.BigInteger(), nullable=True),
        sa.CheckConstraint("octet_length(nonce) = 12", name="ck_totp_credentials_nonce"),
        sa.CheckConstraint(
            "octet_length(secret_ciphertext) BETWEEN 17 AND 64",
            name="ck_totp_credentials_secret",
        ),
        sa.CheckConstraint(
            "(enabled_at IS NULL) = (last_used_step IS NULL)",
            name="ck_totp_credentials_enabled",
        ),
    )

    # Failed password and code checks, counted per user over a sliding window.
    op.create_table(
        "auth_failures",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("clock_timestamp()"),
            nullable=False,
        ),
        sa.CheckConstraint("kind IN ('password','totp')", name="ck_auth_failures_kind"),
    )
    op.create_index(
        "ix_auth_failures_user_kind_created", "auth_failures", ["user_id", "kind", "created_at"]
    )

    # Audit: a confirmation refused for a missing or wrong code.
    op.drop_constraint("ck_assistant_tool_calls_result", "assistant_tool_calls", type_="check")
    op.create_check_constraint(
        "ck_assistant_tool_calls_result",
        "assistant_tool_calls",
        f"result_code IN ({RESULT_CODES},{STEP_UP_CODES})",
    )


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")

    # Refuse, rather than delete audit rows the old check cannot hold.
    op.execute(
        "DO $$ BEGIN IF EXISTS (SELECT 1 FROM assistant_tool_calls "
        f"WHERE result_code IN ({STEP_UP_CODES})) THEN RAISE EXCEPTION "
        "'assistant_tool_calls holds step-up audit rows; downgrade would lose them'; "
        "END IF; END $$"
    )
    op.drop_constraint("ck_assistant_tool_calls_result", "assistant_tool_calls", type_="check")
    op.create_check_constraint(
        "ck_assistant_tool_calls_result",
        "assistant_tool_calls",
        f"result_code IN ({RESULT_CODES})",
    )
    op.drop_index("ix_auth_failures_user_kind_created", table_name="auth_failures")
    op.drop_table("auth_failures")
    op.drop_table("totp_credentials")
