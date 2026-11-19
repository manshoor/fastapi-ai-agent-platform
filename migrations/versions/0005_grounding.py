"""Grounding: chunk headings, no-context reasons, reasoning effort, chat calls, claim checks.

Revision ID: 0005_grounding
Revises: 0004_resident_assistant
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_grounding"
down_revision: str | None = "0004_resident_assistant"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Literal copies of the model lists: a migration must not change when the models do.
NO_CONTEXT_REASONS = (
    "'no_chunks','no_passages','no_claims','quote_too_short','quote_not_found',"
    "'number_not_in_passage','rerank_empty','verifier_rejected'"
)
CLAIM_FAILURES = (
    "'quote_too_short','quote_not_found','number_not_in_passage','tool_claim_ungrounded'"
)


def upgrade() -> None:
    # Fail fast instead of queueing every API statement behind a blocked ALTER.
    op.execute("SET LOCAL lock_timeout = '5s'")

    # Chunks: a heading path for Markdown; existing chunks keep null until re-uploaded.
    op.add_column("rag_chunks", sa.Column("heading", sa.Text(), nullable=True))
    op.create_check_constraint(
        "ck_rag_chunks_heading",
        "rag_chunks",
        "heading IS NULL OR char_length(heading) BETWEEN 1 AND 300",
    )

    # Queries: why there was no answer, the answer call's effort, and completed chat calls.
    op.add_column("rag_queries", sa.Column("no_context_reason", sa.String(32), nullable=True))
    op.add_column(
        "rag_queries",
        sa.Column("reasoning_effort", sa.String(16), nullable=False, server_default="none"),
    )
    op.add_column("rag_queries", sa.Column("chat_calls", sa.SmallInteger(), nullable=True))
    op.create_check_constraint(
        "ck_rag_queries_no_context_reason",
        "rag_queries",
        "no_context_reason IS NULL OR (status = 'no_context' "
        f"AND no_context_reason IN ({NO_CONTEXT_REASONS}))",
    )
    op.create_check_constraint(
        "ck_rag_queries_reasoning_effort",
        "rag_queries",
        "reasoning_effort IN ('none','low','medium')",
    )
    op.create_check_constraint(
        "ck_rag_queries_chat_calls", "rag_queries", "chat_calls IS NULL OR chat_calls >= 0"
    )

    # Assistant turns: which claim check refused the reply.
    op.add_column("assistant_turns", sa.Column("claim_failure", sa.String(32), nullable=True))
    op.create_check_constraint(
        "ck_assistant_turns_claim_failure",
        "assistant_turns",
        "claim_failure IS NULL OR (status IN ('no_context','incomplete') "
        f"AND claim_failure IN ({CLAIM_FAILURES}))",
    )


def downgrade() -> None:
    op.drop_constraint("ck_assistant_turns_claim_failure", "assistant_turns", type_="check")
    op.drop_column("assistant_turns", "claim_failure")
    for name in (
        "ck_rag_queries_chat_calls",
        "ck_rag_queries_reasoning_effort",
        "ck_rag_queries_no_context_reason",
    ):
        op.drop_constraint(name, "rag_queries", type_="check")
    for column in ("chat_calls", "reasoning_effort", "no_context_reason"):
        op.drop_column("rag_queries", column)
    op.drop_constraint("ck_rag_chunks_heading", "rag_chunks", type_="check")
    op.drop_column("rag_chunks", "heading")
