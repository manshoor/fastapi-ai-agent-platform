"""Memory tools in the audit trail: tool names remember and forget, detail key scope.

Revision ID: 0013_memory_tools
Revises: 0012_assistant_memory
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0013_memory_tools"
down_revision: str | None = "0012_assistant_memory"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Literal copies of the model lists: a migration must not change when the models do.
TOOLS = (
    "'search_documents','list_bookings','create_ticket','set_thermostat',"
    "'lock_door','unlock_door','unknown'"
)
MEMORY_TOOLS = "'remember','forget'"
DETAIL_KEYS = "'celsius','category','passage_count'"
MEMORY_DETAIL_KEYS = "'scope'"


def _tool_check(tools: str) -> None:
    op.drop_constraint("ck_assistant_tool_calls_tool", "assistant_tool_calls", type_="check")
    op.create_check_constraint(
        "ck_assistant_tool_calls_tool", "assistant_tool_calls", f"tool IN ({tools})"
    )


def _details_check(keys: str) -> None:
    # Typed values only: any other key (a question, a summary, a note) is rejected.
    op.drop_constraint("ck_assistant_tool_calls_details", "assistant_tool_calls", type_="check")
    op.create_check_constraint(
        "ck_assistant_tool_calls_details",
        "assistant_tool_calls",
        f"jsonb_typeof(details) = 'object' AND (details - ARRAY[{keys}]) = '{{}}'::jsonb",
    )


def upgrade() -> None:
    # Fail fast instead of queueing every API statement behind a blocked ALTER.
    op.execute("SET LOCAL lock_timeout = '5s'")
    _tool_check(f"{TOOLS},{MEMORY_TOOLS}")
    _details_check(f"{DETAIL_KEYS},{MEMORY_DETAIL_KEYS}")


def downgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    # Refuse, rather than delete audit rows the old checks cannot hold. Re-adding the old
    # checks validates every row too, so a row written after this test still fails it.
    op.execute(
        "DO $$ BEGIN IF EXISTS (SELECT 1 FROM assistant_tool_calls "
        f"WHERE tool IN ({MEMORY_TOOLS}) OR details ? 'scope') THEN RAISE EXCEPTION "
        "'assistant_tool_calls holds memory tool audit rows; downgrade would lose them'; "
        "END IF; END $$"
    )
    _details_check(DETAIL_KEYS)
    _tool_check(TOOLS)
