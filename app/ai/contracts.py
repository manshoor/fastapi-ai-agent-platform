from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AfterValidator, AwareDatetime, Field, StringConstraints

from app.schemas import Description, StrictSchema, Title, reject_nul

TERMINAL_STATES = frozenset({"succeeded", "failed", "needs_reconciliation"})
JobStatus = Literal["pending", "leased", "calling", "succeeded", "failed", "needs_reconciliation"]
Instruction = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=1000),
    AfterValidator(reject_nul),
]


class RewriteRequest(StrictSchema):
    instruction: Instruction = Field(repr=False)


class RewriteSuggestion(StrictSchema):
    title: Title
    description: Description


# Passage labels the answer prompt assigns, in retrieval order: P1 is the closest chunk.
PassageLabel = Annotated[str, StringConstraints(pattern=r"^P[1-8]$")]
Answer = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4000),
    AfterValidator(reject_nul),
]


class GroundedAnswer(StrictSchema):
    """Structured model output for a RAG answer. Empty citations mean "not in passages"."""

    answer: Answer
    citations: list[PassageLabel] = Field(max_length=8)


class RewriteEvent(StrictSchema):
    version: Literal[1]
    event_id: UUID
    job_id: UUID
    attempt: int = Field(ge=1, le=3, strict=True)
    created_at: AwareDatetime


class JobResponse(StrictSchema):
    id: UUID
    item_id: UUID
    status: JobStatus
    source_item_updated_at: datetime
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None
    content_expires_at: datetime | None
    suggestion: RewriteSuggestion | None
    error_code: str | None
    status_url: str
