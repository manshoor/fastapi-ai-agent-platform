from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AfterValidator, AwareDatetime, Field, StringConstraints, model_validator

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


Statement = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=480),
    AfterValidator(reject_nul),
]
Quote = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=300),
    AfterValidator(reject_nul),
]


class Claim(StrictSchema):
    """One answer sentence, the passage it comes from, and a word-for-word quote from it."""

    statement: Statement
    citation: PassageLabel
    quote: Quote


class QuotedAnswer(StrictSchema):
    """Structured model output for a RAG answer. No claims means "not in passages"."""

    claims: list[Claim] = Field(max_length=8)


# Where an assistant claim comes from: a search passage this turn, or a tool result.
TOOL_SOURCE = "tool"
ClaimSource = Annotated[str, StringConstraints(pattern=r"^(P[1-8]|tool)$")]
AssistantQuote = Annotated[
    str,
    StringConstraints(strip_whitespace=True, max_length=300),
    AfterValidator(reject_nul),
]


class AssistantClaim(StrictSchema):
    """One reply sentence. A passage claim quotes its passage; a tool claim has no quote."""

    statement: Statement
    source: ClaimSource
    quote: AssistantQuote

    @model_validator(mode="after")
    def _quote_fits_source(self) -> "AssistantClaim":
        if (self.source == TOOL_SOURCE) != (self.quote == ""):
            raise ValueError("A tool claim has no quote; a passage claim has one")
        return self


class AssistantReply(StrictSchema):
    """Final assistant output. No claims means the assistant could not answer."""

    claims: list[AssistantClaim] = Field(max_length=8)


class JudgeVerdict(StrictSchema):
    """Evaluation-only grade of one answer against the passages it cites."""

    grounded: bool
    answers_question: bool


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
