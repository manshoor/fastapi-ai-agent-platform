from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AfterValidator, AwareDatetime, Field, StringConstraints

from app.rag.ontology import MAX_NAME_CHARACTERS, NodeType, Relation
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


class PassageRanking(StrictSchema):
    """Rerank output: labels of the passages that help answer, most useful first."""

    passages: list[PassageLabel] = Field(max_length=8)


ClaimLabel = Annotated[str, StringConstraints(pattern=r"^C[1-8]$")]


class ClaimVerdict(StrictSchema):
    claim: ClaimLabel
    supported: bool


class ClaimVerdicts(StrictSchema):
    """Verifier output: one verdict per claim label, each claim exactly once."""

    verdicts: list[ClaimVerdict] = Field(max_length=8)


class QuotedAnswer(StrictSchema):
    """Structured model output for a RAG answer. No claims means "not in passages"."""

    claims: list[Claim] = Field(max_length=8)


# Chunks of one extraction call, in order: C1 is the first chunk sent.
ChunkLabel = Annotated[str, StringConstraints(pattern=r"^C[1-8]$")]
NodeName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_NAME_CHARACTERS),
    AfterValidator(reject_nul),
]
# Per call: at most 8 chunks, at most 24 items of each kind per chunk.
MAX_GRAPH_ITEMS = 8 * 24


class GraphNode(StrictSchema):
    chunk: ChunkLabel
    type: NodeType
    name: NodeName


class GraphEdge(StrictSchema):
    """source -relation-> target, with a quote of the chunk that states it."""

    chunk: ChunkLabel
    source: NodeName
    source_type: NodeType
    relation: Relation
    target: NodeName
    target_type: NodeType
    quote: Quote


class GraphExtraction(StrictSchema):
    """Graph extraction output. The server keeps only items grounded in their chunk."""

    nodes: list[GraphNode] = Field(max_length=MAX_GRAPH_ITEMS)
    edges: list[GraphEdge] = Field(max_length=MAX_GRAPH_ITEMS)


# Where an assistant claim comes from this turn: a search passage (P1-P8), the
# server-written fact of a tool result (T1-T8), or a recalled memory (M1-M20). The quote
# is copied from that text.
ClaimSource = Annotated[str, StringConstraints(pattern=r"^(?:[PT][1-8]|M(?:[1-9]|1[0-9]|20))$")]
AssistantQuote = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=300),
    AfterValidator(reject_nul),
]


class AssistantClaim(StrictSchema):
    """One reply sentence, quoting the passage or tool fact it names as its source."""

    statement: Statement
    source: ClaimSource
    quote: AssistantQuote


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
