"""Public RAG request/response schemas, event payloads, and fixed codes."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AfterValidator, AwareDatetime, Field, StringConstraints

from app.schemas import StrictSchema, Title, reject_nul

# Readers never see "deleted": a tombstone is a 404.
DocumentStatus = Literal["pending", "processing", "ready", "failed"]
TERMINAL_STATES = frozenset({"ready", "failed", "deleted"})
SourceKind = Literal["upload", "connector"]
UNSETTLED_STATES = ("pending", "processing")
CHUNKER_VERSION = "recursive-v2"


class ContentType(StrEnum):
    TEXT = "text/plain"
    MARKDOWN = "text/markdown"
    PDF = "application/pdf"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    HTML = "text/html"


# Parsed in the sandboxed child; text and Markdown are decoded in the worker.
CHILD_FORMATS = {ContentType.PDF: "pdf", ContentType.DOCX: "docx", ContentType.HTML: "html"}
# Formats whose "#" heading lines become each chunk's heading path.
HEADED_TYPES = frozenset({ContentType.MARKDOWN, ContentType.DOCX, ContentType.HTML})


class IngestError(StrEnum):
    """Fixed, content-free reasons a document ends as failed."""

    PDF_ENCRYPTED = "pdf_encrypted"
    PDF_PARSE_FAILED = "pdf_parse_failed"
    PDF_PARSE_TIMEOUT = "pdf_parse_timeout"
    PDF_PAGE_LIMIT = "pdf_page_limit"
    DOCX_PARSE_FAILED = "docx_parse_failed"
    HTML_PARSE_FAILED = "html_parse_failed"
    PARSE_TIMEOUT = "parse_timeout"
    TEXT_LIMIT = "text_limit"
    NO_TEXT = "no_text"
    CHUNK_LIMIT = "chunk_limit"
    EMBEDDING_REJECTED = "embedding_rejected"
    ATTEMPTS_EXHAUSTED = "attempts_exhausted"
    QUEUE_EXPIRED = "queue_expired"


# Accepted file name suffixes; the declared part content type is never trusted.
SUFFIXES = {
    ".txt": ContentType.TEXT,
    ".md": ContentType.MARKDOWN,
    ".markdown": ContentType.MARKDOWN,
    ".pdf": ContentType.PDF,
    ".docx": ContentType.DOCX,
    ".html": ContentType.HTML,
    ".htm": ContentType.HTML,
}


class DocumentResponse(StrictSchema):
    """Document metadata only; neither source bytes nor chunk text is ever returned."""

    id: UUID
    # Set for building documents; residents of that building can search them.
    building_id: UUID | None
    title: str
    content_type: ContentType
    size_bytes: int
    status: DocumentStatus
    chunk_count: int | None
    page_count: int | None
    error_code: str | None
    # The newest version, and the one queries read (null until a version is ready).
    version: int
    active_version: int | None
    # Why the newest version failed while active_version stays served.
    update_error_code: str | None
    source_kind: SourceKind
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None
    status_url: str


class DocumentList(StrictSchema):
    items: list[DocumentResponse]


class UploadTitle(StrictSchema):
    title: Title = Field(repr=False)


Question = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2000),
    AfterValidator(reject_nul),
]


class QueryRequest(StrictSchema):
    question: Question = Field(repr=False)
    top_k: int = Field(default=4, ge=1, le=8, strict=True)
    document_ids: list[UUID] | None = Field(default=None, min_length=1, max_length=20)


class Citation(StrictSchema):
    document_id: UUID
    title: str
    ordinal: int
    page: int | None
    snippet: str
    # The verified word-for-word quotes this answer takes from the passage.
    quotes: list[str]


class DiagnosticOverrides(StrictSchema):
    """Eval-only answer-context settings for one query; unset fields use the server's."""

    neighbor_chunks: int | None = Field(default=None, ge=0, le=2, strict=True)
    relevance_margin: float | None = Field(default=None, gt=0, le=2)
    rerank: Literal["off", "llm"] | None = None
    reasoning_effort: Literal["none", "low", "medium"] | None = None
    verifier: Literal["off", "on"] | None = None


class DiagnosticsRequest(QueryRequest):
    overrides: DiagnosticOverrides | None = None


class QueryResponse(StrictSchema):
    status: Literal["answered", "no_context"]
    answer: str
    citations: list[Citation]


class DiagnosticPassage(StrictSchema):
    """One passage exactly as the answer model received it."""

    label: str
    document_id: UUID
    title: str
    ordinal: int
    page: int | None
    distance: float
    section: str | None
    text: str


class DiagnosticClaim(StrictSchema):
    statement: str
    citation: str
    quote: str
    check: str


class DiagnosticVerdict(StrictSchema):
    claim: str
    supported: bool


class DiagnosticUsage(StrictSchema):
    embedding_tokens: int | None
    input_tokens: int | None
    output_tokens: int | None
    chat_calls: int | None
    reasoning_effort: str


class DiagnosticsResponse(QueryResponse):
    """Eval-only view of one query. Fields of later grounding steps stay null until they run."""

    no_context_reason: str | None
    passages: list[DiagnosticPassage]
    claims: list[DiagnosticClaim] | None
    verdicts: list[DiagnosticVerdict] | None
    usage: DiagnosticUsage


class DocumentEventType(StrEnum):
    CREATED = "doc.created"
    UPDATED = "doc.updated"
    DELETED = "doc.deleted"


# Outbox kind for each event type, and back.
EVENT_KINDS = {
    DocumentEventType.CREATED: "created",
    DocumentEventType.UPDATED: "updated",
    DocumentEventType.DELETED: "deleted",
}
INGEST_TYPES = frozenset({DocumentEventType.CREATED, DocumentEventType.UPDATED})


def ingest_kind(version: int) -> str:
    """Outbox kind for ingesting a version: the first is a create, later ones updates."""
    return "created" if version == 1 else "updated"


class DocumentEvent(StrictSchema):
    """Kafka payload. Identifiers only: no title, text or connector settings."""

    version: Literal[1]
    event_id: UUID
    type: DocumentEventType
    document_id: UUID
    document_version: int = Field(ge=1, strict=True)
    attempt: int = Field(ge=1, le=3, strict=True)
    source_kind: SourceKind
    created_at: AwareDatetime
