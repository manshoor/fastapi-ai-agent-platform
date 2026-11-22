from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, StringConstraints, model_validator

from app.ai.config import AISettings
from app.models import RAG_EMBEDDING_DIMENSIONS

MIB = 1024 * 1024
# off: answer from the nearest chunks; llm: one paid call picks and orders them first.
RerankMode = Literal["off", "llm"]
RERANK_LLM = "llm"
# Reasoning effort of the answer call only; every other call runs with none.
ReasoningEffort = Literal["none", "low", "medium"]
# on: one paid call judges every verified claim against its passage before answering.
VerifierMode = Literal["off", "on"]
VERIFIER_ON = "on"
# Graph expansion of one query (eval override); the default follows RAG_GRAPH_ENABLED.
GraphMode = Literal["off", "on"]
GRAPH_ON = "on"
GRAPH_OFF = "off"
# Event sinks: Kafka always (it carries recovery's redispatch); SQS optionally beside it.
EventSinks = Annotated[str, StringConstraints(pattern=r"^kafka(,sqs)?$|^sqs,kafka$")]
SQS_SINK = "sqs"
# FIFO queue names: per-document order and deduplication by event id.
FIFOQueue = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,75}\.fifo$")]


class RAGSettings(AISettings):
    """RAG limits and topics. RAG reuses the AI stack's Kafka, provider and collector."""

    rag_enabled: bool = False

    # Upload limits.
    rag_text_max_bytes: int = Field(default=1 * MIB, ge=1, le=5 * MIB)
    rag_pdf_max_bytes: int = Field(default=5 * MIB, ge=1, le=20 * MIB)
    rag_upload_timeout_seconds: float = Field(default=30, gt=0, le=120)
    rag_upload_concurrency: int = Field(default=4, ge=1, le=64)

    # Parsing and chunking limits.
    rag_pdf_max_pages: int = Field(default=300, ge=1, le=2000)
    rag_pdf_timeout_seconds: float = Field(default=30, gt=0, le=300)
    rag_pdf_memory_bytes: int = Field(default=1024 * MIB, ge=64 * MIB)
    # OCR of PDF pages with almost no text. It runs inside RAG_PDF_TIMEOUT_SECONDS:
    # pages past the cap or the deadline stay unread and the parse records truncation.
    rag_ocr_enabled: bool = True
    rag_ocr_max_pages: int = Field(default=25, ge=1, le=300)
    rag_ocr_page_seconds: float = Field(default=20, gt=0, le=120)
    rag_max_characters: int = Field(default=1_000_000, ge=1, le=10_000_000)
    rag_chunk_size: int = Field(default=1200, ge=100, le=4000)
    rag_chunk_overlap: int = Field(default=200, ge=0, le=1000)
    rag_max_chunks: int = Field(default=1200, ge=1, le=10_000)
    rag_embedding_batch_size: int = Field(default=64, ge=1, le=256)

    # Quotas: documents are kept until deleted; unsettled = pending or processing.
    rag_owner_document_limit: int = Field(default=50, ge=1, le=10_000)
    rag_owner_unsettled_limit: int = Field(default=5, ge=1, le=1000)
    rag_global_unsettled_limit: int = Field(default=100, ge=1, le=100_000)
    # Connector documents in progress, all connectors together: kept under the global
    # limit so a large sync leaves room for uploads.
    rag_connector_unsettled_limit: int = Field(default=20, ge=1, le=10_000)

    # Connector sync service (python -m app.rag.connectors).
    rag_connectors_file: str | None = None
    rag_connector_poll_seconds: float = Field(default=15, gt=0, le=3600)

    # Query limits.
    rag_query_limit: int = Field(default=10, ge=1, le=10_000)
    rag_query_window_seconds: float = Field(default=60, gt=0, le=86400)
    rag_query_concurrency: int = Field(default=4, ge=1, le=64)
    rag_query_deadline_seconds: float = Field(default=60, gt=0, le=120)

    # Answer context for /api/v1/rag/query; the assistant's search is unaffected.
    # Neighbor chunks on each side of a retrieved chunk, merged into one passage.
    rag_neighbor_chunks: int = Field(default=1, ge=0, le=2)
    # Drop candidates more than this cosine distance behind the closest; unset keeps all.
    rag_relevance_margin: float | None = Field(default=None, gt=0, le=2)
    rag_rerank: RerankMode = "off"
    rag_answer_reasoning_effort: ReasoningEffort = "none"
    rag_verifier: VerifierMode = "off"

    # Knowledge graph (app/rag/graph.py): extraction at ingest (one call per batch of
    # chunks, stored with that batch) and expansion at query time. Off by default: each
    # batch is a paid call.
    rag_graph_enabled: bool = False
    # Chunks per extraction call and per stored batch while the graph is on.
    # 4 chunks x 8 nodes and 8 edges with short quotes is about 2,500 output tokens,
    # inside the default AI_MAX_OUTPUT_TOKENS (4096). Output cut short has no graph.
    rag_graph_batch_size: int = Field(default=4, ge=1, le=8)
    rag_graph_max_items_per_chunk: int = Field(default=8, ge=1, le=24)
    rag_graph_hops: int = Field(default=1, ge=1, le=2)
    # Graph passages added after the vector passages; the total stays at most 8.
    rag_graph_extra_passages: int = Field(default=2, ge=0, le=4)

    # Ingestion lifecycle.
    rag_lease_seconds: float = Field(default=120, gt=0)
    rag_max_attempts: int = Field(default=3, ge=1, le=3)
    rag_queue_timeout_seconds: float = Field(default=3600, gt=0, le=86400)
    # doc.created, doc.updated and doc.deleted, keyed by document id.
    kafka_rag_event_topic: str = "rag.document-events.v1"
    kafka_rag_dead_letter_topic: str = "rag.document-events.dead-letter.v1"
    kafka_rag_group_id: str = "rag-ingest-worker-v1"

    # Event sinks. With sqs, RAG_SQS_ENDPOINT is required: nothing falls back to real AWS.
    rag_event_sinks: EventSinks = "kafka"
    rag_sqs_endpoint: str | None = None
    rag_sqs_region: str = Field(default="us-east-1", pattern=r"^[a-z]{2}-[a-z]+-\d$")
    rag_sqs_queue: FIFOQueue = "rag-document-events.fifo"
    rag_sqs_dead_letter_queue: FIFOQueue = "rag-document-events-dlq.fifo"
    # Above the document lease; the SQS worker extends it while a message is handled.
    rag_sqs_visibility_seconds: int = Field(default=180, ge=30, le=43_200)
    rag_sqs_wait_seconds: int = Field(default=10, ge=1, le=20)

    @model_validator(mode="after")
    def validate_rag(self) -> "RAGSettings":
        if not self.rag_enabled:
            # The rewrite dispatcher builds these settings too; RAG-only rules apply when on.
            return self
        if not self.ai_enabled:
            raise ValueError("RAG_ENABLED requires AI_ENABLED")
        if self.ai_embedding_dimensions != RAG_EMBEDDING_DIMENSIONS:
            raise ValueError("AI_EMBEDDING_DIMENSIONS must match the vector column")
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")
        # The first renewal comes only after parsing and the first batch: its embedding
        # call, and its graph extraction call when the graph is on.
        first_renewal = self.rag_pdf_timeout_seconds + self.ai_embedding_timeout_seconds
        if self.rag_graph_enabled:
            first_renewal += self.ai_provider_timeout_seconds
        if self.rag_lease_seconds <= first_renewal + 5:
            raise ValueError(
                "RAG_LEASE_SECONDS must exceed parse + embedding (+ graph extraction) "
                "timeouts by 5s"
            )
        if SQS_SINK in self.event_sinks:
            self._validate_sqs()
        floor = query_deadline_floor(self, self.rag_rerank, self.rag_verifier)
        if self.rag_query_deadline_seconds <= floor:
            raise ValueError(
                f"RAG_QUERY_DEADLINE_SECONDS must exceed {floor:g}: the embedding timeout "
                "plus the provider timeout per chat call (one more each for RAG_RERANK=llm "
                "and RAG_VERIFIER=on)"
            )
        return self

    def _validate_sqs(self) -> None:
        endpoint = urlsplit(self.rag_sqs_endpoint or "")
        if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
            raise ValueError("The sqs sink requires RAG_SQS_ENDPOINT (an http or https URL)")
        if endpoint.username is not None or endpoint.password is not None:
            raise ValueError("RAG_SQS_ENDPOINT must carry no credentials")
        if self.rag_sqs_visibility_seconds <= self.rag_lease_seconds:
            raise ValueError("RAG_SQS_VISIBILITY_SECONDS must exceed RAG_LEASE_SECONDS")

    @property
    def event_sinks(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.rag_event_sinks.split(",")))

    @property
    def rag_upload_ceiling_bytes(self) -> int:
        # Largest accepted file plus room for multipart headers and the title field.
        return max(self.rag_text_max_bytes, self.rag_pdf_max_bytes) + 64 * 1024


def query_deadline_floor(
    settings: RAGSettings, rerank: RerankMode, verifier: VerifierMode
) -> float:
    """Provider timeouts must fire before the query deadline: a billed answer is never
    thrown away by the deadline. One embedding plus every chat call in sequence."""
    chat_calls = 1 + (rerank == RERANK_LLM) + (verifier == VERIFIER_ON)
    return settings.ai_embedding_timeout_seconds + settings.ai_provider_timeout_seconds * chat_calls
