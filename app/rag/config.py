from pydantic import Field, model_validator

from app.ai.config import AISettings
from app.models import RAG_EMBEDDING_DIMENSIONS

MIB = 1024 * 1024


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
    rag_max_characters: int = Field(default=1_000_000, ge=1, le=10_000_000)
    rag_chunk_size: int = Field(default=1200, ge=100, le=4000)
    rag_chunk_overlap: int = Field(default=200, ge=0, le=1000)
    rag_max_chunks: int = Field(default=1200, ge=1, le=10_000)
    rag_embedding_batch_size: int = Field(default=64, ge=1, le=256)

    # Quotas: documents are kept until deleted; unsettled = pending or processing.
    rag_owner_document_limit: int = Field(default=50, ge=1, le=10_000)
    rag_owner_unsettled_limit: int = Field(default=5, ge=1, le=1000)
    rag_global_unsettled_limit: int = Field(default=100, ge=1, le=100_000)

    # Query limits.
    rag_query_limit: int = Field(default=10, ge=1, le=10_000)
    rag_query_window_seconds: float = Field(default=60, gt=0, le=86400)
    rag_query_concurrency: int = Field(default=4, ge=1, le=64)
    rag_query_deadline_seconds: float = Field(default=60, gt=0, le=120)

    # Ingestion lifecycle.
    rag_lease_seconds: float = Field(default=120, gt=0)
    rag_max_attempts: int = Field(default=3, ge=1, le=3)
    rag_queue_timeout_seconds: float = Field(default=3600, gt=0, le=86400)
    kafka_rag_request_topic: str = "rag.document-ingest.requested.v1"
    kafka_rag_dead_letter_topic: str = "rag.document-ingest.dead-letter.v1"
    kafka_rag_group_id: str = "rag-ingest-worker-v1"

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
        # The first renewal comes only after parsing and the first embedding batch.
        first_renewal = self.rag_pdf_timeout_seconds + self.ai_embedding_timeout_seconds
        if self.rag_lease_seconds <= first_renewal + 5:
            raise ValueError("RAG_LEASE_SECONDS must exceed parse + embedding timeouts by 5s")
        # Provider timeouts must fire before the query deadline: a billed answer is never
        # thrown away by the deadline.
        provider_total = self.ai_embedding_timeout_seconds + self.ai_provider_timeout_seconds
        if self.rag_query_deadline_seconds <= provider_total:
            raise ValueError("RAG_QUERY_DEADLINE_SECONDS must exceed both provider timeouts")
        return self

    @property
    def rag_upload_ceiling_bytes(self) -> int:
        # Largest accepted file plus room for multipart headers and the title field.
        return max(self.rag_text_max_bytes, self.rag_pdf_max_bytes) + 64 * 1024
