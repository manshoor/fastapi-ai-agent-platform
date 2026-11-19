from pydantic import Field, model_validator

from app.rag.config import RAGSettings


class AssistantSettings(RAGSettings):
    """Resident assistant limits. The assistant reuses the RAG runtime and provider."""

    assistant_enabled: bool = False
    assistant_demo_enabled: bool = False

    # Turn admission.
    assistant_turn_limit: int = Field(default=10, ge=1, le=10_000)
    assistant_turn_window_seconds: float = Field(default=60, gt=0, le=86400)
    # Turns hold no database connection while they wait on the provider.
    assistant_concurrency: int = Field(default=64, ge=1, le=256)
    assistant_max_model_calls: int = Field(default=4, ge=2, le=8)
    assistant_turn_deadline_seconds: float = Field(default=120, gt=0, le=300)

    # Action limits, counted per user in the database.
    assistant_device_action_limit: int = Field(default=10, ge=1, le=1000)
    assistant_device_action_window_seconds: float = Field(default=3600, gt=0, le=86400)
    assistant_ticket_limit: int = Field(default=3, ge=1, le=1000)
    assistant_ticket_window_seconds: float = Field(default=86400, gt=0, le=604800)
    assistant_ticket_dedupe_seconds: float = Field(default=600, ge=0, le=86400)
    assistant_unlock_limit: int = Field(default=3, ge=1, le=100)
    assistant_unlock_window_seconds: float = Field(default=3600, gt=0, le=86400)
    assistant_unlock_ttl_seconds: float = Field(default=60, gt=0, le=300)

    @model_validator(mode="after")
    def validate_assistant(self) -> "AssistantSettings":
        if not self.assistant_enabled:
            return self
        if not self.rag_enabled:
            raise ValueError("ASSISTANT_ENABLED requires RAG_ENABLED")
        # One search and one answer must fit: model call, embedding, model call.
        needed = 2 * self.ai_provider_timeout_seconds + self.ai_embedding_timeout_seconds
        if self.assistant_turn_deadline_seconds <= needed:
            raise ValueError(
                "ASSISTANT_TURN_DEADLINE_SECONDS must exceed two provider timeouts "
                "plus the embedding timeout"
            )
        return self
