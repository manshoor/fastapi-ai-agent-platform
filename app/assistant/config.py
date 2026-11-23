import base64

from pydantic import Field, SecretStr, model_validator

from app.rag.config import RAGSettings

# AES-256 seals each user's authenticator secret, and each memory.
TOTP_KEY_BYTES = 32
DAY_SECONDS = 86400


def _is_totp_key(key: SecretStr | None) -> bool:
    """Base64 for exactly 32 bytes; used for both AES-256 keys."""
    if key is None:
        return False
    try:
        return len(base64.b64decode(key.get_secret_value(), validate=True)) == TOTP_KEY_BYTES
    except ValueError:
        return False


def _key_bytes(key: SecretStr) -> bytes:
    return base64.b64decode(key.get_secret_value(), validate=True)


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

    # Step-up before an unlock runs: an authenticator code (RFC 6238). The key seals each
    # user's secret; scripts/init_ai_env.py generates one.
    assistant_totp_key: SecretStr | None = Field(default=None, repr=False)
    # Shown in the authenticator app; no colon, which separates issuer and account.
    assistant_totp_issuer: str = Field(
        default="Resident assistant", pattern=r"^[A-Za-z0-9 ._-]{1,64}$"
    )
    assistant_totp_failure_limit: int = Field(default=5, ge=1, le=100)
    assistant_totp_failure_window_seconds: float = Field(default=900, gt=0, le=86400)

    # Memory (app/assistant/memory.py): notes the resident asked to keep, sealed under
    # their own key (never the TOTP key). scripts/init_ai_env.py generates one.
    assistant_memory_enabled: bool = False
    assistant_memory_key: SecretStr | None = Field(default=None, repr=False)
    assistant_memory_retention_seconds: float = Field(
        default=90 * DAY_SECONDS, ge=DAY_SECONDS, le=365 * DAY_SECONDS
    )
    # Active memories per user, and writes per user per day (a forget gives none back).
    assistant_memory_limit: int = Field(default=50, ge=1, le=200)
    assistant_memory_writes_per_day: int = Field(default=10, ge=1, le=100)

    @model_validator(mode="after")
    def validate_assistant(self) -> "AssistantSettings":
        if self.assistant_memory_enabled and not self.assistant_enabled:
            raise ValueError("ASSISTANT_MEMORY_ENABLED requires ASSISTANT_ENABLED")
        if not self.assistant_enabled:
            return self
        if not self.rag_enabled:
            raise ValueError("ASSISTANT_ENABLED requires RAG_ENABLED")
        if not _is_totp_key(self.assistant_totp_key):
            raise ValueError("ASSISTANT_TOTP_KEY must hold 32 bytes encoded as base64")
        if self.assistant_memory_enabled:
            if not _is_totp_key(self.assistant_memory_key):
                raise ValueError("ASSISTANT_MEMORY_KEY must hold 32 bytes encoded as base64")
            # One key per purpose: a leak or rotation of one never touches the other.
            # Compared as bytes: base64 can spell one key in more than one way.
            if _key_bytes(self.assistant_memory_key) == _key_bytes(self.assistant_totp_key):
                raise ValueError("ASSISTANT_MEMORY_KEY must differ from ASSISTANT_TOTP_KEY")
        # One search and one answer must fit: model call, embedding, model call.
        needed = 2 * self.ai_provider_timeout_seconds + self.ai_embedding_timeout_seconds
        if self.assistant_turn_deadline_seconds <= needed:
            raise ValueError(
                "ASSISTANT_TURN_DEADLINE_SECONDS must exceed two provider timeouts "
                "plus the embedding timeout"
            )
        return self
