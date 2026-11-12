from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings

from app.config import DatabaseSettings


class ProviderSettings(BaseSettings):
    model_config = DatabaseSettings.model_config

    ai_model: str = Field(default="gpt-5.6-terra", min_length=1, max_length=128)
    ai_prompt_version: str = Field(default="item-rewrite-v1", min_length=1, max_length=64)
    ai_schema_version: str = Field(default="1", min_length=1, max_length=16)
    ai_provider_timeout_seconds: float = Field(default=30, gt=0, le=60)
    ai_max_output_tokens: int = Field(default=4096, ge=1, le=16384)
    openai_api_key: SecretStr | None = None
    openai_base_url: str = Field(default="https://api.openai.com/v1", repr=False)


class AISettings(DatabaseSettings, ProviderSettings):
    db_pool_size: int = Field(default=4, ge=1)
    ai_enabled: bool = False
    ai_global_limit: int = Field(default=1000, ge=1, le=100000)
    ai_owner_limit: int = Field(default=5, ge=1, le=1000)
    ai_concurrency: int = Field(default=2, ge=1, le=32)
    ai_lease_seconds: float = Field(default=90, gt=0)
    ai_max_attempts: int = Field(default=3, ge=1, le=3)
    ai_queue_timeout_seconds: float = Field(default=86400, gt=0, le=86400)
    ai_content_retention_seconds: float = Field(default=604800, gt=0, le=604800)
    ai_metadata_retention_seconds: float = Field(default=2592000, ge=2592000)
    ai_poll_seconds: float = Field(default=1, gt=0, le=60)
    ai_redispatch_seconds: float = Field(default=60, gt=0)
    kafka_bootstrap_servers: str = Field(default="kafka:9092", min_length=1, repr=False)
    kafka_request_topic: str = "ai.item-rewrite.requested.v1"
    kafka_dead_letter_topic: str = "ai.item-rewrite.dead-letter.v1"
    kafka_group_id: str = "item-rewrite-worker-v1"

    @model_validator(mode="after")
    def validate_lease(self) -> "AISettings":
        if self.ai_lease_seconds <= self.ai_provider_timeout_seconds + 5:
            raise ValueError(
                "AI_LEASE_SECONDS must exceed provider timeout by more than five seconds"
            )
        return self
