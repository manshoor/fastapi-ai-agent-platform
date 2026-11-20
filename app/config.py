import re
from collections.abc import Mapping
from typing import Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False, env_file=None, extra="ignore", hide_input_in_errors=True
    )

    database_url: str | None = Field(default=None, repr=False)
    db_host: str | None = None
    db_port: int | None = Field(default=None, ge=1, le=65535)
    db_name: str | None = None
    db_user: str | None = None
    db_password: SecretStr | None = None
    db_pool_size: int = Field(default=10, ge=1)
    db_pool_timeout_seconds: float = Field(default=1, gt=0)
    db_statement_timeout_ms: int = Field(default=2000, ge=1)

    @model_validator(mode="before")
    @classmethod
    def select_database_source(cls, values: object) -> object:
        if isinstance(values, Mapping) and values.get("database_url"):
            # Settings sources have already resolved constructor/environment precedence.
            # Unused components must not undergo field validation.
            values = dict(values)
            for field in ("db_host", "db_port", "db_name", "db_user", "db_password"):
                values.pop(field, None)
        return values

    @model_validator(mode="after")
    def validate_database(self) -> Self:
        if self.database_url:
            try:
                url = make_url(self.database_url)
            except ArgumentError as exc:
                raise ValueError("DATABASE_URL must be a valid SQLAlchemy URL") from exc
            if url.drivername != "postgresql+asyncpg":
                raise ValueError("DATABASE_URL must use postgresql+asyncpg")
        elif not all((self.db_host, self.db_port, self.db_name, self.db_user, self.db_password)):
            raise ValueError("Set DATABASE_URL or all five DB_HOST/PORT/NAME/USER/PASSWORD fields")
        elif not self.db_password.get_secret_value():
            raise ValueError("DB_PASSWORD must not be empty")
        return self

    @property
    def sqlalchemy_url(self) -> URL:
        if self.database_url:
            return make_url(self.database_url)
        return URL.create(
            "postgresql+asyncpg",
            username=self.db_user,
            password=self.db_password.get_secret_value() if self.db_password else None,
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
        )


class Settings(DatabaseSettings):
    redis_url: str = Field(min_length=1, repr=False)
    jwt_secret: str = Field(repr=False)
    redis_max_connections: int = Field(default=50, ge=1)
    redis_timeout_seconds: float = Field(default=0.1, gt=0)
    cache_ttl_ms: int = Field(default=5000, ge=1, le=5000)
    auth_token_ttl_seconds: int = Field(default=900, ge=1, le=86400)
    auth_hash_concurrency: int = Field(default=2, ge=1)
    auth_hash_queue_timeout_seconds: float = Field(default=0.1, gt=0)
    # Failed password checks per account, at login and authenticator enrollment, before
    # the account answers 429 until the oldest failure leaves the window.
    auth_password_failure_limit: int = Field(default=10, ge=1, le=1000)
    auth_password_failure_window_seconds: float = Field(default=900, gt=0, le=86400)
    jwt_issuer: str = Field(default="fastapi-crud", min_length=1)
    jwt_audience: str = Field(default="fastapi-crud-api", min_length=1)
    ai_enabled: bool = False
    rag_enabled: bool = False
    assistant_enabled: bool = False
    # Demo-only: lets any account create a demo residence for itself.
    assistant_demo_enabled: bool = False
    # Eval-only: a hidden query route that also returns the passages sent to the model.
    rag_diagnostics_enabled: bool = False

    @field_validator("jwt_secret")
    @classmethod
    def validate_jwt_secret(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-fA-F]{64,}", value) or len(value) % 2:
            raise ValueError("JWT_SECRET must contain at least 32 bytes encoded as hex")
        return value
