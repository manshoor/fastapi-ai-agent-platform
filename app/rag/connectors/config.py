"""Strict operator configuration. Secrets are named by environment variable, never inlined."""

import re
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit
from uuid import UUID

import httpx
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

# An environment variable name: a DSN or token in its place is refused.
EnvName = Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9_]{0,63}$")]
ConnectorName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9-]{0,63}$")]
# Dotted path into a JSON response; "" is the response itself.
JSONPath = Annotated[str, StringConstraints(pattern=r"^([A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+)*)?$")]
FieldName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
MAX_CONNECTORS = 50
_SELECT = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


class ConnectorConfigError(ValueError):
    """The file is missing or invalid. The message names fields, never their values."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PostgresSource(_Strict):
    """One SELECT returning external_id, title, content; run READ ONLY with a timeout."""

    kind: Literal["postgres"]
    dsn_env: EnvName
    query: str = Field(min_length=1, max_length=4000)

    @field_validator("query")
    @classmethod
    def _one_select(cls, query: str) -> str:
        # The read-only transaction refuses writes; this keeps the wrapper well formed.
        if not _SELECT.match(query) or ";" in query or "\x00" in query:
            raise ValueError("query must be one SELECT statement")
        if "--" in query or "/*" in query:
            # A comment would cut off the wrapper's row cap and size bounds.
            raise ValueError("query must not contain comments")
        return query


class Pagination(_Strict):
    mode: Literal["none", "page", "cursor"] = "none"
    page_param: FieldName = "page"
    cursor_param: FieldName = "cursor"
    # Where the next cursor is in each response; required for cursor mode.
    cursor_path: JSONPath | None = None

    @model_validator(mode="after")
    def _cursor_path(self) -> "Pagination":
        if self.mode == "cursor" and not self.cursor_path:
            raise ValueError("cursor pagination needs cursor_path")
        return self


class HTTPSource(_Strict):
    """GET a JSON list, page by page. No redirects; https unless the host is allowed."""

    kind: Literal["http_json"]
    url: str = Field(min_length=1, max_length=2000)
    auth_header_env: EnvName | None = None
    items_path: JSONPath = ""
    id_field: FieldName = "id"
    title_field: FieldName = "title"
    content_field: FieldName = "content"
    pagination: Pagination = Pagination()
    max_pages: int = Field(default=10, ge=1, le=100)
    # Hosts that may use plain http, such as a local fake. https needs no entry.
    allowed_hosts: list[Annotated[str, StringConstraints(max_length=253)]] = Field(
        default_factory=list, max_length=20
    )

    @model_validator(mode="after")
    def _safe_url(self) -> "HTTPSource":
        if self.url != self.url.strip():
            raise ValueError("url must not have surrounding whitespace")
        try:
            # httpx must accept it too, or every fetch would fail at request time.
            httpx.URL(self.url).port  # noqa: B018
            parts = urlsplit(self.url)
            parts.port  # noqa: B018
        except (httpx.InvalidURL, ValueError):
            raise ValueError("url must be a valid URL") from None
        if parts.username is not None or parts.password is not None or not parts.hostname:
            raise ValueError("url must name a host and carry no credentials")
        if parts.scheme == "https":
            return self
        if parts.scheme == "http" and parts.hostname in self.allowed_hosts:
            return self
        raise ValueError("url must be https, or http to a host in allowed_hosts")


class Connector(_Strict):
    name: ConnectorName
    # Documents belong to this building and are owned by this manager of it.
    building_id: UUID
    owner_user_id: UUID
    interval_seconds: int = Field(ge=60, le=86400)
    max_records: int = Field(ge=1, le=5000)
    source: PostgresSource | HTTPSource = Field(discriminator="kind")


class ConnectorFile(_Strict):
    connectors: list[Connector] = Field(max_length=MAX_CONNECTORS)

    @model_validator(mode="after")
    def _unique_names(self) -> "ConnectorFile":
        names = [connector.name for connector in self.connectors]
        if len(names) != len(set(names)):
            raise ValueError("connector names must be unique")
        return self


def parse_connectors(raw: object) -> list[Connector]:
    try:
        return ConnectorFile.model_validate(raw).connectors
    except ValidationError as exc:
        # Locations and error types only: values may be secrets put in the wrong field.
        # An unknown key is reported by its parent: the key itself may be a secret.
        fields = sorted(
            {
                ".".join(
                    str(part)
                    for part in (
                        error["loc"][:-1] + ("<unknown field>",)
                        if error["type"] == "extra_forbidden"
                        else error["loc"]
                    )
                )
                for error in exc.errors()
            }
        )
        raise ConnectorConfigError(f"Invalid connector config at: {', '.join(fields)}") from None


def load_connectors(path: str) -> list[Connector]:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        raise ConnectorConfigError("Connector config cannot be read") from None
    return parse_connectors(raw)
