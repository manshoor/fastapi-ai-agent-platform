import os
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy.engine import make_url

from app.config import Settings


@pytest.fixture(autouse=True)
def isolated_settings_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    fields = {name.lower() for name in Settings.model_fields}
    for key in tuple(os.environ):
        if key.lower() in fields:
            monkeypatch.delenv(key)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/1")
    monkeypatch.setenv("JWT_SECRET", "ab" * 32)


def components(monkeypatch: pytest.MonkeyPatch, password: str = "test-password") -> None:
    for key, value in {
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "DB_NAME": "app_test",
        "DB_USER": "app_test",
        "DB_PASSWORD": password,
    }.items():
        monkeypatch.setenv(key, value)


def test_component_only_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    components(monkeypatch)
    settings = Settings()
    url = settings.sqlalchemy_url
    assert url.drivername == "postgresql+asyncpg"
    assert (url.host, url.port, url.database, url.username, url.password) == (
        "localhost",
        5432,
        "app_test",
        "app_test",
        "test-password",
    )
    assert settings.db_pool_size == 10
    assert settings.cache_ttl_ms == 5000


def test_explicit_url_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    components(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://other:secret@db:5433/other")
    url = Settings().sqlalchemy_url
    assert (url.host, url.port, url.database, url.username, url.password) == (
        "db",
        5433,
        "other",
        "other",
        "secret",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param(field, None, id=f"{field}-missing")
        for field in ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")
    ]
    + [
        pytest.param(field, "", id=f"{field}-empty")
        for field in ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")
    ]
    + [
        pytest.param("DB_PORT", "bad", id="port-malformed"),
        pytest.param("DB_PORT", "0", id="port-too-low"),
        pytest.param("DB_PORT", "65536", id="port-too-high"),
    ],
)
@pytest.mark.parametrize("url_source", ["environment", "constructor"])
def test_explicit_url_ignores_environment_components(
    monkeypatch: pytest.MonkeyPatch, field: str, value: str | None, url_source: str
) -> None:
    components(monkeypatch)
    if value is None:
        monkeypatch.delenv(field)
    else:
        monkeypatch.setenv(field, value)
    explicit_url = "postgresql+asyncpg://winner:secret@chosen:5433/selected"
    if url_source == "environment":
        monkeypatch.setenv("DATABASE_URL", explicit_url)
        settings = Settings()
    else:
        settings = Settings(database_url=explicit_url)
    assert settings.sqlalchemy_url == make_url(explicit_url)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param(field, None, id=f"{field}-missing")
        for field in ("db_host", "db_port", "db_name", "db_user", "db_password")
    ]
    + [
        pytest.param(field, "", id=f"{field}-empty")
        for field in ("db_host", "db_port", "db_name", "db_user", "db_password")
    ]
    + [
        pytest.param("db_host", object(), id="host-invalid-type"),
        pytest.param("db_port", "bad", id="port-malformed"),
        pytest.param("db_port", 0, id="port-too-low"),
        pytest.param("db_port", 65536, id="port-too-high"),
        pytest.param("db_name", {}, id="name-invalid-type"),
        pytest.param("db_user", [], id="user-invalid-type"),
        pytest.param("db_password", [], id="password-invalid-type"),
    ],
)
@pytest.mark.parametrize("url_source", ["environment", "constructor"])
def test_explicit_url_ignores_constructor_components(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object, url_source: str
) -> None:
    supplied = {
        "db_host": "localhost",
        "db_port": 5432,
        "db_name": "app_test",
        "db_user": "app_test",
        "db_password": "test-password",
    }
    if value is None:
        supplied.pop(field)
    else:
        supplied[field] = value
    explicit_url = "postgresql+asyncpg://winner:secret@chosen:5433/selected"
    if url_source == "environment":
        monkeypatch.setenv("DATABASE_URL", explicit_url)
    else:
        supplied["database_url"] = explicit_url
    assert Settings(**supplied).sqlalchemy_url == make_url(explicit_url)


def test_constructor_url_overrides_environment_url_and_unused_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components(monkeypatch)
    monkeypatch.setenv("DB_PORT", "bad")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///unused.db")
    explicit_url = "postgresql+asyncpg://winner:secret@chosen:5433/selected"
    settings = Settings(database_url=explicit_url, db_host=object(), db_password=["unused"])
    assert settings.sqlalchemy_url == make_url(explicit_url)


@pytest.mark.parametrize("url_source", ["environment", "constructor"])
def test_empty_url_uses_components(monkeypatch: pytest.MonkeyPatch, url_source: str) -> None:
    components(monkeypatch)
    if url_source == "environment":
        monkeypatch.setenv("DATABASE_URL", "")
        settings = Settings()
    else:
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://unused:p@db/unused")
        settings = Settings(database_url="")
    assert settings.sqlalchemy_url == make_url(
        "postgresql+asyncpg://app_test:test-password@localhost:5432/app_test"
    )


@pytest.mark.parametrize("url_source", ["environment", "constructor"])
@pytest.mark.parametrize(
    "invalid_url",
    [
        "not-a-url",
        "sqlite:///unused.db",
        "postgresql://u:p@db/app",
        "postgresql+asyncpg://db:bad/app",
    ],
)
def test_invalid_explicit_url_does_not_fall_back_to_components(
    monkeypatch: pytest.MonkeyPatch, url_source: str, invalid_url: str
) -> None:
    components(monkeypatch)
    if url_source == "environment":
        monkeypatch.setenv("DATABASE_URL", invalid_url)
        supplied = {}
    else:
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://unused:p@db/unused")
        supplied = {"database_url": invalid_url}
    with pytest.raises(ValidationError):
        Settings(**supplied)


@pytest.mark.parametrize("port", ["bad", "0", "65536"])
@pytest.mark.parametrize("source", ["environment", "constructor"])
def test_component_only_invalid_port_rejected(
    monkeypatch: pytest.MonkeyPatch, port: str, source: str
) -> None:
    components(monkeypatch)
    if source == "environment":
        monkeypatch.setenv("DB_PORT", port)
        supplied = {}
    else:
        supplied = {"db_port": port}
    with pytest.raises(ValidationError):
        Settings(**supplied)


@pytest.mark.parametrize("field", ["db_host", "db_name", "db_user", "db_password"])
@pytest.mark.parametrize("source", ["environment", "constructor"])
def test_component_only_empty_credentials_rejected(
    monkeypatch: pytest.MonkeyPatch, field: str, source: str
) -> None:
    components(monkeypatch)
    if source == "environment":
        monkeypatch.setenv(field.upper(), "")
        supplied = {}
    else:
        supplied = {field: ""}
    with pytest.raises(ValidationError):
        Settings(**supplied)


@pytest.mark.parametrize("missing", ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"])
@pytest.mark.parametrize("source", ["environment", "constructor"])
def test_missing_component_rejected(
    monkeypatch: pytest.MonkeyPatch, missing: str, source: str
) -> None:
    if source == "environment":
        components(monkeypatch)
        monkeypatch.delenv(missing)
        supplied = {}
    else:
        supplied = {
            "db_host": "localhost",
            "db_port": 5432,
            "db_name": "app_test",
            "db_user": "app_test",
            "db_password": "test-password",
        }
        supplied.pop(missing.lower())
    with pytest.raises(ValidationError, match="Set DATABASE_URL or all five"):
        Settings(**supplied)


def test_reserved_password_characters_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    password = "p@ss:/?#% word+&=雪"
    components(monkeypatch, password)
    url = Settings().sqlalchemy_url
    assert url.password == password
    assert make_url(url.render_as_string(hide_password=False)).password == password


def test_environment_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    components(monkeypatch)
    monkeypatch.setenv("cache_ttl_ms", "123")
    assert Settings().cache_ttl_ms == 123


def test_mixed_case_url_environment_ignores_unused_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components(monkeypatch)
    monkeypatch.setenv("DaTaBaSe_Url", "postgresql+asyncpg://winner:secret@chosen/selected")
    monkeypatch.setenv("DB_PORT", "not-a-port")
    assert Settings().sqlalchemy_url == make_url(
        "postgresql+asyncpg://winner:secret@chosen/selected"
    )


@pytest.mark.parametrize("ttl", ["0", "5001"])
def test_cache_bound_rejected(monkeypatch: pytest.MonkeyPatch, ttl: str) -> None:
    components(monkeypatch)
    monkeypatch.setenv("CACHE_TTL_MS", ttl)
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize("secret", ["ab" * 31, "z" * 64, "a" * 65])
def test_invalid_signing_secret_rejected(monkeypatch: pytest.MonkeyPatch, secret: str) -> None:
    components(monkeypatch)
    monkeypatch.setenv("JWT_SECRET", secret)
    with pytest.raises(ValidationError):
        Settings()


def test_no_implicit_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("DATABASE_URL=postgresql+asyncpg://u:p@db/app\n")
    with pytest.raises(ValidationError, match="Set DATABASE_URL or all five"):
        Settings()
