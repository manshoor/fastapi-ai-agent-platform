"""Exercise the destructive integration fixture's guard without starting infrastructure."""

from types import SimpleNamespace

import conftest
import pytest

import app.main
from app.config import Settings


class CreateAppReached(Exception):
    """Stop an approved destination before application construction or startup."""


@pytest.fixture
def cleanup_boundaries(monkeypatch):
    touched = []

    def forbidden(name):
        def fail(*args, **kwargs):
            touched.append(name)
            pytest.fail(f"Cleanup safety test touched forbidden boundary: {name}")

        return fail

    # Even an accidentally bypassed application interception must never connect
    # or invoke either destructive cleanup operation.
    application = SimpleNamespace(
        state=SimpleNamespace(
            engine=SimpleNamespace(begin=forbidden("database cleanup")),
            redis=SimpleNamespace(flushdb=forbidden("Redis cleanup")),
        )
    )

    def intercept_create_app(settings):
        touched.append("create_app")
        raise CreateAppReached(application)

    monkeypatch.setattr(app.main, "create_app", intercept_create_app)
    monkeypatch.setattr(app.main, "create_engine", forbidden("database engine"))
    monkeypatch.setattr(app.main.Redis, "from_url", forbidden("Redis client"))
    monkeypatch.setattr(conftest, "LifespanManager", forbidden("lifespan"))
    return touched


def cleanup_settings(database_url, redis_url="redis://redis-user:redis-secret@localhost/1"):
    return Settings(
        database_url=database_url,
        redis_url=redis_url,
        jwt_secret="ab" * 32,
    )


@pytest.mark.parametrize(
    "destination",
    [
        pytest.param("/app_test?database=app", id="query-production-override"),
        pytest.param("/app_test?database=other", id="query-other-override"),
        pytest.param("/app", id="production-database"),
        pytest.param("", id="missing-database"),
        pytest.param("/", id="empty-database"),
        pytest.param("?database=", id="empty-effective-database"),
        pytest.param(
            "/app_test?database=app_test&database=app",
            id="ambiguous-mixed-query-database",
        ),
        pytest.param(
            "/app_test?database=app_test&database=app_test",
            id="ambiguous-identical-query-database",
        ),
    ],
)
async def test_unsafe_database_rejected_before_any_side_effect(destination, cleanup_boundaries):
    url = f"postgresql+asyncpg://cleanup-user:cleanup-secret@localhost{destination}"
    settings = cleanup_settings(url)
    generator = conftest.app.__wrapped__(settings)
    try:
        with pytest.raises(RuntimeError) as rejected:
            await anext(generator)
        message = str(rejected.value)
        assert message
        for private in (url, "cleanup-user", "cleanup-secret", "redis-secret", settings.jwt_secret):
            assert private not in message
        assert "://" not in message
        assert cleanup_boundaries == []
    finally:
        await generator.aclose()


@pytest.mark.parametrize(
    "destination",
    [
        pytest.param("/app_test", id="direct-test-database"),
        pytest.param("/app?database=app_test", id="query-overrides-production-authority"),
        pytest.param("?database=app_test", id="query-supplies-missing-authority"),
        pytest.param("/app_test?database=app_test", id="single-test-query-database"),
        pytest.param("/app_test?database=", id="blank-query-retains-test-authority"),
    ],
)
async def test_effective_test_database_reaches_application_boundary(
    destination, cleanup_boundaries
):
    settings = cleanup_settings(
        f"postgresql+asyncpg://cleanup-user:cleanup-secret@localhost{destination}"
    )
    generator = conftest.app.__wrapped__(settings)
    try:
        with pytest.raises(CreateAppReached):
            await anext(generator)
        assert cleanup_boundaries == ["create_app"]
    finally:
        await generator.aclose()


@pytest.mark.parametrize("redis_database", ["0", "2", ""])
async def test_non_test_redis_rejected_before_any_side_effect(redis_database, cleanup_boundaries):
    redis_url = f"redis://redis-user:redis-secret@localhost/{redis_database}"
    settings = cleanup_settings(
        "postgresql+asyncpg://cleanup-user:cleanup-secret@localhost/app_test",
        redis_url=redis_url,
    )
    generator = conftest.app.__wrapped__(settings)
    try:
        with pytest.raises(RuntimeError) as rejected:
            await anext(generator)
        message = str(rejected.value)
        assert message
        for private in (
            settings.database_url,
            redis_url,
            "cleanup-user",
            "cleanup-secret",
            "redis-secret",
            settings.jwt_secret,
        ):
            assert private not in message
        assert "://" not in message
        assert cleanup_boundaries == []
    finally:
        await generator.aclose()
