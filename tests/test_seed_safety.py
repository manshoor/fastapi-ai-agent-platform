"""Exercise the seed destination guard without engines or network connections."""

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy.engine import make_url

# Scripts are shipped in the tools image but are not an installed Python package.
_spec = importlib.util.spec_from_file_location(
    "seed_load", Path(__file__).resolve().parents[1] / "scripts" / "seed_load.py"
)
assert _spec is not None and _spec.loader is not None
seed_load = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seed_load)


@pytest.mark.parametrize(
    "destination",
    [
        "localhost/app",
        "db/app_test",
        "127.0.0.1/app",
        "[::1]/app_test",
        "localhost/app?host=db&database=app_test",
        "remote.example/production?host=localhost&database=app",
        "localhost/app?host=localhost:5432&host=db:5432",
        "localhost/app?host=localhost,127.0.0.1&port=5432,5432",
    ],
)
def test_effective_local_destinations_accepted(destination: str) -> None:
    seed_load.validate_destination(
        make_url(f"postgresql+asyncpg://u:secret@{destination}"), allow_remote=False
    )


UNSAFE_DESTINATIONS = [
    "localhost/app?host=remote.example",
    "localhost/app?database=production",
    "localhost/app?host=remote.example&database=production",
    "localhost/app?host=localhost:5432&host=remote.example:5432",
    "localhost/app?host=remote.example:5432&host=localhost:5432",
    "localhost/app?host=localhost,remote.example&port=5432,5432",
    "remote.example/production",
]


@pytest.mark.parametrize("destination", UNSAFE_DESTINATIONS)
def test_effective_remote_destinations_rejected(destination: str, capsys) -> None:
    with pytest.raises(ValueError, match="use --allow-remote explicitly") as raised:
        seed_load.validate_destination(
            make_url(f"postgresql+asyncpg://u:secret@{destination}"), allow_remote=False
        )
    output = capsys.readouterr()
    assert "secret" not in str(raised.value) + output.out + output.err
    assert "postgresql" not in str(raised.value) + output.out + output.err


@pytest.mark.parametrize("destination", UNSAFE_DESTINATIONS)
def test_explicit_remote_opt_in_accepted(destination: str) -> None:
    seed_load.validate_destination(
        make_url(f"postgresql+asyncpg://u:secret@{destination}"), allow_remote=True
    )


async def test_guard_precedes_output_and_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    url = make_url("postgresql+asyncpg://u:secret@localhost/app?host=remote.example")
    monkeypatch.setattr(seed_load, "Settings", lambda: SimpleNamespace(sqlalchemy_url=url))

    def forbidden(*args, **kwargs):
        pytest.fail("unsafe destination reached output creation or engine construction")

    monkeypatch.setattr(seed_load, "open_output", forbidden)
    monkeypatch.setattr(seed_load, "create_engine", forbidden)
    with pytest.raises(ValueError, match="use --allow-remote explicitly"):
        await seed_load.seed(argparse.Namespace(allow_remote=False))
