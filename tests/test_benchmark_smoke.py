"""Connection-free regressions for benchmark isolation before Docker startup."""

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "benchmark_smoke", Path(__file__).resolve().parents[1] / "scripts" / "benchmark_smoke.py"
)
assert _spec is not None and _spec.loader is not None
benchmark_smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(benchmark_smoke)
PROJECT = "fastapi-bench-smoke-test"
ARTIFACTS = Path("/private/benchmark-artifacts")


@pytest.mark.parametrize(
    ("arguments", "exit_code"),
    [(["--help"], 0), (["--users", "100"], 2), (["--scenario", "5000"], 2)],
)
def test_cli_never_runs_setup_for_help_or_a_higher_load(monkeypatch, arguments, exit_code):
    monkeypatch.setattr(sys, "argv", ["benchmark_smoke.py", *arguments])

    def forbidden(*args, **kwargs):
        pytest.fail("CLI parsing reached benchmark setup")

    monkeypatch.setattr(benchmark_smoke.os, "chdir", forbidden)
    with pytest.raises(SystemExit) as stopped:
        benchmark_smoke.main()
    assert stopped.value.code == exit_code


def test_default_scenario_remains_the_bounded_20_user_smoke(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark_smoke.py"])
    name, scenario = benchmark_smoke.parse_scenario()
    assert name == "smoke"
    assert (scenario["users"], scenario["duration_seconds"]) == (20, 90)


def test_explicit_1000_scenario_is_fixed_and_has_enough_ramp_and_measurement_time(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark_smoke.py", "--scenario", "1000"])
    name, scenario = benchmark_smoke.parse_scenario()
    assert name == "1000"
    assert (scenario["users"], scenario["spawn_rate"], scenario["seed_users"]) == (1000, 50, 5000)
    assert scenario["warmup_seconds"] == 90
    assert scenario["duration_seconds"] == 480
    assert scenario["minimum_steady_seconds"] == 300
    assert scenario["users"] / scenario["spawn_rate"] < scenario["warmup_seconds"]
    assert (
        scenario["duration_seconds"] - scenario["warmup_seconds"]
        > scenario["minimum_steady_seconds"]
    )


def test_explicit_2500_scenario_is_fixed_and_preserves_the_benchmark_corpus(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark_smoke.py", "--scenario", "2500"])
    name, scenario = benchmark_smoke.parse_scenario()
    assert name == "2500"
    assert (scenario["users"], scenario["spawn_rate"], scenario["seed_users"]) == (2500, 100, 5000)
    assert scenario["warmup_seconds"] == 120
    assert scenario["duration_seconds"] == 780
    assert scenario["minimum_steady_seconds"] == 600
    assert scenario["maximum_failure_ratio"] == "0.01"
    assert scenario["zero_failures"] == "false"
    assert scenario["users"] / scenario["spawn_rate"] < scenario["warmup_seconds"]
    assert (
        scenario["duration_seconds"] - scenario["warmup_seconds"]
        > scenario["minimum_steady_seconds"]
    )


@pytest.fixture
def topology():
    environment = {
        "DATABASE_URL": "",
        "DB_HOST": "db",
        "DB_PORT": "5432",
        "DB_NAME": "app",
        "DB_USER": "app",
        "DB_PASSWORD": "test-only",
        "REDIS_URL": "redis://redis:6379/0",
        "AUTH_TOKEN_TTL_SECONDS": "7200",
    }
    return {
        "name": PROJECT,
        "networks": {"default": {"name": f"{PROJECT}_default"}},
        "volumes": {"benchmark_postgres_data": {"name": f"{PROJECT}_benchmark_postgres_data"}},
        "services": {
            "db": {
                "networks": {"default": None},
                "environment": {"POSTGRES_PASSWORD": "test-only"},
                "volumes": [
                    {
                        "type": "volume",
                        "source": "benchmark_postgres_data",
                        "target": "/var/lib/postgresql",
                    }
                ],
            },
            "redis": {"networks": {"default": None}},
            "migrate": {"networks": {"default": None}, "environment": deepcopy(environment)},
            "api": {
                "networks": {"default": None},
                "environment": deepcopy(environment),
                "ports": [{"host_ip": "127.0.0.1", "published": "8001", "target": 8000}],
            },
            "tools": {
                "networks": {"default": None},
                "environment": deepcopy(environment),
                "volumes": [{"type": "bind", "source": str(ARTIFACTS), "target": "/artifacts"}],
            },
        },
    }


def test_accepts_isolated_benchmark_topology(topology):
    benchmark_smoke.validate_topology(topology, PROJECT, ARTIFACTS)


@pytest.mark.parametrize("service", ["api", "migrate", "tools"])
@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("DATABASE_URL", "postgresql+asyncpg://u:example@normal/app"),
        ("DB_HOST", "normal-db"),
        ("REDIS_URL", "redis://normal-redis:6379/0"),
    ],
)
def test_refuses_normal_dependency_overrides(topology, service, key, value):
    topology["services"][service]["environment"][key] = value
    with pytest.raises(ValueError):
        benchmark_smoke.validate_topology(topology, PROJECT, ARTIFACTS)


@pytest.mark.parametrize("resource", ["volume", "network"])
@pytest.mark.parametrize("override", ["normal-name", "external"])
def test_refuses_shared_storage_and_networks(topology, resource, override):
    entry = (
        topology["volumes"]["benchmark_postgres_data"]
        if resource == "volume"
        else topology["networks"]["default"]
    )
    entry["name" if override == "normal-name" else "external"] = (
        "pythonfastapi_default" if override == "normal-name" else True
    )
    with pytest.raises(ValueError):
        benchmark_smoke.validate_topology(topology, PROJECT, ARTIFACTS)


@pytest.mark.parametrize("service", ["db", "redis"])
def test_refuses_dependency_host_ports(topology, service):
    topology["services"][service]["ports"] = [{"published": "5433", "target": 5432}]
    with pytest.raises(ValueError):
        benchmark_smoke.validate_topology(topology, PROJECT, ARTIFACTS)


def test_refuses_normal_project_database_mount(topology):
    topology["services"]["db"]["volumes"][0]["source"] = "pythonfastapi_postgres_data"
    with pytest.raises(ValueError):
        benchmark_smoke.validate_topology(topology, PROJECT, ARTIFACTS)


def test_refuses_public_api_binding(topology):
    topology["services"]["api"]["ports"][0]["host_ip"] = "0.0.0.0"
    with pytest.raises(ValueError):
        benchmark_smoke.validate_topology(topology, PROJECT, ARTIFACTS)
