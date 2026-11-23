"""Langfuse overlay for the AI-local stack: config shape and launcher. No connections."""

import base64
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("ai_langfuse", ROOT / "scripts" / "ai_langfuse.py")
assert _SPEC is not None and _SPEC.loader is not None
launcher = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(launcher)

ADDON = "otlphttp/langfuse-local"
PUBLIC = "pk-lf-" + "a" * 64
SECRET = "sk-lf-" + "b" * 64
PROJECT = "lf-local-" + "c" * 24


def _yaml(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text())


def state_values() -> dict[str, str]:
    return {
        "LANGFUSE_LOCAL_PROJECT": PROJECT,
        "LANGFUSE_INIT_PROJECT_PUBLIC_KEY": PUBLIC,
        "LANGFUSE_INIT_PROJECT_SECRET_KEY": SECRET,
    }


@pytest.fixture
def state() -> dict[str, str]:
    return state_values()


# Collector add-on


def test_addon_only_adds_one_exporter_and_the_exporter_list():
    addon = _yaml("observability/collector.langfuse.yaml")
    assert set(addon) == {"exporters", "service"}
    assert set(addon["exporters"]) == {ADDON}
    assert set(addon["service"]) == {"pipelines"}
    assert set(addon["service"]["pipelines"]) == {"traces/langfuse"}
    assert set(addon["service"]["pipelines"]["traces/langfuse"]) == {"exporters"}


def test_addon_exporter_list_is_the_base_list_plus_the_addon():
    base = _yaml("observability/collector.yaml")["service"]["pipelines"]["traces/langfuse"]
    addon = _yaml("observability/collector.langfuse.yaml")["service"]["pipelines"]
    # A list in the second file replaces the first: drift in the base must show here.
    assert addon["traces/langfuse"]["exporters"] == [*base["exporters"], ADDON]


def test_addon_exporter_is_local_bounded_and_holds_no_key():
    exporter = _yaml("observability/collector.langfuse.yaml")["exporters"][ADDON]
    assert exporter["endpoint"] == "http://langfuse-web:3000/api/public/otel"
    assert exporter["headers"] == {
        "Authorization": "${env:LANGFUSE_OTLP_AUTH}",
        "x-langfuse-ingestion-version": "4",
    }
    assert exporter["timeout"] == "3s"
    assert exporter["retry_on_failure"]["max_elapsed_time"] == "5s"
    assert exporter["sending_queue"]["queue_size"] == 128


def test_addon_leaves_the_privacy_filter_to_the_base():
    addon = _yaml("observability/collector.langfuse.yaml")
    assert "processors" not in addon
    assert "processors" not in addon["service"]["pipelines"]["traces/langfuse"]
    base = _yaml("observability/collector.yaml")["service"]["pipelines"]["traces/langfuse"]
    assert "transform/langfuse" in base["processors"]


# Compose overlay


def test_overlay_changes_only_the_collector():
    overlay = _yaml("compose.ai.langfuse.yaml")
    assert set(overlay) == {"services", "networks"}
    assert set(overlay["services"]) == {"collector"}
    collector = overlay["services"]["collector"]
    assert set(collector) == {"command", "volumes", "environment", "networks"}
    assert "ports" not in collector


def test_overlay_loads_base_config_then_addon():
    collector = _yaml("compose.ai.langfuse.yaml")["services"]["collector"]
    assert collector["command"] == [
        "--config=/etc/otelcol-contrib/config.yaml",
        "--config=/etc/otelcol-contrib/langfuse.yaml",
    ]
    assert collector["volumes"] == [
        "./observability/collector.langfuse.yaml:/etc/otelcol-contrib/langfuse.yaml:ro"
    ]


def test_overlay_requires_launcher_values_and_an_external_network():
    overlay = _yaml("compose.ai.langfuse.yaml")
    collector = overlay["services"]["collector"]
    assert collector["environment"] == {
        "LANGFUSE_OTLP_AUTH": "${AI_LANGFUSE_OTLP_AUTH:?Start with scripts/ai_langfuse.py}",
        "HTTP_PROXY": "",
        "HTTPS_PROXY": "",
        "NO_PROXY": "langfuse-web",
        "http_proxy": "",
        "https_proxy": "",
        "no_proxy": "langfuse-web",
    }
    assert collector["networks"] == ["default", "langfuse_ingress"]
    assert overlay["networks"] == {
        "langfuse_ingress": {
            "external": True,
            "name": "${AI_LANGFUSE_NETWORK:?Start with scripts/ai_langfuse.py}",
        }
    }


# Launcher


def test_auth_is_basic_public_secret(state):
    value = launcher.otlp_auth(state)
    assert value.startswith("Basic ")
    assert base64.b64decode(value.removeprefix("Basic ")).decode() == f"{PUBLIC}:{SECRET}"


def test_child_environment_drops_ambient_settings(state):
    environ = {
        "PATH": "/bin",
        "HOME": "/home/x",
        "DOCKER_HOST": "unix:///d.sock",
        "AI_DB_PASSWORD": "ambient",
        "AI_OPENAI_API_KEY": "ambient-key",
        "COMPOSE_FILE": "compose.yaml",
        "COMPOSE_PROFILES": "connectors",
        "OPENAI_API_KEY": "ambient-key",
    }
    child = launcher.child_environment(state, launcher.Provider.SIMULATOR, environ)
    assert child == {
        "PATH": "/bin",
        "HOME": "/home/x",
        "DOCKER_HOST": "unix:///d.sock",
        "AI_LANGFUSE_OTLP_AUTH": launcher.otlp_auth(state),
        "AI_LANGFUSE_NETWORK": f"{PROJECT}_api_ingress",
    }


def test_child_environment_passes_the_live_key_only_when_live(state):
    environ = {"PATH": "/bin", "AI_OPENAI_API_KEY": "live-key"}
    child = launcher.child_environment(state, launcher.Provider.LIVE, environ)
    assert child["AI_OPENAI_API_KEY"] == "live-key"
    # Absent from the environment: .ai-local/.env supplies it.
    child = launcher.child_environment(state, launcher.Provider.LIVE, {"PATH": "/bin"})
    assert "AI_OPENAI_API_KEY" not in child


def test_compose_commands_simulator():
    collector, services = launcher.compose_commands(
        launcher.Provider.SIMULATOR, launcher.Ingress.LOOPBACK
    )
    prefix = [
        "docker", "compose", "--env-file", ".ai-local/.env", "-p", "fastapi-ai-local",
        "-f", "compose.ai.yaml", "-f", "compose.ai.langfuse.yaml",
    ]  # fmt: skip
    assert collector == prefix + ["up", "-d", "--wait", "--force-recreate", "collector"]
    assert services == prefix + [
        "up", "--build", "-d", "--wait",
        "api", "dispatcher", "worker", "rag-worker", "rag-dispatcher-sqs", "rag-worker-sqs",
    ]  # fmt: skip


def _files(command: list[str]) -> list[str]:
    return [command[i + 1] for i, part in enumerate(command) if part == "-f"]


def test_compose_commands_live_puts_the_overlay_after_live():
    for command in launcher.compose_commands(launcher.Provider.LIVE, launcher.Ingress.LOOPBACK):
        assert _files(command) == [
            "compose.ai.yaml", "compose.ai.live.yaml", "compose.ai.langfuse.yaml",
        ]  # fmt: skip


def test_compose_commands_edge_lists_the_edge_last_and_starts_it():
    collector, services = launcher.compose_commands(launcher.Provider.LIVE, launcher.Ingress.EDGE)
    for command in (collector, services):
        assert _files(command) == [
            "compose.ai.yaml", "compose.ai.live.yaml", "compose.ai.langfuse.yaml",
            "compose.edge.yaml",
        ]  # fmt: skip
    assert services[-1] == "edge"
    assert collector[-1] == "collector"


@pytest.fixture
def calls(monkeypatch, state):
    recorded = []
    monkeypatch.setattr(launcher.langfuse_local, "load_environment", lambda _: dict(state))
    monkeypatch.setattr(launcher.langfuse_local, "resume", lambda _: recorded.append("resume"))

    def run(command, **kwargs):
        recorded.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, "run", run)
    return recorded


def _launch() -> None:
    launcher.launch(
        Path(".langfuse-demo"),
        launcher.Provider.SIMULATOR,
        launcher.Ingress.LOOPBACK,
        {"PATH": "/bin"},
    )


def test_launch_resumes_langfuse_then_collector_then_services(calls):
    _launch()
    assert calls[0] == "resume"
    (collector, collector_kwargs), (services, services_kwargs) = calls[1:]
    assert collector[-1] == "collector"
    assert services[-1] == "rag-worker-sqs"
    for kwargs in (collector_kwargs, services_kwargs):
        assert kwargs["env"]["AI_LANGFUSE_NETWORK"] == f"{PROJECT}_api_ingress"
        assert set(kwargs["env"]) == {"PATH", "AI_LANGFUSE_OTLP_AUTH", "AI_LANGFUSE_NETWORK"}
        # Output is discarded: Compose may echo interpolated values.
        assert kwargs["stdout"] is launcher.subprocess.DEVNULL
        assert kwargs["stderr"] is launcher.subprocess.DEVNULL


@pytest.mark.parametrize("failing", ["load_environment", "resume"])
def test_unsafe_state_or_failed_resume_starts_nothing(monkeypatch, calls, failing):
    def fail(_):
        raise launcher.langfuse_local.ProvisionError("unsafe-private-state")

    monkeypatch.setattr(launcher.langfuse_local, failing, fail)
    with pytest.raises(launcher.LaunchError):
        _launch()
    assert not [call for call in calls if call != "resume"]


def test_failed_collector_stops_before_services(monkeypatch, calls):
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(launcher.subprocess, "run", run)
    with pytest.raises(launcher.LaunchError, match="^collector: .*exit 1"):
        _launch()
    assert len([call for call in calls if call != "resume"]) == 1


def test_failed_services_say_how_to_stop_the_collector(monkeypatch, calls):
    codes = iter([0, 2])

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=next(codes))

    monkeypatch.setattr(launcher.subprocess, "run", run)
    with pytest.raises(launcher.LaunchError) as error:
        _launch()
    assert str(error.value).startswith("services: docker compose failed (exit 2)")
    assert launcher.TEARDOWN in str(error.value)


@pytest.mark.parametrize(
    ("raised", "message"),
    [
        (launcher.subprocess.TimeoutExpired("docker", 1), "collector: docker compose timed out"),
        (FileNotFoundError("docker"), "collector: docker compose unavailable"),
    ],
)
def test_run_maps_timeout_and_missing_docker(monkeypatch, calls, raised, message):
    def run(command, **kwargs):
        raise raised

    monkeypatch.setattr(launcher.subprocess, "run", run)
    with pytest.raises(launcher.LaunchError, match=f"^{message}$"):
        _launch()


def test_main_never_prints_a_key(monkeypatch, capsys, calls):
    monkeypatch.setattr("sys.argv", ["ai_langfuse.py", "--langfuse-state", ".langfuse-demo"])
    launcher.main()
    output = capsys.readouterr()
    for secret in (PUBLIC, SECRET, launcher.otlp_auth(state_values())):
        assert secret not in output.out + output.err
