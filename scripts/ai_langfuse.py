"""Start the AI-local stack with its traces sent to a saved local Langfuse as well.

    python scripts/ai_langfuse.py --langfuse-state .langfuse-<name> [--live] [--edge]

    1. load the saved Langfuse state (owner, mode and descriptor checks)
    2. resume Langfuse and wait for web and worker; stop here on failure
    3. recreate the collector with compose.ai.langfuse.yaml
    4. start the API and workers with the same files

Spans pass the collector's existing allowlist first: no prompts, questions or answers.
Never prints a key or the state file. Teardown, in this order (volumes kept):

    docker compose --env-file .ai-local/.env -p fastapi-ai-local -f compose.ai.yaml down
    python scripts/langfuse_local.py down --directory .langfuse-<name>
"""

import argparse
import base64
import importlib.util
import os
import subprocess
from collections.abc import Mapping
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ".ai-local/.env"
PROJECT = "fastapi-ai-local"
BASE_FILE = "compose.ai.yaml"
LIVE_FILE = "compose.ai.live.yaml"
OVERLAY_FILE = "compose.ai.langfuse.yaml"
# Listed last, as compose.edge.yaml asks.
EDGE_FILE = "compose.edge.yaml"
SERVICES = ("api", "dispatcher", "worker", "rag-worker", "rag-dispatcher-sqs", "rag-worker-sqs")
TEARDOWN = f"docker compose --env-file {ENV_FILE} -p {PROJECT} -f {BASE_FILE} down"
# Langfuse's web joins this network of its Compose project.
INGRESS_NETWORK = "api_ingress"
LIVE_KEY = "AI_OPENAI_API_KEY"
# Docker client plumbing only: no ambient AI_*, COMPOSE_* or provider settings.
_PASSED = (
    "PATH",
    "HOME",
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
    "DOCKER_CONFIG",
    "DOCKER_TLS_VERIFY",
    "DOCKER_CERT_PATH",
    "SSH_AUTH_SOCK",
    "XDG_RUNTIME_DIR",
    "TMPDIR",
)
COLLECTOR_TIMEOUT_SECONDS = 300
SERVICES_TIMEOUT_SECONDS = 1200


def _load_langfuse_local():
    # A sibling script, not a package: load it by path.
    spec = importlib.util.spec_from_file_location(
        "langfuse_local", Path(__file__).resolve().parent / "langfuse_local.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


langfuse_local = _load_langfuse_local()


class Provider(Enum):
    SIMULATOR = "simulator"
    # Adds compose.ai.live.yaml: billable model calls.
    LIVE = "live"


class Ingress(Enum):
    # The API's own loopback port.
    LOOPBACK = "loopback"
    # Adds compose.edge.yaml: TLS on 8443, the API's plain port unpublished. Without it,
    # a launch would recreate the API without the edge.
    EDGE = "edge"


class LaunchError(RuntimeError):
    """A fixed, secret-free reason."""


def otlp_auth(state: Mapping[str, str]) -> str:
    pair = (
        f"{state['LANGFUSE_INIT_PROJECT_PUBLIC_KEY']}:{state['LANGFUSE_INIT_PROJECT_SECRET_KEY']}"
    )
    return "Basic " + base64.b64encode(pair.encode()).decode()


def child_environment(
    state: Mapping[str, str], provider: Provider, environ: Mapping[str, str]
) -> dict[str, str]:
    """Only these values beat .ai-local/.env, which keeps every other setting."""
    child = {key: environ[key] for key in _PASSED if key in environ}
    child["AI_LANGFUSE_OTLP_AUTH"] = otlp_auth(state)
    child["AI_LANGFUSE_NETWORK"] = f"{state['LANGFUSE_LOCAL_PROJECT']}_{INGRESS_NETWORK}"
    if provider is Provider.LIVE and environ.get(LIVE_KEY):
        # Otherwise compose.ai.live.yaml reads it from .ai-local/.env.
        child[LIVE_KEY] = environ[LIVE_KEY]
    return child


def compose_commands(provider: Provider, ingress: Ingress) -> list[list[str]]:
    files = [BASE_FILE]
    if provider is Provider.LIVE:
        files.append(LIVE_FILE)
    files.append(OVERLAY_FILE)
    services = list(SERVICES)
    if ingress is Ingress.EDGE:
        files.append(EDGE_FILE)
        services.append("edge")
    base = ["docker", "compose", "--env-file", ENV_FILE, "-p", PROJECT]
    for name in files:
        base += ["-f", name]
    return [
        # Recreated so it joins a Langfuse network made again after a down and resume.
        base + ["up", "-d", "--wait", "--force-recreate", "collector"],
        base + ["up", "--build", "-d", "--wait", *services],
    ]


def _run(step: str, command: list[str], environment: Mapping[str, str], timeout: int) -> None:
    try:
        result = subprocess.run(
            command,
            env=dict(environment),
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            # Compose may echo interpolated values on failure: keep its output off screen.
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise LaunchError(f"{step}: docker compose timed out") from None
    except OSError:
        raise LaunchError(f"{step}: docker compose unavailable") from None
    if result.returncode:
        raise LaunchError(f"{step}: docker compose failed (exit {result.returncode})")


def launch(
    state_directory: Path, provider: Provider, ingress: Ingress, environ: Mapping[str, str]
) -> None:
    try:
        state = langfuse_local.load_environment(state_directory)
        # Waits for web and worker; nothing of the AI stack starts if this fails.
        langfuse_local.resume(state_directory)
    except langfuse_local.ProvisionError as error:
        raise LaunchError(str(error)) from None
    environment = child_environment(state, provider, environ)
    collector, services = compose_commands(provider, ingress)
    _run("collector", collector, environment, COLLECTOR_TIMEOUT_SECONDS)
    try:
        _run("services", services, environment, SERVICES_TIMEOUT_SECONDS)
    except LaunchError as error:
        # The collector is up with the key in its environment: say how to stop it.
        raise LaunchError(f"{error}; the collector is still running. Stop: {TEARDOWN}") from None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--langfuse-state", type=Path, required=True)
    parser.add_argument(
        "--live", action="store_true", help="Add compose.ai.live.yaml (billable provider)"
    )
    parser.add_argument(
        "--edge", action="store_true", help="Add compose.edge.yaml (run init_edge_tls.py first)"
    )
    args = parser.parse_args()
    try:
        launch(
            args.langfuse_state,
            Provider.LIVE if args.live else Provider.SIMULATOR,
            Ingress.EDGE if args.edge else Ingress.LOOPBACK,
            os.environ,
        )
    except LaunchError as error:
        parser.exit(1, f"{error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted\n")
    print(f"Langfuse: {langfuse_local.WEB_URL} (Tracing -> Traces)")
    print(f"Login: {args.langfuse_state}/state.json (read it yourself; never paste it)")
    if args.edge:
        print("API: https://127.0.0.1:<EDGE_PORT> (8443 unless set)")
        return
    print("API: http://127.0.0.1:<AI_API_PORT> (8002 unless .ai-local/.env sets it)")


if __name__ == "__main__":
    main()
