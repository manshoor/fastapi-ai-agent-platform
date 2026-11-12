"""Provision an isolated, real local Langfuse without loading any dotenv file.

Import generate_environment(), retain its result privately, then call start()/stop().
CLI: python scripts/langfuse_local.py up --directory .langfuse-local-<unique-name>
     python scripts/langfuse_local.py down --directory .langfuse-local-<unique-name>
Stopping preserves all volumes and private state. Add --remove-volumes to down only
when explicitly deleting this isolated project's stored observations.
The optional state.json holds fresh credentials (0600 inside a new 0700 directory).
Never print that file or run Compose config/inspect against this secret environment.
Docker logs are disabled; startup output is discarded, including on failure.
This local single-node stack is not a production deployment or a media-upload demo.

Sources:
https://github.com/langfuse/langfuse/blob/v4.72.0/docker-compose.yml
https://langfuse.com/self-hosting/administration/headless-initialization
https://langfuse.com/self-hosting/configuration/health-readiness-endpoints
https://github.com/langfuse/langfuse/blob/v4.72.0/fern/apis/server/definition/api.yml
"""

import argparse
import json
import os
import re
import secrets
import stat
import subprocess
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = ROOT / "compose.langfuse.yaml"
WEB_URL = "http://127.0.0.1:8005"
READINESS_URL = WEB_URL + "/api/public/ready"
LANGFUSE_VERSION = "4.72.0"
_PROJECT_PATTERN = re.compile(r"lf-local-[0-9a-f]{24}\Z")
_SECRET_LENGTHS = {
    "POSTGRES_PASSWORD": 64,
    "CLICKHOUSE_PASSWORD": 64,
    "REDIS_AUTH": 64,
    "MINIO_ROOT_USER": 32,
    "MINIO_ROOT_PASSWORD": 64,
    "SALT": 64,
    "ENCRYPTION_KEY": 64,
    "NEXTAUTH_SECRET": 64,
    "LANGFUSE_INIT_USER_PASSWORD": 64,
}
_ENVIRONMENT_KEYS = set(_SECRET_LENGTHS) | {
    "LANGFUSE_LOCAL_PROJECT",
    "LANGFUSE_INIT_ORG_ID",
    "LANGFUSE_INIT_PROJECT_ID",
    "LANGFUSE_INIT_PROJECT_PUBLIC_KEY",
    "LANGFUSE_INIT_PROJECT_SECRET_KEY",
    "LANGFUSE_INIT_USER_EMAIL",
}
# Only projects created in this process or explicitly loaded from private state may
# be destroyed. A name prefix alone never authorizes deleting Docker resources.
_OWNED_PROJECTS: set[str] = set()


class ProvisionError(RuntimeError):
    """Fixed, secret-free error category and optional subprocess exit code."""

    def __init__(self, category: str, returncode: int | None = None) -> None:
        self.category = category
        self.returncode = returncode
        suffix = "" if returncode is None else f" (exit {returncode})"
        super().__init__(f"Langfuse {category}{suffix}")


def generate_environment() -> dict[str, str]:
    """Return fresh credentials in memory; never inspect root .env or provider keys."""
    suffix = secrets.token_hex(12)
    values = {key: secrets.token_hex(length // 2) for key, length in _SECRET_LENGTHS.items()}
    values.update(
        LANGFUSE_LOCAL_PROJECT=f"lf-local-{suffix}",
        LANGFUSE_INIT_ORG_ID=f"local-org-{suffix}",
        LANGFUSE_INIT_PROJECT_ID=f"local-project-{suffix}",
        LANGFUSE_INIT_PROJECT_PUBLIC_KEY=f"pk-lf-{secrets.token_hex(32)}",
        LANGFUSE_INIT_PROJECT_SECRET_KEY=f"sk-lf-{secrets.token_hex(32)}",
        LANGFUSE_INIT_USER_EMAIL=f"operator-{suffix}@example.invalid",
    )
    return values


def compose_command(project: str) -> list[str]:
    """Return a direct-binary argv with explicit isolation and no dotenv loading."""
    if not isinstance(project, str) or not _PROJECT_PATTERN.fullmatch(project):
        raise ProvisionError("invalid-project")
    return [
        "docker",
        "compose",
        "--env-file",
        os.devnull,
        "--project-directory",
        str(ROOT),
        "--project-name",
        project,
        "--file",
        str(COMPOSE_FILE),
    ]


def _validate_environment(project: str, environment: Mapping[str, str]) -> None:
    compose_command(project)
    if set(environment) != _ENVIRONMENT_KEYS:
        raise ProvisionError("invalid-environment")
    if any(not isinstance(value, str) for value in environment.values()):
        raise ProvisionError("invalid-environment")
    suffix = project.removeprefix("lf-local-")
    expected = {
        "LANGFUSE_LOCAL_PROJECT": project,
        "LANGFUSE_INIT_ORG_ID": f"local-org-{suffix}",
        "LANGFUSE_INIT_PROJECT_ID": f"local-project-{suffix}",
        "LANGFUSE_INIT_USER_EMAIL": f"operator-{suffix}@example.invalid",
    }
    if any(environment[key] != value for key, value in expected.items()):
        raise ProvisionError("invalid-environment")
    for key, length in _SECRET_LENGTHS.items():
        if re.fullmatch(rf"[0-9a-f]{{{length}}}", environment[key]) is None:
            raise ProvisionError("invalid-environment")
    for key, prefix in (
        ("LANGFUSE_INIT_PROJECT_PUBLIC_KEY", "pk-lf-"),
        ("LANGFUSE_INIT_PROJECT_SECRET_KEY", "sk-lf-"),
    ):
        if re.fullmatch(prefix + r"[0-9a-f]{64}", environment[key]) is None:
            raise ProvisionError("invalid-environment")


def child_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Pass Docker client plumbing, not ambient Compose/provider/OTLP settings."""
    allowed = (
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
        "SYSTEMROOT",
    )
    child = {key: os.environ[key] for key in allowed if key in os.environ}
    child.update(environment)
    child["COMPOSE_DISABLE_ENV_FILE"] = "true"
    child["COMPOSE_ANSI"] = "never"
    return child


def _run(
    command: list[str],
    environment: Mapping[str, str],
    *,
    capture: bool = False,
    timeout: int = 60,
) -> bytes:
    try:
        result = subprocess.run(
            command,
            env=child_environment(environment),
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise ProvisionError("command-timeout") from None
    except OSError:
        raise ProvisionError("command-unavailable") from None
    if result.returncode:
        raise ProvisionError("command-failed", result.returncode)
    return result.stdout if capture else b""


def start(
    project: str,
    environment: Mapping[str, str],
    *,
    wait_seconds: int = 240,
) -> None:
    """Start only a fresh project; wait for web AND ingestion-worker health.

    Startup failure stops owned containers but preserves volumes and credentials.
    The caller retains the environment for readback authentication and later stop().
    """
    _validate_environment(project, environment)
    if (
        isinstance(wait_seconds, bool)
        or not isinstance(wait_seconds, int)
        or not 1 <= wait_seconds <= 900
    ):
        raise ProvisionError("invalid-wait")
    if project in _OWNED_PROJECTS:
        raise ProvisionError("project-already-owned")
    label = f"label=com.docker.compose.project={project}"
    for command in (
        ["docker", "ps", "--all", "--quiet", "--filter", label],
        ["docker", "volume", "ls", "--quiet", "--filter", label],
        ["docker", "network", "ls", "--quiet", "--filter", label],
    ):
        if _run(command, environment, capture=True).strip():
            raise ProvisionError("project-already-exists")
    _OWNED_PROJECTS.add(project)
    try:
        _run(
            compose_command(project)
            + [
                "up",
                "--detach",
                "--wait",
                "--wait-timeout",
                str(wait_seconds),
            ],
            environment,
            timeout=wait_seconds + 600,
        )
    except (ProvisionError, KeyboardInterrupt):
        try:
            stop(project, environment)
        except ProvisionError:
            raise ProvisionError("startup-and-cleanup-failed") from None
        raise


def stop(
    project: str,
    environment: Mapping[str, str],
    *,
    remove_volumes: bool = False,
) -> None:
    """Stop an owned project; delete its volumes only with explicit opt-in."""
    _validate_environment(project, environment)
    if project not in _OWNED_PROJECTS:
        raise ProvisionError("project-not-owned")
    command = compose_command(project) + ["down", "--timeout", "30"]
    if remove_volumes:
        command.append("--volumes")
    _run(command, environment, timeout=120)
    if remove_volumes:
        _OWNED_PROJECTS.remove(project)


def _check_private(metadata: os.stat_result, *, directory: bool) -> None:
    expected = 0o700 if directory else 0o600
    kind_ok = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    if (
        not kind_ok
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != expected
        or (not directory and metadata.st_nlink != 1)
    ):
        raise ProvisionError("unsafe-private-state")


def _state_directory(directory: Path) -> None:
    if (
        len(directory.parts) != 1
        or re.fullmatch(r"\.langfuse-[a-z0-9][a-z0-9-]{0,79}", directory.name) is None
    ):
        raise ProvisionError("unsafe-private-state")


def save_environment(directory: Path, environment: Mapping[str, str]) -> None:
    """Create private state exclusively; never reuse or repair an existing path."""
    _state_directory(directory)
    _validate_environment(environment.get("LANGFUSE_LOCAL_PROJECT", ""), environment)
    try:
        directory.mkdir(mode=0o700)
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            _check_private(os.fstat(directory_fd), directory=True)
            state_fd = os.open(
                "state.json",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory_fd,
            )
            with os.fdopen(state_fd, "w") as handle:
                _check_private(os.fstat(handle.fileno()), directory=False)
                json.dump(dict(environment), handle)
                handle.write("\n")
        finally:
            os.close(directory_fd)
    except OSError:
        raise ProvisionError("private-state-create-failed") from None


def load_environment(directory: Path) -> dict[str, str]:
    """Load only current-owner 0700/0600 descriptor-checked, nonsymlink state."""
    _state_directory(directory)
    try:
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            _check_private(os.fstat(directory_fd), directory=True)
            state_fd = os.open(
                "state.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd
            )
            with os.fdopen(state_fd, "r") as handle:
                _check_private(os.fstat(handle.fileno()), directory=False)
                values = json.load(handle)
        finally:
            os.close(directory_fd)
    except (OSError, ValueError, UnicodeError):
        raise ProvisionError("private-state-read-failed") from None
    if not isinstance(values, dict):
        raise ProvisionError("invalid-environment")
    _validate_environment(values.get("LANGFUSE_LOCAL_PROJECT", ""), values)
    return values


def resume(directory: Path) -> None:
    """Explicitly restart a saved local project without regenerating credentials or data."""
    environment = load_environment(directory)
    project = environment["LANGFUSE_LOCAL_PROJECT"]
    _OWNED_PROJECTS.add(project)
    _run(
        compose_command(project) + ["up", "--detach", "--wait", "--wait-timeout", "240"],
        environment,
        timeout=840,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("up", "resume", "down"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument(
        "--remove-volumes", action="store_true", help="Explicitly delete owned data on down"
    )
    args = parser.parse_args()
    if args.action != "down" and args.remove_volumes:
        parser.error("--remove-volumes is only valid with down")
    try:
        if args.action == "up":
            environment = generate_environment()
            save_environment(args.directory, environment)
            project = environment["LANGFUSE_LOCAL_PROJECT"]
            start(project, environment)
            print(f"Langfuse {LANGFUSE_VERSION} ready: {WEB_URL}; project {project}")
            print("Fresh login/API credentials retained only in private state.json.")
        elif args.action == "resume":
            resume(args.directory)
            print(f"Saved Langfuse project ready: {WEB_URL}; volumes retained.")
        else:
            environment = load_environment(args.directory)
            project = environment["LANGFUSE_LOCAL_PROJECT"]
            _OWNED_PROJECTS.add(project)
            stop(project, environment, remove_volumes=args.remove_volumes)
            print(
                "Owned Langfuse containers and volumes removed; private state retained."
                if args.remove_volumes
                else "Owned Langfuse containers removed; volumes and private state retained."
            )
    except ProvisionError as error:
        parser.exit(1, f"{error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Langfuse interrupted\n")


if __name__ == "__main__":
    main()
