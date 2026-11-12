"""Local provisioning regressions: no Docker, network, dotenv, or paid requests."""

import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "langfuse_local.py"
_SPEC = importlib.util.spec_from_file_location("langfuse_local", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
langfuse = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(langfuse)


@pytest.fixture(autouse=True)
def isolated_ownership(monkeypatch):
    monkeypatch.setattr(langfuse, "_OWNED_PROJECTS", set())


@pytest.fixture
def environment():
    return langfuse.generate_environment()


@pytest.fixture
def docker(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=b"")

    monkeypatch.setattr(langfuse.subprocess, "run", run)
    return calls


def test_generated_identity_and_credentials_are_fresh_and_in_memory(monkeypatch):
    def forbid(*args, **kwargs):
        pytest.fail("generating credentials must not read any file")

    monkeypatch.setattr(Path, "open", forbid)
    monkeypatch.setattr(os, "open", forbid)
    monkeypatch.setenv("LANGFUSE_INIT_PROJECT_SECRET_KEY", "ambient-secret-not-approved")
    first = langfuse.generate_environment()
    second = langfuse.generate_environment()
    assert first.keys() == second.keys() == langfuse._ENVIRONMENT_KEYS
    assert all(first[key] != second[key] for key in first)
    assert re.fullmatch(r"lf-local-[0-9a-f]{24}", first["LANGFUSE_LOCAL_PROJECT"])
    assert re.fullmatch(r"[0-9a-f]{64}", first["ENCRYPTION_KEY"])
    assert first["LANGFUSE_INIT_PROJECT_PUBLIC_KEY"].startswith("pk-lf-")
    assert first["LANGFUSE_INIT_PROJECT_SECRET_KEY"].startswith("sk-lf-")
    assert "ambient-secret-not-approved" not in first.values()


def test_compose_argv_never_implicitly_loads_dotenv(environment):
    project = environment["LANGFUSE_LOCAL_PROJECT"]
    command = langfuse.compose_command(project)
    assert command[:4] == ["docker", "compose", "--env-file", os.devnull]
    assert command[command.index("--project-name") + 1] == project
    assert command[command.index("--file") + 1] == str(langfuse.COMPOSE_FILE)
    assert command[command.index("--project-directory") + 1] == str(langfuse.ROOT)
    assert ".env" not in command
    assert not any(value in " ".join(command) for value in environment.values() if value != project)


@pytest.mark.parametrize(
    "project",
    ["", "fastapi", "lf-local-prod", "../lf-local-x", "--help", "lf-local-" + "a" * 24 + "\n"],
)
def test_arbitrary_projects_are_rejected_before_docker(project, docker):
    with pytest.raises(langfuse.ProvisionError, match="invalid-project"):
        langfuse.compose_command(project)
    assert docker == []


def test_child_environment_discards_ambient_destinations_and_secrets(monkeypatch, environment):
    for key in (
        "COMPOSE_FILE",
        "COMPOSE_ENV_FILES",
        "COMPOSE_PROJECT_NAME",
        "DATABASE_URL",
        "AI_OPENAI_API_KEY",
        "LANGSMITH_API_KEY",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "LANGFUSE_INIT_PROJECT_SECRET_KEY",
    ):
        monkeypatch.setenv(key, "unsafe-ambient-value")
    monkeypatch.setenv("DOCKER_CONFIG", "/private/docker-client")
    child = langfuse.child_environment(environment)
    assert child["COMPOSE_DISABLE_ENV_FILE"] == "true"
    assert child["DOCKER_CONFIG"] == "/private/docker-client"
    assert "unsafe-ambient-value" not in child.values()
    assert all(child[key] == value for key, value in environment.items())


@pytest.mark.parametrize(
    "mutation", ["extra", "missing", "mismatched-project", "bad-secret", "non-string"]
)
def test_invalid_environments_fail_before_side_effects(mutation, environment, docker):
    project = environment["LANGFUSE_LOCAL_PROJECT"]
    if mutation == "extra":
        environment["COMPOSE_FILE"] = "production.yaml"
    elif mutation == "missing":
        del environment["REDIS_AUTH"]
    elif mutation == "mismatched-project":
        environment["LANGFUSE_LOCAL_PROJECT"] = "lf-local-" + "f" * 24
    elif mutation == "bad-secret":
        environment["POSTGRES_PASSWORD"] = "private@remote/database"
    else:
        environment["REDIS_AUTH"] = None
    with pytest.raises(langfuse.ProvisionError, match="invalid-environment"):
        langfuse.start(project, environment)
    assert docker == []


@pytest.mark.parametrize("collision_at", [0, 1, 2], ids=["container", "volume", "network"])
def test_existing_project_is_never_started_or_destroyed(collision_at, environment, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        output = b"existing-resource\n" if len(calls) - 1 == collision_at else b""
        return SimpleNamespace(returncode=0, stdout=output)

    monkeypatch.setattr(langfuse.subprocess, "run", run)
    with pytest.raises(langfuse.ProvisionError, match="project-already-exists"):
        langfuse.start(environment["LANGFUSE_LOCAL_PROJECT"], environment)
    assert len(calls) == collision_at + 1
    assert not langfuse._OWNED_PROJECTS
    assert all("down" not in call and "up" not in call for call in calls)


def test_start_cannot_reuse_owned_project(environment, docker):
    project = environment["LANGFUSE_LOCAL_PROJECT"]
    langfuse.start(project, environment)
    with pytest.raises(langfuse.ProvisionError, match="already-owned"):
        langfuse.start(project, environment)
    assert len(docker) == 4


@pytest.mark.parametrize("failure", ["exit", "timeout", "oserror", "interrupt"])
def test_partial_start_failure_cleans_only_its_owned_resources(failure, environment, monkeypatch):
    calls = []
    private = environment["LANGFUSE_INIT_PROJECT_SECRET_KEY"]

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if "up" in command:
            if failure == "timeout":
                raise subprocess.TimeoutExpired(command + [private], 10, output=private)
            if failure == "oserror":
                raise OSError(private)
            if failure == "interrupt":
                raise KeyboardInterrupt
            return SimpleNamespace(returncode=19, stdout=private.encode())
        return SimpleNamespace(returncode=0, stdout=b"")

    monkeypatch.setattr(langfuse.subprocess, "run", run)
    expected_error = KeyboardInterrupt if failure == "interrupt" else langfuse.ProvisionError
    with pytest.raises(expected_error) as caught:
        langfuse.start(environment["LANGFUSE_LOCAL_PROJECT"], environment)
    assert private not in str(caught.value)
    assert calls[-1][0][-3:] == ["down", "--timeout", "30"]
    assert "--volumes" not in calls[-1][0]
    assert environment["LANGFUSE_LOCAL_PROJECT"] in langfuse._OWNED_PROJECTS


def test_failed_cleanup_preserves_authority_for_explicit_cleanup(environment, monkeypatch):
    def run(command, **kwargs):
        return SimpleNamespace(
            returncode=17 if "up" in command or "down" in command else 0, stdout=b""
        )

    monkeypatch.setattr(langfuse.subprocess, "run", run)
    project = environment["LANGFUSE_LOCAL_PROJECT"]
    with pytest.raises(langfuse.ProvisionError, match="startup-and-cleanup-failed"):
        langfuse.start(project, environment)
    assert project in langfuse._OWNED_PROJECTS


def test_unowned_project_cannot_be_deleted(environment, docker):
    with pytest.raises(langfuse.ProvisionError, match="project-not-owned"):
        langfuse.stop(environment["LANGFUSE_LOCAL_PROJECT"], environment)
    assert docker == []


@pytest.mark.parametrize("remove_volumes", [True, False])
def test_stop_volume_policy_is_explicit(environment, docker, remove_volumes):
    project = environment["LANGFUSE_LOCAL_PROJECT"]
    langfuse._OWNED_PROJECTS.add(project)
    langfuse.stop(project, environment, remove_volumes=remove_volumes)
    command, options = docker[0]
    assert ("--volumes" in command) is remove_volumes
    assert "--remove-orphans" not in command
    assert options["stdout"] == options["stderr"] == subprocess.DEVNULL
    assert (project not in langfuse._OWNED_PROJECTS) is remove_volumes


def test_default_stop_preserves_data_and_allows_later_explicit_cleanup(environment, docker):
    project = environment["LANGFUSE_LOCAL_PROJECT"]
    langfuse._OWNED_PROJECTS.add(project)
    langfuse.stop(project, environment)
    assert "--volumes" not in docker[0][0]
    assert project in langfuse._OWNED_PROJECTS
    langfuse.stop(project, environment, remove_volumes=True)
    assert "--volumes" in docker[1][0]
    assert project not in langfuse._OWNED_PROJECTS


def test_private_state_roundtrip_without_overwrite(tmp_path, monkeypatch, environment):
    monkeypatch.chdir(tmp_path)
    directory = Path(".langfuse-regression")
    langfuse.save_environment(directory, environment)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    state = directory / "state.json"
    assert stat.S_IMODE(state.stat().st_mode) == 0o600
    assert langfuse.load_environment(directory) == environment
    before = state.read_bytes()
    with pytest.raises(langfuse.ProvisionError, match="private-state-create-failed"):
        langfuse.save_environment(directory, langfuse.generate_environment())
    assert state.read_bytes() == before
    assert not Path(".env").exists()


@pytest.mark.parametrize(
    "directory", [".", "..", ".env", "normal", "a/.langfuse-test", "/tmp/.langfuse-test"]
)
def test_private_state_cannot_target_root_env_or_external_paths(directory, environment):
    with pytest.raises(langfuse.ProvisionError, match="unsafe-private-state"):
        langfuse.save_environment(Path(directory), environment)
    with pytest.raises(langfuse.ProvisionError, match="unsafe-private-state"):
        langfuse.load_environment(Path(directory))


@pytest.mark.parametrize(
    "unsafe",
    [
        "directory-symlink",
        "file-symlink",
        "hardlink",
        "file-mode",
        "directory-mode",
        "foreign-owner",
        "fifo",
    ],
)
def test_unsafe_private_state_is_rejected_without_repair(
    unsafe, tmp_path, monkeypatch, environment
):
    monkeypatch.chdir(tmp_path)
    directory = Path(".langfuse-regression")
    langfuse.save_environment(directory, environment)
    state = directory / "state.json"
    if unsafe == "directory-symlink":
        directory.rename("private-target")
        directory.symlink_to("private-target", target_is_directory=True)
    elif unsafe == "file-symlink":
        state.rename(directory / "private-target.json")
        state.symlink_to("private-target.json")
    elif unsafe == "hardlink":
        os.link(state, directory / "linked.json")
    elif unsafe == "file-mode":
        state.chmod(0o644)
    elif unsafe == "directory-mode":
        directory.chmod(0o755)
    elif unsafe == "foreign-owner":
        actual_uid = os.getuid()
        monkeypatch.setattr(os, "getuid", lambda: actual_uid + 1)
    else:
        state.unlink()
        os.mkfifo(state, 0o600)
    before = (directory.lstat(), state.lstat())
    with pytest.raises(langfuse.ProvisionError):
        langfuse.load_environment(directory)
    after = (directory.lstat(), state.lstat())
    assert [(x.st_mode, x.st_ino, x.st_uid) for x in before] == [
        (x.st_mode, x.st_ino, x.st_uid) for x in after
    ]


def test_invalid_state_json_is_not_in_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    directory = Path(".langfuse-invalid")
    directory.mkdir(mode=0o700)
    state = directory / "state.json"
    state.write_text('{"secret": "never-display-this"')
    state.chmod(0o600)
    with pytest.raises(langfuse.ProvisionError) as caught:
        langfuse.load_environment(directory)
    assert "never-display-this" not in str(caught.value)


def test_cli_outputs_only_safe_readiness_not_credentials(
    tmp_path, monkeypatch, environment, capsys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(langfuse, "generate_environment", lambda: environment)
    monkeypatch.setattr(langfuse, "start", lambda *args: None)
    monkeypatch.setattr(sys, "argv", [str(_SCRIPT), "up", "--directory", ".langfuse-cli"])
    langfuse.main()
    captured = capsys.readouterr()
    assert langfuse.WEB_URL in captured.out
    for key in langfuse._SECRET_LENGTHS:
        assert environment[key] not in captured.out + captured.err
    assert environment["LANGFUSE_INIT_PROJECT_SECRET_KEY"] not in captured.out + captured.err
    assert json.loads(Path(".langfuse-cli/state.json").read_text()) == environment


def test_cli_down_authorizes_only_explicit_private_state(
    tmp_path, monkeypatch, environment, docker, capsys
):
    monkeypatch.chdir(tmp_path)
    langfuse.save_environment(Path(".langfuse-cli"), environment)
    monkeypatch.setattr(sys, "argv", [str(_SCRIPT), "down", "--directory", ".langfuse-cli"])
    langfuse.main()
    assert "down" in docker[0][0] and "--volumes" not in docker[0][0]
    assert "removed" in capsys.readouterr().out
    assert Path(".langfuse-cli/state.json").is_file()


def test_cli_destructive_cleanup_requires_explicit_flag(tmp_path, monkeypatch, environment, docker):
    monkeypatch.chdir(tmp_path)
    langfuse.save_environment(Path(".langfuse-cli"), environment)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(_SCRIPT), "down", "--directory", ".langfuse-cli", "--remove-volumes"],
    )
    langfuse.main()
    assert "--volumes" in docker[0][0]
    assert Path(".langfuse-cli/state.json").is_file()
