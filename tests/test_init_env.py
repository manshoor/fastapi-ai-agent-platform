"""Exercise local initialization in temporary directories without service fixtures."""

import importlib.util
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "init_env.py"
_spec = importlib.util.spec_from_file_location("init_env", _SCRIPT)
assert _spec is not None and _spec.loader is not None
init_env = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(init_env)


@pytest.fixture
def local_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(_SCRIPT)])
    return tmp_path


def _read_values(path: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in path.read_text().splitlines()
        if line and not line.startswith("#")
    )


def _assert_private_credentials(path: Path, output: str) -> dict[str, str]:
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    values = _read_values(path)
    for key, length in (("POSTGRES_PASSWORD", 48), ("JWT_SECRET", 64)):
        assert re.fullmatch(rf"[0-9a-f]{{{length}}}", values[key]) is not None
        assert values[key] not in output
    return values


def _forbid_metadata_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("initialization must not chmod/chown an existing artifact directory")

    monkeypatch.setattr(os, "chmod", forbidden)
    monkeypatch.setattr(os, "chown", forbidden)


def test_new_directory_and_credentials_are_private_in_script_subprocess(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0
    artifacts = tmp_path / "artifacts"
    assert not artifacts.is_symlink()
    assert artifacts.is_dir()
    assert stat.S_IMODE(artifacts.stat().st_mode) == 0o700
    assert artifacts.stat().st_uid == os.getuid()
    values = _assert_private_credentials(tmp_path / ".env", result.stdout + result.stderr)
    assert values["API_PORT"] == "8000"
    assert values["AUTH_TOKEN_TTL_SECONDS"] == "900"
    assert values["HOST_UID"] == str(os.getuid())
    assert values["HOST_GID"] == str(os.getgid())


def test_existing_private_directory_is_accepted_without_metadata_changes(
    local_cli: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    artifacts = local_cli / "artifacts"
    artifacts.mkdir(mode=0o700)
    artifacts.chmod(0o700)
    contents = artifacts / "keep.txt"
    contents.write_text("existing artifact\n")
    before = artifacts.stat()
    monkeypatch.setattr(
        sys,
        "argv",
        [str(_SCRIPT), "--output", "custom.env", "--api-port", "8123", "--token-ttl", "7200"],
    )
    _forbid_metadata_changes(monkeypatch)

    init_env.main()

    captured = capsys.readouterr()
    values = _assert_private_credentials(local_cli / "custom.env", captured.out + captured.err)
    assert values["API_PORT"] == "8123"
    assert values["AUTH_TOKEN_TTL_SECONDS"] == "7200"
    assert not (local_cli / ".env").exists()
    assert contents.read_text() == "existing artifact\n"
    after = artifacts.stat()
    assert (after.st_mode, after.st_uid, after.st_gid, after.st_ino) == (
        before.st_mode,
        before.st_uid,
        before.st_gid,
        before.st_ino,
    )


def test_existing_env_is_preserved_with_private_directory(
    local_cli: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    artifacts = local_cli / "artifacts"
    artifacts.mkdir(mode=0o700)
    artifacts.chmod(0o700)
    output = local_cli / ".env"
    original = (
        b"POSTGRES_PASSWORD=existing-private-password\nJWT_SECRET=existing-private-signing-key\n"
    )
    output.write_bytes(original)
    output.chmod(0o600)
    before = output.stat()
    _forbid_metadata_changes(monkeypatch)

    init_env.main()

    captured = capsys.readouterr()
    assert "unchanged" in captured.out.lower()
    assert output.read_bytes() == original
    after = output.stat()
    assert (after.st_mode, after.st_ino, after.st_mtime_ns) == (
        before.st_mode,
        before.st_ino,
        before.st_mtime_ns,
    )
    assert "existing-private-password" not in captured.out + captured.err
    assert "existing-private-signing-key" not in captured.out + captured.err


@pytest.mark.parametrize("existing_env", [False, True], ids=["new-env", "existing-env"])
@pytest.mark.parametrize("unsafe", ["0755", "0777", "symlink", "file", "foreign-owner"])
def test_unsafe_artifacts_rejected_before_credentials_without_mutation(
    unsafe: str,
    existing_env: bool,
    local_cli: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    artifacts = local_cli / "artifacts"
    if unsafe == "file":
        artifacts.write_bytes(b"not a directory\n")
        artifacts.chmod(0o600)
        preserved = artifacts
    elif unsafe == "symlink":
        target = local_cli / "private-target"
        target.mkdir(mode=0o700)
        target.chmod(0o700)
        artifacts.symlink_to(target, target_is_directory=True)
        preserved = target / "keep.txt"
        preserved.write_bytes(b"private target contents\n")
    else:
        artifacts.mkdir(mode=0o700)
        artifacts.chmod(int(unsafe, 8) if unsafe in {"0755", "0777"} else 0o700)
        preserved = artifacts / "keep.txt"
        preserved.write_bytes(b"existing artifact contents\n")

    before = artifacts.lstat()
    preserved_before = preserved.stat()
    original_contents = preserved.read_bytes()
    target_before = artifacts.stat() if unsafe == "symlink" else None
    output = local_cli / ".env"
    original_env = b"POSTGRES_PASSWORD=do-not-replace\nJWT_SECRET=do-not-disclose\n"
    if existing_env:
        output.write_bytes(original_env)
        output.chmod(0o600)
    env_before = output.stat() if existing_env else None

    if unsafe == "foreign-owner":
        # Change only the claimed caller identity, not access checks or root behavior.
        # The actual directory remains private, writable, and owned by the real caller.
        actual_uid = os.getuid()
        assert before.st_uid == actual_uid
        assert stat.S_IMODE(before.st_mode) == 0o700
        assert os.access(artifacts, os.W_OK | os.X_OK)
        monkeypatch.setattr(os, "getuid", lambda: actual_uid + 1)
        assert artifacts.lstat().st_uid != os.getuid()

    _forbid_metadata_changes(monkeypatch)
    with pytest.raises(SystemExit) as raised:
        init_env.main()

    assert raised.value.code == 2
    captured = capsys.readouterr()
    diagnostic = captured.err.lower()
    assert "artifacts" in diagnostic
    assert re.search(r"remove|replace|recreate|chmod|chown|correct|fix|choose|create", diagnostic)
    assert "do-not-replace" not in captured.out + captured.err
    assert "do-not-disclose" not in captured.out + captured.err
    after = artifacts.lstat()
    assert (after.st_mode, after.st_uid, after.st_gid, after.st_ino) == (
        before.st_mode,
        before.st_uid,
        before.st_gid,
        before.st_ino,
    )
    assert preserved.read_bytes() == original_contents
    assert preserved.stat().st_mode == preserved_before.st_mode
    if target_before is not None:
        assert artifacts.is_symlink()
        assert artifacts.readlink() == local_cli / "private-target"
        target_after = artifacts.stat()
        assert (target_after.st_mode, target_after.st_uid, target_after.st_ino) == (
            target_before.st_mode,
            target_before.st_uid,
            target_before.st_ino,
        )
    if env_before is not None:
        assert output.read_bytes() == original_env
        after_env = output.stat()
        assert (after_env.st_mode, after_env.st_ino, after_env.st_mtime_ns) == (
            env_before.st_mode,
            env_before.st_ino,
            env_before.st_mtime_ns,
        )
    else:
        assert not output.exists()
