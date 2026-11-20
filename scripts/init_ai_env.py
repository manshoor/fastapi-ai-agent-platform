"""Create a separate private AI Compose environment; never read normal .env credentials.

An existing private environment keeps every value. Only a missing AI_ASSISTANT_TOTP_KEY
is appended, so environments made before unlock step-up keep working.
"""

import argparse
import base64
import os
import secrets
import stat
from enum import StrEnum
from pathlib import Path

TOTP_KEY_NAME = "AI_ASSISTANT_TOTP_KEY"
# AES-256 key that seals authenticator secrets (ASSISTANT_TOTP_KEY).
TOTP_KEY_BYTES = 32


class EnvResult(StrEnum):
    CREATED = "created"
    KEY_ADDED = "key_added"
    UNCHANGED = "unchanged"


class EnvKeyError(ValueError):
    """The private .env holds a TOTP key this script must not keep or fix by itself."""


def require_private(metadata: os.stat_result, *, directory: bool) -> None:
    expected = 0o700 if directory else 0o600
    kind_ok = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    if (
        not kind_ok
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != expected
        or (not directory and metadata.st_nlink != 1)
    ):
        raise ValueError(
            "unsafe private path: require current-owner nonsymlink 0700 directory/0600 file"
        )


def _totp_key() -> str:
    return base64.b64encode(secrets.token_bytes(TOTP_KEY_BYTES)).decode("ascii")


def _key_values(content: str) -> list[str]:
    """Every value the .env assigns to the TOTP key, read as Compose reads a line."""
    values = []
    for line in content.splitlines():
        name, separator, value = line.strip().removeprefix("export ").partition("=")
        if separator and name.strip() == TOTP_KEY_NAME:
            # Base64 holds no space or '#': " #" can only start an inline comment.
            values.append(value.split(" #", 1)[0].strip().strip("\"'"))
    return values


def _is_totp_key(value: str) -> bool:
    try:
        return len(base64.b64decode(value, validate=True)) == TOTP_KEY_BYTES
    except ValueError:
        return False


def _add_missing_key(env_fd: int) -> EnvResult:
    """Append the TOTP key to an existing private .env that lacks it; change nothing else."""
    with os.fdopen(env_fd, "r+", encoding="utf-8") as handle:
        # Checked before any write: a public, linked or foreign file is never touched.
        require_private(os.fstat(handle.fileno()), directory=False)
        try:
            content = handle.read()
        except UnicodeDecodeError:
            raise EnvKeyError("The private AI .env is not UTF-8 text; nothing changed.") from None

        # A present key is never replaced: secrets sealed with it would stop opening.
        # An unusable one is reported for the user to remove.
        values = _key_values(content)
        if len(values) > 1:
            raise EnvKeyError(f"{TOTP_KEY_NAME} is set more than once; keep one line, then rerun.")
        if values and not _is_totp_key(values[0]):
            raise EnvKeyError(
                f"{TOTP_KEY_NAME} is not 32 bytes encoded as base64; remove that line, then rerun."
            )
        if values:
            return EnvResult.UNCHANGED
        separator = "" if not content or content.endswith("\n") else "\n"
        handle.write(f"{separator}{TOTP_KEY_NAME}={_totp_key()}\n")
    return EnvResult.KEY_ADDED


def initialize(directory: Path, api_port: int) -> EnvResult:
    # A single dedicated child avoids traversing attacker-controlled parent symlinks and
    # prevents accidental use of normal .env, artifacts, or benchmark directories.
    if len(directory.parts) != 1 or not directory.name.startswith(".ai-"):
        raise ValueError("directory must be a single project child name beginning .ai-")
    if not 1024 <= api_port <= 65535 or api_port in {8000, 8001}:
        raise ValueError(
            "AI API port must be 1024..65535, excluding normal 8000 and benchmark 8001"
        )
    try:
        directory.mkdir(mode=0o700)
    except FileExistsError:
        pass
    require_private(directory.lstat(), directory=True)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    root_fd = os.open(directory, flags)
    try:
        require_private(os.fstat(root_fd), directory=True)
        try:
            os.mkdir("artifacts", mode=0o700, dir_fd=root_fd)
        except FileExistsError:
            pass
        artifact_fd = os.open("artifacts", flags, dir_fd=root_fd)
        try:
            require_private(os.fstat(artifact_fd), directory=True)
        finally:
            os.close(artifact_fd)
        try:
            # O_APPEND: the only write this can make is a line at the end.
            env_fd = os.open(".env", os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW, dir_fd=root_fd)
        except FileNotFoundError:
            env_fd = None
        if env_fd is not None:
            return _add_missing_key(env_fd)
        values = {
            "AI_DB_PASSWORD": secrets.token_hex(32),
            "AI_JWT_SECRET": secrets.token_hex(48),
            "AI_API_PORT": str(api_port),
            "AI_AUTH_TOKEN_TTL_SECONDS": "900",
            TOTP_KEY_NAME: _totp_key(),
        }
        env_fd = os.open(
            ".env", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=root_fd
        )
        with os.fdopen(env_fd, "w") as handle:
            handle.write("# Private isolated AI local credentials; do not commit.\n")
            handle.writelines(f"{key}={value}\n" for key, value in values.items())
        return EnvResult.CREATED
    finally:
        os.close(root_fd)


MESSAGES = {
    EnvResult.CREATED: "Created private AI environment (0600), directories (0700).",
    EnvResult.KEY_ADDED: (
        f"Added {TOTP_KEY_NAME} to the existing private AI environment; other values unchanged."
    ),
    EnvResult.UNCHANGED: "Preserved existing private AI environment; credentials unchanged.",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(".ai-local"))
    parser.add_argument("--api-port", type=int, default=8002)
    args = parser.parse_args()
    try:
        result = initialize(args.directory, args.api_port)
    except EnvKeyError as error:
        parser.error(str(error))
    except (OSError, ValueError):
        parser.error(
            "Unsafe or inaccessible AI paths/port. Use a dedicated .ai-* child with current-owner "
            "0700 directories and a nonsymlink, single-link 0600 .env; paths are never repaired."
        )
    print(MESSAGES[result])


if __name__ == "__main__":
    main()
