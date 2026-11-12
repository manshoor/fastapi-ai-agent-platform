"""Create a separate private AI Compose environment; never read normal .env credentials."""

import argparse
import os
import secrets
import stat
from pathlib import Path


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


def initialize(directory: Path, api_port: int) -> bool:
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
            env_fd = os.open(".env", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
        except FileNotFoundError:
            env_fd = None
        if env_fd is not None:
            try:
                require_private(os.fstat(env_fd), directory=False)
            finally:
                os.close(env_fd)
            return False
        values = {
            "AI_DB_PASSWORD": secrets.token_hex(32),
            "AI_JWT_SECRET": secrets.token_hex(48),
            "AI_API_PORT": str(api_port),
            "AI_AUTH_TOKEN_TTL_SECONDS": "900",
        }
        env_fd = os.open(
            ".env", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=root_fd
        )
        with os.fdopen(env_fd, "w") as handle:
            handle.write("# Private isolated AI local credentials; do not commit.\n")
            handle.writelines(f"{key}={value}\n" for key, value in values.items())
        return True
    finally:
        os.close(root_fd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(".ai-local"))
    parser.add_argument("--api-port", type=int, default=8002)
    args = parser.parse_args()
    try:
        created = initialize(args.directory, args.api_port)
    except (OSError, ValueError):
        parser.error(
            "Unsafe or inaccessible AI paths/port. Use a dedicated .ai-* child with current-owner "
            "0700 directories and a nonsymlink, single-link 0600 .env; paths are never repaired."
        )
    print(
        "Created private AI environment (0600), directories (0700)."
        if created
        else "Preserved existing private AI environment; credentials unchanged."
    )


if __name__ == "__main__":
    main()
