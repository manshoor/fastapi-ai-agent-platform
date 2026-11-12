"""Generate private local Compose configuration without overwriting credentials."""

import argparse
import os
import secrets
import stat
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".env"))
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--token-ttl", type=int, default=900)
    args = parser.parse_args()
    if not 1 <= args.api_port <= 65535 or not 1 <= args.token_ttl <= 86400:
        parser.error("port must be 1–65535 and token lifetime 1–86400 seconds")
    artifacts = Path("artifacts")
    try:
        artifacts.mkdir(mode=0o700)
    except FileExistsError:
        pass
    except OSError as exc:
        parser.error(f"Cannot create artifact directory {artifacts.absolute()}: {exc}")
    metadata = artifacts.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        parser.error(
            f"Unsafe artifact directory {artifacts.absolute()}: must be a nonsymlink directory "
            "owned by the current UID with exact mode 0700. Manually correct its ownership "
            "and permissions, or safely replace it with a private directory; "
            "existing files and credentials are left unchanged."
        )
    if not os.access(artifacts, os.W_OK | os.X_OK):
        parser.error(
            f"Artifact directory is not writable: {artifacts.absolute()}. "
            "Manually correct its access permissions before retrying."
        )
    values = {
        "POSTGRES_PASSWORD": secrets.token_hex(24),
        "JWT_SECRET": secrets.token_hex(32),
        "API_PORT": str(args.api_port),
        "AUTH_TOKEN_TTL_SECONDS": str(args.token_ttl),
        "HOST_UID": str(os.getuid()),
        "HOST_GID": str(os.getgid()),
    }
    try:
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        print(f"Leaving existing {args.output} unchanged")
        return
    with os.fdopen(fd, "w") as handle:
        handle.write("# Generated local-only credentials; do not commit.\n")
        handle.writelines(f"{key}={value}\n" for key, value in values.items())
    print(f"Created {args.output} (0600) and prepared {artifacts}/")


if __name__ == "__main__":
    main()
