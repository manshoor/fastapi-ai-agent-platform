"""Append real load fixtures; effective driver destinations require local/app or explicit opt-in."""

import argparse
import asyncio
import csv
import os
import secrets
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import jwt
from sqlalchemy import insert
from sqlalchemy.engine import URL

from app.config import Settings
from app.db import create_engine
from app.models import Item, User
from app.security import PasswordService, create_access_token

BATCH_SIZE = 1000


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=positive, required=True)
    parser.add_argument("--active-users", type=positive, required=True)
    parser.add_argument("--items-per-user", type=positive, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-remote", action="store_true")
    args = parser.parse_args()
    if args.active_users > args.users:
        parser.error("--active-users must not exceed --users")
    return args


def open_output(path: Path, overwrite: bool):
    # Open before any inserts. O_NOFOLLOW rejects symlinks even with --overwrite;
    # O_NONBLOCK also prevents an existing FIFO from hanging this CLI.
    flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK
    flags |= os.O_TRUNC if overwrite else os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("output must be a regular file")
        os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, "w", newline="", encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        raise


def validate_destination(url: URL, *, allow_remote: bool) -> None:
    """Check driver kwargs, not URL authority fields that query parameters can override."""
    if allow_remote:
        return
    # This conversion does not construct an engine or open a connection. In
    # particular, asyncpg converts repeated/comma-separated query hosts to a list.
    _, connect_kwargs = url.get_dialect()().create_connect_args(url)
    host = connect_kwargs.get("host")
    hosts = host if isinstance(host, (list, tuple)) else [host]
    database = connect_kwargs.get("database")
    if (
        not hosts
        or any(
            not isinstance(value, str) or value not in {"db", "localhost", "127.0.0.1", "::1"}
            for value in hosts
        )
        or not isinstance(database, str)
        or database not in {"app", "app_test"}
    ):
        print(
            "Refusing nonlocal/non-app database; use --allow-remote explicitly.",
            file=sys.stderr,
        )
        raise ValueError("refusing nonlocal/non-app database; use --allow-remote explicitly")


async def seed(args: argparse.Namespace) -> None:
    settings = Settings()
    validate_destination(settings.sqlalchemy_url, allow_remote=args.allow_remote)

    run_id = uuid4().hex
    committed_users = committed_items = 0
    engine = None
    try:
        with open_output(args.output, args.overwrite) as output:
            password_hash = await PasswordService(settings).hash(secrets.token_urlsafe(32))
            engine = create_engine(settings)
            active: list[tuple[UUID, UUID]] = []
            for offset in range(0, args.users, BATCH_SIZE):
                users = [
                    {
                        "id": uuid4(),
                        "email": f"load-{run_id}-{index}@example.com",
                        "password_hash": password_hash,
                    }
                    for index in range(offset, min(offset + BATCH_SIZE, args.users))
                ]
                async with engine.begin() as connection:
                    await connection.execute(insert(User), users)
                committed_users += len(users)

                items = []
                for user in users:
                    for item_index in range(args.items_per_user):
                        item_id = uuid4()
                        if item_index == 0 and len(active) < args.active_users:
                            active.append((user["id"], item_id))
                        items.append(
                            {
                                "id": item_id,
                                "owner_id": user["id"],
                                "title": f"Load item {item_index}",
                                "description": "Seeded load fixture",
                            }
                        )
                        if len(items) == BATCH_SIZE:
                            async with engine.begin() as connection:
                                await connection.execute(insert(Item), items)
                            committed_items += len(items)
                            items.clear()
                if items:
                    async with engine.begin() as connection:
                        await connection.execute(insert(Item), items)
                    committed_items += len(items)

            # Tokens are minted only after every database batch committed, keeping
            # even a long million-row seed out of the fixture's token lifetime.
            writer = csv.writer(output)
            writer.writerow(("user_id", "item_id", "access_token"))
            earliest_expiry = None
            for user_id, item_id in active:
                token = create_access_token(user_id, settings)
                expiry = jwt.decode(token, options={"verify_signature": False})["exp"]
                earliest_expiry = (
                    expiry if earliest_expiry is None else min(earliest_expiry, expiry)
                )
                writer.writerow((user_id, item_id, token))
            output.flush()
            os.fsync(output.fileno())
        expiry_text = datetime.fromtimestamp(earliest_expiry, UTC).isoformat()
        print(
            f"run={run_id} committed_users={committed_users} committed_items={committed_items} "
            f"active_users={len(active)} earliest_token_expiry={expiry_text} output={args.output}"
        )
    except BaseException as error:
        print(
            f"Seed failed ({type(error).__name__}); run={run_id} "
            f"confirmed_committed_users={committed_users} "
            f"confirmed_committed_items={committed_items}. "
            "Existing data was not removed. A failed commit acknowledgement may hide additional "
            f"committed rows; output={args.output} is not a complete fixture.",
            file=sys.stderr,
        )
        raise
    finally:
        if engine is not None:
            await engine.dispose()


def main() -> None:
    try:
        asyncio.run(seed(arguments()))
    except (Exception, KeyboardInterrupt) as error:
        # Never print connection URLs, signing material, fixture passwords or SQL values.
        if isinstance(error, FileExistsError):
            message = (
                f"Output already exists and was left untouched: {error.filename}; "
                "use --overwrite to replace only that file"
            )
        elif isinstance(error, PermissionError):
            message = f"Permission denied: {error.filename}"
        else:
            message = f"Cannot seed load fixtures: {type(error).__name__}"
        print(message, file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
