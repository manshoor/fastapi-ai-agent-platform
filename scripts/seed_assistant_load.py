"""Append resident assistant load fixtures; destinations are checked as in seed_load.py.

Each building holds up to 50 apartments with one resident each, the demo device set and
bookings, and a manager who owns one ready building document. Chunks are embedded the way
the provider simulator embeds questions. Writes `user_id,apartment_id,access_token` rows
for the residents to a new mode-0600 CSV.
"""

import argparse
import asyncio
import csv
import importlib.util
import math
import os
import secrets
import sys
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from uuid import UUID, uuid4

import jwt
from sqlalchemy import func, insert, select

from app.ai.config import ProviderSettings
from app.assistant.contracts import Role
from app.assistant.residence import DEMO_BOOKINGS, DEMO_DEVICES
from app.config import Settings
from app.db import create_engine
from app.models import (
    RAG_EMBEDDING_DIMENSIONS,
    Apartment,
    Booking,
    Building,
    Device,
    Membership,
    RAGChunk,
    RAGDocument,
    User,
)
from app.security import PasswordService, create_access_token

ROOT = Path(__file__).resolve().parents[1]
APARTMENTS_PER_BUILDING = 50
# A multi-row insert takes its columns from the first row: give every row every key.
DEVICE_STATE = {"target_celsius": None, "locked": None, "armed": None}
# Single-line, single-spaced prose: the simulator quotes the first 8 words of a passage.
HOUSE_RULES = (
    "Quiet hours run from 22:00 to 07:00 every day, so keep music and voices low.",
    "Dogs and cats are welcome, and dogs stay on a lead in all shared areas.",
    "Recycling bins stand in the courtyard: glass goes in green bins and paper in blue.",
    "The gym is open from 06:00 to 23:00, and residents wipe down equipment after use.",
    "Residents reserve the party room at the front desk up to 30 days ahead.",
    "Visitor parking is in bays V1 to V6 on level minus one for up to 4 hours.",
    "Bulky waste is collected on the first Monday of each month at the service entrance.",
    "Smoking is not allowed inside the building, on balconies or on the roof terrace.",
    "Laundry machines on the ground floor run from 07:00 to 21:00 every day.",
    "The front desk holds parcels for 14 days, and residents bring photo ID to collect them.",
    "The roof terrace is open from May to September between 10:00 and 22:00.",
    "Moving furniture in or out uses the service lift between 09:00 and 17:00 on weekdays.",
)
DOCUMENT_TITLE = "House rules"
DOCUMENT_CHUNKER = "recursive-v1"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SEED_LOAD = _load("seed_load_shared", ROOT / "scripts" / "seed_load.py")
SIMULATOR = _load("provider_simulator_seed", ROOT / "tests" / "support" / "provider_simulator.py")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residents", type=SEED_LOAD.positive, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-remote", action="store_true")
    return parser.parse_args()


def _document_rows(manager_id: UUID, building_id: UUID, model: str) -> tuple[dict, list[dict]]:
    """One ready building document and its chunks, as the ingestion worker stores them."""
    document_id = uuid4()
    joined = "\n".join(HOUSE_RULES).encode()
    document = {
        "id": document_id,
        "owner_id": manager_id,
        "building_id": building_id,
        "title": DOCUMENT_TITLE,
        "content_type": "text/plain",
        "size_bytes": len(joined),
        "content_sha256": sha256(joined).hexdigest(),
        "source": None,
        "idempotency_key_hash": uuid4().hex,
        "request_fingerprint": uuid4().hex,
        "status": "ready",
        "active_version": 1,
        "deadline_at": func.now() + timedelta(hours=1),
        "embedding_model": model,
        "embedding_dimensions": RAG_EMBEDDING_DIMENSIONS,
        "chunker_version": DOCUMENT_CHUNKER,
        "chunk_count": len(HOUSE_RULES),
        "page_count": 1,
        "finished_at": func.now(),
    }
    chunks = [
        {
            "document_id": document_id,
            "version": 1,
            "owner_id": manager_id,
            "ordinal": ordinal,
            "page": None,
            "content": text,
            "embedding": SIMULATOR.embed_text(text, RAG_EMBEDDING_DIMENSIONS),
        }
        for ordinal, text in enumerate(HOUSE_RULES)
    ]
    return document, chunks


async def _seed_building(connection, *, run_id, number, residents, password_hash, model, now):
    """One building in one transaction. Returns (user_id, apartment_id) per resident."""
    building_id, manager_id = uuid4(), uuid4()
    users = [
        {
            "id": manager_id,
            "email": f"assistant-load-{run_id}-manager-{number}@example.com",
            "password_hash": password_hash,
        }
    ]
    apartments, memberships, devices, bookings, active = [], [], [], [], []
    memberships.append(
        {
            "user_id": manager_id,
            "building_id": building_id,
            "apartment_id": None,
            "role": Role.MANAGER,
        }
    )
    for index in residents:
        user_id, apartment_id = uuid4(), uuid4()
        place = {"building_id": building_id, "apartment_id": apartment_id}
        users.append(
            {
                "id": user_id,
                "email": f"assistant-load-{run_id}-{index}@example.com",
                "password_hash": password_hash,
            }
        )
        apartments.append(
            {"id": apartment_id, "building_id": building_id, "label": f"A{index + 1}"}
        )
        memberships.append({"user_id": user_id, "role": Role.RESIDENT, **place})
        devices.extend({**DEVICE_STATE, **device, **place} for device in DEMO_DEVICES)
        bookings.extend(
            {
                "user_id": user_id,
                "amenity": amenity,
                "starts_at": now + offset,
                "ends_at": now + offset + length,
                **place,
            }
            for amenity, offset, length in DEMO_BOOKINGS
        )
        active.append((user_id, apartment_id))
    document, chunks = _document_rows(manager_id, building_id, model)
    await connection.execute(insert(Building).values(id=building_id, name=f"Load {number}"))
    await connection.execute(insert(User), users)
    await connection.execute(insert(Apartment), apartments)
    await connection.execute(insert(Membership), memberships)
    await connection.execute(insert(Device), devices)
    await connection.execute(insert(Booking), bookings)
    await connection.execute(insert(RAGDocument).values(**document))
    await connection.execute(insert(RAGChunk), chunks)
    return active


async def seed(args: argparse.Namespace) -> None:
    settings = Settings()
    SEED_LOAD.validate_destination(settings.sqlalchemy_url, allow_remote=args.allow_remote)
    model = ProviderSettings().ai_embedding_model

    run_id = uuid4().hex
    buildings = math.ceil(args.residents / APARTMENTS_PER_BUILDING)
    committed_buildings = 0
    engine = None
    try:
        with SEED_LOAD.open_output(args.output, args.overwrite) as output:
            password_hash = await PasswordService(settings).hash(secrets.token_urlsafe(32))
            engine = create_engine(settings)
            active: list[tuple[UUID, UUID]] = []
            for number in range(buildings):
                first = number * APARTMENTS_PER_BUILDING
                residents = range(first, min(first + APARTMENTS_PER_BUILDING, args.residents))
                async with engine.begin() as connection:
                    now = await connection.scalar(select(func.now()))
                    active += await _seed_building(
                        connection,
                        run_id=run_id,
                        number=number + 1,
                        residents=residents,
                        password_hash=password_hash,
                        model=model,
                        now=now,
                    )
                committed_buildings += 1

            # Tokens are minted only after every building committed.
            writer = csv.writer(output)
            writer.writerow(("user_id", "apartment_id", "access_token"))
            earliest_expiry = None
            for user_id, apartment_id in active:
                token = create_access_token(user_id, settings)
                expiry = jwt.decode(token, options={"verify_signature": False})["exp"]
                if earliest_expiry is None or expiry < earliest_expiry:
                    earliest_expiry = expiry
                writer.writerow((user_id, apartment_id, token))
            output.flush()
            os.fsync(output.fileno())
        expiry_text = datetime.fromtimestamp(earliest_expiry, UTC).isoformat()
        print(
            f"run={run_id} committed_buildings={committed_buildings} residents={len(active)} "
            f"chunks_per_building={len(HOUSE_RULES)} earliest_token_expiry={expiry_text} "
            f"output={args.output}"
        )
    except BaseException as error:
        print(
            f"Seed failed ({type(error).__name__}); run={run_id} "
            f"confirmed_committed_buildings={committed_buildings}. Existing data was not "
            f"removed; output={args.output} is not a complete fixture.",
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
            message = f"Cannot seed assistant load fixtures: {type(error).__name__}"
        print(message, file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
