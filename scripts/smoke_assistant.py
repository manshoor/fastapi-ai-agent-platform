"""Exercise the resident assistant over real HTTP against the simulator's keyword policy.

    demo residence -> upload house rules as manager -> wait for ready
      -> "set the living room to 21 degrees"  -> thermostat at 21
      -> "unlock the front door"              -> pending, door still locked
      -> stranger confirm 404 -> owner confirm 200 -> door unlocked
      -> house-rules question                 -> cited answer
      -> account without a residence         -> 403

No paid service: run it against the isolated AI stack with the provider simulator.
Prints no bearer tokens, messages, document text or answers.
"""

import argparse
import importlib.util
import time
from pathlib import Path
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parent.parent
MESSAGES = "/api/v1/assistant/messages"
RESIDENCE = "/api/v1/residence"
DOCUMENTS = "/api/v1/rag/documents"
RULES = "# House rules\n\nQuiet hours start at 22:00 and end at 07:00 every day."


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_smoke_ai = _load("smoke_ai", "scripts/smoke_ai.py")
expect = _smoke_ai.expect


def _account(client: httpx.Client) -> dict:
    email = f"assistant-smoke-{uuid4().hex}@example.com"
    password = f"synthetic-{uuid4().hex}!"
    expect(client.post("/api/v1/auth/register", json={"email": email, "password": password}), 201)
    login = expect(
        client.post("/api/v1/auth/login", data={"username": email, "password": password}), 200
    ).json()
    return {"Authorization": f"Bearer {login['access_token']}"}


def _say(client: httpx.Client, owner: dict, message: str) -> dict:
    return expect(client.post(MESSAGES, headers=owner, json={"message": message}), 200).json()


def _device(client: httpx.Client, owner: dict, name: str) -> dict:
    devices = expect(client.get(RESIDENCE, headers=owner), 200).json()["devices"]
    return next(device for device in devices if device["name"] == name)


def run_smoke(base_url: str, *, timeout_seconds: float = 60) -> None:
    _smoke_ai.validate_destination(base_url)
    with httpx.Client(base_url=base_url, timeout=60, trust_env=False) as client:
        expect(client.get("/health/ready"), 200)
        owner, stranger = _account(client), _account(client)
        view = expect(client.post(f"{RESIDENCE}/demo", headers=owner), 201).json()
        expect(client.post(f"{RESIDENCE}/demo", headers=stranger), 201)

        # Building document, uploaded by the demo manager, then ingested by the worker.
        upload = client.post(
            DOCUMENTS,
            params={"building_id": view["building"]["id"]},
            headers={**owner, "Idempotency-Key": f"smoke-{uuid4().hex}"},
            data={"title": "House rules"},
            files={"file": ("rules.md", RULES.encode())},
        )
        document_id = expect(upload, 202).json()["id"]
        deadline = time.monotonic() + timeout_seconds
        while True:
            status = expect(client.get(f"{DOCUMENTS}/{document_id}", headers=owner), 200).json()
            if status["status"] in {"ready", "failed"}:
                assert status["status"] == "ready", "Ingestion failed"
                break
            if time.monotonic() >= deadline:
                raise AssertionError("Ingestion deadline exceeded")
            time.sleep(0.2)

        # A device action changes the simulated thermostat.
        turn = _say(client, owner, "Set the living room thermostat to 23 degrees")
        assert turn["status"] == "answered", "Thermostat turn not answered"
        assert _device(client, owner, "Living room thermostat")["target_celsius"] == 23.0

        # An unlock only creates a pending action until the owner confirms it.
        turn = _say(client, owner, "Unlock the front door")
        [pending] = turn["pending_actions"]
        assert _device(client, owner, "Front door")["locked"] is True
        confirm = f"/api/v1/assistant/actions/{pending['id']}/confirm"
        expect(client.post(confirm, headers=stranger), 404)
        expect(client.post(confirm, headers=owner), 200)
        assert _device(client, owner, "Front door")["locked"] is False

        # A building question is answered with a citation.
        turn = _say(client, owner, "When do quiet hours start?")
        assert turn["status"] == "answered", "Document question not answered"
        assert [item["document_id"] for item in turn["citations"]] == [document_id]

        # No residence, no assistant.
        expect(client.post(MESSAGES, headers=_account(client), json={"message": "hi"}), 403)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 180:
        parser.error("timeout must be 1..180 seconds")
    try:
        run_smoke(args.base_url, timeout_seconds=args.timeout)
    except ValueError:
        parser.error("unsafe smoke destination; select the isolated AI API")
    except httpx.HTTPError:
        raise SystemExit("Assistant smoke HTTP transport failed (details withheld)") from None
    print("PASS: demo, device action, pending unlock, confirm, cited answer, 403")


if __name__ == "__main__":
    main()
