"""Exercise isolated RAG upload, ingestion and query over real HTTP, with no paid service.

    register two accounts -> upload Markdown + PDF -> wait for ready
      -> owner query cites the right document -> stranger sees nothing
      -> delete -> the same query has no context

Prints no bearer tokens, document text or answers.
"""

import argparse
import importlib.util
import time
from pathlib import Path
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS = "/api/v1/rag/documents"
QUERY = "/api/v1/rag/query"
SETTLED = frozenset({"ready", "failed"})
PUMP = "# Pump manual\n\nThe synthetic pump runs at 40 litres per minute when the valve is open."
GARDEN = "Water the synthetic tomato plants every morning during the summer."


def _load(name: str, relative: str):
    # Reuse the AI smoke checks and the test PDF builder without packaging scripts/ or tests/.
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_smoke_ai = _load("smoke_ai", "scripts/smoke_ai.py")
_pdfs = _load("smoke_pdfs", "tests/support/pdfs.py")
expect = _smoke_ai.expect


def _account(client: httpx.Client) -> dict:
    email = f"rag-smoke-{uuid4().hex}@example.com"
    password = f"synthetic-{uuid4().hex}!"
    expect(client.post("/api/v1/auth/register", json={"email": email, "password": password}), 201)
    login = expect(
        client.post("/api/v1/auth/login", data={"username": email, "password": password}), 200
    ).json()
    return {"Authorization": f"Bearer {login['access_token']}"}


def _upload(client: httpx.Client, owner: dict, title: str, name: str, data: bytes) -> str:
    headers = {**owner, "Idempotency-Key": f"smoke-{uuid4().hex}"}
    response = client.post(
        DOCUMENTS, headers=headers, data={"title": title}, files={"file": (name, data)}
    )
    return expect(response, 202).json()["id"]


def _wait_ready(client: httpx.Client, owner: dict, document_id: str, deadline: float) -> dict:
    while True:
        document = expect(client.get(f"{DOCUMENTS}/{document_id}", headers=owner), 200).json()
        if document["status"] in SETTLED:
            assert document["status"] == "ready", f"Ingestion failed: {document['error_code']}"
            return document
        if time.monotonic() >= deadline:
            raise AssertionError("Ingestion deadline exceeded")
        time.sleep(0.2)


def _ask(client: httpx.Client, owner: dict, question: str, **body) -> dict:
    response = client.post(QUERY, headers=owner, json={"question": question, **body})
    return expect(response, 200).json()


def run_smoke(base_url: str, *, timeout_seconds: float = 60) -> None:
    _smoke_ai.validate_destination(base_url)
    with httpx.Client(base_url=base_url, timeout=60, trust_env=False) as client:
        expect(client.get("/health/ready"), 200)
        owner, stranger = _account(client), _account(client)

        # Upload one Markdown and one PDF document, then wait for the worker.
        pump = _upload(client, owner, "Pump manual", "pump.md", PUMP.encode())
        garden = _upload(client, owner, "Garden notes", "garden.pdf", _pdfs.text_pdf([GARDEN]))
        deadline = time.monotonic() + timeout_seconds
        for document_id in (pump, garden):
            document = _wait_ready(client, owner, document_id, deadline)
            assert document["chunk_count"] >= 1, "Ready document has no chunks"

        # The owner's answer cites the matching document; a filter selects the other one.
        question = "How many litres per minute does the pump run?"
        answer = _ask(client, owner, question, top_k=1)
        assert answer["status"] == "answered", "Owner query found no context"
        assert [item["document_id"] for item in answer["citations"]] == [pump]
        filtered = _ask(client, owner, question, document_ids=[garden])
        assert {item["document_id"] for item in filtered["citations"]} <= {garden}
        assert all(item["page"] == 1 for item in filtered["citations"])

        # Another account can neither read nor search these documents.
        expect(client.get(f"{DOCUMENTS}/{pump}", headers=stranger), 404)
        assert _ask(client, stranger, question)["status"] == "no_context"
        assert _ask(client, stranger, question, document_ids=[pump])["status"] == "no_context"

        # Deleting removes the chunks from search.
        for document_id in (pump, garden):
            expect(client.delete(f"{DOCUMENTS}/{document_id}", headers=owner), 204)
        assert _ask(client, owner, question)["status"] == "no_context"


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
        raise SystemExit("RAG smoke HTTP transport failed (details withheld)") from None
    print("PASS: upload, ingestion, cited answer, owner isolation, delete")


if __name__ == "__main__":
    main()
