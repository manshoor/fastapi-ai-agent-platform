"""Opt-in live evaluation of the resident assistant over real HTTP.

For each golden case (tests/evals/assistant_v1.jsonl):
    register an account -> demo residence -> upload the fixture documents as manager
      -> wait until ready -> send the message -> score the response and residence

The API under test calls whatever model it is configured with, so every run is
treated as paid: --allow-paid-requests is required. --judge also grades answered
cases with an LLM judge (are the claims supported by the cited snippets, and is the
question answered); it calls the provider directly and needs OPENAI_API_KEY.

Example (isolated AI stack on 8002, after explicit authorization):
  uv run --locked python scripts/evaluate_assistant.py --allow-paid-requests
  uv run --locked python scripts/evaluate_assistant.py --allow-paid-requests --judge \
    --judge-model MODEL --case doc-quiet-hours

Exit 0 when every check passes, 1 for failed checks or judge verdicts, 2 for invalid
configuration or transport failures. Prints only case IDs, check names and fixed
codes: never messages, answers, document text or credentials.
"""

import argparse
import asyncio
import importlib.util
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parent.parent
MESSAGES = "/api/v1/assistant/messages"
DOCUMENTS = "/api/v1/rag/documents"
SETTLED = frozenset({"ready", "failed"})


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_smoke_ai = _load("smoke_ai", "scripts/smoke_ai.py")
_evals = _load("assistant_evals", "tests/support/assistant_evals.py")
expect = _smoke_ai.expect


def _resident(client: httpx.Client) -> tuple[dict, dict]:
    email = f"assistant-eval-{uuid4().hex}@example.com"
    password = f"synthetic-{uuid4().hex}!"
    expect(client.post("/api/v1/auth/register", json={"email": email, "password": password}), 201)
    login = expect(
        client.post("/api/v1/auth/login", data={"username": email, "password": password}), 200
    ).json()
    headers = {"Authorization": f"Bearer {login['access_token']}"}
    view = expect(client.post("/api/v1/residence/demo", headers=headers), 201).json()
    return headers, view


def _upload_documents(client: httpx.Client, headers: dict, building_id: str, timeout: float):
    ids = []
    for document in _evals.load_documents():
        response = client.post(
            DOCUMENTS,
            params={"building_id": building_id},
            headers={**headers, "Idempotency-Key": f"eval-{uuid4().hex}"},
            data={"title": document["title"]},
            files={"file": (document["name"], document["text"].encode())},
        )
        ids.append(expect(response, 202).json()["id"])
    deadline = time.monotonic() + timeout
    for document_id in ids:
        while True:
            status = expect(client.get(f"{DOCUMENTS}/{document_id}", headers=headers), 200).json()
            if status["status"] in SETTLED:
                if status["status"] != "ready":
                    raise AssertionError("Fixture ingestion failed")
                break
            if time.monotonic() >= deadline:
                raise AssertionError("Fixture ingestion deadline exceeded")
            time.sleep(0.5)


async def _judge(settings, case: dict, body: dict, model: str | None) -> str:
    from app.ai.provider import OpenAIJudge

    # One client per event loop: each verdict runs in its own asyncio.run.
    judge = OpenAIJudge(settings)
    try:
        verdict = await judge.judge(
            case["message"],
            body["answer"],
            [citation["snippet"] for citation in body["citations"]],
            model=model,
        )
    finally:
        await judge.aclose()
    if verdict.grounded and verdict.answers_question:
        return "pass"
    return "ungrounded" if not verdict.grounded else "off_topic"


def run_case(client: httpx.Client, case: dict, timeout: float) -> tuple[dict, dict | None]:
    headers, view = _resident(client)
    _upload_documents(client, headers, view["building"]["id"], timeout)
    response = client.post(MESSAGES, headers=headers, json={"message": case["message"]})
    body = response.json() if response.status_code == 200 else None
    residence = expect(client.get("/api/v1/residence", headers=headers), 200).json()
    return _evals.score(case, body, residence), body


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument("--allow-paid-requests", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-model")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args(argv)
    if not args.allow_paid_requests:
        parser.error("Live evaluation requires --allow-paid-requests")
    if args.judge and not os.environ.get("OPENAI_API_KEY"):
        parser.error("--judge requires OPENAI_API_KEY")
    if args.judge_model and not args.judge:
        parser.error("--judge-model requires --judge")
    if not 1 <= args.timeout <= 600:
        parser.error("timeout must be 1..600 seconds")
    try:
        _smoke_ai.validate_destination(args.base_url)
    except ValueError:
        parser.error("unsafe destination; select the isolated AI API")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    cases = _evals.load_cases()
    if args.case:
        unknown = set(args.case) - {case["id"] for case in cases}
        if unknown:
            print("Unknown case ID", file=sys.stderr)
            return 2
        cases = [case for case in cases if case["id"] in args.case]
    judge = None
    if args.judge:
        from app.ai.config import ProviderSettings

        judge = ProviderSettings()
    failed = 0
    try:
        with httpx.Client(base_url=args.base_url, timeout=args.timeout, trust_env=False) as client:
            expect(client.get("/health/ready"), 200)
            for case in cases:
                scores, body = run_case(client, case, args.timeout)
                misses = sorted(name for name, ok in scores.items() if not ok)
                verdict = "-"
                if judge is not None and body is not None and body["status"] == "answered":
                    verdict = asyncio.run(_judge(judge, case, body, args.judge_model))
                passed = not misses and verdict in {"-", "pass"}
                failed += not passed
                print(
                    f"{case['id']}: {'PASS' if passed else 'FAIL'}"
                    f" failed={','.join(misses) or '-'} judge={verdict}"
                )
    except (httpx.HTTPError, AssertionError):
        print("Evaluation transport or setup failed (details withheld)", file=sys.stderr)
        return 2
    print(f"{len(cases) - failed}/{len(cases)} cases passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
