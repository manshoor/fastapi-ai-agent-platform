"""Exercise isolated AI rewrite over real HTTP, with no paid service or credential output."""

import argparse
import time
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

TERMINAL = frozenset({"succeeded", "failed", "needs_reconciliation"})


def expect(response: httpx.Response, status: int) -> httpx.Response:
    if response.status_code != status:
        raise AssertionError(
            f"{response.request.method} {response.request.url.path}: "
            f"expected {status}, received {response.status_code}"
        )
    return response


def validate_destination(base_url: str) -> None:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "http"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1", "api"}
    ):
        raise ValueError("smoke requires an isolated loopback or Compose api HTTP destination")
    if parsed.hostname == "api":
        if parsed.port != 8000:
            raise ValueError("Compose api must use internal port 8000")
    elif parsed.port is None or not 1024 <= parsed.port <= 65535 or parsed.port in {8000, 8001}:
        raise ValueError("use the isolated AI port, not normal 8000 or benchmark 8001")


def run_smoke(base_url: str, *, timeout_seconds: float = 60) -> None:
    validate_destination(base_url)
    with httpx.Client(base_url=base_url, timeout=5, trust_env=False) as client:
        expect(client.get("/health/ready"), 200)
        headers = []
        for _ in range(2):
            email = f"ai-smoke-{uuid4().hex}@example.com"
            password = f"synthetic-{uuid4().hex}!"
            expect(
                client.post("/api/v1/auth/register", json={"email": email, "password": password}),
                201,
            )
            login = expect(
                client.post("/api/v1/auth/login", data={"username": email, "password": password}),
                200,
            ).json()
            headers.append({"Authorization": f"Bearer {login['access_token']}"})
        owner, stranger = headers
        source = expect(
            client.post(
                "/api/v1/items",
                headers=owner,
                json={
                    "title": "Synthetic source",
                    "description": "Synthetic original description.",
                },
            ),
            201,
        ).json()
        item_path = f"/api/v1/items/{source['id']}"
        rewrite_path = f"{item_path}/ai/rewrite"
        key = f"smoke-{uuid4().hex}"
        request_headers = {**owner, "Idempotency-Key": key}
        payload = {"instruction": "Make the synthetic wording clearer without adding facts."}
        submitted = expect(
            client.post(rewrite_path, headers=request_headers, json=payload), 202
        ).json()
        job_path = f"/api/v1/ai/jobs/{submitted['id']}"
        assert submitted["status_url"] == job_path, "Unexpected job status location"
        expect(client.get(job_path, headers=stranger), 404)
        expect(
            client.post(
                rewrite_path,
                headers={**stranger, "Idempotency-Key": f"smoke-{uuid4().hex}"},
                json=payload,
            ),
            404,
        )
        expect(
            client.post(
                rewrite_path,
                headers=request_headers,
                json={"instruction": "Different synthetic request."},
            ),
            409,
        )
        replay = client.post(rewrite_path, headers=request_headers, json=payload)
        assert replay.status_code in {200, 202}, "Idempotent replay was not accepted"
        assert replay.json()["id"] == submitted["id"], "Idempotent replay created a new job"
        deadline = time.monotonic() + timeout_seconds
        while True:
            job = expect(client.get(job_path, headers=owner), 200).json()
            if job["status"] in TERMINAL:
                break
            if time.monotonic() >= deadline:
                raise AssertionError("Rewrite completion deadline exceeded")
            time.sleep(0.1)
        assert job["status"] == "succeeded", "Synthetic rewrite did not succeed"
        suggestion = job["suggestion"]
        assert isinstance(suggestion, dict) and set(suggestion) == {"title", "description"}
        assert suggestion["title"] and suggestion["description"]
        unchanged = expect(client.get(item_path, headers=owner), 200).json()
        for field in ("title", "description", "updated_at"):
            assert unchanged[field] == source[field], "Rewrite silently changed source item"
        expect(client.patch(item_path, headers=stranger, json=suggestion), 404)
        applied = expect(client.patch(item_path, headers=owner, json=suggestion), 200).json()
        assert all(applied[field] == suggestion[field] for field in suggestion)
        deadline = time.monotonic() + 6
        while True:
            current = expect(client.get(item_path, headers=owner), 200).json()
            if all(current[field] == suggestion[field] for field in suggestion):
                break
            if time.monotonic() >= deadline:
                raise AssertionError("Explicit PATCH cache convergence deadline exceeded")
            time.sleep(0.1)
        expect(client.delete(item_path, headers=owner), 204)


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
        raise SystemExit("AI smoke HTTP transport failed (details withheld)") from None
    print("PASS: durable rewrite, idempotency, ownership, no auto-apply, explicit PATCH")


if __name__ == "__main__":
    main()
