"""Exercise real account and owner-scoped CRUD behavior over HTTP."""

import argparse
import time
from uuid import uuid4

import httpx


def expect(response: httpx.Response, status: int) -> httpx.Response:
    if response.status_code != status:
        # Never print response bodies: validation errors can contain submitted credentials.
        raise AssertionError(
            f"{response.request.method} {response.request.url.path}: "
            f"expected {status}, received {response.status_code}"
        )
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    with httpx.Client(base_url=args.base_url, timeout=10) as client:
        expect(client.get("/health/ready"), 200)
        headers = []
        for _ in range(2):
            email = f"smoke-{uuid4().hex}@example.com"
            password = f"{uuid4().hex}!"
            registered = expect(
                client.post(
                    "/api/v1/auth/register",
                    json={
                        "email": email,
                        "password": password,
                    },
                ),
                201,
            ).json()
            assert registered["email"] == email
            assert "password" not in registered and "password_hash" not in registered
            token = expect(
                client.post(
                    "/api/v1/auth/login",
                    data={
                        "username": email,
                        "password": password,
                    },
                ),
                200,
            ).json()
            headers.append({"Authorization": f"Bearer {token['access_token']}"})
        owner, stranger = headers
        item = expect(
            client.post(
                "/api/v1/items",
                headers=owner,
                json={"title": "Smoke item", "description": "initial"},
            ),
            201,
        ).json()
        path = f"/api/v1/items/{item['id']}"
        assert expect(client.get(path, headers=owner), 200).json()["title"] == "Smoke item"
        assert expect(client.get(path, headers=owner), 200).json()["description"] == "initial"
        expect(client.get(path, headers=stranger), 404)
        expect(client.patch(path, headers=stranger, json={"title": "stolen"}), 404)
        expect(client.delete(path, headers=stranger), 404)
        page = expect(client.get("/api/v1/items", headers=owner), 200).json()
        assert any(row["id"] == item["id"] for row in page["items"])
        changed = expect(
            client.patch(path, headers=owner, json={"description": "changed"}), 200
        ).json()
        assert changed["description"] == "changed"
        deadline = time.monotonic() + 6
        while True:
            read = expect(client.get(path, headers=owner), 200).json()
            if read["description"] == "changed":
                break
            if time.monotonic() >= deadline:
                raise AssertionError("Item cache did not converge after PATCH")
            time.sleep(0.1)
        deleted = expect(client.delete(path, headers=owner), 204)
        assert not deleted.content
        deadline = time.monotonic() + 6
        while True:
            read = client.get(path, headers=owner)
            if read.status_code == 404:
                break
            expect(read, 200)
            if time.monotonic() >= deadline:
                raise AssertionError("Item cache did not converge after DELETE")
            time.sleep(0.1)
        expect(client.delete(path, headers=owner), 404)
        print("PASS: register/login, persisted CRUD, owner isolation, cache convergence, deletion")


if __name__ == "__main__":
    main()
