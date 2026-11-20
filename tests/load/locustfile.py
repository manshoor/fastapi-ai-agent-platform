"""Prepared-token CRUD workload with explicit local and distributed capacity gates."""

import time

from capacity import LatencyGate, canonical_uuid, install
from locust import FastHttpUser, between, task
from locust.exception import StopUser

# One gate over every request: p95 <= 500 ms and p99 <= 1000 ms.
install("item_id", (LatencyGate("all", "", p95_ms=500, p99_ms=1000),))


class ItemUser(FastHttpUser):
    wait_time = between(2, 5)

    def on_start(self):
        self.run = self.environment.capacity
        try:
            self.fixture = next(self.run.accounts)
        except StopIteration:
            self.run.abort("fixture exhausted; each virtual user requires a unique account")
            raise StopUser from None
        self.sequence = 0
        self.headers = {"Authorization": f"Bearer {self.fixture.token}"}

    def validate_item(self, value, expected_id=None):
        if not isinstance(value, dict) or set(value) != {
            "id",
            "title",
            "description",
            "created_at",
            "updated_at",
        }:
            raise ValueError("invalid item response shape")
        canonical_uuid(value["id"])
        if expected_id is not None and value["id"] != expected_id:
            raise ValueError("wrong item identity")
        known_owner = self.run.known_owners.get(value["id"])
        if known_owner is not None and known_owner != self.fixture.user_id:
            raise ValueError("response exposed another fixture account's item")
        if not all(isinstance(value[key], str) for key in value):
            raise ValueError("invalid item field type")

    def perform(self, method, path, name, payload=None):
        if time.time() >= self.fixture.expiry:
            self.run.abort("fixture token expired during traffic; seed fresh fixtures")
            raise StopUser
        with self.client.request(
            method, path, name=name, headers=self.headers, json=payload, catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"unexpected HTTP status {response.status_code}")
                if response.status_code == 401:
                    self.run.abort("authentication rejected prepared fixture token")
                return
            try:
                value = response.json()
                if method == "GET" and name == "/api/v1/items?limit=20":
                    if not isinstance(value, dict) or set(value) != {"items", "next_cursor"}:
                        raise ValueError("invalid list response shape")
                    if not isinstance(value["items"], list) or not 1 <= len(value["items"]) <= 20:
                        raise ValueError("invalid list size")
                    if value["next_cursor"] is not None and not isinstance(
                        value["next_cursor"], str
                    ):
                        raise ValueError("invalid cursor type")
                    for item in value["items"]:
                        self.validate_item(item)
                    ids = [item["id"] for item in value["items"]]
                    if len(ids) != len(set(ids)):
                        raise ValueError("duplicate items in list")
                    if value["next_cursor"] is None and self.fixture.resource_id not in ids:
                        raise ValueError("complete owner list omitted its known fixture item")
                else:
                    self.validate_item(value, self.fixture.resource_id)
                    if method == "PATCH" and value["description"] != payload["description"]:
                        raise ValueError("PATCH response does not reflect its mutation")
            except (ValueError, TypeError, KeyError) as error:
                response.failure(f"invalid response: {error}")

    @task(70)
    def detail(self):
        self.perform("GET", f"/api/v1/items/{self.fixture.resource_id}", "/api/v1/items/{item_id}")

    @task(20)
    def listing(self):
        self.perform("GET", "/api/v1/items?limit=20", "/api/v1/items?limit=20")

    @task(10)
    def patch(self):
        self.sequence += 1
        self.perform(
            "PATCH",
            f"/api/v1/items/{self.fixture.resource_id}",
            "/api/v1/items/{item_id}",
            {
                "description": (
                    f"worker={self.run.index} user={self.fixture.user_id} seq={self.sequence}"
                )
            },
        )
