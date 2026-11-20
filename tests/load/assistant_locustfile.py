"""Prepared-token resident assistant workload with the shared capacity gates.

One seeded resident per virtual user (scripts/seed_assistant_load.py). Turns run against
the provider simulator, whose keyword policy picks one tool per message:

    message keyword        tool              success
    (none of the below)    search_documents  200, "answered", at least one citation
    "booking"              list_bookings     200, "answered", action allowed/ok
    "thermostat"           set_thermostat    200, "answered", action allowed/ok
    "lock"                 lock_door         200, "answered", action allowed/ok

Unlock and confirm are left out: they are limited to 3 per hour and covered by tests.
"""

import os
import random
import time

from capacity import LatencyGate, install, number
from locust import FastHttpUser, between, task
from locust.exception import StopUser

TURN_PREFIX = "turn: "
MESSAGES_PATH = "/api/v1/assistant/messages"
RESIDENCE_PATH = "/api/v1/residence"
HTTP_OK = 200
HTTP_UNAUTHORIZED = 401
# Document questions must avoid every routing keyword of the simulator, including
# "lock" inside words such as "clock" and "block".
QUESTIONS = (
    "What are the quiet hours?",
    "Can I keep a dog in my apartment?",
    "Where do the recycling bins go?",
    "When is the gym open?",
    "Where can visitors park?",
    "How long does the front desk hold parcels?",
)
BOOKINGS_MESSAGE = "Show my upcoming bookings."
# The simulator sets the first number in the message; keep it the only number.
THERMOSTAT_TARGETS = (20.0, 20.5, 21.0, 21.5, 22.0)
LOCK_MESSAGE = "Please lock the front door."
GATE_OFF = "off"

TURN_P95_MS = number("LOAD_TURN_P95_MS", "5000", 1)
# "off" drops the residence gate; the 200-user tier gates turns only.
RESIDENCE_P95 = os.getenv("LOAD_RESIDENCE_P95_MS", "500")
GATES = (LatencyGate("turns", TURN_PREFIX, p95_ms=TURN_P95_MS),)
if RESIDENCE_P95 != GATE_OFF:
    GATES += (
        LatencyGate("residence", RESIDENCE_PATH, p95_ms=number("LOAD_RESIDENCE_P95_MS", "500", 1)),
    )
install("apartment_id", GATES)


def _detail(response) -> str:
    """The server's fixed error text, or the body's type when it has none."""
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        return "no JSON body"
    return detail if isinstance(detail, str) else type(detail).__name__


class ResidentUser(FastHttpUser):
    wait_time = between(8, 15)

    def on_start(self):
        self.run = self.environment.capacity
        try:
            self.fixture = next(self.run.accounts)
        except StopIteration:
            self.run.abort("fixture exhausted; each virtual user requires a unique resident")
            raise StopUser from None
        self.headers = {"Authorization": f"Bearer {self.fixture.token}"}

    def check_token(self):
        if time.time() >= self.fixture.expiry:
            self.run.abort("fixture token expired during traffic; seed fresh fixtures")
            raise StopUser

    def refused(self, response) -> bool:
        """Mark a non-200 response failed with its status and fixed detail."""
        if response.status_code == HTTP_OK:
            return False
        response.failure(f"HTTP {response.status_code} {_detail(response)}")
        if response.status_code == HTTP_UNAUTHORIZED:
            self.run.abort("authentication rejected prepared fixture token")
        return True

    def turn(self, kind: str, message: str, tool: str | None):
        """One assistant turn; `tool` is the action it must report, None for a search."""
        self.check_token()
        with self.client.post(
            MESSAGES_PATH,
            name=f"{TURN_PREFIX}{kind}",
            headers=self.headers,
            json={"message": message},
            catch_response=True,
        ) as response:
            if self.refused(response):
                return
            try:
                body = response.json()
                if body["status"] != "answered":
                    response.failure(f"turn status {body['status']}")
                    return
                if tool is None and not body["citations"]:
                    response.failure("answered without citations")
                    return
                actions = [(a["tool"], a["decision"], a["result"]) for a in body["actions"]]
                if tool is not None and (tool, "allowed", "ok") not in actions:
                    response.failure(f"no allowed {tool} action")
            except (ValueError, TypeError, KeyError) as error:
                response.failure(f"invalid turn response: {type(error).__name__}")

    @task(50)
    def document(self):
        self.turn("document", random.choice(QUESTIONS), None)

    @task(15)
    def bookings(self):
        self.turn("bookings", BOOKINGS_MESSAGE, "list_bookings")

    @task(15)
    def residence(self):
        self.check_token()
        with self.client.get(
            RESIDENCE_PATH, name=RESIDENCE_PATH, headers=self.headers, catch_response=True
        ) as response:
            if self.refused(response):
                return
            try:
                body = response.json()
                if body["apartment"]["id"] != self.fixture.resource_id:
                    response.failure("residence of another apartment")
                elif not body["devices"]:
                    response.failure("residence without devices")
            except (ValueError, TypeError, KeyError) as error:
                response.failure(f"invalid residence response: {type(error).__name__}")

    @task(10)
    def thermostat(self):
        celsius = random.choice(THERMOSTAT_TARGETS)
        message = f"Set the living room thermostat to {celsius:g} degrees."
        self.turn("thermostat", message, "set_thermostat")

    @task(10)
    def lock(self):
        self.turn("lock", LOCK_MESSAGE, "lock_door")
