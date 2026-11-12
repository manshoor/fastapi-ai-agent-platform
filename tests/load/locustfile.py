"""Prepared-token CRUD workload with explicit local and distributed capacity gates."""

import csv
import hashlib
import json
import logging
import math
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import gevent
import jwt
from locust import FastHttpUser, between, events, task
from locust.exception import StopUser
from locust.runners import MasterRunner, WorkerRunner
from locust.stats import calculate_response_time_percentile

logger = logging.getLogger(__name__)
HANDSHAKE_SECONDS = 30


@dataclass(frozen=True)
class Fixture:
    user_id: str
    item_id: str
    token: str
    expiry: float


def canonical_uuid(value: str) -> str:
    if str(UUID(value)) != value:
        raise ValueError("fixture UUID is not canonical")
    return value


def boolean(name: str, default: str) -> bool:
    value = os.getenv(name, default).lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return value == "true"


def number(name: str, default: str, minimum: float) -> float:
    value = float(os.getenv(name, default))
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return value


class CapacityRun:
    def __init__(self, environment):
        self.environment = environment
        self.runner = environment.runner
        self.quitting_requested = False
        self.master = isinstance(self.runner, MasterRunner)
        self.worker = isinstance(self.runner, WorkerRunner)
        self.errors: list[str] = []
        self.started = False
        self.stopping = False
        self.prepared = False
        self.measuring = False
        self.measure_start = None
        self.measure_end = None
        self.minimum_users = None
        self.whole_failures = 0
        self.measured_requests = 0
        self.measured_failures = 0
        self.histogram: Counter = Counter()
        self.ready = {}
        self.snapshots = {}
        self.reset_acks = set()
        self.final_reports = set()
        self.worker_ids = set()
        self.initialized_workers = set()
        self.fixtures: tuple[Fixture, ...] = ()
        self.known_owners: dict[str, str] = {}
        self.accounts = iter(())
        self.index = 0
        self.count = 1
        self.target = 0
        self.duration = 0
        self.warmup = number("LOAD_WARMUP_SECONDS", "30", 0)
        self.minimum_steady = number("LOAD_MIN_STEADY_SECONDS", "30", 0.001)
        self.maximum_failure_ratio = number("LOAD_MAX_FAILURE_RATIO", "0.01", 0)
        if self.maximum_failure_ratio > 1:
            raise ValueError("LOAD_MAX_FAILURE_RATIO must be <= 1")
        self.zero_failures = boolean("LOAD_ZERO_FAILURES", "false")
        if self.worker:
            self.runner.register_message("capacity_prepare", self.prepare_worker, concurrent=True)
            self.runner.register_message("capacity_measure", self.measure_worker)
            self.runner.register_message("capacity_finish", self.finish_worker, concurrent=True)
            # client_ready precedes events.init, so it cannot prove these handlers exist.
            self.runner.send_message("capacity_initialized", {})
        elif self.master:
            # Locust's main greenlet waits independently of runner.quit(). Bound
            # its own worker wait as well, or a missing worker can hang forever.
            options = environment.parsed_options
            if not options.expect_workers_max_wait:
                options.expect_workers_max_wait = HANDSHAKE_SECONDS
            self.runner.register_message("capacity_initialized", self.receive_initialized)
            self.runner.register_message("capacity_ready", self.receive_ready)
            self.runner.register_message("capacity_reset_ack", self.receive_ack)
            self.runner.register_message("capacity_snapshot", self.receive_snapshot)
            self.runner.register_message("capacity_abort", self.receive_abort)
            # Locust's --expect-workers otherwise waits forever when a shard is absent.
            gevent.spawn(self.startup_deadline)
        if not self.worker:
            original_start = self.runner.start

            def guarded_start(user_count, spawn_rate, **kwargs):
                if self.started or self.prepared:
                    self.abort("one fixed-target run per process is required")
                    return
                try:
                    self.prepare(user_count)
                except Exception as error:
                    self.abort(f"preflight failed: {error}")
                    return
                original_start(user_count, spawn_rate, **kwargs)

            self.runner.start = guarded_start

    def startup_deadline(self):
        gevent.sleep(HANDSHAKE_SECONDS)
        if not self.prepared and not self.errors:
            self.abort("missing workers: startup/preflight did not finish within 30 seconds")

    def abort(self, reason):
        if reason not in self.errors:
            self.errors.append(reason)
            logger.error("CAPACITY FAIL: %s", reason)
        self.environment.process_exit_code = 1
        if self.worker:
            self.runner.send_message("capacity_abort", {"reason": reason})
        elif not self.stopping and not self.quitting_requested:
            self.quitting_requested = True
            gevent.spawn(self.runner.quit)

    def load_fixtures(self, duration, target, distributed):
        if distributed:
            # Explicit indices, never Locust's automatic worker index or --processes.
            self.index = int(os.environ["LOAD_WORKER_INDEX"])
            self.count = int(os.environ["LOAD_WORKER_COUNT"])
            if self.count < 1 or not 0 <= self.index < self.count:
                raise ValueError("invalid LOAD_WORKER_INDEX/LOAD_WORKER_COUNT")
        elif "LOAD_WORKER_INDEX" in os.environ or "LOAD_WORKER_COUNT" in os.environ:
            raise ValueError("shard settings require a distributed worker")
        path = Path(os.environ["LOAD_USERS_FILE"])
        assigned = []
        seen = set()
        digest = hashlib.sha256()
        expiry_floor = time.time() + duration + 120
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["user_id", "item_id", "access_token"]:
                raise ValueError("fixture CSV columns must be user_id,item_id,access_token")
            for row_index, row in enumerate(reader):
                user_id = canonical_uuid(row["user_id"])
                item_id = canonical_uuid(row["item_id"])
                if user_id in seen:
                    raise ValueError("fixture contains duplicate user accounts")
                seen.add(user_id)
                if item_id in self.known_owners:
                    raise ValueError("fixture contains duplicate item IDs")
                self.known_owners[item_id] = user_id
                digest.update(json.dumps(row, sort_keys=True).encode())
                if row_index % self.count != self.index:
                    continue
                # Expiry validation only. The API still verifies the signature.
                claims = jwt.decode(row["access_token"], options={"verify_signature": False})
                expiry = claims.get("exp")
                if (
                    claims.get("sub") != user_id
                    or isinstance(expiry, bool)
                    or not isinstance(expiry, (int, float))
                    or not math.isfinite(expiry)
                    or expiry < expiry_floor
                ):
                    raise ValueError("fixture token invalid or too old; seed fresh fixtures")
                assigned.append(Fixture(user_id, item_id, row["access_token"], expiry))
        required = math.ceil(target / self.count)
        if len(assigned) < required:
            raise ValueError(f"insufficient shard accounts: need {required}, found {len(assigned)}")
        self.fixtures = tuple(assigned)
        self.accounts = iter(self.fixtures)
        return {"index": self.index, "count": self.count, "digest": digest.hexdigest()}

    def prepare(self, target):
        options = self.environment.parsed_options
        self.target = int(target)
        self.duration = float(options.run_time or 0)
        if self.target < 1 or self.duration < self.warmup + self.minimum_steady:
            raise ValueError("--users must be positive; --run-time must cover warmup + steady time")
        if self.environment.reset_stats:
            raise ValueError("do not use --reset-stats; this harness owns the warmup reset")
        if self.master:
            self.worker_ids = {client.id for client in self.runner.clients.all}
            expected = int(options.expect_workers)
            if expected < 1 or len(self.worker_ids) != expected:
                raise ValueError("connected worker count must equal --expect-workers")
            deadline = time.monotonic() + HANDSHAKE_SECONDS
            while self.initialized_workers != self.worker_ids and time.monotonic() < deadline:
                self.check_membership()
                if self.errors:
                    raise ValueError("worker initialization rejected")
                if self.initialized_workers - self.worker_ids:
                    raise ValueError("initialization acknowledgement from unexpected worker")
                gevent.sleep(0.05)
            self.check_membership()
            if self.errors:
                raise ValueError("worker initialization rejected")
            if self.initialized_workers != self.worker_ids:
                raise ValueError("missing worker initialization acknowledgements")
            self.runner.send_message(
                "capacity_prepare",
                {
                    "target": self.target,
                    "duration": self.duration,
                    "warmup": self.warmup,
                    "minimum_steady": self.minimum_steady,
                    "maximum_failure_ratio": self.maximum_failure_ratio,
                    "zero_failures": self.zero_failures,
                },
            )
            while set(self.ready) != self.worker_ids and time.monotonic() < deadline:
                if self.errors:
                    raise ValueError("worker preflight rejected")
                gevent.sleep(0.05)
            if set(self.ready) != self.worker_ids:
                raise ValueError("missing worker preflight acknowledgements")
            reports = list(self.ready.values())
            counts = {report["count"] for report in reports}
            indices = [report["index"] for report in reports]
            if counts != {expected} or sorted(indices) != list(range(expected)):
                raise ValueError(
                    "duplicate/missing shard indices or inconsistent LOAD_WORKER_COUNT"
                )
            if len({report["digest"] for report in reports}) != 1:
                raise ValueError("workers must use the same fixture CSV")
        else:
            self.load_fixtures(self.duration, self.target, False)
        if self.errors:
            raise ValueError("preflight rejected")
        self.prepared = True
        logger.info("Capacity preflight passed: target=%s duration=%ss", self.target, self.duration)

    def prepare_worker(self, environment, msg, **kwargs):
        try:
            if self.prepared:
                raise ValueError("worker cannot be reused for another run")
            for name, value in msg.data.items():
                setattr(self, name, value)
            report = self.load_fixtures(self.duration, self.target, True)
            self.prepared = True
            self.runner.send_message("capacity_ready", report)
        except Exception as error:
            self.abort(f"worker preflight failed: {error}")

    def receive_initialized(self, environment, msg, **kwargs):
        # An acknowledgement may arrive before prepare() snapshots connected workers.
        self.initialized_workers.add(msg.node_id)

    def receive_ready(self, environment, msg, **kwargs):
        if msg.node_id not in self.worker_ids or msg.node_id in self.ready:
            self.abort("unexpected or duplicate worker preflight response")
            return
        self.ready[msg.node_id] = msg.data

    def receive_abort(self, environment, msg, **kwargs):
        self.abort(f"worker {msg.node_id}: {msg.data['reason']}")

    def start(self):
        if not self.prepared:
            self.abort("traffic attempted before successful preflight")
            return
        self.started = True
        if not self.worker:
            gevent.spawn(self.monitor)
        if self.master:
            gevent.spawn(self.watch_membership)

    def check_membership(self):
        connected = {client.id for client in self.runner.clients.all if client.state != "missing"}
        if connected != self.worker_ids:
            self.abort("worker disconnected or membership changed")

    def watch_membership(self):
        while not self.stopping:
            self.check_membership()
            gevent.sleep(0.1)

    def check_users(self):
        actual = self.runner.user_count
        if self.minimum_users is None:
            self.minimum_users = actual
        else:
            self.minimum_users = min(self.minimum_users, actual)
        if actual != self.target:
            self.abort(f"global active users {actual} != target {self.target}")
        if self.master:
            self.check_membership()

    def monitor(self):
        gevent.sleep(self.warmup)
        if self.stopping or self.errors:
            return
        self.check_users()
        if self.errors:
            return
        self.measure_start = time.monotonic()
        if self.master:
            self.runner.send_message("capacity_measure", {})
            deadline = time.monotonic() + 5
            while self.reset_acks != self.worker_ids and time.monotonic() < deadline:
                gevent.sleep(0.01)
            if self.reset_acks != self.worker_ids:
                self.abort("missing warmup reset acknowledgement")
                return
            # Each worker's earlier RPC stats precede its acknowledgement. Reset
            # the master's aggregate only after the barrier drains warmup reports.
            self.environment.stats.reset_all()
            # Start the conservative steady-duration clock after all workers reset.
            self.measure_start = time.monotonic()
        else:
            self.environment.stats.reset_all()
            self.measuring = True
        logger.info("Warmup complete; measured capacity window started at target=%s", self.target)
        while not self.stopping:
            self.check_users()
            gevent.sleep(0.1)

    def measure_worker(self, environment, msg, **kwargs):
        self.environment.stats.reset_all()
        self.measuring = True
        self.runner.send_message("capacity_reset_ack", {})

    def receive_ack(self, environment, msg, **kwargs):
        self.reset_acks.add(msg.node_id)

    def request(self, response_time, exception):
        if exception is not None:
            self.whole_failures += 1
        if self.measuring:
            self.measured_requests += 1
            self.measured_failures += int(exception is not None)
            # Same bounded histogram rounding used by Locust StatsEntry.
            elapsed = response_time or 0
            if elapsed < 100:
                bucket = round(elapsed)
            elif elapsed < 1000:
                bucket = round(elapsed, -1)
            elif elapsed < 10000:
                bucket = round(elapsed, -2)
            else:
                bucket = round(elapsed, -3)
            self.histogram[int(bucket)] += 1

    def snapshot(self):
        return {
            "requests": self.measured_requests,
            "failures": self.measured_failures,
            "histogram": dict(self.histogram),
            "whole_failures": self.whole_failures,
            "cpu_warning": self.runner.cpu_warning_emitted,
        }

    def finish_worker(self, environment, msg, **kwargs):
        self.measuring = False
        self.stopping = True
        # Complete admitted requests before reporting whole-run failures. This
        # handshake precedes the master's built-in quit, whose RPC loop exits early.
        self.runner.stop()
        self.runner.send_message("capacity_snapshot", {**self.snapshot(), "final": True})

    def receive_snapshot(self, environment, msg, **kwargs):
        if msg.node_id not in self.worker_ids:
            self.abort("snapshot from unexpected worker")
            return
        self.snapshots[msg.node_id] = msg.data
        if msg.data.get("final"):
            self.final_reports.add(msg.node_id)

    def stopping_test(self):
        if self.stopping:
            return
        if not self.worker and self.measure_start is not None:
            self.check_users()
            self.measure_end = time.monotonic()
        self.stopping = True
        self.measuring = False
        if self.master and self.prepared:
            self.runner.send_message("capacity_finish", {})
            deadline = time.monotonic() + (self.environment.stop_timeout or 0) + 10
            while self.final_reports != self.worker_ids and time.monotonic() < deadline:
                gevent.sleep(0.01)
            if self.final_reports != self.worker_ids:
                self.abort("missing final worker counters/histograms")

    def evaluate(self):
        if self.worker:
            self.environment.process_exit_code = int(bool(self.errors))
            return
        reports = list(self.snapshots.values()) if self.master else [self.snapshot()]
        histogram = Counter()
        for report in reports:
            histogram.update({int(key): count for key, count in report["histogram"].items()})
        requests = sum(report["requests"] for report in reports)
        failures = sum(report["failures"] for report in reports)
        whole_failures = sum(report["whole_failures"] for report in reports)
        steady = (
            self.measure_end - self.measure_start
            if self.measure_end is not None and self.measure_start is not None
            else 0
        )
        ratio = failures / requests if requests else 1
        p95 = calculate_response_time_percentile(histogram, requests, 0.95) if requests else 0
        p99 = calculate_response_time_percentile(histogram, requests, 0.99) if requests else 0
        reasons = []
        if not self.prepared or not self.started:
            reasons.append("run never started after preflight")
        if steady < self.minimum_steady:
            reasons.append(f"steady duration {steady:.2f}s < {self.minimum_steady}s")
        if not requests:
            reasons.append("no measured requests")
        if ratio > self.maximum_failure_ratio:
            reasons.append(f"failure ratio {ratio:.5f} > {self.maximum_failure_ratio}")
        if p95 > 500 or p99 > 1000:
            reasons.append("latency gate exceeded (p95 <=500ms, p99 <=1000ms)")
        if self.zero_failures and whole_failures:
            reasons.append(f"whole-run failures={whole_failures}, including warmup/shutdown")
        if self.runner.exceptions:
            reasons.append("unhandled user/runner exceptions")
        self.errors.extend(reasons)
        self.environment.process_exit_code = int(bool(self.errors))
        logger.info(
            "CAPACITY %s target=%s minimum_active=%s steady=%.2fs requests=%s "
            "failures=%s ratio=%.5f whole_run_failures=%s p95=%sms p99=%sms "
            "generator_cpu_warning=%s reasons=%s",
            "FAIL" if self.errors else "PASS",
            self.target,
            self.minimum_users,
            steady,
            requests,
            failures,
            ratio,
            whole_failures,
            p95,
            p99,
            self.runner.cpu_warning_emitted or any(r["cpu_warning"] for r in reports),
            self.errors,
        )


@events.init.add_listener
def initialize(environment, **kwargs):
    try:
        environment.capacity = CapacityRun(environment)
    except Exception as error:
        logger.error("Capacity configuration rejected: %s", error)
        environment.process_exit_code = 1
        gevent.spawn(environment.runner.quit)


@events.test_start.add_listener
def start(environment, **kwargs):
    if hasattr(environment, "capacity"):
        environment.capacity.start()


@events.test_stopping.add_listener
def stopping(environment, **kwargs):
    if hasattr(environment, "capacity"):
        environment.capacity.stopping_test()


@events.request.add_listener
def request(response_time, exception, **kwargs):
    # The event hook supplies request context but not Environment.
    if _run is not None:
        _run.request(response_time, exception)


_run = None


@events.init.add_listener
def remember_run(environment, **kwargs):
    global _run
    _run = getattr(environment, "capacity", None)


@events.report_to_master.add_listener
def report_to_master(client_id, data, **kwargs):
    if _run is not None:
        data["capacity"] = _run.snapshot()


@events.worker_report.add_listener
def worker_report(client_id, data, **kwargs):
    if _run is not None and "capacity" in data and client_id not in _run.final_reports:
        _run.snapshots[client_id] = data["capacity"]
    if _run is not None and _run.measure_start is not None and not _run.stopping:
        _run.check_users()


@events.quitting.add_listener
def quitting(environment, **kwargs):
    if hasattr(environment, "capacity"):
        environment.capacity.evaluate()
    else:
        environment.process_exit_code = 1


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
                    if value["next_cursor"] is None and self.fixture.item_id not in ids:
                        raise ValueError("complete owner list omitted its known fixture item")
                else:
                    self.validate_item(value, self.fixture.item_id)
                    if method == "PATCH" and value["description"] != payload["description"]:
                        raise ValueError("PATCH response does not reflect its mutation")
            except (ValueError, TypeError, KeyError) as error:
                response.failure(f"invalid response: {error}")

    @task(70)
    def detail(self):
        self.perform("GET", f"/api/v1/items/{self.fixture.item_id}", "/api/v1/items/{item_id}")

    @task(20)
    def listing(self):
        self.perform("GET", "/api/v1/items?limit=20", "/api/v1/items?limit=20")

    @task(10)
    def patch(self):
        self.sequence += 1
        self.perform(
            "PATCH",
            f"/api/v1/items/{self.fixture.item_id}",
            "/api/v1/items/{item_id}",
            {
                "description": (
                    f"worker={self.run.index} user={self.fixture.user_id} seq={self.sequence}"
                )
            },
        )
