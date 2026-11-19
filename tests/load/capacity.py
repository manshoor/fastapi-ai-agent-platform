"""Shared capacity run for the Locust workloads: fixtures, warmup, gates and the verdict.

Each locustfile calls `install()` once with its fixture column and latency gates:

    tests/load/locustfile.py            user_id,item_id,access_token       all requests
    tests/load/assistant_locustfile.py  user_id,apartment_id,access_token  turns, residence

The run passes only when the failure ratio, the whole-run failure rule and every latency
gate hold over the measured (post-warmup) window. Locust's exit code carries the verdict.
"""

import csv
import hashlib
import json
import logging
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import gevent
import jwt
from locust import events
from locust.runners import MasterRunner, WorkerRunner
from locust.stats import calculate_response_time_percentile

logger = logging.getLogger(__name__)
HANDSHAKE_SECONDS = 30


@dataclass(frozen=True)
class Fixture:
    user_id: str
    # The one resource the account owns: an item, or an apartment.
    resource_id: str
    token: str
    expiry: float


@dataclass(frozen=True)
class LatencyGate:
    """Latency limits over measured requests whose Locust name starts with `prefix`."""

    label: str
    prefix: str
    p95_ms: float
    p99_ms: float | None = None

    def describe(self) -> str:
        limits = f"p95 <={self.p95_ms:g}ms"
        if self.p99_ms is not None:
            limits += f", p99 <={self.p99_ms:g}ms"
        return limits


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


def _bucket(response_time) -> int:
    # Same bounded histogram rounding used by Locust StatsEntry.
    elapsed = response_time or 0
    if elapsed < 100:
        return int(round(elapsed))
    if elapsed < 1000:
        return int(round(elapsed, -1))
    if elapsed < 10000:
        return int(round(elapsed, -2))
    return int(round(elapsed, -3))


def _percentiles(histogram: Counter) -> tuple[int, int]:
    count = sum(histogram.values())
    if not count:
        return 0, 0
    return (
        calculate_response_time_percentile(histogram, count, 0.95),
        calculate_response_time_percentile(histogram, count, 0.99),
    )


class CapacityRun:
    def __init__(self, environment, resource_column: str, gates: tuple[LatencyGate, ...]):
        self.environment = environment
        self.runner = environment.runner
        self.resource_column = resource_column
        self.gates = gates
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
        # Measured latency histogram per Locust request name.
        self.histograms: defaultdict[str, Counter] = defaultdict(Counter)
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
        columns = ["user_id", self.resource_column, "access_token"]
        assigned = []
        seen = set()
        digest = hashlib.sha256()
        expiry_floor = time.time() + duration + 120
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != columns:
                raise ValueError(f"fixture CSV columns must be {','.join(columns)}")
            for row_index, row in enumerate(reader):
                user_id = canonical_uuid(row["user_id"])
                resource_id = canonical_uuid(row[self.resource_column])
                if user_id in seen:
                    raise ValueError("fixture contains duplicate user accounts")
                seen.add(user_id)
                if resource_id in self.known_owners:
                    raise ValueError(f"fixture contains duplicate {self.resource_column} values")
                self.known_owners[resource_id] = user_id
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
                assigned.append(Fixture(user_id, resource_id, row["access_token"], expiry))
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

    def request(self, name, response_time, exception):
        if exception is not None:
            self.whole_failures += 1
        if self.measuring:
            self.measured_requests += 1
            self.measured_failures += int(exception is not None)
            self.histograms[name][_bucket(response_time)] += 1

    def snapshot(self):
        return {
            "requests": self.measured_requests,
            "failures": self.measured_failures,
            "histograms": {name: dict(histogram) for name, histogram in self.histograms.items()},
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

    def gate_reasons(self, histograms: dict[str, Counter]) -> tuple[list[str], dict]:
        """Each gate's measured latency, and a reason for every gate that fails."""
        reasons, latency = [], {}
        for gate in self.gates:
            merged = Counter()
            for name, histogram in histograms.items():
                if name.startswith(gate.prefix):
                    merged.update(histogram)
            count = sum(merged.values())
            p95, p99 = _percentiles(merged)
            latency[gate.label] = {"requests": count, "p95": p95, "p99": p99}
            if not count:
                reasons.append(f"no measured requests for latency gate {gate.label}")
            elif p95 > gate.p95_ms or (gate.p99_ms is not None and p99 > gate.p99_ms):
                reasons.append(f"latency gate {gate.label} exceeded ({gate.describe()})")
        return reasons, latency

    def evaluate(self):
        if self.worker:
            self.environment.process_exit_code = int(bool(self.errors))
            return
        reports = list(self.snapshots.values()) if self.master else [self.snapshot()]
        histograms: defaultdict[str, Counter] = defaultdict(Counter)
        for report in reports:
            for name, histogram in report["histograms"].items():
                histograms[name].update({int(key): count for key, count in histogram.items()})
        requests = sum(report["requests"] for report in reports)
        failures = sum(report["failures"] for report in reports)
        whole_failures = sum(report["whole_failures"] for report in reports)
        steady = (
            self.measure_end - self.measure_start
            if self.measure_end is not None and self.measure_start is not None
            else 0
        )
        ratio = failures / requests if requests else 1
        overall = Counter()
        for histogram in histograms.values():
            overall.update(histogram)
        p95, p99 = _percentiles(overall)
        reasons = []
        if not self.prepared or not self.started:
            reasons.append("run never started after preflight")
        if steady < self.minimum_steady:
            reasons.append(f"steady duration {steady:.2f}s < {self.minimum_steady}s")
        if not requests:
            reasons.append("no measured requests")
        if ratio > self.maximum_failure_ratio:
            reasons.append(f"failure ratio {ratio:.5f} > {self.maximum_failure_ratio}")
        gate_reasons, latency = self.gate_reasons(histograms)
        reasons.extend(gate_reasons)
        if self.zero_failures and whole_failures:
            reasons.append(f"whole-run failures={whole_failures}, including warmup/shutdown")
        if self.runner.exceptions:
            reasons.append("unhandled user/runner exceptions")
        self.errors.extend(reasons)
        self.environment.process_exit_code = int(bool(self.errors))
        logger.info(
            "CAPACITY %s target=%s minimum_active=%s steady=%.2fs requests=%s "
            "failures=%s ratio=%.5f whole_run_failures=%s p95=%sms p99=%sms "
            "generator_cpu_warning=%s reasons=%s gates=%s",
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
            latency,
        )


_run: CapacityRun | None = None


def install(resource_column: str, gates: tuple[LatencyGate, ...]) -> None:
    """Create the run at Locust init with this workload's fixture column and gates."""

    @events.init.add_listener
    def initialize(environment, **kwargs):
        global _run
        try:
            environment.capacity = CapacityRun(environment, resource_column, gates)
        except Exception as error:
            logger.error("Capacity configuration rejected: %s", error)
            environment.process_exit_code = 1
            gevent.spawn(environment.runner.quit)
        # The request hooks receive no Environment; they reach the run through `_run`.
        _run = getattr(environment, "capacity", None)


@events.test_start.add_listener
def start(environment, **kwargs):
    if hasattr(environment, "capacity"):
        environment.capacity.start()


@events.test_stopping.add_listener
def stopping(environment, **kwargs):
    if hasattr(environment, "capacity"):
        environment.capacity.stopping_test()


@events.request.add_listener
def request(name, response_time, exception, **kwargs):
    if _run is not None:
        _run.request(name, response_time, exception)


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
