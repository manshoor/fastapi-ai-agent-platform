"""Run one fresh LOCAL 20-VU/90-second mixed smoke, never production capacity evidence.

Simulator: uv run --locked python scripts/ai_load_smoke.py --provider simulator --max-ai-jobs 5
Live: set AI_OPENAI_API_KEY in the calling process (never source/read root .env), then add
--provider live --allow-paid-requests --max-ai-jobs 5. There is no automatic retry or drain.
All five admissions include preflight. Unknown submissions are spent, never replaced.

API contracts: app/ai/router.py, app/ai/contracts.py; provider request implementation:
app/ai/provider.py. Compose --env-file semantics:
https://docs.docker.com/compose/how-tos/environment-variables/variable-interpolation/
"""

import argparse
import asyncio
import json
import math
import os
import random
import re
import secrets
import socket
import stat
import subprocess
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from app.ai.contracts import RewriteSuggestion

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8004"
DURATION = 90.0
USERS = 20
MAX_JOBS = 5
TERMINAL = {"succeeded", "failed", "needs_reconciliation"}
SOURCE = {"title": "Synthetic source", "description": "Synthetic original description."}
INSTRUCTION = "Make the synthetic wording clearer without adding facts."
PROJECT_PATTERN = re.compile(r"fastapi-ai-load-[0-9]{8}t[0-9]{6}z-[a-f0-9]{12}\Z")


class SmokeError(RuntimeError):
    """Public error with no request body, credentials, or dependency URLs."""


class UnexpectedStatus(SmokeError):
    def __init__(self, status: int):
        self.status = status
        super().__init__(f"Unexpected HTTP status {status}")


def authorize(provider: str, allow_paid_requests: bool, max_ai_jobs: int, env: dict) -> None:
    if provider not in {"live", "simulator"}:
        raise ValueError("Select explicitly --provider simulator or --provider live")
    if (
        isinstance(max_ai_jobs, bool)
        or not isinstance(max_ai_jobs, int)
        or not 1 <= max_ai_jobs <= 5
    ):
        raise ValueError("max-ai-jobs must be 1..5, including preflight")
    if provider == "live" and (not allow_paid_requests or not env.get("AI_OPENAI_API_KEY")):
        raise ValueError("Live requires --allow-paid-requests and process AI_OPENAI_API_KEY")
    if provider == "simulator" and allow_paid_requests:
        raise ValueError("Paid authorization is invalid for simulator mode")


def generate_environment(provider: str, source: dict[str, str]) -> dict[str, str]:
    """Fresh secrets in child memory only; never import dotenv or inherit export settings."""
    child = {
        k: v
        for k, v in source.items()
        if k
        in {
            "PATH",
            "HOME",
            "USER",
            "TMPDIR",
            "DOCKER_HOST",
            "DOCKER_CONTEXT",
            "DOCKER_CONFIG",
            "SSH_AUTH_SOCK",
        }
    }
    child.update(
        {
            "COMPOSE_DISABLE_ENV_FILE": "1",
            "AI_LOAD_DB_PASSWORD": secrets.token_hex(24),
            "AI_LOAD_JWT_SECRET": secrets.token_hex(32),
            "AI_LOAD_MODEL": source.get("AI_LOAD_MODEL", "gpt-5.6-terra"),
            "AI_LOAD_PROVIDER_KEY": source["AI_OPENAI_API_KEY"]
            if provider == "live"
            else "local-simulator-only-not-a-real-key",
            "AI_LOAD_PROVIDER_URL": "https://api.openai.com/v1"
            if provider == "live"
            else "http://provider:8080/v1",
            "AI_LOAD_EGRESS_INTERNAL": "false" if provider == "live" else "true",
        }
    )
    return child


def private_directory(path: Path, *, create: bool = False) -> None:
    if create:
        path.mkdir(mode=0o700, exist_ok=True)
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise ValueError("Artifact directory must be owned, nonsymlink, mode 0700")


def private_json(path: Path, value: Any) -> None:
    with open(path, "x", opener=lambda p, flags: os.open(p, flags, 0o600)) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def new_run(artifacts_root: Path) -> tuple[str, Path]:
    private_directory(artifacts_root, create=True)
    project = (
        "fastapi-ai-load-"
        + datetime.now(UTC).strftime("%Y%m%dt%H%M%Sz")
        + "-"
        + secrets.token_hex(6)
    )
    output = artifacts_root / project
    # Exclusive creation is also a persistent no-replay lock. No resume API exists.
    output.mkdir(mode=0o700, exist_ok=False)
    private_json(
        output / "run.json",
        {"project": project, "api": BASE_URL, "label": "LOCAL smoke, not production capacity"},
    )
    return project, output


def validate_topology(config: dict, project: str, provider: str) -> None:
    if not PROJECT_PATTERN.fullmatch(project) or config.get("name") != project:
        raise ValueError("Expected a fresh owned load project")
    services = config["services"]
    expected = {
        "db",
        "redis",
        "kafka",
        "topic-init",
        "migrate",
        "otlp",
        "collector",
        "collector-ready",
        "api",
        "dispatcher",
        "worker",
    }
    if provider == "simulator":
        expected.add("provider")
    # Compose may retain inactive profile definitions in its rendered model.
    if set(services) - expected - {"provider"} or not expected <= set(services):
        raise ValueError("Unexpected load services")
    for name, entry in config["volumes"].items():
        if (
            name not in {"load_postgres_data", "load_kafka_data"}
            or entry.get("external")
            or (entry.get("name") != f"{project}_{name}")
        ):
            raise ValueError("Load storage must be project-owned")
    if set(config["volumes"]) != {"load_postgres_data", "load_kafka_data"}:
        raise ValueError("Unexpected storage")
    networks = config["networks"]
    if set(networks) != {"default", "api_ingress", "provider_egress"}:
        raise ValueError("Unexpected load networks")
    for name, entry in networks.items():
        if entry.get("external") or entry.get("name") != f"{project}_{name}":
            raise ValueError("Load network must be project-owned")
    if not networks["default"].get("internal") or bool(
        networks["provider_egress"].get("internal")
    ) != (provider == "simulator"):
        raise ValueError("Unexpected load egress")
    for name, service in services.items():
        allowed = {"default"}
        if name == "api":
            allowed.add("api_ingress")
        if name == "worker":
            allowed.add("provider_egress")
        if set(service.get("networks", {})) != allowed or service.get("network_mode"):
            raise ValueError("Unexpected service network")
        if service.get("env_file") or service.get("container_name"):
            raise ValueError("Unowned service configuration")
        if name != "api" and service.get("ports"):
            raise ValueError("Only API may publish a host port")
        if name in {"db", "kafka", "collector"} and len(service.get("volumes", [])) != 1:
            raise ValueError("Required dedicated mount is missing or duplicated")
        for mount in service.get("volumes", []):
            if name in {"db", "kafka"}:
                volume = "load_postgres_data" if name == "db" else "load_kafka_data"
                target = "/var/lib/postgresql" if name == "db" else "/var/lib/kafka/data"
                if (mount["type"], mount["source"], mount["target"]) != ("volume", volume, target):
                    raise ValueError("Unexpected persistent destination")
            elif name != "collector" or (
                mount["type"],
                mount["source"],
                mount["target"],
                mount.get("read_only"),
            ) != (
                "bind",
                str(ROOT / "observability/collector.yaml"),
                "/etc/otelcol-contrib/config.yaml",
                True,
            ):
                raise ValueError("Unexpected bind mount")
    ports = services["api"].get("ports", [])
    if len(ports) != 1 or (
        ports[0].get("host_ip"),
        str(ports[0].get("published")),
        ports[0].get("target"),
    ) != ("127.0.0.1", "8004", 8000):
        raise ValueError("Unexpected API binding")
    for name in ("api", "migrate", "dispatcher", "worker"):
        env = services[name]["environment"]
        for key, value in {
            "DATABASE_URL": "",
            "DB_HOST": "db",
            "DB_PORT": "5432",
            "DB_NAME": "app_ai_load",
            "DB_USER": "app_ai_load",
        }.items():
            if str(env.get(key)) != value:
                raise ValueError("Unexpected database destination")
        if env.get("DB_PASSWORD") != services["db"]["environment"]["POSTGRES_PASSWORD"]:
            raise ValueError("Database credentials disagree")
        if str(env.get("DB_POOL_SIZE")) != {"api": "10", "dispatcher": "2"}.get(name, "4"):
            raise ValueError("Unexpected role pool")
        if name != "migrate":
            for key, value in {
                "AI_MAX_ATTEMPTS": "1",
                "AI_CONCURRENCY": "2",
                "AI_MAX_OUTPUT_TOKENS": "512",
                "AI_PROVIDER_TIMEOUT_SECONDS": "30",
                "KAFKA_BOOTSTRAP_SERVERS": "kafka:9092",
                "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4318",
            }.items():
                if str(env.get(key)) != value:
                    raise ValueError("Unexpected provider bounds or export destination")
    if services["api"]["environment"].get("REDIS_URL") != "redis://redis:6379/2":
        raise ValueError("Unexpected Redis destination")
    worker = services["worker"]["environment"]
    target = "https://api.openai.com/v1" if provider == "live" else "http://provider:8080/v1"
    if worker.get("OPENAI_BASE_URL") != target or not worker.get("OPENAI_API_KEY"):
        raise ValueError("Unexpected provider destination")


class Stack:
    def __init__(self, project: str, output: Path, provider: str, environment: dict):
        self.project, self.output, self.environment = project, output, environment
        self.compose = [
            "docker",
            "compose",
            "--env-file",
            "/dev/null",
            "-p",
            project,
            "-f",
            str(ROOT / "compose.ai.load.yaml"),
            "--profile",
            "load",
        ]
        if provider == "simulator":
            self.compose += ["--profile", "simulator"]

    def command(self, args: list[str], *, compose: bool = True, timeout: float = 180) -> str:
        try:
            result = subprocess.run(
                [*self.compose, *args] if compose else args,
                env=self.environment,
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            raise SmokeError("Docker command failed or exceeded its deadline") from None
        if result.returncode:
            # Neither rendered Compose nor arbitrary stderr may escape: both can contain secrets.
            raise SmokeError("Docker command failed (sensitive output withheld)")
        return result.stdout

    def ensure_fresh(self) -> None:
        label = f"label=com.docker.compose.project={self.project}"
        for args in (
            ["docker", "ps", "-aq", "--filter", label],
            ["docker", "volume", "ls", "-q", "--filter", label],
            ["docker", "network", "ls", "-q", "--filter", label],
            ["docker", "volume", "ls", "-q", "--filter", f"name={self.project}_"],
            ["docker", "network", "ls", "-q", "--filter", f"name={self.project}_"],
        ):
            if self.command(args, compose=False).strip():
                raise ValueError("Refusing reuse of an existing Docker project")

    def metadata(self) -> dict:
        sql = """SELECT json_build_object(
          'users', (SELECT count(*) FROM users),
          'items', (SELECT count(*) FROM items),
          'jobs', COALESCE((SELECT json_agg(json_build_object(
            'id', id, 'status', status, 'attempt', attempt,
            'provider_started_at', provider_started_at, 'created_at', created_at,
            'finished_at', finished_at, 'model', model, 'response_model', response_model,
            'input_tokens', input_tokens, 'output_tokens', output_tokens,
            'error_code', error_code)) FROM ai_jobs), '[]'::json),
          'queue_depth', (SELECT count(*) FROM ai_jobs
            WHERE status IN ('pending','leased','calling')),
          'oldest_queue_age_seconds', (SELECT COALESCE(EXTRACT(EPOCH FROM now()-min(created_at)),0)
            FROM ai_jobs WHERE status IN ('pending','leased','calling')));"""
        return json.loads(
            self.command(
                [
                    "exec",
                    "-T",
                    "db",
                    "psql",
                    "-U",
                    "app_ai_load",
                    "-d",
                    "app_ai_load",
                    "-At",
                    "-c",
                    sql,
                ],
                timeout=10,
            )
        )

    def resource_sample(self) -> dict:
        ids = self.command(["ps", "-q"], timeout=5).split()
        if not ids:
            raise SmokeError("No load containers available for resources")
        text = self.command(
            ["docker", "stats", "--no-stream", "--format", "{{json .}}", *ids],
            compose=False,
            timeout=8,
        )
        return {
            "at": datetime.now(UTC).isoformat(),
            "containers": [json.loads(line) for line in text.splitlines() if line.strip()],
        }

    def simulator_control(self) -> None:
        self.command(
            [
                "exec",
                "-T",
                "provider",
                "python",
                "-c",
                "import json,urllib.request; "
                "r=urllib.request.Request('http://127.0.0.1:8080/__control',"
                "data=json.dumps({'max_calls':5}).encode(),"
                "headers={'Content-Type':'application/json'}); "
                "assert urllib.request.urlopen(r,timeout=2).status==200",
            ]
        )

    def simulator_state(self) -> dict:
        return json.loads(
            self.command(
                [
                    "exec",
                    "-T",
                    "provider",
                    "python",
                    "-c",
                    "import urllib.request; "
                    "print(urllib.request.urlopen('http://127.0.0.1:8080/__state',"
                    "timeout=2).read().decode())",
                ]
            )
        )

    def provider_telemetry(self) -> dict:
        state = json.loads(
            self.command(
                [
                    "exec",
                    "-T",
                    "otlp",
                    "python",
                    "-c",
                    "import urllib.request; "
                    "print(urllib.request.urlopen("
                    "'http://127.0.0.1:4318/__state?channel=langfuse&service=item-rewrite-worker',"
                    "timeout=2).read().decode())",
                ]
            )
        )
        generations = {
            span["span_id"]: span for span in state["spans"] if span["name"] == "ai.generate"
        }
        return {
            "generation_spans_observed": len(generations),
            "receiver_dropped": state["dropped"],
            "models": dict(
                Counter(
                    span["attributes"].get("gen_ai.response.model", "unknown")
                    for span in generations.values()
                )
            ),
            "input_tokens": sum(
                span["attributes"].get("gen_ai.usage.input_tokens", 0)
                for span in generations.values()
            ),
            "output_tokens": sum(
                span["attributes"].get("gen_ai.usage.output_tokens", 0)
                for span in generations.values()
            ),
        }


@dataclass
class Ledger:
    output: Path
    maximum: int
    entries: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if (
            isinstance(self.maximum, bool)
            or not isinstance(self.maximum, int)
            or not 1 <= self.maximum <= MAX_JOBS
        ):
            raise ValueError("Admission budget must be 1..5 including preflight")
        # Refuse stale/partial ledgers too: a restart must never replay ambiguous submissions.
        with open(
            self.output / "admissions.jsonl", "x", opener=lambda p, flags: os.open(p, flags, 0o600)
        ):
            pass

    def record(self, value: dict) -> None:
        with open(
            self.output / "admissions.jsonl", "a", opener=lambda p, flags: os.open(p, flags, 0o600)
        ) as handle:
            handle.write(json.dumps(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def reserve(self) -> dict:
        if len(self.entries) >= self.maximum:
            raise SmokeError("Five-job total admission budget exhausted")
        entry = {"slot": len(self.entries) + 1, "status": "unknown", "job_id": None}
        # Persist before HTTP. A timeout/crash spends the slot; no replay/new-key recovery.
        self.record({"event": "reserved", **entry})
        self.entries.append(entry)
        return entry


@dataclass
class Account:
    headers: dict = field(repr=False)
    items: list[dict]
    rewrite: dict


@dataclass
class Measurements:
    requests: list[dict] = field(default_factory=list)
    user_starts: dict[int, float] = field(default_factory=dict)
    job_latencies: list[float] = field(default_factory=list)
    unfinished_requests: int = 0

    def summary(self) -> dict:
        def bucket(rows: list[dict]) -> dict:
            values = sorted(row["latency_ms"] for row in rows)

            def percentile(q: float) -> float | None:
                return values[max(0, math.ceil(len(values) * q) - 1)] if values else None

            return {
                "requests": len(rows),
                "errors": sum(bool(r["error"]) for r in rows),
                "p50_ms": percentile(0.5),
                "p95_ms": percentile(0.95),
                "p99_ms": percentile(0.99),
                "by_operation": dict(Counter(row["operation"] for row in rows)),
                "error_codes": dict(Counter(row["error"] for row in rows if row["error"])),
            }

        steady = [row for row in self.requests if row["started_seconds"] >= 30]
        return {
            "crud": bucket(self.requests),
            "crud_steady": bucket(steady),
            "crud_errors": bucket([row for row in self.requests if row["error"]]),
            "users_started": len(self.user_starts),
            "full_user_steady_seconds": max(0, DURATION - max([30.0, *self.user_starts.values()]))
            if len(self.user_starts) == USERS
            else 0,
            "unfinished_requests": self.unfinished_requests,
            "job_end_to_end_ms": self.job_latencies,
            "job_latency_slo": "Separate from CRUD; never apply 500ms to jobs",
        }


def expect(response: httpx.Response, status: int) -> dict:
    if response.status_code != status:
        raise UnexpectedStatus(response.status_code)
    return response.json() if response.content else {}


async def seed(client: httpx.AsyncClient) -> list[Account]:
    accounts = []
    for index in range(USERS):
        email = f"ai-load-{secrets.token_hex(12)}@example.com"
        password = secrets.token_hex(20)
        expect(
            await client.post("/api/v1/auth/register", json={"email": email, "password": password}),
            201,
        )
        token = expect(
            await client.post("/api/v1/auth/login", data={"username": email, "password": password}),
            200,
        )
        headers = {"Authorization": f"Bearer {token['access_token']}"}
        items = []
        for item in range(5):
            items.append(
                expect(
                    await client.post(
                        "/api/v1/items",
                        headers=headers,
                        json={
                            "title": f"Synthetic account {index} item {item}",
                            "description": "Synthetic CRUD fixture only.",
                        },
                    ),
                    201,
                )
            )
        rewrite = expect(await client.post("/api/v1/items", headers=headers, json=SOURCE), 201)
        accounts.append(Account(headers, items, rewrite))
    for index, owner in enumerate(accounts):
        foreign = accounts[(index + 1) % USERS]
        path = f"/api/v1/items/{owner.items[0]['id']}"
        expect(await client.get(path, headers=foreign.headers), 404)
        expect(
            await client.patch(path, headers=foreign.headers, json={"title": "Forbidden change"}),
            404,
        )
        page = expect(await client.get("/api/v1/items", headers=owner.headers), 200)
        if {item["id"] for item in page["items"]} != {
            item["id"] for item in [*owner.items, owner.rewrite]
        }:
            raise SmokeError("Synthetic item count or ownership mismatch")
    return accounts


async def verify_suggestion(client: httpx.AsyncClient, owner: Account, job: dict) -> None:
    try:
        suggestion = RewriteSuggestion.model_validate(job.get("suggestion")).model_dump()
    except ValidationError:
        raise SmokeError("Invalid rewrite result schema") from None
    if job.get("source_item_updated_at") != owner.rewrite["updated_at"]:
        raise SmokeError("Rewrite snapshot timestamp mismatch")
    path = f"/api/v1/items/{owner.rewrite['id']}"
    current = expect(await client.get(path, headers=owner.headers), 200)
    if any(
        current[field] != owner.rewrite[field] for field in ("title", "description", "updated_at")
    ):
        raise SmokeError("Rewrite auto-applied to dedicated source item")
    # Dedicated sources are never touched by CRUD PATCH, eliminating assertion races.
    applied = expect(await client.patch(path, headers=owner.headers, json=suggestion), 200)
    if any(applied[field] != suggestion[field] for field in suggestion):
        raise SmokeError("Explicit suggestion PATCH failed")


async def rewrite(
    client: httpx.AsyncClient,
    owner: Account,
    stranger: Account,
    ledger: Ledger,
    metrics: Measurements,
    abort: asyncio.Event,
    deadline: float,
) -> bool:
    if abort.is_set() or time.monotonic() >= deadline:
        return False
    entry = ledger.reserve()
    start = time.monotonic()
    try:
        async with asyncio.timeout(max(0.001, deadline - start)):
            headers = {**owner.headers, "Idempotency-Key": f"load-{secrets.token_hex(20)}"}
            submitted = expect(
                await client.post(
                    f"/api/v1/items/{owner.rewrite['id']}/ai/rewrite",
                    headers=headers,
                    json={"instruction": INSTRUCTION},
                ),
                202,
            )
            entry.update(job_id=submitted["id"], status="pending")
            path = f"/api/v1/ai/jobs/{submitted['id']}"
            if submitted["status_url"] != path:
                raise SmokeError("Invalid job status location")
            expect(await client.get(path, headers=stranger.headers), 404)
            ledger.record({"event": "accepted", **entry})
            while time.monotonic() < deadline:
                job = expect(await client.get(path, headers=owner.headers), 200)
                entry["status"] = job["status"]
                if job["status"] in TERMINAL:
                    entry["error_code"] = job.get("error_code")
                    metrics.job_latencies.append((time.monotonic() - start) * 1000)
                    if job["status"] != "succeeded":
                        abort.set()
                        return False
                    await verify_suggestion(client, owner, job)
                    entry["verified"] = True
                    return True
                await asyncio.sleep(min(0.2, max(0, deadline - time.monotonic())))
    except asyncio.CancelledError:
        abort.set()
        raise
    except (httpx.HTTPError, TimeoutError, SmokeError, KeyError, ValueError):
        entry["verification_error"] = "request_or_result_inconclusive"
        abort.set()
    finally:
        ledger.record({"event": "observed", **entry})
    abort.set()
    return False


async def run_window(
    client: httpx.AsyncClient,
    accounts: list[Account],
    ledger: Ledger,
    metrics: Measurements,
    abort: asyncio.Event,
    *,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> dict:
    """Hard observation window: cancel, never drain or resubmit, at exactly start+90s."""
    started = now()
    deadline = started + DURATION

    async def user(index: int, account: Account) -> None:
        await sleep(index / 5)
        metrics.user_starts[index] = now() - started
        rng = random.Random(secrets.randbits(64))
        choices = ["detail"] * 7 + ["list"] * 2 + ["patch"]
        while now() < deadline:
            rng.shuffle(choices)
            for operation in choices:
                if now() >= deadline:
                    return
                begin = now()
                error = None
                try:
                    async with asyncio.timeout(max(0.001, deadline - begin)):
                        item = rng.choice(account.items)
                        path = f"/api/v1/items/{item['id']}"
                        if operation == "detail":
                            result = expect(await client.get(path, headers=account.headers), 200)
                            if result["id"] != item["id"]:
                                raise SmokeError("Detail ownership mismatch")
                        elif operation == "list":
                            result = expect(
                                await client.get("/api/v1/items", headers=account.headers), 200
                            )
                            if {x["id"] for x in result["items"]} != {
                                x["id"] for x in [*account.items, account.rewrite]
                            }:
                                raise SmokeError("List ownership/count mismatch")
                        else:
                            result = expect(
                                await client.patch(
                                    path,
                                    headers=account.headers,
                                    json={
                                        "description": f"Synthetic update {secrets.token_hex(4)}"
                                    },
                                ),
                                200,
                            )
                            if result["id"] != item["id"]:
                                raise SmokeError("PATCH ownership mismatch")
                except asyncio.CancelledError:
                    metrics.unfinished_requests += 1
                    raise
                except (httpx.HTTPError, TimeoutError, SmokeError, KeyError, ValueError) as exc:
                    error = (
                        f"http_{exc.status}"
                        if isinstance(exc, UnexpectedStatus)
                        else type(exc).__name__
                    )
                metrics.requests.append(
                    {
                        "operation": operation,
                        "started_seconds": begin - started,
                        "latency_ms": (now() - begin) * 1000,
                        "error": error,
                    }
                )
                await sleep(min(rng.uniform(2, 5), max(0, deadline - now())))

    async def scheduled_job(index: int) -> None:
        await sleep((index // 2) * 10 + (index % 2) * 2)
        await rewrite(
            client,
            accounts[index + 1],
            accounts[(index + 2) % USERS],
            ledger,
            metrics,
            abort,
            min(deadline, time.monotonic() + 40),
        )

    tasks = [asyncio.create_task(user(i, account)) for i, account in enumerate(accounts)]
    tasks += [asyncio.create_task(scheduled_job(i)) for i in range(ledger.maximum - 1)]
    try:
        await sleep(max(0, deadline - now()))
    finally:
        # No HTTP is permitted to extend the measurement. Cancelled jobs remain unfinished.
        for task in tasks:
            task.cancel()
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    failures = [
        type(outcome).__name__
        for outcome in outcomes
        if isinstance(outcome, BaseException) and not isinstance(outcome, asyncio.CancelledError)
    ]
    return {
        "duration_seconds": DURATION,
        "observed_seconds": now() - started,
        "warmup_seconds": 30,
        "minimum_steady_seconds": 45,
        "task_errors": failures,
        **metrics.summary(),
    }


def evaluate(report: dict, ledger: Ledger, metadata: dict) -> list[str]:
    failures = []
    if len(ledger.entries) != ledger.maximum or len(metadata["jobs"]) != ledger.maximum:
        failures.append("not_all_approved_jobs_admitted")
    if any(
        entry.get("status") != "succeeded" or not entry.get("verified") for entry in ledger.entries
    ):
        failures.append("unfinished_failed_or_unknown_ai_jobs")
    attempts = sum(job["provider_started_at"] is not None for job in metadata["jobs"])
    if (
        attempts > ledger.maximum
        or len(metadata["jobs"]) > MAX_JOBS
        or any(job["attempt"] != 1 for job in metadata["jobs"])
    ):
        failures.append("provider_budget_invariant")
    for job in metadata["jobs"]:
        if (
            job["status"] != "succeeded"
            or not job.get("response_model")
            or job.get("input_tokens") is None
            or job.get("output_tokens") is None
            or not 0 <= job["output_tokens"] <= 512
        ):
            failures.append("missing_or_invalid_model_usage")
        if job.get("provider_started_at") and job.get("finished_at"):
            elapsed = (
                datetime.fromisoformat(job["finished_at"])
                - datetime.fromisoformat(job["provider_started_at"])
            ).total_seconds()
            if elapsed > 30:
                failures.append("generation_over_30s")
    if metadata["users"] != USERS or metadata["items"] != USERS * 6:
        failures.append("synthetic_fixture_count")
    if report.get("users_started") != USERS or report.get("full_user_steady_seconds", 0) < 45:
        failures.append("insufficient_20_user_steady_window")
    for name in ("crud", "crud_steady"):
        bucket = report.get(name, {})
        if (
            not bucket.get("requests")
            or bucket.get("errors")
            or bucket.get("p95_ms", math.inf) > 500
            or bucket.get("p99_ms", math.inf) > 1000
        ):
            failures.append(f"{name}_latency_or_errors")
    if report.get("unfinished_requests") or report.get("task_errors"):
        failures.append("unfinished_or_failed_crud_requests")
    return sorted(set(failures))


async def exercise(stack: Stack, provider: str, maximum: int) -> dict:
    ledger = Ledger(stack.output, maximum)
    metrics = Measurements()
    abort = asyncio.Event()
    report: dict = {
        "label": "LOCAL simulator smoke (NO live-provider evidence)"
        if provider == "simulator"
        else "LOCAL live-provider smoke, not production capacity",
        "provider": provider,
        "max_ai_jobs_including_preflight": maximum,
        "output_token_ceiling_per_attempt": 512,
        "maximum_output_tokens": maximum * 512,
        "budget_currency": None,
        "budget_note": "Actual tokens reported; currency not inferred from unknown model pricing",
    }
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=5, trust_env=False) as client:
        expect(await client.get("/health/ready"), 200)
        accounts = await seed(client)
        # Preflight is one real admission and provider attempt, outside 90 seconds but inside five.
        preflight = await rewrite(
            client, accounts[0], accounts[1], ledger, metrics, abort, time.monotonic() + 40
        )
        report["preflight_succeeded"] = preflight
        if preflight:
            report.update(await run_window(client, accounts, ledger, metrics, abort))
        else:
            report["failure"] = "preflight_failed_or_ambiguous; no additional jobs admitted"
    # Stop calling before post-window metadata; never silently drain outstanding provider calls.
    await asyncio.to_thread(stack.command, ["stop", "-t", "0", "worker"], timeout=10)
    metadata = await asyncio.to_thread(stack.metadata)
    private_json(stack.output / "crud_requests.json", metrics.requests)
    report["metadata"] = metadata
    report["provider_telemetry"] = await asyncio.to_thread(stack.provider_telemetry)
    report["job_timing"] = [
        {
            "job_id": job["id"],
            "queue_wait_ms": (
                datetime.fromisoformat(job["provider_started_at"])
                - datetime.fromisoformat(job["created_at"])
            ).total_seconds()
            * 1000
            if job.get("provider_started_at")
            else None,
            "generation_and_persistence_ms": (
                datetime.fromisoformat(job["finished_at"])
                - datetime.fromisoformat(job["provider_started_at"])
            ).total_seconds()
            * 1000
            if job.get("finished_at") and job.get("provider_started_at")
            else None,
        }
        for job in metadata["jobs"]
    ]
    report["ai"] = {
        "admissions_reserved": len(ledger.entries),
        "accepted": sum(e["job_id"] is not None for e in ledger.entries),
        "observed_status_counts": dict(Counter(e["status"] for e in ledger.entries)),
        "verified_completed": sum(bool(e.get("verified")) for e in ledger.entries),
        "provider_attempts_observed": sum(
            j["provider_started_at"] is not None for j in metadata["jobs"]
        ),
        "input_tokens": sum(j.get("input_tokens") or 0 for j in metadata["jobs"]),
        "output_tokens": sum(j.get("output_tokens") or 0 for j in metadata["jobs"]),
        "unknown_usage_jobs": sum(j.get("output_tokens") is None for j in metadata["jobs"]),
        "observations": ledger.entries,
    }
    if provider == "simulator":
        report["simulator"] = await asyncio.to_thread(stack.simulator_state)
    report["failures"] = evaluate(report, ledger, metadata)
    if (
        provider == "simulator"
        and report["simulator"]["calls"] != report["ai"]["provider_attempts_observed"]
    ):
        report["failures"].append("simulator_attempt_counter_mismatch")
    observed = report["provider_telemetry"]["generation_spans_observed"]
    if (
        observed != report["ai"]["provider_attempts_observed"]
        or observed > maximum
        or report["provider_telemetry"]["receiver_dropped"]
    ):
        report["failures"].append("provider_telemetry_incomplete_or_budget_mismatch")
    report["passed"] = not report["failures"]
    return report


async def capture_resources(stack: Stack, stop: asyncio.Event) -> dict:
    path = stack.output / "resources.jsonl"
    summary = {
        "successful_samples": 0,
        "failed_samples": 0,
        "artifact": "resources.jsonl",
        "source": "docker stats CPU/memory/network/block IO",
    }
    with open(path, "x", opener=lambda p, flags: os.open(p, flags, 0o600)) as handle:
        while not stop.is_set():
            try:
                sample = await asyncio.to_thread(stack.resource_sample)
                summary["successful_samples"] += 1
            except (SmokeError, ValueError):
                sample = {
                    "at": datetime.now(UTC).isoformat(),
                    "error": "resource_sample_unavailable",
                }
                summary["failed_samples"] += 1
            handle.write(json.dumps(sample) + "\n")
            handle.flush()
            try:
                await asyncio.wait_for(stop.wait(), timeout=5)
            except TimeoutError:
                pass
    return summary


async def orchestrate(stack: Stack, provider: str, maximum: int) -> dict:
    stop = asyncio.Event()
    resources = asyncio.create_task(capture_resources(stack, stop))
    try:
        report = await exercise(stack, provider, maximum)
    finally:
        stop.set()
        observed = await resources
    report["resources"] = observed
    if not observed["successful_samples"]:
        report["failures"].append("resource_measurement_unavailable")
        report["passed"] = False
    return report


def run_stack(
    *,
    provider: str,
    allow_paid_requests: bool = False,
    max_ai_jobs: int = 5,
    environment: dict[str, str] | None = None,
    artifacts_root: Path | None = None,
) -> dict:
    """Parent integration API; accepts only process-derived env, never dotenv paths.

    Raises sanitized exceptions; retained report.json also records failure/cleanup evidence.
    No caller-supplied project or run directory can resume a previous paid admission.
    """
    source = dict(os.environ if environment is None else environment)
    authorize(
        provider, allow_paid_requests, max_ai_jobs, source
    )  # Before filesystem/Docker/network.
    child = generate_environment(provider, source)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 8004))
    project, output = new_run(artifacts_root or ROOT / "artifacts")
    stack = Stack(project, output, provider, child)
    report: dict = {
        "project": project,
        "artifacts": str(output),
        "provider": provider,
        "passed": False,
        "cleanup": "not_started",
    }
    started = False
    try:
        stack.ensure_fresh()
        config = json.loads(stack.command(["config", "--format", "json"]))
        validate_topology(config, project, provider)
        # Rendered config contains credentials: neither print nor persist it.
        del config
        started = True
        services = ["api", "dispatcher", "worker"]
        if provider == "simulator":
            services.append("provider")
        stack.command(["up", "--build", "-d", "--wait", *services], timeout=600)
        if provider == "simulator":
            stack.simulator_control()
        report.update(asyncio.run(orchestrate(stack, provider, max_ai_jobs)))
    except BaseException as exc:
        report["failure"] = type(exc).__name__  # Exception strings can contain credentials.
        raise
    finally:
        if started:
            try:
                # Stops only this unique project. No down --volumes, deletion, or database reset.
                stack.command(["stop", "-t", "0"], timeout=60)
                report["cleanup"] = "owned_containers_stopped_volumes_preserved"
            except SmokeError:
                report["cleanup"] = "stop_failed_manual_owned_project_stop_required"
                report["passed"] = False
        private_json(output / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=["simulator", "live"])
    parser.add_argument("--allow-paid-requests", action="store_true")
    parser.add_argument("--max-ai-jobs", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_stack(
            provider=args.provider,
            allow_paid_requests=args.allow_paid_requests,
            max_ai_jobs=args.max_ai_jobs,
        )
    except (ValueError, SmokeError, OSError, httpx.HTTPError):
        print("FAIL: smoke rejected or interrupted; sensitive details withheld")
        return 1
    print(
        json.dumps(
            {key: result[key] for key in ("project", "artifacts", "provider", "passed", "cleanup")}
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
