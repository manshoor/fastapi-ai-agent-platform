"""Behavioral regressions for the bounded runner; no Docker, paid calls, or live exports."""

import asyncio
import heapq
import importlib.util
import json
import stat
import sys
from contextlib import aclosing
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs
from uuid import uuid4

import httpx
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ai_load_smoke.py"
_SPEC = importlib.util.spec_from_file_location("ai_load_smoke", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
load = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = load
_SPEC.loader.exec_module(load)


@pytest.mark.parametrize(
    ("provider", "paid", "maximum", "environment"),
    [
        ("live", False, 5, {"AI_OPENAI_API_KEY": "approved-process-fixture"}),
        ("live", True, 5, {}),
        ("live", True, 6, {"AI_OPENAI_API_KEY": "approved-process-fixture"}),
        ("simulator", False, 6, {}),
        ("simulator", False, 0, {}),
        ("simulator", False, True, {}),
        ("simulator", True, 5, {}),
    ],
)
def test_rejects_unauthorized_budget_before_any_side_effect(
    monkeypatch, provider, paid, maximum, environment
):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid authorization reached socket, filesystem, or Docker")

    monkeypatch.setattr(load.socket, "socket", forbidden)
    monkeypatch.setattr(load, "new_run", forbidden)
    monkeypatch.setattr(load.subprocess, "run", forbidden)
    with pytest.raises(ValueError):
        load.run_stack(
            provider=provider,
            allow_paid_requests=paid,
            max_ai_jobs=maximum,
            environment=environment,
        )


def test_child_credentials_do_not_inherit_dotenv_or_export_configuration():
    source = {
        "PATH": "/usr/bin",
        "AI_OPENAI_API_KEY": "private-process-key",
        "OPENAI_BASE_URL": "https://unapproved.example",
        "COMPOSE_FILE": "compose.yaml",
        "LANGSMITH_API_KEY": "not-for-this-run",
        "DATABASE_URL": "normal-db",
    }
    first = load.generate_environment("live", source)
    second = load.generate_environment("simulator", source)
    assert first["AI_LOAD_PROVIDER_KEY"] == source["AI_OPENAI_API_KEY"]
    assert second["AI_LOAD_PROVIDER_KEY"] != source["AI_OPENAI_API_KEY"]
    assert first["AI_LOAD_DB_PASSWORD"] != second["AI_LOAD_DB_PASSWORD"]
    assert first["AI_LOAD_JWT_SECRET"] != second["AI_LOAD_JWT_SECRET"]
    assert not (
        {"LANGSMITH_API_KEY", "DATABASE_URL", "COMPOSE_FILE", "OPENAI_BASE_URL"} & first.keys()
    )
    assert first["COMPOSE_DISABLE_ENV_FILE"] == "1"


def test_fresh_destination_cannot_be_reused_and_credentials_are_not_persisted(
    tmp_path, monkeypatch
):
    root = tmp_path / "private"
    monkeypatch.setattr(load.secrets, "token_hex", lambda count: "a" * (count * 2))

    class FrozenDate:
        @staticmethod
        def now(zone):
            return datetime(2026, 11, 1, 12, tzinfo=UTC)

    monkeypatch.setattr(load, "datetime", FrozenDate)
    project, output = load.new_run(root)
    assert load.PROJECT_PATTERN.fullmatch(project)
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert stat.S_IMODE((output / "run.json").stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        load.new_run(root)
    assert list(output.iterdir()) == [output / "run.json"]


@pytest.mark.parametrize("kind", ["symlink", "public"])
def test_refuses_unsafe_artifact_parent(tmp_path, kind):
    actual = tmp_path / "actual"
    actual.mkdir(mode=0o700)
    destination = tmp_path / "artifacts"
    if kind == "symlink":
        destination.symlink_to(actual, target_is_directory=True)
    else:
        destination.mkdir(mode=0o755)
        destination.chmod(0o755)
    with pytest.raises(ValueError):
        load.new_run(destination)
    assert not list(actual.iterdir())


def test_unknown_preflight_reservation_consumes_one_of_five_forever(tmp_path):
    ledger = load.Ledger(tmp_path, 5)
    first = ledger.reserve()
    assert first["status"] == "unknown"
    for _ in range(4):
        ledger.reserve()
    with pytest.raises(load.SmokeError):
        ledger.reserve()
    persisted = [
        json.loads(line) for line in (tmp_path / "admissions.jsonl").read_text().splitlines()
    ]
    assert [row["slot"] for row in persisted] == [1, 2, 3, 4, 5]
    assert all(row["event"] == "reserved" and row["status"] == "unknown" for row in persisted)
    assert stat.S_IMODE((tmp_path / "admissions.jsonl").stat().st_mode) == 0o600


class SyntheticHTTP:
    """A stateful remote-consumer fixture: owner checks, persisted items, asynchronous jobs."""

    def __init__(self, *, outcome="succeeded", ambiguous_submit=False):
        self.outcome = outcome
        self.ambiguous_submit = ambiguous_submit
        self.users = {}
        self.items = {}
        self.jobs = {}
        self.submissions = 0
        self.patches = []
        self.calls = []

    def handle(self, request):
        path = request.url.path
        self.calls.append((request.method, path))
        owner = request.headers.get("Authorization")

        def response(code, value=None):
            return httpx.Response(code, json=value) if value is not None else httpx.Response(code)

        if path == "/health/ready":
            return response(200, {"status": "ok"})
        if path == "/api/v1/auth/register":
            body = json.loads(request.content)
            self.users[body["email"]] = f"Bearer fixture-{uuid4().hex}"
            return response(201, {"id": str(uuid4()), "email": body["email"]})
        if path == "/api/v1/auth/login":
            form = parse_qs(request.content.decode())
            return response(200, {"access_token": self.users[form["username"][0]].split()[1]})
        if path == "/api/v1/items" and request.method == "POST":
            item = {
                **json.loads(request.content),
                "id": str(uuid4()),
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            }
            self.items[item["id"]] = (owner, item)
            return response(201, item)
        if path == "/api/v1/items":
            return response(
                200,
                {
                    "items": [
                        deepcopy(item)
                        for identity, item in self.items.values()
                        if identity == owner
                    ],
                    "next_cursor": None,
                },
            )
        if path.startswith("/api/v1/ai/jobs/"):
            job = self.jobs[path.rsplit("/", 1)[1]]
            if job["owner"] != owner:
                return response(404)
            return response(200, {k: v for k, v in job.items() if k != "owner"})
        if path.startswith("/api/v1/items/"):
            identifier = path.split("/")[4]
            if identifier not in self.items or self.items[identifier][0] != owner:
                return response(404)
            item = self.items[identifier][1]
            if path.endswith("/ai/rewrite"):
                self.submissions += 1
                identifier = str(uuid4())
                job = {
                    "id": identifier,
                    "owner": owner,
                    "status": self.outcome,
                    "status_url": f"/api/v1/ai/jobs/{identifier}",
                    "source_item_updated_at": item["updated_at"],
                    "error_code": None
                    if self.outcome == "succeeded"
                    else "provider_authentication",
                    "suggestion": {
                        "title": "Clear synthetic title",
                        "description": "A clear synthetic description.",
                    }
                    if self.outcome == "succeeded"
                    else None,
                }
                self.jobs[identifier] = job
                if self.ambiguous_submit:
                    raise httpx.ReadTimeout("Uncertain remote acceptance", request=request)
                # Real submit can finish before HTTP returns but still returns fresh 202.
                return response(202, job)
            if request.method == "PATCH":
                payload = json.loads(request.content)
                self.patches.append(item["id"])
                item.update(payload, updated_at="2026-01-01T00:00:01Z")
            return response(200, deepcopy(item))
        raise AssertionError("Unexpected request in synthetic consumer fixture")

    def metadata(self):
        return {
            "users": len(self.users),
            "items": len(self.items),
            "jobs": [
                {
                    "id": job["id"],
                    "status": job["status"],
                    "attempt": 1,
                    "provider_started_at": "2026-01-01T00:00:00+00:00",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "finished_at": "2026-01-01T00:00:01+00:00",
                    "model": "requested-model",
                    "response_model": "actual-model",
                    "input_tokens": 25,
                    "output_tokens": 15,
                    "error_code": job["error_code"],
                }
                for job in self.jobs.values()
            ],
            "queue_depth": 0,
            "oldest_queue_age_seconds": 0,
        }


async def seeded_client(remote):
    client = httpx.AsyncClient(base_url=load.BASE_URL, transport=httpx.MockTransport(remote.handle))
    accounts = await load.seed(client)
    return client, accounts


async def test_five_rewrites_include_preflight_and_never_touch_crud_patch_targets(tmp_path):
    remote = SyntheticHTTP()
    client, accounts = await seeded_client(remote)
    ledger = load.Ledger(tmp_path, 5)
    metrics = load.Measurements()
    abort = asyncio.Event()
    async with aclosing(client):
        for index in range(5):
            assert await load.rewrite(
                client,
                accounts[index],
                accounts[index + 1],
                ledger,
                metrics,
                abort,
                load.time.monotonic() + 30,
            )
    assert remote.submissions == 5
    assert len(ledger.entries) == 5
    assert all(entry["verified"] for entry in ledger.entries)
    assert set(remote.patches) == {account.rewrite["id"] for account in accounts[:5]}
    assert not set(remote.patches) & {item["id"] for account in accounts for item in account.items}
    with pytest.raises(load.SmokeError):
        ledger.reserve()


@pytest.mark.parametrize("ambiguous", [False, True])
async def test_preflight_failure_or_unknown_acceptance_aborts_remaining_paid_admissions(
    tmp_path, monkeypatch, ambiguous
):
    remote = SyntheticHTTP(outcome="failed", ambiguous_submit=ambiguous)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        load.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(**kwargs, transport=httpx.MockTransport(remote.handle)),
    )
    commands = []

    class OwnedStack:
        output = tmp_path
        metadata = remote.metadata

        def command(self, args, **kwargs):
            commands.append(args)

        def simulator_state(self):
            return {"calls": remote.submissions}

        def provider_telemetry(self):
            return {
                "generation_spans_observed": remote.submissions,
                "receiver_dropped": 0,
            }

    report = await load.exercise(OwnedStack(), "simulator", 5)
    assert remote.submissions == 1
    assert not report["preflight_succeeded"]
    assert not report["passed"]
    assert "duration_seconds" not in report  # No measurement began after failed preflight.
    assert report["ai"]["admissions_reserved"] == 1
    assert report["ai"]["observed_status_counts"] == (
        {"unknown": 1} if ambiguous else {"failed": 1}
    )
    assert commands == [["stop", "-t", "0", "worker"]]


async def test_no_autoapply_assertion_rejects_changed_source_before_explicit_patch(tmp_path):
    remote = SyntheticHTTP()
    client, accounts = await seeded_client(remote)
    owner = accounts[0]
    remote.items[owner.rewrite["id"]][1]["description"] = "Unexpected automatic mutation"
    async with aclosing(client):
        result = await load.rewrite(
            client,
            owner,
            accounts[1],
            load.Ledger(tmp_path, 5),
            load.Measurements(),
            asyncio.Event(),
            load.time.monotonic() + 30,
        )
    assert not result
    assert remote.submissions == 1
    assert not remote.patches


@pytest.mark.parametrize(
    "invalid",
    [
        {"title": "Invalid\0title", "description": "Text"},
        {"title": "Valid", "description": "Invalid\0text"},
        {"title": "Valid", "description": "x" * 5001},
        {"title": None, "description": "Text"},
        {"title": "Valid", "description": "Text", "owner_id": "untrusted"},
    ],
)
async def test_result_boundaries_preserve_unicode_empty_descriptions_and_reject_bad_writes(invalid):
    remote = SyntheticHTTP()
    client, accounts = await seeded_client(remote)
    owner, other = accounts[:2]
    untouched = deepcopy(remote.items[other.rewrite["id"]])
    async with aclosing(client):
        await load.verify_suggestion(
            client,
            owner,
            {
                "suggestion": {"title": "Résumé 更新", "description": ""},
                "source_item_updated_at": owner.rewrite["updated_at"],
            },
        )
        assert remote.items[owner.rewrite["id"]][1]["description"] == ""
        assert remote.items[owner.rewrite["id"]][1]["title"] == "Résumé 更新"
        with pytest.raises(load.SmokeError):
            await load.verify_suggestion(
                client,
                other,
                {
                    "suggestion": invalid,
                    "source_item_updated_at": other.rewrite["updated_at"],
                },
            )
    assert remote.items[other.rewrite["id"]] == untouched
    assert remote.patches == [owner.rewrite["id"]]


class VirtualClock:
    def __init__(self):
        self.value = 0.0
        self.sequence = 0
        self.waiters = []

    def now(self):
        return self.value

    async def sleep(self, delay):
        future = asyncio.get_running_loop().create_future()
        self.sequence += 1
        heapq.heappush(self.waiters, (self.value + delay, self.sequence, future))
        await future

    def advance(self):
        while self.waiters and self.waiters[0][2].done():
            heapq.heappop(self.waiters)
        assert self.waiters, "Virtual run stalled without a timer"
        self.value = self.waiters[0][0]
        while self.waiters and self.waiters[0][0] <= self.value:
            _, _, future = heapq.heappop(self.waiters)
            if not future.done():
                future.set_result(None)


async def test_observation_is_exactly_90_seconds_with_20_accounts_and_no_drain(tmp_path):
    remote = SyntheticHTTP()
    client, accounts = await seeded_client(remote)
    clock = VirtualClock()
    metrics = load.Measurements()
    async with aclosing(client):
        task = asyncio.create_task(
            load.run_window(
                client,
                accounts,
                load.Ledger(tmp_path, 1),
                metrics,
                asyncio.Event(),
                now=clock.now,
                sleep=clock.sleep,
            )
        )
        for _ in range(2000):
            for _ in range(4):
                await asyncio.sleep(0)
            if task.done():
                break
            clock.advance()
        report = await task
    assert clock.value == 90
    assert report["observed_seconds"] == 90
    assert report["users_started"] == 20
    assert report["full_user_steady_seconds"] == 60
    assert sorted(metrics.user_starts.values()) == pytest.approx([i / 5 for i in range(20)])
    assert all(row["started_seconds"] < 90 for row in metrics.requests)
    assert report["crud"]["errors"] == 0
    assert report["crud_steady"]["requests"] > 0
    assert not remote.submissions
    assert {"detail", "list", "patch"} == report["crud"]["by_operation"].keys()


def test_crud_percentiles_are_not_poll_or_generation_latency():
    metrics = load.Measurements()
    metrics.user_starts = {index: index / 5 for index in range(20)}
    metrics.requests = [
        {"operation": "detail", "latency_ms": value, "started_seconds": timestamp, "error": None}
        for value, timestamp in [(1, 0), (2, 29), (10, 30), (20, 50), (30, 89)]
    ]
    metrics.job_latencies = [29000, 28000]
    report = metrics.summary()
    assert report["crud"]["p50_ms"] == 10
    assert report["crud"]["p95_ms"] == 30
    assert report["crud_steady"]["requests"] == 3
    assert report["crud_steady"]["p50_ms"] == 20
    assert report["job_end_to_end_ms"] == [29000, 28000]


def good_topology(project):
    base = {
        "DATABASE_URL": "",
        "DB_HOST": "db",
        "DB_PORT": "5432",
        "DB_NAME": "app_ai_load",
        "DB_USER": "app_ai_load",
        "DB_PASSWORD": "fresh-synthetic-password",
        "DB_POOL_SIZE": "4",
    }
    processing = {
        **base,
        "AI_MAX_ATTEMPTS": "1",
        "AI_CONCURRENCY": "2",
        "AI_MAX_OUTPUT_TOKENS": "512",
        "AI_PROVIDER_TIMEOUT_SECONDS": "30",
        "KAFKA_BOOTSTRAP_SERVERS": "kafka:9092",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4318",
    }
    services = {
        name: {"networks": {"default": None}}
        for name in (
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
            "provider",
        )
    }
    services["db"]["environment"] = {"POSTGRES_PASSWORD": base["DB_PASSWORD"]}
    services["db"]["volumes"] = [
        {"type": "volume", "source": "load_postgres_data", "target": "/var/lib/postgresql"}
    ]
    services["kafka"]["volumes"] = [
        {"type": "volume", "source": "load_kafka_data", "target": "/var/lib/kafka/data"}
    ]
    services["collector"]["volumes"] = [
        {
            "type": "bind",
            "source": str(load.ROOT / "observability/collector.yaml"),
            "target": "/etc/otelcol-contrib/config.yaml",
            "read_only": True,
        }
    ]
    for name in ("migrate", "api", "dispatcher", "worker"):
        services[name]["environment"] = deepcopy(base if name == "migrate" else processing)
    services["api"]["environment"].update(DB_POOL_SIZE="10", REDIS_URL="redis://redis:6379/2")
    services["dispatcher"]["environment"]["DB_POOL_SIZE"] = "2"
    services["worker"]["environment"].update(
        OPENAI_API_KEY="synthetic-only", OPENAI_BASE_URL="http://provider:8080/v1"
    )
    services["api"]["networks"]["api_ingress"] = None
    services["worker"]["networks"]["provider_egress"] = None
    services["api"]["ports"] = [{"host_ip": "127.0.0.1", "published": "8004", "target": 8000}]
    return {
        "name": project,
        "services": services,
        "volumes": {
            name: {"name": f"{project}_{name}"}
            for name in ("load_postgres_data", "load_kafka_data")
        },
        "networks": {
            name: {"name": f"{project}_{name}", "internal": name != "api_ingress"}
            for name in ("default", "api_ingress", "provider_egress")
        },
    }


@pytest.mark.parametrize(
    ("service", "key", "value"),
    [
        ("api", "DATABASE_URL", "postgresql://normal-db/app"),
        ("worker", "DB_NAME", "app"),
        ("api", "REDIS_URL", "redis://redis:6379/0"),
        ("worker", "AI_MAX_ATTEMPTS", "2"),
        ("worker", "AI_MAX_OUTPUT_TOKENS", "513"),
        ("worker", "OPENAI_BASE_URL", "https://api.openai.com/v1"),
    ],
)
def test_refuses_unsafe_destinations_or_budget_before_start(service, key, value):
    project = "fastapi-ai-load-20261101t120000z-aaaaaaaaaaaa"
    config = good_topology(project)
    load.validate_topology(config, project, "simulator")
    config["services"][service]["environment"][key] = value
    with pytest.raises(ValueError):
        load.validate_topology(config, project, "simulator")


def test_existing_docker_project_is_never_started_or_stopped(tmp_path):
    stack = load.Stack("fastapi-ai-load-20261101t120000z-aaaaaaaaaaaa", tmp_path, "simulator", {})
    stack.command = lambda *args, **kwargs: "existing-container-id"
    with pytest.raises(ValueError):
        stack.ensure_fresh()


@pytest.mark.parametrize("stop_fails", [False, True])
def test_setup_failure_always_attempts_only_owned_cleanup_and_retains_private_report(
    tmp_path, monkeypatch, stop_fails
):
    project = "fastapi-ai-load-20261101t120000z-aaaaaaaaaaaa"
    commands = []
    output = tmp_path / project
    output.mkdir(mode=0o700)
    monkeypatch.setattr(load, "new_run", lambda root: (project, output))

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def bind(self, address):
            assert address == ("127.0.0.1", 8004)

    monkeypatch.setattr(load.socket, "socket", Socket)
    monkeypatch.setattr(load.Stack, "ensure_fresh", lambda self: None)

    def command(self, args, **kwargs):
        commands.append(args)
        if args == ["config", "--format", "json"]:
            return json.dumps(good_topology(project))
        if args[0] == "up" or stop_fails:
            raise load.SmokeError("failed")
        return ""

    monkeypatch.setattr(load.Stack, "command", command)
    with pytest.raises(load.SmokeError):
        load.run_stack(provider="simulator", environment={}, artifacts_root=tmp_path)
    assert commands[-1] == ["stop", "-t", "0"]
    assert all("down" not in args and "--volumes" not in args for args in commands)
    report = json.loads((output / "report.json").read_text())
    assert not report["passed"]
    assert report["cleanup"] == (
        "stop_failed_manual_owned_project_stop_required"
        if stop_fails
        else "owned_containers_stopped_volumes_preserved"
    )
    assert stat.S_IMODE((output / "report.json").stat().st_mode) == 0o600


async def test_inflight_http_is_reported_unfinished_not_drained_past_90_seconds(tmp_path):
    remote = SyntheticHTTP()
    seed_client, accounts = await seeded_client(remote)
    await seed_client.aclose()
    clock = VirtualClock()

    async def stalled_request(request):
        await clock.sleep(100)
        return remote.handle(request)

    metrics = load.Measurements()
    async with httpx.AsyncClient(
        base_url=load.BASE_URL, transport=httpx.MockTransport(stalled_request)
    ) as client:
        task = asyncio.create_task(
            load.run_window(
                client,
                accounts,
                load.Ledger(tmp_path, 1),
                metrics,
                asyncio.Event(),
                now=clock.now,
                sleep=clock.sleep,
            )
        )
        for _ in range(100):
            for _ in range(4):
                await asyncio.sleep(0)
            if task.done():
                break
            clock.advance()
        report = await task
    assert clock.value == 90
    assert report["unfinished_requests"] == 20
    assert not report["crud"]["requests"]
    assert report["observed_seconds"] == 90


def test_local_provider_telemetry_counts_unique_generations_and_actual_usage(tmp_path):
    stack = load.Stack("fastapi-ai-load-20261101t120000z-aaaaaaaaaaaa", tmp_path, "simulator", {})
    first = {
        "span_id": "01",
        "name": "ai.generate",
        "attributes": {
            "gen_ai.response.model": "actual-model",
            "gen_ai.usage.input_tokens": 29,
            "gen_ai.usage.output_tokens": 19,
        },
    }
    second = {
        "span_id": "02",
        "name": "ai.generate",
        "attributes": {
            "gen_ai.response.model": "actual-model",
            "gen_ai.usage.input_tokens": 33,
            "gen_ai.usage.output_tokens": 21,
        },
    }
    state = {
        "spans": [first, deepcopy(first), second, {"span_id": "03", "name": "ai.poll"}],
        "dropped": 0,
    }
    stack.command = lambda *args, **kwargs: json.dumps(state)
    observed = stack.provider_telemetry()
    assert observed["generation_spans_observed"] == 2
    assert observed["models"] == {"actual-model": 2}
    assert observed["input_tokens"] == 62
    assert observed["output_tokens"] == 40


def test_errors_retain_http_status_and_their_own_latency_bucket():
    response = httpx.Response(503, request=httpx.Request("GET", load.BASE_URL + "/api/v1/items"))
    with pytest.raises(load.UnexpectedStatus) as failure:
        load.expect(response, 200)
    assert failure.value.status == 503
    metrics = load.Measurements(
        requests=[
            {"operation": "list", "started_seconds": 40, "latency_ms": 80, "error": "http_503"},
            {"operation": "detail", "started_seconds": 41, "latency_ms": 3, "error": None},
        ]
    )
    report = metrics.summary()
    assert report["crud"]["error_codes"] == {"http_503": 1}
    assert report["crud_errors"]["p50_ms"] == 80
    assert report["crud_errors"]["requests"] == 1


def test_budget_evaluation_rejects_unknowns_retry_attempts_and_excess_actual_tokens(tmp_path):
    ledger = load.Ledger(tmp_path, 1)
    ledger.reserve().update(status="succeeded", verified=True, job_id="known-job")
    bucket = {"requests": 10, "errors": 0, "p95_ms": 12, "p99_ms": 30}
    report = {
        "users_started": 20,
        "full_user_steady_seconds": 60,
        "crud": bucket,
        "crud_steady": bucket,
        "unfinished_requests": 0,
    }
    metadata = {
        "users": 20,
        "items": 120,
        "jobs": [
            {
                "status": "succeeded",
                "attempt": 1,
                "provider_started_at": "2026-01-01T00:00:00+00:00",
                "finished_at": "2026-01-01T00:00:02+00:00",
                "response_model": "actual-model",
                "input_tokens": 30,
                "output_tokens": 20,
            }
        ],
    }
    assert not load.evaluate(report, ledger, metadata)
    metadata["jobs"][0].update(attempt=2, output_tokens=513)
    failures = load.evaluate(report, ledger, metadata)
    assert "provider_budget_invariant" in failures
    assert "missing_or_invalid_model_usage" in failures
    ledger.entries[0].update(status="unknown", verified=False)
    assert "unfinished_failed_or_unknown_ai_jobs" in load.evaluate(report, ledger, metadata)


def test_restarting_even_an_empty_or_ambiguous_ledger_is_refused(tmp_path):
    load.Ledger(tmp_path, 5)
    with pytest.raises(FileExistsError):
        load.Ledger(tmp_path, 5)
    (tmp_path / "admissions.jsonl").write_text('{"event":"reserved","slot":1')
    with pytest.raises(FileExistsError):
        load.Ledger(tmp_path, 5)


@pytest.mark.parametrize("maximum", [0, 6, True, 2.5])
def test_ledger_itself_cannot_expand_approved_provider_budget(tmp_path, maximum):
    with pytest.raises(ValueError):
        load.Ledger(tmp_path, maximum)
    assert not (tmp_path / "admissions.jsonl").exists()
