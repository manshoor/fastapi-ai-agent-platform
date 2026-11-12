"""Real local HTTP/provider/Kafka/OTLP scenarios; explicitly opt in via AI_QA=1.

The app fixture finishes destructive cleanup BEFORE children start. Every child is
stopped BEFORE that fixture tears down. Kafka topics/groups are unique per test.
Never run this suite concurrently with the explicit live-HTTP smoke profile.
"""

import asyncio
import json
import os
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from aiokafka import AIOKafkaConsumer, TopicPartition
from sqlalchemy import select

from app.ai.config import AISettings
from app.ai.contracts import RewriteEvent
from app.ai.kafka import close_producer, create_producer
from app.models import AIJob, AIOutbox
from tests.support.e2e import ROOT, eventually, isolated_runtime

pytestmark = pytest.mark.skipif(
    os.environ.get("AI_QA") != "1", reason="requires isolated AI QA stack"
)
TERMINAL = {"succeeded", "failed", "needs_reconciliation"}


@pytest.fixture
async def runtime(app):
    prefix = f"ai.qa.{uuid4().hex}"
    overrides = {
        "KAFKA_REQUEST_TOPIC": f"{prefix}.requests",
        "KAFKA_DEAD_LETTER_TOPIC": f"{prefix}.dead-letter",
        "KAFKA_GROUP_ID": prefix,
    }
    async with isolated_runtime(
        app,
        settings_type=AISettings,
        overrides=overrides,
        topics=[f"{prefix}.requests", f"{prefix}.dead-letter"],
        roles=["dispatcher", "worker"],
    ) as children:
        yield children


async def submit(client, owner, item, *, instruction="Clarify the synthetic wording.", key=None):
    key = key or f"e2e-{uuid4().hex}"
    response = await client.post(
        f"/api/v1/items/{item['id']}/ai/rewrite",
        headers={**owner["headers"], "Idempotency-Key": key},
        json={"instruction": instruction},
    )
    assert response.status_code == 202
    return response.json(), key


async def completed(client, owner, job, runtime):
    async def read():
        runtime.healthy()
        response = await client.get(job["status_url"], headers=owner["headers"])
        assert response.status_code == 200
        result = response.json()
        return result if result["status"] in TERMINAL else None

    return await eventually(read, message="durable rewrite did not become terminal")


async def wait_committed(runtime, metadata):
    # No subscription: querying the real group offset does not join/rebalance the worker group.
    consumer = AIOKafkaConsumer(
        bootstrap_servers=runtime.settings.kafka_bootstrap_servers,
        group_id=runtime.settings.kafka_group_id,
        enable_auto_commit=False,
    )
    await consumer.start()
    try:

        async def acknowledged():
            runtime.healthy()
            value = await consumer.committed(TopicPartition(metadata.topic, metadata.partition))
            return value is not None and value >= metadata.offset + 1

        await eventually(acknowledged, message="duplicate delivery offset was not committed")
    finally:
        await consumer.stop()


async def duplicate(app, runtime, job):
    async with app.state.session_factory() as session:
        outbox = await session.scalar(
            select(AIOutbox)
            .where(AIOutbox.job_id == UUID(job["id"]), AIOutbox.event_kind == "requested")
            .order_by(AIOutbox.attempt)
            .limit(1)
        )
        event = RewriteEvent(
            version=1,
            event_id=outbox.id,
            job_id=outbox.job_id,
            attempt=outbox.attempt,
            created_at=outbox.created_at,
        )
    producer = await create_producer(runtime.settings)
    try:
        for _ in range(2):
            metadata = await producer.send_and_wait(
                runtime.settings.kafka_request_topic,
                value=event.model_dump_json().encode(),
                key=str(event.job_id).encode(),
            )
    finally:
        await close_producer(producer)
    await wait_committed(runtime, metadata)


async def dead_letter(runtime, job):
    consumer = AIOKafkaConsumer(
        runtime.settings.kafka_dead_letter_topic,
        bootstrap_servers=runtime.settings.kafka_bootstrap_servers,
        group_id=f"qa-check-{uuid4().hex}",
        auto_offset_reset="earliest",
        enable_auto_commit=False,
    )
    await consumer.start()
    try:
        async with asyncio.timeout(20):
            while True:
                record = await consumer.getone()
                payload = json.loads(record.value)
                if payload.get("job_id") == job["id"]:
                    assert set(payload) == {
                        "version",
                        "event_id",
                        "job_id",
                        "attempt",
                        "created_at",
                        "status",
                        "error_code",
                    }
                    return payload
    finally:
        await consumer.stop()


async def test_http_durable_success_ownership_no_autoapply_and_duplicates(
    app, client, account, item, runtime
):
    owner, stranger = await account(), await account()
    source = await item(
        owner, title=f"private-title-{uuid4().hex}", description="Synthetic original."
    )
    instruction = f"private-instruction-{uuid4().hex}"
    job, key = await submit(client, owner, source, instruction=instruction)
    path = f"/api/v1/items/{source['id']}"
    assert (await client.get(job["status_url"], headers=stranger["headers"])).status_code == 404
    foreign, _ = await submit_foreign(client, stranger, source)
    assert foreign.status_code == 404
    replay = await client.post(
        f"{path}/ai/rewrite",
        headers={**owner["headers"], "Idempotency-Key": key},
        json={"instruction": instruction},
    )
    assert replay.status_code in {200, 202}
    assert replay.json()["id"] == job["id"]
    result = await completed(client, owner, job, runtime)
    assert result["status"] == "succeeded"
    assert result["source_item_updated_at"] == source["updated_at"]
    assert await runtime.calls() == 1
    current = (await client.get(path, headers=owner["headers"])).json()
    assert all(current[field] == source[field] for field in ("title", "description", "updated_at"))
    await duplicate(app, runtime, job)
    assert await runtime.calls() == 1
    response = await client.patch(path, headers=owner["headers"], json=result["suggestion"])
    assert response.status_code == 200
    assert all(
        response.json()[field] == result["suggestion"][field] for field in result["suggestion"]
    )

    # Query actual Collector-transformed protobuf deliveries on both independent routes.
    async def routed():
        response = await runtime.receiver.get("/__state")
        assert response.status_code == 200
        spans = response.json()["spans"]
        workers = [
            span
            for span in spans
            if span["name"] == "ai.worker.process" and span["attributes"].get("job_id") == job["id"]
        ]
        channels = {span["channel"] for span in workers}
        return spans if channels == {"langfuse", "langsmith"} else None

    spans = await eventually(routed, seconds=15, message="two local OTLP routes not observed")
    serialized = json.dumps(spans)
    private_values = [
        source["title"],
        source["description"],
        instruction,
        key,
        owner["access_token"],
    ]
    if any(value in serialized for value in private_values):
        raise AssertionError("Private business data or credentials escaped into telemetry")
    for channel, kind in (
        ("langfuse", "langfuse.observation.type"),
        ("langsmith", "langsmith.span.kind"),
    ):
        worker = next(
            span
            for span in spans
            if span["channel"] == channel
            and span["name"] == "ai.worker.process"
            and span["attributes"].get("job_id") == job["id"]
        )
        generations = [
            span
            for span in spans
            if span["channel"] == channel
            and span["trace_id"] == worker["trace_id"]
            and span["name"] == "ai.generate"
        ]
        assert len(generations) == 1
        assert generations[0]["parent_span_id"] == worker["span_id"]
        assert kind in generations[0]["attributes"]
        attributes = generations[0]["attributes"]
        if channel == "langfuse":
            assert attributes["gen_ai.request.model"] == runtime.settings.ai_model
            assert attributes["langfuse.observation.type"] == "generation"
        else:
            assert attributes["langsmith.metadata.model"] == runtime.settings.ai_model
            assert attributes["langsmith.metadata.prompt_version"] == "item-rewrite-v1"
        assert not generations[0]["events"]
        assert not generations[0]["status_description"]

    assert await asyncio.to_thread(
        app.state.telemetry.meter_provider.force_flush, timeout_millis=2000
    )

    async def routed_metrics():
        response = await runtime.receiver.get("/__state")
        assert response.status_code == 200
        names = {
            metric["name"]
            for metric in response.json()["metrics"]
            if metric["resource"].get("service.name") == "rewrite-api"
            and metric["channel"] == "direct"
        }
        return {"ai.requests", "ai.jobs"} <= names

    await eventually(
        routed_metrics, seconds=10, message="Collector did not forward real API metrics"
    )


async def submit_foreign(client, stranger, source):
    key = f"foreign-{uuid4().hex}"
    return await client.post(
        f"/api/v1/items/{source['id']}/ai/rewrite",
        headers={**stranger["headers"], "Idempotency-Key": key},
        json={"instruction": "Synthetic foreign request."},
    ), key


@pytest.mark.parametrize(
    ("modes", "status", "calls", "code"),
    [
        (["refusal"], "failed", 1, "provider_refusal"),
        (["invalid"], "failed", 1, "provider_invalid_output"),
        (["incomplete"], "failed", 1, "provider_incomplete"),
        (["rate_limit", "success"], "succeeded", 2, None),
        (["rate_limit"] * 3, "failed", 3, "provider_rate_limit"),
        (["timeout"], "needs_reconciliation", 1, "provider_timeout"),
        (["server_error"], "needs_reconciliation", 1, "provider_unavailable"),
    ],
)
async def test_real_http_provider_outcomes_and_bounded_calls(
    app, client, account, item, runtime, modes, status, calls, code
):
    await runtime.control(modes)
    owner = await account()
    source = await item(owner)
    job, _ = await submit(client, owner, source)
    result = await completed(client, owner, job, runtime)
    assert result["status"] == status
    assert result["error_code"] == code
    assert await runtime.calls() == calls
    if status != "succeeded":
        assert result["suggestion"] is None
        dlq = await dead_letter(runtime, job)
        assert dlq["status"] == status and dlq["error_code"] == code
    await duplicate(app, runtime, job)
    assert await runtime.calls() == calls


async def test_broker_unreachable_admission_survives_then_recovers(
    app, client, account, item, runtime
):
    # Real refused network connection, not a mocked send; avoid stopping the shared broker
    # underneath other QA modules. A replacement dispatcher uses the healthy real broker.
    await runtime.stop("dispatcher")
    runtime.start("dispatcher", KAFKA_BOOTSTRAP_SERVERS="127.0.0.1:1")
    owner = await account()
    source = await item(owner)
    job, _ = await submit(client, owner, source)
    process = runtime.processes["dispatcher"]

    async def refused():
        return process.poll() is not None

    await eventually(refused, seconds=20, message="unreachable broker startup was not bounded")
    assert process.returncode != 0
    async with app.state.session_factory() as session:
        persisted = await session.get(AIJob, UUID(job["id"]))
        outbox = await session.scalar(select(AIOutbox).where(AIOutbox.job_id == persisted.id))
        assert persisted.status == "pending"
        assert outbox.published_at is None
        event_id = outbox.id
    assert await runtime.calls() == 0
    await runtime.stop("dispatcher")
    runtime.start("dispatcher")
    result = await completed(client, owner, job, runtime)
    assert result["status"] == "succeeded"
    async with app.state.session_factory() as session:
        assert (await session.get(AIOutbox, event_id)).published_at is not None
    assert await runtime.calls() == 1


async def test_worker_crash_after_send_is_reconciled_without_replay(
    app, client, account, item, runtime
):
    await runtime.control(["timeout"], delay_seconds=3)
    owner = await account()
    source = await item(owner)
    job, _ = await submit(client, owner, source)

    async def entered_provider():
        response = await runtime.simulator.get("/__state")
        data = response.json()
        return data["calls"] == 1 and data["in_flight"] == 1

    await eventually(entered_provider, message="worker never entered provider HTTP call")
    await runtime.stop("worker", crash=True)
    async with app.state.session_factory() as session:
        persisted = await session.get(AIJob, UUID(job["id"]))
        assert persisted.status == "calling"
        assert persisted.provider_started_at is not None
    runtime.start("worker")
    result = await completed(client, owner, job, runtime)
    assert result["status"] == "needs_reconciliation"
    dlq = await dead_letter(runtime, job)
    assert dlq["status"] == "needs_reconciliation"
    await duplicate(app, runtime, job)
    assert await runtime.calls() == 1


async def test_item_deleted_after_admission_does_not_erase_job_snapshot(
    app, client, account, item, runtime
):
    await runtime.stop("worker")
    owner = await account()
    source = await item(
        owner, title="Synthetic captured title", description="Captured description."
    )
    job, _ = await submit(client, owner, source)
    deleted = await client.delete(f"/api/v1/items/{source['id']}", headers=owner["headers"])
    assert deleted.status_code == 204
    async with app.state.session_factory() as session:
        persisted = await session.get(AIJob, UUID(job["id"]))
        assert persisted.source_title == source["title"]
        assert persisted.source_description == source["description"]
    runtime.start("worker")
    result = await completed(client, owner, job, runtime)
    assert result["status"] == "succeeded"
    assert result["source_item_updated_at"] == source["updated_at"]
    assert await runtime.calls() == 1


def test_private_ai_environment_preserves_normal_credentials(tmp_path, monkeypatch):
    import runpy
    import stat

    initialize = runpy.run_path(str(ROOT / "scripts/init_ai_env.py"))["initialize"]
    monkeypatch.chdir(tmp_path)
    normal = tmp_path / ".env"
    normal.write_bytes(b"unrelated-normal-credentials\n")
    assert initialize(Path(".ai-local"), 8002)
    private = tmp_path / ".ai-local"
    env = private / ".env"
    assert stat.S_IMODE(private.stat().st_mode) == 0o700
    assert stat.S_IMODE((private / "artifacts").stat().st_mode) == 0o700
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    before = env.read_bytes()
    assert not initialize(Path(".ai-local"), 8002)
    assert env.read_bytes() == before
    assert normal.read_bytes() == b"unrelated-normal-credentials\n"
    values = dict(line.split("=", 1) for line in before.decode().splitlines() if "=" in line)
    assert len(values["AI_DB_PASSWORD"]) == 64
    assert len(values["AI_JWT_SECRET"]) == 96
    assert values["AI_DB_PASSWORD"] != values["AI_JWT_SECRET"]


@pytest.mark.parametrize(
    "unsafe",
    [
        "directory_symlink",
        "public_directory",
        "artifact_symlink",
        "env_symlink",
        "env_hardlink",
        "public_env",
    ],
)
def test_private_ai_initializer_rejects_unsafe_existing_paths(tmp_path, monkeypatch, unsafe):
    import runpy

    initialize = runpy.run_path(str(ROOT / "scripts/init_ai_env.py"))["initialize"]
    monkeypatch.chdir(tmp_path)
    private = tmp_path / ".ai-local"
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    preserved = target / "preserved"
    preserved.write_bytes(b"preserved-private-data")
    preserved.chmod(0o600)
    if unsafe == "directory_symlink":
        private.symlink_to(target, target_is_directory=True)
    else:
        private.mkdir(mode=0o700)
        if unsafe == "public_directory":
            private.chmod(0o755)
        elif unsafe == "artifact_symlink":
            (private / "artifacts").symlink_to(target, target_is_directory=True)
        elif unsafe == "env_symlink":
            (private / ".env").symlink_to(preserved)
        elif unsafe == "env_hardlink":
            os.link(preserved, private / ".env")
        else:
            (private / ".env").write_bytes(b"existing-private-data")
            (private / ".env").chmod(0o644)
    with pytest.raises((OSError, ValueError)):
        initialize(Path(".ai-local"), 8002)
    assert preserved.read_bytes() == b"preserved-private-data"
    if unsafe in {"directory_symlink", "public_directory", "artifact_symlink"}:
        assert not (private / ".env").exists()


@pytest.mark.parametrize(
    "directory", [".", "..", "artifacts", "nested/.ai-local", "/tmp/.ai-local"]
)
def test_private_ai_initializer_refuses_escaping_or_normal_paths(tmp_path, monkeypatch, directory):
    import runpy

    initialize = runpy.run_path(str(ROOT / "scripts/init_ai_env.py"))["initialize"]
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        initialize(Path(directory), 8002)


@pytest.mark.parametrize("port", [0, 80, 8000, 8001, 65536])
def test_private_ai_initializer_refuses_normal_or_invalid_ports(tmp_path, monkeypatch, port):
    import runpy

    initialize = runpy.run_path(str(ROOT / "scripts/init_ai_env.py"))["initialize"]
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        initialize(Path(".ai-local"), port)
    assert not (tmp_path / ".ai-local").exists()
