"""Shared harness for AI_QA end-to-end tests: guarded child processes and unique topics.

    isolated_runtime(app)
      guards: AI enabled, local simulator + receiver only, one E2E run at a time
      -> unique Kafka topics -> start child roles -> yield Runtime
      -> stop children -> wait for the simulator to drain -> delete topics

The app fixture finishes destructive cleanup BEFORE children start. Every child is
stopped BEFORE that fixture tears down.
"""

import asyncio
import fcntl
import os
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from aiokafka.admin import AIOKafkaAdminClient, NewTopic

from app.ai.config import AISettings

ROOT = Path(__file__).resolve().parents[2]
# Child role -> module run with `python -m`.
ROLES = {
    "dispatcher": "app.ai.dispatcher",
    "worker": "app.ai.worker",
    "rag-worker": "app.rag.worker",
}
# Processing roles receive neither JWT nor Redis credentials, even in subprocess QA.
_CHILD_PREFIXES = ("DB_", "AI_", "RAG_", "KAFKA_", "OTEL_")
_CHILD_KEYS = {"DATABASE_URL", "PATH", "PYTHONPATH", "HOME", "LANG", "PYTHONUNBUFFERED"}
_LOCK_PATH = "/tmp/item-rewrite-ai-e2e.lock"


async def eventually(check, *, seconds: float = 45, message: str = "condition deadline exceeded"):
    deadline = time.monotonic() + seconds
    while True:
        value = await check()
        if value:
            return value
        if time.monotonic() >= deadline:
            raise AssertionError(message)
        await asyncio.sleep(0.05)


@dataclass
class Runtime:
    settings: AISettings
    environment: dict[str, str]
    simulator: httpx.AsyncClient
    receiver: httpx.AsyncClient
    processes: dict[str, subprocess.Popen] = field(default_factory=dict)

    def start(self, role: str, **overrides: str) -> None:
        if role in self.processes:
            raise AssertionError("role already started")
        self.processes[role] = subprocess.Popen(
            [sys.executable, "-m", ROLES[role]],
            cwd=ROOT,
            env={**self.environment, **overrides},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    async def stop(self, role: str, *, crash: bool = False) -> None:
        process = self.processes.pop(role, None)
        if process is None:
            return
        if process.poll() is None:
            try:
                process.kill() if crash else process.terminate()
            except ProcessLookupError:
                pass
        try:
            await asyncio.to_thread(process.wait, timeout=50)
        except subprocess.TimeoutExpired:
            process.kill()
            await asyncio.to_thread(process.wait, timeout=5)
            if not crash:
                raise AssertionError("AI child exceeded graceful shutdown deadline") from None

    async def close(self) -> None:
        failures = []
        for role in tuple(self.processes):
            try:
                await self.stop(role)
            except AssertionError as exc:
                failures.append(exc)
        if failures:
            raise failures[0]

    def healthy(self) -> None:
        if any(process.poll() is not None for process in self.processes.values()):
            raise AssertionError(
                "AI child exited unexpectedly; no provider response details emitted"
            )

    async def control(self, modes: list[str], *, delay_seconds: float = 3, max_calls: int = 10):
        response = await self.simulator.post(
            "/__control",
            json={"modes": modes, "delay_seconds": delay_seconds, "max_calls": max_calls},
        )
        assert response.status_code == 200

    async def calls(self) -> int:
        response = await self.simulator.get("/__state")
        assert response.status_code == 200
        return response.json()["calls"]


def _local_urls() -> tuple[str, str]:
    provider_url = os.environ.get("AI_PROVIDER_SIMULATOR_URL", "http://provider:8080")
    receiver_url = os.environ.get("AI_OTLP_RECEIVER_URL", "http://otlp:4318")
    for value, hosts in (
        (provider_url, {"provider", "localhost", "127.0.0.1"}),
        (receiver_url, {"otlp", "localhost", "127.0.0.1"}),
    ):
        parsed = urlsplit(value)
        if parsed.scheme != "http" or parsed.hostname not in hosts or parsed.username:
            raise AssertionError("AI QA refuses nonlocal infrastructure")
    return provider_url, receiver_url


@asynccontextmanager
async def isolated_runtime(
    app,
    *,
    settings_type: type[AISettings],
    overrides: dict[str, str],
    topics: list[str],
    roles: list[str],
) -> AsyncIterator[Runtime]:
    # Explicit guards in addition to the ordinary fixture's resolved driver safety guards.
    if not app.state.ai_settings or not app.state.ai_settings.ai_enabled:
        raise AssertionError("AI QA requires AI_ENABLED")
    provider_url, receiver_url = _local_urls()
    lock_fd = os.open(_LOCK_PATH, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(lock_fd)
        raise AssertionError("AI E2E cannot run concurrently") from None
    overrides = {
        **overrides,
        "OPENAI_BASE_URL": f"{provider_url}/v1",
        "OPENAI_API_KEY": "local-simulator-only-not-a-real-key",
    }
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(_CHILD_PREFIXES) or key in _CHILD_KEYS
    }
    environment.update(overrides)
    settings = settings_type(**{key.lower(): value for key, value in overrides.items()})
    admin = AIOKafkaAdminClient(bootstrap_servers=settings.kafka_bootstrap_servers)
    started = False
    created = False
    children = None
    try:
        async with asyncio.timeout(15):
            await admin.start()
        started = True
        async with asyncio.timeout(15):
            await admin.create_topics(
                [
                    NewTopic(
                        topic,
                        num_partitions=6,
                        replication_factor=1,
                        topic_configs={"retention.ms": "60000", "retention.bytes": "1048576"},
                    )
                    for topic in topics
                ]
            )
        created = True
        async with (
            httpx.AsyncClient(base_url=provider_url, timeout=5, trust_env=False) as simulator,
            httpx.AsyncClient(base_url=receiver_url, timeout=5, trust_env=False) as receiver,
        ):
            children = Runtime(settings, environment, simulator, receiver)
            await children.control([])
            assert (await receiver.post("/__reset")).status_code == 200
            for role in roles:
                children.start(role)
            try:
                yield children
            finally:
                await children.close()

                async def idle():
                    response = await simulator.get("/__state")
                    return response.status_code == 200 and response.json()["in_flight"] == 0

                await eventually(idle, seconds=10, message="simulator did not drain")
    finally:
        try:
            if children is not None and children.processes:
                await children.close()
            if created:
                async with asyncio.timeout(15):
                    await admin.delete_topics(topics)
        finally:
            if started:
                await admin.close()
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
