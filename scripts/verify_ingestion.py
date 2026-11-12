"""Send one content-free synthetic trace through a Collector and READ IT BACK.

No provider is called. Langfuse must be local; LangSmith requires an explicit cloud
endpoint, project/agent routing, credentials, and --allow-external-export. This tool
never loads dotenv or prints keys/product response bodies. A successful OTLP POST
alone is not a pass. Generated trace/span IDs identify only this verification run.
"""

import argparse
import base64
import json
import os
import re
import secrets
import socket
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import httpx
import yaml
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.resource.v1.resource_pb2 import Resource
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span

ROOT = Path(__file__).resolve().parents[1]
COLLECTOR_IMAGE = "otel/opentelemetry-collector-contrib:0.123.0"
MODEL = "verification-simulator-v1"
INPUT_TOKENS, OUTPUT_TOKENS = 7, 3
_CLOUD_HOSTS = frozenset(
    {
        "api.smith.langchain.com",
        "eu.api.smith.langchain.com",
        "apac.api.smith.langchain.com",
        "aws.api.smith.langchain.com",
    }
)
_PROJECT = re.compile(r"lf-local-[0-9a-f]{24}\Z")


class VerificationError(RuntimeError):
    """Only application-owned, secret-free error codes cross the CLI boundary."""


@dataclass(frozen=True)
class Settings:
    backends: tuple[str, ...]
    langfuse_url: str = "http://127.0.0.1:8005"
    langfuse_public: str = field(default="", repr=False)
    langfuse_secret: str = field(default="", repr=False)
    langfuse_project: str = ""
    langsmith_url: str = ""
    langsmith_key: str = field(default="", repr=False)
    langsmith_project: str = ""
    langsmith_agent: str = ""
    langsmith_environment: str = "development"
    langsmith_workspace: str = ""


def load_settings(
    backend: str, environment: dict[str, str], *, allow_external_export=False
) -> Settings:
    if backend not in {"langfuse", "langsmith", "both"}:
        raise VerificationError("invalid-backend")
    backends = ("langfuse", "langsmith") if backend == "both" else (backend,)
    if "langsmith" in backends and not allow_external_export:
        raise VerificationError("cloud-export-not-authorised")
    lf_url = environment.get("LANGFUSE_BASE_URL", "http://127.0.0.1:8005").rstrip("/")
    ls_url = environment.get("LANGSMITH_ENDPOINT", "").rstrip("/")
    public = environment.get(
        "LANGFUSE_PUBLIC_KEY", environment.get("LANGFUSE_INIT_PROJECT_PUBLIC_KEY", "")
    )
    secret = environment.get(
        "LANGFUSE_SECRET_KEY", environment.get("LANGFUSE_INIT_PROJECT_SECRET_KEY", "")
    )
    lf_project = environment.get("LANGFUSE_LOCAL_PROJECT", "")
    if "langfuse" in backends:
        parsed = urlsplit(lf_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.path
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
            or parsed.port is None
            or not 1024 <= parsed.port <= 65535
        ):
            raise VerificationError("langfuse-must-be-local")
        if not public or not secret:
            raise VerificationError("langfuse-credentials-unavailable")
        if lf_project and not _PROJECT.fullmatch(lf_project):
            raise VerificationError("invalid-langfuse-project")
    project = environment.get("LANGSMITH_PROJECT", "")
    agent = environment.get("LANGSMITH_AGENT_ID", "")
    agent_environment = environment.get("LANGSMITH_AGENT_ENVIRONMENT", "development")
    workspace = environment.get("LANGSMITH_WORKSPACE_ID", "")
    key = environment.get("LANGSMITH_API_KEY", "")
    if "langsmith" in backends:
        parsed = urlsplit(ls_url)
        if (
            parsed.scheme != "https"
            or parsed.hostname not in _CLOUD_HOSTS
            or parsed.path
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
            or parsed.port not in {None, 443}
        ):
            raise VerificationError("explicit-langsmith-cloud-endpoint-required")
        if not key:
            raise VerificationError("langsmith-credentials-unavailable")
        if bool(project) == bool(agent):
            raise VerificationError("select-one-langsmith-project-or-agent")
        if project and (len(project) > 128 or any(ord(c) < 32 for c in project)):
            raise VerificationError("invalid-langsmith-project")
        if agent and (
            not re.fullmatch(r"[a-z][a-z0-9-]{0,61}[a-z0-9]|[a-z]", agent)
            or agent_environment not in {"local", "development", "staging", "production"}
        ):
            raise VerificationError("invalid-langsmith-agent")
        if workspace:
            try:
                UUID(workspace)
            except ValueError:
                raise VerificationError("invalid-langsmith-workspace") from None
    return Settings(
        backends,
        lf_url,
        public,
        secret,
        lf_project,
        ls_url,
        key,
        project,
        agent,
        agent_environment,
        workspace,
    )


def _attribute(name: str, value: str | int) -> KeyValue:
    return KeyValue(
        key=name,
        value=AnyValue(int_value=value) if isinstance(value, int) else AnyValue(string_value=value),
    )


def synthetic_trace() -> tuple[bytes, dict]:
    trace_id = secrets.token_hex(16)
    root_span, child_span = secrets.token_hex(8), secrets.token_hex(8)
    root_run, child_run = str(uuid4()), str(uuid4())
    started = time.time_ns() - 10_000_000
    order = datetime.fromtimestamp(started / 1e9, UTC).strftime("%Y%m%dT%H%M%S%fZ")
    root_order = order + root_run
    root_attributes = {
        "langsmith.trace.id": root_run,
        "langsmith.span.id": root_run,
        "langsmith.span.dotted_order": root_order,
    }
    child_attributes = {
        "langsmith.trace.id": root_run,
        "langsmith.span.id": child_run,
        "langsmith.span.parent_id": root_run,
        "langsmith.span.dotted_order": root_order + "." + order + child_run,
        "provider": "verification-simulator",
        "model": MODEL,
        "prompt_version": "ingestion-verification-v1",
        "schema_version": "1",
        "gen_ai.system": "verification-simulator",
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": MODEL,
        "gen_ai.response.model": MODEL,
        "gen_ai.usage.input_tokens": INPUT_TOKENS,
        "gen_ai.usage.output_tokens": OUTPUT_TOKENS,
    }
    root = Span(
        trace_id=bytes.fromhex(trace_id),
        span_id=bytes.fromhex(root_span),
        name="ai.request",
        kind=Span.SPAN_KIND_INTERNAL,
        start_time_unix_nano=started,
        end_time_unix_nano=started + 5_000_000,
        attributes=[_attribute(k, v) for k, v in root_attributes.items()],
    )
    child = Span(
        trace_id=bytes.fromhex(trace_id),
        span_id=bytes.fromhex(child_span),
        parent_span_id=bytes.fromhex(root_span),
        name="ai.generate",
        kind=Span.SPAN_KIND_CLIENT,
        start_time_unix_nano=started + 1_000_000,
        end_time_unix_nano=started + 4_000_000,
        attributes=[_attribute(k, v) for k, v in child_attributes.items()],
    )
    request = ExportTraceServiceRequest(
        resource_spans=[
            ResourceSpans(
                resource=Resource(
                    attributes=[
                        _attribute("service.name", "ingestion-verification"),
                        _attribute("service.namespace", "item-rewrite"),
                    ]
                ),
                scope_spans=[ScopeSpans(spans=[root, child])],
            )
        ]
    )
    expected = {
        "trace_id": trace_id,
        "root_span_id": root_span,
        "child_span_id": child_span,
        "root_run_id": root_run,
        "child_run_id": child_run,
        "synthetic_usage": True,
    }
    return request.SerializeToString(), expected


def collector_configuration(settings: Settings) -> tuple[dict, dict[str, str]]:
    config = yaml.safe_load((ROOT / "observability/collector.verify.yaml").read_text())
    config["service"]["pipelines"] = {
        name: value
        for name, value in config["service"]["pipelines"].items()
        if name.split("/")[-1] in settings.backends
    }
    config["exporters"] = {
        name: value
        for name, value in config["exporters"].items()
        if name.split("/")[-1] in settings.backends
    }
    config["processors"] = {
        name: value
        for name, value in config["processors"].items()
        if not name.startswith("transform/") or name.split("/")[-1] in settings.backends
    }
    environment = {}
    if "langfuse" in settings.backends:
        local = urlsplit(settings.langfuse_url)
        destination = (
            "http://langfuse-web:3000"
            if settings.langfuse_project
            else f"http://host.docker.internal:{local.port}"
        )
        environment["VERIFY_LANGFUSE_OTLP_BASE"] = destination + "/api/public/otel"
        encoded = base64.b64encode(
            f"{settings.langfuse_public}:{settings.langfuse_secret}".encode()
        ).decode()
        environment["VERIFY_LANGFUSE_AUTH"] = "Basic " + encoded
    if "langsmith" in settings.backends:
        environment["VERIFY_LANGSMITH_OTLP_BASE"] = settings.langsmith_url + "/otel"
        environment["LANGSMITH_API_KEY"] = settings.langsmith_key
        headers = config["exporters"]["otlphttp/langsmith"]["headers"]
        if settings.langsmith_project:
            environment["VERIFY_LANGSMITH_PROJECT"] = settings.langsmith_project
            headers["Langsmith-Project"] = "${env:VERIFY_LANGSMITH_PROJECT}"
        else:
            environment["VERIFY_LANGSMITH_AGENT"] = settings.langsmith_agent
            environment["VERIFY_LANGSMITH_ENVIRONMENT"] = settings.langsmith_environment
            headers["Langsmith-Agent-Id"] = "${env:VERIFY_LANGSMITH_AGENT}"
            headers["Langsmith-Agent-Environment"] = "${env:VERIFY_LANGSMITH_ENVIRONMENT}"
        if settings.langsmith_workspace:
            environment["VERIFY_LANGSMITH_WORKSPACE"] = settings.langsmith_workspace
            headers["x-tenant-id"] = "${env:VERIFY_LANGSMITH_WORKSPACE}"
    return config, environment


def _json_request(client: httpx.Client, method: str, url: str, **kwargs) -> dict:
    try:
        response = client.request(method, url, **kwargs)
        if response.status_code in {401, 403}:
            raise VerificationError("product-authentication-rejected")
        if response.status_code != 200:
            raise VerificationError(f"product-http-{response.status_code}")
        value = response.json()
    except (httpx.HTTPError, ValueError):
        raise VerificationError("product-readback-transport-or-json-failure") from None
    if not isinstance(value, dict):
        raise VerificationError("product-readback-invalid-shape")
    return value


def read_langfuse(client: httpx.Client, settings: Settings, expected: dict) -> bool:
    body = _json_request(
        client,
        "GET",
        settings.langfuse_url + "/api/public/v2/observations",
        auth=(settings.langfuse_public, settings.langfuse_secret),
        params={
            "traceId": expected["trace_id"],
            "limit": 10,
            "fields": "core,basic,io,metadata,model,usage,trace_context",
        },
    )
    rows = body.get("data")
    if not isinstance(rows, list):
        raise VerificationError("langfuse-invalid-observations")
    rows = {
        row.get("id"): row
        for row in rows
        if isinstance(row, dict) and row.get("traceId") == expected["trace_id"]
    }
    root, child = rows.get(expected["root_span_id"]), rows.get(expected["child_span_id"])
    if root is None or child is None:
        return False
    usage = child.get("usageDetails", {})
    if (
        root.get("name") != "ai.request"
        or child.get("name") != "ai.generate"
        or child.get("parentObservationId") != expected["root_span_id"]
        or child.get("type") != "GENERATION"
        or child.get("model") != MODEL
        or usage.get("input") != INPUT_TOKENS
        or usage.get("output") != OUTPUT_TOKENS
    ):
        raise VerificationError("langfuse-stored-mapping-mismatch")
    if any(
        row.get(key) not in (None, "", {}, [])
        for row in (root, child)
        for key in ("input", "output")
    ):
        raise VerificationError("langfuse-unexpected-stored-content")
    return True


def read_langsmith(client: httpx.Client, settings: Settings, expected: dict) -> bool:
    headers = {"x-api-key": settings.langsmith_key}
    if settings.langsmith_workspace:
        headers["x-tenant-id"] = settings.langsmith_workspace
    body = _json_request(
        client,
        "POST",
        settings.langsmith_url + "/runs/query",
        headers=headers,
        json={
            "trace": expected["root_run_id"],
            "limit": 10,
            "id": [expected["root_run_id"], expected["child_run_id"]],
            "select": [
                "id",
                "name",
                "run_type",
                "parent_run_id",
                "trace_id",
                "inputs",
                "outputs",
                "extra",
                "prompt_tokens",
                "completion_tokens",
            ],
        },
    )
    rows = body.get("runs")
    if not isinstance(rows, list):
        raise VerificationError("langsmith-invalid-runs")
    rows = {
        str(row.get("id")): row
        for row in rows
        if isinstance(row, dict) and str(row.get("trace_id")) == expected["root_run_id"]
    }
    root, child = rows.get(expected["root_run_id"]), rows.get(expected["child_run_id"])
    if root is None or child is None:
        return False
    metadata = child.get("extra", {}).get("metadata", {})
    if (
        root.get("name") != "ai.request"
        or child.get("name") != "ai.generate"
        or child.get("parent_run_id") != expected["root_run_id"]
        or child.get("run_type") != "llm"
        or metadata.get("model") != MODEL
        or child.get("prompt_tokens") != INPUT_TOKENS
        or child.get("completion_tokens") != OUTPUT_TOKENS
    ):
        raise VerificationError("langsmith-stored-mapping-mismatch")
    if any(
        row.get(key) not in (None, "", {}, [])
        for row in (root, child)
        for key in ("inputs", "outputs")
    ):
        raise VerificationError("langsmith-unexpected-stored-content")
    return True


def await_readback(settings: Settings, expected: dict, *, deadline_seconds: float = 90) -> dict:
    if not 1 <= deadline_seconds <= 180:
        raise VerificationError("invalid-readback-deadline")
    deadline = time.monotonic() + deadline_seconds
    pending = set(settings.backends)
    with httpx.Client(timeout=3, trust_env=False, follow_redirects=False) as client:
        while pending:
            for backend in tuple(pending):
                read = read_langfuse if backend == "langfuse" else read_langsmith
                if read(client, settings, expected):
                    pending.remove(backend)
            if not pending:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise VerificationError("product-readback-deadline-exceeded")
            time.sleep(min(1, remaining))
    return {
        "status": "passed",
        "backends": list(settings.backends),
        **expected,
        "model": MODEL,
        "input_tokens": INPUT_TOKENS,
        "output_tokens": OUTPUT_TOKENS,
        "provider_calls": 0,
        "content_recorded": False,
    }


def _docker(arguments: list[str], environment: dict[str, str], *, capture=False) -> str:
    try:
        result = subprocess.run(
            ["docker", *arguments],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise VerificationError("collector-command-unavailable-or-timeout") from None
    if result.returncode:
        raise VerificationError("collector-command-failed")
    return result.stdout.decode().strip() if capture else ""


def run_verification(
    settings: Settings, *, artifacts_root: Path = ROOT / "artifacts", deadline_seconds: float = 90
) -> dict:
    artifacts_root.mkdir(mode=0o700, exist_ok=True)
    metadata = artifacts_root.lstat()
    if (
        not artifacts_root.is_dir()
        or artifacts_root.is_symlink()
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o777 != 0o700
    ):
        raise VerificationError("unsafe-artifact-directory")
    directory = artifacts_root / f"ingestion-{uuid4().hex}"
    directory.mkdir(mode=0o700)
    config, exporter_environment = collector_configuration(settings)
    config_path = directory / "collector.yaml"
    # The template contains environment references, not secrets; the nonroot
    # Collector must be able to read its bind-mounted copy inside this 0700 directory.
    with open(config_path, "x", opener=lambda path, flags: os.open(path, flags, 0o644)) as handle:
        yaml.safe_dump(config, handle)
    for port in (14318, 14333):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    allowed = (
        "PATH",
        "HOME",
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_CONFIG",
        "DOCKER_TLS_VERIFY",
        "DOCKER_CERT_PATH",
        "SSH_AUTH_SOCK",
    )
    environment = {key: os.environ[key] for key in allowed if key in os.environ}
    environment.update(exporter_environment)
    name = "ingestion-verify-" + uuid4().hex
    arguments = [
        "create",
        "--name",
        name,
        "--log-driver",
        "none",
        "--publish",
        "127.0.0.1:14318:4318",
        "--publish",
        "127.0.0.1:14333:13133",
        "--mount",
        f"type=bind,src={config_path},dst=/etc/verify.yaml,readonly",
        "--add-host",
        "host.docker.internal:host-gateway",
    ]
    if settings.langfuse_project:
        arguments += ["--network", settings.langfuse_project + "_api_ingress"]
    for key in exporter_environment:
        arguments += ["--env", key]
    arguments += [COLLECTOR_IMAGE, "--config=/etc/verify.yaml"]
    container_id = None
    try:
        container_id = _docker(arguments, environment, capture=True)
        _docker(["start", container_id], environment)
        with httpx.Client(timeout=2, trust_env=False, follow_redirects=False) as client:
            ready_deadline = time.monotonic() + 15
            while True:
                try:
                    if client.get("http://127.0.0.1:14333/").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() >= ready_deadline:
                    raise VerificationError("collector-readiness-deadline-exceeded")
                time.sleep(0.2)
            payload, expected = synthetic_trace()
            response = client.post(
                "http://127.0.0.1:14318/v1/traces",
                content=payload,
                headers={"Content-Type": "application/x-protobuf"},
            )
            if response.status_code != 200:
                raise VerificationError("collector-rejected-synthetic-trace")
        result = await_readback(settings, expected, deadline_seconds=deadline_seconds)
        result["artifacts"] = (
            str(directory.relative_to(ROOT)) if directory.is_relative_to(ROOT) else str(directory)
        )
        with open(
            directory / "result.json", "x", opener=lambda path, flags: os.open(path, flags, 0o600)
        ) as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        return result
    finally:
        if container_id is not None:
            _docker(["rm", "--force", container_id], environment)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("langfuse", "langsmith", "both"), required=True)
    parser.add_argument("--allow-external-export", action="store_true")
    parser.add_argument("--langfuse-state", type=Path)
    args = parser.parse_args(argv)
    try:
        environment = dict(os.environ)
        if args.langfuse_state:
            from scripts.langfuse_local import load_environment

            environment.update(load_environment(args.langfuse_state))
        settings = load_settings(
            args.backend, environment, allow_external_export=args.allow_external_export
        )
        result = run_verification(settings)
        print(json.dumps(result, sort_keys=True))
        return 0
    except VerificationError as error:
        print(f"Ingestion verification failed: {error}")
        return 1
    except (OSError, ValueError):
        print("Ingestion verification failed: local configuration or infrastructure error.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
