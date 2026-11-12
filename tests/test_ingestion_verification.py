import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_ingestion.py"
_SPEC = importlib.util.spec_from_file_location("verify_ingestion", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
verification = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = verification
_SPEC.loader.exec_module(verification)

INPUT_TOKENS = verification.INPUT_TOKENS
MODEL = verification.MODEL
OUTPUT_TOKENS = verification.OUTPUT_TOKENS
Settings = verification.Settings
VerificationError = verification.VerificationError
await_readback = verification.await_readback
collector_configuration = verification.collector_configuration
load_settings = verification.load_settings
read_langfuse = verification.read_langfuse
read_langsmith = verification.read_langsmith
synthetic_trace = verification.synthetic_trace


@pytest.fixture
def product_server():
    state = {"status": 200, "body": {"data": [], "runs": []}}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.reply()

        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", 0)))
            self.reply()

        def reply(self):
            data = json.dumps(state["body"]).encode()
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def expected_trace():
    _, expected = synthetic_trace()
    return expected


def langfuse_rows(expected):
    return [
        {
            "id": expected["root_span_id"],
            "traceId": expected["trace_id"],
            "name": "ai.request",
            "type": "SPAN",
            "input": None,
            "output": None,
        },
        {
            "id": expected["child_span_id"],
            "traceId": expected["trace_id"],
            "name": "ai.generate",
            "parentObservationId": expected["root_span_id"],
            "type": "GENERATION",
            "model": MODEL,
            "usageDetails": {"input": INPUT_TOKENS, "output": OUTPUT_TOKENS},
            "input": None,
            "output": None,
        },
    ]


def langsmith_rows(expected):
    return [
        {
            "id": expected["root_run_id"],
            "trace_id": expected["root_run_id"],
            "name": "ai.request",
            "run_type": "chain",
            "inputs": {},
            "outputs": {},
        },
        {
            "id": expected["child_run_id"],
            "trace_id": expected["root_run_id"],
            "name": "ai.generate",
            "parent_run_id": expected["root_run_id"],
            "run_type": "llm",
            "extra": {"metadata": {"model": MODEL}},
            "prompt_tokens": INPUT_TOKENS,
            "completion_tokens": OUTPUT_TOKENS,
            "inputs": {},
            "outputs": {},
        },
    ]


@pytest.mark.parametrize("backend", ["langfuse", "langsmith"])
def test_http_acceptance_is_not_success_without_our_stored_trace(product_server, backend):
    url, state = product_server
    expected = expected_trace()
    settings = Settings(
        (backend,),
        langfuse_url=url,
        langfuse_public="test-public",
        langfuse_secret="test-secret",
        langsmith_url=url,
        langsmith_key="test-api-key",
        langsmith_project="verification",
    )
    read = read_langfuse if backend == "langfuse" else read_langsmith
    with httpx.Client(timeout=1, trust_env=False) as client:
        assert read(client, settings, expected) is False
        rows = langfuse_rows(expected) if backend == "langfuse" else langsmith_rows(expected)
        state["body"] = {"data" if backend == "langfuse" else "runs": rows}
        assert read(client, settings, expected) is True
        rows[1]["traceId" if backend == "langfuse" else "trace_id"] = "unrelated"
        assert read(client, settings, expected) is False


@pytest.mark.parametrize("backend", ["langfuse", "langsmith"])
@pytest.mark.parametrize("failure", ["parent", "model", "usage", "content"])
def test_readback_rejects_mislinked_mapped_or_private_records(product_server, backend, failure):
    url, state = product_server
    expected = expected_trace()
    settings = Settings((backend,), langfuse_url=url, langsmith_url=url)
    rows = langfuse_rows(expected) if backend == "langfuse" else langsmith_rows(expected)
    child = rows[1]
    if failure == "parent":
        child["parentObservationId" if backend == "langfuse" else "parent_run_id"] = "unrelated"
    elif failure == "model":
        if backend == "langfuse":
            child["model"] = "wrong-model"
        else:
            child["extra"]["metadata"]["model"] = "wrong-model"
    elif failure == "usage":
        if backend == "langfuse":
            child["usageDetails"]["input"] = 999
        else:
            child["prompt_tokens"] = 999
    else:
        child["input" if backend == "langfuse" else "inputs"] = "private-sentinel"
    state["body"] = {"data" if backend == "langfuse" else "runs": rows}
    read = read_langfuse if backend == "langfuse" else read_langsmith
    with (
        httpx.Client(timeout=1, trust_env=False) as client,
        pytest.raises(VerificationError) as error,
    ):
        read(client, settings, expected)
    assert "private-sentinel" not in str(error.value)


@pytest.mark.parametrize("backend", ["langfuse", "langsmith"])
def test_authentication_error_never_echoes_product_body(product_server, backend):
    url, state = product_server
    state.update(status=401, body={"error": "private-auth-response-sentinel"})
    settings = Settings((backend,), langfuse_url=url, langsmith_url=url)
    read = read_langfuse if backend == "langfuse" else read_langsmith
    with (
        httpx.Client(timeout=1, trust_env=False) as client,
        pytest.raises(VerificationError) as error,
    ):
        read(client, settings, expected_trace())
    assert str(error.value) == "product-authentication-rejected"


def test_no_readback_eventually_fails_instead_of_claiming_ingestion(product_server):
    url, _ = product_server
    with pytest.raises(VerificationError, match="product-readback-deadline-exceeded"):
        await_readback(
            Settings(("langfuse",), langfuse_url=url), expected_trace(), deadline_seconds=1
        )


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://api.smith.langchain.com.evil.invalid",
        "http://api.smith.langchain.com",
        "https://secret@api.smith.langchain.com",
        "https://api.smith.langchain.com?key=private",
    ],
)
def test_cloud_secret_cannot_be_sent_to_unselected_or_credentialed_endpoint(endpoint):
    environment = {
        "LANGSMITH_API_KEY": "test-secret",
        "LANGSMITH_ENDPOINT": endpoint,
        "LANGSMITH_PROJECT": "verification",
    }
    with pytest.raises(VerificationError, match="explicit-langsmith-cloud-endpoint-required"):
        load_settings("langsmith", environment, allow_external_export=True)


def test_cloud_export_requires_explicit_authorisation_and_unambiguous_workspace_routing():
    environment = {
        "LANGSMITH_API_KEY": "test-secret",
        "LANGSMITH_ENDPOINT": "https://eu.api.smith.langchain.com",
        "LANGSMITH_PROJECT": "verification",
    }
    with pytest.raises(VerificationError, match="cloud-export-not-authorised"):
        load_settings("langsmith", environment)
    environment["LANGSMITH_AGENT_ID"] = "verification-agent"
    with pytest.raises(VerificationError, match="select-one-langsmith-project-or-agent"):
        load_settings("langsmith", environment, allow_external_export=True)
    environment.pop("LANGSMITH_PROJECT")
    settings = load_settings("langsmith", environment, allow_external_export=True)
    config, child = collector_configuration(settings)
    encoded = json.dumps(config)
    assert "test-secret" not in encoded and "test-secret" not in repr(settings)
    assert child["LANGSMITH_API_KEY"] == "test-secret"
    headers = config["exporters"]["otlphttp/langsmith"]["headers"]
    assert "Langsmith-Agent-Id" in headers and "Langsmith-Project" not in headers
