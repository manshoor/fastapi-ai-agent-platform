"""Use real TCP + the installed SDK, never an SDK-shaped mock response."""

import asyncio
import importlib.util
import json
import socket
import sys
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import SecretStr

from app.ai.config import AISettings, ProviderSettings
from app.ai.provider import Generation, OpenAIRewriter, ProviderFailure


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
async def simulator():
    module = load_script(
        "provider_simulator_regression", Path(__file__).parent / "support/provider_simulator.py"
    )
    app = module.create_app()
    # Capture requests ONLY in this test process to assert the actual wire contract.
    captured = []

    @app.middleware("http")
    async def capture(request, call_next):
        if request.url.path == "/v1/responses":
            captured.append(json.loads(await request.body()))
        return await call_next(request)

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    server = uvicorn.Server(
        uvicorn.Config(app, access_log=False, log_level="critical", lifespan="off")
    )
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(3):
            while not server.started:
                if task.done():
                    await task
                await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{sock.getsockname()[1]}", app, captured
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 3)
        sock.close()


def provider_for(base_url, **updates):
    return OpenAIRewriter(
        AISettings(
            openai_api_key=SecretStr("synthetic-key-never-external"),
            openai_base_url=base_url + "/v1",
            ai_provider_timeout_seconds=1,
        ).model_copy(update=updates)
    )


async def configure(base_url, mode, delay=0.4):
    async with httpx.AsyncClient(base_url=base_url, trust_env=False) as client:
        response = await client.post("/__control", json={"modes": [mode], "delay_seconds": delay})
        assert response.status_code == 200


async def test_real_sdk_success_wire_contract_and_pinned_metadata(simulator):
    url, _, captured = simulator
    provider = provider_for(url, ai_model="pinned-model")
    try:
        result = await provider.rewrite("Source café", "Facts: 48 pages.", "Be clear.")
    finally:
        await provider.aclose()
    assert result == Generation(
        title="Clear synthetic title",
        description="A clear synthetic description.",
        model="simulator-item-rewrite",
        input_tokens=25,
        output_tokens=15,
        request_id=result.request_id,
    )
    assert result.request_id.startswith("req_sim_")
    assert len(captured) == 1
    wire = captured[0]
    assert wire["model"] == "pinned-model"
    assert wire["store"] is False and wire["stream"] is False
    assert wire["tools"] == [] and wire["tool_choice"] == "none"
    assert wire["reasoning"] == {"effort": "none"}
    assert wire["truncation"] == "disabled" and wire["max_output_tokens"] == 4096
    assert json.loads(wire["input"]) == {
        "title": "Source café",
        "description": "Facts: 48 pages.",
        "instruction": "Be clear.",
    }
    schema = wire["text"]["format"]
    assert schema["type"] == "json_schema" and schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False
    assert set(schema["schema"]["required"]) == {"title", "description"}


@pytest.mark.parametrize(
    ("mode", "code", "ambiguous", "retryable"),
    [
        ("refusal", "provider_refusal", False, False),
        ("invalid", "provider_invalid_output", False, False),
        ("incomplete", "provider_incomplete", False, False),
        ("rate_limit", "provider_rate_limit", False, True),
        ("server_error", "provider_unavailable", True, False),
        ("timeout", "provider_timeout", True, False),
    ],
)
async def test_real_sdk_classifies_one_attempt(simulator, mode, code, ambiguous, retryable):
    url, app, captured = simulator
    await configure(url, mode)
    provider = provider_for(url, ai_provider_timeout_seconds=0.1 if mode == "timeout" else 1)
    try:
        with pytest.raises(ProviderFailure) as error:
            await provider.rewrite("Synthetic title", "Synthetic description", "Rewrite clearly")
        assert (error.value.code, error.value.ambiguous, error.value.retryable) == (
            code,
            ambiguous,
            retryable,
        )
        assert str(error.value) == code
        assert error.value.__cause__ is None
        assert app.state.simulator.calls == 1
        assert len(captured) == 1  # SDK retries are disabled, including 429 and 500.
    finally:
        await provider.aclose()


async def test_transport_connect_failure_is_proven_presend():
    # Bound but deliberately non-listening port cannot accept a request; no internet/DNS.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    provider = provider_for(f"http://127.0.0.1:{sock.getsockname()[1]}")
    try:
        with pytest.raises(ProviderFailure) as error:
            await provider.rewrite("Synthetic", "", "Polish")
        assert error.value.code == "provider_connection"
        assert error.value.retryable and not error.value.ambiguous
    finally:
        await provider.aclose()
        sock.close()


async def test_cancellation_propagates_without_replay(simulator):
    url, app, _ = simulator
    await configure(url, "timeout")
    provider = provider_for(url)
    task = asyncio.create_task(provider.rewrite("Synthetic", "", "Polish"))
    try:
        async with asyncio.timeout(2):
            while not app.state.simulator.in_flight:
                await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert app.state.simulator.calls == 1
    finally:
        await provider.aclose()


async def test_unknown_pinned_versions_never_send(simulator):
    url, _, captured = simulator
    provider = provider_for(url)
    try:
        for kwargs, code in [
            ({"schema_version": "2"}, "provider_schema_version"),
            ({"prompt_version": "unknown"}, "provider_prompt_version"),
        ]:
            with pytest.raises(ProviderFailure) as error:
                await provider.rewrite("Synthetic", "", "Polish", **kwargs)
            assert error.value.code == code
        assert captured == []
    finally:
        await provider.aclose()


@pytest.mark.parametrize(
    "bad_output",
    [
        {"title": "", "description": ""},
        {"title": "x" * 201, "description": ""},
        {"title": "Valid", "description": "x" * 5001},
        {"title": "Valid\u0000", "description": ""},
        {"title": "Valid"},
        {"title": "Valid", "description": "", "extra": "must reject"},
    ],
)
async def test_real_sdk_app_schema_rejects_invalid_generated_fields(simulator, bad_output):
    url, app, _ = simulator
    # Replace only the simulator HTTP route: the SDK still parses a real wire response.
    original = next(route for route in app.routes if getattr(route, "path", "") == "/v1/responses")
    app.routes.remove(original)

    @app.post("/v1/responses")
    async def malformed(request: Request):
        return JSONResponse(
            {
                "id": "resp_invalid",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "synthetic-model",
                "output": [
                    {
                        "id": "msg_invalid",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(bad_output),
                                "annotations": [],
                            }
                        ],
                    }
                ],
            }
        )

    provider = provider_for(url)
    try:
        with pytest.raises(ProviderFailure) as error:
            await provider.rewrite("Synthetic", "", "Polish")
        assert error.value.code == "provider_invalid_output"
        assert not error.value.retryable and not error.value.ambiguous
    finally:
        await provider.aclose()


async def test_quota_rejection_is_not_retryable(simulator):
    url, _, _ = simulator
    async with httpx.AsyncClient(base_url=url, trust_env=False) as client:
        assert (await client.post("/__control", json={"max_calls": 1})).status_code == 200
    provider = provider_for(url)
    try:
        await provider.rewrite("Synthetic", "", "Polish")
        with pytest.raises(ProviderFailure) as error:
            await provider.rewrite("Synthetic", "", "Polish")
        assert error.value.code == "provider_rate_limit" and not error.value.retryable
    finally:
        await provider.aclose()


def test_external_provider_without_key_is_rejected(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        OpenAIRewriter(AISettings(openai_api_key=None, openai_base_url="https://api.openai.com/v1"))


def test_offline_rubric_and_network_consent(monkeypatch, capsys):
    runner = load_script(
        "rewrite_evaluation_regression", Path(__file__).parents[1] / "scripts/evaluate_rewrites.py"
    )
    cases = runner.load_dataset()
    assert all(all(runner.score_suggestion(case, case["expected"]).values()) for case in cases)
    broken = dict(cases[0]["expected"], description="500 ml. Dishwasher safe.")
    scores = runner.score_suggestion(cases[0], broken)
    assert scores["schema"] and not scores["required_facts"]
    assert not scores["forbidden_claims"] and not scores["numbers"]
    assert not runner.score_suggestion(cases[0], {"title": ""})["schema"]
    for key in ("OPENAI_API_KEY", "LANGSMITH_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    for flags in (
        ["--provider"],
        ["--provider", "--allow-paid-requests"],
        ["--publish-langsmith"],
        ["--publish-langsmith", "--allow-external-publication"],
    ):
        with pytest.raises(SystemExit) as error:
            runner.parse_args(flags)
        assert error.value.code == 2
    assert runner.main([]) == 0
    output = capsys.readouterr().out
    assert '"mode": "offline"' in output
    assert "350 ml" not in output and "Synthetic" not in output


async def test_provider_comparison_uses_identical_cases_and_real_sdk(simulator):
    url, app, captured = simulator
    runner = load_script(
        "rewrite_evaluation_network", Path(__file__).parents[1] / "scripts/evaluate_rewrites.py"
    )
    cases = runner.load_dataset()[:2]
    settings = AISettings(
        openai_base_url=url + "/v1",
        openai_api_key=SecretStr("synthetic-key"),
        ai_provider_timeout_seconds=1,
    )
    records = await runner.evaluate(
        cases, settings=settings, models=["comparison-one", "comparison-two"]
    )
    assert app.state.simulator.calls == 4
    assert [wire["model"] for wire in captured] == [
        "comparison-one",
        "comparison-one",
        "comparison-two",
        "comparison-two",
    ]
    assert captured[0]["input"] == captured[2]["input"]
    assert captured[1]["input"] == captured[3]["input"]
    # Simulator returns fixed content unrelated to source: the rubric must fail,
    # not quietly substitute the reference answer or treat JSON validity as quality.
    assert all(record["scores"]["schema"] for record in records)
    assert not any(record["scores"]["required_facts"] for record in records)


async def test_borrowed_job_wrappers_share_client_and_do_not_close_it(simulator):
    url, app, captured = simulator
    owning = provider_for(url)
    borrowed = OpenAIRewriter(
        owning.settings.model_copy(update={"ai_model": "job-pinned-model"}),
        client=owning.client,
    )
    try:
        assert borrowed.client is owning.client
        await borrowed.rewrite("Synthetic", "", "Polish")
        await borrowed.aclose()
        await owning.rewrite("Synthetic", "", "Polish")
        assert app.state.simulator.calls == 2
        assert [wire["model"] for wire in captured] == [
            "job-pinned-model",
            owning.settings.ai_model,
        ]
    finally:
        await owning.aclose()


def test_saved_candidate_ids_must_match_dataset(tmp_path):
    runner = load_script(
        "rewrite_evaluation_ids", Path(__file__).parents[1] / "scripts/evaluate_rewrites.py"
    )
    cases = runner.load_dataset()
    path = tmp_path / "responses.jsonl"
    path.write_text(
        "\n".join(json.dumps({"id": case["id"], **case["expected"]}) for case in cases),
        encoding="utf-8",
    )
    assert set(runner.load_responses(path, cases)) == {case["id"] for case in cases}
    path.write_text(json.dumps({"id": "missing", "title": "Synthetic", "description": ""}))
    with pytest.raises(ValueError):
        runner.load_responses(path, cases)


async def test_real_langsmith_sdk_publishes_linked_dataset_runs_and_feedback(
    simulator,
    monkeypatch,
):
    from datetime import UTC, datetime
    from uuid import uuid4

    url, app, _ = simulator
    runner = load_script(
        "rewrite_evaluation_publication", Path(__file__).parents[1] / "scripts/evaluate_rewrites.py"
    )
    dataset_id = str(uuid4())
    tenant_id = str(uuid4())
    stored = {"dataset": None, "examples": [], "projects": [], "runs": [], "feedback": []}
    now = datetime.now(UTC).isoformat()

    @app.get("/info")
    async def info():
        return {"version": "0.8.0", "instance_flags": {"dataset_examples_multipart_enabled": False}}

    @app.get("/datasets")
    async def datasets():
        return [stored["dataset"]] if stored["dataset"] else []

    @app.post("/datasets")
    async def create_dataset(request: Request):
        value = await request.json()
        stored["dataset"] = {**value, "id": dataset_id, "created_at": now}
        return stored["dataset"]

    @app.get("/sessions")
    async def sessions():
        return []

    @app.post("/sessions")
    async def create_project(request: Request):
        value = await request.json()
        stored["projects"].append(value)
        return {**value, "tenant_id": tenant_id, "start_time": now}

    @app.get("/examples")
    async def examples():
        return []

    @app.post("/examples/bulk")
    async def create_examples(request: Request):
        rows = await request.json()
        stored["examples"].extend(rows)
        return [{"id": row["id"]} for row in rows]

    @app.post("/runs")
    async def create_run(request: Request):
        stored["runs"].append(await request.json())
        return {}

    @app.post("/feedback")
    async def feedback(request: Request):
        stored["feedback"].append(await request.json())
        return {}

    monkeypatch.setenv("LANGSMITH_ENDPOINT", url)
    monkeypatch.setenv("LANGSMITH_API_KEY", "synthetic-local-only")
    monkeypatch.setenv("LANGSMITH_TRACING_SAMPLING_RATE", "1")
    monkeypatch.delenv("LANGSMITH_TRACING_MODE", raising=False)
    cases = runner.load_dataset()[:2]
    records = await runner.evaluate(cases)
    # Actual synchronous SDK uses a real loopback HTTP server, serviced by this loop.
    await asyncio.to_thread(runner.publish_langsmith, cases, records, dataset_name="synthetic-test")
    assert stored["dataset"]["name"] == "synthetic-test"
    assert len(stored["examples"]) == len(stored["runs"]) == 2
    assert len(stored["projects"]) == 1 and len(stored["feedback"]) == 10
    assert stored["projects"][0]["reference_dataset_id"] == dataset_id
    example_ids = {row["id"] for row in stored["examples"]}
    run_ids = {row["id"] for row in stored["runs"]}
    assert {row["reference_example_id"] for row in stored["runs"]} == example_ids
    assert {row["run_id"] for row in stored["feedback"]} == run_ids
    assert all(row["score"] == 1 for row in stored["feedback"])
    assert {row["key"] for row in stored["feedback"]} == {
        "schema",
        "required_facts",
        "forbidden_claims",
        "numbers",
        "bounds",
    }
    assert all(row["outputs"] for row in stored["examples"])


async def test_provider_only_settings_need_no_database_configuration(simulator, monkeypatch):
    url, app, _ = simulator
    for name in ("DATABASE_URL", "DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    settings = ProviderSettings(
        openai_api_key=SecretStr("synthetic-local-key"), openai_base_url=f"{url}/v1"
    )
    provider = OpenAIRewriter(settings)
    try:
        result = await provider.rewrite("Synthetic", "", "Clarify")
        assert result.title == "Clear synthetic title" and app.state.simulator.calls == 1
    finally:
        await provider.aclose()


async def test_sdk_debug_logging_cannot_expose_item_content(simulator, caplog):
    import logging

    url, app, _ = simulator
    caplog.set_level(logging.DEBUG, logger="openai._base_client")
    provider = provider_for(url)
    try:
        result = await provider.rewrite(
            "private-title-sentinel", "private-description-sentinel", "private-instruction-sentinel"
        )
        assert result.title == "Clear synthetic title" and app.state.simulator.calls == 1
        assert "private-title-sentinel" not in caplog.text
        assert "private-description-sentinel" not in caplog.text
        assert "private-instruction-sentinel" not in caplog.text
    finally:
        await provider.aclose()
