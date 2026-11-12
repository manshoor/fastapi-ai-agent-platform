"""Bounded, synthetic OpenAI Responses HTTP simulator; never contacts a provider."""

import argparse
import asyncio
import hashlib
import json
import math
import re
import time
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass, field
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

MODES = frozenset(
    {
        "success",
        "refusal",
        "invalid",
        "incomplete",
        "timeout",
        "rate_limit",
        "server_error",
        # Grounded answer that cites no passage.
        "uncited",
    }
)
SUGGESTION = {"title": "Clear synthetic title", "description": "A clear synthetic description."}
EMBEDDING_DIMENSIONS = 1536
# Embedding requests carry up to 64 chunks of about 1200 characters each.
EMBEDDING_BODY_LIMIT = 2 * 1024 * 1024
WORD = re.compile(r"\w+")


def embed_text(text: str, dimensions: int) -> list[float]:
    """Deterministic hashed bag-of-words vector: shared words give nearby vectors."""
    vector = [0.0] * dimensions
    for word in WORD.findall(text.lower()):
        digest = hashlib.sha256(word.encode()).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        vector[index] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        vector[0] = 1.0
        return vector
    return [value / norm for value in vector]


def grounded_answer(request: dict) -> dict:
    """Cite the first passage; with no passages, answer that nothing was found."""
    try:
        passages = json.loads(request.get("input", "")).get("passages", [])
    except (ValueError, AttributeError):
        passages = []
    first = passages[0] if isinstance(passages, list) and passages else None
    if not isinstance(first, dict) or not isinstance(first.get("text"), str):
        return {"answer": "The passages do not contain the answer.", "citations": []}
    answer = f"Synthetic grounded answer: {first['text'][:200]}"
    return {"answer": answer, "citations": ["P1"]}


def wants_grounded_answer(request: dict) -> bool:
    text_format = request.get("text", {}).get("format", {})
    properties = text_format.get("schema", {}).get("properties", {})
    return isinstance(properties, dict) and "citations" in properties


@dataclass
class SimulatorState:
    modes: list[str] = field(default_factory=list)
    delay_seconds: float = 5.0
    max_calls: int = 100
    calls: int = 0
    in_flight: int = 0
    counts: Counter = field(default_factory=Counter)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


async def bounded_json(request: Request, limit: int = 65536) -> dict | None:
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            return None
    try:
        value = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def create_app() -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    state = SimulatorState()
    app.state.simulator = state

    @app.get("/health/live")
    async def live() -> dict:
        return {"status": "ok"}

    @app.get("/__state")
    async def inspect() -> dict:
        async with state.lock:
            return {
                "calls": state.calls,
                "in_flight": state.in_flight,
                "counts": dict(state.counts),
                "max_calls": state.max_calls,
            }

    @app.post("/__control")
    async def control(request: Request) -> Response:
        value = await bounded_json(request)
        if value is None or set(value) - {"modes", "delay_seconds", "max_calls"}:
            return JSONResponse({"error": "invalid_control"}, status_code=400)
        modes = value.get("modes", [])
        delay = value.get("delay_seconds", 5.0)
        maximum = value.get("max_calls", 100)
        if (
            not isinstance(modes, list)
            or len(modes) > 100
            or any(not isinstance(mode, str) or mode not in MODES for mode in modes)
            or isinstance(delay, bool)
            or not isinstance(delay, (int, float))
            or not 0 <= delay <= 120
            or isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or not 1 <= maximum <= 1000
        ):
            return JSONResponse({"error": "invalid_control"}, status_code=400)
        async with state.lock:
            if state.in_flight:
                return JSONResponse({"error": "requests_in_flight"}, status_code=409)
            state.modes = list(modes)
            state.delay_seconds = float(delay)
            state.max_calls = maximum
            state.calls = 0
            state.counts.clear()
        return JSONResponse({"status": "configured"})

    async def admit() -> str | Response:
        async with state.lock:
            if state.calls >= state.max_calls:
                return JSONResponse(
                    {"error": {"type": "insufficient_quota", "code": "insufficient_quota"}},
                    status_code=429,
                )
            mode = state.modes.pop(0) if state.modes else "success"
            state.calls += 1
            state.counts[mode] += 1
            state.in_flight += 1
            return mode

    async def fail(mode: str) -> Response | None:
        if mode == "timeout":
            await asyncio.sleep(state.delay_seconds)
        if mode == "rate_limit":
            return JSONResponse(
                {
                    "error": {
                        "message": "Synthetic rate limit",
                        "type": "rate_limit_exceeded",
                        "code": "rate_limit_exceeded",
                    }
                },
                status_code=429,
                headers={"Retry-After": "1"},
            )
        if mode == "server_error":
            return JSONResponse({"error": {"code": "server_error"}}, status_code=500)
        return None

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> Response:
        # Never retains or echoes inputs; vectors depend only on the words they contain.
        value = await bounded_json(request, EMBEDDING_BODY_LIMIT)
        inputs = value.get("input") if value else None
        dimensions = value.get("dimensions", EMBEDDING_DIMENSIONS) if value else None
        if (
            not isinstance(inputs, list)
            or not 1 <= len(inputs) <= 2048
            or any(not isinstance(item, str) for item in inputs)
            or type(dimensions) is not int
            or not 1 <= dimensions <= 3072
        ):
            return JSONResponse({"error": {"code": "invalid_request_error"}}, status_code=400)
        admitted = await admit()
        if isinstance(admitted, Response):
            return admitted
        try:
            failure = await fail(admitted)
            if failure is not None:
                return failure
            vectors = [embed_text(item, dimensions) for item in inputs]
            if admitted == "invalid":
                vectors = vectors[:-1]
            return JSONResponse(
                {
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": index, "embedding": vector}
                        for index, vector in enumerate(vectors)
                    ],
                    "model": "simulator-embedding",
                    "usage": {
                        "prompt_tokens": sum(len(WORD.findall(item)) for item in inputs),
                        "total_tokens": sum(len(WORD.findall(item)) for item in inputs),
                    },
                },
                headers={"x-request-id": f"req_sim_{uuid4().hex}"},
            )
        finally:
            async with state.lock:
                state.in_flight -= 1

    @app.post("/v1/responses")
    async def responses(request: Request) -> Response:
        # Parse the real request without retaining or echoing prompts, instructions, or headers.
        value = await bounded_json(request)
        if value is None:
            return JSONResponse({"error": {"code": "invalid_request_error"}}, status_code=400)
        admitted = await admit()
        if isinstance(admitted, Response):
            return admitted
        mode = admitted
        try:
            output = grounded_answer(value) if wants_grounded_answer(value) else SUGGESTION
            if mode == "uncited":
                output = {"answer": "Unsupported synthetic claim.", "citations": []}
            failure = await fail(mode)
            if failure is not None:
                return failure
            content = (
                [{"type": "refusal", "refusal": "Synthetic refusal"}]
                if mode == "refusal"
                else [
                    {
                        "type": "output_text",
                        "text": "not valid JSON" if mode == "invalid" else json.dumps(output),
                        "annotations": [],
                        "logprobs": [],
                    }
                ]
            )
            return JSONResponse(
                {
                    "id": f"resp_sim_{uuid4().hex}",
                    "object": "response",
                    "created_at": int(time.time()),
                    "status": "incomplete" if mode == "incomplete" else "completed",
                    "model": "simulator-item-rewrite",
                    "error": None,
                    "incomplete_details": {"reason": "max_output_tokens"}
                    if mode == "incomplete"
                    else None,
                    "output": [
                        {
                            "id": f"msg_sim_{uuid4().hex}",
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": content,
                        }
                    ],
                    "usage": {
                        "input_tokens": 25,
                        "output_tokens": 15,
                        "total_tokens": 40,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens_details": {"reasoning_tokens": 0},
                    },
                    "parallel_tool_calls": False,
                    "tools": [],
                    "tool_choice": "none",
                    "store": False,
                },
                headers={"x-request-id": f"req_sim_{uuid4().hex}"},
            )
        finally:
            async with state.lock:
                state.in_flight -= 1

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    with suppress(KeyboardInterrupt):
        uvicorn.run(app, host=args.host, port=args.port, access_log=False, log_level="critical")


if __name__ == "__main__":
    main()
