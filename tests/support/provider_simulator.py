"""Bounded, synthetic OpenAI Responses HTTP simulator; never contacts a provider."""

import argparse
import asyncio
import json
import time
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass, field
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

MODES = frozenset(
    {"success", "refusal", "invalid", "incomplete", "timeout", "rate_limit", "server_error"}
)
SUGGESTION = {"title": "Clear synthetic title", "description": "A clear synthetic description."}


@dataclass
class SimulatorState:
    modes: list[str] = field(default_factory=list)
    delay_seconds: float = 5.0
    max_calls: int = 100
    calls: int = 0
    in_flight: int = 0
    counts: Counter = field(default_factory=Counter)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


async def bounded_json(request: Request) -> dict | None:
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 65536:
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

    @app.post("/v1/responses")
    async def responses(request: Request) -> Response:
        # Parse the real request without retaining or echoing prompts, instructions, or headers.
        value = await bounded_json(request)
        if value is None:
            return JSONResponse({"error": {"code": "invalid_request_error"}}, status_code=400)
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
            delay = state.delay_seconds
        try:
            if mode == "timeout":
                await asyncio.sleep(delay)
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
            content = (
                [{"type": "refusal", "refusal": "Synthetic refusal"}]
                if mode == "refusal"
                else [
                    {
                        "type": "output_text",
                        "text": "not valid JSON" if mode == "invalid" else json.dumps(SUGGESTION),
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
