"""Bounded, synthetic OpenAI Responses HTTP simulator; never contacts a provider."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
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
        # Wait until the test sets `release`, then answer as success.
        "hold",
        "rate_limit",
        "server_error",
        # RAG answers: no claims, a quote absent from the passage, a number absent from it.
        "uncited",
        "misquote",
        "wrong_number",
        # Rerank: keep no passage.
        "unranked",
        # Verifier: reject the first claim.
        "unsupported",
        # A reasoning item precedes the message, as with a reasoning effort above none.
        "reasoning",
    }
)
SUGGESTION = {"title": "Clear synthetic title", "description": "A clear synthetic description."}
EMBEDDING_DIMENSIONS = 1536
# Embedding requests carry up to 64 chunks of about 1200 characters each.
EMBEDDING_BODY_LIMIT = 2 * 1024 * 1024
WORD = re.compile(r"\w+")
# Agent requests carry tool schemas, passages and resident context.
RESPONSES_BODY_LIMIT = 256 * 1024
SCRIPT_LIMIT = 50
TOOL_NAMES = frozenset(
    {
        "search_documents",
        "list_bookings",
        "create_ticket",
        "set_thermostat",
        "lock_door",
        "unlock_door",
        "remember",
        "forget",
    }
)
NUMBER = re.compile(r"\d+(?:\.\d+)?")
TICKET_WORDS = ("ticket", "broken", "leak", "repair")
REMEMBER_WORD = "remember"
# The server's quote check needs at least three words.
MIN_NOTE_WORDS = 3
RECALL_WORDS = ("what did i tell", "recall")
THERMOSTAT_WORDS = ("degree", "thermostat", "temperature")
# Benchmark knobs, read once at startup. Unset: no added latency and a cap of
# 100 calls, as the tests expect.
LATENCY_ENV = "SIMULATOR_LATENCY_MS"
EMBEDDING_LATENCY_ENV = "SIMULATOR_EMBEDDING_LATENCY_MS"
MAX_CALLS_ENV = "SIMULATOR_MAX_CALLS"
DEFAULT_MAX_CALLS = 100
MAX_LATENCY_MS = 60_000


def latency_range(name: str) -> tuple[float, float]:
    """Seconds from "N" or "LOW-HIGH" milliseconds in the environment; unset is none."""
    value = os.environ.get(name, "").strip()
    if not value:
        return 0.0, 0.0
    low, _, high = value.partition("-")
    bounds = (float(low), float(high or low))
    if bounds[0] > bounds[1] or not all(
        math.isfinite(bound) and 0 <= bound <= MAX_LATENCY_MS for bound in bounds
    ):
        raise ValueError(f"{name} must be N or LOW-HIGH milliseconds, 0 to {MAX_LATENCY_MS}")
    return bounds[0] / 1000, bounds[1] / 1000


def initial_max_calls() -> int:
    value = int(os.environ.get(MAX_CALLS_ENV, DEFAULT_MAX_CALLS))
    if value < 1:
        raise ValueError(f"{MAX_CALLS_ENV} must be a positive integer")
    return value


async def pause(bounds: tuple[float, float]) -> None:
    # Outside state.lock: concurrent requests wait in parallel, as with a real provider.
    low, high = bounds
    if high > 0:
        await asyncio.sleep(random.uniform(low, high))


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


# Words of the first passage the synthetic claim quotes.
QUOTE_WORDS = 8
# Quoted from no tool fact: a final claim citing it grounds nothing.
UNGROUNDED_QUOTE = "the request is done"


def quoted_answer(request: dict, mode: str) -> dict:
    """One claim quoting the first words of P1; no passages or "uncited" give no claims."""
    try:
        passages = json.loads(request.get("input", "")).get("passages", [])
    except (ValueError, AttributeError):
        passages = []
    first = passages[0] if isinstance(passages, list) and passages else None
    if mode == "uncited" or not isinstance(first, dict) or not isinstance(first.get("text"), str):
        return {"claims": []}
    quote = " ".join(first["text"].split()[:QUOTE_WORDS])
    statement = quote
    if mode == "misquote":
        quote = "words that appear in no passage"
    elif mode == "wrong_number":
        statement = f"{quote} 9999"
    return {"claims": [{"statement": statement, "citation": "P1", "quote": quote}]}


def _schema_properties(request: dict) -> dict:
    text_format = request.get("text", {}).get("format", {})
    properties = text_format.get("schema", {}).get("properties", {})
    return properties if isinstance(properties, dict) else {}


def wants_claims(request: dict) -> bool:
    return "claims" in _schema_properties(request)


def wants_claim_verdicts(request: dict) -> bool:
    return "verdicts" in _schema_properties(request)


def claim_verdicts(request: dict, mode: str) -> dict:
    """Support every claim label in the request; "unsupported" rejects the first."""
    try:
        claims = json.loads(request.get("input", "")).get("claims", [])
    except (ValueError, AttributeError):
        claims = []
    labels = [item.get("label") for item in claims if isinstance(item, dict)]
    verdicts = [{"claim": label, "supported": True} for label in labels if isinstance(label, str)]
    if mode == "unsupported" and verdicts:
        verdicts[0]["supported"] = False
    return {"verdicts": verdicts}


def wants_graph(request: dict) -> bool:
    return {"nodes", "edges"} <= set(_schema_properties(request))


# Graph extraction: "The <source> is <relation> the <target>." sentences become an edge.
# Types come from the last word of a name; the server checks the ontology, not this.
GRAPH_SENTENCE = re.compile(
    r"^(?:the\s+)?(?P<source>[a-z][\w' -]*?)\s+(?:is|are)\s+"
    r"(?P<relation>located in|part of|operated by|governed by|scheduled on|the contact for)"
    r"\s+(?:the\s+)?(?P<target>[a-z][\w' -]*?)$",
    re.IGNORECASE,
)
GRAPH_RELATIONS = {
    "located in": "located_in",
    "part of": "part_of",
    "operated by": "operated_by",
    "governed by": "governed_by",
    "scheduled on": "scheduled_on",
    "the contact for": "contact_for",
}
GRAPH_TYPES = {
    "floor": "location",
    "basement": "location",
    "garage": "location",
    "level": "location",
    "wing": "location",
    "courtyard": "location",
    "lobby": "location",
    "building": "building",
    "tower": "building",
    "rules": "rule",
    "rule": "rule",
    "policy": "rule",
    "regulations": "rule",
    "weekdays": "schedule",
    "weekends": "schedule",
    "schedule": "schedule",
    "concierge": "contact_role",
    "caretaker": "contact_role",
    "manager": "contact_role",
    "office": "contact_role",
    "service": "service",
    "company": "service",
    "team": "service",
    "contractor": "service",
    "lift": "device",
    "elevator": "device",
    "boiler": "device",
    "thermostat": "device",
}


def _graph_type(name: str) -> str:
    return GRAPH_TYPES.get(name.lower().split()[-1], "facility")


def graph_extraction(request: dict, mode: str) -> dict:
    """Edges from matching sentences of every chunk; "misquote" quotes no chunk."""
    try:
        chunks = json.loads(request.get("input", "")).get("chunks", [])
    except (ValueError, AttributeError):
        chunks = []
    nodes, edges = [], []
    for item in chunks if isinstance(chunks, list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", item["text"]):
            match = GRAPH_SENTENCE.match(sentence.strip().rstrip(".!?"))
            if match is None:
                continue
            source, target = match["source"].strip(), match["target"].strip()
            quote = sentence.strip().rstrip(".!?")
            if mode == "misquote":
                quote = "words that appear in no chunk"
            for name in (source, target):
                nodes.append({"chunk": item["label"], "type": _graph_type(name), "name": name})
            edges.append(
                {
                    "chunk": item["label"],
                    "source": source,
                    "source_type": _graph_type(source),
                    "relation": GRAPH_RELATIONS[match["relation"].lower()],
                    "target": target,
                    "target_type": _graph_type(target),
                    "quote": quote,
                }
            )
    return {"nodes": nodes, "edges": edges}


def wants_ranking(request: dict) -> bool:
    return "passages" in _schema_properties(request)


def ranking(request: dict, mode: str) -> dict:
    """Keep the passages in their given order, up to the request's limit."""
    try:
        payload = json.loads(request.get("input", ""))
    except (ValueError, AttributeError):
        payload = {}
    passages = payload.get("passages", []) if isinstance(payload, dict) else []
    limit = payload.get("limit", 0) if isinstance(payload, dict) else 0
    if mode == "unranked" or not isinstance(passages, list) or type(limit) is not int:
        return {"passages": []}
    labels = [item.get("label") for item in passages if isinstance(item, dict)]
    return {"passages": [label for label in labels if isinstance(label, str)][:limit]}


def wants_verdict(request: dict) -> bool:
    return "grounded" in _schema_properties(request)


def valid_step(step) -> bool:
    """A scripted agent step: one tool call, or one final reply of claims."""
    if not isinstance(step, dict):
        return False
    if set(step) == {"claims"}:
        claims = step["claims"]
        return isinstance(claims, list) and all(
            isinstance(claim, dict)
            and set(claim) == {"statement", "source", "quote"}
            and all(isinstance(value, str) for value in claim.values())
            for claim in claims
        )
    if "tool" not in step or not isinstance(step["tool"], str) or len(step["tool"]) > 64:
        return False
    rest = set(step) - {"tool"}
    if rest == {"arguments"}:
        return isinstance(step["arguments"], dict)
    # Raw text lets tests send malformed argument JSON.
    return rest == {"raw_arguments"} and isinstance(step["raw_arguments"], str)


def agent_context(request: dict) -> tuple[dict, list[dict]]:
    """The resident context of the first input item, and every parsed tool output."""
    items = request.get("input")
    if not isinstance(items, list) or not items or not isinstance(items[0], dict):
        return {}, []
    try:
        context = json.loads(items[0].get("content", ""))
    except (TypeError, ValueError):
        context = {}
    outputs = []
    for item in items[1:]:
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            try:
                outputs.append(json.loads(item.get("output", "")))
            except (TypeError, ValueError):
                continue
    return (context if isinstance(context, dict) else {}), outputs


def _device(devices: list, kind: str, message: str) -> str:
    candidates = [d for d in devices if isinstance(d, dict) and d.get("kind") == kind]
    for device in candidates:
        if str(device.get("name", "")).lower() in message:
            return device["name"]
    return candidates[0]["name"] if candidates else "unknown"


def _remember_step(message: str) -> dict:
    """remember with the words after "remember" (and "that"), or the whole message."""
    start = message.lower().index(REMEMBER_WORD) + len(REMEMBER_WORD)
    text = message[start:].strip(" :,")
    if text.lower().startswith("that "):
        text = text[5:]
    if len(text.split()) < MIN_NOTE_WORDS:
        text = message
    scope = "residence" if "this residence" in message.lower() else "resident"
    return {"tool": "remember", "arguments": {"text": text[:300], "scope": scope}}


def _memory_step(context: dict, message: str) -> dict | None:
    """Memory keywords come first: "forget" and "remember" may sit beside "lock"."""
    lowered = message.lower()
    memories = context.get("memories")
    memories = [m for m in memories if isinstance(m, dict)] if isinstance(memories, list) else []
    if "forget" in lowered and memories:
        arguments = {"label": memories[0].get("label", "M1"), "request_quote": message[:300]}
        return {"tool": "forget", "arguments": arguments}
    if REMEMBER_WORD in lowered:
        return _remember_step(message)
    if any(word in lowered for word in RECALL_WORDS) and memories:
        first = memories[0]
        quote = " ".join(str(first.get("text", "")).split()[:QUOTE_WORDS])
        claim = {"statement": quote, "source": first.get("label", "M1"), "quote": quote}
        return {"claims": [claim]}
    return None


def keyword_step(request: dict) -> dict:
    """Toy policy for unscripted agent requests: one tool from keywords, then an answer."""
    context, outputs = agent_context(request)
    if outputs:
        for output in outputs:
            passages = output.get("passages") if isinstance(output, dict) else None
            first = passages[0] if isinstance(passages, list) and passages else None
            if isinstance(first, dict) and isinstance(first.get("text"), str):
                quote = " ".join(first["text"].split()[:QUOTE_WORDS])
                label = first.get("label", "P1")
                return {"claims": [{"statement": quote, "source": label, "quote": quote}]}
        # A labelled tool result: claim its server-written fact, quoting its first words.
        for output in outputs:
            if not isinstance(output, dict):
                continue
            label, fact = output.get("label"), output.get("fact")
            if isinstance(label, str) and isinstance(fact, str):
                quote = " ".join(fact.split()[:QUOTE_WORDS])
                return {"claims": [{"statement": quote, "source": label, "quote": quote}]}
        # Nothing labelled, such as a denial: a claim that no fact supports.
        claim = {"statement": "Done.", "source": "T1", "quote": UNGROUNDED_QUOTE}
        return {"claims": [claim]}
    memory = _memory_step(context, str(context.get("message", "")))
    if memory is not None:
        return memory
    message = str(context.get("message", "")).lower()
    resident = context.get("resident")
    devices = resident.get("devices", []) if isinstance(resident, dict) else []
    if "unlock" in message:
        return {
            "tool": "unlock_door",
            "arguments": {"device": _device(devices, "door_lock", message)},
        }
    if "lock" in message:
        return {
            "tool": "lock_door",
            "arguments": {"device": _device(devices, "door_lock", message)},
        }
    if any(word in message for word in THERMOSTAT_WORDS):
        number = NUMBER.search(message)
        celsius = float(number.group()) if number else 21.0
        arguments = {"device": _device(devices, "thermostat", message), "celsius": celsius}
        return {"tool": "set_thermostat", "arguments": arguments}
    if "booking" in message:
        return {"tool": "list_bookings", "arguments": {}}
    if any(word in message for word in TICKET_WORDS):
        summary = "Synthetic maintenance request"
        return {"tool": "create_ticket", "arguments": {"category": "other", "summary": summary}}
    return {"tool": "search_documents", "arguments": {"question": message[:500] or "building"}}


def agent_item(step: dict) -> dict:
    if "tool" in step:
        arguments = step.get("raw_arguments")
        if arguments is None:
            arguments = json.dumps(step["arguments"])
        return {
            "id": f"fc_sim_{uuid4().hex}",
            "type": "function_call",
            "status": "completed",
            "call_id": f"call_sim_{uuid4().hex}",
            "name": step["tool"],
            "arguments": arguments,
        }
    return message_item(
        [
            {
                "type": "output_text",
                "text": json.dumps({"claims": step["claims"]}),
                "annotations": [],
                "logprobs": [],
            }
        ]
    )


def message_item(content: list) -> dict:
    return {
        "id": f"msg_sim_{uuid4().hex}",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": content,
    }


@dataclass
class SimulatorState:
    script: list[dict] = field(default_factory=list)
    tool_counts: Counter = field(default_factory=Counter)
    modes: list[str] = field(default_factory=list)
    delay_seconds: float = 5.0
    max_calls: int = DEFAULT_MAX_CALLS
    # Added to every admitted call, in seconds: (low, high).
    latency: tuple[float, float] = (0.0, 0.0)
    embedding_latency: tuple[float, float] = (0.0, 0.0)
    calls: int = 0
    in_flight: int = 0
    counts: Counter = field(default_factory=Counter)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Opens every "hold" call; tests set it once their checks ran.
    release: asyncio.Event = field(default_factory=asyncio.Event)


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
    state = SimulatorState(
        max_calls=initial_max_calls(),
        latency=latency_range(LATENCY_ENV),
        embedding_latency=latency_range(EMBEDDING_LATENCY_ENV),
    )
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
                "tool_counts": dict(state.tool_counts),
                "max_calls": state.max_calls,
            }

    @app.post("/__control")
    async def control(request: Request) -> Response:
        value = await bounded_json(request)
        if value is None or set(value) - {"modes", "delay_seconds", "max_calls", "script"}:
            return JSONResponse({"error": "invalid_control"}, status_code=400)
        modes = value.get("modes", [])
        delay = value.get("delay_seconds", 5.0)
        maximum = value.get("max_calls", 100)
        script = value.get("script", [])
        if (
            not isinstance(script, list)
            or len(script) > SCRIPT_LIMIT
            or not all(valid_step(step) for step in script)
        ):
            return JSONResponse({"error": "invalid_control"}, status_code=400)
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
            state.script = list(script)
            state.calls = 0
            state.counts.clear()
            state.tool_counts.clear()
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
        if mode == "hold":
            await state.release.wait()
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
            await pause(state.embedding_latency)
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

    async def agent_response(value: dict, mode: str) -> Response:
        """Scripted step if one is queued, else the keyword policy. Modes still apply."""
        async with state.lock:
            step = state.script.pop(0) if state.script else None
        step = step or keyword_step(value)
        failure = await fail(mode)
        if failure is not None:
            return failure
        if mode == "refusal":
            items = [message_item([{"type": "refusal", "refusal": "Synthetic refusal"}])]
        elif mode == "invalid":
            text = {
                "type": "output_text",
                "text": "not valid JSON",
                "annotations": [],
                "logprobs": [],
            }
            items = [message_item([text])]
        else:
            items = [agent_item(step)]
            if "tool" in step:
                name = step["tool"] if step["tool"] in TOOL_NAMES else "other"
                async with state.lock:
                    state.tool_counts[name] += 1
        return JSONResponse(
            {
                "id": f"resp_sim_{uuid4().hex}",
                "object": "response",
                "created_at": int(time.time()),
                "status": "incomplete" if mode == "incomplete" else "completed",
                "model": "simulator-assistant",
                "error": None,
                "incomplete_details": {"reason": "max_output_tokens"}
                if mode == "incomplete"
                else None,
                "output": items,
                "usage": {
                    "input_tokens": 40,
                    "output_tokens": 20,
                    "total_tokens": 60,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
                "parallel_tool_calls": False,
                "tools": value.get("tools"),
                "tool_choice": "auto",
                "store": False,
            },
            headers={"x-request-id": f"req_sim_{uuid4().hex}"},
        )

    @app.post("/v1/responses")
    async def responses(request: Request) -> Response:
        # Parse the real request without retaining or echoing prompts, instructions, or headers.
        value = await bounded_json(request, RESPONSES_BODY_LIMIT)
        if value is None:
            return JSONResponse({"error": {"code": "invalid_request_error"}}, status_code=400)
        admitted = await admit()
        if isinstance(admitted, Response):
            return admitted
        mode = admitted
        try:
            await pause(state.latency)
            if value.get("tools"):
                return await agent_response(value, mode)
            if wants_verdict(value):
                output = {"grounded": True, "answers_question": True}
            elif wants_claim_verdicts(value):
                output = claim_verdicts(value, mode)
            elif wants_claims(value):
                output = quoted_answer(value, mode)
            elif wants_graph(value):
                output = graph_extraction(value, mode)
            elif wants_ranking(value):
                output = ranking(value, mode)
            else:
                output = SUGGESTION
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
                        *(
                            [{"id": f"rs_sim_{uuid4().hex}", "type": "reasoning", "summary": []}]
                            if mode == "reasoning"
                            else []
                        ),
                        {
                            "id": f"msg_sim_{uuid4().hex}",
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": content,
                        },
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
