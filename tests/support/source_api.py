"""A small JSON source for connector tests and the local `source-api` Compose fake.

    GET    /records?page=N          page mode, PAGE_SIZE records per page, [] past the end
    GET    /cursor?cursor=C         cursor mode, {"data": [...], "next": C | null}
    GET    /redirect                302 to /records (the connector must not follow)
    GET    /large                   a body past the connector's response cap
    GET    /gzip                    a small gzip body that inflates past the cap
    GET    /gzip-bomb               a gzip body that inflates far past the cap
    GET    /brotli                  a body declared brotli (the connector refuses it)
    GET    /surrogate               a record whose id is a lone surrogate
    GET    /status/{code}           an empty answer with that status
    PUT    /control/records/{id}    add or change a record: {"title": ..., "content": ...}
    DELETE /control/records/{id}    remove a record

Records live in memory; the Compose fake starts with SEED_RECORDS.
"""

import gzip
import json
from collections import deque

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse

PAGE_SIZE = 2
# Past sources.MAX_RESPONSE_BYTES (5 MiB).
OVERSIZE_BYTES = 6 * 1024 * 1024
SEED_RECORDS = {
    "notice-1": {"title": "Water shutoff", "content": "Water is off on Monday from 9 to 11."},
    "notice-2": {"title": "Lift service", "content": "The lift is serviced on Wednesday."},
    "notice-3": {"title": "Recycling", "content": "Glass goes in the green bin."},
}


def make_source_app(records: dict[str, dict] | None = None) -> FastAPI:
    app = FastAPI()
    app.state.records = dict(SEED_RECORDS if records is None else records)
    # Bounded: the Compose fake's health check calls it every 2 s.
    app.state.requests = deque(maxlen=200)

    def items() -> list[dict]:
        return [{"id": key, **value} for key, value in sorted(app.state.records.items())]

    @app.middleware("http")
    async def remember(request: Request, call_next):
        # Tests check what the connector sent (path, query, auth header).
        app.state.requests.append(
            (request.url.path, dict(request.query_params), request.headers.get("authorization"))
        )
        return await call_next(request)

    @app.get("/records")
    async def page(page: int = 1) -> list[dict]:
        start = (page - 1) * PAGE_SIZE
        return items()[start : start + PAGE_SIZE]

    @app.get("/cursor")
    async def cursor(cursor: int = 0) -> dict:
        chunk = items()[cursor : cursor + PAGE_SIZE]
        following = cursor + PAGE_SIZE
        return {"data": chunk, "next": following if following < len(items()) else None}

    @app.get("/redirect")
    async def redirect() -> Response:
        return RedirectResponse("/records", status_code=302)

    @app.get("/large")
    async def large() -> Response:
        return Response(b"[" + b" " * OVERSIZE_BYTES + b"]", media_type="application/json")

    @app.get("/gzip")
    async def bomb() -> Response:
        body = gzip.compress(b"[" + b" " * OVERSIZE_BYTES + b"]")
        return Response(body, media_type="application/json", headers={"Content-Encoding": "gzip"})

    @app.get("/gzip-bomb")
    async def gzip_bomb() -> Response:
        body = gzip.compress(b" " * (64 * 1024 * 1024))
        return Response(body, media_type="application/json", headers={"Content-Encoding": "gzip"})

    @app.get("/gzip-trailing")
    async def gzip_trailing() -> Response:
        # A valid member, then junk that zlib would keep as unused data.
        body = gzip.compress(b"[]") + b"\x00" * (1024 * 1024)
        return Response(body, media_type="application/json", headers={"Content-Encoding": "gzip"})

    @app.get("/gzip-truncated")
    async def gzip_truncated() -> Response:
        body = gzip.compress(b"[]")[:-4]
        return Response(body, media_type="application/json", headers={"Content-Encoding": "gzip"})

    @app.get("/brotli")
    async def brotli() -> Response:
        return Response(b"\x0b\x01\x80[]\x03", headers={"Content-Encoding": "br"})

    @app.get("/surrogate")
    async def surrogate() -> Response:
        body = (
            b'[{"id": "\\ud800", "title": "T", "content": "C"},'
            b' {"id": "ok", "title": "T", "content": "C"}]'
        )
        return Response(body, media_type="application/json")

    @app.get("/status/{code}")
    async def status(code: int) -> Response:
        return Response(status_code=code)

    @app.put("/control/records/{record_id}")
    async def put_record(record_id: str, request: Request) -> JSONResponse:
        body = json.loads(await request.body())
        app.state.records[record_id] = {"title": body["title"], "content": body["content"]}
        return JSONResponse({"id": record_id})

    @app.delete("/control/records/{record_id}", status_code=204)
    async def delete_record(record_id: str) -> Response:
        app.state.records.pop(record_id, None)
        return Response(status_code=204)

    return app


# The Compose fake: `uvicorn tests.support.source_api:app`.
app = make_source_app()
