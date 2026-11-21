"""Fetch records from a connector's source. No database transaction of ours is open here.

A fetch is `complete` only when every page answered and no cap stopped it; only a
complete fetch may delete documents. Errors are fixed codes: exception text can carry a
DSN, URL, query or record content, so it is never logged or stored.
"""

import asyncio
import json
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import asyncpg
import httpx

from app.rag.connectors.config import HTTPSource, PostgresSource

SOURCE_TIMEOUT_SECONDS = 10
# The whole fetch, all pages together.
FETCH_DEADLINE_SECONDS = 120
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
# One fetch, all database rows or all HTTP pages together.
# One row's content is capped at MAX_RESPONSE_BYTES.
MAX_FETCH_BYTES = 64 * 1024 * 1024
# Rows asyncpg reads ahead: at most MAX_FETCH_BYTES of maximal rows are buffered.
CURSOR_PREFETCH = max(1, MAX_FETCH_BYTES // MAX_RESPONSE_BYTES)
# gzip window bits: header and trailer expected.
GZIP_WBITS = 16 + zlib.MAX_WBITS
MAX_ID_CHARACTERS = 256
MAX_CURSOR_CHARACTERS = 1024


@dataclass(frozen=True)
class Record:
    external_id: str
    title: object
    content: object


@dataclass
class Fetch:
    records: list[Record] = field(default_factory=list)
    complete: bool = False
    # Why the fetch failed, or why it is incomplete: a fixed code.
    error_code: str | None = None


def _external_id(value: object) -> str | None:
    # bool is an int subclass; True is not an identifier.
    if isinstance(value, bool) or not isinstance(value, str | int):
        return None
    text = str(value)
    if not 1 <= len(text) <= MAX_ID_CHARACTERS or "\x00" in text:
        return None
    try:
        # A lone surrogate from JSON cannot be stored or compared in PostgreSQL.
        text.encode("utf-8", "strict")
    except UnicodeEncodeError:
        return None
    return text


async def fetch_postgres(
    source: PostgresSource, max_records: int, environ: Mapping[str, str]
) -> Fetch:
    dsn = environ.get(source.dsn_env)
    if not dsn:
        return Fetch(error_code="dsn_missing")
    # The query is operator config; the wrapper fixes columns, row cap and per-row size.
    # An oversize id or content arrives as NULL: the id is invalid, the record skipped.
    statement = (
        "SELECT "
        f"CASE WHEN char_length(external_id::text) <= {MAX_ID_CHARACTERS} "
        "THEN external_id::text END AS external_id, "
        f"CASE WHEN char_length(title::text) <= {MAX_ID_CHARACTERS} "
        "THEN title::text END AS title, "
        f"CASE WHEN octet_length(content::text) <= {MAX_RESPONSE_BYTES} "
        "THEN content::text END AS content "
        f"FROM ({source.query}) AS source LIMIT {max_records + 1}"
    )
    try:
        connection = await asyncpg.connect(
            dsn, timeout=SOURCE_TIMEOUT_SECONDS, command_timeout=SOURCE_TIMEOUT_SECONDS
        )
    except (OSError, TimeoutError, ValueError, asyncpg.PostgresError, asyncpg.InterfaceError):
        # ValueError: a DSN that does not parse.
        return Fetch(error_code="source_unreachable")
    try:
        # The whole fetch, all cursor round trips together, like an HTTP fetch.
        async with asyncio.timeout(FETCH_DEADLINE_SECONDS):
            async with connection.transaction(readonly=True):
                await connection.execute(
                    f"SET LOCAL statement_timeout = '{SOURCE_TIMEOUT_SECONDS}s'"
                )
                rows = await _bounded_rows(connection, statement)
    except _SourceError as error:
        return Fetch(error_code=error.code)
    except (
        OSError,
        TimeoutError,
        UnicodeDecodeError,
        asyncpg.PostgresError,
        asyncpg.InterfaceError,
    ):
        # UnicodeDecodeError: asyncpg decodes text as strict UTF-8.
        return Fetch(error_code="source_query_failed")
    finally:
        try:
            await connection.close(timeout=5)
        except (OSError, TimeoutError, asyncpg.InterfaceError):
            connection.terminate()
    fetch = Fetch(complete=True)
    for row in rows[:max_records]:
        external_id = _external_id(row["external_id"])
        if external_id is None:
            # An unusable id cannot be matched, so nothing may be deleted this run.
            fetch.complete, fetch.error_code = False, "invalid_record_id"
            continue
        fetch.records.append(Record(external_id, row["title"], row["content"]))
    if len(rows) > max_records:
        fetch.complete, fetch.error_code = False, "record_limit"
    return fetch


async def _bounded_rows(connection, statement: str) -> list:
    """Rows through a cursor, stopping once their text passes MAX_FETCH_BYTES."""
    rows, total = [], 0
    async for row in connection.cursor(
        statement, prefetch=CURSOR_PREFETCH, timeout=SOURCE_TIMEOUT_SECONDS
    ):
        total += sum(len(value.encode()) for value in row.values() if isinstance(value, str))
        if total > MAX_FETCH_BYTES:
            raise _SourceError("source_too_large")
        rows.append(row)
    return rows


# A path that is absent, as opposed to present with a JSON null.
_MISSING = object()


def _at(data: Any, path: str) -> Any:
    for part in path.split(".") if path else ():
        if not isinstance(data, dict) or part not in data:
            return _MISSING
        data = data[part]
    return data


def _inflate(inflater, chunk: bytes, body: bytearray) -> None:
    """Inflate at most one byte past the cap per call, so memory stays bounded.

    One gzip member only: bytes after its end (junk or a second member) are refused,
    since zlib would keep them as unused data outside the cap.
    """
    while chunk:
        if inflater.eof:
            raise _SourceError("source_malformed")
        body.extend(inflater.decompress(chunk, MAX_RESPONSE_BYTES + 1 - len(body)))
        if len(body) > MAX_RESPONSE_BYTES:
            raise _SourceError("source_too_large")
        if inflater.unused_data:
            raise _SourceError("source_malformed")
        chunk = inflater.unconsumed_tail


async def _get_json(client: httpx.AsyncClient, url: httpx.URL, headers: dict) -> tuple[Any, int]:
    """One GET: 200 only, identity or gzip decoded here under the cap, then JSON and size."""
    async with client.stream("GET", url, headers=headers) as response:
        if response.status_code != 200:
            raise _SourceError("source_http_status")
        encoding = response.headers.get("content-encoding", "identity").strip().lower()
        if encoding not in {"identity", "gzip"}:
            # httpx would decode brotli or zstd whole before any cap applies.
            raise _SourceError("source_malformed")
        inflater = zlib.decompressobj(GZIP_WBITS) if encoding == "gzip" else None
        body = bytearray()
        try:
            async for chunk in response.aiter_raw():
                if inflater is not None:
                    _inflate(inflater, chunk, body)
                    continue
                body.extend(chunk)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise _SourceError("source_too_large")
        except zlib.error:
            raise _SourceError("source_malformed") from None
        if inflater is not None and not inflater.eof:
            # Cut short: the gzip trailer (and its CRC check) never arrived.
            raise _SourceError("source_malformed")
    try:
        return json.loads(body), len(body)
    except (ValueError, RecursionError):
        raise _SourceError("source_malformed") from None


class _SourceError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


async def fetch_http(
    source: HTTPSource,
    max_records: int,
    environ: Mapping[str, str],
    transport: httpx.AsyncBaseTransport | None = None,
) -> Fetch:
    headers = {"Accept": "application/json", "Accept-Encoding": "gzip"}
    if source.auth_header_env is not None:
        value = environ.get(source.auth_header_env)
        if not value:
            return Fetch(error_code="auth_missing")
        headers["Authorization"] = value
    fetch = Fetch()
    # No proxies or netrc from the environment, no redirects: the configured URL only.
    async with httpx.AsyncClient(
        timeout=SOURCE_TIMEOUT_SECONDS,
        follow_redirects=False,
        trust_env=False,
        transport=transport,
    ) as client:
        try:
            async with asyncio.timeout(FETCH_DEADLINE_SECONDS):
                await _pages(client, source, max_records, headers, fetch)
        except _SourceError as error:
            return Fetch(error_code=error.code)
        except (httpx.HTTPError, httpx.InvalidURL, TimeoutError):
            # InvalidURL is not an HTTPError: a bad port would end the service.
            return Fetch(error_code="source_unreachable")
    return fetch


async def _pages(client, source: HTTPSource, max_records: int, headers: dict, fetch: Fetch):
    paging = source.pagination
    cursor: str | None = None
    base = httpx.URL(source.url)
    # All pages together stay under the database source's total bound.
    total = 0
    for page in range(1, source.max_pages + 1):
        params: dict[str, str] = {}
        if paging.mode == "page":
            params[paging.page_param] = str(page)
        elif paging.mode == "cursor" and cursor is not None:
            # A value, never a URL: the next request still goes to the configured URL.
            params[paging.cursor_param] = cursor
        # Merged into the URL's own query: httpx's params= would replace it.
        data, size = await _get_json(client, base.copy_merge_params(params), headers)
        total += size
        if total > MAX_FETCH_BYTES:
            raise _SourceError("source_too_large")
        items = _at(data, source.items_path)
        if not isinstance(items, list):
            # A missing list is a failure, never "zero records" (that would delete all).
            raise _SourceError("source_malformed")
        for item in items:
            if len(fetch.records) >= max_records:
                fetch.error_code = "record_limit"
                return
            external_id = (
                _external_id(item.get(source.id_field)) if isinstance(item, dict) else None
            )
            if external_id is None:
                fetch.error_code = "invalid_record_id"
                continue
            fetch.records.append(
                Record(external_id, item.get(source.title_field), item.get(source.content_field))
            )
        if paging.mode == "none" or (paging.mode == "page" and not items):
            break
        if paging.mode == "cursor":
            following = _at(data, paging.cursor_path or "")
            if following is _MISSING:
                # A renamed or mistyped cursor key is not "last page" (that would delete).
                raise _SourceError("source_malformed")
            if following is None or following == "":
                break
            if not isinstance(following, str | int) or isinstance(following, bool):
                raise _SourceError("source_malformed")
            cursor = str(following)
            if len(cursor) > MAX_CURSOR_CHARACTERS:
                raise _SourceError("source_malformed")
    else:
        # Pages left when max_pages ran out: the source may hold more.
        fetch.error_code = "page_limit"
        return
    fetch.complete = fetch.error_code is None
