# Owner-scoped FastAPI CRUD

A local, runnable API with built-in accounts, PostgreSQL persistence, async SQLAlchemy, and Redis detail caching with **at most five seconds of cache staleness** under the assumptions below. An opt-in, isolated [durable item-rewrite stack](#durable-item-rewriting) adds Apache Kafka, an async OpenAI worker, and local OpenTelemetry verification. Python 3.13, uv, Docker Compose, and a reproducible Locust harness are included. The requested 15,000–30,000 concurrent-user target is **not verified capacity**.

> **Local development/measurement only.** PostgreSQL is deliberately pinned to the user-selected prerelease `postgres:19rc1-bookworm`. Accept its prerelease risks; do not treat this stack as a production deployment. HTTP binds to loopback. The optional [TLS edge](#optional-tls-edge) adds local TLS and per-address request limits; it is not a production reverse proxy or abuse protection. There is no deployment automation.

## Start the stack

Prerequisites: Docker with Compose, uv, and a standard-library `python3` for environment initialization. Run commands from the repository root. Project dependencies require Python `>=3.13,<3.14`; do not resolve them with the host's unrelated default interpreter. No Node tooling is needed.

```bash
uv python install 3.13
uv sync --locked
python3 scripts/init_env.py
docker compose up --build -d --wait api
curl --fail http://127.0.0.1:8000/health/ready
```

Open **http://localhost:8000/docs** for interactive OpenAPI documentation. The schema is at `/openapi.json`.

`init_env.py` exclusively creates `.env` with mode `0600`, fresh random database/signing secrets, port `8000`, token lifetime `900` seconds, and the host UID/GID. It leaves an existing file unchanged. Before creating credentials, it creates `artifacts/` as the host user with mode `0700`, or validates the existing path using `lstat`: it must be a nonsymlink directory owned by the current UID with exact mode `0700` and writable by that user. Symlinks, files, foreign-owned directories, and modes such as `0755` or `0777` are rejected, even if writable. Initialization never changes existing contents, permissions, or ownership. Inspect a rejected path and manually correct its ownership/permissions (for example, `chmod 700 artifacts` on your own actual directory), or safely replace it with a private directory before retrying; do not run Docker as root or change unrelated ownership. `.env.example` is a reference, not usable default credentials. Do not commit `.env` or artifacts.

Compose starts PostgreSQL and Redis, runs `alembic upgrade head` through the one-shot `migrate` service, then starts the API after migration succeeds. Migrations do not run inside each API worker. The default is one Uvicorn process, keep-alive 2 seconds, graceful shutdown 15 seconds. `Dockerfile` has a nonroot **runtime** target with locked, non-editable application dependencies and a **test** target that additionally includes dev dependencies, tests, and scripts; the `tools` profile uses the latter with the host UID/GID.

The API publishes `127.0.0.1:${API_PORT:-8000}`. The current normal Compose configuration also publishes PostgreSQL at `127.0.0.1:5433`; Redis has no host port. The isolated benchmark and AI stacks publish no database/cache ports. PostgreSQL's named volume mounts **`/var/lib/postgresql`**, as required by the PG19 image layout (`PGDATA=/var/lib/postgresql/19/docker`). Redis is disposable: 128 MiB, allkeys-LRU eviction, no AOF or RDB persistence. `docker compose down` stops the stack without removing its database volume; do not use `down -v` unless deliberately discarding that project's data.

## Use accounts and items

These shell examples generate an example-only password and capture returned tokens rather than embedding actual credentials. They require the locked environment from setup, not `jq`. Keep tokens out of logs and shell tracing.

```bash
BASE=http://127.0.0.1:8000
EMAIL="engineer-$(uv run --locked python -c 'import uuid; print(uuid.uuid4().hex)')@example.com"
PASSWORD=$(uv run --locked python -c 'import secrets; print(secrets.token_hex(24))')

# JSON registration: 201, user id/email/created_at; no password or hash.
curl --fail-with-body "$BASE/api/v1/auth/register" \
  -H 'Content-Type: application/json' \
  -d "{\"email\":\"$EMAIL\",\"password\":\"$PASSWORD\"}"

# OAuth2 form login: username is the email, not a JSON field named email.
TOKEN=$(curl --fail-with-body "$BASE/api/v1/auth/login" \
  --data-urlencode "username=$EMAIL" --data-urlencode "password=$PASSWORD" \
  | uv run --locked python -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')

# Create: 201 with id, title, description, created_at, updated_at.
ITEM=$(curl --fail-with-body "$BASE/api/v1/items" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"title":"First item","description":"Owned by this account"}')
ITEM_ID=$(printf '%s' "$ITEM" \
  | uv run --locked python -c 'import json,sys; print(json.load(sys.stdin)["id"])')

curl --fail-with-body "$BASE/api/v1/items/$ITEM_ID" -H "Authorization: Bearer $TOKEN"
curl --fail-with-body "$BASE/api/v1/items?limit=20" -H "Authorization: Bearer $TOKEN"
curl --fail-with-body -X PATCH "$BASE/api/v1/items/$ITEM_ID" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"description":"Changed; title stays unchanged"}'
curl --fail-with-body -X DELETE "$BASE/api/v1/items/$ITEM_ID" \
  -H "Authorization: Bearer $TOKEN"
unset TOKEN PASSWORD
```

### HTTP contract

| Method and path | Success | Contract |
| --- | --- | --- |
| `POST /api/v1/auth/register` | 201 | JSON `email`, `password`; returns user metadata. |
| `POST /api/v1/auth/login` | 200 | Form `username`, `password`; returns `access_token`, `token_type="bearer"`, `expires_in`. 429 after repeated wrong passwords for one account. |
| `POST /api/v1/items` | 201 | JSON `title`, optional `description` (default `""`). |
| `GET /api/v1/items/{item_id}` | 200 | Owner-scoped detail; may use cache. |
| `GET /api/v1/items?limit=50&cursor=...` | 200 | Owner-scoped `{ "items": [...], "next_cursor": ... }`; DB-only. |
| `PATCH /api/v1/items/{item_id}` | 200 | Supplied `title` and/or `description` only; returns committed mutation. |
| `DELETE /api/v1/items/{item_id}` | 204 | Empty body; repeating deletion returns 404. |
| `GET /health/live` | 200 | Process liveness, no dependency checks. |
| `GET /health/ready` | 200/503 | Real bounded database/cache checks; see below. |

All item routes require `Authorization: Bearer <access_token>`. Ownership comes from the signed token, never client JSON. Foreign-owned and nonexistent IDs both return `404 {"detail":"Item not found"}`. Item responses contain no owner field; user responses contain no password/hash.

- Emails are validated, surrounding whitespace stripped, and lowercased at registration and login; maximum normalized length is 254. Passwords are **12–128 characters**, never stripped or normalized; hashing is Argon2id.
- Titles are stripped and must contain 1–200 characters afterward. Descriptions are at most 5,000 characters, may be empty, and cannot be null. Extra JSON fields, including `owner_id`, are rejected.
- NUL (`U+0000`) is rejected in both item fields on create and PATCH with a sanitized 422; it is never removed or passed to PostgreSQL. Other valid Unicode remains unchanged.
- PATCH must supply at least one field. Omitted fields remain unchanged; explicit null and `{}` are invalid. Overlapping concurrent updates use the last committed value, with no version/precondition API.
- Pagination limit is 1–100 (default 50), ordered by immutable `(created_at DESC, id DESC)`, not `updated_at`. Pass `next_cursor` back unchanged and URL-encode it; final/empty pages return null. Cursors are canonical URL-safe unpadded base64 of `[timezone-aware timestamp, canonical UUID]`, at most 512 encoded characters. Invalid cursors return 422. They are not credentials and never remove the owner filter. There is no OFFSET or total-count query.
- Validation failures return 422 with sanitized error details. Request bodies larger than **32 KiB**, including chunked bodies, return 413.
- Duplicate normalized email: `409 {"detail":"Email already registered"}`. Wrong password/unknown email: identical `401 {"detail":"Invalid credentials"}`, until the failure limit below. Missing, malformed, expired, or invalid bearer token: `401 {"detail":"Invalid authentication credentials"}`. Every 401 includes `WWW-Authenticate: Bearer`.
- After `AUTH_PASSWORD_FAILURE_LIMIT` (10) wrong passwords for one existing account within `AUTH_PASSWORD_FAILURE_WINDOW_SECONDS` (900), login returns `429 {"detail":"Too many failed attempts"}` before Argon2 runs, with `Retry-After` until the oldest counted failure leaves the window. Assistant authenticator enrollment shares this counter. Unknown emails record nothing and never get 429, so, like registration's 409, the 429 shows that an account exists.
- Known transient database failures, pool exhaustion, or hash admission exhaustion return `503 {"detail":"Service temporarily unavailable"}` with `Retry-After: 1`; programming/unexpected integrity errors remain 500. Writes are **not automatically retried**: a lost response does not prove a mutation failed to commit.

### Authentication and local security boundary

JWTs use HS256 with a generated secret decoded from hex (at least 32 bytes), a fixed algorithm allowlist, canonical UUID subject, and required/verified `sub`, `iat`, `exp`, `iss`, and `aud` claims. Default lifetime is 900 seconds. Authorization is stateless: valid tokens resolve an owner UUID without an account lookup, including on cache hits. There are **no account deletion/disable, password-change, logout/revocation, or refresh-token endpoints**; issued tokens remain valid until expiry. Do not infer revocation from clearing a browser token.

Registration/login have bounded password-hashing work and the per-account login failure limit above, not public-service abuse controls. Trade-offs: anyone who knows an account's email can block its password logins for one window. The count runs before Argon2 without a lock, because no transaction may span hashing, so parallel wrong guesses can pass the limit by the attempts already admitted to hashing when it is reached (per process: the 2 running plus those its 100 ms queue admits). Keep the loopback binding. The optional remote benchmark below uses authenticated SSH forwarding; it does not turn this into a production TLS deployment. Treat `.env` and CSV bearer fixtures as secrets, never publish them or paste them into reports.

## Configuration and resource bounds

`app/config.py` reads process environment only, with **no implicit dotenv loading**. Compose explicitly supplies its environment. Required application settings are `REDIS_URL`, `JWT_SECRET`, and either `DATABASE_URL` or all of `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`. An explicit `DATABASE_URL` wins and must use `postgresql+asyncpg`. Component-based URL construction safely handles password URL-reserved characters.
Source selection happens before component-field validation: a nonempty explicit URL ignores all five unused `DB_*` values, even malformed ones. An empty URL falls back to components; an invalid nonempty URL fails rather than falling back.

Normal Compose configures database/user `app`, database host `db`, and Redis database 0. `.env` supplies `POSTGRES_PASSWORD`, `JWT_SECRET`, `API_PORT`, `AUTH_TOKEN_TTL_SECONDS`, `HOST_UID`, `HOST_GID`, and optionally `DATABASE_URL`. Other application tuning settings below require an explicit service environment change; merely adding them to `.env` does not forward them into containers.

| Application setting | Default/bound |
| --- | --- |
| `DB_POOL_SIZE` / `DB_POOL_TIMEOUT_SECONDS` | 10 connections/process; 1-second acquisition/connect timeout; overflow is always 0. |
| `DB_STATEMENT_TIMEOUT_MS` | 2,000 ms; asyncpg command timeout is 3 seconds. |
| `REDIS_MAX_CONNECTIONS` / `REDIS_TIMEOUT_SECONDS` | 50/process; 100 ms connect/read; zero retries and a 200 ms outer deadline per cache operation. |
| `CACHE_TTL_MS` | 5,000; accepted range 1–5,000. |
| `AUTH_TOKEN_TTL_SECONDS` | 900; accepted range 1–86,400. |
| `AUTH_HASH_CONCURRENCY` / `AUTH_HASH_QUEUE_TIMEOUT_SECONDS` | 2 hashing workers/process; 100 ms admission wait. Cancellation does not free an occupied permit before hashing ends. |
| `AUTH_PASSWORD_FAILURE_LIMIT` / `AUTH_PASSWORD_FAILURE_WINDOW_SECONDS` | 10 wrong passwords per account per 900 s, then 429; accepted ranges 1–1,000 and up to 86,400 s. |
| `JWT_ISSUER` / `JWT_AUDIENCE` | `fastapi-crud` / `fastapi-crud-api`. |

Each process owns its engine/Redis pool; each request gets its own session. Cache hits do not acquire a database connection. PostgreSQL allows 100 connections: budget **replicas × API workers × 10**, retaining **at least 20** connections for migrations, tools, and other non-API work. More virtual users do not mean more database connections. Do not scale workers past that budget or treat this as autoscaling guidance.

### Cache consistency

Only successful owner-scoped detail reads are cached, using `item:v1:{owner_uuid}:{item_uuid}`. Lists, missing rows, and authentication are not cached. Writes commit first; PATCH/DELETE then best-effort invalidate. Redis errors fall back to PostgreSQL, and failed invalidation never turns a committed write into an apparent rollback. Corrupt cached values are misses.

On a miss, the service reads Redis **TIME before the database SELECT**, computes an absolute deadline at that time plus `CACHE_TTL_MS`, and fills using **SET PXAT**, not a fresh relative TTL. Thus a reader paused before a concurrent mutation cannot resume after invalidation and grant the old row a new five-second lifetime. If TIME fails, that read does not populate the cache.

The maximum five-second staleness guarantee applies to **new cache reads after a mutation**, assumes a normally advancing Redis clock, and does not bound delivery of responses already in flight. Concurrent reads do not promise immediate read-your-writes. Clock rollback or restoring old Redis snapshots is outside the contract; persistence is disabled intentionally.

## Optional TLS edge

`compose.edge.yaml` is an overlay for `compose.yaml` or `compose.ai.yaml`; list it last. nginx 1.30.7 serves `https://127.0.0.1:${EDGE_PORT:-8443}`, and the API's plain HTTP port is no longer published. Default stacks, tests and benchmarks do not use it.

```bash
uv run --locked python scripts/init_edge_tls.py
docker compose -f compose.yaml -f compose.edge.yaml up --build -d --wait api edge
curl --fail --cacert .edge-local/ca.crt https://localhost:8443/health/ready
uv run --locked python scripts/smoke_edge.py --plain-port 8000 --project pythonfastapi
```

`--project` names the Compose project whose edge log the smoke reads; the normal stack's is the checkout directory's name in lowercase (see `docker compose ls`). For the AI stack, start it with `docker compose --env-file .ai-local/.env -p fastapi-ai-local -f compose.ai.yaml -f compose.edge.yaml up --build -d --wait api edge` and run the smoke with its defaults (plain port 8002, project `fastapi-ai-local`). To return to plain HTTP, remove the edge with the same files (`... rm -sf edge`), then run `up -d --wait api` without the overlay; the `<project>_edge_ingress` network stays until you remove it.

- **Certificates.** `init_edge_tls.py` creates `.edge-local/` (0700) with `ca.crt`, `server.crt` (localhost, 127.0.0.1 and ::1; 397 days) and `server.key` (P-256, 0600). The CA key is never written, and the CA's name constraints allow only localhost (and names under it), 127.0.0.1 and ::1. Pass `ca.crt` to clients explicitly (`curl --cacert`, httpx `verify`). Existing files are kept; unsafe paths and partial sets are refused. To renew, delete `.edge-local/` and rerun. `.edge-*/` is excluded from git and the Docker build context.
- **TLS and headers.** TLS 1.2 and 1.3 only, no session tickets, no version in `Server`. Every response gets `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer` and `X-Frame-Options: DENY`; `/api/` responses also get `Content-Security-Policy: default-src 'none'; frame-ancestors 'none'`. No HSTS: on localhost it would force HTTPS on every local site.
- **Sizes and timeouts.** Bodies over 32 KiB get 413 (uploads: 6 MiB). The whole header must arrive within 10 s, and a body may pause at most 10 s between reads (uploads: 30 s). nginx waits up to 30 s for the API's response, or 150 s for assistant messages and RAG queries, then answers 504. These values are fixed in `docker/edge/nginx.conf`: when you raise `ASSISTANT_TURN_DEADLINE_SECONDS`, `RAG_QUERY_DEADLINE_SECONDS`, `RAG_PDF_MAX_BYTES` or `RAG_TEXT_MAX_BYTES`, raise the matching `proxy_read_timeout` or `client_max_body_size` too.
- **Blocked routes.** `/api/v1/rag/query/diagnostics` and `/api/v1/residence/demo` get 404 at the edge.
- **Edge replies.** Replies nginx makes without the API (limits, sizes, blocked routes, 502, 504) have JSON bodies and carry `X-Edge-Reply: 1`.
- **Client addresses.** nginx replaces any `X-Forwarded-For` with the address it sees; the API trusts no forwarded header. Through Docker's port proxy all host clients may arrive from one gateway address, and then the per-address limits act as global caps: on the development Mac nginx saw every host client as 192.168.160.1. Behind a trusted load balancer, per-client limits need a config edit: `set_real_ip_from <balancer range>;` and `real_ip_header X-Forwarded-For;`.
- **Logs.** The access log keeps address, time, method, path, status, bytes, duration and the API's status; never headers or query strings. The error log keeps only critical lines, because nginx writes the request line, Host and Referer into each one.
- **Container.** Read-only root, tmpfs buffers, `no-new-privileges`, and only the CHOWN, SETUID, SETGID and DAC_OVERRIDE capabilities (the last reads the 0600 key). On Linux the host can still reach the API container's bridge address; the overlay removes the published port, not that route.

Request limits per client address (429 past rate plus burst), plus 20 open connections per address:

| Zone | Paths | Rate, burst |
| --- | --- | --- |
| `auth` | `/api/v1/auth/`, `/api/v1/assistant/totp/` | 10/min, 10 |
| `assistant` | `/api/v1/assistant/messages`, `/api/v1/assistant/actions/{id}/confirm` | 20/min, 10 |
| `rag` | POSTs under `/api/v1/rag/`; status reads do not count | 30/min, 10 |
| `api` | `/api/v1/rag/` and every path not listed above | 20/s, 40 |

`smoke_edge.py` sends no credentials. It checks TLS versions and names (TLS 1.1 refused), that the plain port is closed, the headers, an edge 413, the two edge 404s and a login burst that reaches 429, then reports the address nginx saw. The burst uses up the `auth` zone for about a minute.

## Tests, smoke, and outage exercises

The integration suite uses actual isolated PG19/Redis, not SQLite. Its standalone Compose file has literal **test-only** credentials, database/user `app_test`, Redis database 1, a separate project volume, no host ports, and a migration gate. Fixtures refuse cleanup of other database destinations; tests truncate that dedicated test data, so do not point them at a normal stack. Run sequentially, not with parallel pytest workers.
Before app creation or lifespan, fixture guards inspect SQLAlchemy's **resolved driver connection arguments**, including query overrides: the effective database must be exactly the string `app_test`, and Redis must select database 1. Missing or ambiguous database values are rejected without connecting. Connection-free safety tests protect these boundaries; never reproduce a guard defect by truncating another database.

```bash
docker compose --env-file /dev/null -p fastapi-qa -f compose.test.yaml run --build --rm tests
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python scripts/smoke.py
```

The test service runs `pytest --cov=app --cov-report=term-missing`. Use one-off `run`, not `--abort-on-container-exit`: the successful migration is expected to exit. Optional teardown: `docker compose --env-file /dev/null -p fastapi-qa -f compose.test.yaml down` (preserves the test volume). The HTTP smoke creates unique accounts and proves login, persisted CRUD, owner isolation, and bounded cache convergence; it accepts `--base-url` for a nondefault port.

Readiness checks database `SELECT 1` and Redis PING within a one-second total budget:

| Dependencies | `/health/ready` | Body |
| --- | --- | --- |
| Both available | 200 | `{"status":"ok","database":"ok","cache":"ok"}` |
| Redis unavailable, DB available | 200 | `{"status":"degraded","database":"ok","cache":"unavailable"}` |
| Database unavailable | 503 | `{"status":"unavailable","database":"unavailable","cache":"ok"}` (or cache `"unavailable"`). |

Readiness proves reachability, not schema correctness; migrations establish schema before API startup. Exercise only this project's services:

```bash
docker compose stop redis
uv run --locked python scripts/smoke.py
curl --fail http://127.0.0.1:8000/health/ready  # 200, degraded
docker compose start redis
uv run --locked python scripts/smoke.py

docker compose stop db
curl --fail http://127.0.0.1:8000/health/live   # 200
curl -i http://127.0.0.1:8000/health/ready     # 503
# A DB-dependent write using an existing token should return bounded 503.
# A warmed detail cache hit may still succeed briefly; that is allowed.
docker compose start db
# Allow readiness to recover, then prove fresh writes/reads/deletes again:
curl --fail http://127.0.0.1:8000/health/ready
uv run --locked python scripts/smoke.py
```

## Durable item rewriting

Rewriting is disabled in the normal application unless `AI_ENABLED=true`. Enabling the API alone does not start business loops: the API, outbox dispatcher, and Kafka worker are separate processes. PostgreSQL remains authoritative; Redis is not used for jobs or input snapshots. Apply the new Alembic migration before running any updated image, including the normal test image.

### Start the isolated local stack

```bash
python3 scripts/init_ai_env.py
docker compose --env-file .ai-local/.env -p fastapi-ai-local -f compose.ai.yaml up --build -d --wait api worker dispatcher
uv run --locked python scripts/smoke_ai.py
```

The default stack uses a **local provider simulator**, not OpenAI or a paid model. The smoke creates real accounts/items, submits through HTTP, waits for Kafka/worker completion, verifies ownership and no auto-apply, then explicitly PATCHes the suggestion. It prints no bearer tokens or content. Port 8002 must be free; normal port 8000 and benchmark port 8001 are rejected by the AI initializer/smoke.

The initializer creates a separate `.ai-local/.env` with random credentials and mode `0600`, plus private `0700` directories. Existing credentials are preserved; unsafe ownership, modes, links, and normal/escaping paths are rejected rather than repaired. For an existing file it appends only a missing `AI_ASSISTANT_TOTP_KEY` (the assistant's authenticator sealing key) or `AI_ASSISTANT_MEMORY_KEY` (its memory sealing key) and leaves every other line alone; an empty, malformed or repeated key, or one memory key equal to the TOTP key, is reported, not repaired. Until both keys exist, `compose.ai.yaml` commands stop with an error naming the initializer. `.ai-*/` is excluded from version control and Docker build context. Settings read process environment only, never dotenv implicitly.

`compose.ai.yaml` is standalone: **never merge it with normal, benchmark, or QA Compose**; only the [TLS edge](#optional-tls-edge) and [Langfuse](#send-the-ai-stacks-traces-to-local-langfuse) overlays are layered on it. It uses database `app_ai`, Redis DB 2, and project-owned PostgreSQL/Kafka volumes. Only the API publishes `127.0.0.1:8002` (with the edge overlay, only the edge publishes, on 8443); dependencies and simulator/telemetry control endpoints have no host ports. An API-only ingress network enables loopback publication; all dependency traffic stays on the internal network (the Langfuse overlay is the one exception: its collector also joins Langfuse's web network). The local image uses the test target to include simulators and operational scripts. This remains a development deployment, not production TLS, abuse protection, or Kafka HA.

Stop only this stack, retaining its volumes:

```bash
docker compose --env-file .ai-local/.env -p fastapi-ai-local -f compose.ai.yaml down
```

### HTTP contract

Use the same account and item contracts above against `http://127.0.0.1:8002`. With a token and an existing owner-owned item:

```http
POST /api/v1/items/{item_id}/ai/rewrite
Authorization: Bearer <token>
Idempotency-Key: <unique-opaque-key>
Content-Type: application/json

{"instruction":"Make the description clearer without adding facts."}
```

The key is case-sensitive, 16–128 ASCII letters/digits/`.`/`_`/`-`. Instructions are stripped at the edges, must contain 1–1,000 characters, and reject NUL. Extra JSON fields are forbidden. The existing 32 KiB body limit and sanitised 422 responses apply.

New submissions return **202**, a `Location` header, and a job containing `id`, `item_id`, `status`, `status_url`, `source_item_updated_at`, lifecycle timestamps, `suggestion`, and `error_code`. Poll `GET /api/v1/ai/jobs/{job_id}` using the same owner token. Results use `Cache-Control: no-store`; polling never exposes the input snapshot or instruction.

| Condition | HTTP response |
| --- | --- |
| New submission or matching active replay | 202; same durable job for a replay |
| Poll existing job or replay a terminal job | 200 |
| Missing/foreign item on a fresh submission; missing/foreign job | 404 with no ownership disclosure |
| Same owner reuses a key for another item/instruction | 409 |
| Owner admission limit | 429 and `Retry-After` |
| Feature disabled, global admission exhausted, or transient DB failure | 503 |
| Retained job whose content expired | 410, even before physical purge |
| Invalid request/header | Sanitised 422 |

Idempotency is owner-scoped and compares the item ID and validated instruction, not the item's current contents. Exact replays therefore keep the original snapshot after edits or deletion. Matching replays are resolved before a fresh ownership lookup; conflicting replays disclose no target-item existence. Only key/request digests are retained, not raw idempotency keys.

Statuses are `pending`, `leased`, `calling`, `succeeded`, `failed`, and `needs_reconciliation`. Success exposes a validated `{title, description}` suggestion using the existing item limits and Unicode/NUL rules. **No worker writes to `items`.** Saving requires the existing owner-scoped PATCH endpoint. The source timestamp is advisory: PATCH still has no atomic version/precondition protection, so clients must handle stale suggestions explicitly.

### Delivery, recovery, and billing safety

- Submission atomically commits the owner-scoped PostgreSQL snapshot, job, and outbox before returning 202. It bypasses the bounded-stale item cache.
- The dispatcher leases outbox rows, releases the connection, publishes with broker acknowledgement, then records that acknowledgement. A crash in between can duplicate an event; it cannot lose the durable job. Pending unclaimed jobs are periodically redispatched using the same event identity.
- Kafka events contain versioned identifiers and bounded W3C trace context, not item text, instructions, passwords, tokens, or output. Request/DLQ topics have six partitions and finite retention/storage. Single-broker local storage is not an HA guarantee.
- Workers process one record at a time per partition and commit explicit contiguous offsets only after durable handling. Rebalancing drains/cancels work within a bound. Job mutations are fenced by state, attempt, lease token/generation, and database-clock expiry.
- An expired `leased` job can recover safely. Before any provider call, the worker must receive acknowledgement of the committed `calling` transition. An uncertain commit never authorises a call.
- An expired `calling` lease, uncertain provider response, or lost result persistence becomes `needs_reconciliation`. **There is no automatic replay or retry endpoint for uncertain calls.** Inspect the fixed error code and protected provider request metadata before deciding whether to explicitly submit a new key; a new submission may incur another charge. A false-positive uncertain outcome is possible if the process crashes after the marker but before sending.
- The pinned SDK has hidden retries disabled. At most three total attempts are scheduled for proven pre-send failures or explicit rate-limit rejection. Timeouts/uncertain delivery/408/5xx are not automatically replayed. Refusal, incomplete output, invalid output, or unsupported prompt/schema versions never become successful suggestions; no automatic repair generation is made.
- Terminal failures and uncertain outcomes create a durable sanitised DLQ event atomically with the state change. Malformed Kafka messages are dead-lettered using coordinates/digests and a fixed code, never raw bytes; the source offset advances only after the DLQ acknowledgement.
- Broker publication failure durably defers the row and terminates the producer-owning process. Compose restarts it; this avoids accumulating unbounded idempotent-producer batches. No automatic database-write retries were added.

Normal defaults are two concurrent generations per worker, a 30-second total provider deadline, 4,096 maximum output tokens, a 90-second lease, and a 24-hour latest-start deadline. The current model is `gpt-5.6-terra` with reasoning effort `none`. Each job persists its model, prompt version `item-rewrite-v1`, and schema version `1`; deployment changes do not silently change accepted work.

Admission is limited to **1,000 unsettled jobs globally and five per owner**. A terminal job releases admission only after all required publications are acknowledged. A Kafka/DLQ outage therefore eventually stops acceptance rather than growing an unlimited backlog. These limits are not rate limiting or a demonstrated throughput target.

The local/QA runtime pool budget is API 10 + dispatcher 2 + worker 4 connections. Every replica/worker multiplies its own pool; keep the total below PostgreSQL's 100 connections with at least 20 reserved elsewhere. Sessions/transactions are released before Kafka, provider, and cache I/O. `/health/live` and `/health/ready` retain their existing CRUD semantics; broker/provider failure does not make unrelated CRUD unready. Worker/dispatcher container probes check broker TCP reachability, not end-to-end job progress; monitor queue age, lag, and outcomes as well.

### Retention and privacy

Input snapshots, instructions, and suggestions expire **seven days after the terminal outcome**, including uncertain outcomes. The pending interval can add up to 24 hours before this window begins. Polling enforces logical expiry independently of maintenance.

The dispatcher physically clears expired content in bounded batches and retains minimal idempotency metadata for at least 30 days after terminal handling. Unsettled publications keep the metadata needed to finish delivery. Dispatcher downtime or dependency outages can delay physical purge/recovery; a 410 response is not proof that storage has already been erased. Old events never recreate expired jobs or trigger new calls.

OpenAI requests use `store=False`, no tools, no conversations, no streaming, and no background Responses mode. **Application retention is not provider zero-data-retention**: review the account's [OpenAI data controls](https://developers.openai.com/api/docs/guides/your-data) before live use.

One application-owned telemetry provider per process emits allowlisted spans and low-cardinality metrics: queue depth/wait, consumer lag, worker utilisation, provider duration/outcomes/usage, publication timing, retries, and export loss. Job IDs appear only in traces, not metric dimensions. Prompt/output text, account identifiers, auth headers, raw exception text/tracebacks, SQL parameters, arbitrary baggage, and connection strings are excluded. Export queues/deadlines/shutdown are bounded; export failure does not undo successful business operations.

The Collector forwards separate Langfuse/LangSmith trace transformations and local OTLP metrics to internal recording receivers. **These receivers verify real protobuf delivery, redaction, and mapping—not hosted product UI ingestion.** External exporters are not configured by default. Product credentials/region and actual backend acceptance require separate configuration and verification.

### Real product ingestion verification

The verification-only `observability/collector.verify.yaml` sends one content-free
synthetic parent/child trace to selected products. `scripts/verify_ingestion.py`
requires product-side API readback of the exact trace, parent relationship, model,
usage, and absent input/output content. Collector HTTP acceptance alone cannot pass.
The fixture's **7 input / 3 output tokens are synthetic**, not provider billing.
This does not redirect normal application or load-stack telemetry.

#### Local Langfuse

```bash
uv run --locked python scripts/langfuse_local.py up --directory .langfuse-verification
uv run --locked python scripts/verify_ingestion.py --backend langfuse --langfuse-state .langfuse-verification
uv run --locked python scripts/langfuse_local.py down --directory .langfuse-verification
# Explicitly restore the same saved credentials and data:
uv run --locked python scripts/langfuse_local.py resume --directory .langfuse-verification
```

`compose.langfuse.yaml` provisions real Langfuse **4.72.0** with its PostgreSQL,
ClickHouse, Redis, and MinIO dependencies, independently of application data.
Fresh `up` refuses an existing state directory/project. Random login/API credentials
stay in a descriptor-checked, owner-only `0700` directory / `0600` `state.json`;
retrieve them privately, never paste them into chat or logs. These directories are
excluded from version control and Docker build context.

Only the web UI publishes `127.0.0.1:8005`; dependencies have no host ports.
The transient verification Collector uses loopback ports 14318/14333 and is removed
afterwards. Web/worker health checks wait for real readiness; headless initialization
creates one project with three-day trace retention. Browser media uploads are
intentionally unsupported. This is local OSS verification, not a production deployment.
`down` preserves volumes and state; deletion requires explicit `--remove-volumes`.

Actual verification read back the stored `ai.request` / `ai.generate` observations
and inspected them in the Langfuse UI, including model and usage mapping with no
input/output content. No model call was made. Evidence:
`artifacts/ingestion-f072fadff3cb41aea82525bc4d5ef528/result.json`.
The same observations survived an explicit down/resume cycle. Verification services
were then stopped with their volumes and private credentials preserved.

#### Send the AI stack's traces to local Langfuse

```bash
uv run --locked python scripts/ai_langfuse.py --langfuse-state .langfuse-verification
uv run --locked python scripts/smoke_assistant.py
uv run --locked python scripts/smoke_rag.py
```

Open http://127.0.0.1:8005, sign in with the login in the state directory's
`state.json` (read it yourself), and go to **Tracing → Traces**. The launcher resumes
the saved Langfuse and waits for it, then recreates the AI collector with
`compose.ai.langfuse.yaml` and starts the API and workers. `--live` adds
`compose.ai.live.yaml` (billable model calls); `--edge` adds `compose.edge.yaml` and
starts the edge (run `scripts/init_edge_tls.py` first). It prints no key. If a step
fails, the error names it (`collector` or `services`).

```
api / workers --OTLP--> collector: transform/langfuse (allowlist, observation types)
                           -> otlp stand-in receiver          (unchanged)
                           -> langfuse-web:3000/api/public/otel (observability/collector.langfuse.yaml)
```

- You see `assistant.turn` (agent), `assistant.tool` (tool), `rag.query` (chain),
  `rag.retrieve` (retriever), generations and embeddings with model, tokens, timing
  and IDs. Input and output stay empty; there are no user or session fields.
- The collector joins Langfuse's non-internal `api_ingress` network, so it also gains
  outbound access, and its environment holds the project key. Never run
  `docker compose config` or `docker inspect` against this stack with real values.
- The collector's OTLP receiver and health port become reachable from `langfuse-web`.
  Proxy variables are blanked in the collector, so a Docker client proxy setting cannot
  carry the key off the machine.
- If Langfuse goes away, the exporter retries for at most 5 s and drops spans; the API
  and the stand-in receiver keep working. LangSmith stays on the stand-in receiver.
- Any later `up` of these services without the overlay (for example the plain
  `compose.ai.yaml` command) recreates the collector without the Langfuse export, and
  without `--edge` the launcher recreates the API without the edge. Use the launcher
  again to restore it.

Tear down the AI stack first (the collector is attached to Langfuse's network), then
Langfuse; volumes stay:

```bash
docker compose --env-file .ai-local/.env -p fastapi-ai-local -f compose.ai.yaml down
uv run --locked python scripts/langfuse_local.py down --directory .langfuse-verification
```

#### Cloud LangSmith

Provision/authenticate the account separately and inject `LANGSMITH_API_KEY`.
Select an explicit regional `LANGSMITH_ENDPOINT` from the documented cloud hosts;
the verifier does not guess the region. Supply **either** `LANGSMITH_PROJECT`
**or** `LANGSMITH_AGENT_ID` plus `LANGSMITH_AGENT_ENVIRONMENT`, never both.
Use `LANGSMITH_WORKSPACE_ID` when the key needs explicit workspace routing.
See the [official OTLP setup](https://docs.langchain.com/langsmith/trace-with-opentelemetry)
and [regional endpoints](https://docs.langchain.com/langsmith/cloud#regional-storage).

```bash
uv run --locked python scripts/verify_ingestion.py --backend langsmith --allow-external-export
```

The export gate is independent of paid-provider authorization. Readback uses the
exact run IDs, verifies parent/LLM/model/token mapping, and rejects stored content.
Only the bundled synthetic trace is exported. **Cloud LangSmith has not been
verified:** no authenticated account, selected region/project, or process credential
was available. Local receiver tests do not establish cloud acceptance.

### Bounded mixed CRUD + AI validation

Use only the fresh-project runner, never merge `compose.ai.load.yaml` with another
Compose stack:

```bash
uv run --locked python scripts/ai_load_smoke.py --provider simulator --max-ai-jobs 5
# Only after explicit paid authorization and process-secret injection:
uv run --locked python scripts/ai_load_smoke.py --provider live --allow-paid-requests --max-ai-jobs 5
```

Each invocation has its own private artifacts, random database/JWT credentials,
database `app_ai_load`, Redis DB 2, Kafka volumes/network, and loopback API port 8004.
The runner checks destinations before seeding 20 distinct accounts. The fixed
90-second window ramps at five users/second, uses 2–5-second think time and 70/20/10
detail/list/PATCH weights, and requires at least 45 seconds with all 20 users after
30 seconds of warmup. Both whole-run and steady CRUD must have zero errors,
p95 <=500 ms and p99 <=1,000 ms. Generation latency is reported separately.

The ceiling is **five total rewrite admissions/attempts including the preflight**,
not five plus a preflight: concurrency two, one attempt per job, 30-second provider
deadline and 512 maximum output tokens per attempt. A failed/unknown preflight
aborts additional submissions. An append-only private ledger reserves each slot
before HTTP; ambiguous requests are never resubmitted. Dedicated rewrite sources
prove owner isolation, no auto-apply, and explicit PATCH without racing CRUD updates.
At the observation deadline work is cancelled and the worker stopped, with no
drain/replay window. Incomplete/unknown jobs fail; no automatic escalation occurs.
Do not rerun a live command under the same five-attempt authorization.

The runner never reads dotenv. On a **local operator's terminal**, uv can explicitly
inject protected `.env` values without printing keys by adding `--env-file .env`
to `uv run`. This is not implicit application dotenv loading. The live runner
requires `AI_OPENAI_API_KEY`; `AI_LOAD_MODEL` optionally selects the model.
Output-token caps are not a dollar budget or an input-token cap; actual model/token
usage is reported without inventing currency pricing.

The exercised **simulator-only** run passed, with 20 users, 60 seconds of full-user
steady time, all five rewrites verified, no unfinished requests, and five observed
generation spans with no receiver drops:

| CRUD measurement | Whole 90-second run | Steady window |
| --- | ---: | ---: |
| Requests | 509 | 340 |
| Errors | 0 | 0 |
| p50 (ms) | 5.04 | 5.10 |
| p95 (ms) | 9.81 | 9.59 |
| p99 (ms) | 12.74 | 12.21 |

Private evidence is under
`artifacts/fastapi-ai-load-20261031t233312z-98b7e1554242/`:
`report.json`, `admissions.jsonl`, `crud_requests.json`, and `resources.jsonl`
(14 successful Docker CPU/memory/network/block-I/O samples, none failed).
Only that run's containers were stopped; its volumes/artifacts were preserved.
**No live OpenAI calls or production load were validated.** The process key was
unavailable, and root `.env` is tool-protected. Simulator success proves the local
harness and integration path, not live-provider quality/latency, billing, cloud
ingestion, production capacity, or the 15,000–30,000-user target.

### Explicit live OpenAI opt-in

The simulator-only stack has no worker egress. If paid generation has been authorised, supply `AI_OPENAI_API_KEY` through an approved secret source in the invoking environment and add **only** the AI live override:

```bash
docker compose --env-file .ai-local/.env -p fastapi-ai-local -f compose.ai.yaml -f compose.ai.live.yaml up --build -d --wait api dispatcher worker
```

This override changes the worker to the real OpenAI endpoint and attaches its provider-egress network. It does not enable external telemetry. Do not use the live override with QA, normal, or benchmark data; do not put keys on command lines or in logs. Provider availability, account permissions, billing/data-policy approval, and model quality are not established by simulator success. No live OpenAI call or hosted export was made during implementation verification.

### Isolated AI QA and evaluations

Run sequentially. The AI QA stack uses only `app_test`, Redis DB 1, unique per-test Kafka topics/groups, local HTTP simulation, and local OTLP capture:

```bash
docker compose --env-file /dev/null -p fastapi-ai-qa -f compose.ai.test.yaml run --build --rm tests
docker compose --env-file /dev/null -p fastapi-ai-qa -f compose.ai.test.yaml --profile smoke run --build --rm smoke
uv run --locked python scripts/smoke_ai.py
uv run --locked python scripts/smoke.py --base-url http://127.0.0.1:8002
docker compose --env-file /dev/null -p fastapi-ai-qa -f compose.ai.test.yaml --profile smoke stop api dispatcher worker
```

**Stop smoke API/worker/dispatcher processes before another pytest invocation.** Function fixtures validate resolved test destinations before startup/cleanup and truncate job/item/account tables only after those guards. E2E children start after cleanup and stop before teardown. Never parallelise this suite or point it at normal data. Without `AI_QA=1`, the ordinary `compose.test.yaml` suite skips infrastructure-specific Kafka/E2E cases but still exercises the API, provider adapter over loopback HTTP, state transitions, retention, and telemetry contracts.

```bash
uv run --locked python scripts/evaluate_rewrites.py
uv run --locked python scripts/evaluate_rewrites.py --responses candidates.jsonl
```

The bundled eight-case synthetic dataset covers facts/quantities, uncertainty, dates/prices, Unicode, empty descriptions, title bounds, embedded prompt injection, and unsupported claims. Saved candidates must have exactly its case IDs and `{id,title,description}` JSONL rows. Offline reference scoring is a deterministic rubric sanity check, not evidence of model quality or general semantic correctness.

Real model comparisons require `--provider --allow-paid-requests`, an explicit `OPENAI_API_KEY`, and optional repeated `--model` arguments. LangSmith dataset/experiment/run/feedback publication requires `--publish-langsmith --allow-external-publication` and `LANGSMITH_API_KEY` (optional `LANGSMITH_ENDPOINT`). Publication is limited to the bundled synthetic dataset; never supply private production responses. The two network gates are independent and disabled by default. Evaluations do not require database credentials.

Implementation verification: 543 tests passed in the isolated AI topology, including actual Kafka/manual commits/rebalance, provider failure/retry bounds, worker crash after send, stale fences, publication commit gaps, logical/physical expiry, SDK debug-log privacy, Collector trace/metric routing, product-readback acceptance/privacy, and bounded-load safety. The deployed HTTP smoke and ordinary CRUD/cache smoke also passed through loopback port 8002. The original QA topology previously passed with AI infrastructure absent (infrastructure-specific cases skipped), and `alembic check` found no model/migration drift. These checks establish local behaviour, not AI quality, production capacity, cloud ingestion, or HA; the separately exercised local Langfuse ingestion is documented above.

## Document question answering (RAG)

RAG (retrieval-augmented generation) answers a question from the owner's own uploaded documents, with citations. It is off unless `RAG_ENABLED=true` **and** `AI_ENABLED=true`; it reuses the AI stack's Kafka, provider, and Collector. `compose.ai.yaml` and `compose.ai.test.yaml` enable it and add a `rag-worker` service.

```
upload / PUT content --+
connector-sync --------+--> rag_documents (new version) + rag_outbox, one transaction
                                 | insert trigger: one delivery row per registered sink
           dispatcher -----------+--> Kafka rag.document-events.v1 ------> rag-worker
           rag-dispatcher-sqs ---+--> SQS rag-document-events.fifo ---> rag-worker-sqs
                both workers: claim (first wins, the duplicate settles)
                  created/updated: parse (sandboxed: PDF + OCR, Word, HTML) -> chunk -> embed
                                   (+ graph extraction per batch, RAG_GRAPH_ENABLED)
                                   -> swap to the new version in one transaction
                  deleted: purge the tombstone
query  --> API: auth -> query slot -> admit (rate limit) -> embed question -> owner-scoped pgvector KNN
                 (+ graph-linked chunks) -> neighbor windows (+ optional rerank) -> claims, each with a quote
                 -> server checks every quote and number (+ optional verifier) -> 200
```

Start and smoke the local stack (simulator only, no paid calls):

```bash
python3 scripts/init_ai_env.py
docker compose --env-file .ai-local/.env -p fastapi-ai-local -f compose.ai.yaml up --build -d --wait api worker rag-worker dispatcher rag-dispatcher-sqs rag-worker-sqs
uv run --locked python scripts/smoke_rag.py
```

The smoke uploads Markdown, PDF, Word and HTML files (and a scanned PDF when `pdftoppm` is on the host), waits for `ready`, and checks that the answer cites the right document and that a second account sees nothing. It then replaces one document's content and waits for version 2 to be served. Last, it deletes every document, checks each is 404 at once, and checks that the answer becomes `no_context`. It prints no tokens or content.

### RAG HTTP contract

All routes need `Authorization: Bearer` and return `Cache-Control: no-store`.

| Route | Success | Notes |
| --- | --- | --- |
| `POST /api/v1/rag/documents` | 202; 200 on replay of a finished upload | Multipart `title` (1–200 chars) + one `file` (`.txt`, `.md`, `.pdf`, `.docx`, `.html`/`.htm`). `Idempotency-Key` header. `Location` header. |
| `PUT /api/v1/rag/documents/{id}/content` | 202; 200 for the same content or a replay | Multipart `file` only, same types and limits, `Idempotency-Key`. Starts the next version; queries read the current `active_version` until the new one is ready. 409 while a version is in progress. |
| `GET /api/v1/rag/documents` | 200 | Owner's uploaded documents, newest first (connector documents are not listed). |
| `GET /api/v1/rag/documents/{id}` | 200 | Status, `version`, `active_version`, `update_error_code`, `source_kind`, chunk and page counts, error code, timestamps. Never the content. |
| `GET /api/v1/rag/documents/{id}/graph` | 200 | Distinct nodes and edges of the active version (at most 1,000 each, `truncated`), and `graph_error_code`. Owner only; empty while no version is active. |
| `DELETE /api/v1/rag/documents/{id}` | 204 | One transaction removes the chunks and the source and leaves a tombstone (404 to every reader) plus a `doc.deleted` event; the worker purges the row and **its idempotency record**, so the same key can then create a new document. |
| `POST /api/v1/rag/query` | 200 | JSON `question` (1–2,000 chars), `top_k` (1–8, default 4), optional `document_ids` (1–20). Returns `status` (`answered`/`no_context`), `answer`, `citations[{document_id,title,ordinal,page,snippet,quotes}]`. `quotes` are the checked, word-for-word spans the answer relies on. |

Upload errors: 401 before any body byte is read; 400 malformed multipart or disconnect; 408 slow body; 409 key reused with other content; 413 too large; 415 wrong type; 422 bad title, empty file, invalid UTF-8, NUL, or bad PDF header; 429 owner quota or another upload by the same owner in progress; 503 disabled, busy, or global backlog full.

Query errors: 429 owner rate limit (`Retry-After`); 503 disabled, busy, or provider rate-limited; 504 provider timeout or the 60 s deadline; 502 any other provider failure. Foreign and missing documents share one 404, and a foreign `document_ids` filter gives `no_context`.

### RAG rules and limits

- Every SQL statement on documents, chunks and queries filters by owner. A composite foreign key stops a chunk from belonging to another owner's document.
- No question with zero searchable chunks reaches the provider. The model returns claims, not free text; an answer whose claims fail any check becomes `no_context` with fixed text, so unchecked model text is never shown (see below).
- No database transaction spans provider, Kafka, or child-process I/O. Each provider call is one attempt; the worker owns embedding retries (at most three attempts). Answers are never replayed.
- Worker writes are fenced by status, attempt, lease token, lease generation, and database-clock expiry. A document deleted mid-ingestion stops the worker without writes.
- Uploaded bytes are deleted when ingestion ends (`ready` or `failed`). Chunk text stays until the owner deletes the document. `rag_queries` keeps usage metadata only, never question or answer text.
- The LCEL chain runs under `langsmith.tracing_context(enabled=False)`, so `LANGSMITH_TRACING` in the environment cannot export content. Telemetry carries IDs, counts, models, and fixed codes only; the Collector maps embedding, retriever, and chain observation types.
- PDF, Word and HTML parsing runs in one child process (`app/rag/extract.py`) with CPU, memory, and file-size limits, a scrubbed environment, and a wall timeout. The Linux container is the supported runtime; on macOS the limits cannot be set, so parsing fails closed.
- Word: paragraphs and tables in order, headings become the heading path, images ignored; more than 1,000 ZIP entries or 50 MiB uncompressed is refused before parsing. HTML: UTF-8 only, `lxml` with no network or entity resolution, `script`, `style`, `noscript`, `template` and comments dropped.
- OCR (`RAG_OCR_ENABLED`, default on): a PDF page with fewer than 20 extracted characters is rendered by `pdftoppm` and read by `tesseract`, through pipes only, inside the child's limits. At most `RAG_OCR_MAX_PAGES` (25) pages, `RAG_OCR_PAGE_SECONDS` (20) each, all within the parse timeout; pages past either stay unread. A scan with no text after OCR fails as `no_text`.
- Versions: a document serves `active_version` while the next one is ingested; the swap to the new version and the removal of the old chunks are one fenced transaction. A failed update keeps the old version served and reports `update_error_code`. One version is in flight per document.
- Delete keeps its privacy timing: chunks and source go in the delete transaction. The tombstone is purged by the worker when it consumes `doc.deleted`, or by recovery after an hour if the event is lost.

Defaults: text and HTML 1 MiB; PDF and Word 5 MiB; PDF 300 pages, 30 s, 1 GiB; 1,000,000 extracted characters; chunks of 1,200 characters with 200 overlap, at most 1,200 per document; 64 chunks per embedding request. 50 documents and 5 unsettled per owner; 100 unsettled globally; one upload per owner and 4 uploads and 4 queries per API process; 10 queries per 60 s per owner. Embeddings use `text-embedding-3-small` (1,536 dimensions) in pgvector 0.8.7 with an HNSW cosine index. Pools: API 10 + dispatcher 2 + worker 4 + rag-worker 4 + rag-dispatcher-sqs 2 + rag-worker-sqs 4 + connector-sync 2 = 28 of PostgreSQL's 100.

With `compose.ai.live.yaml`, `rag-worker` and `rag-worker-sqs` pay for document embeddings (either may claim a document first) and the API pays for question embeddings and answers. The same paid-use authorisation applies.

### Document events and sinks

Every document change writes one `rag_outbox` row: `doc.created`, `doc.updated`, `doc.deleted`, or a dead letter. The event (`DocumentEvent` v1) carries its id, type, document id and version, attempt, `source_kind` and time; never a title, text or connector detail.

| Sink | Where | Publisher | Consumer |
| --- | --- | --- | --- |
| Kafka | `rag.document-events.v1` (6 partitions, key = document id); DLQ `rag.document-events.dead-letter.v1` | `dispatcher` | `rag-worker` |
| SQS (local ElasticMQ 1.6.14) | FIFO `rag-document-events.fifo`, group = document id; DLQ `rag-document-events-dlq.fifo` | `rag-dispatcher-sqs` (`python -m app.rag.sinks`) | `rag-worker-sqs` (`python -m app.rag.sqs_worker`) |

- Sinks are registered in `rag_event_sinks` (`kafka` by the migration, `sqs` by its dispatcher on start). An insert trigger gives every new outbox row one `rag_outbox_deliveries` row per registered sink, so no write path can miss one and a sink registered later never replays history.
- Each sink claims, sends and acknowledges its own delivery rows, and each publisher is its own process: one broker down never holds back the other. A still-pending ingest event is redispatched per sink.
- Both workers claim through the same fence (status, version, attempt, lease), so the first claim wins and the duplicate settles with no writes. The SQS worker deletes a message only after the outcome is committed, puts a deferred one back in 5 s, extends visibility while it works, handles at most `AI_CONCURRENCY` at once, and sends malformed messages to the DLQ with fixed fields only. There is no redrive policy.
- `RAG_EVENT_SINKS` is `kafka` (default) or `kafka,sqs` (`compose.ai.yaml`). With `sqs`, `RAG_SQS_ENDPOINT` is required; the client uses fixed local credentials, no AWS config or credential files, no proxy environment and no retries. Real AWS SQS is not tested.
- A registered sink stays registered. To stop SQS for good, stop its services and run `DELETE FROM rag_event_sinks WHERE sink = 'sqs'`; its unpublished deliveries are removed when their documents are.

### Database and API connectors

`connector-sync` (`python -m app.rag.connectors`, Compose profile `connectors`) pulls records into building documents owned by a manager of that building. Connectors are operator config only (`RAG_CONNECTORS_FILE`, read once at start; no route creates them). `docker/connectors/connectors.example.yaml` shows both kinds; its placeholder ids make every run fail with `owner_not_manager` until you set a real building and manager.

| Kind | Source | Bounds |
| --- | --- | --- |
| `postgres` | One `SELECT` (no `;` or comments) returning `external_id`, `title`, `content`; DSN from the variable named by `dsn_env` | `READ ONLY` transaction, 10 s statement timeout, `max_records` (≤ 5,000), per-row and 64 MiB total text, bounded read-ahead |
| `http_json` | `GET url`, items at `items_path`, field names for id, title and content, `none`/`page`/`cursor` paging, optional `Authorization` from `auth_header_env` | https unless the host is in `allowed_hosts`; no redirects or proxy environment; 10 s per request, 120 s per fetch, 5 MiB per response (gzip inflated under the cap, other encodings refused), `max_pages` (≤ 100) |

- One runner per connector across processes (PostgreSQL advisory lock); each run checks that the owner still manages the building. New id: document and `doc.created`. Changed content: next version and `doc.updated`. Same content: nothing (a title change is applied in place; a version that ran out of time or attempts is retried). Duplicate ids in one fetch: the first wins.
- Only a complete, non-empty fetch with no cap hit deletes: documents whose ids were not seen get a tombstone and `doc.deleted`. A partial, failed, malformed or empty fetch deletes nothing, so the sync can never remove a source's last record. Documents left in another building or owner by an earlier config are removed at the start of a run.
- Connector documents are refused (409) by the document update and delete routes, do not count toward owner upload quotas, and are capped in progress by `RAG_CONNECTOR_UNSETTLED_LIMIT` (20).
- `rag_connector_runs` keeps counts and fixed error codes only; logs keep the connector name and exception type. Nothing stores or logs content, queries, URLs or DSNs.
- Known limits: page mode can miss a record that shifts pages during a fetch (its document is removed and re-created next run); documents of a removed connector stay until an operator acts; postgres DSNs are not forced to TLS; run rows are not pruned.

Local fakes on the internal `connector_sources` network: `source-db` (seeded `notices`, a read-only role) and `source-api` (`tests/support/source_api.py`, paged, with `/control/records/{id}` to change records). The connector smoke runs inside the stack, creates its own manager and connectors, changes and restores the fakes:

```bash
docker compose --env-file .ai-local/.env -p fastapi-ai-local -f compose.ai.yaml --profile connectors run --build --rm -e SOURCE_DB_ADMIN_DSN=postgresql://source_admin:local-fake-source-only@source-db:5432/notices connector-sync python scripts/smoke_connectors.py
```

Real sources need their own egress overlay and private env file; the fake credentials above are for the local fakes only.

### Answer grounding

Retrieval can find the right passage while the model still misreads it. So the answer is built only from claims the server can check (prompt `rag-answer-v3`):

```
model -> claims[{statement, citation P<n>, quote}]   (no claims -> no_context: no_claims)
  each claim: quote has >= 3 words                   else quote_too_short
              normalized quote is in the cited
              passage, on word edges                 else quote_not_found
              every number in the statement is
              in the cited passage                   else number_not_in_passage
  RAG_VERIFIER=on: a second model grades each claim  else verifier_rejected
answer = statements joined; citations = cited passages with their quotes
```

- **All or nothing.** One failed claim refuses the whole answer, because dropping a claim can change the meaning of the rest. The reason goes to `rag_queries.no_context_reason` and the `outcome` tag of `rag.queries`, never to the response.
- **Context.** Markdown chunks carry their heading path (`Pets > Dogs`, in the prompt only, not embedded). Each passage is widened to its neighbor chunks on the same page, with the overlap removed; quotes are checked against exactly that text.
- **Usage.** `input_tokens` and `output_tokens` are totals over every chat call of the query; `chat_calls` counts them and `reasoning_effort` records the effort. A call that fails has unknown tokens and is not counted; earlier billed calls still are.
- **Known false refusals.** Scripts without spaces (Chinese, Japanese, Thai), PDF words split by a line-break hyphen, and numbers the model writes as words.

| Setting | Default | Effect |
| --- | --- | --- |
| `RAG_NEIGHBOR_CHUNKS` | `1` | Neighbor chunks on each side of a hit (0–2). No extra call; input tokens grow up to 3×. |
| `RAG_RELEVANCE_MARGIN` | unset | Drops hits farther than the best distance plus this margin. |
| `RAG_RERANK` | `off` | `llm`: one more paid call ranks `min(8, 2 × top_k)` candidates; an empty ranking gives `rerank_empty`. |
| `RAG_ANSWER_REASONING_EFFORT` | `none` | `low` or `medium` for the answer call only. |
| `RAG_VERIFIER` | `off` | `on`: one more paid call; any unsupported claim gives `verifier_rejected`. A failed verifier call fails the query (fails closed). |

`RAG_QUERY_DEADLINE_SECONDS` (default 60, at most 120) must exceed the embedding timeout plus one provider timeout per chat call; rerank and the verifier each add one. With the default 20 s and 30 s timeouts that is 80 s for one of them and 110 s for both. Startup fails otherwise. `compose.ai.yaml` sets 115 s so the eval can try either.

### Knowledge graph (GraphRAG)

Off unless `RAG_GRAPH_ENABLED=true`. Each extraction batch is one paid chat call at ingest (prompt `graph-extract-v1`), so the graph costs money only while documents are ingested; queries make no extra call.

```
ingest, per batch of RAG_GRAPH_BATCH_SIZE chunks (after the embed call, no transaction open)
  model -> nodes{chunk C<n>, type, name} + edges{source, relation, target, quote}
  server keeps:  type and triple in the ontology (building-ontology-v1)
                 name: in its chunk on word edges, no email or phone number
                 edge: quote of 3+ words in its chunk naming both ends; two distinct keys
  -> rag_graph_nodes / rag_graph_edges, stored with the batch's chunks (same fence)
query (after vector search, one REPEATABLE READ snapshot, same scope and active versions)
  seeds:    nodes the question names (whole words), then nodes of the vector hits
            within the relevance margin; at most 50
  hops:     RAG_GRAPH_HOPS, each reads at most 50 unread edges outside the vector hits
  evidence: their chunks, most links first, then closest; up to RAG_GRAPH_EXTRA_PASSAGES
  mentions: slots the edges left go to chunks naming a seed or an entity the hops set out
            from, most such entities first, then closest; no further hop from there
            and no text a neighbor window already holds
```

- **Grounded, not trusted.** Graph passages are ordinary chunk text, so every answer quote is still checked against what the model received. The graph only decides which extra chunks are sent; it is never a source of its own. The relation, its direction and the node types are the model's reading of the quote; retrieval follows edges as undirected links.
- **Scope.** Nodes and edges belong to one chunk of one version and cascade with it, so swaps, retries, deletes and purges remove them. Entities merge across documents only at query time, by `node_key` (the name's normalized words, without its type: a live model typed "cellar" as a location in one document and a facility in another), inside one owner's or one building's documents. Rows stored before this change keep `type:name` keys and share no edges with newer rows until their document gets a new version (a mention can still reach them by name).
- **Failures.** A rejected, invalid or cut-short extraction stores no graph for that batch and records `graph_error_code` (`graph_rejected`, `graph_invalid`, `graph_incomplete`); the document is still served. A transient provider failure (timeout, rate limit, 5xx) retries the whole document like an embedding failure and can use up its three attempts.
- **Privacy.** People are kept out by the prompt only; the server refuses names that look like an email address or phone number. Spans carry counts only (`node_count`, `edge_count`, `graph_dropped`, `graph_passages`), never names or quotes.
- **Limits.** Graph passages fill only the slots left under 8: with `top_k` 8 (or rerank with `top_k` 4 or more) the graph adds nothing. In the assistant, graph passages use the turn's 8 passage labels. A graph passage can repeat text the neighbor window already holds. Existing documents get a graph on their next version; there is no backfill.

| Setting | Default | Effect |
| --- | --- | --- |
| `RAG_GRAPH_ENABLED` | `false` | Extraction at ingest and expansion at query time. Adds one provider timeout to the lease floor. |
| `RAG_GRAPH_BATCH_SIZE` | `1` | Chunks per extraction call (1–8); also the stored batch size while the graph is on. With 4, a live model filled the first chunk of each call and left the rest empty, so the default gives each chunk its own paid call. `compose.ai.yaml` reads it from `AI_RAG_GRAPH_BATCH_SIZE`. |
| `RAG_GRAPH_MAX_ITEMS_PER_CHUNK` | `8` | Nodes and edges kept per chunk (1–24). Larger values can exceed `AI_MAX_OUTPUT_TOKENS`. |
| `RAG_GRAPH_HOPS` | `1` | Edge hops from the seeds (1–2). |
| `RAG_GRAPH_EXTRA_PASSAGES` | `2` | Graph passages after the vector passages (0–4). |

### RAG evaluation

`POST /api/v1/rag/query/diagnostics` is an eval-only route. It is 404 (before authentication) and hidden from the schema unless `RAG_DIAGNOSTICS_ENABLED=true`, which only `compose.ai.yaml` sets. It runs the normal query (same owner scope, rate limit and usage row) and also returns, to the owner only, the passages sent to the model, each claim with its check result, verifier verdicts, the refusal reason and usage. It takes optional `overrides`: `neighbor_chunks`, `relevance_margin`, `rerank`, `reasoning_effort`, `verifier`, `graph`. Each passage carries its `source` (`vector` or `graph`). An override set that does not fit the deadline is 422.

`tests/evals/rag_v1.jsonl` holds 34 golden cases over the five documents in `tests/evals/rag_documents/` (single fact, numeric, condition, multi-passage, heading, split across chunks, distractor, not covered, injection, multi-hop). A multi-hop case needs two facts from two documents that share one entity (storage cages -> cellar -> opening hours). Each case is scored on retrieval, citations, status, facts and forbidden strings, so a wrong answer after a correct retrieval is counted on its own (`wrong_given_hit`). CI checks the data and the scorer only; the simulator cannot judge answers.

Live, paid, after explicit authorisation, against the AI stack started with `compose.ai.live.yaml`:

```bash
uv run --locked python scripts/evaluate_rag.py --allow-paid-requests --repeat 3 --config base --config checked=verifier:on --output artifacts/rag-eval.json
uv run --locked python scripts/evaluate_rag.py --allow-paid-requests --judge --judge-model MODEL
```

`--graph` runs every selected case twice, as configurations `vector` (`graph:off`) and `graph` (`graph:on`). Start the stack with `AI_RAG_GRAPH_ENABLED=true` in the invoking shell (it sets `RAG_GRAPH_ENABLED` for the API and workers; default off) so the fixture documents are ingested with a graph; extraction is paid too:

```bash
AI_RAG_GRAPH_ENABLED=true docker compose --env-file .ai-local/.env -p fastapi-ai-local -f compose.ai.yaml -f compose.ai.live.yaml up --build -d --wait api worker rag-worker rag-worker-sqs dispatcher rag-dispatcher-sqs
```

```bash
uv run --locked python scripts/evaluate_rag.py --allow-paid-requests --graph --case multi-hop-storage-hours --case multi-hop-guest-apartment
```

`--judge` first runs six judge controls with known verdicts and exits 2 if the judge gets one wrong. The script accepts loopback destinations only (not ports 8000 or 8001) and prints case IDs, check names, rates and fixed codes; `--output` saves scores without text, mode 0600. Against the simulator it exits 1 by design: it only checks the wiring.

## Resident assistant (tool-using agent)

A resident asks one question or request per turn. A LangGraph agent answers from their building's documents with citations, reads live data, and acts on **simulated** devices (rows in PostgreSQL; no hardware) through a fixed allowlist of typed tools. The server, not the prompt, enforces identity, permissions and limits. It is off unless `ASSISTANT_ENABLED=true`, which requires RAG. `compose.ai.yaml` enables it, with the demo setup route (`ASSISTANT_DEMO_ENABLED=true`).

```
POST /api/v1/assistant/messages
  JWT -> resident (membership -> apartment, building) -> 403 if none
  -> at most 64 turns per process -> admit: per-user lock, one running turn per user
     across processes, rate limit, turn row
  -> LangGraph:  model --tool call--> tools --> model ... --reply--> END
                   at most 4 model calls; a call starts only if its timeout fits the 120 s budget
                 tools: allowlist -> validate arguments -> server checks
                        -> one transaction: write + audit row
  -> grounding rule -> usage row -> 200

POST /api/v1/assistant/totp/enroll {password}   -> 201 secret + otpauth URI
POST /api/v1/assistant/totp/activate {code}     -> 200 enabled

POST /api/v1/assistant/actions/{id}/confirm {code}
  JWT -> resident -> step-up: lock authenticator row -> failure count -> code for a
         step after the last spent one -> step spent, committed
  -> one transaction: own live pending unlock -> still a resident -> unlock -> audit -> 200
```

Smoke it against the simulator (no paid calls), with the AI stack above running. It enrolls an authenticator and confirms unlocks with real codes:

```bash
uv run --locked python scripts/smoke_assistant.py
```

### Assistant HTTP contract

All routes need `Authorization: Bearer` and return `Cache-Control: no-store`.

| Route | Success | Errors |
| --- | --- | --- |
| `POST /api/v1/assistant/messages` with `{message}` (1–2,000 chars, no NUL) | 200 `{turn_id, status, answer, citations[], actions[], pending_actions[]}` | 403 no residence; 422; 429 turn limit or a turn already running (`Retry-After`); 503 disabled, busy, or provider rate-limited with nothing committed; 502/504 provider failure, timeout or budget with nothing committed |
| `POST /api/v1/assistant/totp/enroll` with `{password}` | 201 `{secret, otpauth_uri}`: a base32 secret for an authenticator app; replaces one not yet activated | 403 wrong password (counts toward the login limit); 409 already enabled; 422; 429 password limit (`Retry-After`); 503 disabled or hashing busy |
| `POST /api/v1/assistant/totp/activate` with `{code}` (6 digits) | 200 `{enabled: true}` | 403 wrong code; 409 not enrolled or already enabled; 422; 429 code limit (`Retry-After`); 503 disabled |
| `POST /api/v1/assistant/actions/{id}/confirm` with `{code}` (6 digits) | 200 `{id, action, status: "confirmed", device: {name, locked}}` | 403 no residence, authenticator not enabled, or wrong code; 404 foreign or missing; 409 expired or already decided; 422 missing or malformed code; 429 code limit (`Retry-After`); 503 disabled. The code is checked before the action, so 404 and 409 need a valid code |
| `GET /api/v1/residence` | 200 building, apartment, devices, upcoming bookings, newest tickets, live pending unlocks | 403; 503 |
| `POST /api/v1/residence/demo` | 201: a demo building, apartment `1A`, two thermostats, two door locks, an alarm, two bookings; the caller becomes resident and manager | 404 when the demo flag is off (before authentication); 409 already a resident |
| `POST /api/v1/rag/documents?building_id=` | As for RAG uploads; the document belongs to the building | 404 unless the caller manages that building |

The final reply is claims, checked like RAG answers (prompt `assistant-v5`): each passage claim needs a quote from a passage handed out this turn, and the reply shows its statement. Each successful device, ticket or booking result gets a label `T1`–`T8` and a `fact` the server writes from checked values. A tool claim cites that label and quotes the fact, and the reply shows the fact in place of the model's sentence, so a reply cannot misstate a tool result. A later result for the same device replaces its earlier fact; a denial has no fact. While memory is on, a memory claim cites a recalled note `M1`–`M20` and quotes at least three of its words; the reply shows the whole note as `You told me: "…".`, never the model's sentence, so a fragment cannot drop a "not" or cut a number ([Assistant memory](#assistant-memory)). `citations[]` items carry `quotes`. `status` is `answered` (every claim checked), `no_context` (fixed text: nothing grounded the answer, or a claim failed with nothing committed), or `incomplete` (fixed text: the turn stopped after committing a change, including a failed claim after a write, `error_code` `claim_unverified`; `actions` lists what happened). The failed check is stored in `assistant_turns.claim_failure`. `actions[]` items are `{tool, decision, result, device, ticket_id, pending_action_id}`, built by the server from tool results, never from model text.

### Tools and rules

| Tool | Arguments | Effect | Limit |
| --- | --- | --- | --- |
| `search_documents` | `question` | Top passages from the resident's building documents, labels P1–P8 per turn | 8 passages per turn |
| `list_bookings` | none | Up to 10 upcoming bookings of the apartment | none |
| `create_ticket` | `category`, `summary` | Opens a ticket; the same category within 10 minutes returns the existing one | 3 per user per 24 h |
| `set_thermostat` | `device`, `celsius` (16–26, steps of 0.5) | Sets the target | 10 device actions per user per hour |
| `lock_door` | `device` | Locks the door | shares the device-action limit |
| `unlock_door` | `device` | Creates a 60 s pending unlock; never unlocks | 3 per user per hour |
| `remember` (memory on) | `text`, `scope` (`resident` or `residence`) | Saves `text`, which must be whole words of the current message | 50 active, 10 writes per user per 24 h, shared with the route |
| `forget` (memory on) | `label` (`M1`–`M20`), `request_quote` | Deletes that recalled note; `request_quote` must be words of the message that ask to forget | none |

- **Identity from the server.** The model supplies only tool names and values. User, apartment and building come from the caller's token and membership. Extra arguments are rejected.
- **Checked, audited writes.** Each write runs in one transaction: per-user lock, membership share lock, limit count on the database clock, a write guarded by apartment, device name and kind, and an `assistant_tool_calls` audit row. Unknown tool names are audited as `unknown`.
- **Only the resident confirms, with a code.** No tool can confirm an unlock, and confirm needs an authenticator code ([Unlock step-up](#unlock-step-up)). The alarm exists but no tool can reach it, and the model never sees it.
- **Untrusted text.** The message, recalled memories, tool results and passages are data, never instructions. A memory never authorizes an action. Once a search hands out passages, only `search_documents` and `list_bookings` run for the rest of the turn; every other valid call is denied with `not_permitted` and audited, because a document could be steering the model. A message that asks for an action and a document answer works when the model acts first; otherwise the server adds a fixed sentence that an action was not run, and the resident can send it as its own message. An action that depends on a document takes two messages. Planted document text therefore cannot open a ticket, change a device, request an unlock, or change a note. It can still make the model search again or list bookings, and shape the reply within the claim checks.
- **No content stored or traced.** Turns and audit rows hold counts, IDs and fixed codes; ticket summaries live only in `tickets`. The graph runs with LangSmith tracing disabled; spans `assistant.turn` (Langfuse `agent`) and `assistant.tool` (Langfuse and LangSmith `tool`) carry IDs and counts only.
- Defaults: 10 turns per 60 s per user. One running turn per user across all API processes: another gets 429 `A turn is already running` (`Retry-After: 1`). A turn's lease ends with the turn, or 30 s after its deadline if its process died. At most 64 running turns per API process (`ASSISTANT_CONCURRENCY`, up to 256), then 503 at once; turns hold no database connection while they wait on the provider.

### Unlock step-up

Enroll once with the account password, add the returned secret (or `otpauth_uri`) to an authenticator app, and activate it with a first code. Each unlock confirm then sends a current code.

- Codes follow RFC 6238: SHA-1, 6 digits, 30 s steps, one step of clock drift either way. A code must match a step later than the last one spent, on the database clock, so each code works once.
- Every check locks the user's authenticator row, so parallel guesses cannot pass the failure count together and one code cannot confirm twice. At `ASSISTANT_TOTP_FAILURE_LIMIT` (5) wrong codes in `ASSISTANT_TOTP_FAILURE_WINDOW_SECONDS` (900), activate and confirm return 429 without checking the code. A replayed code counts as a wrong one. At that rate a guesser succeeds with about 0.14% chance per day.
- A matching code is spent and committed before the action is claimed, so a 404 or 409 after the match cannot reuse it.
- A confirm without an enabled authenticator, or with a wrong code, writes a `denied` audit row (`step_up_required`, `step_up_failed`) with no action or device id: the action is not yet proven to be the caller's. `step_up_required` denials are not counted as failures; the edge's `assistant` zone limits their rate.
- Secrets are sealed with AES-256-GCM under `ASSISTANT_TOTP_KEY` (32 bytes as base64, required when the assistant is on), with the user id as associated data, so a row copied to another user does not open. Secrets, codes and passwords are never logged, traced or stored in plain text. `compose.ai.yaml` reads the key from `AI_ASSISTANT_TOTP_KEY`, `compose.ai.test.yaml` sets a test-only key, and the benchmark runner generates `BENCH_ASSISTANT_TOTP_KEY` per run.
- Not included: reset, disable and recovery codes. A lost authenticator blocks that user's unlock confirms until an operator deletes their `totp_credentials` row; a lost or changed `ASSISTANT_TOTP_KEY` does this for every enrolled user (500 `Authenticator unavailable`). Assistant memory is sealed the same way under its own `ASSISTANT_MEMORY_KEY` (`AI_ASSISTANT_MEMORY_KEY`, a test-only key, `BENCH_ASSISTANT_MEMORY_KEY`); losing or changing it makes every memory read a 500 `Memory unavailable`. Downgrading migration 0006 drops all enrollments and failure counts, and refuses while step-up audit rows exist.

### Assistant memory

A resident can ask the assistant to keep a note ("remember that I prefer the gym after 19:00") and to forget it later. It is off unless `ASSISTANT_MEMORY_ENABLED=true`, which requires the assistant and `ASSISTANT_MEMORY_KEY`. `compose.ai.yaml` enables it; the QA smoke API enables it; the pytest stack leaves it off except in the memory tests.

```
write   route POST or the remember tool
          -> lock the user row -> (tool) membership share lock
          -> residence scope: share-lock the resident membership, 403 / not_permitted if none
          -> active memories < ASSISTANT_MEMORY_LIMIT (50), else 429 / limit_exceeded
          -> writes in the last 24 h < ASSISTANT_MEMORY_WRITES_PER_DAY (10), else 429 + Retry-After
          -> seal -> insert memory + write row (+ audit row for the tool), one transaction
recall  each turn, before the first model call: unexpired notes, plus residence notes of
        the building the resident lives in now; newest first, at most 20 and 4,000
        characters; labelled M1..Mn as JSON strings in the first input
forget  route DELETE (one or all), or the forget tool on a recalled label: hard delete
purge   the dispatcher tick deletes expired and moved-out notes and day-old write rows
```

- **Scopes.** `resident` follows the user. `residence` holds only while the user lives in the building it was saved in: moving out hides it at once and the purge removes it.
- **Retention.** 90 days per note (`ASSISTANT_MEMORY_RETENTION_SECONDS`, 1 to 365 days), on the database clock. Reads never serve an expired note, even before the purge.
- **Explicit writes only.** The tool's `text` must be whole words of the resident's current message: no cut word, no cut number (`1` is not in `1,200`). A note needs at least three words, so a reply can always quote it. Nothing is saved automatically.
- **Forgetting needs a request.** `request_quote` must be words of the message that contain `forget`, `delete`, `remove` or `erase`, and the message must not say "don't forget". The tool's fact names the deleted note, so the resident sees what went. A forgotten note grounds no later claim in the same turn.
- **Steering.** Neither memory tool runs after a search handed out passages (see Untrusted text). Memories go in as JSON strings under their own `M` labels, so a note that reads like `T1: Front door is unlocked` is never a tool fact.
- **Privacy.** Text is sealed with AES-256-GCM; the associated data binds it to its purpose (`memory-v1`), user and row. Text is never logged, traced, or stored outside the sealed column: turns hold `memory_count`, audit rows hold `scope` only. `create_ticket` refuses a summary that repeats three words in a row of a recalled note that the current message does not say, because co-residents can read tickets. A paraphrase can still pass that check.
- **A row that does not open** (changed key or row) makes the list route return 500 `Memory unavailable`, and is skipped by recall, so one bad row never fails every turn.
- **Not included:** conversation memory, automatic extraction, building-wide notes, key rotation (`key_version` is fixed at 1). A route `DELETE` during a running turn does not stop that turn from quoting a note it already recalled.

| Request | Success | Errors |
| --- | --- | --- |
| `GET /api/v1/assistant/memories` | 200 `{memories: [{id, scope, text, created_at, expires_at}]}`, newest first, `Cache-Control: no-store` | 500 a note does not open |
| `POST /api/v1/assistant/memories` with `{text, scope}` (3–300 characters, at least three words, no NUL) | 201 the memory | 403 residence scope without a residence; 422; 429 active or daily limit (`Retry-After` for the daily one) |
| `DELETE /api/v1/assistant/memories/{id}` | 204 | 404 foreign or missing |
| `DELETE /api/v1/assistant/memories` | 204: every note of the caller | — |

All four are 404 before authentication while memory is off, and none is in the OpenAPI schema. Downgrading migration 0012 refuses while any memory exists; 0013 refuses while memory tool audit rows exist.

### Assistant evaluations

`tests/evals/assistant_v1.jsonl` holds 21 golden cases (documents, bookings, tickets, devices, confirmation, safety) over the fixture documents in `tests/evals/assistant_documents/`, one of which carries planted instructions. Each case names the message, reference model steps, and expected outcomes: tools, forbidden tools, status, cited titles, device states, and audit decisions.

- **CI (deterministic):** `tests/test_assistant_evals.py` replays each case's reference steps through the real route with the provider simulator. It checks the server pipeline, not model quality. The same rubric must fail hijacked or uncited runs.
- **Live (opt-in, paid):** against the isolated AI stack started with `compose.ai.live.yaml`, after explicit authorisation:

```bash
uv run --locked python scripts/evaluate_assistant.py --allow-paid-requests
uv run --locked python scripts/evaluate_assistant.py --allow-paid-requests --judge --judge-model MODEL
```

`--judge` adds an LLM judge (needs `OPENAI_API_KEY`) that checks whether answers that cite passages are grounded in those snippets and answer the question. Answers grounded only in tool results are scored by the rubric alone. The script accepts loopback destinations only and prints case IDs, failed check names and fixed codes.

### Assistant load benchmark

```bash
uv run --locked python scripts/benchmark_assistant.py
uv run --locked python scripts/benchmark_assistant.py --scenario 200
```

Run the second only after reviewing the first and authorising it. The runner uses only `compose.assistant.bench.yaml`: a fresh `fastapi-bench-assistant-<tier>-<run-id>` project, random private credentials and TOTP key, no database or Redis host ports, and one API process with AI, RAG and the assistant on at `127.0.0.1:8006`. Port 8006 must be free. The provider is the simulator with 800–1,500 ms response and 100–200 ms embedding latency: turn times mostly measure that simulated latency, and no run reaches a paid API.

`scripts/seed_assistant_load.py` adds one resident per user, in buildings of up to 50 apartments, with the demo devices, bookings and one ready building document per building. `tests/load/assistant_locustfile.py` gives each user its own resident and 8–15 s think time, with task weights of 50 document questions, 15 bookings, 15 `GET /residence`, 10 thermostat changes and 10 door locks. A turn passes only with 200 and status `answered`; 429 and 503 are failures. Unlocks are left out: they need a code and allow 3 per hour.

| Tier | Users | Ramp | Duration | Gate |
| --- | --- | --- | --- | --- |
| `smoke` (default) | 20 | 5/s | 120 s | zero failures; turn p95 ≤ 5 s; residence p95 ≤ 500 ms |
| `200` | 200 | 10/s | 300 s | failures ≤ 1%; turn p95 ≤ 6 s |

Artifacts stay in the printed mode-0700 `artifacts/<project>/`: `run.json`, `setup.log`, `seed.log`, `containers.jsonl`, `locust.log`, Locust CSVs, `resources.jsonl`, `api.log` and `turns.csv` (turns by status, error and claim failure). `.env` and `residents.csv` hold real secrets and tokens (mode 0600). The runner also prints how many API log lines point at pool pressure or failed turns. Afterwards, including on failure, it runs `down` without `-v` on its own project: its containers and network go, its database volume and artifacts stay. (The CRUD runner below still uses `stop` and keeps its network.) It never starts a higher tier.

Recorded on the development laptop (Locust, the Docker VM and the target share it). Measured requests include `GET /residence`; answered turns count the whole run, warmup included.

| Run | Measured window | Measured requests / failures | Turn p95 / p99 | Turns answered | Peak API CPU |
| --- | --- | --- | --- | --- | --- |
| 20 users, 4 turns per process (before) | 89.9 s | 138 / 27 (503 capacity busy) | 2.8 / 3.0 s | 121 | 21% |
| 20 users, 64 turns per process | 89.8 s | 132 / 0 | 2.9 / 3.0 s | 150 | 21% |
| 200 users, 64 turns per process | 254.9 s | 3,757 / 0 | 2.9 / 3.0 s | 3,695 | 38% |

Turn latency is mostly two simulated model calls of 0.8–1.5 s each, plus one embedding for questions. Peak CPU is the highest 5 s `docker stats` sample, as a share of one core. No run logged database request failures, failed turns or timeouts. These runs show that one API process serves 200 simulated residents without 503s; they say nothing about a real provider's latency, rate limits or cost.

## Isolated 20-user benchmark smoke

Run this bounded setup from the repository root:

```bash
uv run --locked python scripts/benchmark_smoke.py
```

The runner uses **only** standalone `compose.bench.yaml`, a fresh
`fastapi-bench-smoke-<run-id>` project/network/database volume, and random private
credentials. It does not merge the normal Compose file or inherit its database
URL. PostgreSQL and Redis have no host ports; only the benchmark API is published
at `127.0.0.1:8001`. Port 8001 must be free. Resolved configuration, the running
database mount, and effective database/Redis driver destinations are checked
before seeding. No commands or data cleanup target the normal stack.

The default invocation seeds **20 accounts with 25 items each**, then starts the existing
Locust harness at **20 users**, **5 users/second**, for **90 seconds**. It allows
30 seconds of warmup and requires at least 45 measured seconds, zero whole-run
failures, p95 <=500 ms, p99 <=1000 ms, and the harness's correctness/user-count
checks. The runner requires a zero exit code and `CAPACITY PASS`. It never starts
a higher tier. The one-worker runtime, pool sizes, and cache limits are unchanged.

Artifacts are retained in the printed mode-0700 `artifacts/<project>/` directory:
`run.json`, `setup.log`, `seed.log`, `containers.jsonl`, `locust.log`, Locust CSVs,
and periodic benchmark-container `resources.jsonl` samples. `.env` and `users.csv`
contain real secrets/tokens and are mode 0600; do not commit or share them.
Inherited `COMPOSE_*`, `BENCH_*`, `LOAD_*`, and `LOCUST_*` environment overrides are
removed for child processes. The runner stops **only its own** benchmark containers
afterwards, including on failure, and preserves the database volume for review.
It does not delete volumes or truncate/flush either stack.

This smoke uses a host-side generator sharing hardware with the Docker VM. Record
the VM CPU/RAM allocation and other running workloads; the result proves only
this small workload on this machine, not production or 15k/30k-user capacity.
Review results before authorising any increased load. Each rerun creates a new
project and retains its resources; remove old benchmark resources only when
deliberately authorised, using the exact project and environment file recorded
in its artifacts, never the normal Compose file.

### Explicit isolated 1,000-user tier

After reviewing the smoke results and authorising this tier, run:

```bash
uv run --locked python scripts/benchmark_smoke.py --scenario 1000
```

This selects exactly one fixed scenario: 5,000 seeded accounts, 25 items each,
1,000 active accounts, 50 users/second spawn rate, 90-second warmup, and an
eight-minute total duration. It requires at least five measured minutes at the
full target, measured failure ratio <=1%, p95 <=500 ms, and p99 <=1,000 ms.
Whole-run failures remain reported, but the zero-whole-run-failures smoke setting
is not applied to this capacity tier. Review per-endpoint errors/latency and
generator CPU warnings alongside the aggregate harness verdict. A saturated
generator makes application capacity inconclusive.

The default remains the 20-user smoke. Unsupported scenario names and arbitrary user
counts are rejected. No subsequent tier is run automatically; review the results
before authorising any further load. This local run still shares the Mac and
Docker VM with other workloads, and does not establish production capacity.

The first authorised run passed at 1,000 users: 111,300 measured requests, zero
failures, p95 5 ms / p99 8 ms, and 389.92 measured seconds (about 285 requests/sec).
See [the 1,000-user report](docs/benchmark-1000-results.md) for isolation evidence,
resource observations, counter boundaries, and limitations. No higher tier was part of that run.

### Explicit isolated 2,500-user tier

After review and authorisation, run exactly this tier with:

```bash
uv run --locked python scripts/benchmark_smoke.py --scenario 2500
```

It seeds a fresh isolated corpus of 5,000 accounts and 125,000 items, matching
the 1,000-user tier's corpus size, and assigns 2,500 unique active accounts.
The fixed profile uses 100 users/second, a 120-second warmup, and 780 seconds
(13 minutes) total, requiring at least 600 measured seconds at the full target.
The aggregate measured failure limit remains 1%, p95 <=500 ms, and p99 <=1,000 ms.
Review per-endpoint correctness/latency and generator conditions as well. No API,
pool, database, or cache tuning is applied. The runner stops its isolated
containers afterwards and does not run a repeat or any higher tier automatically.

The first authorised 2,500-user run passed with 470,507 measured requests, zero
failures, p95 20 ms / p99 31 ms, and 659.89 measured seconds (about 713 requests/sec).
See [the 2,500-user report](docs/benchmark-2500-results.md) for isolation evidence,
resource observations, and limitations. Its containers are stopped, and no higher
tier was run.

## Local 20-user harness smoke using an existing stack

The following older recipe writes fixtures into the selected application stack.
Use the isolated runner above when the normal database must remain untouched.

Seed **real appended rows** and a signed, mode-`0600` CSV of `user_id,item_id,access_token`. This bypasses normal login only for deliberate load fixtures; prepared-token CRUD measurements **exclude Argon2/login throughput**.

```bash
docker compose --profile tools run --build --rm tools python scripts/seed_load.py \
  --users 20 --active-users 20 --items-per-user 2 --output /artifacts/load-smoke.csv

LOAD_USERS_FILE=artifacts/load-smoke.csv LOAD_ZERO_FAILURES=true LOAD_MAX_FAILURE_RATIO=0 \
  uv run --locked locust -f tests/load/locustfile.py --headless \
  --host http://127.0.0.1:8000 -u 20 -r 5 -t 90s \
  --csv artifacts/smoke --csv-full-history
```

An existing seed output is refused **before inserts**: choose a new path or explicitly use `--overwrite` to replace **only that CSV**. Neither choice deletes database rows; repeated seeding appends unique accounts/items. Batches commit independently, and a failure reports confirmed partial counts rather than pretending million-row atomicity. Tokens are minted after seeding finishes; the output reports earliest expiry. The seeder permits only effective asyncpg hosts `db`, `localhost`, `127.0.0.1`, `::1` and databases `app`/`app_test`, unless `--allow-remote` is explicit. The guard checks resolved SQLAlchemy driver connection arguments, including URL query parameters that override host/database, not just the URL authority. Remote opt-in is not a safety substitute for reviewing the destination.

Each VU receives a unique account. FastHttpUser waits 2–5 seconds between single-request tasks, weighted 70% detail GET, 20% first-page list GET (`limit=20`), and 10% description PATCH. These are request-selection weights, not guaranteed exact finite-run percentages. Responses are checked for shape/identity/ownership where applicable; a GET is not required to immediately reflect a concurrent PATCH. Registration/create/delete correctness is exercised separately.

The default warmup is 30 seconds, then stats reset once; at least 30 continuous measured seconds at the full target are required. Target users must be reached at the cutoff and maintained through measurement. Defaults gate aggregate measured failure ratio ≤1%, p95 ≤500 ms, p99 ≤1,000 ms, and nonzero measured requests. The command above additionally requires **zero whole-run failures**, including warmup/shutdown; that counter survives stats reset. Invalid duration, expired/insufficient fixtures, user-count drops, worker loss, or unhandled exceptions fail the run with a nonzero exit. Do not add Locust's `--reset-stats`: the harness owns the coordinated reset. These are **proposed benchmark gates, not established production SLOs**.

Each assigned token must cover the configured **entire run duration plus 120 seconds** before spawning; stale fixtures fail instead of refreshing credentials silently. Reseed promptly before runs; never edit JWT expiry client-side.

## Optional capacity measurement: separate target and generator

This is an unexecuted capacity recipe, not evidence of 15k/30k support. Stop escalation at a failed gate or generator/tunnel saturation. Use an isolated benchmark Compose project on the target, with a separate port and 7,200-second token lifetime:

```bash
python3 scripts/init_env.py --output .env.bench --api-port 8001 --token-ttl 7200
docker compose --env-file .env.bench -p fastapi-bench up --build -d --wait api
docker compose --env-file .env.bench -p fastapi-bench --profile tools run --build --rm tools \
  python scripts/seed_load.py --users 1000000 --active-users 30000 --items-per-user 1 \
  --output /artifacts/load-capacity.csv
```

If `.env.bench` already exists, initialization intentionally does not update it; inspect its nonsecret port/lifetime settings and select a new env filename if necessary. Shell environment can override Compose interpolation: do not carry a normal-stack `DATABASE_URL`, `API_PORT`, or secret override into this separate project. The shared host `artifacts/` directory means output names must also be unique across projects.

On a **separate generator host**, use at least four available CPU cores, this project's same locked dependencies and uv-managed Python 3.13 (`uv python install 3.13`, `uv sync --locked`). Establish authenticated SSH access to the target first. Set `BENCH_HOST` to its SSH alias and replace the example absolute project path with the target's real path:

```bash
BENCH_HOST=your-target-ssh-alias
TARGET_PROJECT=/absolute/path/to/project
mkdir -m 700 -p artifacts
# Use a new local filename if this one already exists; keep the fixture private.
scp "$BENCH_HOST:$TARGET_PROJECT/artifacts/load-capacity.csv" artifacts/load-capacity.csv
chmod 600 artifacts/load-capacity.csv

# Keep running in its own terminal; the target API remains loopback-only.
ssh -N -L 18001:127.0.0.1:8001 "$BENCH_HOST"
```

For **N=1000, then 5000, 15000, 30000**, run a fresh master and four fresh worker processes per tier. Set `N` in the master terminal for each invocation; do not automatically escalate past a failure. Start workers promptly in four separate terminals once the master is listening (missing workers/handshake time out after 30 seconds).

```bash
# Master terminal; change N for each tier and use a fresh CSV prefix on reruns.
N=1000
LOAD_WARMUP_SECONDS=180 LOAD_MIN_STEADY_SECONDS=600 \
  uv run --locked locust -f tests/load/locustfile.py \
  --master --master-bind-host 127.0.0.1 --expect-workers 4 --headless \
  --host http://127.0.0.1:18001 -u "$N" -r 250 -t 15m --stop-timeout 30 \
  --csv "artifacts/capacity-$N" --csv-full-history
```

```bash
# Four worker terminals: set I to 0, 1, 2, 3 respectively, exactly once each.
I=0
LOAD_USERS_FILE=artifacts/load-capacity.csv LOAD_WORKER_INDEX="$I" LOAD_WORKER_COUNT=4 \
  LOAD_WARMUP_SECONDS=180 LOAD_MIN_STEADY_SECONDS=600 \
  uv run --locked locust -f tests/load/locustfile.py --worker \
  --master-host 127.0.0.1 --host http://127.0.0.1:18001
```

Do **not** use `--processes`: inherited duplicate shard indices would violate account uniqueness. Workers use the same CSV and select rows by row index modulo worker count; the master consumes no accounts. It sends target/duration for expiry and sufficiency preflight, rejects duplicate/missing indices or mismatched fixtures, and coordinates the warmup reset and aggregate evaluation. Worker failures abort the global run rather than silently reducing capacity. Global targets are not evaluated independently on workers.

A tier is 15 minutes total: 3 minutes warmup/ramp at 250 users/second, nominally 12 minutes measured, requiring at least 10 continuous measured minutes at full target. Default measured failure/latency gates above apply; zero whole-run failures is optional via `LOAD_ZERO_FAILURES=true` on the master. Four workers are an initial generator configuration, **not a capacity guarantee**. Fixtures must pass run-plus-120-second expiry preflight for every tier. If they become too old, append 30,000 fresh users/active users to the benchmark database with a **new output filename**, preserving the million-row corpus, then privately copy that fixture and update `LOAD_USERS_FILE` in every worker. Do not delete the corpus or fake token renewal.

During each tier retain Locust CSV/history and final gate logs (target/minimum active users, measured duration, failures, whole-run failures, p95/p99, generator CPU warning). Record per-endpoint RPS/latency/errors, achieved VUs, p50, generator CPU/memory/network and SSH tunnel CPU/network; a saturated generator or tunnel invalidates an API-capacity conclusion. On the **target**, capture these lightweight observations while traffic is running:

```bash
docker stats --no-stream $(docker compose --env-file .env.bench -p fastapi-bench ps -q)
docker compose --env-file .env.bench -p fastapi-bench exec -T db psql -U app -d app -c \
  "SELECT state, wait_event_type, wait_event, count(*) FROM pg_stat_activity GROUP BY 1,2,3 ORDER BY 1,2,3;"
docker compose --env-file .env.bench -p fastapi-bench exec -T redis redis-cli INFO stats
docker compose --env-file .env.bench -p fastapi-bench exec -T redis redis-cli INFO memory
```

Record hardware, images/settings, corpus size, process count, tunnel arrangement and bottlenecks alongside results. Idle connected VUs alone, a short ramp-only run, or a successful harness exit on a laptop does not establish production throughput or the requested capacity.

## Recorded verification and limits

Initial implementation evidence:

The implementation run observed **138 passing pytest tests, 91% application coverage, no warnings**, using isolated actual PG19 RC1/Redis; Ruff passed, and Alembic reported no schema drift. Live HTTP smoke passed normal operation, Redis-down database fallback/degraded readiness, database-down liveness 200/readiness 503/bounded write 503, and read/write/delete recovery after restart. Private initialization permissions/no-overwrite behavior and a real 20-user/40-item seed were exercised. Expired/insufficient fixtures, duplicate worker shards, and missing-worker startup were observed to exit nonzero; missing-worker startup was fixed and re-exercised with a bounded wait.

The cleanup/input/configuration hardening pass established failing regressions before each fix, then passed **277 sequential tests with 91% application coverage** through isolated `compose.test.yaml`. Focused checks, Ruff lint/format, real HTTP NUL rejection/atomicity, a PostgreSQL connection with malformed unused components, and initializer CLI permission rejection/preservation also passed. Unsafe cleanup destinations were tested without connections or truncation.

Final local observations (generator and target on the **same laptop**):

| Harness run | Target / total duration | Measured window | Measured requests / failures | p95 / p99 |
| --- | --- | --- | --- | --- |
| Local single generator | 20 VUs / 90 s | 59.93 s | 332 / 0 | 11 / 13 ms |
| Four distributed workers | 20 VUs / 90 s | 59.93 s | 348 / 0 | 11 / 12 ms |

Recorded final CSV prefixes are `artifacts/local-final` and `artifacts/distributed-final` (fixture `artifacts/load-final.csv`); the repeatable example above uses `load-smoke.csv` and `smoke`. These small runs prove local workflow/harness operation only. **No million-row seed, 15,000-user tier, or 30,000-user tier was executed.** No production capacity, availability SLO, or authentication throughput claim follows from these observations.

Assistant hardening pass (`feature/assistant-hardening`): **1,058 passing main-suite tests (63 skipped) and 1,121 passing AI-suite tests**; Ruff and `alembic check` clean; the Alembic round trip passed, and the 0006 downgrade refuses while step-up audit rows exist. `scripts/smoke_assistant.py` (step-up flow) and `scripts/smoke_edge.py` passed against the isolated AI QA stack; the edge with the normal and AI-local stacks was checked with `docker compose config` only. The paid live assistant eval passed 21/21 before and again after the step-up change (no judge). The CRUD `benchmark_smoke.py` still passes (335 measured requests, 0 failures, p95 13 ms, p99 14 ms); assistant benchmark results are in [Assistant load benchmark](#assistant-load-benchmark).

GraphRAG and memory pass (`feature/graph-memory`): **1,313 passing main-suite tests (87 skipped) and 1,400 passing AI-suite tests**; Ruff and `alembic check` clean. Migrations 0011, 0012 and 0013 round-trip, and each downgrade refuses while its data exists (graph rows or a graph error code, memories, memory tool audit rows). `scripts/smoke_assistant.py` (now with remember, recall and forget) and `scripts/smoke_rag.py` passed against the isolated AI QA stack with the simulator. No paid model call was made: the `evaluate_rag.py --graph` comparison and live memory behaviour are not measured.

Paid graph eval (2026-11-22, `gpt-5.6-terra`, one repeat, the two multi-hop cases): `vector` and `graph` scored the same, 0/2 runs passing each, retrieval hit 50%, answer correct 50%, mean 1,262 tokens, with identical tokens per case, so expansion added no passage. Extraction stored 71 nodes and 13 edges over the 5 fixture documents with no extraction error, but none of the edges the cases need (storage cage to cellar, cellar to opening hours, guest apartment to the service that runs it). The limit is extraction recall, not expansion. With n = 1 per case this is a wiring and direction check, not a quality measure.

Diagnostic rerun the same day with `AI_RAG_GRAPH_BATCH_SIZE=1`: every chunk of every fixture document got nodes (before, mostly only the first chunk of each 4-chunk call), and the bridging edges appeared (storage cages located in the cellar, guest apartment operated by the concierge service). The graph configuration added a passage on the storage case (graph passages 1 vs 0; mean tokens 1,724 vs 1,472), but both cases still failed: the two documents typed "cellar" differently, and node keys then included the type. Fixed: one chunk per call is now the default and node keys are type-free. The citations check fails with the graph off too; that is a separate problem.

Confirming run with the fix (defaults: one chunk per call, type-free keys): `graph` reached retrieval hit 100% and answer correct 100%, against 50% and 50% for `vector`; it added one passage on each case (mean tokens 2,052 vs 1,472). `multi-hop-guest-apartment` passed with the graph and failed without it. `multi-hop-storage-hours` gave the right hours both ways but failed its citations check: the answer cited the cellar hours, not the passage that puts the storage cages in the cellar. That is the answer prompt, not the graph. One repeat per case: a direction, not a quality measure.

With prompt `rag-answer-v3` (cite each linking fact): `multi-hop-storage-hours` passed in both configurations, citing both the storage-cage and the cellar-hours passages. `multi-hop-guest-apartment` failed in both: this run's extraction stored "guest apartment operated by concierge service" but no edge from the concierge service in the moving guide (the model linked the parcel room to the main entrance instead). The moving guide held a concierge node (its chunk was not checked), but expansion then followed edges only, so it was not reached. Mentions were added after this run.

Run with mentions (2026-11-23, same two cases, one repeat): `graph` passed both cases, `vector` one (`multi-hop-guest-apartment` failed). Retrieval hit and answer correct were 100% against 50%; mean tokens 1,996 against 1,538; one graph passage per case. This run does not show that mentions help: extraction stored "concierge service located in parcel room" this time, so the edge alone links the chunk. It shows no regression on these cases.
