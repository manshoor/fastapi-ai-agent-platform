# Owner-scoped FastAPI CRUD

A local, runnable API with built-in accounts, PostgreSQL persistence, async SQLAlchemy, and Redis detail caching with **at most five seconds of cache staleness** under the assumptions below. An opt-in, isolated [durable item-rewrite stack](#durable-item-rewriting) adds Apache Kafka, an async OpenAI worker, and local OpenTelemetry verification. Python 3.13, uv, Docker Compose, and a reproducible Locust harness are included. The requested 15,000–30,000 concurrent-user target is **not verified capacity**.

> **Local development/measurement only.** PostgreSQL is deliberately pinned to the user-selected prerelease `postgres:19rc1-bookworm`. Accept its prerelease risks; do not treat this stack as a production deployment. HTTP binds to loopback, not TLS. There is no production reverse proxy, rate limiter, abuse protection, or deployment automation.

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
| `POST /api/v1/auth/login` | 200 | Form `username`, `password`; returns `access_token`, `token_type="bearer"`, `expires_in`. |
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
- Duplicate normalized email: `409 {"detail":"Email already registered"}`. Wrong password/unknown email: identical `401 {"detail":"Invalid credentials"}`. Missing, malformed, expired, or invalid bearer token: `401 {"detail":"Invalid authentication credentials"}`. Every 401 includes `WWW-Authenticate: Bearer`.
- Known transient database failures, pool exhaustion, or hash admission exhaustion return `503 {"detail":"Service temporarily unavailable"}` with `Retry-After: 1`; programming/unexpected integrity errors remain 500. Writes are **not automatically retried**: a lost response does not prove a mutation failed to commit.

### Authentication and local security boundary

JWTs use HS256 with a generated secret decoded from hex (at least 32 bytes), a fixed algorithm allowlist, canonical UUID subject, and required/verified `sub`, `iat`, `exp`, `iss`, and `aud` claims. Default lifetime is 900 seconds. Authorization is stateless: valid tokens resolve an owner UUID without an account lookup, including on cache hits. There are **no account deletion/disable, password-change, logout/revocation, or refresh-token endpoints**; issued tokens remain valid until expiry. Do not infer revocation from clearing a browser token.

Registration/login have bounded password-hashing work, not public-service abuse controls. Keep the loopback binding. The optional remote benchmark below uses authenticated SSH forwarding; it does not turn this into a production TLS deployment. Treat `.env` and CSV bearer fixtures as secrets, never publish them or paste them into reports.

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
| `JWT_ISSUER` / `JWT_AUDIENCE` | `fastapi-crud` / `fastapi-crud-api`. |

Each process owns its engine/Redis pool; each request gets its own session. Cache hits do not acquire a database connection. PostgreSQL allows 100 connections: budget **replicas × API workers × 10**, retaining **at least 20** connections for migrations, tools, and other non-API work. More virtual users do not mean more database connections. Do not scale workers past that budget or treat this as autoscaling guidance.

### Cache consistency

Only successful owner-scoped detail reads are cached, using `item:v1:{owner_uuid}:{item_uuid}`. Lists, missing rows, and authentication are not cached. Writes commit first; PATCH/DELETE then best-effort invalidate. Redis errors fall back to PostgreSQL, and failed invalidation never turns a committed write into an apparent rollback. Corrupt cached values are misses.

On a miss, the service reads Redis **TIME before the database SELECT**, computes an absolute deadline at that time plus `CACHE_TTL_MS`, and fills using **SET PXAT**, not a fresh relative TTL. Thus a reader paused before a concurrent mutation cannot resume after invalidation and grant the old row a new five-second lifetime. If TIME fails, that read does not populate the cache.

The maximum five-second staleness guarantee applies to **new cache reads after a mutation**, assumes a normally advancing Redis clock, and does not bound delivery of responses already in flight. Concurrent reads do not promise immediate read-your-writes. Clock rollback or restoring old Redis snapshots is outside the contract; persistence is disabled intentionally.

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

The initializer creates a separate `.ai-local/.env` with random credentials and mode `0600`, plus private `0700` directories. Existing credentials are preserved; unsafe ownership, modes, links, and normal/escaping paths are rejected rather than repaired. `.ai-*/` is excluded from version control and Docker build context. Settings read process environment only, never dotenv implicitly.

`compose.ai.yaml` is standalone: **never merge it with normal, benchmark, or QA Compose**. It uses database `app_ai`, Redis DB 2, and project-owned PostgreSQL/Kafka volumes. Only the API publishes `127.0.0.1:8002`; dependencies and simulator/telemetry control endpoints have no host ports. An API-only ingress network enables loopback publication; all dependency traffic stays on the internal network. The local image uses the test target to include simulators and operational scripts. This remains a development deployment, not production TLS, abuse protection, or Kafka HA.

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
upload --> API: auth -> bounded multipart read -> rag_documents(pending) + rag_outbox, one transaction -> 202
                 dispatcher --> Kafka rag.document-ingest.requested.v1 --> rag-worker
                 rag-worker: lease -> parse (PDF in a sandboxed child process) -> chunk -> embed -> rag_chunks -> ready
query  --> API: auth -> query slot -> admit (rate limit) -> embed question -> owner-scoped pgvector KNN
                 -> neighbor windows (+ optional rerank) -> claims, each with a quote
                 -> server checks every quote and number (+ optional verifier) -> 200
```

Start and smoke the local stack (simulator only, no paid calls):

```bash
python3 scripts/init_ai_env.py
docker compose --env-file .ai-local/.env -p fastapi-ai-local -f compose.ai.yaml up --build -d --wait api worker rag-worker dispatcher
uv run --locked python scripts/smoke_rag.py
```

The smoke uploads a Markdown file and a generated PDF, waits for `ready`, checks that the answer cites the right document, that a second account sees nothing, and that deletion turns the answer into `no_context`. It prints no tokens or content.

### RAG HTTP contract

All routes need `Authorization: Bearer` and return `Cache-Control: no-store`.

| Route | Success | Notes |
| --- | --- | --- |
| `POST /api/v1/rag/documents` | 202; 200 on replay of a finished upload | Multipart `title` (1–200 chars) + one `file` (`.txt`, `.md`, `.pdf`). `Idempotency-Key` header. `Location` header. |
| `GET /api/v1/rag/documents` | 200 | Owner's documents, newest first. |
| `GET /api/v1/rag/documents/{id}` | 200 | Status, chunk and page counts, error code, timestamps. Never the content. |
| `DELETE /api/v1/rag/documents/{id}` | 204 | Removes the document, its chunks, outbox rows **and its idempotency record**: the same key can then create a new document. |
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
- PDF parsing runs in a child process with CPU, memory, and file-size limits, a scrubbed environment, and a wall timeout. The Linux container is the supported runtime; on macOS the limits cannot be set, so PDF parsing fails closed.

Defaults: text 1 MiB; PDF 5 MiB, 300 pages, 30 s, 1 GiB; 1,000,000 extracted characters; chunks of 1,200 characters with 200 overlap, at most 1,200 per document; 64 chunks per embedding request. 50 documents and 5 unsettled per owner; 100 unsettled globally; one upload per owner and 4 uploads and 4 queries per API process; 10 queries per 60 s per owner. Embeddings use `text-embedding-3-small` (1,536 dimensions) in pgvector 0.8.7 with an HNSW cosine index. Pools: API 10 + dispatcher 2 + worker 4 + rag-worker 4 = 20 of PostgreSQL's 100.

With `compose.ai.live.yaml`, the rag-worker pays for document embeddings and the API pays for question embeddings and answers. The same paid-use authorisation applies.

### Answer grounding

Retrieval can find the right passage while the model still misreads it. So the answer is built only from claims the server can check (prompt `rag-answer-v2`):

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

### RAG evaluation

`POST /api/v1/rag/query/diagnostics` is an eval-only route. It is 404 (before authentication) and hidden from the schema unless `RAG_DIAGNOSTICS_ENABLED=true`, which only `compose.ai.yaml` sets. It runs the normal query (same owner scope, rate limit and usage row) and also returns, to the owner only, the passages sent to the model, each claim with its check result, verifier verdicts, the refusal reason and usage. It takes optional `overrides`: `neighbor_chunks`, `relevance_margin`, `rerank`, `reasoning_effort`, `verifier`. An override set that does not fit the deadline is 422.

`tests/evals/rag_v1.jsonl` holds 32 golden cases over the five documents in `tests/evals/rag_documents/` (single fact, numeric, condition, multi-passage, heading, split across chunks, distractor, not covered, injection). Each case is scored on retrieval, citations, status, facts and forbidden strings, so a wrong answer after a correct retrieval is counted on its own (`wrong_given_hit`). CI checks the data and the scorer only; the simulator cannot judge answers.

Live, paid, after explicit authorisation, against the AI stack started with `compose.ai.live.yaml`:

```bash
uv run --locked python scripts/evaluate_rag.py --allow-paid-requests --repeat 3 --config base --config checked=verifier:on --output artifacts/rag-eval.json
uv run --locked python scripts/evaluate_rag.py --allow-paid-requests --judge --judge-model MODEL
```

`--judge` first runs six judge controls with known verdicts and exits 2 if the judge gets one wrong. The script accepts loopback destinations only (not ports 8000 or 8001) and prints case IDs, check names, rates and fixed codes; `--output` saves scores without text, mode 0600. Against the simulator it exits 1 by design: it only checks the wiring.

## Resident assistant (tool-using agent)

A resident asks one question or request per turn. A LangGraph agent answers from their building's documents with citations, reads live data, and acts on **simulated** devices (rows in PostgreSQL; no hardware) through a fixed allowlist of typed tools. The server, not the prompt, enforces identity, permissions and limits. It is off unless `ASSISTANT_ENABLED=true`, which requires RAG. `compose.ai.yaml` enables it, with the demo setup route (`ASSISTANT_DEMO_ENABLED=true`).

```
POST /api/v1/assistant/messages
  JWT -> resident (membership -> apartment, building) -> 403 if none
  -> one running turn per user, bounded per process -> admit (rate limit, turn row)
  -> LangGraph:  model --tool call--> tools --> model ... --reply--> END
                   at most 4 model calls; a call starts only if its timeout fits the 120 s budget
                 tools: allowlist -> validate arguments -> server checks
                        -> one transaction: write + audit row
  -> grounding rule -> usage row -> 200

POST /api/v1/assistant/actions/{id}/confirm
  JWT -> one transaction: own live pending unlock -> still a resident -> unlock -> audit -> 200
```

Smoke it against the simulator (no paid calls), with the AI stack above running:

```bash
uv run --locked python scripts/smoke_assistant.py
```

### Assistant HTTP contract

All routes need `Authorization: Bearer` and return `Cache-Control: no-store`.

| Route | Success | Errors |
| --- | --- | --- |
| `POST /api/v1/assistant/messages` with `{message}` (1–2,000 chars, no NUL) | 200 `{turn_id, status, answer, citations[], actions[], pending_actions[]}` | 403 no residence; 422; 429 turn limit or a turn already running (`Retry-After`); 503 disabled, busy, or provider rate-limited with nothing committed; 502/504 provider failure, timeout or budget with nothing committed |
| `POST /api/v1/assistant/actions/{id}/confirm` | 200 `{id, action, status: "confirmed", device: {name, locked}}` | 403 no residence; 404 foreign or missing; 409 expired or already decided; 503 disabled |
| `GET /api/v1/residence` | 200 building, apartment, devices, upcoming bookings, newest tickets, live pending unlocks | 403; 503 |
| `POST /api/v1/residence/demo` | 201: a demo building, apartment `1A`, two thermostats, two door locks, an alarm, two bookings; the caller becomes resident and manager | 404 when the demo flag is off (before authentication); 409 already a resident |
| `POST /api/v1/rag/documents?building_id=` | As for RAG uploads; the document belongs to the building | 404 unless the caller manages that building |

The final reply is claims, checked like RAG answers (prompt `assistant-v2`): each passage claim needs a quote from a passage handed out this turn, and a `tool` claim needs a successful non-search tool result. `citations[]` items carry `quotes`. `status` is `answered` (every claim checked), `no_context` (fixed text: nothing grounded the answer, or a claim failed with nothing committed), or `incomplete` (fixed text: the turn stopped after committing a change, including a failed claim after a write, `error_code` `claim_unverified`; `actions` lists what happened). The failed check is stored in `assistant_turns.claim_failure`. `actions[]` items are `{tool, decision, result, device, ticket_id, pending_action_id}`, built by the server from tool results, never from model text.

### Tools and rules

| Tool | Arguments | Effect | Limit |
| --- | --- | --- | --- |
| `search_documents` | `question` | Top passages from the resident's building documents, labels P1–P8 per turn | 8 passages per turn |
| `list_bookings` | none | Up to 10 upcoming bookings of the apartment | none |
| `create_ticket` | `category`, `summary` | Opens a ticket; the same category within 10 minutes returns the existing one | 3 per user per 24 h |
| `set_thermostat` | `device`, `celsius` (16–26, steps of 0.5) | Sets the target | 10 device actions per user per hour |
| `lock_door` | `device` | Locks the door | shares the device-action limit |
| `unlock_door` | `device` | Creates a 60 s pending unlock; never unlocks | 3 per user per hour |

- **Identity from the server.** The model supplies only tool names and values. User, apartment and building come from the caller's token and membership. Extra arguments are rejected.
- **Checked, audited writes.** Each write runs in one transaction: per-user lock, membership share lock, limit count on the database clock, a write guarded by apartment, device name and kind, and an `assistant_tool_calls` audit row. Unknown tool names are audited as `unknown`.
- **Only the resident confirms.** No tool can confirm an unlock. The alarm exists but no tool can reach it, and the model never sees it.
- **Untrusted text.** The message, tool results and passages are data, never instructions. A hijacked model can at worst create an unconfirmed unlock request.
- **No content stored or traced.** Turns and audit rows hold counts, IDs and fixed codes; ticket summaries live only in `tickets`. The graph runs with LangSmith tracing disabled; spans `assistant.turn` (Langfuse `agent`) and `assistant.tool` (Langfuse and LangSmith `tool`) carry IDs and counts only.
- Defaults: 10 turns per 60 s per user, one running turn per user, 4 turns per API process.

### Assistant evaluations

`tests/evals/assistant_v1.jsonl` holds 21 golden cases (documents, bookings, tickets, devices, confirmation, safety) over the fixture documents in `tests/evals/assistant_documents/`, one of which carries planted instructions. Each case names the message, reference model steps, and expected outcomes: tools, forbidden tools, status, cited titles, device states, and audit decisions.

- **CI (deterministic):** `tests/test_assistant_evals.py` replays each case's reference steps through the real route with the provider simulator. It checks the server pipeline, not model quality. The same rubric must fail hijacked or uncited runs.
- **Live (opt-in, paid):** against the isolated AI stack started with `compose.ai.live.yaml`, after explicit authorisation:

```bash
uv run --locked python scripts/evaluate_assistant.py --allow-paid-requests
uv run --locked python scripts/evaluate_assistant.py --allow-paid-requests --judge --judge-model MODEL
```

`--judge` adds an LLM judge (needs `OPENAI_API_KEY`) that checks whether answers that cite passages are grounded in those snippets and answer the question. Answers grounded only in tool results are scored by the rubric alone. The script accepts loopback destinations only and prints case IDs, failed check names and fixed codes.

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
