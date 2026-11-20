# Repository Guidelines

## Project Overview

This is an async FastAPI application with built-in accounts and user-owned item CRUD, PostgreSQL persistence, and disposable Redis detail caching. The supported deployment is local Docker Compose, using the explicitly chosen PostgreSQL 19 prerelease (`postgres:19rc1-bookworm`). The 15,000–30,000 concurrent-user goal is a measurement target, not demonstrated capacity or an RPS promise. See [README.md](README.md) for API usage, operational caveats, and the full load-testing recipe.

## Architecture & Data Flow

- `app/main.py:create_app()` assembles routers, a 32 KiB ASGI body limit, private validation errors, and transient-database-error handling. Lifespan owns one engine/session factory, Redis client/cache wrapper, and initialized password service per process; application state contains infrastructure, not business records.
- PostgreSQL is the source of truth. Registration hashes before opening a database transaction; login reads credentials and releases its transaction before verifying. For an existing account, login checks its password failures before Argon2 (429 at `AUTH_PASSWORD_FAILURE_LIMIT`, 10 per 900 s, rows in `auth_failures`) and records one after a wrong password; unknown emails record nothing. Argon2 runs in AnyIO worker threads with two admitted operations by default, a 100 ms queue timeout, and cancellation-safe permit ownership until the thread finishes.
- HS256 JWT verification supplies the owner UUID without a database lookup. Every item detail read/update/delete filters by both owner and item ID; lists always filter by owner. Never accept ownership from JSON. Foreign and missing items share the same 404. Tokens default to 900 seconds; there are no refresh, logout/revocation, account deletion, or password-change endpoints.
- Detail GET uses `item:v1:{owner_uuid}:{item_uuid}` cache-aside keys. A hit avoids PostgreSQL entirely. On a miss, obtain Redis `TIME` **before** the READ COMMITTED database read, release the database transaction, then fill using that original absolute `PXAT` deadline. `CACHE_TTL_MS` defaults to 5000 and cannot exceed 5000; never restart a relative TTL after a delayed read. The bound covers new cache reads, not responses already in flight, and assumes a normally advancing Redis clock and no restored snapshots.
- Mutations return committed database state. Commit **before** invalidating cache on PATCH/DELETE; create does not fill cache. Expected Redis failures fall back to PostgreSQL and failed invalidation must not undo or misreport a commit. Lists are database-only, using descending immutable `(created_at, id)` keyset pagination, not OFFSET or `updated_at` ordering.
- Each request gets its own `AsyncSession`; never share a session across requests or concurrent tasks. Keep transactions short and release read connections before cache work. The per-process database pool defaults to 10 with `max_overflow=0`, one-second acquisition/connect bounds, and a two-second statement timeout. Redis defaults to 50 connections, 100 ms socket timeouts, zero retries, and a 200 ms per-operation deadline. Do not add automatic write retries.
- `/health/live` is dependency-free. Readiness checks real database/Redis reachability within one second: database failure is 503; Redis-only failure is degraded 200. Schema readiness comes from the successful one-shot migration dependency, not the health query.
- RAG (`app/rag/`, off unless `RAG_ENABLED` and `AI_ENABLED`) mirrors the rewrite pipeline: upload commits `rag_documents` + `rag_outbox` in one transaction; the dispatcher publishes; `python -m app.rag.worker` leases, parses, chunks, embeds, and stores `rag_chunks` (pgvector, HNSW cosine). Filter every document, chunk and query statement by owner. The upload route has no form parameters so authentication runs before any body read; `BoundedBodyMiddleware` streams only that path under its own ceiling. Fence worker writes by status, attempt, lease token, generation and database-clock expiry. Parse PDFs only in the sandboxed child (`app/rag/pdf_extract.py`). Queries run `has_context` before any provider call, admit under a per-owner advisory lock counting `rag_queries`, run the LCEL chain under `langsmith.tracing_context(enabled=False)`, and accept only quoted claims: `app/rag/verification.py` checks every quote (word edges) and number against the exact passage text sent, all or nothing, else `no_context` with a fixed `no_context_reason`. Rerank, verifier and reasoning effort are off by default; each paid call must fit `query_deadline_floor`. `input_tokens`/`output_tokens` sum every completed chat call. `/rag/query/diagnostics` is eval-only (404 before auth unless `RAG_DIAGNOSTICS_ENABLED`). Never store question or answer text; never hold a transaction across provider, Kafka, or child-process I/O.

- The resident assistant (`app/assistant/`, off unless `ASSISTANT_ENABLED`, which requires RAG) runs a LangGraph `StateGraph` with no checkpointer; per-turn state goes in `config["configurable"]`. Building documents (`rag_documents.building_id`, uploaded by a building manager) are the one exception to owner filtering: the assistant searches them by the resident's building from their membership, never from model output. Tools are a fixed allowlist with strict argument models; write tools run lock, membership, limit, guarded write and audit row in one transaction on the database clock. `unlock_door` only creates a pending action that the resident confirms through `/api/v1/assistant/actions/{id}/confirm` with an authenticator code; no tool may confirm, and no tool reaches the alarm. Step-up (`app/assistant/step_up.py`) locks the credential row, checks the failure count, accepts only a step after `last_used_step` on the database clock, and commits the spent step before the claim. Secrets are AES-GCM sealed under `ASSISTANT_TOTP_KEY` with the user id as associated data; never log, trace or store secrets, codes or passwords. Every assistant stack needs that key: `AI_ASSISTANT_TOTP_KEY` (appended to an existing `.ai-local/.env` by `scripts/init_ai_env.py`, which changes nothing else), a test-only key in `compose.ai.test.yaml`, and a per-run benchmark key. Turns and audit rows hold no message, argument, passage or answer text. Final replies are claims checked with the same verification module: passage claims quote a `P` passage, tool claims quote the server-written fact of a `T` label (`ToolContext.facts`), and denials get no label; a failed claim is `no_context`, or `incomplete` with `claim_unverified` after a committed write. A turn never hides a committed write: it returns `incomplete`. One running turn per user holds across processes: admission checks for an `admitted` turn inside its lease under the per-user advisory lock. `ASSISTANT_CONCURRENCY` (64) caps running turns per process; past it, 503 at once.

## Key Directories

- `app/`: feature routers use direct SQLAlchemy statements; `security.py`, `cache.py`, and `db.py` own their respective boundaries. Extend these patterns rather than introducing a generic repository/service framework.
- `migrations/`: explicit Alembic schema history and async migration setup with `NullPool`. Change models and add a migration together; never use runtime `create_all()` or migrate inside API workers.
- `docker/postgres/`: PostgreSQL 19 rc1 image with pgvector 0.8.7; every Compose `db` service builds it. Migration `0003` creates the `vector` extension.
- `tests/`: real PostgreSQL/Redis integration regressions; `tests/load/locustfile.py` is a separate concurrent-user harness, not a correctness-suite substitute.
- `compose.bench.yaml` and `scripts/benchmark_smoke.py`: standalone, fresh-project benchmark with private credentials/volumes, no database/Redis host ports, and API port 8001. Defaults to the 20-user smoke; explicit `--scenario 1000` or `--scenario 2500` selects only that fixed tier after authorisation. Other tiers/arbitrary counts are rejected. The runner validates destinations before seeding, stops only its own containers, preserves artifacts/data, and never escalates load automatically. Do not merge it with normal Compose or use normal data.
- `compose.assistant.bench.yaml` and `scripts/benchmark_assistant.py`: fresh-project assistant benchmark on port 8006 against the latency simulator (seeder `scripts/seed_assistant_load.py`, `tests/load/assistant_locustfile.py`, gates shared through `tests/load/capacity.py`). Defaults to the 20-user smoke; `--scenario 200` only after authorisation. Teardown is `down` without `-v`: containers and network go, the volume and artifacts stay.
- `compose.edge.yaml`, `docker/edge/nginx.conf`, `scripts/init_edge_tls.py` and `scripts/smoke_edge.py`: optional local TLS edge overlay (nginx on 127.0.0.1:8443, per-address limits) for `compose.yaml` or `compose.ai.yaml`. `.edge-local/` holds the server key; never commit it. Use prefix locations: the API also routes a path with a trailing `%0A`. Keep nginx timeouts and body sizes in step with the assistant and RAG deadlines and upload limits.
- `scripts/`: private environment initialization, real HTTP smoke checks, and append-only load-fixture seeding. `artifacts/` holds private generated bearer-token CSVs and measurement output; never commit or echo credentials.

## Development Commands

Run from the repository root with Docker available. Use uv-managed Python 3.13 rather than the host default:

```bash
uv python install 3.13
uv sync --locked
python3 scripts/init_env.py
docker compose config --quiet
docker compose build
docker compose up --build -d --wait api
curl --fail http://127.0.0.1:8000/health/ready
uv run --locked python scripts/smoke.py
```

`init_env.py` exclusively creates mode-0600 `.env` with random secrets and prepares host-owned `artifacts/` as mode 0700; existing credentials are preserved. Compose runs `alembic upgrade head` before the API. The smoke script creates real accounts/items and checks owner isolation and bounded cache convergence; use `--base-url` if the API port differs.
Existing `artifacts/` must be a nonsymlink directory owned by the current UID with exact mode 0700. Initialization rejects unsafe paths before creating credentials; it never repairs existing ownership or permissions automatically.

## Code Conventions & Common Patterns

Use typed Python, snake_case functions/modules, PascalCase models, and async network/database calls. Ruff targets Python 3.13, line length 100, and `E,F,I,UP,B`. Reuse `Annotated[..., Depends(...)]` dependencies and Pydantic response models; keep infrastructure construction in lifespan.

Schemas reject extra fields. Normalize validated email consistently, but never strip or normalize passwords. PATCH uses `exclude_unset=True`: omitted fields stay unchanged, explicit null and empty patches are invalid. Preserve owner-scoped `RETURNING` mutations, database timestamps, named uniqueness constraints/indexes, and rollback before translating known conflicts. Catch only expected dependency failures; programming failures must not become successful fallbacks. Never log passwords, hashes, JWTs, Authorization headers, connection strings, SQL parameter values, or raw validation inputs.
Shared `Title` and `Description` validators reject NUL characters rather than stripping them; keep create/PATCH 422 errors sanitized and preserve valid Unicode.

## Important Files

- `app/config.py`: authoritative settings validation and shared SQLAlchemy URL resolver. Select a nonempty explicit `DATABASE_URL` before validating unused `DB_*` components; empty URL uses component fallback, invalid nonempty URL fails.
- `app/models.py`, `app/schemas.py`, and `migrations/versions/`: persistent schema, public request/response contracts, and migration history; keep their changes coordinated.
- `compose.yaml`, `compose.test.yaml`, and `Dockerfile`: local service lifecycle, isolated QA topology, and locked runtime/test image targets. PostgreSQL 19 mounts at `/var/lib/postgresql`, not an older version's data path.
- `pyproject.toml`, `uv.lock`, `.python-version`, and `.env.example`: dependency/tooling contracts and non-secret configuration template. Update the lock intentionally when changing dependencies; do not replace generated secrets with fixed normal defaults.

## Runtime/Tooling Preferences

Use Python 3.13 and uv (`uv sync --locked`, `uv run --locked`); no Node tooling is needed. Settings read process environment only, with **no implicit dotenv loading**. Compose interpolates `.env` and explicitly supplies container environment; host commands needing settings must receive their own environment.

The runtime image installs no development dependencies, runs nonroot, and starts one Uvicorn worker. API publication is loopback-only; database/Redis have no host ports. The edge overlay adds local TLS and per-address limits, but through Docker's port proxy all host clients may share one address, so locally those limits act as global caps. This is not production TLS or abuse protection. Before increasing replicas/workers, budget `replicas × workers × DB_POOL_SIZE` against PostgreSQL's 100 connections, reserving at least 20 outside application pools.

Load seeding appends rows and refuses nonlocal/non-app effective driver destinations unless explicitly opted in. Fixture CSVs contain real signed tokens; keep them mode 0600. The Locust workload uses a unique account per VU, 2–5 second think time, and 70/20/10 detail/list/PATCH weights. Distributed workers require distinct explicit shard indices; never use inherited duplicate indices through `--processes`. Follow README for fixture expiry checks, distributed commands, capacity gates, and resource observations; local smoke results are not production capacity evidence.

## Testing & QA

Use the standalone isolated test Compose project, not the normal application database:

```bash
docker compose --env-file /dev/null -p fastapi-qa -f compose.test.yaml run --build --rm tests
uv run --locked ruff check .
uv run --locked ruff format --check .
```

The test container runs `pytest --cov=app --cov-report=term-missing` after healthy dependencies and successful migration. Use `compose run`, **not** `--abort-on-container-exit`: migration is a successful one-shot service. Test configuration uses literal test-only credentials and does not inherit normal secrets.
Coverage is reported by pytest-cov; no minimum percentage threshold is configured. A passing suite does not replace live HTTP smoke or establish the concurrency target.

`tests/conftest.py` creates function-scoped, lifespan-managed apps and HTTPX ASGI clients. Cleanup truncates users/items and flushes Redis only after guards require database **`app_test`** and Redis database **1**. Never weaken those guards, run destructive fixtures against normal data, or parallelize this suite. Exercise actual PostgreSQL/Redis, not SQLite replacements. Existing regressions cover auth/privacy and hashing cancellation, persisted CRUD/ownership and tied-cursor pagination, cache deadlines/races/fallback, bounded health/pool failures, configuration, seeder destination safety, login and authenticator throttles, step-up races, and the edge TLS initializer. Use controlled events at real dependency boundaries for race tests; avoid sleep-only assertions and mocks that merely echo the implementation. Live dependency stop/restart checks belong in the isolated operational workflow described in README.
The cleanup guard must check SQLAlchemy's resolved driver arguments, including query overrides, before app creation/lifespan—not just `URL.database`. Require an unambiguous effective `app_test`; keep guard regressions connection-free and Redis restricted to database 1.
