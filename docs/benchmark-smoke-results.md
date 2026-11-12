# Isolated 20-user benchmark smoke — 31 October 2026

**PASS. No higher-load tier was run.** This is a local workflow and small-load
verification, not a production-capacity measurement.

## Scope and isolation

- Command: `uv run --locked python scripts/benchmark_smoke.py`.
- Standalone configuration: `compose.bench.yaml`; normal Compose was not merged.
- Project: `fastapi-bench-smoke-20261030t212829z-d2b45d` (run ID uses UTC).
- Dedicated project network and PostgreSQL volume:
  `fastapi-bench-smoke-20261030t212829z-d2b45d_benchmark_postgres_data`.
- Fresh random credentials; database `app` and Redis database 0 belong to this
  separate topology. Database names alone are not the isolation mechanism.
- API: `127.0.0.1:8001`; no PostgreSQL/Redis host ports.
- Resolved Compose destinations, running database volume ownership, and effective
  PostgreSQL/Redis driver destinations passed checks before fixture seeding.
- Seeded only 20 users and 500 items (25 per user), using the append-only seeder.
- No statements, seeding, truncation, or Redis flush targeted the normal database
  or normal Redis. Normal containers retained their original IDs and healthy
  running status. No normal containers were restarted/recreated.

## Environment and workload

Docker reported 5 CPUs and 6,210,437,120 bytes of VM memory (about 5.78 GiB usable).
The screenshot's configured allocation was 6 GiB and 1 GiB swap; swap activity was
not measured. The generator ran on the same Mac as Docker, and the normal stack
remained running. These shared resources limit interpretation of the result.

The existing one-worker runtime, 10-connection database pool, cache settings,
PostgreSQL 19 RC1, and Redis 7.4 were retained. No application code was tuned.
Locust used unique prepared-token accounts, 2–5 second think time, and the existing
70/20/10 detail/list/PATCH task weights. Authentication throughput is excluded.

The run started at approximately 01:28:53 and finished at 01:30:23 Asia/Dubai.
Target: 20 users, spawn rate 5 users/sec, total duration 90 seconds, warmup
30 seconds, required measured duration at least 45 seconds.

## Observed results

The final harness summary reported:

```text
CAPACITY PASS target=20 minimum_active=20 steady=59.86s requests=321
failures=0 ratio=0.00000 whole_run_failures=0 p95=12ms p99=14ms
generator_cpu_warning=False reasons=[]
```

Locust exited with code 0. Approximate measured throughput was 5.4 requests/sec.
The final console endpoint statistics were:

| Endpoint | Requests | Failures | p95 | p99 |
| --- | ---: | ---: | ---: | ---: |
| Detail GET | 236 | 0 | 12 ms | 14 ms |
| List GET (`limit=20`) | 63 | 0 | 13 ms | 13 ms |
| PATCH | 22 | 0 | 14 ms | 15 ms |
| Aggregate | 321 | 0 | 12 ms | 14 ms |

All endpoints met the proposed p95 <=500 ms / p99 <=1,000 ms gates. There were
zero whole-run request/validation failures and no unhandled exceptions reported.
The active-user count was held at 20 during measurement.

The periodic `smoke_stats.csv` snapshot contains 318 requests; the final console
and harness counters contain 321. Use the final log for this run's totals: the
periodic CSV writer did not capture the final three requests before shutdown.
This difference is recorded, not treated as three failed requests.

Fourteen resource samples were collected, approximately seven seconds apart
(five-second pauses plus sampling overhead). Sampled peaks are not continuous
maximums. Docker CPU percentages use approximately 100% per fully used core.

| Container | Peak sampled CPU | Peak sampled memory |
| --- | ---: | ---: |
| API | 17.13% | 81.28 MiB |
| PostgreSQL | 4.58% | 47.20 MiB |
| Redis | 2.11% | 10.11 MiB |

No generator CPU warning was reported. No OOM kills or container restarts were
observed; API, database, Redis, and migration containers all exited with code 0.
This smoke did not collect database wait-event samples, Redis hit-rate deltas,
host swap telemetry, or sustained-load evidence.

## Artifacts and final state

Private artifacts are under:

`artifacts/fastapi-bench-smoke-20261030t212829z-d2b45d/`

The directory contains setup/seed logs, run metadata, container metadata,
resource samples, Locust logs and CSVs, plus private credentials and bearer-token
fixtures. Do not publish the directory wholesale or disclose `.env`/`users.csv`.

Only the benchmark containers were stopped after the run. Its volume and
artifacts remain available for review; no volumes were deleted. The normal stack
was left running. Further load requires review and explicit authorisation.
