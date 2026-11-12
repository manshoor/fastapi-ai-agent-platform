# Isolated 1,000-user benchmark — 31 October 2026

**PASS. Only the authorised 1,000-user tier was run. No further load was started.**

## Setup and isolation

- Command: `uv run --locked python scripts/benchmark_smoke.py --scenario 1000`.
- Standalone Compose file: `compose.bench.yaml`, never merged with normal Compose.
- Fresh project: `fastapi-bench-smoke-1000-20261030t213319z-e5f616`.
- Dedicated project network, PostgreSQL volume, Redis instance, and random credentials.
- API published only at `127.0.0.1:8001`; PostgreSQL and Redis had no host ports.
- Resolved Compose configuration, running database volume, and effective driver
  destinations were verified before seeding. No normal-stack database URL was used.
- Seeded 5,000 accounts and 125,000 items (25 per account), with 1,000 distinct
  active bearer-token fixtures. A benchmark-only SELECT confirmed both row counts.
- No queries, seed operations, truncation, or Redis flush targeted the normal stack.
  Normal container IDs were unchanged before/after; API/database/Redis remained healthy.

## Environment and workload

Docker reported 5 CPUs and 6,210,437,120 bytes of VM memory, approximately 5.78 GiB.
The load generator ran on the same Mac; the normal stack remained running and
shared hardware. This is a local benchmark, not a separate-generator production
capacity test. Host swap activity and disk latency were not measured.

The application retained one Uvicorn worker, a database pool of 10 with no overflow,
the existing cache/timeouts, PostgreSQL 19 RC1, and Redis 7.4. No application tuning
or normal-stack configuration changes were made for this run.

The existing Locust harness used 2–5 second think time and 70/20/10 detail GET /
list GET / PATCH task weights, with one unique account per virtual user. This is
a paced, closed-loop workload, not 1,000 simultaneous in-flight HTTP requests.
Authentication/register/create/delete throughput is outside this workload.

- Total duration: 480 seconds.
- Spawn rate: 50 users/second; target: 1,000 users.
- Warmup: 90 seconds; minimum required measured duration: 300 seconds.
- Measurement began at 01:35:12 and ended at 01:41:42 Asia/Dubai.
- Gates: maintain the target; measured failure ratio <=1%; aggregate and per-endpoint
  p95 <=500 ms / p99 <=1,000 ms; no correctness failures or invalid generator conditions.

## Final results

The harness's final measurement summary was:

```text
CAPACITY PASS target=1000 minimum_active=1000 steady=389.92s requests=111300
failures=0 ratio=0.00000 whole_run_failures=0 p95=5ms p99=8ms
generator_cpu_warning=False reasons=[]
```

Locust exited with code 0. Approximately **285 requests/sec** were sustained for
**6.5 measured minutes**, with the full target maintained and zero whole-run
request/response-validation failures reported.

Final console statistics, including shutdown completions:

| Endpoint | Requests | Failures | p95 | p99 |
| --- | ---: | ---: | ---: | ---: |
| Detail GET | 78,011 | 0 | 5 ms | 8 ms |
| List GET (`limit=20`) | 22,059 | 0 | 5 ms | 8 ms |
| PATCH | 11,231 | 0 | 6 ms | 9 ms |
| Aggregate | 111,301 | 0 | 5 ms | 8 ms |

All per-endpoint gates passed. The console has one more completion than the
harness's bounded measurement counter. The periodic CSV snapshot has 111,258
requests; it precedes the final console totals. These counter boundaries are
recorded explicitly; the differences are not failed requests. The harness's
111,300-request summary is used for the measured-window verdict.

## Resource observations

Seventy container samples were captured approximately seven seconds apart.
These are sampled peaks, not continuously observed maxima. Docker CPU percentages
use approximately 100% per fully utilised core.

| Container | Peak sampled CPU | Peak sampled memory |
| --- | ---: | ---: |
| API | 40.51% | 89.99 MiB |
| PostgreSQL | 7.01% | 90.12 MiB |
| Redis | 2.66% | 14.59 MiB |

Thirty benchmark-only dependency samples were collected, approximately 15 seconds
apart. PostgreSQL samples showed at most 11 connections: the pool plus the observer
connection. Observed wait events were `ClientRead` or no wait event; no lock/I/O
wait event appeared in those snapshots. This does not measure pool-acquisition
latency or exclude waits between samples.

Redis reported zero evictions and rejected connections throughout sampling.
The sampled Redis hit/miss counters rose from 1,997/3,108 to 39,018/53,662.
No generator CPU warning or unhandled user/runner exception was reported.
All benchmark containers stopped with exit code 0, zero restarts, and no OOM kills.

No bottleneck was demonstrated at this paced workload. The result does not locate
maximum capacity, establish production availability, or predict 15k/30k-user results.

## Artifacts and final state

Private artifacts remain under:

`artifacts/fastapi-bench-smoke-1000-20261030t213319z-e5f616/`

They include `run.json`, `source-metadata.json` (source hashes and resource limits),
setup/seed logs, row counts, container metadata, resource/dependency samples,
`resource-summary.json`, and Locust logs/CSVs. Credentials and signed fixtures are
private: do not share `.env`, `users.csv`, or the directory wholesale.

The runner stopped only this benchmark's containers. The dedicated database volume
and artifacts are preserved, and the normal stack remains running unchanged.
No 2,500/5,000-user tier, repeat run, or additional load was started. Review these
results before authorising any further benchmark.

Verification of the runner update: 23 connection-free safety/scenario tests passed;
Ruff lint and formatting passed. The default command remains the 20-user smoke;
only explicit `--scenario 1000` selects this tier, and unsupported larger tiers
are rejected.
