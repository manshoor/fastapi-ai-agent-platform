# Isolated 2,500-user benchmark — 31 October 2026

**PASS. Only the authorised 2,500-user tier was run. No repeat or higher tier was started.**

## Setup and isolation

- Command: `uv run --locked python scripts/benchmark_smoke.py --scenario 2500`.
- Standalone file: `compose.bench.yaml`, never merged with normal Compose.
- Fresh project: `fastapi-bench-smoke-2500-20261030t214526z-88839e`.
- Dedicated project network, PostgreSQL volume, Redis instance, and random credentials.
- API published only at `127.0.0.1:8001`; PostgreSQL/Redis had no host ports.
- Resolved configuration, running database volume ownership, and effective driver
  destinations passed checks before seeding.
- Seeded 5,000 accounts and 125,000 items, with 2,500 distinct active bearer-token
  fixtures. A benchmark-only SELECT confirmed the database row counts.
- No queries, seeding, truncation, or Redis flush targeted the normal stack.
  Its original container IDs were unchanged; API/database/Redis remained healthy.

## Environment and workload

Docker reported 5 CPUs and 6,210,437,120 bytes of VM memory (about 5.78 GiB).
The generator ran on the same Mac, and the normal stack remained running on shared
hardware. Host swap activity, disk latency, and generator network limits were not
measured. No separate-generator or production-capacity claim follows.

One Uvicorn worker, a 10-connection database pool with no overflow, existing cache
and timeout settings, PostgreSQL 19 RC1, and Redis 7.4 were retained. Source hashes
confirmed the application, Dockerfile, standalone Compose file, dependencies,
seeder, and Locust harness were unchanged from the 1,000-user run. Only the runner's
explicit scenario selection was extended; no application tuning was applied.

The corpus size matched the 1,000-user tier, but used fresh isolated data. Each VU
had a unique account; requests followed the existing 70/20/10 detail GET / first-page
list GET / PATCH weights, with 2–5 seconds of think time. This is a paced closed-loop
workload, not 2,500 simultaneous in-flight requests. Authentication throughput is
excluded because signed tokens are prepared before traffic.

- Target: 2,500 users; spawn rate: 100 users/second.
- Duration: 780 seconds (13 minutes); warmup: 120 seconds.
- Minimum measured duration: 600 seconds at the full target.
- Measurement began at 01:47:50 and ended at 01:58:50 Asia/Dubai.
- Gates: measured failure ratio <=1%; p95 <=500 ms; p99 <=1,000 ms, checked
  overall and per endpoint; no correctness failures or invalid generator conditions.

## Final results

The harness's final bounded-window summary was:

```text
CAPACITY PASS target=2500 minimum_active=2500 steady=659.89s requests=470507
failures=0 ratio=0.00000 whole_run_failures=0 p95=20ms p99=31ms
generator_cpu_warning=False reasons=[]
```

Locust exited with code 0. Throughput was approximately **713 requests/sec** for
**11 measured minutes**, with the full user target maintained. There were zero
whole-run request/response-validation failures and no unhandled exceptions reported.

Final console statistics, including shutdown completions:

| Endpoint | Requests | Failures | p95 | p99 |
| --- | ---: | ---: | ---: | ---: |
| Detail GET | 329,079 | 0 | 20 ms | 31 ms |
| List GET (`limit=20`) | 94,242 | 0 | 20 ms | 31 ms |
| PATCH | 47,191 | 0 | 21 ms | 32 ms |
| Aggregate | 470,512 | 0 | 20 ms | 31 ms |

All endpoint gates passed. The console includes five more completions than the
harness's bounded measurement window; the periodic CSV snapshot contains 470,144
requests. These are different counter/snapshot boundaries, not failed requests.
The measured-window verdict uses the harness's 470,507-request summary.

| Single-run local result | Measured requests | Approx. RPS | p95 / p99 |
| --- | ---: | ---: | ---: |
| 1,000 paced users | 111,300 | 285 | 5 / 8 ms |
| 2,500 paced users | 470,507 | 713 | 20 / 31 ms |

Latency increased with this higher workload but remained below the proposed gates.
These individual runs do not establish repeatability, maximum capacity, a soak
result, outage tolerance, or production availability.

## Resource and dependency observations

There were 112 container samples, approximately seven seconds apart. These are
sampled peaks rather than continuously observed maxima. Docker CPU percentages
use approximately 100% per fully utilised core.

| Container | Peak sampled CPU | Peak sampled memory |
| --- | ---: | ---: |
| API | 76.56% | 96.34 MiB |
| PostgreSQL | 12.61% | 91.23 MiB |
| Redis | 3.90% | 10.65 MiB |

Fifty-one dependency samples were collected approximately 15 seconds apart.
PostgreSQL showed at most 11 connections, including the observer. Observed wait
events were `ClientRead`, `WalSync`, or no wait event. Sparse snapshots cannot
exclude waits between samples or establish pool-acquisition latency.

Redis reported zero evictions and rejected connections throughout sampling.
The sampled hit/miss counters rose from 1,296/2,628 to 159,974/219,511. No generator
CPU warning was reported. API, database, Redis, and migration containers all
stopped with exit code 0, zero restarts, and no OOM kills.

## Artifacts, verification, and final state

Private artifacts remain under:

`artifacts/fastapi-bench-smoke-2500-20261030t214526z-88839e/`

They include run metadata, source hashes, setup/seed logs, row counts, container
metadata, resource/dependency samples, a resource summary, and Locust logs/CSVs.
Do not share `.env`, signed `users.csv` fixtures, or the directory wholesale.

The runner stopped only this benchmark's containers, preserving its dedicated
volume and artifacts. Normal services remain running unchanged. No 5,000-user
tier, repeat run, or other load was started. Further load requires review and
explicit authorisation.

The new scenario regression failed before implementation, then all 24
connection-free safety/scenario tests passed. Ruff lint and formatting passed.
The default remains the 20-user smoke; 1,000 and 2,500 users require explicit
scenario selection, and arbitrary counts or unsupported higher tiers are rejected.
