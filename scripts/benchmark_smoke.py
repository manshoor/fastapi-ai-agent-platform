"""Create a fresh isolated stack and run one explicitly selected, fixed Locust scenario.

Retain private results, credentials, and the database volume for review. Stop only
this benchmark's containers afterwards, including on failure. No normal-stack
commands, database cleanup, or automatic load escalation are run.
"""

import argparse
import json
import os
import secrets
import socket
import stat
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = {
    "smoke": {
        "users": 20,
        "seed_users": 20,
        "spawn_rate": 5,
        "duration_seconds": 90,
        "warmup_seconds": 30,
        "minimum_steady_seconds": 45,
        "maximum_failure_ratio": "0",
        "zero_failures": "true",
    },
    "1000": {
        "users": 1000,
        "seed_users": 5000,
        "spawn_rate": 50,
        "duration_seconds": 480,
        "warmup_seconds": 90,
        "minimum_steady_seconds": 300,
        "maximum_failure_ratio": "0.01",
        "zero_failures": "false",
    },
    "2500": {
        "users": 2500,
        "seed_users": 5000,
        "spawn_rate": 100,
        "duration_seconds": 780,
        "warmup_seconds": 120,
        "minimum_steady_seconds": 600,
        "maximum_failure_ratio": "0.01",
        "zero_failures": "false",
    },
}


def parse_scenario() -> tuple[str, dict]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, default="smoke")
    args = parser.parse_args()
    return args.scenario, SCENARIOS[args.scenario]


def validate_topology(config: dict, project: str, artifacts: Path) -> None:
    """Fail closed before building/starting anything if destinations escape isolation."""
    if config.get("name") != project or not project.startswith("fastapi-bench-smoke-"):
        raise ValueError("Unexpected benchmark project")
    services = config["services"]
    if set(services) != {"db", "redis", "migrate", "api", "tools"}:
        raise ValueError("Unexpected benchmark services")
    volume = config["volumes"]["benchmark_postgres_data"]
    network = config["networks"]["default"]
    if volume.get("external") or volume["name"] != f"{project}_benchmark_postgres_data":
        raise ValueError("Benchmark database volume is not project-owned")
    if network.get("external") or network["name"] != f"{project}_default":
        raise ValueError("Benchmark network is not project-owned")
    for name, service in services.items():
        if set(service.get("networks", {})) != {"default"} or service.get("network_mode"):
            raise ValueError("Unexpected service network")
        if name != "api" and service.get("ports"):
            raise ValueError("Only the benchmark API may publish a port")
    ports = services["api"]["ports"]
    if len(ports) != 1 or (
        ports[0].get("host_ip"),
        str(ports[0].get("published")),
        ports[0].get("target"),
    ) != ("127.0.0.1", "8001", 8000):
        raise ValueError("Unexpected benchmark API binding")
    mounts = services["db"]["volumes"]
    if len(mounts) != 1 or (mounts[0]["type"], mounts[0]["source"], mounts[0]["target"]) != (
        "volume",
        "benchmark_postgres_data",
        "/var/lib/postgresql",
    ):
        raise ValueError("Unexpected database mount")
    mounts = services["tools"]["volumes"]
    if len(mounts) != 1 or (mounts[0]["type"], mounts[0]["source"], mounts[0]["target"]) != (
        "bind",
        str(artifacts),
        "/artifacts",
    ):
        raise ValueError("Unexpected artifact mount")
    for name in ("api", "migrate", "tools"):
        env = services[name]["environment"]
        expected = {
            "DATABASE_URL": "",
            "DB_HOST": "db",
            "DB_PORT": "5432",
            "DB_NAME": "app",
            "DB_USER": "app",
            "REDIS_URL": "redis://redis:6379/0",
            "AUTH_TOKEN_TTL_SECONDS": "7200",
        }
        if any(str(env.get(key)) != value for key, value in expected.items()):
            raise ValueError("Unexpected benchmark dependency destination")
        if env["DB_PASSWORD"] != services["db"]["environment"]["POSTGRES_PASSWORD"]:
            raise ValueError("Benchmark database credentials disagree")


def main() -> int:
    scenario_name, scenario = parse_scenario()
    os.chdir(ROOT)
    os.umask(0o077)
    # Fail before Docker mutations when the dedicated HTTP port is unavailable.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 8001))
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(mode=0o700, exist_ok=True)
    metadata = artifacts.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ValueError("artifacts must be a current-user-owned nonsymlink directory, mode 0700")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)
    project = f"fastapi-bench-smoke-{scenario_name}-{run_id.lower()}"
    output = artifacts / project
    output.mkdir(mode=0o700)
    env_file = output / ".env"
    with env_file.open("x") as handle:
        handle.write(
            f"BENCH_POSTGRES_PASSWORD={secrets.token_hex(24)}\n"
            f"BENCH_JWT_SECRET={secrets.token_hex(32)}\n"
            f"BENCH_UID={os.getuid()}\nBENCH_GID={os.getgid()}\n"
            f"BENCH_ARTIFACTS='{output}'\n"
        )
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith(("BENCH_", "COMPOSE_", "LOCUST_", "LOAD_"))
    }
    compose = [
        "docker",
        "compose",
        "--env-file",
        str(env_file),
        "-p",
        project,
        "-f",
        str(ROOT / "compose.bench.yaml"),
        "--profile",
        "tools",
    ]

    def run(args: list[str], *, log: str | None = None) -> str:
        if log:
            with (output / log).open("a") as handle:
                result = subprocess.run(
                    args, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=False
                )
            if result.returncode:
                raise RuntimeError(f"Step failed; inspect private {output / log}")
            return ""
        return subprocess.check_output(args, env=environment, text=True, stderr=subprocess.PIPE)

    print(f"Benchmark project: {project}\nPrivate artifacts: {output}", flush=True)
    config = json.loads(run([*compose, "config", "--format", "json"]))
    validate_topology(config, project, output)
    # Never persist/print rendered Compose configuration: it contains credentials.
    (output / "run.json").write_text(
        json.dumps(
            {
                "project": project,
                "scenario": scenario_name,
                **scenario,
                "items_per_user": 25,
                "generator": "host, sharing hardware with Docker VM",
                "api": "http://127.0.0.1:8001",
            },
            indent=2,
        )
        + "\n"
    )
    started = False
    try:
        started = True
        print("Building and starting isolated dependencies/API...", flush=True)
        run([*compose, "up", "--build", "-d", "--wait", "api"], log="setup.log")
        run([*compose, "build", "tools"], log="setup.log")
        # Verify the running DB volume and resolved driver target before any seeding.
        container_id = run([*compose, "ps", "-q", "db"]).strip()
        db = json.loads(run(["docker", "inspect", container_id]))[0]
        if db["Config"]["Labels"].get("com.docker.compose.project") != project:
            raise ValueError("Running database belongs to another project")
        if any(
            m["Type"] != "volume" or m["Name"] != f"{project}_benchmark_postgres_data"
            for m in db["Mounts"]
        ):
            raise ValueError("Running database has unexpected storage")
        probe = (
            "from app.config import Settings; from redis.asyncio.connection import parse_url; "
            "s=Settings(); u=s.sqlalchemy_url; _,k=u.get_dialect()().create_connect_args(u); "
            "assert (k['host'],k['database'],k['user']) == ('db','app','app'); "
            "r=parse_url(s.redis_url); assert r['host']=='redis' and r.get('db',0)==0; "
            "print('Effective benchmark destinations verified')"
        )
        run([*compose, "exec", "-T", "api", "python", "-c", probe], log="setup.log")
        run([*compose, "ps", "-a", "--format", "json"], log="containers.jsonl")
        print(
            f"Isolation verified. Seeding {scenario['seed_users']} accounts / "
            f"{scenario['seed_users'] * 25} items; {scenario['users']} active accounts...",
            flush=True,
        )
        run(
            [
                *compose,
                "run",
                "--rm",
                "--no-deps",
                "tools",
                "python",
                "scripts/seed_load.py",
                "--users",
                str(scenario["seed_users"]),
                "--active-users",
                str(scenario["users"]),
                "--items-per-user",
                "25",
                "--output",
                "/artifacts/users.csv",
            ],
            log="seed.log",
        )
        environment.update(
            {
                "LOAD_USERS_FILE": str(output / "users.csv"),
                "LOAD_WARMUP_SECONDS": str(scenario["warmup_seconds"]),
                "LOAD_MIN_STEADY_SECONDS": str(scenario["minimum_steady_seconds"]),
                "LOAD_ZERO_FAILURES": scenario["zero_failures"],
                "LOAD_MAX_FAILURE_RATIO": scenario["maximum_failure_ratio"],
            }
        )
        print(
            f"Running ONLY {scenario['users']} users, {scenario['spawn_rate']} users/sec, "
            f"{scenario['duration_seconds']} seconds...",
            flush=True,
        )
        with (
            (output / "locust.log").open("w") as log,
            (output / "resources.jsonl").open("w") as stats,
        ):
            process = subprocess.Popen(
                [
                    "uv",
                    "run",
                    "--locked",
                    "locust",
                    "-f",
                    "tests/load/locustfile.py",
                    "--headless",
                    "--host",
                    "http://127.0.0.1:8001",
                    "-u",
                    str(scenario["users"]),
                    "-r",
                    str(scenario["spawn_rate"]),
                    "-t",
                    f"{scenario['duration_seconds']}s",
                    "--stop-timeout",
                    "10",
                    "--csv",
                    str(output / "smoke"),
                    "--csv-full-history",
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + scenario["duration_seconds"] + 60
                while process.poll() is None:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Locust exceeded benchmark deadline")
                    ids = run([*compose, "ps", "-q", "api", "db", "redis"]).split()
                    if len(ids) != 3:
                        raise RuntimeError("Benchmark dependency disappeared")
                    sample = run(["docker", "stats", "--no-stream", "--format", "{{json .}}", *ids])
                    stats.write(
                        json.dumps({"time": datetime.now(UTC).isoformat(), "stats": sample}) + "\n"
                    )
                    stats.flush()
                    time.sleep(5)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            locust_log = (output / "locust.log").read_text()
            if process.returncode or "CAPACITY PASS" not in locust_log:
                raise RuntimeError(f"Smoke failed; inspect private {output / 'locust.log'}")
        print(f"{scenario['users']}-user benchmark PASS. No higher tier started.", flush=True)
        return 0
    finally:
        if started:
            print(
                "Stopping only this benchmark's containers; preserving results and volume.",
                flush=True,
            )
            run([*compose, "stop"], log="setup.log")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Do not print subprocess output/configuration or raw URL/settings errors.
        print(
            f"Benchmark stopped: {type(error).__name__}. Inspect its private artifacts.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
