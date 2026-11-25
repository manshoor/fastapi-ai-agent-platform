"""Prove the GitOps demo in the cluster made by scripts/gitops_cluster.py.

python scripts/smoke_gitops.py --revision <40-hex commit>

1. Argo CD is Synced and Healthy at that commit
2. scripts/smoke.py passes through a loopback port-forward
3. smoke_ai.py, smoke_rag.py (with OCR) and smoke_assistant.py pass the same way
4. only the planned network paths are open: no other pod to pod, none to the internet
5. scaling `api` to 0 by hand is undone by Argo CD
6. a failing migration stops the sync and leaves the API pods alone
"""

import argparse
import importlib.util
import json
import re
import select
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path


def _load(name: str):
    # A sibling script, not a package: load it by path.
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cluster = _load("gitops_cluster")
checker = _load("check_deploy")

APPLICATION = "fastapi-demo"
API = "deployment/api"
REPLICAS = 2
# Marks the sync operation that carries the failing patch.
MARKER = "gitops-smoke-failing-migration"
FAILING_PATCHES = [
    {
        "target": {"kind": "Job", "name": "migrate"},
        "patch": json.dumps(
            [
                {
                    "op": "replace",
                    "path": "/spec/template/spec/containers/0/command",
                    "value": ["python", "-c", "raise SystemExit(1)"],
                }
            ]
        ),
    },
    {
        # Makes the app OutOfSync, so Argo CD starts a sync.
        "target": {"kind": "Deployment", "name": "api"},
        "patch": json.dumps(
            [
                {
                    "op": "add",
                    "path": "/spec/template/metadata/annotations",
                    "value": {MARKER: "true"},
                }
            ]
        ),
    },
]
FAILED_PHASES = {"Failed", "Error"}
FORWARDING = re.compile(r"Forwarding from 127\.0\.0\.1:(\d+) ")
POLL_SECONDS = 3
# Their sum must stay under the step limit in gitops.yml, or no log would be shown.
SYNC_SECONDS = 240
HEAL_SECONDS = 90
FAIL_SECONDS = 160
RECOVER_SECONDS = 120
# How long Argo CD gets to start a sync by itself before one is requested.
START_SECONDS = 20
FORWARD_SECONDS = 20
CORE_SMOKES = {"smoke.py": 45}
AI_SMOKES = {"smoke_ai.py": 60, "smoke_rag.py": 90, "smoke_assistant.py": 90}
NETWORK_SECONDS = 30
# Argo CD's repo server fetches from it, so the runner's pods can reach it.
INTERNET = "github.com:443"
OPEN = "open"
BLOCKED = "timeout"
PYTHON_PODS = ("api", "dispatcher", "worker", "rag-worker", "provider")
# Images without Python: they only try the internet and, as a control, their own port.
BASH_PODS = {
    "statefulset/db": "127.0.0.1:5432",
    "deployment/redis": "127.0.0.1:6379",
    "statefulset/kafka": "127.0.0.1:9092",
}
# Both probes tell a dropped connection (timeout) from a missing name or a refusal.
PYTHON_PROBE = """
import json, socket, sys
from concurrent.futures import ThreadPoolExecutor

def probe(target):
    host, port = target.rsplit(":", 1)
    try:
        address = socket.getaddrinfo(host, int(port), socket.AF_INET, socket.SOCK_STREAM)[0][4]
    except OSError:
        return "no address"
    try:
        socket.create_connection(address, timeout=4).close()
    except TimeoutError:
        return "timeout"
    except OSError as error:
        return type(error).__name__
    return "open"

targets = sys.argv[1:]
print(json.dumps(dict(zip(targets, ThreadPoolExecutor().map(probe, targets)))))
"""
BASH_PROBE = """
for target in "$@"; do
  host="${target%:*}"
  if ! getent hosts "$host" > /dev/null; then echo "$target no address"; continue; fi
  # 124 and 143: timeout's own exit codes (coreutils, busybox).
  timeout 4 bash -c 'exec 3<> "/dev/tcp/$0/$1"' "$host" "${target##*:}" 2> /dev/null
  case $? in
    0) echo "$target open" ;;
    124 | 143) echo "$target timeout" ;;
    *) echo "$target error" ;;
  esac
done
"""
# smoke_rag.py renders its scanned page with it, and skips OCR without it.
OCR_TOOL = "pdftoppm"


class SmokeError(RuntimeError):
    pass


def _expect(ok: object, message: str) -> None:
    if not ok:
        raise SmokeError(message)


def _wait(what: str, check: Callable[[], bool], seconds: int) -> None:
    deadline = time.monotonic() + seconds
    while not check():
        if time.monotonic() >= deadline:
            raise SmokeError(f"timed out after {seconds} s: {what}")
        time.sleep(POLL_SECONDS)


def _get(*arguments: str, namespace: str = cluster.NAMESPACE) -> dict:
    output = cluster.kubectl("--namespace", namespace, "get", *arguments, "--output", "json")
    return json.loads(output)


def _get_app() -> dict:
    return _get("application", APPLICATION, namespace=cluster.ARGOCD_NAMESPACE)


def _patch_app(patch: Mapping) -> None:
    cluster.kubectl(
        *("--namespace", cluster.ARGOCD_NAMESPACE, "patch", "application", APPLICATION),
        *("--type", "merge", "--patch", json.dumps(patch)),
    )


# Reading an Application


def _operation(app: Mapping) -> Mapping:
    return app.get("status", {}).get("operationState") or {}


def _sync_result(app: Mapping) -> Mapping:
    return _operation(app).get("syncResult") or {}


def has_marker(app: Mapping) -> bool:
    """Did the last sync operation carry the failing patch?"""
    return MARKER in json.dumps(_sync_result(app).get("source") or {})


def is_synced(app: Mapping, revision: str) -> bool:
    status = app.get("status", {})
    sync = status.get("sync") or {}
    return (
        sync.get("status") == "Synced"
        and sync.get("revision") == revision
        and (status.get("health") or {}).get("status") == "Healthy"
        and _operation(app).get("phase") == "Succeeded"
        and _sync_result(app).get("revision") == revision
        and not has_marker(app)
    )


def _hook_failed(app: Mapping) -> bool:
    return any(
        (item.get("kind"), item.get("name"), item.get("hookPhase")) == ("Job", "migrate", "Failed")
        for item in _sync_result(app).get("resources") or []
    )


def failed_on_migrate(app: Mapping) -> bool:
    """The patched sync used up its retries, because of the migrate Job."""
    operation = _operation(app)
    limit = app["spec"]["syncPolicy"]["retry"]["limit"]
    return (
        has_marker(app)
        and operation.get("phase") in FAILED_PHASES
        and operation.get("retryCount", 0) == limit
        and _hook_failed(app)
    )


# Reading the API workload


def ready_pods(pods: Mapping) -> dict[str, bool]:
    """Pod UID -> ready, for the pods that are not being deleted."""
    return {
        pod["metadata"]["uid"]: any(
            condition["type"] == "Ready" and condition["status"] == "True"
            for condition in pod.get("status", {}).get("conditions") or []
        )
        for pod in pods["items"]
        if "deletionTimestamp" not in pod["metadata"]
    }


def _api_pods() -> dict[str, bool]:
    return ready_pods(_get("pods", "--selector", "app.kubernetes.io/name=api"))


def _all_ready(pods: Mapping[str, bool]) -> bool:
    return len(pods) == REPLICAS and all(pods.values())


# The checks


@contextmanager
def _port_forward() -> Iterator[str]:
    command = cluster.kubectl_command("--namespace", cluster.NAMESPACE, "port-forward")
    command += ["--address", "127.0.0.1", "service/api", ":8000"]
    with subprocess.Popen(command, stdout=subprocess.PIPE, text=True) as process:
        try:
            readable, _, _ = select.select([process.stdout], [], [], FORWARD_SECONDS)
            match = FORWARDING.match(process.stdout.readline()) if readable else None
            _expect(match, "port-forward did not start")
            yield f"http://127.0.0.1:{match[1]}"
        finally:
            process.terminate()


def _check_synced(revision: str, seconds: int = SYNC_SECONDS) -> None:
    _wait("Synced and Healthy", lambda: is_synced(_get_app(), revision), seconds)


def _run_smokes(smokes: Mapping[str, int]) -> None:
    """Run sibling scripts one after another through one port-forward."""
    with _port_forward() as url:
        for script, seconds in smokes.items():
            path = Path(__file__).resolve().parent / script
            command = [sys.executable, str(path), "--base-url", url]
            try:
                result = subprocess.run(command, check=False, timeout=seconds)
            except subprocess.TimeoutExpired:
                raise SmokeError(f"{script} timed out") from None
            _expect(result.returncode == 0, f"{script} failed")


def _check_ai() -> None:
    _expect(shutil.which(OCR_TOOL), f"{OCR_TOOL} is missing: OCR would not run")
    _run_smokes(AI_SMOKES)


def expected_paths(sender: str) -> dict[str, str]:
    """Target -> what one pod's probe must find, from the checked table."""
    expected = {INTERNET: BLOCKED}
    if sender != "api":
        expected["api:8000"] = BLOCKED
    for receiver, (port, senders) in checker.PATHS.items():
        if receiver != sender:
            expected[f"{receiver}:{port}"] = OPEN if sender in senders else BLOCKED
    return expected


def parse_bash_probe(output: str) -> dict[str, str]:
    return dict(line.split(" ", 1) for line in output.splitlines())


def _exec(target: str, *command: str, namespace: str = cluster.NAMESPACE) -> str:
    return cluster.kubectl(
        *("--namespace", namespace, "exec", target, "--", *command), timeout=NETWORK_SECONDS
    )


def _probe_python(name: str) -> None:
    expected = expected_paths(name)
    output = _exec(f"deployment/{name}", "python", "-c", PYTHON_PROBE, *expected)
    _expect(json.loads(output) == expected, f"{name} reached {output}, expected {expected}")


def _probe_bash(target: str, expected: Mapping[str, str], namespace: str) -> None:
    command = ("bash", "-c", BASH_PROBE, "bash", *expected)
    output = _exec(target, *command, namespace=namespace)
    found = parse_bash_probe(output)
    _expect(found == expected, f"{target} reached {found}, expected {expected}")


def _check_network() -> None:
    # The control: without policies the same probe reaches the internet.
    control = "deployment/argocd-repo-server"
    _probe_bash(control, {INTERNET: OPEN}, cluster.ARGOCD_NAMESPACE)
    with ThreadPoolExecutor() as pool:
        probes = [pool.submit(_probe_python, name) for name in PYTHON_PODS]
        for target, own in BASH_PODS.items():
            expected = {own: OPEN, INTERNET: BLOCKED}
            probes.append(pool.submit(_probe_bash, target, expected, cluster.NAMESPACE))
    for probe in probes:
        probe.result()


def _check_self_repair(revision: str) -> None:
    before = _get(API)["metadata"]["generation"]
    old_pods = set(_api_pods())
    cluster.kubectl("--namespace", cluster.NAMESPACE, "scale", API, "--replicas=0")

    # Two changes: ours, then Argo CD's. The new pods prove ours took effect.
    def repaired() -> bool:
        live = _get(API)
        pods = _api_pods()
        return (
            live["metadata"]["generation"] >= before + 2
            and live["spec"]["replicas"] == REPLICAS
            and _all_ready(pods)
            and not old_pods & set(pods)
            and is_synced(_get_app(), revision)
        )

    _wait("Argo CD undoes the scale to 0", repaired, HEAL_SECONDS)


def _request_sync(revision: str) -> None:
    options = _get_app()["spec"]["syncPolicy"]["syncOptions"]
    _patch_app(
        {
            "operation": {
                "initiatedBy": {"username": "smoke_gitops"},
                "sync": {"revision": revision, "prune": True, "syncOptions": options},
            }
        }
    )


def _started_clean(failed_at: str) -> bool:
    app = _get_app()
    started = _operation(app).get("startedAt")
    return bool(app.get("operation")) or (started != failed_at and not has_marker(app))


def _check_failing_migration(revision: str) -> None:
    before = _api_pods()
    generation = _get(API)["metadata"]["generation"]
    _expect(_all_ready(before), "API pods are not ready before the failing migration")
    _patch_app({"spec": {"source": {"kustomize": {"patches": FAILING_PATCHES}}}})
    _wait("the sync fails on migrate", lambda: failed_on_migrate(_get_app()), FAIL_SECONDS)
    _expect(_api_pods() == before, "the failing migration touched the API pods")
    _expect(
        _get(API)["metadata"]["generation"] == generation,
        "the failing migration changed the API Deployment",
    )

    # Without the patch the live objects already match git, so Argo CD may start nothing.
    failed_at = _operation(_get_app())["startedAt"]
    _patch_app({"spec": {"source": {"kustomize": None}}})
    try:
        _wait("Argo CD starts a sync", lambda: _started_clean(failed_at), START_SECONDS)
    except SmokeError:
        _request_sync(revision)
    _check_synced(revision, RECOVER_SECONDS)
    _expect(_api_pods() == before, "the recovery replaced the API pods")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--revision", required=True, help="The commit Argo CD must be at")
    args = parser.parse_args()
    if not cluster.COMMIT.fullmatch(args.revision):
        parser.error("--revision must be 40 lowercase hex characters")
    checks: list[tuple[str, Callable[[], None]]] = [
        ("Synced and Healthy at the commit", lambda: _check_synced(args.revision)),
        ("smoke.py through a port-forward", lambda: _run_smokes(CORE_SMOKES)),
        ("the AI smokes, with OCR, through a port-forward", _check_ai),
        ("only the planned network paths are open", _check_network),
        ("Argo CD undoes a manual scale to 0", lambda: _check_self_repair(args.revision)),
        ("a failing migration is contained", lambda: _check_failing_migration(args.revision)),
    ]
    try:
        cluster.require_loopback()
        for name, check in checks:
            check()
            print(f"PASS: {name}", flush=True)
    except (SmokeError, cluster.ClusterError) as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
