"""GitOps smoke: how it reads Argo CD and pod states. No cluster."""

import copy
import importlib.util
import json
import socket
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("smoke_gitops", ROOT / "scripts" / "smoke_gitops.py")
assert _SPEC is not None and _SPEC.loader is not None
smoke = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke)

COMMIT = "a" * 40
OTHER = "b" * 40


@pytest.fixture
def synced() -> dict:
    return {
        "spec": {"syncPolicy": {"retry": {"limit": 2}}},
        "status": {
            "sync": {"status": "Synced", "revision": COMMIT},
            "health": {"status": "Healthy"},
            "operationState": {
                "phase": "Succeeded",
                "syncResult": {"revision": COMMIT, "source": {"path": "deploy/app"}},
            },
        },
    }


@pytest.fixture
def failed(synced) -> dict:
    app = copy.deepcopy(synced)
    operation = app["status"]["operationState"]
    operation.update(phase="Failed", retryCount=2)
    operation["syncResult"]["source"]["kustomize"] = {"patches": smoke.FAILING_PATCHES}
    operation["syncResult"]["resources"] = [
        {"kind": "Job", "name": "migrate", "hookPhase": "Failed"}
    ]
    return app


def test_synced_and_healthy_at_the_commit(synced):
    assert smoke.is_synced(synced, COMMIT)


@pytest.mark.parametrize(
    ("keys", "value"),
    [
        (("sync", "status"), "OutOfSync"),
        (("sync", "revision"), OTHER),
        (("health", "status"), "Degraded"),
        (("health", "status"), "Progressing"),
        (("operationState", "phase"), "Running"),
        (("operationState", "phase"), "Failed"),
        (("operationState", "syncResult", "revision"), OTHER),
        (
            ("operationState", "syncResult", "source"),
            {"kustomize": {"patches": smoke.FAILING_PATCHES}},
        ),
    ],
)
def test_any_other_state_is_not_synced(synced, keys, value):
    node = synced["status"]
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value
    assert not smoke.is_synced(synced, COMMIT)


def test_a_new_application_is_not_synced():
    assert not smoke.is_synced({"spec": {}}, COMMIT)


def test_the_failed_migration_is_recognised(failed):
    assert smoke.failed_on_migrate(failed)


def test_a_failure_for_another_reason_does_not_count(failed):
    for change in (
        {"phase": "Running"},
        {"phase": "Succeeded"},
        {"retryCount": 1},
        {"syncResult": {**failed["status"]["operationState"]["syncResult"], "resources": []}},
        {"syncResult": {**failed["status"]["operationState"]["syncResult"], "source": {}}},
    ):
        app = copy.deepcopy(failed)
        app["status"]["operationState"].update(change)
        assert not smoke.failed_on_migrate(app), change


def test_ready_pods_skips_pods_being_deleted():
    def pod(uid: str, ready: str, **metadata) -> dict:
        return {
            "metadata": {"uid": uid, **metadata},
            "status": {"conditions": [{"type": "Ready", "status": ready}]},
        }

    pods = {
        "items": [
            pod("a", "True"),
            pod("b", "False"),
            pod("c", "True", deletionTimestamp="2026-01-01T00:00:00Z"),
            {"metadata": {"uid": "d"}},
        ]
    }
    assert smoke.ready_pods(pods) == {"a": True, "b": False, "d": False}


def test_the_failing_patch_fits_the_manifests():
    job, api = (
        yaml.safe_load_all((ROOT / "deploy" / "app" / f"{name}.yaml").read_text())
        for name in ("migrate", "api")
    )
    (job,) = job
    (api,) = [doc for doc in api if doc["kind"] == "Deployment"]
    migrate_patch, api_patch = smoke.FAILING_PATCHES
    assert migrate_patch["target"] == {"kind": job["kind"], "name": job["metadata"]["name"]}
    assert api_patch["target"] == {"kind": api["kind"], "name": api["metadata"]["name"]}
    # `replace` needs the command; `add` must not overwrite annotations.
    ((operation,), (added,)) = (json.loads(patch["patch"]) for patch in smoke.FAILING_PATCHES)
    assert operation["op"] == "replace"
    assert "command" in job["spec"]["template"]["spec"]["containers"][0]
    assert added["op"] == "add"
    assert "annotations" not in api["spec"]["template"]["metadata"]
    assert api["spec"]["replicas"] == smoke.REPLICAS


def test_port_forward_output_gives_the_port():
    match = smoke.FORWARDING.match("Forwarding from 127.0.0.1:40123 -> 8000\n")
    assert match[1] == "40123"
    assert not smoke.FORWARDING.match("Forwarding from 0.0.0.0:40123 -> 8000\n")


def test_the_ai_smokes_need_the_ocr_tool(monkeypatch):
    # smoke_rag.py would skip OCR silently.
    monkeypatch.setattr(smoke.shutil, "which", lambda name: None)
    monkeypatch.setattr(smoke, "_run_smokes", lambda _: pytest.fail("ran without OCR"))
    with pytest.raises(smoke.SmokeError, match="pdftoppm"):
        smoke._check_ai()


def test_smokes_share_one_port_forward_and_stop_at_a_failure(monkeypatch):
    commands = []

    @contextmanager
    def forward():
        yield "http://127.0.0.1:40123"

    def run(command, check, timeout):
        commands.append((Path(command[1]).name, command[2:], timeout))
        return subprocess.CompletedProcess(command, len(commands) == 2)

    monkeypatch.setattr(smoke, "_port_forward", forward)
    monkeypatch.setattr(smoke.subprocess, "run", run)
    with pytest.raises(smoke.SmokeError, match="smoke_rag.py failed"):
        smoke._run_smokes(smoke.AI_SMOKES)
    url = ["--base-url", "http://127.0.0.1:40123"]
    assert commands == [("smoke_ai.py", url, 60), ("smoke_rag.py", url, 90)]


def test_every_smoke_exists():
    for script in {**smoke.CORE_SMOKES, **smoke.AI_SMOKES}:
        assert (ROOT / "scripts" / script).is_file()


def test_the_probed_paths_are_the_planned_ones():
    assert smoke.expected_paths("api") == {
        "db:5432": "open",
        "redis:6379": "open",
        "kafka:9092": "timeout",
        "provider:8080": "open",
        "github.com:443": "timeout",
    }
    assert smoke.expected_paths("provider") == {
        "db:5432": "timeout",
        "redis:6379": "timeout",
        "kafka:9092": "timeout",
        "api:8000": "timeout",
        "github.com:443": "timeout",
    }


def test_every_running_workload_is_probed():
    probed = {*smoke.PYTHON_PODS, *(target.partition("/")[2] for target in smoke.BASH_PODS)}
    assert probed == set(smoke.checker.WAVES) - {"migrate", "topic-init"}


def test_the_python_probe_tells_open_refused_and_unknown_apart():
    with socket.create_server(("127.0.0.1", 0)) as server:
        opened = f"127.0.0.1:{server.getsockname()[1]}"
        with socket.create_server(("127.0.0.1", 0)) as other:
            closed = f"127.0.0.1:{other.getsockname()[1]}"
        targets = [opened, closed, "no-such-host.invalid:1"]
        command = [sys.executable, "-c", smoke.PYTHON_PROBE, *targets]
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
    assert list(json.loads(result.stdout).values()) == [
        "open",
        "ConnectionRefusedError",
        "no address",
    ]


@pytest.mark.parametrize("found", ["open", "no address", "ConnectionRefusedError"])
def test_only_a_timeout_counts_as_blocked(monkeypatch, found):
    answer = {**smoke.expected_paths("api"), "kafka:9092": found}
    monkeypatch.setattr(smoke, "_exec", lambda *_, **__: json.dumps(answer))
    with pytest.raises(smoke.SmokeError, match="api reached"):
        smoke._probe_python("api")


def test_the_network_check_fails_if_any_pod_does(monkeypatch):
    def run(target, *command, namespace=smoke.cluster.NAMESPACE):
        if command[0] == "python":
            name = target.partition("/")[2]
            return json.dumps(smoke.expected_paths(name))
        lines = []
        for probed in command[4:]:
            reachable = namespace == "argocd" or target == leaking or probed != smoke.INTERNET
            lines.append(f"{probed} {'open' if reachable else 'timeout'}\n")
        return "".join(lines)

    monkeypatch.setattr(smoke, "_exec", run)
    leaking = ""
    smoke._check_network()
    leaking = "statefulset/kafka"
    with pytest.raises(smoke.SmokeError, match="statefulset/kafka reached"):
        smoke._check_network()


def test_waiting_gives_up(monkeypatch):
    monkeypatch.setattr(smoke, "POLL_SECONDS", 0)
    with pytest.raises(smoke.SmokeError, match="timed out"):
        smoke._wait("never", lambda: False, 0)


@pytest.mark.parametrize("starts", ["by itself", "on request"])
def test_the_recovery_requests_a_sync_only_if_needed(monkeypatch, synced, failed, starts):
    """The failing check, with Argo CD played by one Application document."""
    state = {"app": copy.deepcopy(failed)}
    state["app"]["status"]["operationState"]["startedAt"] = "1"
    state["app"]["spec"]["syncPolicy"]["syncOptions"] = ["PruneLast=true"]
    patches = []

    def succeed() -> None:
        state["app"]["status"] = copy.deepcopy(synced["status"])
        state["app"]["status"]["operationState"]["startedAt"] = "2"

    def patch(value) -> None:
        patches.append(value)
        removed = value.get("spec", {}).get("source", {}).get("kustomize", "") is None
        if "operation" in value or (removed and starts == "by itself"):
            succeed()

    monkeypatch.setattr(smoke, "_get_app", lambda: state["app"])
    monkeypatch.setattr(smoke, "_patch_app", patch)
    monkeypatch.setattr(smoke, "_api_pods", lambda: {"a": True, "b": True})
    monkeypatch.setattr(smoke, "_get", lambda *_: {"metadata": {"generation": 1}})
    monkeypatch.setattr(smoke, "START_SECONDS", 0)
    smoke._check_failing_migration(COMMIT)

    requested = [value["operation"]["sync"] for value in patches if "operation" in value]
    assert len(requested) == (starts == "on request")
    for sync in requested:
        assert sync == {"revision": COMMIT, "prune": True, "syncOptions": ["PruneLast=true"]}


def test_the_waits_fit_their_step_and_the_job():
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "gitops.yml").read_text())
    job = workflow["jobs"]["cluster"]
    minutes = {step.get("name"): step.get("timeout-minutes") for step in job["steps"]}
    limits = [
        minutes[name]
        for name in (
            "Install poppler",
            "Start the cluster",
            "Smoke-test the cluster",
            "Show pods, events and logs",
            "Delete the cluster",
        )
    ]
    # A step limit fails the step; the job limit would cancel the job and skip the dump.
    assert all(limits)
    assert sum(limits) <= job["timeout-minutes"] - 3
    waits = (
        smoke.SYNC_SECONDS
        + 2 * smoke.FORWARD_SECONDS
        + sum(smoke.CORE_SMOKES.values())
        + sum(smoke.AI_SMOKES.values())
        + smoke.NETWORK_SECONDS * 2  # the control, then every pod at once
        + smoke.HEAL_SECONDS
        + smoke.FAIL_SECONDS
        + smoke.START_SECONDS
        + smoke.RECOVER_SECONDS
    )
    assert waits < minutes["Smoke-test the cluster"] * 60
